"""Distance-controlled retrieval through Model B or legacy historical-read inference.

Model B uses full-sequence no-cache forward, projecting ONLY the final hidden
state to vocabulary logits. No artificial restriction to its training window.
The benchmark records full-vocabulary target ranks even on incorrect answers.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
import math
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, asnumpy
from benchmarks.live_suite import (
    _generator, _preserve_global_rng, _strict_graph_decode_for_generation,
)


FILLER = (
    'The quarterly register lists the status of equipment inspections and records '
    'whether the office lights were tested. A technician checks the inventory '
    'sheets before closing the workshop. The entries in this paragraph contain '
    'no project verification words. The archived notes discuss ordinary '
    'administrative tasks rather than access codes.\n'
)
WORDS = ('red','blue','green','white','black','orange','yellow','gold',
         'amber','cobalt','marble','violet','velvet','copper','silver','yellow',
         'garden','forest','winter','summer','planet','river','castle','orange',
         'purple','golden','crystal','falcon','harbor','meadow','signal','timber',
         'bamboo','sunset','meteor','orchid','tunnel','whisper','silvered',
         'sphere','apricot','puzzle','magnet','sailor','comet','island')


def _tokens(tokenizer, text):
    return [int(t) for t in tokenizer.encode(text)]


def _one_token_choices(tokenizer, prefix, target, n=4):
    """Candidates must be a *single exact token* at the given tokenizer boundary."""
    tokens = _tokens(tokenizer, prefix)
    options = []
    for word in (target,) + WORDS:
        if word in [v[0] for v in options]:
            continue
        joined = _tokens(tokenizer, prefix+' '+word)
        if joined[:len(tokens)] == tokens and len(joined) == len(tokens)+1:
            options.append((word, int(joined[-1])))
        if len(options) >= n and any(v[0] == target for v in options):
            break
    if not any(w == target for w, _ in options):
        raise ValueError('Answer is not one token in this tokenizer; change reference word')
    if len(options) < n:
        raise ValueError('Fewer than four distinct single-token candidate words')
    if len({t for _, t in options}) != n:
        raise ValueError('Candidate tokenizer collision')
    return options[:n]


def _build_prefix(tokenizer, case, distance, *, replace_answer=None, lead_tokens=0):
    word = replace_answer if replace_answer is not None else case['answer']
    fact = (f"Archived entry: project {case['project']} has verification word {word}.\n")
    query = (f"\nThe official verification word for project {case['project']} is")
    fact_ids, query_ids = _tokens(tokenizer, fact), _tokens(tokenizer, query)
    if distance <= len(query_ids):
        raise ValueError('Retrieval distance must exceed question size')
    filler_tokens = _tokens(tokenizer, FILLER)
    required = int(distance)-len(query_ids)
    if not filler_tokens:
        raise ValueError('Filler tokenization empty')
    # Prefix padding controls the source token's absolute stride-16 phase,
    # without changing the requested *fact-end to query-end* distance.
    newline = _tokens(tokenizer, '\n')
    if lead_tokens and not newline:
        raise ValueError('Tokenizer does not encode newline phase padding')
    lead = [newline[0]] * int(lead_tokens) if lead_tokens else []
    ids = lead + fact_ids + (filler_tokens*(required//len(filler_tokens)) + filler_tokens[:required%len(filler_tokens)]) + query_ids
    fact_end = len(lead) + len(fact_ids)
    assert len(ids) - fact_end == distance
    return ids, query, fact, fact_end


def _answer_source_offset(tokenizer, case):
    """Index of the first answer-dependent token in the historical fact.

    Usually this is exactly the leading answer token. Some BPE tokenizers
    merge the whitespace or previous word with the answer, so use a
    counterfactual first-difference fallback instead of failing a whole suite.
    """
    head = f"Archived entry: project {case['project']} has verification word"
    bare = _tokens(tokenizer, head)
    with_answer = _tokens(tokenizer, head + ' ' + case['answer'])
    if with_answer[:len(bare)] == bare and len(with_answer) > len(bare):
        return len(bare)
    alternative = 'blue' if case['answer'] == 'red' else 'red'
    counterfactual = _tokens(tokenizer, head + ' ' + alternative)
    first_difference = 0
    while (first_difference < min(len(with_answer), len(counterfactual))
           and with_answer[first_difference] == counterfactual[first_difference]):
        first_difference += 1
    if first_difference >= len(with_answer):
        raise ValueError('Cannot identify a distinct source-answer token')
    return first_difference


def _repeat_case(case, repeat, case_index):
    # Rotating the association prevents repeated probes from merely testing
    # the same word prior. The original case is retained for repeat zero.
    if repeat == 0:
        return dict(case)
    palette = ('red', 'blue', 'green', 'white', 'black', 'orange', 'yellow', 'gold')
    variant = dict(case)
    variant['project'] = f"{case['project']}-R{repeat:02d}"
    variant['answer'] = palette[(case_index + repeat) % len(palette)]
    return variant


def _model_b_final_logits(model, ids):
    # A full [T,vocab] LM head at 32K is >2 GiB even in BF16. Run all layers,
    # but project only the final time step. Never build an entire logits cube.
    hidden = model.forward_body(
        xp.asarray(np.asarray(ids, dtype=np.int32)[None, :]),
        return_cache=False,
    )
    logits, _ = model.output_proj.forward(hidden[:, -1:, :].copy())
    result = logits[0, -1]
    del hidden, logits
    return result



def _rank_logits(logits, candidates, answer):
    vec = np.asarray(asnumpy(logits), dtype=np.float64).reshape(-1)
    vec -= np.max(vec)
    lse = math.log(float(np.exp(vec).sum()))
    vals = {word: float(vec[token]-lse) for word, token in candidates}
    best = max(vals, key=vals.get)
    correct_lp = vals[answer]
    others = [v for word,v in vals.items() if word != answer]
    delta = correct_lp - max(others)
    # The top-1 over these four candidate words, not the entire vocabulary.
    all_lps = np.asarray(list(vals.values()))
    relative = np.exp(all_lps-all_lps.max())
    relative /= relative.sum()
    return {'prediction': best, 'correct': best == answer,
            'correct_rank': 1+sum(v > correct_lp for v in vals.values()),
            'full_vocab_rank': int(1+np.count_nonzero(vec>vec[dict(candidates)[answer]])),
            'correct_logprob': correct_lp, 'correct_probability': float(math.exp(correct_lp)),
            'correct_margin': float(delta),
            'relative_choice_probability': float(relative[list(vals).index(answer)]),
            'candidate_logprobs': vals}


@contextmanager
def _disable_cuda_graphs_for_long_prefill():
    # Native graph decoder is intended for short-context one-token decoding.
    # Long-history prefill is a different kernel path; preserve user settings.
    keys = ('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE',
            'MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT')
    original = {key:os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key]='0'
        yield
    finally:
        for key,value in original.items():
            if value is None: os.environ.pop(key,None)
            else:os.environ[key]=value


def run_long_retrieval(model, tokenizer, *, tasks,
                       distances=(1000, 4000, 6000, 8000, 12000, 16000, 24000, 32000),
                       task_ids=None, include_counterfactual=False, max_cases=None,
                       repetitions=1, skip_unsupported_distances=False):
    """Measure factual retrieval at exact distances, including >training-context.

    Repetitions vary the project/answer association and (for sparse Model B)
    systematically vary the answer-source absolute position modulo 16.
    """
    source = Path(tasks)
    blob = source.read_bytes()
    data = json.loads(blob)
    if data.get('version') != 1 or data.get('family') != 'long_retrieval':
        raise ValueError('Long retrieval task file requires family=long_retrieval version=1')
    cases = data.get('tasks', [])
    if task_ids is not None:
        wanted = set(task_ids)
        known = {r['id'] for r in cases}
        if wanted - known:
            raise ValueError('Unknown retrieval IDs: ' + str(sorted(wanted - known)))
        cases = [r for r in cases if r['id'] in wanted]
    if max_cases is not None:
        cases = cases[:int(max_cases)]
    distances = [int(d) for d in distances]
    if repetitions < 1 or int(repetitions) != repetitions:
        raise ValueError('repetitions must be a positive integer')
    repetitions = int(repetitions)
    if not cases or not distances or len(set(distances)) != len(distances) or any(d < 64 for d in distances):
        raise ValueError('Need nonempty cases and unique positive retrieval distances')

    memory_cfg = getattr(model.config, 'memory_context', None)
    working = int(model.config.context_length)
    supports = bool(memory_cfg is not None and memory_cfg.enabled and
                    memory_cfg.integration_mode == 'historical_read')
    dual_cfg = getattr(model.config, 'dual_resolution', None)
    bounded_full_sequence = bool(
        not supports and dual_cfg is not None and getattr(dual_cfg, 'enabled', False)
        and getattr(dual_cfg, 'compression_mode', None) == 'matrix_pool'
    )
    # Model B's max_blocks gives its architectural horizon, even if checkpoint
    # config.context_length was set to 8192 by the training launch.
    # No weights, training config, or model object are mutated.
    horizon = (int(dual_cfg.max_blocks) * int(dual_cfg.block_size)
               if bounded_full_sequence else
               int(memory_cfg.memory_length) if supports else working)
    sparse_stride = (int(getattr(dual_cfg, 'sparse_retrieval_stride', 0))
                     if bounded_full_sequence else 0)

    built = []
    for case_index, case in enumerate(cases):
        for repeat in range(repetitions):
            variant = _repeat_case(case, repeat, case_index)
            offset = _answer_source_offset(tokenizer, variant)
            # Cover phase classes in order (with eight cases and two repeats,
            # sixteen exact phases). Repeated examples keep their own IDs.
            wanted_phase = ((case_index * repetitions + repeat) % sparse_stride
                            if sparse_stride else None)
            pad = ((wanted_phase - offset) % sparse_stride if sparse_stride else 0)
            for distance in distances:
                ids, query, fact, fact_end = _build_prefix(
                    tokenizer, variant, distance, lead_tokens=pad)
                source_position = pad + offset
                built.append(dict(case=variant, base_id=case['id'],
                                  id=(case['id'] if repeat == 0 else
                                      f"{case['id']}_r{repeat:02d}"),
                                  repetition=repeat, distance=distance,
                                  ids=ids, query=query, fact=fact, fact_end=fact_end,
                                  source_position=source_position,
                                  phase=(source_position % sparse_stride
                                         if sparse_stride else None), pad=pad))
    runnable, unsupported = [], []
    for row in built:
        prompt_tokens = len(row['ids'])
        reason = None
        if prompt_tokens > working and not (supports or bounded_full_sequence):
            reason = 'historical_read is required beyond the working context'
        elif prompt_tokens > horizon:
            reason = (f'prompt exceeds supported inference horizon ({horizon} tokens)')
        if reason is None:
            runnable.append(row)
        else:
            unsupported.append({
                'id': row['id'], 'base_case_id': row['base_id'],
                'repetition': row['repetition'],
                'requested_distance_tokens': row['distance'],
                'prompt_tokens': prompt_tokens, 'reason': reason,
            })
    if unsupported and not skip_unsupported_distances:
        if any('historical_read' in x['reason'] for x in unsupported):
            raise NotImplementedError(
                '0097 long retrieval requires historical_read for >working context; never truncate')
        raise ValueError('Retrieval prompt exceeds the supported inference horizon; never truncate')

    runnable.sort(key=lambda row: (
        row['distance'], row['phase'] if row['phase'] is not None else -1, row['id']))
    max_horizon = max((len(x['ids']) for x in runnable), default=working)
    graph_context = (_disable_cuda_graphs_for_long_prefill()
                     if supports or bounded_full_sequence else
                     _strict_graph_decode_for_generation(False, True))
    results = []
    benchmark_start = time.perf_counter()
    with _preserve_global_rng(), graph_context:
        generator = impl = reusable_state = None
        if not bounded_full_sequence and runnable:
            generator = _generator(model, tokenizer, max_context=max_horizon)
            generator._ensure_inference_model(reset=True)
            impl = generator._inference_model
            reusable_state = getattr(generator, '_state', None) if not supports else None

        total_cases = len(runnable)
        for case_index, example in enumerate(runnable):
            case_start = time.perf_counter()
            case = example['case']
            distance = example['distance']
            ids, query, fact, fact_end = (example[k] for k in
                                          ('ids', 'query', 'fact', 'fact_end'))
            print(
                f"[0097] Long retrieval {case_index + 1}/{total_cases} "
                f"id={example['id']} distance={distance} tokens={len(ids)}",
                flush=True,
            )
            choices = _one_token_choices(tokenizer, query, case['answer'])
            if bounded_full_sequence:
                next_logits = _model_b_final_logits(model, ids)
                state = None
            else:
                if reusable_state is not None:
                    state = reusable_state
                    if case_index:
                        state.reset()
                else:
                    state = impl.create_generation_state(batch_size=1, max_length=max_horizon)
                next_logits = impl.prefill(
                    xp.asarray(np.asarray(ids, dtype=np.int32)[None, :]), state)
            score = _rank_logits(next_logits, choices, case['answer'])
            del next_logits
            row = {
                'id': example['id'], 'base_case_id': example['base_id'],
                'repetition': example['repetition'],
                'category': case.get('category', 'retrieval'),
                'difficulty': ('short' if distance <= 1024 else
                               'medium' if distance <= 4096 else 'long'),
                'requested_distance_tokens': distance,
                'measured_distance_tokens': len(ids) - fact_end,
                'prompt_tokens': len(ids),
                'history_path_active': bool(state is not None and state.long_memory_enabled),
                'fact_in_historical_store': bool(len(ids) > working and
                                                 fact_end <= len(ids) - working),
                'model_b_full_sequence': bounded_full_sequence,
                'source_answer_position': example['source_position'],
                'source_answer_phase_mod16': (example['source_position'] % 16),
                'sparse_phase': example['phase'],
                'answer': case['answer'], 'question': query.strip(),
                'fact': fact.strip(), 'choices': [word for word, _ in choices], **score,
            }
            if include_counterfactual:
                replacement = next(word for word, _ in choices if word != case['answer'])
                modified, _, _, modified_fact_end = _build_prefix(
                    tokenizer, case, distance, replace_answer=replacement,
                    lead_tokens=example['pad'])
                if len(modified) > horizon:
                    raise ValueError('Counterfactual prompt exceeds inference horizon')
                if bounded_full_sequence:
                    neg_logits = _model_b_final_logits(model, modified)
                else:
                    if reusable_state is not None:
                        state.reset()
                        control_state = state
                    else:
                        control_state = impl.create_generation_state(
                            batch_size=1, max_length=max_horizon)
                    neg_logits = impl.prefill(
                        xp.asarray(np.asarray(modified, dtype=np.int32)[None, :]),
                        control_state)
                control_score = _rank_logits(neg_logits, choices, replacement)
                del neg_logits
                row['counterfactual'] = {
                    'replacement': replacement,
                    'prediction': control_score['prediction'],
                    'follows_replacement': bool(control_score['correct']),
                    'original_answer_logprob_change': float(
                        control_score['candidate_logprobs'][case['answer']] -
                        score['correct_logprob']),
                    'measured_distance_tokens': len(modified) - modified_fact_end,
                }
            case_elapsed = time.perf_counter() - case_start
            processed_tokens = len(ids) * (2 if include_counterfactual else 1)
            row['elapsed_seconds'] = case_elapsed
            row['prompt_tokens_per_second'] = (
                processed_tokens / case_elapsed if case_elapsed > 0 else None)
            print(
                f"[0097] Long retrieval complete {case_index + 1}/{total_cases} "
                f"elapsed={case_elapsed:.3f}s rank={row['full_vocab_rank']} "
                f"correct={int(row['correct'])}",
                flush=True,
            )
            results.append(row)
        if generator is not None:
            del generator

    by_distance = {}
    for distance in distances:
        rows = [r for r in results if r['requested_distance_tokens'] == distance]
        skipped = [r for r in unsupported if r['requested_distance_tokens'] == distance]
        ranks = [r['full_vocab_rank'] for r in rows]
        by_distance[str(distance)] = {
            'count': len(rows), 'supported': bool(rows),
            'skipped_cases': len(skipped),
            'skip_reason': skipped[0]['reason'] if skipped else None,
            'accuracy': sum(r['correct'] for r in rows) / len(rows) if rows else None,
            'mean_correct_margin': (float(np.mean([r['correct_margin'] for r in rows]))
                                    if rows else None),
            'mean_correct_logprob': (float(np.mean([r['correct_logprob'] for r in rows]))
                                     if rows else None),
            'mean_full_vocab_rank': float(np.mean(ranks)) if ranks else None,
            'median_full_vocab_rank': float(np.median(ranks)) if ranks else None,
            'geomean_full_vocab_rank': (float(np.exp(np.mean(np.log(ranks))))
                                        if ranks else None),
            'full_vocab_top1_rate': sum(r == 1 for r in ranks) / len(ranks) if ranks else None,
            'full_vocab_top128_rate': sum(r <= 128 for r in ranks) / len(ranks) if ranks else None,
            'counterfactual_flip_rate': (
                sum(r['counterfactual']['follows_replacement'] for r in rows) / len(rows)
                if include_counterfactual and rows else None),
            'mean_counterfactual_original_logprob_change': (
                float(np.mean([r['counterfactual']['original_answer_logprob_change']
                               for r in rows])) if include_counterfactual and rows else None),
            'by_source_phase_mod16': {
                str(phase): {
                    'count': len(group),
                    'mean_full_vocab_rank': float(np.mean([g['full_vocab_rank'] for g in group])),
                    'accuracy': sum(g['correct'] for g in group) / len(group),
                }
                for phase in range(16)
                if (group := [g for g in rows if g['source_answer_phase_mod16'] == phase])
            },
        }
    mode = ('native_inference_prefill_historical_read' if supports else
            'native_full_sequence_bounded_context' if bounded_full_sequence else
            'native_inference_prefill_bounded_context')
    elapsed_seconds = time.perf_counter() - benchmark_start
    processed_prompt_tokens = sum(
        row['prompt_tokens'] * (2 if include_counterfactual else 1) for row in results)
    return {
        'family': 'long_retrieval', 'tasks_sha256': hashlib.sha256(blob).hexdigest(),
        'cases': results, 'count': len(results), 'repetitions': repetitions,
        'requested_distances': distances, 'by_distance': by_distance,
        'unsupported_cases': unsupported, 'mode': mode,
        'training_context_length': working, 'evaluation_horizon': horizon,
        'timing': {
            'elapsed_seconds': elapsed_seconds,
            'processed_prompt_tokens': processed_prompt_tokens,
            'prompt_tokens_per_second': (
                processed_prompt_tokens / elapsed_seconds if elapsed_seconds > 0 else None),
        },
        'warning': ('No prompt is truncated. Model B is evaluated using full-sequence '
                    'no-cache attention and last-token-only LM projection; its context '
                    'length may exceed the training window. Full-vocabulary rank and '
                    'counterfactual sensitivity are stronger diagnostics than four-choice '
                    'accuracy. Exact retrieval is not established by rank alone.'),
    }
