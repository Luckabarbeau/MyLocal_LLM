"""0097 distance-controlled, checkpoint-compatible long-context retrieval.

Uses the *actual* InferenceModel prefill path (historical_read when the prompt
exceeds the 4096-token working window).  Never calls the old 4096-token
teacher-forced scorer, and never silently truncates the 16K-token challenge.

Compares single-token candidate probabilities, which avoids additional costly
full-history prefill passes per multi-token answer. Reports measured distance
in tokenizer tokens, retrieval mode, and negative-control likelihood shifts.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
import math
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


def _build_prefix(tokenizer, case, distance, *, replace_answer=None):
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
    ids = fact_ids + (filler_tokens*(required//len(filler_tokens)) + filler_tokens[:required%len(filler_tokens)]) + query_ids
    # Exact measurement starts immediately AFTER the fact's last token.
    assert len(ids)-len(fact_ids) == distance
    return ids, query, fact, len(fact_ids)


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


def run_long_retrieval(model, tokenizer, *, tasks, distances=(1000,4000,16000),
                       task_ids=None, include_counterfactual=False, max_cases=None,
                       skip_unsupported_distances=False):
    """Benchmark 1K/4K/16K factual retrieval through actual inference memory.

    Not a pure memory score: reading the question, inference path and tokenizer
    each affect results. Counterfactual swaps the fact and checks whether the
    original answer becomes less likely (causal sensitivity diagnostic).
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
            raise ValueError('Unknown retrieval IDs: '+str(sorted(wanted-known)))
        cases = [r for r in cases if r['id'] in wanted]
    if max_cases is not None:
        cases = cases[:int(max_cases)]
    distances = [int(d) for d in distances]
    if not cases or not distances or any(d < 64 for d in distances):
        raise ValueError('Need nonempty cases and positive retrieval distances')
    memory_cfg = getattr(model.config, 'memory_context', None)
    working = int(model.config.context_length)
    results = []
    # No blanket catch: unsupported long memory is explicitly flagged on the
    # affected cases; actual prefill/cuda failures propagate as benchmark errors.
    built = []
    for case in cases:
        for distance in distances:
            ids, query, fact, fact_len = _build_prefix(tokenizer, case, distance)
            built.append((case, distance, ids, query, fact, fact_len))
    supports = bool(memory_cfg is not None and memory_cfg.enabled and
                    memory_cfg.integration_mode == 'historical_read')
    runnable = []
    unsupported = []
    for row in built:
        prompt_tokens = len(row[2])
        reason = None
        if prompt_tokens > working and not supports:
            reason = 'historical_read is required beyond the working context'
        elif (prompt_tokens > working and
              prompt_tokens + 8 > int(memory_cfg.memory_length)):
            reason = 'prompt exceeds the trained historical horizon'
        if reason is None:
            runnable.append(row)
        else:
            unsupported.append({
                'id': row[0]['id'],
                'requested_distance_tokens': row[1],
                'prompt_tokens': prompt_tokens,
                'reason': reason,
            })
    if unsupported and not skip_unsupported_distances:
        if any('historical_read' in row['reason'] for row in unsupported):
            raise NotImplementedError(
                '0097 long retrieval requires historical_read for >working context; never truncate')
        raise ValueError('Retrieval prompt exceeds trained historical horizon')
    max_horizon = max((len(row[2]) for row in runnable), default=working) + 8
    dual_cfg = getattr(model.config, 'dual_resolution', None)
    bounded_full_sequence = bool(
        not supports and dual_cfg is not None and getattr(dual_cfg, 'enabled', False)
        and getattr(dual_cfg, 'compression_mode', None) == 'matrix_pool'
    )
    graph_context = (
        _disable_cuda_graphs_for_long_prefill() if supports or bounded_full_sequence
        else _strict_graph_decode_for_generation(False, True)
    )
    with _preserve_global_rng(), graph_context:
        # The generator is an adapter to the validated native InferenceModel.
        # Do not use TextGenerator.generate(): it produces arbitrary free text,
        # while our test needs logits from a long-context prefill.
        generator = None
        impl = None
        reusable_state = None
        if not bounded_full_sequence:
            generator = _generator(model, tokenizer, max_context=max_horizon)
            generator._ensure_inference_model(reset=True)
            impl = generator._inference_model
            reusable_state = getattr(generator, '_state', None) if not supports else None
        for case_index, (case, distance, ids, query, fact, fact_len) in enumerate(runnable):
            # Because these options share identical single-token next-token
            # boundaries, one prefill evaluates all candidates exactly.
            # Build choices from the *exact prompt last boundary*.
            choices = _one_token_choices(tokenizer, query, case['answer'])
            if bounded_full_sequence:
                state = None
                full_logits = model.forward(
                    xp.asarray(np.asarray(ids, dtype=np.int32)[None,:]),
                    return_cache=False,
                )
                next_logits = full_logits[0, -1]
            else:
                if reusable_state is not None:
                    state = reusable_state
                    if case_index:
                        state.reset()
                else:
                    state = impl.create_generation_state(batch_size=1, max_length=max_horizon)
                next_logits = impl.prefill(xp.asarray(np.asarray(ids, dtype=np.int32)[None,:]), state)
            score = _rank_logits(next_logits, choices, case['answer'])
            if bounded_full_sequence:
                del next_logits, full_logits
            row = {'id': case['id'], 'category': case.get('category','retrieval'),
                   'difficulty': ('short' if distance <=1024 else 'medium' if distance <=4096 else 'long'),
                   'requested_distance_tokens': distance, 'measured_distance_tokens': len(ids)-fact_len,
                   'prompt_tokens': len(ids),
                   'history_path_active': bool(state is not None and state.long_memory_enabled),
                   'fact_in_historical_store': bool(len(ids)>working and fact_len<=len(ids)-working),
                   'answer': case['answer'], 'question': query.strip(),
                   'fact': fact.strip(), 'choices': [word for word,_ in choices], **score}
            if include_counterfactual:
                replacement = next(word for word,_ in choices if word != case['answer'])
                modified, _, _, _ = _build_prefix(tokenizer, case, distance,
                                                    replace_answer=replacement)
                if bounded_full_sequence:
                    control_state = None
                    control_logits = model.forward(
                        xp.asarray(np.asarray(modified, dtype=np.int32)[None,:]),
                        return_cache=False,
                    )
                    neg_logits = control_logits[0, -1]
                else:
                    if reusable_state is not None:
                        state.reset()
                        control_state = state
                    else:
                        control_state = impl.create_generation_state(batch_size=1, max_length=max_horizon)
                    neg_logits = impl.prefill(xp.asarray(np.asarray(modified, dtype=np.int32)[None,:]), control_state)
                control_score = _rank_logits(neg_logits, choices, replacement)
                if bounded_full_sequence:
                    del neg_logits, control_logits
                row['counterfactual'] = {
                    'replacement': replacement,
                    'prediction': control_score['prediction'],
                    'follows_replacement': bool(control_score['correct']),
                    'original_answer_logprob_change': float(
                        control_score['candidate_logprobs'][case['answer']] -
                        score['correct_logprob']),
                    'measured_distance_tokens': len(modified)-len(_tokens(tokenizer,
                        f"Archived entry: project {case['project']} has verification word {replacement}.\n")),
                }
            results.append(row)
        if generator is not None:
            del generator
    by_distance = {}
    for distance in distances:
        rows = [r for r in results if r['requested_distance_tokens']==distance]
        skipped = [r for r in unsupported if r['requested_distance_tokens']==distance]
        by_distance[str(distance)] = {
            'count': len(rows),
            'supported': bool(rows),
            'skipped_cases': len(skipped),
            'skip_reason': skipped[0]['reason'] if skipped else None,
            'accuracy': sum(r['correct'] for r in rows)/len(rows) if rows else None,
            'mean_correct_margin': (float(np.mean([r['correct_margin'] for r in rows]))
                                    if rows else None),
            'mean_correct_logprob': (float(np.mean([r['correct_logprob'] for r in rows]))
                                     if rows else None),
            'mean_full_vocab_rank': (float(np.mean([r['full_vocab_rank'] for r in rows]))
                                     if rows else None),
            'counterfactual_flip_rate': (
                sum(r['counterfactual']['follows_replacement'] for r in rows)/len(rows)
                if include_counterfactual and rows else None),
        }
    mode = ('native_inference_prefill_historical_read' if supports else
            'native_full_sequence_bounded_context' if bounded_full_sequence else
            'native_inference_prefill_bounded_context')
    return {'family':'long_retrieval', 'tasks_sha256':hashlib.sha256(blob).hexdigest(),
            'cases':results, 'count':len(results), 'by_distance':by_distance,
            'unsupported_cases':unsupported, 'mode':mode,
            'warning':('No model may truncate a long retrieval prompt. Unsupported distances are '
                       'explicitly skipped when capability-aware evaluation is enabled. Filler is '
                       'synthetic and repetitive; retrieval also depends on training/inference '
                       'memory parity. Candidate preference is not unconstrained free answer generation.')}
