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
from benchmarks.live_suite import _generator, _preserve_global_rng


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
                       task_ids=None, include_counterfactual=False, max_cases=None):
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
    max_horizon = max(len(row[2]) for row in built)+8
    supports = bool(memory_cfg is not None and memory_cfg.enabled and
                    memory_cfg.integration_mode == 'historical_read')
    if max_horizon > working and supports and max_horizon > int(memory_cfg.memory_length):
        raise ValueError('Retrieval prompt exceeds trained historical horizon')
    if max_horizon > working and not supports:
        raise NotImplementedError('0097 long retrieval requires historical_read for >working context; never truncate')
    with _preserve_global_rng(), _disable_cuda_graphs_for_long_prefill():
        # The generator is an adapter to the validated native InferenceModel.
        # Do not use TextGenerator.generate(): it produces arbitrary free text,
        # while our test needs logits from a long-context prefill.
        generator = _generator(model, tokenizer, max_context=max_horizon)
        generator._ensure_inference_model(reset=True)
        impl = generator._inference_model
        for case, distance, ids, query, fact, fact_len in built:
            # Because these options share identical single-token next-token
            # boundaries, one prefill evaluates all candidates exactly.
            # Build choices from the *exact prompt last boundary*.
            choices = _one_token_choices(tokenizer, query, case['answer'])
            state = impl.create_generation_state(batch_size=1, max_length=max_horizon)
            next_logits = impl.prefill(xp.asarray(np.asarray(ids, dtype=np.int32)[None,:]), state)
            score = _rank_logits(next_logits, choices, case['answer'])
            row = {'id': case['id'], 'category': case.get('category','retrieval'),
                   'difficulty': ('short' if distance <=1024 else 'medium' if distance <=4096 else 'long'),
                   'requested_distance_tokens': distance, 'measured_distance_tokens': len(ids)-fact_len,
                   'prompt_tokens': len(ids), 'history_path_active': bool(state.long_memory_enabled),
                   'fact_in_historical_store': bool(len(ids)>working and fact_len<=len(ids)-working),
                   'answer': case['answer'], 'question': query.strip(),
                   'fact': fact.strip(), 'choices': [word for word,_ in choices], **score}
            if include_counterfactual:
                replacement = next(word for word,_ in choices if word != case['answer'])
                modified, _, _, _ = _build_prefix(tokenizer, case, distance,
                                                    replace_answer=replacement)
                control_state = impl.create_generation_state(batch_size=1, max_length=max_horizon)
                neg_logits = impl.prefill(xp.asarray(np.asarray(modified, dtype=np.int32)[None,:]), control_state)
                control_score = _rank_logits(neg_logits, choices, replacement)
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
        del generator
    by_distance = {}
    for distance in distances:
        rows = [r for r in results if r['requested_distance_tokens']==distance]
        by_distance[str(distance)] = {
            'count': len(rows), 'accuracy': sum(r['correct'] for r in rows)/len(rows),
            'mean_correct_margin': float(np.mean([r['correct_margin'] for r in rows])),
            'mean_correct_logprob': float(np.mean([r['correct_logprob'] for r in rows])),
            'mean_full_vocab_rank': float(np.mean([r['full_vocab_rank'] for r in rows])),
            'counterfactual_flip_rate': (sum(r['counterfactual']['follows_replacement'] for r in rows)/len(rows)
                if include_counterfactual else None),
        }
    return {'family':'long_retrieval', 'tasks_sha256':hashlib.sha256(blob).hexdigest(),
            'cases':results, 'count':len(results), 'by_distance':by_distance,
            'mode':'native_inference_prefill_historical_read',
            'warning':('No model may truncate a long retrieval prompt. Filler is synthetic and '
                       'repetitive; retrieval also depends on training/inference memory parity. '
                       'Candidate preference is not unconstrained free answer generation.')}
