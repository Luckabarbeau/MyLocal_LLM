"""0092: fixed-reference token diagnostics and cut-point continuations.

Reference evaluation is intentionally *teacher forced*. A low rank of the
reference next token is disagreement with one valid continuation, not proof
that every higher ranked alternative is incorrect.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, asnumpy
from benchmarks.reference_diagnostics import token_kind, summarize_rows


def load_references(path):
    blob = Path(path).read_bytes()
    suite = json.loads(blob)
    cases = suite.get('references', [])
    if not isinstance(cases, list) or not cases:
        raise ValueError('Reference suite requires nonempty references list')
    ids = set()
    for item in cases:
        if not isinstance(item, dict) or not item.get('id') or not isinstance(item.get('text'), str) or not item['text']:
            raise ValueError(f'Invalid reference case: {item!r}')
        if item['id'] in ids:
            raise ValueError('Duplicate reference ID: ' + str(item['id']))
        ids.add(item['id'])
    return cases, hashlib.sha256(blob).hexdigest()


def _distribution(logits, true_id, top_k):
    v = np.asarray(asnumpy(logits), dtype=np.float64).ravel()
    if true_id < 0 or true_id >= len(v):
        raise ValueError('Reference token out of vocabulary')
    v -= np.max(v)
    probabilities = np.exp(v)
    probabilities /= probabilities.sum()
    actual = float(probabilities[true_id])
    candidates = np.argsort(-probabilities, kind='stable')[:top_k]
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    return {
        'actual_token_id': int(true_id), 'actual_probability': actual,
        'actual_rank': int(1 + np.count_nonzero(probabilities > actual)),
        'nll': float(-math.log(max(actual, 1e-300))),
        'entropy_nats': entropy,
        'top': [{'token_id': int(i), 'probability': float(probabilities[i])} for i in candidates],
    }


def analyze_reference(model, tokenizer, text, *, top_k=10, max_context=4096, category=None,
                      cutoffs=(0.25, 0.5, 0.75), max_new_tokens=128,
                      generator=None, seeds=(42,), strategy='top-k', temperature=0.8,
                      sampling_top_k=40, use_kv_cache=True, branches=0):
    ids = [int(v) for v in tokenizer.encode(text)]
    if len(ids) < 2:
        raise ValueError('Reference must have >=2 tokenizer tokens')
    if len(ids) > max_context + 1:
        raise ValueError('Reference longer than evaluation max_context+1; split into shorter references')
    inputs = xp.asarray(np.asarray(ids[:-1], dtype=np.int64)[None, :])
    logits = model.forward(inputs, return_cache=False)
    if isinstance(logits, tuple):
        raise RuntimeError('Expected forward(..., return_cache=False) to return logits')
    rows = []
    all_logits = np.asarray(asnumpy(logits[0]), dtype=np.float64)
    for pos, true_id in enumerate(ids[1:]):
        rec = _distribution(all_logits[pos], true_id, top_k)
        decoded = tokenizer.decode([true_id])
        rec.update(position=pos + 1, actual_text=decoded,
                   token_kind=token_kind(decoded, category=category),
                   top=[{**c, 'text': tokenizer.decode([c['token_id']])} for c in rec['top']])
        rows.append(rec)
    del logits, inputs
    result = {
        'reference_tokens': len(ids), 'token_ids': ids, 'token_predictions': rows,
        'mean_nll': float(np.mean([r['nll'] for r in rows])),
        'top1_accuracy': float(np.mean([r['actual_rank'] == 1 for r in rows])),
        'top5_accuracy': float(np.mean([r['actual_rank'] <= 5 for r in rows])),
        'top10_accuracy': float(np.mean([r['actual_rank'] <= 10 for r in rows])),
        'mean_target_rank': float(np.mean([r['actual_rank'] for r in rows])),
        'cutoff_generations': [],
        'diagnostics': summarize_rows(rows, category=category),
    }
    if generator is None:
        return result
    # Interpret float fractions as proportions of full tokenized reference;
    # integers as explicit token indices. Never cut within a token.
    cutoff_ids = []
    for c in cutoffs:
        index = int(round(c * len(ids))) if isinstance(c, float) else int(c)
        if not 1 <= index < len(ids):
            raise ValueError(f'Cutoff {c} resolved to invalid token index {index}')
        if index not in cutoff_ids:
            cutoff_ids.append(index)
    for index in cutoff_ids:
        prefix = tokenizer.decode(ids[:index])
        reconstructed = [int(v) for v in tokenizer.encode(prefix)]
        if reconstructed != ids[:index]:
            # BPE decode/re-encode around a split can differ. Explicitly track
            # the actual conditioned prefix; never claim the tokens are equal.
            note = 'decoded_prefix_retokenized_differently'
        else:
            note = 'exact_token_prefix'
        for seed in seeds:
            text_out, timing = generator.generate(
                prompt=prefix, max_new_tokens=max_new_tokens, strategy=strategy,
                temperature=temperature, top_k=sampling_top_k if strategy == 'top-k' else None,
                seed=seed, use_kv_cache=use_kv_cache, return_token_ids=True,
            )
            generated_ids = [int(v) for v in timing.get('generated_token_ids', [])]
            expected_ids = ids[index:]
            shared = 0
            for predicted, expected in zip(generated_ids, expected_ids):
                if predicted != expected:
                    break
                shared += 1
            from benchmarks.generation_metrics import repetition_metrics
            result['cutoff_generations'].append({
                'token_cutoff': index, 'prefix': prefix, 'prefix_contract': note,
                'seed': int(seed), 'strategy': strategy, 'generated_text': text_out,
                'generated_token_ids': generated_ids,
                'reference_suffix': tokenizer.decode(expected_ids),
                'reference_next_token_rank': rows[index-1]['actual_rank'],
                'reference_next_token_probability': rows[index-1]['actual_probability'],
                'reference_suffix_mean_nll': float(np.mean(
                    [row['nll'] for row in rows[index-1:]])),
                'matched_reference_prefix_tokens': shared,
                'matched_reference_next_token': bool(shared),
                'generation_repetition': repetition_metrics(generated_ids),
                'matching_note': 'Exact token-ID overlap with one reference; not a coherence or correctness score.',
            })
        if branches:
            # Teacher-forced top candidates at the precise cutoff. A separate
            # generation adapter is needed for strict token-ID forcing; report
            # proposals only here rather than misleadingly retokenizing them.
            result.setdefault('branch_candidates', []).append({
                'token_cutoff': index,
                'candidates': rows[index - 1]['top'][:branches],
                'note': 'Ranked candidates only: forced-token generation not implemented in 0092',
            })
    return result
