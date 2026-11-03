#!/usr/bin/env python3
"""SmolLM2 GGUF control for Model B's *same* synthetic retrieval task bank.

Reuses 0097 prompt construction, 4-choice scoring and ranking. Model-specific
embedded tokenizers imply that token-ID sequences are not identical. Does not
extend the pretrained model's 8192-token context or truncate long prompts.

Run: MINI_LLM_BACKEND=cupy .venv/bin/python -m benchmarks.evaluate_external_long_retrieval
    --model path/to/SmolLM2-360M-Instruct-F16.gguf
    --output-dir benchmarks/results/smollm2_retrieval_control
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from benchmarks.long_retrieval_0097 import (
    _answer_source_offset, _build_prefix, _one_token_choices,
    _rank_logits, _repeat_case,
)
from benchmarks.live_report import long_retrieval_case_csv, long_retrieval_html
from mini_llm.backend import BACKEND_NAME, xp

DEFAULT_TASKS = Path(__file__).resolve().parent / 'tasks' / 'long_retrieval_0097.json'
DEFAULT_DISTANCES = (1000, 4000, 8000, 16000, 24000, 32000)


class _TokenizerIDs:
    """Use the GGUF embedded tokenizer with NO per-fragment BOS insertion.

    The shared 0097 builder tokenizes fact, filler and query separately. An
    automatic BOS on every fragment would silently invalidate the prompts.
    A single model-specific BOS is added to the assembled prompt instead.
    """
    def __init__(self, embedded):
        self.embedded = embedded
        self.bos_id = (embedded.spec.bos_id if embedded.spec.add_bos else None)
        if embedded.spec.add_bos and self.bos_id is None:
            raise ValueError('GGUF tokenizer requires a BOS token but does not define it')

    def encode(self, text):
        return list(self.embedded.encode(text, add_special_tokens=False).ids)


def build_cases(tokenizer, task_path, distances=DEFAULT_DISTANCES, repetitions=2,
                max_context=8192):
    """Return scoreable rows and explicit exclusions; never truncate context."""
    if repetitions <= 0 or any(int(d) < 64 for d in distances):
        raise ValueError('repetitions must be positive and distances >= 64')
    task_path = Path(task_path)
    raw = task_path.read_bytes()
    bank = json.loads(raw)
    if bank.get('version') != 1 or bank.get('family') != 'long_retrieval':
        raise ValueError('Expected 0097 long_retrieval version 1 task bank')
    cases = bank['tasks']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Task bank must have nonempty, unique case IDs')
    output, unsupported = [], []
    for i, case in enumerate(cases):
        for repetition in range(repetitions):
            variant = _repeat_case(case, repetition, i)
            offset = _answer_source_offset(tokenizer, variant)
            phase = (i * repetitions + repetition) % 16
            pad = (phase - offset) % 16
            for distance in distances:
                ids, query, fact, fact_end = _build_prefix(
                    tokenizer, variant, int(distance), lead_tokens=pad)
                source_position = pad + offset
                if tokenizer.bos_id is not None:
                    ids = [int(tokenizer.bos_id)] + ids
                    source_position += 1
                    fact_end += 1
                row = {
                    'id': variant['id'] if 'id' in variant else (
                        case['id'] if repetition == 0 else f"{case['id']}_r{repetition:02d}"),
                    'base_case_id': case['id'], 'repetition': repetition,
                    'category': case.get('category', 'retrieval'),
                    'distance': int(distance), 'ids': ids, 'query': query,
                    'fact': fact, 'fact_end': fact_end,
                    'source_position': source_position,
                    'answer': variant['answer'], 'case': variant,
                    'lead_padding_tokens': pad,
                }
                if len(ids) > int(max_context):
                    unsupported.append({
                        'id': row['id'], 'base_case_id': case['id'],
                        'repetition': repetition,
                        'requested_distance_tokens': int(distance),
                        'prompt_tokens': len(ids),
                        'reason': f'prompt exceeds model-native context ({max_context} tokens)',
                    })
                else:
                    output.append(row)
    output.sort(key=lambda r: (r['distance'], r['source_position'] % 16, r['id']))
    return output, unsupported, hashlib.sha256(raw).hexdigest()


@contextmanager
def _disable_external_cuda_graph():
    names = ('MINI_LLM_CUDA_GRAPH_DECODE', 'MINI_LLM_CUDA_GRAPH_DECODE_STRICT')
    before = {n: os.environ.get(n) for n in names}
    try:
        for n in names:
            os.environ[n] = '0'
        yield
    finally:
        for n, v in before.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def _prefill_last(model, ids, *, chunk_size=128, chunk_threshold=4096):
    """Return final-token logits without full-prompt headwise attention matrices.

    For small prompts preserve the existing public prefill. For larger prompts
    process *all* tokens, in order, in bounded causal blocks using the model's
    existing GQA attention and KV caches. Unlike truncate/recent-window
    approximations, every query still attends to every preceding key.

    The external llama adapter owns an ordinary dense Transformer and exposes
    the same underlying inference blocks. This benchmark-only path does not
    alter their kernels, the checkpoint, or the global model runtime.
    """
    total = len(ids)
    if chunk_size < 1 or chunk_threshold < 1:
        raise ValueError('chunk_size and chunk_threshold must be positive')
    if total <= chunk_threshold:
        state = model.create_generation_state(batch_size=1, max_length=total)
        return model.prefill(xp.asarray(np.asarray(ids, dtype=np.int32)[None, :]), state)

    engine = model.engine
    if getattr(engine, 'dual_resolution_enabled', False) or getattr(engine, 'sparse_attention', False):
        raise NotImplementedError('Chunked external prefill requires ordinary dense GQA')
    if getattr(engine, 'long_memory_enabled', False):
        raise NotImplementedError('Chunked external prefill does not support long memory')
    state = model.create_generation_state(batch_size=1, max_length=total)
    tokens = xp.asarray(np.asarray(ids, dtype=np.int32)[None, :])
    result = None
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        hidden = engine._embedding_lookup(tokens[:, start:stop])
        for layer, block in enumerate(engine.blocks):
            hidden = block.prefill(hidden, state.k_cache[layer], state.v_cache[layer], start)
        if stop == total:
            normalized = engine.final_norm.forward(hidden)
            last = normalized[:, -1, :].astype(engine.dtype, copy=False)
            result = engine._lm_head_forward(last)
        state.length = stop
    return result


def _distance_metrics(results, unsupported, distances, include_counterfactual):
    summary = {}
    for distance in distances:
        rows = [r for r in results if r['requested_distance_tokens'] == distance]
        skipped = [r for r in unsupported if r['requested_distance_tokens'] == distance]
        ranks = [r['full_vocab_rank'] for r in rows]
        summary[str(distance)] = {
            'count': len(rows), 'supported': bool(rows), 'skipped_cases': len(skipped),
            'skip_reason': skipped[0]['reason'] if skipped else None,
            'accuracy': (sum(r['correct'] for r in rows) / len(rows) if rows else None),
            'mean_correct_margin': float(np.mean([r['correct_margin'] for r in rows])) if rows else None,
            'mean_correct_logprob': float(np.mean([r['correct_logprob'] for r in rows])) if rows else None,
            'mean_full_vocab_rank': float(np.mean(ranks)) if rows else None,
            'median_full_vocab_rank': float(np.median(ranks)) if rows else None,
            'geomean_full_vocab_rank': float(np.exp(np.mean(np.log(ranks)))) if rows else None,
            'full_vocab_top1_rate': sum(r == 1 for r in ranks)/len(ranks) if rows else None,
            'full_vocab_top128_rate': sum(r <= 128 for r in ranks)/len(ranks) if rows else None,
            'counterfactual_flip_rate': (
                sum(r['counterfactual']['follows_replacement'] for r in rows)/len(rows)
                if include_counterfactual and rows else None),
            'mean_counterfactual_original_logprob_change': (
                float(np.mean([r['counterfactual']['original_answer_logprob_change'] for r in rows]))
                if include_counterfactual and rows else None),
            'by_source_phase_mod16': {}, # SmolLM has no stride-16 sparse architecture.
        }
    return summary


def evaluate(model, tokenizer, task_path, distances, repetitions=2,
             include_counterfactual=False, progress=True,
             prefill_chunk_size=128, prefill_chunk_threshold=4096):
    native_limit = model.config.max_position_embeddings
    built, unsupported, digest = build_cases(
        tokenizer, task_path, distances, repetitions, native_limit)
    if not built:
        raise ValueError('No retrieval prompts fit the model native context')
    results, start = [], time.perf_counter()
    with _disable_external_cuda_graph():
        for i, row in enumerate(built, 1):
            choices = _one_token_choices(tokenizer, row['query'], row['answer'])
            logits = _prefill_last(model, row['ids'], chunk_size=prefill_chunk_size, chunk_threshold=prefill_chunk_threshold)
            score = _rank_logits(logits, choices, row['answer'])
            record = {
                'id': row['id'], 'base_case_id': row['base_case_id'],
                'repetition': row['repetition'], 'category': row['category'],
                'difficulty': ('short' if row['distance'] <= 1024 else
                               'medium' if row['distance'] <= 4096 else 'long'),
                'requested_distance_tokens': row['distance'],
                'measured_distance_tokens': len(row['ids']) - row['fact_end'],
                'prompt_tokens': len(row['ids']),
                'source_answer_position': row['source_position'],
                'source_answer_phase_mod16': row['source_position'] % 16,
                'sparse_phase': None, 'history_path_active': False,
                'fact_in_historical_store': False, 'model_b_full_sequence': False,
                'answer': row['answer'], 'question': row['query'].strip(),
                'fact': row['fact'].strip(),
                'choices': [word for word, _ in choices], **score,
            }
            if include_counterfactual:
                replacement = next(w for w, _ in choices if w != row['answer'])
                counter_ids, _, _, end = _build_prefix(
                    tokenizer, row['case'], row['distance'],
                    replace_answer=replacement, lead_tokens=row['lead_padding_tokens'])
                if tokenizer.bos_id is not None:
                    counter_ids = [int(tokenizer.bos_id)] + counter_ids
                    end += 1
                if len(counter_ids) > native_limit:
                    raise ValueError('Counterfactual prompt exceeds native context')
                control = _rank_logits(
                    _prefill_last(model, counter_ids, chunk_size=prefill_chunk_size, chunk_threshold=prefill_chunk_threshold), choices, replacement)
                record['counterfactual'] = {
                    'replacement': replacement, 'prediction': control['prediction'],
                    'follows_replacement': bool(control['correct']),
                    'original_answer_logprob_change': float(
                        control['candidate_logprobs'][row['answer']] - score['correct_logprob']),
                    'measured_distance_tokens': len(counter_ids)-end,
                }
            results.append(record)
            if progress:
                print(f"[SmolLM] {i}/{len(built)} distance={row['distance']} "
                      f"correct={int(score['correct'])} vocab_rank={score['full_vocab_rank']}",
                      flush=True)
    elapsed = time.perf_counter()-start
    return {
        'family': 'long_retrieval', 'tasks_sha256': digest,
        'cases': results, 'count': len(results), 'repetitions': repetitions,
        'requested_distances': list(distances),
        'by_distance': _distance_metrics(results, unsupported, distances, include_counterfactual),
        'unsupported_cases': unsupported,
        'mode': 'external_gguf_native_full_causal_prefill',
        'prefill_chunk_size': prefill_chunk_size,
        'prefill_chunk_threshold': prefill_chunk_threshold,
        'training_context_length': native_limit, 'evaluation_horizon': native_limit,
        'timing': {'elapsed_seconds': elapsed,
                   'processed_prompt_tokens': sum(r['prompt_tokens']*(2 if include_counterfactual else 1) for r in results),
                   'prompt_tokens_per_second': sum(r['prompt_tokens']*(2 if include_counterfactual else 1) for r in results)/elapsed if elapsed else None},
        'warning': ('SmolLM2 GGUF native-context control. Same 0097 task bank, prompt builder and 4-choice scorer as Model B, '
                    'but tokenization and absolute token IDs are model-specific; no truncation, no RoPE extrapolation. '
                    'Unsupported distances are explicitly excluded. Large prompts use bounded query chunks with full causal KV history; a reusable O(T^2) boolean mask may still be allocated.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    parser.add_argument('--distances', default=','.join(map(str, DEFAULT_DISTANCES)))
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--include-counterfactual', action='store_true')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--prefill-chunk-size', type=int, default=128,
                        help='Maximum query chunk for long native-context prompts (default: 128)')
    parser.add_argument('--prefill-chunk-threshold', type=int, default=4096,
                        help='Use cached chunked prefill for prompts longer than this (default: 4096)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f'GGUF not found: {args.model}')
    try:
        distances = tuple(int(x) for x in args.distances.split(','))
    except ValueError:
        parser.error('distances must be comma-separated integers')
    if not distances or len(set(distances)) != len(distances) or any(d < 64 for d in distances):
        parser.error('distances must be unique integers >=64')
    if args.repetitions <= 0:
        parser.error('repetitions must be positive')
    if args.prefill_chunk_size <= 0 or args.prefill_chunk_threshold <= 0:
        parser.error('prefill chunk size and threshold must be positive')
    from gguf import GGUFReader
    from mini_llm.external.gguf_loader import GGUFWeightMap
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from mini_llm.external import load_gguf_model
    inspection = GGUFWeightMap(GGUFReader(str(args.model)))
    embedded = load_gguf_tokenizer(inspection.reader, vocab_size=inspection.config.vocab_size)
    tokenizer = _TokenizerIDs(embedded)
    runnable, skipped, _ = build_cases(
        tokenizer, args.tasks, distances, args.repetitions,
        inspection.config.max_position_embeddings)
    print(f'SmolLM2 native context {inspection.config.max_position_embeddings}; '
          f'eligible={len(runnable)}, beyond context={len(skipped)}; GPU backend={BACKEND_NAME}', flush=True)
    if BACKEND_NAME != 'cupy':
        parser.error('The full 360M model should be run with MINI_LLM_BACKEND=cupy')
    if not runnable:
        parser.error('No retrieval cases fit the GGUF native context')
    model, provenance = load_gguf_model(args.model, dtype=args.dtype)
    result = evaluate(model, tokenizer, args.tasks, distances, args.repetitions,
                      args.include_counterfactual,
                      prefill_chunk_size=args.prefill_chunk_size,
                      prefill_chunk_threshold=args.prefill_chunk_threshold)
    result['model'] = provenance
    result['tokenizer'] = 'GGUF embedded SmolLM tokenizer; no per-fragment BOS'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    long_retrieval_case_csv(result, args.output_dir / 'retrieval_by_case.csv')
    (args.output_dir / 'report.html').write_text(long_retrieval_html(result), encoding='utf-8')
    print(f"Scored {result['count']} cases; skipped {len(result['unsupported_cases'])} beyond native context.")
    print('Wrote', args.output_dir / 'result.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
