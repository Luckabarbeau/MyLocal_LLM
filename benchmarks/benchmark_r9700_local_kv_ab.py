#!/usr/bin/env python3
"""Matched alternating full-model HIP local-KV key-tile comparison.

Run via sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_r9700_local_kv_ab.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from statistics import median

import benchmark_dual_resolution_full_model_0102d as full_model


@contextmanager
def alternating_key_tiles():
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    original = DecoderLanguageModel.zero_grad
    environment = dict(os.environ)
    labels = []

    def zero_grad(model, *args, **kwargs):
        tile = (32, 64)[len(labels) % 2]
        os.environ['MINI_LLM_HIP_LOCAL_KV_KEY_TILE'] = str(tile)
        labels.append(tile)
        return original(model, *args, **kwargs)

    try:
        os.environ['MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD'] = '0'
        DecoderLanguageModel.zero_grad = zero_grad
        yield labels
    finally:
        DecoderLanguageModel.zero_grad = original
        os.environ.clear()
        os.environ.update(environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args(argv)
    args = full_model.parse_args([
        '--context', str(cli.context), '--architecture', 'matrix-pool-sparse',
        '--score-last-k', str(cli.context), '--head-chunk-tokens', '8192',
        '--activation-checkpoint', 'none', '--runtime-profile', 'consumer-gpu',
        '--query-chunk', '128', '--warmup-iterations', '6',
        '--repeat-iterations', '20',
    ])
    # Reserve before importing GPU libraries; never overwrite a previous result.
    with cli.output.open('x', encoding='utf-8') as output:
        with alternating_key_tiles() as labels:
            result = full_model.run(args)
        assert len(labels) == 26, f'unexpected zero_grad call count: {len(labels)}'
        result['warmup_key_tiles'] = labels[:6]
        for index, (row, tile) in enumerate(zip(result['iteration_timings_ms'], labels[6:])):
            row.update(iteration=index, local_kv_key_tile=tile)
        summary = {}
        for tile in (32, 64):
            rows = [row for row in result['iteration_timings_ms']
                    if row['local_kv_key_tile'] == tile]
            assert len(rows) == 10
            summary[str(tile)] = {
                'iterations': len(rows),
                'median_ms': {field: median(row[field] for row in rows)
                              for field in ('total_ms', 'forward_ms', 'backward_ms', 'loss_ms')},
                'range_ms': {field: [min(row[field] for row in rows),
                                     max(row[field] for row in rows)]
                             for field in ('total_ms', 'forward_ms', 'backward_ms', 'loss_ms')},
            }
        result.update(
            alternating_local_kv_key_tile=summary,
            hip_ffn_transpose_wgrad=False,
            query_chunk=args.query_chunk,
            timing_note='Same initialized model and random inputs; 6 alternating warmups, '
                        '20 alternating measured iterations (10 per tile); no optimizer, '
                        'full supervision, no checkpoint; all samples retained, no trimming. '
                        'Total includes forward, LM loss, and backward.',
            correctness_note='Existing run asserts finite loss each iteration and finite '
                             'representative gradients after the final iteration; '
                             'this is not a cross-tile numerical parity test.',
        )
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
        print(json.dumps({'context': cli.context, 'alternating_local_kv_key_tile': summary,
                          'gradients': result['gradients'], 'loss': result['loss']},
                         indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
