#!/usr/bin/env python3
"""Isolate local-attention backward launches; compare staging without source edits.

Synthetic BF16 Q/K/V, FP32 upstream gradient; validates candidates against the
unchanged launch configuration. Not a mathematical oracle or full-model test.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warps', type=int, choices=(4, 8), default=4)
    parser.add_argument('--tile', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--dq-loop-stages', type=int, choices=(1, 2))
    parser.add_argument('--kv-query-tile', type=int, choices=(16, 32, 64))
    parser.add_argument('--kv-key-tile', type=int, choices=(16, 32, 64))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    import mini_llm.ops.triton_dual_resolution_local as local

    rng = np.random.default_rng(107)
    def values(heads):
        return cp.asarray(rng.normal(0, 0.2, size=(1, args.context, heads, 64)),
                          dtype='float32').astype('bfloat16')
    q, k, v = values(16), values(4), values(4)
    docs = cp.zeros((1, args.context), dtype=cp.int32)
    dy = cp.asarray(rng.normal(0, 0.1, size=q.shape), dtype=cp.float32)
    _, cache = local.local_attention_forward(q, k, v, docs, long_window=4096, long_heads=1)
    reference = local.local_attention_backward(dy, cache)
    cp.cuda.runtime.deviceSynchronize()
    original = {name: getattr(local, name) for name in ('_delta_kernel', '_dq_kernel', '_dkdv_kernel')}
    results = []

    class TimedKernel:
        def __init__(self, name, stages, events, metadata):
            self.name, self.stages = name, stages
            self.events, self.metadata = events, metadata

        def __getitem__(self, grid):
            def launch(*values, **options):
                launch_grid = grid
                if self.name != '_delta_kernel':
                    tile_m = args.tile
                    tile_n = args.tile
                    if self.name == '_dkdv_kernel':
                        tile_m = args.kv_query_tile or tile_m
                        tile_n = args.kv_key_tile or tile_n
                    owned_tile = tile_n if self.name == '_dkdv_kernel' else tile_m
                    launch_grid = ((options['n_ctx'] + owned_tile - 1) // owned_tile, grid[1])
                    options['block_m'], options['block_n'] = tile_m, tile_n
                    options['num_warps'] = args.warps
                    if self.stages is not None:
                        options['num_stages'] = self.stages
                if self.name == '_dq_kernel' and args.dq_loop_stages is not None:
                    options['loop_stages'] = args.dq_loop_stages
                label = self.name + (f".window{options['window']}" if 'window' in options else '')
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                compiled = original[self.name][launch_grid](*values, **options)
                end.record()
                self.events.append((label, start, end))
                self.metadata[label] = {
                    'registers': getattr(compiled, 'n_regs', None),
                    'spills': getattr(compiled, 'n_spills', None),
                    'shared_bytes': getattr(compiled.metadata, 'shared', None),
                }
                return compiled
            return launch

    try:
        for stages in (None, 1):
            events, metadata = [], {}
            for name in original:
                setattr(local, name, TimedKernel(name, stages, events, metadata))
            candidate = local.local_attention_backward(dy, cache)
            cp.cuda.runtime.deviceSynchronize()
            errors = []
            for actual, expected in zip(candidate, reference):
                error = float(cp.max(cp.abs(actual - expected)).get())
                errors.append(error)
                np.testing.assert_allclose(actual.get(), expected.get(), atol=1e-5, rtol=1e-5)
            events.clear()
            for _ in range(args.repeats):
                candidate = local.local_attention_backward(dy, cache)
            cp.cuda.runtime.deviceSynchronize()
            times = defaultdict(list)
            for label, start, end in events:
                times[label].append(float(cp.cuda.get_elapsed_time(start, end)))
            result = {'warps': args.warps, 'tile': args.tile,
                      'kv_query_tile': args.kv_query_tile, 'kv_key_tile': args.kv_key_tile,
                      'launch_stages': 3 if stages is None else stages,
                      'kernel_median_ms': {k: median(v) for k, v in times.items()},
                      'compiled_resources': metadata, 'max_abs_errors': errors}
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    finally:
        for name, kernel in original.items():
            setattr(local, name, kernel)
    args.output.write_text(json.dumps({'context': args.context, 'repeats': args.repeats,
                                      'hip': bool(cp.cuda.runtime.is_hip), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
