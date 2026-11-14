#!/usr/bin/env python3
"""Profile unchanged production HIP split-KV=2 local backward (no tuning)."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=107)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'refusing to overwrite {args.output}')
    if args.context <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error('context/repeats must be positive and warmup nonnegative')
    for name, expected in (('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '2'),
                           ('MINI_LLM_HIP_DQ_LOOP_STAGES', '1')):
        if os.environ.get(name, expected) != expected:
            parser.error(f'{name} must be unset or {expected}; production defaults only')

    import cupy as cp
    import mini_llm.ops.triton_dual_resolution_local as local
    import mini_llm.ops.triton_hip_local_kv as kv

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('this profiler is HIP-only')
    rng = np.random.default_rng(args.seed)

    def values(heads):
        return cp.asarray(rng.normal(0, 0.2, (1, args.context, heads, 64)),
                          dtype=cp.float32).astype('bfloat16')

    q, k, v = values(16), values(4), values(4)
    docs = cp.zeros((1, args.context), dtype=cp.int32)
    dy = cp.asarray(rng.normal(0, 0.1, q.shape), dtype=cp.float32)

    def forward():
        return local.local_attention_forward(q, k, v, docs, window=512,
                                             long_window=4096, long_heads=1)

    # Retain exactly one cache for all backward measurements and parity checks.
    _, cache = forward()
    reference = local.local_attention_backward(dy, cache)
    cp.cuda.runtime.deviceSynchronize()
    forward_samples = []
    for index in range(args.warmup + args.repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        temporary = forward()
        end.record()
        end.synchronize()
        if index >= args.warmup:
            forward_samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        del temporary

    events, resources = [], {}
    originals = [(local, name, getattr(local, name))
                 for name in ('_delta_kernel', '_dq_kernel')]
    originals += [(kv, name, getattr(kv, name))
                  for name in ('_kv_kernel', '_reduce_kernel')]

    class TimedKernel:
        def __init__(self, name, original):
            self.name, self.original = name, original

        def __getitem__(self, grid):
            def launch(*values, **options):
                label = {'_delta_kernel': 'delta', '_reduce_kernel': 'merge'}.get(self.name)
                if self.name == '_dq_kernel':
                    label = 'dq_long' if options['head_begin'] == 0 else 'dq_short'
                elif self.name == '_kv_kernel':
                    label = 'dk' if options['compute_dk'] else 'dv'
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                compiled = self.original[grid](*values, **options)
                end.record()
                events.append((label, start, end))
                metadata = compiled.metadata
                resources[label] = {
                    'kernel': self.name,
                    'nregs': getattr(compiled, 'n_regs', None),
                    'nspills': getattr(compiled, 'n_spills', None),
                    'shared_bytes': getattr(metadata, 'shared', None),
                    'grid': list(grid),
                    'num_stages': getattr(metadata, 'num_stages', options.get('num_stages')),
                    'launch_num_stages': options.get('num_stages'),
                    'num_warps': options.get('num_warps'),
                    'block_m': options.get('block_m'),
                    'block_n': options.get('block_n'),
                    'loop_stages': options.get('loop_stages'),
                }
                return compiled
            return launch

    whole, host, samples = [], [], defaultdict(list)
    errors = []
    try:
        for module, name, original in originals:
            setattr(module, name, TimedKernel(name, original))
        candidate = local.local_attention_backward(dy, cache)
        cp.cuda.runtime.deviceSynchronize()
        for actual, expected in zip(candidate, reference):
            errors.append(float(cp.max(cp.abs(actual - expected)).get()))
            np.testing.assert_allclose(actual.get(), expected.get(), atol=1e-5, rtol=1e-5)
        expected_labels = {'delta', 'dq_long', 'dq_short', 'dk', 'dv', 'merge'}
        if {label for label, _, _ in events} != expected_labels or len(events) != 6:
            raise RuntimeError('unexpected production backward launch sequence')
        for index in range(args.warmup + args.repeats):
            events.clear()
            cp.cuda.runtime.deviceSynchronize()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            begin = perf_counter()
            start.record()
            candidate = local.local_attention_backward(dy, cache)
            end.record()
            end.synchronize()
            elapsed_host = (perf_counter() - begin) * 1000
            if index >= args.warmup:
                host.append(elapsed_host)
                whole.append(float(cp.cuda.get_elapsed_time(start, end)))
                for label, first, last in events:
                    samples[label].append(float(cp.cuda.get_elapsed_time(first, last)))
    finally:
        for module, name, original in originals:
            setattr(module, name, original)

    kernel_medians = {label: median(times) for label, times in samples.items()}
    dominant = max(kernel_medians, key=kernel_medians.get)
    result = {
        'hip': True, 'device': str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'shape': {'B': 1, 'T': args.context, 'Q': 16, 'KV': 4, 'D': 64},
        'dtype': 'bfloat16', 'upstream_dtype': 'float32', 'seed': args.seed,
        'window': 512, 'long_window': 4096, 'long_heads': 1,
        'split_kv_policy': 2, 'warmup': args.warmup, 'repeats': args.repeats,
        'parity': {'reference': 'uninstrumented production backward',
                   'atol': 1e-5, 'rtol': 1e-5, 'max_abs_errors_dq_dk_dv': errors},
        'forward_event_median_ms': median(forward_samples),
        'backward_event_median_ms': median(whole),
        'backward_host_synchronized_median_ms': median(host),
        'kernel_median_ms': kernel_medians, 'resources': resources,
        'samples_ms': {'forward': forward_samples, 'backward_event': whole,
                       'backward_host': host, 'kernels': dict(samples)},
        'bottleneck': {'largest_measured_kernel': dominant,
                       'share_of_sum_kernel_medians': kernel_medians[dominant] / sum(kernel_medians.values()),
                       'qualification': 'Launch attribution only; resources do not prove occupancy, bandwidth, or compute limits.'},
        'instrumentation': 'Unchanged launch arguments; temporary wrappers restored in finally. Whole backward includes event/wrapper overhead; host includes synchronization.',
    }
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
