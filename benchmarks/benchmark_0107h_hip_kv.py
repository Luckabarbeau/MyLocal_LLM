#!/usr/bin/env python3
"""HIP split-pass KV parity/resources/timings; no production source changes.

Seed/distribution match benchmark_0107f. The installed backward is a scheduling
reference, not an independent mathematical oracle. Baseline KV timing excludes
dQ so the comparison includes the same work: delta, dK, dV and optional reduce.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.context < 1 or args.repeats < 1:
        parser.error('context and repeats must be positive')
    import cupy as cp
    import mini_llm.ops.triton_dual_resolution_local as local
    import mini_llm.ops.triton_hip_local_kv as candidate

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('AMD HIP device required')
    rng = np.random.default_rng(107)

    def values(heads):
        return cp.asarray(rng.normal(0, 0.2, size=(1, args.context, heads, 64)),
                          dtype=cp.float32).astype('bfloat16')

    q, k, v = values(16), values(4), values(4)
    docs = cp.zeros((1, args.context), dtype=cp.int32)
    dy = cp.asarray(rng.normal(0, 0.1, size=q.shape), dtype=cp.float32)
    _, cache = local.local_attention_forward(q, k, v, docs, long_window=4096, long_heads=1)
    events, resources = [], {}

    def timed(label, launch, *values, **options):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        compiled = launch(*values, **options)
        end.record()
        events.append((label, start, end))
        resources[label] = {
            'n_regs': getattr(compiled, 'n_regs', None),
            'n_spills': getattr(compiled, 'n_spills', None),
            'shared_bytes': getattr(compiled.metadata, 'shared', None),
        }
        return compiled

    class TimedKernel:
        def __init__(self, original, label):
            self.original, self.label = original, label

        def __getitem__(self, grid):
            def launch(*values, **options):
                label = self.label
                if label == 'split':
                    label = 'dk' if options['compute_dk'] else 'dv'
                return timed(label, self.original[grid], *values, **options)
            return launch

    def compare(actual, reference, label):
        errors = {}
        for name, got, want in zip(('dk', 'dv'), actual, reference):
            got, want = cp.asnumpy(got), cp.asnumpy(want)
            error = float(np.max(np.abs(got - want)))
            errors[name] = error
            print(f'{label} {name} max_abs_error={error:.9g}', flush=True)
            np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
        return errors

    def measure(fn):
        events.clear()
        for _ in range(args.repeats):
            fn()
        cp.cuda.runtime.deviceSynchronize()
        times = defaultdict(list)
        for label, start, end in events:
            times[label].append(float(cp.cuda.get_elapsed_time(start, end)))
        totals = [sum(values[i] for values in times.values()) for i in range(args.repeats)]
        return {'kernel_median_ms': {name: median(values) for name, values in times.items()},
                'kv_device_median_ms': median(totals),
                'kv_device_samples_ms': totals}

    # Process-local wrappers only, restored even on failure. Force the installed
    # combined path for a stable reference regardless of the parent's opt-in.
    with patch.dict('os.environ', {'MINI_LLM_HIP_SPLIT_KV_BACKWARD': '0'}):
        with patch.object(local, '_delta_kernel', TimedKernel(local._delta_kernel, 'baseline_delta')), \
             patch.object(local, '_dkdv_kernel', TimedKernel(local._dkdv_kernel, 'baseline_dkdv')):
            _, ref_dk, ref_dv = local.local_attention_backward(dy, cache)  # warm once
            cp.cuda.runtime.deviceSynchronize()
            baseline = measure(lambda: local.local_attention_backward(dy, cache))

        original_delta = candidate._launch_delta

        def timed_delta(*values, **options):
            return timed('delta', original_delta, *values, **options)

        with patch.object(candidate, '_launch_delta', timed_delta), \
             patch.object(candidate, '_kv_kernel', TimedKernel(candidate._kv_kernel, 'split')), \
             patch.object(candidate, '_reduce_kernel', TimedKernel(candidate._reduce_kernel, 'reduce')):
            actual = candidate.backward(dy, cache)  # warm once and correctness
            cp.cuda.runtime.deviceSynchronize()
            errors = compare(actual, (ref_dk, ref_dv), 'random8k')
            split = measure(lambda: candidate.backward(dy, cache))

        # A separate small parity case exercises batch > 1, partial tiles,
        # strided packed QKV/dO/documents, EOS boundaries, and skip-head zero.
        packed_rng = np.random.default_rng(107)
        batch, length = 2, 641
        packed = cp.asarray(packed_rng.normal(0, 0.2, (batch, length * 2, 48, 64)),
                            dtype=cp.float32).astype('bfloat16')[:, ::2, ::2]
        pq, pk, pv = packed[:, :, :16], packed[:, :, 16:20], packed[:, :, 20:24]
        pdy = cp.asarray(packed_rng.normal(0, 0.1, (batch, length * 2, 16, 64)),
                         dtype=cp.float32)[:, ::2]
        pdocs = cp.zeros((batch, length * 2), dtype=cp.int32)[:, ::2]
        pdocs[:, 17:] = 1
        pdocs[:, 599:] = 2
        _, packed_cache = local.local_attention_forward(
            pq, pk, pv, pdocs, long_window=4096, long_heads=3, skip_head=0,
        )
        _, packed_dk, packed_dv = local.local_attention_backward(pdy, packed_cache)
        packed_actual = candidate.backward(pdy, packed_cache)
        packed_errors = compare(packed_actual, (packed_dk, packed_dv), 'packed_eos_skip0')
        repeated = candidate.backward(pdy, packed_cache)
        for got, again in zip(packed_actual, repeated):
            np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(again))

    result = {
        'hip': True, 'context': args.context, 'seed': 107, 'warmup': 1,
        'repeats': args.repeats, 'query_tile': 32, 'key_tile': 32,
        'num_warps': 4, 'num_stages': 1, 'atol': 1e-5, 'rtol': 1e-5,
        'correctness': 'passed', 'max_abs_errors': errors,
        'packed_eos_skip0_max_abs_errors': packed_errors,
        'packed_repeat_bitwise_equal': True,
        'baseline': baseline, 'split_pass': split, 'compiled_resources': resources,
        'kv_speedup': baseline['kv_device_median_ms'] / split['kv_device_median_ms'],
        'scope': 'delta + dK/dV + reduce device events; excludes dQ and allocation/host overhead',
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
