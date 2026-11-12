#!/usr/bin/env python3
"""Bounded HIP dense KV parity and median-five device timing (no source patches).

Run with the ROCm environment, e.g.:
sh scripts/run_r9700_preflight.sh benchmarks/benchmark_0107i_hip_dense_kv.py
Reports JSON to stdout only. SIGALRM bounds the complete GPU experiment to 180s.
The existing dense.backward is a scheduling reference, not an independent oracle.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    def timeout(*_):
        raise TimeoutError('180-second experiment bound reached; stop without retry')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    started = perf_counter()
    # Keep a genuine combined-kernel reference after optimized HIP defaults
    # are adopted by production dispatch. This script owns its process only.
    os.environ['MINI_LLM_HIP_DENSE_KV_BACKWARD'] = '0'
    os.environ['MINI_LLM_HIP_DENSE_BACKWARD_TILES'] = '0'
    import cupy as cp
    import mini_llm.ops.triton_dual_resolution_dense as dense
    import mini_llm.ops.triton_hip_dense_kv as candidate

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('AMD HIP required; no CUDA launch or production dispatch changes')
    result = {'seed': 107, 'atol': 1e-5, 'rtol': 1e-5, 'repeats': 5,
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'correctness': [], 'timings': {}, 'resources': {}, 'status': 'running',
              'scope': 'device events only: baseline dK/dV including inline delta vs candidate delta + dK + dV + reduce; excludes dQ, allocation and host overhead'}
    rng = np.random.default_rng(107)
    events = []

    def bf16(shape):
        return cp.asarray(rng.normal(0, .2, shape), dtype=cp.float32).astype('bfloat16')

    def make_case(branch, length, packed=False):
        batch = 2 if packed else 1
        docs_host = np.zeros((batch, length), np.int32)
        if packed:
            docs_host[0, 257:] = 1
            docs_host[0, 599:] = 2
            docs_host[1, 17:] = 1
            docs_host[1, 513:] = 2
        docs = cp.asarray(docs_host.repeat(2, axis=1))[:, ::2]
        hq, hk = ((4, 1) if packed else (1, 1)) if branch == 'sparse' else ((4, 2) if packed else (4, 4))
        q = bf16((batch, length * 2, hq * 2, 64))[:, ::2, ::2]
        if branch == 'sparse':
            stride, phase = 16, 5
            kv = bf16((batch, length * 2, hk * 4, 64))[:, ::2]
            k = kv[:, phase::stride, :hk * 2:2]
            v = kv[:, phase::stride, hk * 2::2]
            kwargs = dict(documents=docs, sparse_stride=stride, sparse_phase=phase,
                          bounded_causal=True)
        else:
            nk = length // 128 * 4
            kv = bf16((batch, nk * 2, hk * 4, 64))[:, ::2]
            k, v = kv[:, :, :hk * 2:2], kv[:, :, hk * 2::2]
            positions = np.zeros_like(docs_host)
            kd = np.zeros((batch, nk), np.int32)
            ke, valid = np.zeros_like(kd), np.zeros_like(kd)
            for b in range(batch):
                first = 0
                for m in range(length):
                    if m and docs_host[b, m] != docs_host[b, m - 1]:
                        first = m
                    positions[b, m] = m - first
                for n in range(nk):
                    begin, end = n // 4 * 128, (n // 4 + 1) * 128 - 1
                    kd[b, n] = docs_host[b, end]
                    ke[b, n] = positions[b, end]
                    valid[b, n] = docs_host[b, begin] == docs_host[b, end]
            kwargs = dict(documents=docs, positions=cp.asarray(positions),
                          block_documents=cp.asarray(kd), block_ends=cp.asarray(ke),
                          block_valid=cp.asarray(valid), bounded_causal=True,
                          history_block_size=128, summaries_per_block=4)
        dout = cp.asarray(rng.normal(0, .1, (batch, length * 2, hq * 2, 64)),
                          dtype=cp.float32)[:, ::2, ::2]
        _, cache = dense.forward(q, k, v, scale=.125, **kwargs)
        return dout, cache

    def destinations(cache, last_stride):
        b, nk, hk, _ = cache.k.shape
        raw_k = cp.full((b, nk * 2, hk * 2, 64 * last_stride), .25, cp.float32)
        raw_v = cp.full(raw_k.shape, -.5, cp.float32)
        dk = raw_k[:, ::2, ::2, ::last_stride]
        dv = raw_v[:, ::2, ::2, ::last_stride]
        return (dk, dv), (raw_k, raw_v)

    def check_canaries(raws, last_stride):
        for raw, value in zip(raws, (.25, -.5)):
            host = cp.asnumpy(raw)
            mask = np.ones(host.shape, bool)
            mask[:, ::2, ::2, ::last_stride] = False
            assert np.all(host[mask] == value), 'destination canary overwritten'

    def errors(got, want):
        a, b = cp.asnumpy(got).astype(np.float64), cp.asnumpy(want).astype(np.float64)
        err = np.abs(a - b)
        index = np.unravel_index(int(np.argmax(err)), err.shape)
        return {'max_abs': float(err.max()),
                'max_relative': float((err / np.maximum(np.abs(b), 1e-300)).max()),
                'max_tolerance_ratio': float((err / (1e-5 + 1e-5 * np.abs(b))).max()),
                'worst_index': list(map(int, index)), 'actual': float(a[index]),
                'reference': float(b[index]), 'worst_relative': float(err[index] / max(abs(b[index]), 1e-300)),
                'outside_tolerance': int((err > 1e-5 + 1e-5 * np.abs(b)).sum()),
                'dtype': 'float32', 'finite_difference_step': None,
                'passed': bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))}

    class TimedKernel:
        def __init__(self, kernel, label):
            self.kernel, self.label = kernel, label

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                label = self.label
                if label == 'pass':
                    label = 'dk' if kwargs['COMPUTE_DK'] else 'dv'
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                compiled = self.kernel[grid](*args, **kwargs)
                end.record()
                events.append((label, start, end))
                key = f'{label}.NQ{kwargs.get("NQ", "reduce")}.NK{kwargs.get("NK", "delta")}.GROUP{kwargs.get("GROUP", 1)}'
                result['resources'][key] = {
                    'n_regs': getattr(compiled, 'n_regs', None),
                    'n_spills': getattr(compiled, 'n_spills', None),
                    'shared_bytes': getattr(compiled.metadata, 'shared', None),
                    'grid': list(grid), 'num_stages': kwargs.get('num_stages'),
                }
                return compiled
            return launch

    def measure(fn):
        fn()  # Compile/warm outside the recorded samples.
        cp.cuda.runtime.deviceSynchronize()
        events.clear()
        for _ in range(5):
            fn()
        cp.cuda.runtime.deviceSynchronize()
        times = {}
        for label, start, end in events:
            times.setdefault(label, []).append(float(cp.cuda.get_elapsed_time(start, end)))
        assert all(len(samples) == 5 for samples in times.values())
        totals = [sum(samples[i] for samples in times.values()) for i in range(5)]
        return {'kernel_median_ms': {name: median(samples) for name, samples in times.items()},
                'device_samples_ms': totals, 'device_median_ms': median(totals)}

    try:
        for branch, length, packed in [('global', 8192, False), ('global', 641, True),
                                       ('sparse', 641, True), ('sparse', 8192, False)]:
            dout, cache = make_case(branch, length, packed)
            for add in ((False, True) if branch == 'sparse' else (False,)):
                reference_dest, reference_raw = destinations(cache, 1)
                _, ref_k, ref_v = dense.backward(dout, cache, dk=reference_dest[0],
                                                 dv=reference_dest[1], add_dkdv=add)
                check_canaries(reference_raw, 1)
                actual_dest, actual_raw = destinations(cache, 2)
                actual = candidate.backward(dout, cache, dk=actual_dest[0],
                                            dv=actual_dest[1], add_dkdv=add)
                assert all(a is b for a, b in zip(actual, actual_dest))
                check_canaries(actual_raw, 2)
                comparison = {name: errors(a, b) for name, a, b in
                              zip(('dk', 'dv'), actual, (ref_k, ref_v))}
                repeat_dest, _ = destinations(cache, 2)
                repeat = candidate.backward(dout, cache, dk=repeat_dest[0],
                                            dv=repeat_dest[1], add_dkdv=add)
                deterministic = all(np.array_equal(cp.asnumpy(a), cp.asnumpy(b))
                                    for a, b in zip(actual, repeat))
                row = {'case': f'{branch}_{length}_packed{int(packed)}_add{int(add)}',
                       'errors': comparison, 'repeat_bitwise_equal': deterministic,
                       'canaries_preserved': True,
                       'passed': deterministic and all(e['passed'] for e in comparison.values())}
                result['correctness'].append(row)
                print(json.dumps(row), flush=True)
                if not row['passed']:
                    raise AssertionError(f'{row["case"]}: parity failure; no tolerance relaxation')
            # Also cover omitted destinations and add=True's zero initialization.
            allocated = candidate.backward(dout, cache, add_dkdv=True)
            _, ref_k, ref_v = dense.backward(dout, cache)
            allocated_errors = {name: errors(a, b) for name, a, b in
                                zip(('dk', 'dv'), allocated, (ref_k, ref_v))}
            assert all(e['passed'] for e in allocated_errors.values()), allocated_errors
            result['correctness'].append({'case': f'{branch}_{length}_allocated_add',
                                          'errors': allocated_errors, 'passed': True})
            if length == 8192:
                with patch.object(dense, '_dkdv', TimedKernel(dense._dkdv, 'baseline_dkdv')):
                    baseline = measure(lambda: dense.backward(dout, cache))
                with ExitStack() as stack:
                    for name, label in (('_delta', 'delta'), ('_kv_pass', 'pass'), ('_reduce', 'reduce')):
                        stack.enter_context(patch.object(candidate, name, TimedKernel(getattr(candidate, name), label)))
                    split = measure(lambda: candidate.backward(dout, cache))
                result['timings'][branch] = {
                    'baseline': baseline, 'candidate': split,
                    'speedup': baseline['device_median_ms'] / split['device_median_ms'],
                    'scratch_bytes': 2 * int(cache.q.shape[0]) * int(cache.q.shape[2])
                                     * min(16, (length + 511) // 512) * int(cache.k.shape[1]) * 64 * 4,
                }
                print(json.dumps({'timing': branch, **result['timings'][branch]}), flush=True)
        # Rejection checks do not launch GPU work or change the production caller.
        from dataclasses import replace
        _, small_cache = make_case('global', 641, True)
        fp32_cache = replace(small_cache, q=small_cache.q.astype(cp.float32),
                             k=small_cache.k.astype(cp.float32), v=small_cache.v.astype(cp.float32))
        try:
            candidate.backward(cp.zeros(fp32_cache.q.shape, cp.float32), fp32_cache)
        except TypeError as exc:
            assert 'BF16' in str(exc)
        else:
            raise AssertionError('FP32 QKV must be rejected')
        with patch.object(cp.cuda.runtime, 'is_hip', False):
            try:
                candidate.backward(None, None)
            except RuntimeError as exc:
                assert 'HIP' in str(exc) and 'CUDA' in str(exc)
            else:
                raise AssertionError('CUDA must be rejected')
        result['rejections_passed'] = True
        result['status'] = 'passed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        result['elapsed_s'] = perf_counter() - started
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
