#!/usr/bin/env python3
"""Standalone learned pooling CuPy vs Triton forward/backward benchmark.

Reports warmed wall-clock medians with device synchronization.  This tests
only one compressor's learned pooling primitive, not a complete LLM update.
Use the warmed 0102D full-model benchmark for production impact.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', default='1,16,64,256')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error('warmup and repeats must be positive')
    import cupy as cp
    from mini_llm.ops.attention import softmax_forward, softmax_backward
    from mini_llm.ops.triton_dual_resolution_pooling import forward, backward
    results = []
    rng = np.random.default_rng(102)
    h, t, d = 8, 128, 64
    scale = 1.0 / np.sqrt(d)
    for blocks in (int(x) for x in args.blocks.split(',')):
        if blocks < 1:
            parser.error('block counts must be positive')
        k = cp.asarray(rng.normal(size=(blocks, h, t, d)).astype('float32'))
        v = cp.asarray(rng.normal(size=(blocks, h, t, d)).astype('float32'))
        q = cp.asarray(rng.normal(size=(h, d)).astype('float32'))
        dout = cp.asarray(rng.normal(size=(blocks, h, d)).astype('float32'))

        def reference():
            scores = cp.sum(k * q[None, :, None, :], axis=-1) * scale
            p = softmax_forward(scores, axis=-1)
            context = cp.sum(p[..., None] * v, axis=2)
            dv = p[..., None] * dout[:, :, None, :]
            dp = cp.sum(dout[:, :, None, :] * v, axis=-1)
            ds = softmax_backward(dp, p, axis=-1)
            dk = ds[..., None] * q[None, :, None, :] * scale
            dq = cp.sum(ds[..., None] * k, axis=(0, 2)) * scale
            return context, (dk, dv, dq)

        def fused():
            context, probs = forward(k, v, q, scale)
            return context, backward(dout, k, v, q, probs, scale)

        ref_context, ref_grads = reference()
        new_context, new_grads = fused()
        cp.testing.assert_allclose(new_context, ref_context, rtol=2e-4, atol=2e-4)
        for a, b in zip(new_grads, ref_grads):
            cp.testing.assert_allclose(a, b, rtol=4e-4, atol=4e-4)

        case = {'blocks': blocks, 'heads': h, 'block_size': t, 'head_dim': d}
        for name, fn in (('cupy', reference), ('triton', fused)):
            for _ in range(args.warmup):
                fn()
            cp.cuda.Stream.null.synchronize()
            measurements = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                fn()
                cp.cuda.Stream.null.synchronize()
                measurements.append(1000 * (time.perf_counter() - start))
            case[name + '_median_ms'] = median(measurements)
        case['speedup'] = case['cupy_median_ms'] / case['triton_median_ms']
        print(f"blocks={blocks:4d}: CuPy={case['cupy_median_ms']:9.3f} ms "
              f"Triton={case['triton_median_ms']:9.3f} ms "
              f"speedup={case['speedup']:.2f}x")
        results.append(case)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    return results


if __name__ == '__main__':
    main()
