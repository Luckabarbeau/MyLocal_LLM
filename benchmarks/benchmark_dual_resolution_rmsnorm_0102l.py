#!/usr/bin/env python3
"""Warmed comparison of reference vs fused compressor RMSNorm forward/backward.

This is a primitive microbenchmark. The full-model 0102D benchmark determines
whether RMSNorm fusion actually improves end-to-end training throughput.
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
    parser.add_argument('--blocks', default='1,16,64,256,512')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import cupy as cp
    from mini_llm.ops.triton_dual_resolution_rmsnorm import forward, backward
    import mini_llm.ops.dual_resolution_attention as ref
    rng = np.random.default_rng(112)
    records = []
    for blocks in (int(v) for v in args.blocks.split(',')):
        if blocks <= 0:
            parser.error('blocks must be positive')
        x = cp.asarray(rng.normal(size=(blocks, 128, 512)).astype('float32')).astype('bfloat16')
        dy = cp.asarray(rng.normal(size=(blocks, 128, 512)).astype('float32'))
        gamma = cp.asarray(rng.uniform(.8, 1.2, 512).astype('float32')).astype('bfloat16')
        class Param:
            def __init__(self):
                self.data = gamma
                self.grad = cp.zeros((512,), dtype='float32')
        param = Param()
        def cupy_impl():
            # Explicit reference without an environment switch, for stable A/B.
            work = ref._work_array(x)
            inverse = 1.0 / cp.sqrt(cp.mean(work * work, axis=-1, keepdims=True) + 1e-6)
            normalized = work * inverse
            out = (normalized * ref._work_array(gamma)).astype(x.dtype)
            value = work
            din = ref._work_array(dy) * ref._work_array(gamma)
            projection = cp.sum(din * value, axis=-1, keepdims=True)
            dx = din * inverse - value * (inverse**3) * projection / 512.0
            dg = cp.sum(ref._work_array(dy) * normalized, axis=(0, 1))
            return out, dx, dg
        def fused_impl():
            out, inv = forward(x, gamma)
            dx, dg = backward(dy, x, gamma, inv)
            return out, dx, dg
        a = cupy_impl()
        b = fused_impl()
        for left, right in zip(a, b):
            cp.testing.assert_allclose(left.astype('float32'), right.astype('float32'), atol=.02, rtol=.02)
        record = {'blocks': blocks, 'row_count': blocks*128}
        for label, fn in [('cupy', cupy_impl), ('triton', fused_impl)]:
            for _ in range(args.warmup):
                fn()
            cp.cuda.Stream.null.synchronize()
            timings = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                cp.cuda.Stream.null.synchronize()
                timings.append(1000 * (time.perf_counter() - t0))
            record[label + '_ms'] = median(timings)
        record['speedup'] = record['cupy_ms']/record['triton_ms']
        print(f"blocks={blocks:4d}: CuPy={record['cupy_ms']:9.3f} ms "
              f"Triton={record['triton_ms']:9.3f} ms speedup={record['speedup']:.2f}x")
        records.append(record)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
