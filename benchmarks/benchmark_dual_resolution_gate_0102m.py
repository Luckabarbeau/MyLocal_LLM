#!/usr/bin/env python3
"""Measure warmed CuPy vs Triton gate (forward + backward, one invocation)."""
import argparse
import time
from statistics import median
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', default='4096,16384,65536')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    import cupy as cp
    from mini_llm.ops.triton_dual_resolution_gate import forward, backward
    rng = np.random.default_rng(102)
    for t in (int(v) for v in args.contexts.split(',')):
        h, d = 16, 64
        local = cp.asarray(rng.normal(size=(1, t, h, d)).astype('float32'))
        glob = cp.asarray(rng.normal(size=(1, t, h, d)).astype('float32'))
        logits = cp.asarray(rng.normal(size=(1, t, h)).astype('float32')).astype('bfloat16')
        available = cp.asarray(rng.uniform(size=(1, t)) > 0.2)
        dc = cp.asarray(rng.normal(size=local.shape).astype('float32'))

        def reference():
            g = 1 / (1 + cp.exp(-logits.astype('float32')))
            eff = g * available[..., None]
            context = local + eff[..., None] * (glob - local)
            dl = dc * (1 - eff[..., None])
            dg = dc * eff[..., None]
            dgate = cp.sum(dc * (glob - local), axis=-1) * available[..., None] * g * (1 - g)
            return context, (dl, dg, dgate)

        def fused():
            context, g, eff = forward(local, glob, logits, available)
            return context, backward(dc, local, glob, g, eff, available)

        ref, refg = reference()
        got, gotg = fused()
        cp.testing.assert_allclose(ref, got, rtol=3e-5, atol=3e-5)
        for a, b in zip(refg, gotg):
            cp.testing.assert_allclose(a, b, rtol=5e-5, atol=5e-5)
        times = {}
        for name, fn in (('cupy', reference), ('triton', fused)):
            for _ in range(args.warmup):
                fn()
            cp.cuda.Stream.null.synchronize()
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                fn()
                cp.cuda.Stream.null.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            times[name] = median(durations)
        print(f'T={t:6d} CuPy={times["cupy"]:8.3f} ms Triton={times["triton"]:8.3f} ms '
              f'speedup={times["cupy"] / times["triton"]:.2f}x')


if __name__ == '__main__':
    main()
