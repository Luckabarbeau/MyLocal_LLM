"""Compare native packed CPU MoE GEMV to FP32 reference and cache reuse.

MINI_LLM_BACKEND=numpy python benchmarks/benchmark_0106e_cpu_packed.py \\
    --format Q4_K --d-model 2048 --d-ff 512 --tokens 1 --iterations 30

Measures *one expert* only (not model tokens/sec). Native C++ compiles once
without admin rights; compiler startup is excluded from timed loops.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from mini_llm.ops.native_packed_cpu import packed_cpu_expert
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def make_weight(fmt, n, k, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    data = np.random.default_rng(seed).integers(0, 256, (n * k // qk, qb), dtype=np.uint8)
    scale = np.frombuffer(np.float16(.02).tobytes(), dtype=np.uint8)
    if fmt == "Q3_K":
        data[:, 108:110] = scale
    elif fmt == "Q6_K":
        data[:, 208:210] = scale
    else:
        data[:, :2] = scale
        if fmt in ("Q4_K", "Q5_K"):
            data[:, 2:4] = np.frombuffer(np.float16(.01).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(data.ravel(), (n, k), fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=list(BLOCK_FORMATS), default="Q4_K")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--iterations", type=int, default=15)
    args = p.parse_args()
    qk, _ = BLOCK_FORMATS[args.format]
    if min(args.d_model, args.d_ff, args.tokens, args.iterations) < 1 or \
       args.d_model % qk or args.d_ff % qk:
        p.error("positive dimensions and iteration count, d_model and d_ff block aligned")
    g = make_weight(args.format, args.d_ff, args.d_model, 1)
    u = make_weight(args.format, args.d_ff, args.d_model, 2)
    d = make_weight(args.format, args.d_model, args.d_ff, 3)
    x = np.random.default_rng(123).normal(0, .02, (args.tokens, args.d_model)).astype(np.float32)
    # Initialize once; do not include the compiler in benchmark timings.
    native = packed_cpu_expert(x, (g, u, d))
    wg, wu, wd = (w.dequantize_reference() for w in (g, u, d))
    def reference():
        gg, uu = x @ wg.T, x @ wu.T
        return ((gg / (1.0 + np.exp(-gg))) * uu) @ wd.T
    oracle = reference()
    np.testing.assert_allclose(native, oracle, rtol=2e-4, atol=2e-3)
    def reference_uncached():
        ag, au, ad = (w.dequantize_reference() for w in (g, u, d))
        gg, uu = x @ ag.T, x @ au.T
        return ((gg / (1.0 + np.exp(-gg))) * uu) @ ad.T
    def measure(fn):
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            fn()
        return (time.perf_counter() - t0) * 1e3 / args.iterations
    native_ms = measure(lambda: packed_cpu_expert(x, (g, u, d)))
    reference_ms = measure(reference)
    uncached_ms = measure(reference_uncached)
    packed_bytes = sum(w.nbytes for w in (g, u, d))
    print(f"format={args.format}, d_model={args.d_model}, d_ff={args.d_ff}, tokens={args.tokens}")
    print(f"packed expert: {packed_bytes/1024**2:.2f} MiB; "
          f"dense FP32 expert: {sum(w.size for w in (wg, wu, wd))*4/1024**2:.2f} MiB")
    print(f"packed CPU direct:       {native_ms:.3f} ms")
    print(f"FP32 cached BLAS:        {reference_ms:.3f} ms (extra dense RAM)")
    print(f"FP32 dequant every call: {uncached_ms:.3f} ms (no dense RAM cache)")
    print(f"native vs cached FP32:   {reference_ms/native_ms:.3f}x")
    print(f"native vs cold decode:   {uncached_ms/native_ms:.3f}x")
    print("Single-expert CPU microbenchmark; not end-to-end model tokens/sec.")


if __name__ == "__main__":
    main()
