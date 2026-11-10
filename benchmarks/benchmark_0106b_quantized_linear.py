"""0106B single-op packed GEMV benchmark; *not* end-to-end tokens/sec.

MINI_LLM_BACKEND=cupy .venv/bin/python benchmarks/benchmark_0106b_quantized_linear.py \\
    --format Q4_K --k 2048 --n 512 --iterations 100
"""

import argparse
from pathlib import Path
import sys

# Allow direct execution from the repository root without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear


def _time_cuda(cp, fun, repeats):
    fun()
    cp.cuda.Stream.null.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        fun()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end) * 1000.0 / repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=BLOCK_FORMATS, default="Q4_K")
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if BACKEND_NAME != "cupy":
        raise SystemExit("Set MINI_LLM_BACKEND=cupy and install Triton to benchmark GPU")
    import cupy as cp
    qk, qb = BLOCK_FORMATS[args.format]
    if args.k % qk or min(args.k, args.n, args.tokens, args.iterations) <= 0:
        raise SystemExit("K must be block-aligned and dimensions/iterations positive")
    rng = np.random.default_rng(106)
    raw = rng.integers(0, 256, size=(args.n * args.k // qk, qb), dtype=np.uint8)
    if args.format == "Q3_K":
        raw[:, 108:110] = [0, 56]
    elif args.format == "Q6_K":
        raw[:, 208:210] = [0, 56]
    elif args.format in ("Q4_K", "Q5_K"):
        raw[:, :4] = [0, 56, 0, 52]
    else:
        raw[:, :2] = [0, 56]
    weight = PackedGGUFWeight(raw.reshape(-1), (args.n, args.k), args.format)
    x = rng.normal(0, 0.2, (args.tokens, args.k)).astype(np.float32)
    dev_weight = packed_to_device(weight)
    xd = cp.asarray(x)
    y = quantized_linear(xd, dev_weight)
    cp.cuda.Stream.null.synchronize()
    expected = x @ weight.dequantize_reference().T
    maxerr = float(np.max(np.abs(cp.asnumpy(y) - expected)))
    dense = cp.asarray(weight.dequantize_reference(), dtype=cp.float32)
    # This dense baseline is reference GEMV only; not necessarily a comparable
    # specialized kernel (and it is much larger in VRAM).
    fast_us = _time_cuda(cp, lambda: quantized_linear(xd, dev_weight), args.iterations)
    dense_us = _time_cuda(cp, lambda: xd @ dense.T, args.iterations)
    print(f"GGUF {args.format}: N={args.n} K={args.k} tokens={args.tokens}")
    print(f"packed bytes={weight.nbytes:,}; float32 bytes={dense.nbytes:,}; ratio={dense.nbytes/weight.nbytes:.2f}x")
    print(f"native packed GEMV {fast_us:.1f} us/call; dense FP32 baseline {dense_us:.1f} us/call")
    print(f"max absolute vs FP32 CPU oracle: {maxerr:.6g}")
    print("NOTE: Includes kernel launch overhead; no full-model, cache-miss, or CPU/GPU-overlap timing.")


if __name__ == "__main__":
    main()
