"""0106C quantized expert decode kernel benchmark (one expert, not model tok/s).

MINI_LLM_BACKEND=cupy .venv/bin/python benchmarks/benchmark_0106c_packed_expert.py \\
    --format Q4_K --d-model 2048 --d-ff 512 --tokens 1 --iterations 100

Reports launch-inclusive latency of unfused gate/up/activation/down versus
fused gate+up+SwiGLU followed by a packed down GEMV. The results only measure
resident packed weights: no router, cache-miss, PCIe, KV, or CPU execution.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def make_weight(fmt, rows, columns, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    if columns % qk:
        raise ValueError(f"{fmt} requires K divisible by {qk}")
    raw = np.random.default_rng(seed).integers(
        0, 256, size=(rows * columns // qk, qb), dtype=np.uint8,
    )
    scale = np.frombuffer(np.float16(.015).tobytes(), dtype=np.uint8)
    if fmt == "Q3_K":
        raw[:, 108:110] = scale
    elif fmt in ("Q4_K", "Q5_K"):
        raw[:, :2] = scale
        raw[:, 2:4] = scale
    else:
        raw[:, :2] = scale
    return PackedGGUFWeight(raw.reshape(-1), (rows, columns), fmt)


def elapsed_us(cp, fn, repeats):
    for _ in range(5):
        fn()
    cp.cuda.get_current_stream().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=BLOCK_FORMATS, default="Q4_K")
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if BACKEND_NAME != "cupy":
        raise SystemExit("Run with MINI_LLM_BACKEND=cupy and install Triton")
    import cupy as cp
    qk = BLOCK_FORMATS[args.format][0]
    if args.d_model < 1 or args.d_ff < 1 or args.tokens < 1 or args.iterations < 1:
        parser.error("all dimensions and iterations must be positive")
    if args.d_model % qk or args.d_ff % qk:
        parser.error(f"Both widths must be divisible by GGUF block length {qk}")
    gate = packed_to_device(make_weight(args.format, args.d_ff, args.d_model, 106))
    up = packed_to_device(make_weight(args.format, args.d_ff, args.d_model, 107))
    down = packed_to_device(make_weight(args.format, args.d_model, args.d_ff, 108))
    x = cp.asarray(np.random.default_rng(109).normal(0, .03, (args.tokens, args.d_model)),
                   dtype=cp.float32)
    old = PackedExpertFFNInference(gate, up, down, fuse_gate_up=False)
    new = PackedExpertFFNInference(gate, up, down, fuse_gate_up=True)
    expected = old.forward(x)
    actual = new.forward(x)
    cp.cuda.get_current_stream().synchronize()
    diff = cp.asnumpy(actual - expected)
    old_us = elapsed_us(cp, lambda: old.forward(x), args.iterations)
    new_us = elapsed_us(cp, lambda: new.forward(x), args.iterations)
    total_bytes = gate.nbytes + up.nbytes + down.nbytes
    print(f"GGUF {args.format} expert D={args.d_model}, F={args.d_ff}, tokens={args.tokens}")
    print(f"Resident packed weights: {total_bytes:,} bytes")
    print(f"Unfused gate/up/SiLU/down: {old_us:.2f} us/call")
    print(f"Fused gate+up+SiLU / down: {new_us:.2f} us/call")
    print(f"Speedup (same device): {old_us/new_us:.2f}x")
    print(f"Maximum absolute output difference: {np.max(np.abs(diff)):.6g}")
    print("These are single-expert timings, not end-to-end decoding performance.")


if __name__ == '__main__':
    main()
