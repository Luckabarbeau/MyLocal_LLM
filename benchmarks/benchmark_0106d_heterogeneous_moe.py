"""Single-layer decode microbenchmark for 0106D heterogeneous experts.

Examples:
  MINI_LLM_BACKEND=numpy python benchmarks/benchmark_0106d_heterogeneous_moe.py
  MINI_LLM_BACKEND=cupy python benchmarks/benchmark_0106d_heterogeneous_moe.py \\
      --d-model 2048 --d-ff 512 --top-k 10 --n-experts 64 \\
      --device-budget-mb 192 --host-dequant-budget-mb 256 --iterations 20

These are single-layer timings, *not* full-model tokens/s. Packed CPU weights
still use the slow correctness decoder. The goal is to expose whether and how
much cross-device overlap helps, including the cost of host fallback.
"""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def packed_weight(fmt, n, k, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    if k % qk:
        raise ValueError(f"{fmt} requires the input width divisible by {qk}")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n * k // qk, qb), dtype=np.uint8)
    if fmt == "Q3_K":
        raw[:, 108:110] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
    elif fmt in ("Q4_K", "Q5_K"):
        raw[:, 0:2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
        raw[:, 2:4] = np.frombuffer(np.float16(.01).tobytes(), dtype=np.uint8)
    else:
        raw[:, 0:2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(raw.ravel(), (n, k), fmt)


def synchronize():
    if BACKEND_NAME == "cupy":
        xp.cuda.runtime.deviceSynchronize()


def run(args, workers, provider, inputs, weights, ids):
    executor = HeterogeneousExpertExecutor(
        provider, model_id="bench-0106d", d_model=args.d_model, d_ff=args.d_ff,
        device_budget_bytes=int(args.device_budget_mb * 1024 * 1024),
        host_dequant_budget_bytes=int(args.host_dequant_budget_mb * 1024 * 1024),
        promote_after=args.promote_after, compute_dtype="float16",
        cpu_workers=workers,
        cpu_packed_backend=args.cpu_packed_backend,
        idle_eviction_requests=args.idle_eviction_requests,
    )
    layer = ExpertsInference(args.d_model, args.d_ff, args.n_experts,
                             expert_executor=executor, layer_id=0)
    try:
        for _ in range(args.warmup):
            layer.forward(inputs, weights, ids)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            output = layer.forward(inputs, weights, ids)
        synchronize()
        elapsed = (time.perf_counter() - start) / args.iterations
        return output, elapsed, executor.statistics()
    finally:
        executor.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=list(BLOCK_FORMATS), default="Q4_0")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--n-experts", type=int, default=16)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--device-budget-mb", type=float, default=0)
    p.add_argument("--host-dequant-budget-mb", type=float, default=16)
    p.add_argument("--promote-after", type=int, default=3)
    p.add_argument("--cpu-packed-backend", choices=["reference", "native", "auto"], default="reference")
    p.add_argument("--idle-eviction-requests", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.top_k <= args.n_experts or args.iterations <= 0 or args.warmup < 0:
        p.error("Require 1 <= top-k <= n-experts, iterations > 0, warmup >= 0")
    if args.d_model <= 0 or args.d_ff <= 0 or args.device_budget_mb < 0 or args.host_dequant_budget_mb < 0:
        p.error("Dimensions must be positive and budgets nonnegative")
    qk, _ = BLOCK_FORMATS[args.format]
    if args.d_model % qk or args.d_ff % qk:
        p.error(f"d-model and d-ff must be divisible by {qk} for {args.format}")
    if args.idle_eviction_requests < 0:
        p.error("--idle-eviction-requests must be nonnegative")
    if args.workers <= 0:
        p.error("--workers must be a positive integer")

    @lru_cache(maxsize=None)
    def bank(key):
        seed = 12_000 + int(key.expert_id) * 3
        return (packed_weight(args.format, args.d_ff, args.d_model, seed),
                packed_weight(args.format, args.d_ff, args.d_model, seed + 1),
                packed_weight(args.format, args.d_model, args.d_ff, seed + 2))

    raw = np.random.default_rng(123).normal(0, .02, (1, 1, args.d_model)).astype(np.float32)
    ids = np.arange(args.top_k, dtype=np.int32).reshape(1, 1, -1)
    mix = np.full((1, 1, args.top_k), 1 / args.top_k, dtype=np.float32)
    activation_dtype = xp.float16 if BACKEND_NAME == "cupy" else np.float32
    inputs = xp.asarray(raw, dtype=activation_dtype)
    weights = xp.asarray(mix, dtype=activation_dtype)
    indices = xp.asarray(ids)

    baseline, sync_ms, sync_stats = run(args, 0, bank, inputs, weights, indices)
    overlapped, async_ms, async_stats = run(args, args.workers, bank, inputs, weights, indices)
    first = xp.asnumpy(baseline) if BACKEND_NAME == "cupy" else baseline
    second = xp.asnumpy(overlapped) if BACKEND_NAME == "cupy" else overlapped
    np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-5)
    print(f"backend={BACKEND_NAME} format={args.format} d={args.d_model} ff={args.d_ff} "
          f"k={args.top_k} device_budget_mb={args.device_budget_mb}")
    print(f"synchronous mean latency: {sync_ms * 1e3:.2f} ms per layer/token")
    print(f"overlapped  mean latency: {async_ms * 1e3:.2f} ms per layer/token")
    print(f"ratio sync/overlap:        {sync_ms / async_ms:.3f}x")
    print(f"sync cache: {sync_stats}")
    print(f"async cache: {async_stats}")
    print("NOTE: one synthetic MoE layer, not a full-model decoding benchmark.")


if __name__ == "__main__":
    main()
