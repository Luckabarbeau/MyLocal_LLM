"""0106I deterministic single-layer profiling overhead and cache replay.

Routes are synthetic locality phases, NOT measurements of Python/C++ workloads
or full-model generation. All modes receive identical packed weights, inputs,
and routes. No speculative decoding. VRAM budget is explicit, never auto-sized.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.benchmark_0106d_heterogeneous_moe import packed_weight
from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.routing_analysis import analyze_trace
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS


def sync():
    if BACKEND_NAME == "cupy":
        xp.cuda.get_current_stream().synchronize()


def run(args, mode, bank, inputs, weights, routes):
    executor = HeterogeneousExpertExecutor(
        bank, model_id="synthetic-0106i-v1", d_model=args.d_model, d_ff=args.d_ff,
        cpu_workers=args.workers, cpu_packed_backend=args.cpu_backend,
        compute_dtype="float32", device_budget_bytes=args.device_budget_mib * 1024**2,
        promote_after=3,
    )
    layer = ExpertsInference(args.d_model, args.d_ff, args.n_experts,
                             expert_executor=executor, layer_id=0)
    trace = None
    outputs = []
    try:
        # Separate cache/worker/kernel warm-up from the measured trace.
        for i in range(args.warmup):
            layer.forward(inputs[i % args.tokens], weights, routes[i % args.tokens])
        sync()
        if mode != "disabled":
            trace = RoutingTrace(model_id="synthetic-0106i-v1", quantization=args.format,
                                 expert_mapping="identity-v1", workload="synthetic-locality-phases",
                                 mode=mode, max_records=args.tokens * args.n_experts,
                                 max_pending_events=args.tokens * args.n_experts * 4)
        executor.routing_profiler = trace
        start = time.perf_counter()
        for x, ids in zip(inputs, routes):
            outputs.append(layer.forward(x, weights, ids))
        sync()
        elapsed = time.perf_counter() - start
        output = asnumpy(xp.concatenate(outputs))
        snapshot = trace.snapshot() if trace is not None else None
        return output, dict(mode=mode, elapsed_seconds=elapsed,
                            mean_layer_token_ms=elapsed * 1000 / args.tokens,
                            layer_tokens_per_second=args.tokens / elapsed,
                            execution=executor.statistics(), trace=snapshot)
    finally:
        executor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--format", choices=list(BLOCK_FORMATS), default="Q4_0")
    parser.add_argument("--cpu-backend", choices=["reference", "native", "auto"], default="native")
    parser.add_argument("--device-budget-mib", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qk, _ = BLOCK_FORMATS[args.format]
    if (args.tokens < 1 or args.warmup < 0 or args.workers < 0 or
            args.device_budget_mib < 0 or args.d_model < 1 or args.d_ff < 1 or
            args.d_model % qk or args.d_ff % qk or
            not 1 <= args.top_k <= args.n_experts):
        parser.error("Invalid dimensions, top-k, iterations, workers, or budget")

    @lru_cache(maxsize=None)
    def bank(key):
        seed = 106090 + key.expert_id * 3
        return (packed_weight(args.format, args.d_ff, args.d_model, seed),
                packed_weight(args.format, args.d_ff, args.d_model, seed + 1),
                packed_weight(args.format, args.d_model, args.d_ff, seed + 2))

    rng = np.random.default_rng(10609)
    inputs = [xp.asarray(rng.normal(0, .02, (1, 1, args.d_model)).astype(np.float32))
              for _ in range(args.tokens)]
    routes = []
    phase = max(1, args.tokens // 4)
    for i in range(args.tokens):
        hot = (i // phase) * args.top_k
        candidates = (np.arange(args.n_experts) + hot) % args.n_experts
        if i % 8:
            ids = candidates[:args.top_k]
        else:
            ids = rng.choice(args.n_experts, args.top_k, replace=False)
        routes.append(xp.asarray(ids.astype(np.int32).reshape(1, 1, -1)))
    weights = xp.full((1, 1, args.top_k), 1 / args.top_k, dtype=xp.float32)
    baseline = None
    results = []
    for mode in ("disabled", "summary", "detailed"):
        output, result = run(args, mode, bank, inputs, weights, routes)
        if baseline is None:
            baseline = output
        # Placement is unchanged, so even CPU/GPU quantization differences
        # cannot justify an instrumentation-dependent result here.
        np.testing.assert_array_equal(output, baseline)
        result["max_abs_error_vs_disabled"] = float(np.max(np.abs(output - baseline)))
        results.append(result)
        print(f"{mode}: {result['mean_layer_token_ms']:.4f} ms/layer-token, "
              f"{result['layer_tokens_per_second']:.2f} layer-tokens/s, max_abs_error=0")
    detailed = results[-1]["trace"]
    replay = analyze_trace(detailed, capacities=tuple(sorted({0, 1, 2, 4, args.n_experts})))
    payload = dict(benchmark="0106I", backend=BACKEND_NAME,
                   scope="Synthetic single MoE layer; not full-model decode throughput",
                   comparison_order="disabled, summary, detailed; fixed order, no statistical speedup claim",
                   arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   results=results, offline_analysis=replay)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}; trace includes warm cache but excludes warmup observations.")


if __name__ == "__main__":
    main()
