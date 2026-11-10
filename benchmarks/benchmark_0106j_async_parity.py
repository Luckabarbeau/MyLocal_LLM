"""0106J: bounded async-buffer ownership regression, not full-model throughput.

All cases share the original 0106I seeds, packed weights, inputs and routes.
There is deliberately no per-token output copy or synchronization: doing so
hides the cross-stream allocator race this benchmark is intended to detect.

Example (also usable with MINI_LLM_BACKEND=numpy):
  MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 python \
      benchmarks/benchmark_0106j_async_parity.py --output /tmp/0106j.json
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.benchmark_0106d_heterogeneous_moe import packed_weight
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference

MODEL_ID = "synthetic-0106i-v1"
DEVICE_BUDGET_BYTES = 1024**2
RTOL = 1e-4
ATOL = 1e-7
MODES = ("disabled", "summary", "detailed")


@dataclass(frozen=True)
class Fixture:
    d_model: int
    d_ff: int
    n_experts: int
    top_k: int
    inputs: np.ndarray
    routes: np.ndarray
    mixing_weights: np.ndarray
    bank: dict

    @property
    def tokens(self):
        return len(self.inputs)


def build_fixture(*, tokens=128, d_model=256, d_ff=256, n_experts=16, top_k=4):
    """Build host-only Q4_0 data; importing/building never touches CUDA."""
    if (tokens < 1 or d_model < 1 or d_ff < 1 or d_model % 32 or d_ff % 32
            or not 1 <= top_k <= n_experts):
        raise ValueError("Require positive Q4_0 widths, tokens and 1 <= top_k <= n_experts")
    bank = {}
    for expert in range(n_experts):
        seed = 106090 + expert * 3
        bank[ExpertKey(MODEL_ID, 0, expert)] = (
            packed_weight("Q4_0", d_ff, d_model, seed),
            packed_weight("Q4_0", d_ff, d_model, seed + 1),
            packed_weight("Q4_0", d_model, d_ff, seed + 2),
        )
    rng = np.random.default_rng(10609)
    # Preserve RNG consumption order from benchmark_0106i_routing_trace.py.
    inputs = np.stack([rng.normal(0, .02, (1, 1, d_model)).astype(np.float32)
                       for _ in range(tokens)])
    routes = []
    phase = max(1, tokens // 4)
    for i in range(tokens):
        hot = (i // phase) * top_k
        candidates = (np.arange(n_experts) + hot) % n_experts
        ids = candidates[:top_k] if i % 8 else rng.choice(n_experts, top_k, replace=False)
        routes.append(ids.astype(np.int32).reshape(1, 1, top_k))
    routes = np.stack(routes)
    mixing_weights = np.full((1, 1, top_k), 1 / top_k, dtype=np.float32)
    for array in (inputs, routes, mixing_weights):
        array.flags.writeable = False
    return Fixture(d_model, d_ff, n_experts, top_k, inputs, routes, mixing_weights, bank)


def reference(fixture):
    """Independent NumPy dequant/SwiGLU oracle, accumulated in sorted ID order."""
    x = fixture.inputs.reshape(fixture.tokens, fixture.d_model)
    ids = fixture.routes.reshape(fixture.tokens, fixture.top_k)
    mix = fixture.mixing_weights.reshape(fixture.top_k)
    output = np.zeros_like(x)
    for expert in range(fixture.n_experts):
        rows, slots = np.nonzero(ids == expert)
        if not len(rows):
            continue
        gate, up, down = (w.dequantize_reference() for w in
                          fixture.bank[ExpertKey(MODEL_ID, 0, expert)])
        g = x[rows] @ gate.T
        u = x[rows] @ up.T
        expert_y = ((g / (1.0 + np.exp(-g))) * u) @ down.T
        output[rows] += mix[slots, None] * expert_y
    assert output.dtype == np.float32
    return output.reshape(fixture.tokens, 1, fixture.d_model)


def comparison(actual, expected):
    """JSON-safe error diagnostics, including nonfinite/contract failures."""
    contract = actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    result = dict(passed=False, shape=list(actual.shape), dtype=str(actual.dtype),
                  expected_shape=list(expected.shape), expected_dtype=str(expected.dtype),
                  finite=finite, max_abs_error=None, max_relative_error=None,
                  max_tolerance_ratio=None, rtol=RTOL, atol=ATOL)
    if contract and finite:
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        magnitude = np.abs(expected.astype(np.float64))
        result.update(
            passed=bool(np.all(error <= ATOL + RTOL * magnitude)),
            max_abs_error=float(error.max()),
            max_relative_error=float((error / np.maximum(magnitude, np.finfo(np.float32).tiny)).max()),
            max_tolerance_ratio=float((error / (ATOL + RTOL * magnitude)).max()),
        )
    return result


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("0106J regression exceeded its wall-time budget")


def run_case(fixture, *, mode="disabled", workers=2, residency="fixed",
             caller_stream="default", cpu_backend="auto", deadline=None):
    """Return (host output, measurements); synchronize only after a complete run.

    Fixed placement preloads IDs 0..8 then disables subsequent promotion. Dynamic
    placement performs eight warmup tokens with promote_after=3. NumPy callers
    must select dynamic/default: fixed GPU residency is not simulated on CPU.
    A deadline is checked on host without adding CUDA synchronization.
    """
    if mode not in MODES or residency not in ("fixed", "dynamic"):
        raise ValueError("Unknown tracing mode or residency policy")
    if caller_stream not in ("default", "non-default"):
        raise ValueError("Unknown caller stream")
    gpu = BACKEND_NAME == "cupy"
    if not gpu and (residency == "fixed" or caller_stream != "default"):
        raise ValueError("Fixed GPU residency and non-default CUDA streams require CuPy")
    fixed_count = 9
    if residency == "fixed":
        if fixture.n_experts <= fixed_count:
            raise ValueError("Fixed mixed residency requires at least ten experts")
        resident_bytes = sum(w.nbytes for e in range(fixed_count)
                             for w in fixture.bank[ExpertKey(MODEL_ID, 0, e)])
        if resident_bytes > DEVICE_BUDGET_BYTES:
            raise ValueError("Nine fixed experts exceed the bounded 1 MiB GPU cache")
    stream = (xp.cuda.Stream.null if caller_stream == "default"
              else xp.cuda.Stream(non_blocking=True)) if gpu else None
    executor = HeterogeneousExpertExecutor(
        fixture.bank.__getitem__, model_id=MODEL_ID, d_model=fixture.d_model,
        d_ff=fixture.d_ff, compute_dtype="float32", cpu_workers=workers,
        cpu_packed_backend=cpu_backend, device_budget_bytes=DEVICE_BUDGET_BYTES if gpu else 0,
        # Bounded host cache also keeps reference-only CPU runs practical.
        host_dequant_budget_bytes=fixture.n_experts * 3 * fixture.d_model * fixture.d_ff * 4,
        promote_after=1 if residency == "fixed" else 3,
    )
    layer = ExpertsInference(fixture.d_model, fixture.d_ff, fixture.n_experts,
                             expert_executor=executor, layer_id=0)
    completed = False
    try:
        with stream if gpu else nullcontext():
            inputs = [xp.asarray(x) for x in fixture.inputs]
            routes = [xp.asarray(ids) for ids in fixture.routes]
            mix = xp.asarray(fixture.mixing_weights)
            assert all(x.dtype == np.float32 and x.shape == (1, 1, fixture.d_model)
                       for x in inputs)
            assert mix.dtype == np.float32 and mix.shape == (1, 1, fixture.top_k)
            assert all(ids.dtype == np.int32 and ids.shape == (1, 1, fixture.top_k)
                       for ids in routes)
            if residency == "fixed":
                for expert in range(fixed_count):
                    _check_deadline(deadline)
                    executor.forward(0, expert, inputs[0].reshape(1, fixture.d_model))
                executor.cache.promote_after = 10**12
                assert {key.expert_id for key in executor.cache.resident} == set(range(9))
            else:
                for i in range(8):
                    _check_deadline(deadline)
                    layer.forward(inputs[i % fixture.tokens], mix, routes[i % fixture.tokens])
            before = executor.statistics()
            trace = None if mode == "disabled" else RoutingTrace(
                model_id=MODEL_ID, quantization="Q4_0", expert_mapping="identity-v1",
                workload="synthetic-locality-phases", mode=mode,
                max_records=fixture.tokens * fixture.top_k,
                max_pending_events=fixture.tokens * fixture.top_k * 4,
            )
            executor.routing_profiler = trace
            outputs = []
            start = time.perf_counter()
            for x, ids in zip(inputs, routes):
                _check_deadline(deadline)
                output = layer.forward(x, mix, ids)
                assert output.dtype == np.float32 and output.shape == x.shape
                outputs.append(output)
            joined = xp.concatenate(outputs, axis=0)
            if gpu:
                # Queue one terminal D2H on the caller stream, not one per token.
                host_output = np.empty(joined.shape, dtype=np.float32)
                joined.get(out=host_output, stream=stream, blocking=False)
                stream.synchronize()
            else:
                host_output = np.asarray(joined)
            completed = True
            elapsed = time.perf_counter() - start
            after = executor.statistics()
            return host_output, dict(
                mode=mode, workers=workers, residency=residency, caller_stream=caller_stream,
                status="completed", elapsed_seconds=elapsed,
                mean_layer_token_ms=elapsed * 1000 / fixture.tokens,
                layer_tokens_per_second=fixture.tokens / elapsed,
                gpu_cache_before=before["device"], gpu_cache_after=after["device"],
                gpu_cache_delta={key: after["device"].get(key, 0) - before["device"].get(key, 0)
                                 for key in sorted(before["device"].keys() | after["device"].keys())
                                 if isinstance(after["device"].get(key, 0), (int, float))},
                execution=after, trace=trace.snapshot() if trace is not None else None,
            )
    finally:
        executor.close()
        if gpu and not completed:
            # Error cleanup only; never reached between successful decode tokens.
            stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-backend", choices=("reference", "native", "auto"), default="auto")
    args = parser.parse_args()
    if args.repeats < 1 or any(w < 0 for w in args.workers):
        parser.error("Require positive repeats and nonnegative worker counts")
    try:
        fixture = build_fixture(tokens=args.tokens, d_model=args.d_model, d_ff=args.d_ff,
                                n_experts=args.n_experts, top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    oracle = reference(fixture)
    results, failures = [], []
    for residency in ("fixed", "dynamic"):
        for caller_stream in ("default", "non-default"):
            if BACKEND_NAME != "cupy" and (residency == "fixed" or caller_stream != "default"):
                results.append(dict(residency=residency, caller_stream=caller_stream,
                                    status="skipped", reason="GPU-only placement/stream semantics"))
                continue
            for workers in args.workers:
                for repeat in range(args.repeats):
                    baseline = None
                    for mode in MODES:
                        identity = dict(residency=residency, caller_stream=caller_stream,
                                        workers=workers, repeat=repeat, mode=mode)
                        try:
                            output, result = run_case(fixture, mode=mode, workers=workers,
                                residency=residency, caller_stream=caller_stream,
                                cpu_backend=args.cpu_backend)
                            result["repeat"] = repeat
                            result["cpu_oracle_error"] = comparison(output, oracle)
                            if mode == "disabled":
                                baseline = output
                            result["disabled_mode_error"] = (comparison(output, baseline)
                                if baseline is not None else None)
                            passed = (result["cpu_oracle_error"]["passed"]
                                      and result["disabled_mode_error"] is not None
                                      and result["disabled_mode_error"]["passed"])
                            if residency == "fixed":
                                passed = passed and result["gpu_cache_delta"].get("evictions", 0) == 0
                            result["passed"] = passed
                            if not passed:
                                failures.append(dict(**identity, reason="parity/placement failure"))
                            print(f"{identity}: {result['mean_layer_token_ms']:.3f} ms/layer-token; "
                                  f"oracle={result['cpu_oracle_error']['max_abs_error']}; "
                                  f"vs_disabled={result['disabled_mode_error']}; "
                                  f"GPU cache={result['gpu_cache_after']}")
                        except Exception as exc:
                            result = dict(**identity, status="failed", passed=False,
                                          exception=f"{type(exc).__name__}: {exc}")
                            failures.append(result.copy())
                        results.append(result)
    payload = dict(benchmark="0106J", backend=BACKEND_NAME, model_id=MODEL_ID,
                   seed=10609, weight_seed="106090 + expert * 3 + projection",
                   format="Q4_0", device_budget_bytes=DEVICE_BUDGET_BYTES,
                   tolerance=dict(rtol=RTOL, atol=ATOL),
                   scope="Synthetic single MoE layer; not full-model decode throughput",
                   timing_semantics="Warmup queued before timing; one terminal sync, no per-token sync",
                   arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   results=results, failures=failures, passed=not failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; failures={len(failures)}")
    if failures:
        raise AssertionError(f"0106J parity failed in {len(failures)} cases; see {args.output}")


if __name__ == "__main__":
    main()
