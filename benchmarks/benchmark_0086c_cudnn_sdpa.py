#!/usr/bin/env python3
"""PATCH 0086C: full forward+backward SDPA feasibility benchmark.

The production grouped-cuBLAS path and the opt-in 0086B fused-forward path are
measured through their public APIs.  Optional cuDNN candidates are never
emulated: if a real cuDNN Frontend forward+backward route is unavailable, the
report records the exact reason and keeps their timings unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from cudnn_sdpa_0086c import (
    ATTENTION_SCALE,
    D_HEAD,
    LOCAL_WINDOW,
    kv_split_plan,
    logical_bhsd_metadata,
    probe_cudnn_environment,
    regular16_plan,
    resolve_active_heads,
)


_CANDIDATES = ("grouped", "0086b", "cudnn_regular16", "cudnn_kv_split")
_PHASES = (
    "local_forward",
    "local_backward",
    "local_fwd_bwd",
    "full_forward",
    "full_backward",
    "full_fwd_bwd",
    "composite_forward",
    "composite_backward",
    "composite_fwd_bwd",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seq", type=int, default=2304)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument(
        "--heads",
        default="15",
        choices=("15", "13", "9", "scattered"),
        help="active original local-Q-head pattern",
    )
    parser.add_argument("--window", type=int, default=LOCAL_WINDOW)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--compare",
        default=",".join(_CANDIDATES),
        help="comma-separated candidates: " + ",".join(_CANDIDATES),
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--output-json", "--json-out", dest="output_json", type=Path)
    parser.add_argument("--output-csv", "--csv-out", dest="output_csv", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.seq, args.window, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch, seq, window and repeats must be positive; warmup must be nonnegative")
    requested = tuple(part.strip() for part in args.compare.split(",") if part.strip())
    unknown = tuple(name for name in requested if name not in _CANDIDATES)
    if unknown:
        parser.error(f"unknown candidates: {', '.join(unknown)}")
    if not requested:
        parser.error("--compare must request at least one candidate")
    args.compare = requested
    return args


def _summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        "median_ms": median,
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "std_ms": float(np.std(values)),
        "mad_ms": float(np.median(np.abs(values - median))),
        "samples_ms": [float(value) for value in values],
    }


def _time_call(call, stream):
    import cupy as cp

    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    wall_start = time.perf_counter()
    start_event.record(stream)
    result = call()
    end_event.record(stream)
    end_event.synchronize()
    return (
        float(cp.cuda.get_elapsed_time(start_event, end_event)),
        (time.perf_counter() - wall_start) * 1000.0,
        result,
    )


def _time_candidates(candidate_calls, warmup, repeats):
    import cupy as cp

    names = tuple(candidate_calls)
    for _ in range(warmup):
        for name in names:
            candidate_calls[name]()
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    device = {name: [] for name in names}
    wall = {name: [] for name in names}
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            device_ms, wall_ms, _ = _time_call(candidate_calls[name], stream)
            device[name].append(device_ms)
            wall[name].append(wall_ms)
    return {
        name: {"device": _summary(device[name]), "wall": _summary(wall[name])}
        for name in names
    }


def _memory_snapshot():
    import cupy as cp

    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_reserved_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def _error_metrics(actual, reference):
    import cupy as cp

    actual32 = actual.astype(cp.float32, copy=False)
    reference32 = reference.astype(cp.float32, copy=False)
    difference = actual32 - reference32
    absolute = cp.abs(difference)
    flat_worst = int(cp.argmax(absolute).item())
    worst = tuple(int(value) for value in np.unravel_index(flat_worst, actual.shape))
    actual_flat = actual32.reshape(-1)
    reference_flat = reference32.reshape(-1)
    reference_norm = cp.linalg.norm(reference_flat)
    product_norm = cp.linalg.norm(actual_flat) * reference_norm
    equal = bool(cp.array_equal(actual32, reference32).item())
    cosine = 1.0 if float(product_norm.item()) == 0.0 and equal else (
        float((cp.dot(actual_flat, reference_flat) / product_norm).item())
        if float(product_norm.item()) != 0.0 else 0.0
    )
    return {
        "shape": [int(value) for value in actual.shape],
        "max_abs": float(absolute.reshape(-1)[flat_worst].item()),
        "rmse": float(cp.sqrt(cp.mean(difference * difference)).item()),
        "relative_l2": float(
            (cp.linalg.norm(difference.reshape(-1)) / cp.maximum(reference_norm, 1e-30)).item()
        ),
        "cosine": cosine,
        "signed_bias": float(cp.mean(difference).item()),
        "worst_index": list(worst),
        "actual_at_worst": float(actual32[worst].item()),
        "reference_at_worst": float(reference32[worst].item()),
        "finite": bool(cp.all(cp.isfinite(actual32)).item()),
    }


def _selection(heads):
    if heads == tuple(range(heads[0], heads[0] + len(heads))):
        return slice(heads[0], heads[0] + len(heads))
    return heads


def _build_existing_candidate(name, q_all, k, v, d_out_all, active_heads, window):
    import cupy as cp

    from mini_llm.ops.indexed_attention import (
        local_window_attention_backward,
        local_window_attention_forward,
    )

    selection = _selection(active_heads)
    q_local = q_all[:, :, selection, :]
    d_local = d_out_all[:, :, selection, :]
    local_mapping = tuple(head // 4 for head in active_heads)
    q_full = q_all[:, :, 0:1, :]
    d_full = d_out_all[:, :, 0:1, :]
    local_mode = "grouped" if name == "grouped" else "fused"

    def local_forward():
        os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = local_mode
        if local_mode == "fused":
            os.environ["MINI_LLM_TRAIN_ATTN_STRICT"] = "1"
        return local_window_attention_forward(
            q_local,
            k,
            v,
            window,
            kv_head_indices=local_mapping,
            scale=ATTENTION_SCALE,
            return_cache=True,
        )

    def full_forward():
        os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = "grouped"
        return local_window_attention_forward(
            q_full,
            k,
            v,
            int(q_all.shape[1]),
            kv_head_indices=(0,),
            scale=ATTENTION_SCALE,
            return_cache=True,
        )

    initial_local = local_forward()
    initial_full = full_forward()

    def local_backward():
        return local_window_attention_backward(d_local, initial_local[1])

    def full_backward():
        return local_window_attention_backward(d_full, initial_full[1])

    def local_fwd_bwd():
        output, cache = local_forward()
        return output, local_window_attention_backward(d_local, cache)

    def full_fwd_bwd():
        output, cache = full_forward()
        return output, local_window_attention_backward(d_full, cache)

    def composite_forward():
        full_output, full_cache = full_forward()
        local_output, local_cache = local_forward()
        context = cp.zeros(q_all.shape, dtype=cp.float32)
        context[:, :, 0:1, :] = full_output
        context[:, :, selection, :] = local_output
        return context, (full_cache, local_cache)

    initial_composite = composite_forward()

    def _composite_backward(caches):
        full_gradients = local_window_attention_backward(d_full, caches[0])
        local_gradients = local_window_attention_backward(d_local, caches[1])
        d_q = cp.zeros(q_all.shape, dtype=cp.float32)
        d_q[:, :, 0:1, :] = full_gradients[0]
        d_q[:, :, selection, :] = local_gradients[0]
        return d_q, full_gradients[1] + local_gradients[1], full_gradients[2] + local_gradients[2]

    def composite_backward():
        return _composite_backward(initial_composite[1])

    def composite_fwd_bwd():
        context, caches = composite_forward()
        return context, _composite_backward(caches)

    calls = {
        "local_forward": local_forward,
        "local_backward": local_backward,
        "local_fwd_bwd": local_fwd_bwd,
        "full_forward": full_forward,
        "full_backward": full_backward,
        "full_fwd_bwd": full_fwd_bwd,
        "composite_forward": composite_forward,
        "composite_backward": composite_backward,
        "composite_fwd_bwd": composite_fwd_bwd,
    }
    return {
        "name": name,
        "status": "available",
        "calls": calls,
        "initial_local": initial_local,
        "initial_full": initial_full,
        "initial_composite": initial_composite,
        "reference_result": composite_fwd_bwd(),
        "layout": {
            "q_local_bthd_shape": [int(value) for value in q_local.shape],
            "q_local_bthd_strides_bytes": [int(value) for value in q_local.strides],
            "q_local_c_contiguous": bool(q_local.flags.c_contiguous),
            "q_local_logical_bhsd": logical_bhsd_metadata(q_local),
            "k_logical_bhsd": logical_bhsd_metadata(k),
            "v_logical_bhsd": logical_bhsd_metadata(v),
            "d_out_dtype": str(d_local.dtype),
        },
        "cache_contract": {
            "local_attention_backend": initial_local[1].get("attention_backend", "grouped"),
            "grouped_cublas_local": bool(initial_local[1].get("grouped_cublas_local", False)),
            "bf16_pipeline": bool(initial_local[1].get("bf16_pipeline", False)),
            "recompute_local_probs": bool(initial_local[1].get("recompute_local_probs", False)),
            "query_chunk_size": int(initial_local[1]["query_chunk_size"]),
            "output_dtype": str(initial_local[0].dtype),
            "gradient_dtypes": [str(value.dtype) for value in local_backward()],
        },
    }


def _gpu_capabilities():
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name")
    return {
        "cupy": cp.__version__,
        "device_name": name.decode() if isinstance(name, bytes) else str(name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
    }


def _write_csv(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "candidate", "status", "reason", "batch", "seq", "head_pattern",
        "active_heads", "window", "phase", "device_median_ms", "device_p10_ms",
        "device_p90_ms", "device_std_ms", "device_mad_ms", "wall_median_ms",
        "workspace_bytes", "pool_used_bytes", "pool_reserved_bytes",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in report["candidates"]:
            phases = candidate.get("timings") or {"": None}
            for phase, timing in phases.items():
                device = (timing or {}).get("device", {})
                wall = (timing or {}).get("wall", {})
                memory = candidate.get("steady_state_memory") or {}
                writer.writerow({
                    "candidate": candidate["candidate"],
                    "status": candidate["status"],
                    "reason": candidate.get("reason"),
                    "batch": report["shape"]["batch"],
                    "seq": report["shape"]["seq"],
                    "head_pattern": report["shape"]["head_pattern"],
                    "active_heads": len(report["shape"]["active_local_heads"]),
                    "window": report["shape"]["window"],
                    "phase": phase,
                    "device_median_ms": device.get("median_ms"),
                    "device_p10_ms": device.get("p10_ms"),
                    "device_p90_ms": device.get("p90_ms"),
                    "device_std_ms": device.get("std_ms"),
                    "device_mad_ms": device.get("mad_ms"),
                    "wall_median_ms": wall.get("median_ms"),
                    "workspace_bytes": candidate.get("workspace_bytes"),
                    "pool_used_bytes": memory.get("pool_used_bytes"),
                    "pool_reserved_bytes": memory.get("pool_reserved_bytes"),
                })


def _emit(report, output_json, output_csv):
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    if output_csv is not None:
        _write_csv(output_csv, report)


def main():
    args = _parse_args()
    capability = probe_cudnn_environment()
    base_report = {
        "patch": "0086C",
        "experiment": "benchmark-only; no production backend integration",
        "cudnn": capability,
        "requested_candidates": list(args.compare),
        "mask_contract": {
            "alignment": "TOP_LEFT",
            "inclusive_allowed_keys": "max(0, i-W+1) <= j <= i",
            "sentinel": {"query": 519, "excluded_key": 7, "included_key": 8, "window": 512},
            "required_pygraph_diagonal_band_left_bound": 512,
            "installed_api_verified": False,
            "timing_valid_without_mask_verification": False,
        },
    }
    if args.probe_only:
        base_report.update({"gpu": _gpu_capabilities(), "candidates": []})
        _emit(base_report, args.output_json, args.output_csv)
        return

    os.environ.setdefault("MINI_LLM_BACKEND", "cupy")
    os.environ.setdefault("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "1")
    os.environ.setdefault("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT", "0")
    os.environ.setdefault("MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD", "1")
    os.environ.setdefault("MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_CUBLAS_MODE", "grouped")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "auto")
    os.environ.setdefault("MINI_LLM_BF16_MIXED_CONTEXT", "0")
    os.environ.setdefault("MINI_LLM_FUSED_LOCAL_TILE_N", "16")

    import cupy as cp

    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    profile = apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    active_heads = resolve_active_heads(args.heads)
    window = min(args.window, args.seq)
    rng = np.random.default_rng(args.seed)
    q_all = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 16, D_HEAD)).astype(np.float32),
        dtype="bfloat16",
    )
    k = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, D_HEAD)).astype(np.float32),
        dtype="bfloat16",
    )
    v = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, D_HEAD)).astype(np.float32),
        dtype="bfloat16",
    )
    d_out_all = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 16, D_HEAD)).astype(np.float32),
        dtype=cp.float32,
    )
    cp.cuda.get_current_stream().synchronize()

    available = {}
    records = {}
    for name in args.compare:
        if name.startswith("cudnn_"):
            strategy = regular16_plan(active_heads) if name == "cudnn_regular16" else kv_split_plan(active_heads)
            records[name] = {
                "candidate": name,
                "status": "unavailable",
                "reason": capability["reason"],
                "strategy": strategy,
                "kernel_only": {"status": "unavailable", "reason": capability["reason"]},
                "effective_end_to_end": {"status": "unavailable", "reason": capability["reason"]},
                "workspace_bytes": None,
                "selected_plan": None,
                "mask_verified": False,
            }
            continue
        build_start = time.perf_counter()
        try:
            candidate = _build_existing_candidate(
                name, q_all, k, v, d_out_all, active_heads, window
            )
            cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            records[name] = {
                "candidate": name,
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            if name == "grouped":
                raise
            continue
        candidate["build_and_first_execution_ms"] = (
            time.perf_counter() - build_start
        ) * 1000.0
        available[name] = candidate
        records[name] = {
            "candidate": name,
            "status": "available",
            "reason": None,
            "build_and_first_execution_ms": candidate["build_and_first_execution_ms"],
            "layout": candidate["layout"],
            "cache_contract": candidate["cache_contract"],
            "workspace_bytes": None,
            "kernel_only": {
                "status": "not_separately_exposed",
                "reason": "public repository path measured as the effective baseline",
            },
        }

    if available:
        for phase in _PHASES:
            phase_timings = _time_candidates(
                {name: candidate["calls"][phase] for name, candidate in available.items()},
                args.warmup,
                args.repeats,
            )
            for name, timing in phase_timings.items():
                records[name].setdefault("timings", {})[phase] = timing
        for name, candidate in available.items():
            live_result = candidate["calls"]["composite_fwd_bwd"]()
            cp.cuda.get_current_stream().synchronize()
            records[name]["steady_state_memory"] = _memory_snapshot()
            records[name]["peak_memory"] = {
                "status": "not_available_from_cupy_pool_snapshot",
                "note": "steady-state live allocation and reservation are reported instead",
            }
            records[name]["effective_end_to_end"] = {
                "status": "measured",
                "phase": "composite_fwd_bwd",
                "includes": [
                    "full-head forward/backward",
                    "selected local-head forward/backward",
                    "FP32 context composition",
                    "dQ scatter",
                    "shared dK/dV summation",
                    "public Python dispatch and output allocations",
                ],
            }
            del live_result

    if "grouped" in available:
        grouped_output, grouped_gradients = available["grouped"]["reference_result"]
        records["grouped"]["numerical_vs_grouped"] = {
            "output": _error_metrics(grouped_output, grouped_output),
            "d_q": _error_metrics(grouped_gradients[0], grouped_gradients[0]),
            "d_k": _error_metrics(grouped_gradients[1], grouped_gradients[1]),
            "d_v": _error_metrics(grouped_gradients[2], grouped_gradients[2]),
        }
        for name, candidate in available.items():
            if name == "grouped":
                continue
            output, gradients = candidate["reference_result"]
            records[name]["numerical_vs_grouped"] = {
                "output": _error_metrics(output, grouped_output),
                "d_q": _error_metrics(gradients[0], grouped_gradients[0]),
                "d_k": _error_metrics(gradients[1], grouped_gradients[1]),
                "d_v": _error_metrics(gradients[2], grouped_gradients[2]),
            }
            grouped_ms = records["grouped"]["timings"]["composite_fwd_bwd"]["device"]["median_ms"]
            candidate_ms = records[name]["timings"]["composite_fwd_bwd"]["device"]["median_ms"]
            records[name]["grouped_over_candidate_composite_fwd_bwd"] = grouped_ms / candidate_ms

    base_report.update({
        "gpu": _gpu_capabilities(),
        "dtype": "bfloat16 inputs; FP32 context/upstream/gradients",
        "shape": {
            "batch": args.batch,
            "seq": args.seq,
            "d_head": D_HEAD,
            "n_q_heads": 16,
            "n_kv_heads": 4,
            "head_pattern": args.heads,
            "active_local_heads": list(active_heads),
            "window": window,
            "attention_scale": ATTENTION_SCALE,
        },
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runtime_profile": {
            "requested": profile.requested,
            "resolved": profile.resolved,
            "applied": list(profile.applied),
        },
        "head_strategies": {
            "cudnn_regular16": regular16_plan(active_heads),
            "cudnn_kv_split": kv_split_plan(active_heads),
        },
        "acceptance_tolerances_declared_before_vendor_timing": {
            "output": {"rtol": 0.008, "atol": 0.004},
            "d_q_d_k_d_v": {"relative_l2_max": 0.02, "cosine_min": 0.999},
            "mask_sentinel": "exact exclusion for j=7 and nonzero inclusion for j=8",
        },
        "candidates": [records[name] for name in args.compare],
        "decision": {
            "0086d": "no-go",
            "reason": (
                "no importable cuDNN Frontend exists in the active environment, so no real "
                "cuDNN forward+backward plan or mask contract could be validated or timed"
            ),
            "screening_threshold": "20-30% faster complete attention F+B after bridges",
        },
    })
    _emit(base_report, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
