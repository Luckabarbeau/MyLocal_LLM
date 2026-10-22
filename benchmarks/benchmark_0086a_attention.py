#!/usr/bin/env python3
"""0086A benchmark for the existing local-attention training path.

The benchmark uses deterministic post-RoPE Q/K/V/dO tensors and the public
``local_window_attention_forward/backward`` API. It does not alter production
backend selection. Run with ``MINI_LLM_BACKEND=numpy`` or ``cupy`` set before
launch; GPU synchronization is confined to this benchmark.
"""

from __future__ import annotations

import argparse
import csv
import ctypes.util
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reference_fused_attention_oracle import backward_reference, forward_reference


@dataclass(frozen=True)
class Case:
    name: str
    original_heads: tuple[int, ...]
    window: int


CASES = {
    "full-head0": Case("full-head0", (0,), 4096),
    "local-all": Case("local-all", tuple(range(1, 16)), 512),
    "layer0-history-valid": Case(
        "layer0-history-valid", tuple(h for h in range(1, 16) if h not in (12, 13)), 512
    ),
    "layer4-history-valid": Case("layer4-history-valid", tuple(range(1, 10)), 512),
}

_ENV_NAMES = (
    "MINI_LLM_BACKEND",
    "MINI_LLM_RUNTIME_PROFILE",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT",
    "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD",
    "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS",
    "MINI_LLM_BF16_LOCAL_GEMM",
    "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE",
    "MINI_LLM_BF16_MIXED_CONTEXT",
    "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS",
    "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK",
    "MINI_LLM_LOCAL_ATTN_AUTO_WORK_BUDGET",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=129)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--dtype", choices=("auto", "float32", "float64", "bfloat16"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--runtime-profile", choices=("reference", "gpu-fast"), default="gpu-fast")
    parser.add_argument(
        "--head-selection",
        choices=("current", "basic-slice"),
        default="current",
        help="current reproduces tuple indexing; basic-slice models 0085I for contiguous groups",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=tuple(CASES),
    )
    parser.add_argument(
        "--oracle-max-seq",
        type=int,
        default=65,
        help="compare a deterministic prefix to FP64 oracle; 0 disables",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.seq, args.d_head, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch, seq, d-head and repeats must be positive; warmup must be nonnegative")
    return args


def _selection(original_heads, mode):
    if mode == "basic-slice" and original_heads:
        start = original_heads[0]
        if original_heads == tuple(range(start, start + len(original_heads))):
            return slice(start, start + len(original_heads))
    return original_heads


def _to_numpy(array, backend_name):
    if backend_name == "cupy":
        import cupy as cp

        return cp.asnumpy(array).astype(np.float64)
    return np.asarray(array, dtype=np.float64)


def _shares_memory(left, right, backend_name):
    try:
        if backend_name == "cupy":
            import cupy as cp

            return bool(cp.shares_memory(left, right))
        return bool(np.shares_memory(left, right))
    except (AttributeError, NotImplementedError):
        return None


def _error_metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = actual - reference
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat))
    norms = float(np.linalg.norm(actual_flat)) * denominator
    cosine = 1.0 if norms == 0.0 and np.array_equal(actual, reference) else (
        float(actual_flat @ reference_flat) / norms if norms else 0.0
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / max(denominator, 1e-30)),
        "signed_bias": float(np.mean(difference)),
        "cosine": cosine,
        "finite": bool(np.isfinite(actual).all()),
    }


def _percentiles(milliseconds):
    return {
        "median_ms": float(statistics.median(milliseconds)),
        "p10_ms": float(np.percentile(milliseconds, 10)),
        "p90_ms": float(np.percentile(milliseconds, 90)),
    }


def _time_cpu(call, warmup, repeats):
    for _ in range(warmup):
        call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return {"wall": _percentiles(elapsed)}


def _time_gpu(call, warmup, repeats):
    import cupy as cp

    for _ in range(warmup):
        call()
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    event_ms = []
    wall_ms = []
    for _ in range(repeats):
        start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter()
        start_event.record(stream)
        call()
        end_event.record(stream)
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        event_ms.append(float(cp.cuda.get_elapsed_time(start_event, end_event)))
    return {"device": _percentiles(event_ms), "wall": _percentiles(wall_ms)}


def _chunk_metadata(cache):
    chunks = cache.get("chunks", ())
    if not chunks:
        return {"layout": "none", "count": 0, "qk_shapes": []}
    if isinstance(chunks[0], dict):
        shapes = [[int(chunk["q_count"]), int(chunk["k_count"])] for chunk in chunks]
        layout = "grouped_dict"
    else:
        shapes = [[int(c[1] - c[0]), int(c[3] - c[2])] for c in chunks]
        layout = "fallback_tuple"
    return {"layout": layout, "count": len(chunks), "qk_shapes": shapes}


def _capabilities(backend_name):
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "cudnn_library": ctypes.util.find_library("cudnn"),
    }
    if backend_name != "cupy":
        return result
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name")
    result.update(
        {
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
            "compute_capability": str(cp.cuda.Device(0).compute_capability),
            "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "cudnn_python_frontend": importlib.util.find_spec("cudnn") is not None,
        }
    )
    return result


def _memory_snapshot(backend_name):
    if backend_name != "cupy":
        return None
    import cupy as cp

    pool = cp.get_default_memory_pool()
    return {"used_bytes": int(pool.used_bytes()), "reserved_bytes": int(pool.total_bytes())}


def _run_case(args, case, xp, backend_name, dtype, tensors):
    from mini_llm.ops.indexed_attention import (
        local_window_attention_backward,
        local_window_attention_forward,
    )

    q_all, k, v, d_out_all = tensors
    selection = _selection(case.original_heads, args.head_selection)
    q = q_all[:, :, selection, :]
    d_out = d_out_all[:, :, selection, :]
    kv_map = tuple(original_head // 4 for original_head in case.original_heads)
    window = min(case.window, args.seq)

    def call():
        output, cache = local_window_attention_forward(
            q,
            k,
            v,
            window,
            kv_head_indices=kv_map,
            return_cache=True,
            query_chunk_size=None,
        )
        gradients = local_window_attention_backward(d_out, cache)
        return output, cache, gradients

    memory_before = _memory_snapshot(backend_name)
    output, cache, gradients = call()
    if backend_name == "cupy":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    memory_live = _memory_snapshot(backend_name)
    timings = (
        _time_gpu(call, args.warmup, args.repeats)
        if backend_name == "cupy"
        else _time_cpu(call, args.warmup, args.repeats)
    )

    numerical = None
    oracle_length = min(args.seq, args.oracle_max_seq)
    if oracle_length > 0:
        q_ref = _to_numpy(q[:, :oracle_length], backend_name)
        k_ref = _to_numpy(k[:, :oracle_length], backend_name)
        v_ref = _to_numpy(v[:, :oracle_length], backend_name)
        d_out_ref = _to_numpy(d_out[:, :oracle_length], backend_name)
        windows = (min(window, oracle_length),) * len(case.original_heads)
        expected_output, _ = forward_reference(q_ref, k_ref, v_ref, kv_map, windows)
        expected_gradients = backward_reference(
            q_ref, k_ref, v_ref, d_out_ref, kv_map, windows
        )
        actual_output, actual_cache = local_window_attention_forward(
            q[:, :oracle_length],
            k[:, :oracle_length],
            v[:, :oracle_length],
            min(window, oracle_length),
            kv_head_indices=kv_map,
            return_cache=True,
            query_chunk_size=None,
        )
        actual_gradients = local_window_attention_backward(
            d_out[:, :oracle_length], actual_cache
        )
        if backend_name == "cupy":
            import cupy as cp

            cp.cuda.get_current_stream().synchronize()
        numerical = {
            "oracle_length": oracle_length,
            "output": _error_metrics(_to_numpy(actual_output, backend_name), expected_output),
            "d_q": _error_metrics(_to_numpy(actual_gradients[0], backend_name), expected_gradients[0]),
            "d_k": _error_metrics(_to_numpy(actual_gradients[1], backend_name), expected_gradients[1]),
            "d_v": _error_metrics(_to_numpy(actual_gradients[2], backend_name), expected_gradients[2]),
        }

    return {
        "case": case.name,
        "original_heads": list(case.original_heads),
        "kv_map": list(kv_map),
        "window": window,
        "q_shape": list(map(int, q.shape)),
        "q_strides_bytes": list(map(int, q.strides)),
        "k_strides_bytes": list(map(int, k.strides)),
        "v_strides_bytes": list(map(int, v.strides)),
        "q_c_contiguous": bool(q.flags.c_contiguous),
        "q_shares_base": _shares_memory(q_all, q, backend_name),
        "output_dtype": str(output.dtype),
        "gradient_dtypes": [str(gradient.dtype) for gradient in gradients],
        "grouped_cublas_local": bool(cache.get("grouped_cublas_local", False)),
        "bf16_pipeline": bool(cache.get("bf16_pipeline", False)),
        "recompute_local_probs": bool(cache.get("recompute_local_probs", False)),
        "query_chunk_size": int(cache["query_chunk_size"]),
        "chunks": _chunk_metadata(cache),
        "memory_before": memory_before,
        "memory_with_live_result": memory_live,
        "timings": timings,
        "numerical_vs_fp64": numerical,
    }


def _write_csv(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case",
        "backend",
        "dtype",
        "batch",
        "seq",
        "heads",
        "window",
        "grouped_cublas_local",
        "bf16_pipeline",
        "query_chunk_size",
        "device_median_ms",
        "device_p10_ms",
        "device_p90_ms",
        "wall_median_ms",
        "live_used_bytes",
        "live_reserved_bytes",
        "output_max_abs_vs_fp64",
        "dq_max_abs_vs_fp64",
        "dk_max_abs_vs_fp64",
        "dv_max_abs_vs_fp64",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            device = result["timings"].get("device", {})
            wall = result["timings"]["wall"]
            memory = result["memory_with_live_result"] or {}
            numerical = result["numerical_vs_fp64"] or {}
            writer.writerow(
                {
                    "case": result["case"],
                    "backend": report["backend"],
                    "dtype": report["dtype"],
                    "batch": report["batch"],
                    "seq": report["seq"],
                    "heads": len(result["original_heads"]),
                    "window": result["window"],
                    "grouped_cublas_local": result["grouped_cublas_local"],
                    "bf16_pipeline": result["bf16_pipeline"],
                    "query_chunk_size": result["query_chunk_size"],
                    "device_median_ms": device.get("median_ms"),
                    "device_p10_ms": device.get("p10_ms"),
                    "device_p90_ms": device.get("p90_ms"),
                    "wall_median_ms": wall["median_ms"],
                    "live_used_bytes": memory.get("used_bytes"),
                    "live_reserved_bytes": memory.get("reserved_bytes"),
                    "output_max_abs_vs_fp64": numerical.get("output", {}).get("max_abs"),
                    "dq_max_abs_vs_fp64": numerical.get("d_q", {}).get("max_abs"),
                    "dk_max_abs_vs_fp64": numerical.get("d_k", {}).get("max_abs"),
                    "dv_max_abs_vs_fp64": numerical.get("d_v", {}).get("max_abs"),
                }
            )


def main():
    args = _parse_args()
    os.environ.setdefault("MINI_LLM_BACKEND", "numpy")

    from mini_llm.backend import BACKEND_NAME, xp
    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    profile = apply_runtime_defaults(
        args.runtime_profile,
        backend_name=BACKEND_NAME,
        config=config,
        training=True,
        precision="bf16-mixed" if BACKEND_NAME == "cupy" else "fp32",
    )
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if BACKEND_NAME == "cupy" else "float32"
    if BACKEND_NAME == "numpy" and dtype == "bfloat16":
        raise ValueError("use float32/float64 for the NumPy benchmark")

    rng = np.random.default_rng(args.seed)
    host_tensors = (
        rng.normal(0.0, 0.2, (args.batch, args.seq, 16, args.d_head)).astype(np.float32),
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, args.d_head)).astype(np.float32),
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, args.d_head)).astype(np.float32),
        rng.normal(0.0, 0.2, (args.batch, args.seq, 16, args.d_head)).astype(np.float32),
    )
    tensors = tuple(xp.asarray(array, dtype=dtype) for array in host_tensors)
    report = {
        "patch": "0086A",
        "backend": BACKEND_NAME,
        "dtype": dtype,
        "batch": args.batch,
        "seq": args.seq,
        "d_head": args.d_head,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "head_selection": args.head_selection,
        "runtime_profile": {
            "requested": profile.requested,
            "resolved": profile.resolved,
            "applied": list(profile.applied),
        },
        "environment": {name: os.environ.get(name) for name in _ENV_NAMES},
        "capabilities": _capabilities(BACKEND_NAME),
        "model": {
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "n_q_heads": config.n_q_heads,
            "n_kv_heads": config.n_kv_heads,
            "d_head": config.d_head,
        },
        "results": [
            _run_case(args, CASES[name], xp, BACKEND_NAME, dtype, tensors)
            for name in args.cases
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.csv_out is not None:
        _write_csv(args.csv_out, report)


if __name__ == "__main__":
    main()
