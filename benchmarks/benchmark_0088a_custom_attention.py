#!/usr/bin/env python3
"""Benchmark the 0088A custom Triton mixed-attention forward/backward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.reference_0087b_torch_attention import (
    boundary_contract,
    error_metrics,
    load_fixture,
    resolve_active_heads,
)

_TENSOR_NAMES = ("context", "d_q", "d_k", "d_v")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--heads", default="15")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--reference-tensors", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    return args


def _summary(samples):
    values = [float(value) for value in samples]
    return {
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": float(np.std(values)),
    }


def _cupy_arrays(arrays, active_heads):
    import cupy as cp

    def bf16(name):
        return cp.asarray(arrays[name]).view(cp.dtype("bfloat16"))

    q = bf16("q_bits")
    k = bf16("k_bits")
    v = bf16("v_bits")
    d_out = cp.asarray(arrays["d_out"], dtype=cp.float32)
    inactive = sorted(set(range(1, 16)) - set(active_heads))
    if inactive:
        d_out[:, :, inactive, :] = 0.0
    return q, k, v, d_out


def _time_cupy(call, warmup, repeats):
    import cupy as cp

    for _ in range(warmup):
        result = call()
        del result
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    device, wall = [], []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        result = call()
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter_ns() - wall_start) / 1_000_000.0)
        device.append(float(cp.cuda.get_elapsed_time(start, end)))
        del result
    return {"gpu_event": _summary(device), "host_wall": _summary(wall)}


def _load_reference(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        tensors = {
            name: np.array(archive[name], dtype=np.float32, copy=True)
            for name in _TENSOR_NAMES
        }
    return metadata, tensors


def _compare(actual, reference_path, active_heads):
    metadata, reference = _load_reference(reference_path)
    metrics = {
        name: error_metrics(actual[name], reference[name]) for name in _TENSOR_NAMES
    }
    inactive = sorted(set(range(1, 16)) - set(active_heads))
    inactive_output = (
        0.0
        if not inactive
        else float(np.max(np.abs(actual["context"][:, :, inactive, :])))
    )
    inactive_dq = (
        0.0
        if not inactive
        else float(np.max(np.abs(actual["d_q"][:, :, inactive, :])))
    )
    accepted = (
        all(metrics[name]["relative_l2"] <= 0.02 for name in _TENSOR_NAMES)
        and all(metrics[name]["cosine"] >= 0.999 for name in _TENSOR_NAMES)
        and inactive_output == 0.0
        and inactive_dq == 0.0
    )
    return {
        "accepted": bool(accepted),
        "reference": str(reference_path),
        "reference_metadata": metadata,
        "metrics": metrics,
        "inactive_current_heads": inactive,
        "inactive_output_max_abs": inactive_output,
        "inactive_dq_max_abs": inactive_dq,
        "thresholds": {
            "relative_l2_max": 0.02,
            "cosine_min": 0.999,
            "inactive_output_and_dq": "exact zero",
        },
    }


def _save_tensors(path, tensors, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{name: np.asarray(tensors[name], dtype=np.float32) for name in _TENSOR_NAMES},
    )


def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("MINI_LLM_BACKEND", "cupy")

    import cupy as cp
    import triton
    import mini_llm.backend  # noqa: F401 - registers CuPy BF16 support

    from mini_llm.ops.triton_mixed_attention import (
        mixed_attention_backward,
        mixed_attention_forward,
    )

    manifest, arrays = load_fixture(args.fixture)
    active_heads = resolve_active_heads(args.heads)
    boundary = boundary_contract(active_heads)
    q, k, v, d_out = _cupy_arrays(arrays, active_heads)

    def forward():
        return mixed_attention_forward(q, k, v, active_heads)

    compile_wall_start = time.perf_counter_ns()
    initial_output, initial_cache = forward()
    initial_gradients = mixed_attention_backward(d_out, initial_cache)
    cp.cuda.get_current_stream().synchronize()
    cold_compile_and_first_fwd_bwd_ms = (
        time.perf_counter_ns() - compile_wall_start
    ) / 1_000_000.0
    del initial_gradients

    def backward():
        return mixed_attention_backward(d_out, initial_cache)

    def combined():
        output, cache = forward()
        return output, mixed_attention_backward(d_out, cache)

    timings = {
        "forward": _time_cupy(forward, args.warmup, args.repeats),
        "backward": _time_cupy(backward, args.warmup, args.repeats),
        "combined": _time_cupy(combined, args.warmup, args.repeats),
    }

    pool = cp.get_default_memory_pool()
    baseline_used = int(pool.used_bytes())
    baseline_reserved = int(pool.total_bytes())
    output, gradients = combined()
    cp.cuda.get_current_stream().synchronize()
    memory = {
        "baseline_pool_used_bytes": baseline_used,
        "baseline_pool_reserved_bytes": baseline_reserved,
        "post_combined_pool_used_bytes": int(pool.used_bytes()),
        "post_combined_pool_reserved_bytes": int(pool.total_bytes()),
        "peak_semantics": "CuPy pool reservation is a process high-water proxy",
    }
    tensors = {
        "context": cp.asnumpy(output),
        "d_q": cp.asnumpy(gradients[0]),
        "d_k": cp.asnumpy(gradients[1]),
        "d_v": cp.asnumpy(gradients[2]),
    }
    parity = _compare(tensors, args.reference_tensors, active_heads)
    if not parity["accepted"]:
        raise RuntimeError("0088A numerical parity gate failed; timings are invalid")

    if args.cuda_profiler_capture:
        cp.cuda.runtime.profilerStart()
        cp.cuda.nvtx.RangePush("0088A_CUSTOM_MIXED_FWD_BWD")
        try:
            profile_result = combined()
            cp.cuda.get_current_stream().synchronize()
        finally:
            cp.cuda.nvtx.RangePop()
            cp.cuda.runtime.profilerStop()
        del profile_result

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    report = {
        "patch": "0088A",
        "candidate": "custom-triton-mixed-attention",
        "status": "valid" if parity["accepted"] else "invalid",
        "fixture": str(args.fixture),
        "fixture_manifest": manifest,
        "active_local_heads": list(active_heads),
        "boundary_contract": boundary,
        "precision": {
            "qkv": "bfloat16",
            "upstream": "float32 loaded and rounded to bfloat16 inside dot products",
            "output": "float32",
            "gradients": "float32",
            "softmax_and_accumulation": "float32",
        },
        "capability": {
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "triton": triton.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device": (
                device_name.decode()
                if isinstance(device_name, bytes)
                else str(device_name)
            ),
            "compute_capability": str(cp.cuda.Device(0).compute_capability),
        },
        "cold_compile_and_first_fwd_bwd_host_ms": cold_compile_and_first_fwd_bwd_ms,
        "timings_ms": timings,
        "memory": memory,
        "parity_vs_grouped_cupy": parity,
        "integration_included": [
            "FP32 output and gradient allocation",
            "full-head and local-head custom launches",
            "online-softmax probability recomputation",
            "native shared-KV dK/dV accumulation",
            "inactive original-head masking",
        ],
    }
    _save_tensors(
        args.output_tensors,
        tensors,
        {
            "candidate": report["candidate"],
            "active_local_heads": list(active_heads),
            "fixture": str(args.fixture),
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
