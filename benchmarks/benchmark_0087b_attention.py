#!/usr/bin/env python3
"""0087B matched production CuPy vs compiled PyTorch attention F+B benchmark."""

from __future__ import annotations

import argparse
import csv
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
    ATTENTION_SCALE,
    D_HEAD,
    LOCAL_WINDOW,
    N_KV_HEADS,
    N_Q_HEADS,
    boundary_contract,
    error_metrics,
    load_fixture,
    resolve_active_heads,
)

_TENSOR_NAMES = ("context", "d_q", "d_k", "d_v")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cupy", "torch-flex-eager", "torch-flex-compile", "torch-vendor-hybrid"),
        required=True,
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--heads", default="15")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead"), default="default")
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--output-tensors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    args = parser.parse_args()
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


def _selection(active_heads):
    start = active_heads[0]
    if active_heads == tuple(range(start, start + len(active_heads))):
        return slice(start, start + len(active_heads))
    return active_heads


def _save_tensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{name: np.asarray(tensors[name], dtype=np.float32) for name in _TENSOR_NAMES},
    )


def _load_result_tensors(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        tensors = {name: np.array(archive[name], dtype=np.float32, copy=True) for name in _TENSOR_NAMES}
    return metadata, tensors


def _compare(tensors, reference_path, active_heads):
    if reference_path is None:
        return None
    metadata, reference = _load_result_tensors(reference_path)
    metrics = {name: error_metrics(tensors[name], reference[name]) for name in _TENSOR_NAMES}
    accepted = (
        metrics["context"]["relative_l2"] <= 0.02
        and metrics["context"]["cosine"] >= 0.999
        and all(metrics[name]["relative_l2"] <= 0.02 for name in ("d_q", "d_k", "d_v"))
        and all(metrics[name]["cosine"] >= 0.999 for name in ("d_q", "d_k", "d_v"))
    )
    inactive = sorted(set(range(1, N_Q_HEADS)) - set(active_heads))
    inactive_output_max = 0.0 if not inactive else float(np.max(np.abs(tensors["context"][:, :, inactive, :])))
    inactive_dq_max = 0.0 if not inactive else float(np.max(np.abs(tensors["d_q"][:, :, inactive, :])))
    accepted = accepted and inactive_output_max == 0.0 and inactive_dq_max == 0.0
    return {
        "reference": str(reference_path),
        "reference_metadata": metadata,
        "metrics": metrics,
        "inactive_current_heads": inactive,
        "inactive_output_max_abs": inactive_output_max,
        "inactive_dq_max_abs": inactive_dq_max,
        "tolerances_declared_before_timing": {
            "relative_l2_max": 0.02,
            "cosine_min": 0.999,
            "inactive_current_contribution": "exact zero",
        },
        "accepted": bool(accepted),
    }


def _cupy_arrays(arrays):
    import cupy as cp

    def bf16(name):
        return cp.asarray(arrays[name]).view(cp.dtype("bfloat16"))

    return bf16("q_bits"), bf16("k_bits"), bf16("v_bits"), cp.asarray(arrays["d_out"])


def _cupy_candidate(arrays, active_heads):
    import cupy as cp
    from mini_llm.ops.indexed_attention import (
        local_window_attention_backward,
        local_window_attention_forward,
    )

    q_all, k, v, d_out_all = _cupy_arrays(arrays)
    selection = _selection(active_heads)
    active_mapping = tuple(head // 4 for head in active_heads)
    inactive = sorted(set(range(1, N_Q_HEADS)) - set(active_heads))
    if inactive:
        d_out_all[:, :, inactive, :] = 0.0

    def full_forward():
        return local_window_attention_forward(
            q_all[:, :, 0:1, :],
            k,
            v,
            int(q_all.shape[1]),
            kv_head_indices=(0,),
            scale=ATTENTION_SCALE,
            return_cache=True,
        )

    def local_forward():
        return local_window_attention_forward(
            q_all[:, :, selection, :],
            k,
            v,
            LOCAL_WINDOW,
            kv_head_indices=active_mapping,
            scale=ATTENTION_SCALE,
            return_cache=True,
        )

    def forward():
        full_output, full_cache = full_forward()
        local_output, local_cache = local_forward()
        context = cp.zeros(q_all.shape, dtype=cp.float32)
        context[:, :, 0:1, :] = full_output
        context[:, :, selection, :] = local_output
        return context, (full_cache, local_cache)

    def backward(caches):
        full_gradients = local_window_attention_backward(
            d_out_all[:, :, 0:1, :], caches[0]
        )
        local_gradients = local_window_attention_backward(
            d_out_all[:, :, selection, :], caches[1]
        )
        d_q = cp.zeros(q_all.shape, dtype=cp.float32)
        d_q[:, :, 0:1, :] = full_gradients[0]
        d_q[:, :, selection, :] = local_gradients[0]
        return d_q, full_gradients[1] + local_gradients[1], full_gradients[2] + local_gradients[2]

    def combined():
        context, caches = forward()
        return context, backward(caches)

    initial_context, initial_caches = forward()
    cp.cuda.get_current_stream().synchronize()
    return {
        "forward": forward,
        "backward": lambda: backward(initial_caches),
        "combined": combined,
        "correctness": combined,
        "arrays": (q_all, k, v, d_out_all),
        "initial_context": initial_context,
        "cache": initial_caches,
    }


def _time_cupy(call, warmup, repeats):
    import cupy as cp

    for _ in range(warmup):
        call()
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
        wall_end = time.perf_counter_ns()
        device.append(float(cp.cuda.get_elapsed_time(start, end)))
        wall.append((wall_end - wall_start) / 1_000_000.0)
        del result
    return {"gpu_event": _summary(device), "host_wall": _summary(wall)}


def _run_cupy(args, manifest, arrays, active_heads):
    os.environ.setdefault("MINI_LLM_BACKEND", "cupy")
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_BACKEND", "grouped")
    os.environ.setdefault("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "1")
    os.environ.setdefault("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT", "0")
    os.environ.setdefault("MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD", "1")
    os.environ.setdefault("MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_CUBLAS_MODE", "grouped")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "auto")
    os.environ.setdefault("MINI_LLM_BF16_MIXED_CONTEXT", "0")

    import cupy as cp
    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    runtime_profile = apply_runtime_defaults(
        "gpu-fast",
        backend_name="cupy",
        config=ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k(),
        training=True,
        precision="bf16-mixed",
    )
    candidate = _cupy_candidate(arrays, active_heads)
    timings = {
        phase: _time_cupy(candidate[phase], args.warmup, args.repeats)
        for phase in ("forward", "backward", "combined")
    }
    pool = cp.get_default_memory_pool()
    baseline_used = int(pool.used_bytes())
    baseline_reserved = int(pool.total_bytes())
    result = candidate["combined"]()
    cp.cuda.get_current_stream().synchronize()
    memory = {
        "baseline_pool_used_bytes": baseline_used,
        "baseline_pool_reserved_bytes": baseline_reserved,
        "post_combined_pool_used_bytes": int(pool.used_bytes()),
        "post_combined_pool_reserved_bytes": int(pool.total_bytes()),
        "peak_semantics": "CuPy pool reservation is a process high-water proxy, not an allocator peak counter",
    }
    context, gradients = candidate["correctness"]()
    cp.cuda.get_current_stream().synchronize()
    tensors = {
        "context": cp.asnumpy(context),
        "d_q": cp.asnumpy(gradients[0]),
        "d_k": cp.asnumpy(gradients[1]),
        "d_v": cp.asnumpy(gradients[2]),
    }
    if args.cuda_profiler_capture:
        cp.cuda.runtime.profilerStart()
        cp.cuda.nvtx.RangePush("0087B_CUPY_COMBINED_FWD_BWD")
        try:
            profile_result = candidate["combined"]()
            cp.cuda.get_current_stream().synchronize()
        finally:
            cp.cuda.nvtx.RangePop()
            cp.cuda.runtime.profilerStop()
        del profile_result
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    capability = {
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
    }
    return timings, memory, tensors, capability, {
        "runtime_profile": {
            "requested": runtime_profile.requested,
            "resolved": runtime_profile.resolved,
            "applied": list(runtime_profile.applied),
        },
        "output_dtype": str(context.dtype),
        "gradient_dtypes": [str(value.dtype) for value in gradients],
        "integration_included": [
            "original-head selection/packing",
            "full and local public forward dispatch",
            "FP32 context allocation/scatter",
            "full and local public backward dispatch",
            "FP32 dQ scatter",
            "shared dK/dV summation",
        ],
    }


def _torch_tensors(torch, arrays, active_heads):
    def bf16(name):
        host = torch.from_numpy(np.array(arrays[name], copy=True))
        return host.view(torch.bfloat16).to("cuda").requires_grad_(True)

    q, k, v = bf16("q_bits"), bf16("k_bits"), bf16("v_bits")
    d_out = torch.from_numpy(np.array(arrays["d_out"], copy=True)).to("cuda")
    inactive = sorted(set(range(1, N_Q_HEADS)) - set(active_heads))
    if inactive:
        d_out[:, :, inactive, :] = 0.0
    return q, k, v, d_out


def _torch_timing(torch, forward, inputs, d_out, warmup, repeats):
    q, k, v = inputs

    def time_forward():
        device, wall = [], []
        for _ in range(warmup):
            value = forward(q, k, v)
            del value
        torch.cuda.synchronize()
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter_ns()
            start.record()
            value = forward(q, k, v)
            end.record()
            end.synchronize()
            wall_end = time.perf_counter_ns()
            device.append(start.elapsed_time(end))
            wall.append((wall_end - wall_start) / 1_000_000.0)
            del value
        return {"gpu_event": _summary(device), "host_wall": _summary(wall)}

    def one_backward():
        output = forward(q, k, v)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        raw_gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=d_out)
        gradients = tuple(gradient.float() for gradient in raw_gradients)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        return start.elapsed_time(end), (wall_end - wall_start) / 1_000_000.0, gradients

    def time_backward():
        for _ in range(warmup):
            _, _, gradients = one_backward()
            del gradients
        device, wall = [], []
        for _ in range(repeats):
            event_ms, wall_ms, gradients = one_backward()
            device.append(event_ms)
            wall.append(wall_ms)
            del gradients
        return {"gpu_event": _summary(device), "host_wall": _summary(wall)}

    def one_combined():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output = forward(q, k, v)
        raw_gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=d_out)
        gradients = tuple(gradient.float() for gradient in raw_gradients)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        return start.elapsed_time(end), (wall_end - wall_start) / 1_000_000.0, output, gradients

    def time_combined():
        for _ in range(warmup):
            _, _, output, gradients = one_combined()
            del output, gradients
        device, wall = [], []
        for _ in range(repeats):
            event_ms, wall_ms, output, gradients = one_combined()
            device.append(event_ms)
            wall.append(wall_ms)
            del output, gradients
        return {"gpu_event": _summary(device), "host_wall": _summary(wall)}

    return {
        "forward": time_forward(),
        "backward": time_backward(),
        "combined": time_combined(),
    }, one_combined


def _run_torch(args, manifest, arrays, active_heads):
    import torch
    from benchmarks.reference_0087b_torch_attention import (
        build_block_mask,
        flex_forward_bthd,
        vendor_hybrid_forward_bthd,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    q, k, v, d_out = _torch_tensors(torch, arrays, active_heads)
    mask_build_start = time.perf_counter()
    if args.backend == "torch-vendor-hybrid":
        block_mask = build_block_mask(torch, active_heads, manifest["seq"], include_full=False)
        eager_forward = lambda q, k, v: vendor_hybrid_forward_bthd(torch, q, k, v, block_mask)
    else:
        block_mask = build_block_mask(torch, active_heads, manifest["seq"], include_full=True)
        eager_forward = lambda q, k, v: flex_forward_bthd(torch, q, k, v, block_mask)
    torch.cuda.synchronize()
    mask_build_ms = (time.perf_counter() - mask_build_start) * 1000.0

    compiled = args.backend != "torch-flex-eager"
    compile_ms = None
    forward = eager_forward
    if compiled:
        forward = torch.compile(
            eager_forward,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
        compile_start = time.perf_counter()
        compile_output = forward(q, k, v)
        compile_gradients = torch.autograd.grad(
            compile_output, (q, k, v), grad_outputs=d_out
        )
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - compile_start) * 1000.0
        del compile_output, compile_gradients

    timings, one_combined = _torch_timing(
        torch, forward, (q, k, v), d_out, args.warmup, args.repeats
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    _, _, memory_output, memory_gradients = one_combined()
    memory = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del memory_output, memory_gradients

    _, _, output, gradients = one_combined()
    tensors = {
        "context": output.detach().cpu().numpy(),
        "d_q": gradients[0].detach().float().cpu().numpy(),
        "d_k": gradients[1].detach().float().cpu().numpy(),
        "d_v": gradients[2].detach().float().cpu().numpy(),
    }
    if args.cuda_profiler_capture:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("0087B_TORCH_COMBINED_FWD_BWD"):
            _, _, profile_output, profile_gradients = one_combined()
        torch.cuda.cudart().cudaProfilerStop()
        del profile_output, profile_gradients

    capability = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "flash_attention_available": bool(torch.backends.cuda.is_flash_attention_available()),
        "triton": __import__("triton").__version__,
    }
    details = {
        "compiled": compiled,
        "fullgraph_requested": compiled,
        "compile_mode": args.compile_mode if compiled else None,
        "compile_first_fwd_bwd_ms": compile_ms,
        "block_mask_build_ms": mask_build_ms,
        "output_dtype": str(output.dtype),
        "gradient_dtypes_before_export": [str(value.dtype) for value in gradients],
        "flex_leaf_gradient_dtypes_before_timed_upcast": ["torch.bfloat16"] * 3,
        "precision_controls": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "integration_included": [
            "BTHD-to-BHTD permutation views",
            "single logical 16:4 GQA dispatch",
            "per-original-head sparse block mask",
            "inactive historical-head zero-current contribution",
            "BHTD-to-BTHD output permutation",
            "BF16-output to FP32-context cast",
            "PyTorch autograd dispatch",
            "timed BF16-to-FP32 dQ/dK/dV conversion",
        ],
        "gradient_layout_note": "FlexAttention produces BF16 leaf gradients; the effective timing includes explicit FP32 upcasts for dQ/dK/dV. CuPy public attention returns FP32 gradients directly.",
        "vendor_exact_candidate": (
            "Flash SDPA for full head 0 plus FlexAttention local heads"
            if args.backend == "torch-vendor-hybrid"
            else "not used"
        ),
        "vendor_only_complete_candidate": {
            "status": "unavailable",
            "reason": "PyTorch SDPA exposes causal but no sparse per-query-head sliding-window bias; a dense T×T mask is prohibited",
        },
    }
    return timings, memory, tensors, capability, details


def _write_csv(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase, timing in report["timings"].items():
        rows.append({
            "backend": report["backend"],
            "head_pattern": report["head_pattern"],
            "batch": report["fixture"]["batch"],
            "seq": report["fixture"]["seq"],
            "phase": phase,
            "gpu_median_ms": timing["gpu_event"]["median_ms"],
            "gpu_p10_ms": timing["gpu_event"]["p10_ms"],
            "gpu_p90_ms": timing["gpu_event"]["p90_ms"],
            "wall_median_ms": timing["host_wall"]["median_ms"],
            "parity_accepted": None if report["parity"] is None else report["parity"]["accepted"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    manifest, arrays = load_fixture(args.fixture)
    active_heads = resolve_active_heads(args.heads)
    boundary = boundary_contract(active_heads)
    if not (
        boundary["local_key_7_allowed"] is False
        and boundary["local_key_8_allowed"] is True
        and boundary["local_future_key_520_allowed"] is False
        and boundary["full_key_7_allowed"] is True
        and boundary["full_future_key_520_allowed"] is False
    ):
        raise AssertionError(f"mask boundary contract failed: {boundary}")

    if args.backend == "cupy":
        timings, memory, tensors, capability, details = _run_cupy(
            args, manifest, arrays, active_heads
        )
    else:
        timings, memory, tensors, capability, details = _run_torch(
            args, manifest, arrays, active_heads
        )
    result_metadata = {
        "backend": args.backend,
        "fixture_sha256": manifest["sha256"],
        "head_pattern": args.heads,
        "active_local_heads": list(active_heads),
    }
    _save_tensors(args.output_tensors, tensors, result_metadata)
    parity = _compare(tensors, args.reference_tensors, active_heads)
    report = {
        "schema_version": "0087B-attention-result-v1",
        "scope": "complete mixed attention only; post-RoPE Q/K/V through context and dQ/dK/dV",
        "backend": args.backend,
        "status": "valid" if parity is None or parity["accepted"] else "invalid_numerical_parity",
        "fixture_path": str(args.fixture.resolve()),
        "fixture": manifest,
        "head_pattern": args.heads,
        "active_local_heads": list(active_heads),
        "historical_substitution_heads": sorted(set(range(1, N_Q_HEADS)) - set(active_heads)),
        "q_to_kv_original": [head // 4 for head in range(N_Q_HEADS)],
        "mask_boundary": boundary,
        "dtype_contract": {
            "q_k_v": "bfloat16 byte-exact fixture bits",
            "upstream_d_out": "float32",
            "cupy_context_and_gradients": "float32",
            "torch_context": "float32 cast of BF16 Flex/SDPA output",
            "torch_leaf_gradients": "bfloat16, exported as float32",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
        "memory": memory,
        "capability": capability,
        "details": details,
        "parity": parity,
        "result_tensors": str(args.output_tensors.resolve()),
        "timing_note": "GPU events and synchronized host wall are measured independently. Forward, backward and combined medians are separate runs and must not be added.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv:
        _write_csv(args.output_csv, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if parity is not None and not parity["accepted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
