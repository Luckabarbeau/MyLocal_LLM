#!/usr/bin/env python3
"""0089D production CuPy versus compiled-Torch LM-head/CE benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RELATIVE_L2_MAX = 0.02
_COSINE_MIN = 0.999
_LOSS_ABS_MAX = 5e-3
_LOSS_REL_MAX = 2e-3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cupy", "torch-eager", "torch-compile"),
        required=True,
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--output-tensors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    return args


def _reference_module():
    return importlib.import_module("benchmarks.reference_0089d_torch_lm_head")


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    median = float(statistics.median(values))
    return {
        "count": len(values),
        "samples_ms": values,
        "median_ms": median,
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "std_ms": float(np.std(values)),
        "mad_ms": float(statistics.median(abs(value - median) for value in values)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_tensors(
    path: Path,
    names: Sequence[str],
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata), sort_keys=True))
    }
    payload.update(
        {name: np.asarray(tensors[name], dtype=np.float32) for name in names}
    )
    np.savez(path, **payload)


def _load_result_tensors(
    path: Path, names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = (
            json.loads(str(archive["metadata_json"].item()))
            if "metadata_json" in archive
            else {}
        )
        tensors = {
            name: np.array(archive[name], dtype=np.float32, copy=True)
            for name in names
        }
    return metadata, tensors


def _compare(
    tensors: Mapping[str, np.ndarray],
    reference_path: Path | None,
    names: Sequence[str],
    error_metrics: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    metadata, reference = _load_result_tensors(reference_path, names)
    loss_actual = float(np.asarray(tensors["loss"], dtype=np.float32))
    loss_reference = float(np.asarray(reference["loss"], dtype=np.float32))
    loss_abs = abs(loss_actual - loss_reference)
    loss_rel = loss_abs / max(abs(loss_reference), 1e-30)
    tensor_metrics = {
        name: dict(error_metrics(tensors[name], reference[name]))
        for name in names
        if name != "loss"
    }
    tensor_accepted = all(
        metric["relative_l2"] <= _RELATIVE_L2_MAX
        and metric["cosine"] >= _COSINE_MIN
        for metric in tensor_metrics.values()
    )
    loss_accepted = loss_abs <= _LOSS_ABS_MAX and loss_rel <= _LOSS_REL_MAX
    worst_name = max(
        tensor_metrics,
        key=lambda name: tensor_metrics[name]["relative_l2"],
    )
    return {
        "reference": str(reference_path.resolve()),
        "reference_metadata": metadata,
        "loss": {
            "actual": loss_actual,
            "reference": loss_reference,
            "absolute_error": loss_abs,
            "relative_error": loss_rel,
            "accepted": bool(loss_accepted),
        },
        "metrics": tensor_metrics,
        "worst_tensor": worst_name,
        "tolerances_declared_before_timing": {
            "relative_l2_max": _RELATIVE_L2_MAX,
            "cosine_min": _COSINE_MIN,
            "loss_absolute_error_max": _LOSS_ABS_MAX,
            "loss_relative_error_max": _LOSS_REL_MAX,
        },
        "accepted": bool(loss_accepted and tensor_accepted),
    }


def _cupy_timing_forward(harness, head, targets, mask, manifest, warmup, repeats):
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    def call():
        return harness.chunked_lm_head_loss_forward(
            head,
            targets,
            chunk_tokens=int(manifest["chunk_tokens"]),
            loss_mask=mask,
            return_device_loss=True,
            target_slice=tuple(manifest["target_slice"]),
        )

    for _ in range(warmup):
        loss, cache = call()
        del loss, cache
    stream.synchronize()
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        loss, cache = call()
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        wall_samples.append((wall_end - wall_start) / 1_000_000.0)
        del loss, cache
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _cupy_timing_backward(harness, head, targets, mask, manifest, warmup, repeats):
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    def one(measure: bool):
        _, cache = harness.chunked_lm_head_loss_forward(
            head,
            targets,
            chunk_tokens=int(manifest["chunk_tokens"]),
            loss_mask=mask,
            return_device_loss=True,
            target_slice=tuple(manifest["target_slice"]),
        )
        stream.synchronize()
        harness.zero_grad()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        dx = harness.chunked_lm_head_backward(
            cache, grad_scale=float(manifest["grad_scale"])
        )
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(cp.cuda.get_elapsed_time(start, end)),
            (wall_end - wall_start) / 1_000_000.0,
            dx,
        )

    for _ in range(warmup):
        one(False)
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        gpu_ms, wall_ms, dx = measured
        gpu_samples.append(gpu_ms)
        wall_samples.append(wall_ms)
        del dx
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _cupy_timing_combined(harness, head, targets, mask, manifest, warmup, repeats):
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    def one(measure: bool):
        harness.zero_grad()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        loss, cache = harness.chunked_lm_head_loss_forward(
            head,
            targets,
            chunk_tokens=int(manifest["chunk_tokens"]),
            loss_mask=mask,
            return_device_loss=True,
            target_slice=tuple(manifest["target_slice"]),
        )
        dx = harness.chunked_lm_head_backward(
            cache, grad_scale=float(manifest["grad_scale"])
        )
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(cp.cuda.get_elapsed_time(start, end)),
            (wall_end - wall_start) / 1_000_000.0,
            loss,
            dx,
        )

    for _ in range(warmup):
        one(False)
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        gpu_ms, wall_ms, loss, dx = measured
        gpu_samples.append(gpu_ms)
        wall_samples.append(wall_ms)
        del loss, dx
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _run_cupy(args, manifest, arrays, result_names):
    os.environ["MINI_LLM_BACKEND"] = "cupy"
    os.environ["MINI_LLM_FUSED_BF16_CE"] = "1"
    os.environ["MINI_LLM_FUSED_BF16_CE_STRICT"] = "1"

    import cupy as cp
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.linear import TiedOutputProjection
    import mini_llm.ops.loss as loss_ops
    from mini_llm.parameter import Parameter

    class ProductionHeadHarness:
        chunked_lm_head_loss_forward = (
            DecoderLanguageModel.chunked_lm_head_loss_forward
        )
        chunked_lm_head_backward = DecoderLanguageModel.chunked_lm_head_backward

        def __init__(self, embedding_weight):
            self.embedding_weight = embedding_weight
            self.output_proj = TiedOutputProjection(embedding_weight)
            self.hierarchical_memory_enabled = False

        def zero_grad(self):
            self.embedding_weight.zero_grad()

    setup_start = time.perf_counter_ns()
    bf16 = cp.dtype("bfloat16")
    head = cp.asarray(arrays["head_bits"]).view(bf16)
    weight_dv = cp.asarray(arrays["weight_bits"]).view(bf16)
    embedding_data = cp.ascontiguousarray(weight_dv.transpose(1, 0))
    embedding_weight = Parameter(embedding_data, name="embedding.W", decay=False)
    harness = ProductionHeadHarness(embedding_weight)
    targets = cp.asarray(arrays["targets"], dtype=cp.int32)
    mask = cp.asarray(arrays["loss_mask"], dtype=cp.float32)
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    transfer_setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0

    fused_start = time.perf_counter_ns()
    if not loss_ops._fused_bf16_ce_enabled(head.dtype):
        raise RuntimeError("0089D requires fused BF16 CE selection")
    fused_module = loss_ops._get_fused_bf16_ce_module()
    if fused_module is None:
        raise RuntimeError("0089D strict CuPy run could not load fused BF16 CE")
    stream.synchronize()
    fused_ce_setup_ms = (time.perf_counter_ns() - fused_start) / 1_000_000.0

    timings = {
        "forward": _cupy_timing_forward(
            harness, head, targets, mask, manifest, args.warmup, args.repeats
        ),
        "backward": _cupy_timing_backward(
            harness, head, targets, mask, manifest, args.warmup, args.repeats
        ),
        "combined": _cupy_timing_combined(
            harness, head, targets, mask, manifest, args.warmup, args.repeats
        ),
    }

    harness.zero_grad()
    loss, cache = harness.chunked_lm_head_loss_forward(
        head,
        targets,
        chunk_tokens=int(manifest["chunk_tokens"]),
        loss_mask=mask,
        return_device_loss=True,
        target_slice=tuple(manifest["target_slice"]),
    )
    dx = harness.chunked_lm_head_backward(
        cache, grad_scale=float(manifest["grad_scale"])
    )
    stream.synchronize()
    tensors = {
        result_names[0]: np.asarray(cp.asnumpy(loss), dtype=np.float32),
        result_names[1]: cp.asnumpy(dx.astype(cp.float32)),
        result_names[2]: cp.asnumpy(harness.output_proj.W.grad),
    }

    if args.cuda_profiler_capture:
        harness.zero_grad()
        cp.cuda.runtime.profilerStart()
        cp.cuda.nvtx.RangePush("0089D_CUPY_LM_HEAD_FWD_BWD")
        try:
            profile_loss, profile_cache = harness.chunked_lm_head_loss_forward(
                head,
                targets,
                chunk_tokens=int(manifest["chunk_tokens"]),
                loss_mask=mask,
                return_device_loss=True,
                target_slice=tuple(manifest["target_slice"]),
            )
            profile_dx = harness.chunked_lm_head_backward(
                profile_cache, grad_scale=float(manifest["grad_scale"])
            )
            stream.synchronize()
        finally:
            cp.cuda.nvtx.RangePop()
            cp.cuda.runtime.profilerStop()
        del profile_loss, profile_cache, profile_dx

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    capability = {
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
    }
    setup = {
        "fixture_transfer_and_harness_ms": transfer_setup_ms,
        "fused_ce_module_setup_ms": fused_ce_setup_ms,
        "fused_ce_selected": True,
        "fused_ce_strict": True,
    }
    details = {
        "production_methods": [
            "DecoderLanguageModel.chunked_lm_head_loss_forward",
            "DecoderLanguageModel.chunked_lm_head_backward",
        ],
        "tied_parameter_layout": {
            "embedding_data": [int(manifest["vocab"]), int(manifest["d_model"])],
            "output_weight_view": [int(manifest["d_model"]), int(manifest["vocab"])],
            "output_grad_view": [int(manifest["d_model"]), int(manifest["vocab"])],
        },
        "backward_setup": (
            "production forward/cache construction, synchronization, and FP32 "
            "gradient clearing are outside the timed event"
        ),
        "combined_setup": (
            "FP32 gradient clearing is outside the timed event; target-slice copy, "
            "loss-only forward, logits recomputation, dW accumulation, and dx "
            "scatter are inside"
        ),
    }
    return timings, tensors, capability, setup, details


def _torch_from_bf16_bits(torch, value):
    host = np.array(value, dtype=np.uint16, copy=True)
    return torch.from_numpy(host).view(torch.bfloat16).to(device="cuda")


def _torch_timing_forward(torch, forward, head, weight, targets, mask, warmup, repeats):
    for _ in range(warmup):
        loss, selected = forward(head, weight, targets, mask)
        del loss, selected
    torch.cuda.synchronize()
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        loss, selected = forward(head, weight, targets, mask)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_samples.append(float(start.elapsed_time(end)))
        wall_samples.append((wall_end - wall_start) / 1_000_000.0)
        del loss, selected
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _torch_timing_backward(
    torch, forward, backward, head, weight, targets, mask, dweight, warmup, repeats
):
    def one(measure: bool):
        _, selected = forward(head, weight, targets, mask)
        torch.cuda.synchronize()
        dweight.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        dx, accumulated = backward(selected, weight, targets, mask, dweight)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return float(start.elapsed_time(end)), (wall_end - wall_start) / 1_000_000.0, dx, accumulated

    for _ in range(warmup):
        one(False)
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        gpu_ms, wall_ms, dx, accumulated = measured
        gpu_samples.append(gpu_ms)
        wall_samples.append(wall_ms)
        del dx, accumulated
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _torch_timing_combined(
    torch, forward, backward, head, weight, targets, mask, dweight, warmup, repeats
):
    def one(measure: bool):
        dweight.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        loss, selected = forward(head, weight, targets, mask)
        dx, accumulated = backward(selected, weight, targets, mask, dweight)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return float(start.elapsed_time(end)), (wall_end - wall_start) / 1_000_000.0, loss, dx, accumulated

    for _ in range(warmup):
        one(False)
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        gpu_ms, wall_ms, loss, dx, accumulated = measured
        gpu_samples.append(gpu_ms)
        wall_samples.append(wall_ms)
        del loss, dx, accumulated
    return {"gpu_event": _summary(gpu_samples), "host_wall": _summary(wall_samples)}


def _run_torch(args, manifest, arrays, result_names, reference):
    torch = importlib.import_module("torch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    transfer_start = time.perf_counter_ns()
    fixture = reference.torch_fixture_tensors(torch, arrays)
    head = _torch_from_bf16_bits(torch, fixture["head_bits"])
    weight = _torch_from_bf16_bits(torch, fixture["weight_bits"])
    targets = torch.from_numpy(
        np.array(fixture["targets"], dtype=np.int32, copy=True)
    ).to(device="cuda", dtype=torch.int64)
    mask = torch.from_numpy(
        np.array(fixture["loss_mask"], dtype=np.float32, copy=True)
    ).to(device="cuda")
    dweight = torch.zeros(
        (int(manifest["d_model"]), int(manifest["vocab"])),
        device="cuda",
        dtype=torch.float32,
    )
    torch.cuda.synchronize()
    transfer_setup_ms = (time.perf_counter_ns() - transfer_start) / 1_000_000.0

    eager_forward, eager_backward = reference.build_torch_functions(
        torch,
        batch=int(manifest["batch"]),
        seq=int(manifest["seq"]),
        d_model=int(manifest["d_model"]),
        vocab=int(manifest["vocab"]),
        chunk_tokens=int(manifest["chunk_tokens"]),
        target_slice=tuple(manifest["target_slice"]),
        normalizer_count=float(manifest["normalizer_count"]),
        grad_scale=float(manifest["grad_scale"]),
    )
    compiled = args.backend == "torch-compile"
    forward = eager_forward
    backward = eager_backward
    compile_forward_ms = None
    compile_backward_ms = None
    if compiled:
        forward = torch.compile(
            eager_forward,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
        backward = torch.compile(
            eager_backward,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
        compile_start = time.perf_counter_ns()
        compile_loss, compile_selected = forward(head, weight, targets, mask)
        torch.cuda.synchronize()
        compile_forward_ms = (time.perf_counter_ns() - compile_start) / 1_000_000.0
        dweight.zero_()
        compile_start = time.perf_counter_ns()
        compile_dx, compile_dw = backward(
            compile_selected, weight, targets, mask, dweight
        )
        torch.cuda.synchronize()
        compile_backward_ms = (time.perf_counter_ns() - compile_start) / 1_000_000.0
        del compile_loss, compile_selected, compile_dx, compile_dw

    timings = {
        "forward": _torch_timing_forward(
            torch, forward, head, weight, targets, mask, args.warmup, args.repeats
        ),
        "backward": _torch_timing_backward(
            torch,
            forward,
            backward,
            head,
            weight,
            targets,
            mask,
            dweight,
            args.warmup,
            args.repeats,
        ),
        "combined": _torch_timing_combined(
            torch,
            forward,
            backward,
            head,
            weight,
            targets,
            mask,
            dweight,
            args.warmup,
            args.repeats,
        ),
    }

    dweight.zero_()
    loss, selected = forward(head, weight, targets, mask)
    dx, accumulated = backward(selected, weight, targets, mask, dweight)
    torch.cuda.synchronize()
    tensors = {
        result_names[0]: loss.detach().float().cpu().numpy(),
        result_names[1]: dx.detach().float().cpu().numpy(),
        result_names[2]: accumulated.detach().float().cpu().numpy(),
    }

    if args.cuda_profiler_capture:
        dweight.zero_()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("0089D_TORCH_LM_HEAD_FWD_BWD"):
                profile_loss, profile_selected = forward(
                    head, weight, targets, mask
                )
                profile_dx, profile_dw = backward(
                    profile_selected, weight, targets, mask, dweight
                )
                torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        del profile_loss, profile_selected, profile_dx, profile_dw

    try:
        triton_version = importlib.import_module("triton").__version__
    except (ImportError, AttributeError):
        triton_version = None
    capability = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton_version,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    setup = {
        "fixture_transfer_and_allocation_ms": transfer_setup_ms,
        "compile_first_forward_ms": compile_forward_ms,
        "compile_first_backward_ms": compile_backward_ms,
        "compiled": compiled,
        "fullgraph_requested": compiled,
        "dynamic_requested": False if compiled else None,
        "compile_mode": args.compile_mode if compiled else None,
    }
    details = {
        "manual_backward": True,
        "ordinary_cross_entropy_autograd": False,
        "separate_compiled_forward_backward": True,
        "recompute_barrier": (
            "forward and backward are separate callables, so combined timing invokes "
            "two compiled graphs and cannot eliminate logits recomputation"
        ),
        "precision_controls": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "backward_setup": (
            "compiled/eager forward cache construction, synchronization, and FP32 "
            "dweight clearing are outside the timed event"
        ),
        "combined_setup": (
            "FP32 dweight clearing is outside the timed event; target-slice copy, "
            "loss-only forward, explicit logits recomputation, BF16 dW to FP32 "
            "accumulation, and BF16 dx scatter are inside"
        ),
    }
    return timings, tensors, capability, setup, details


def main() -> int:
    args = _parse_args()
    reference = _reference_module()
    required = (
        "RESULT_TENSOR_NAMES",
        "load_fixture",
        "error_metrics",
        "graph_contract",
        "build_torch_functions",
        "torch_fixture_tensors",
    )
    missing = [name for name in required if not hasattr(reference, name)]
    if missing:
        raise ImportError(
            f"reference_0089d_torch_lm_head is missing helpers: {missing}"
        )
    manifest, arrays = reference.load_fixture(args.fixture)
    result_names = tuple(reference.RESULT_TENSOR_NAMES)
    if result_names != ("loss", "dx", "dweight"):
        raise ValueError("0089D result tensor contract changed")

    if args.backend == "cupy":
        timings, tensors, capability, setup, details = _run_cupy(
            args, manifest, arrays, result_names
        )
    else:
        timings, tensors, capability, setup, details = _run_torch(
            args, manifest, arrays, result_names, reference
        )

    metadata = {
        "patch": "0089D",
        "backend": args.backend,
        "fixture_sha256": manifest.get("sha256"),
        "result_tensor_names": list(result_names),
    }
    _save_tensors(args.output_tensors, result_names, tensors, metadata)
    parity = _compare(
        tensors,
        args.reference_tensors,
        result_names,
        reference.error_metrics,
    )
    report = {
        "schema_version": "0089D-production-lm-head-result-v1",
        "patch": "0089D",
        "scope": (
            "exact B3/T2304/D1024/V32768 production chunked tied LM head and "
            "masked BF16 cross-entropy forward plus recomputed numerical backward"
        ),
        "backend": args.backend,
        "status": (
            "valid"
            if parity is None or parity["accepted"]
            else "invalid_numerical_parity"
        ),
        "fixture_path": str(args.fixture.resolve()),
        "fixture": _jsonable(manifest),
        "graph_contract": reference.graph_contract(),
        "dtype_contract": {
            "fixture_head_and_weight": "uint16 byte-exact bfloat16 bits",
            "logits": "bfloat16 transient",
            "loss_and_softmax_compute": "float32",
            "dlogits_and_dx": "bfloat16",
            "dweight_tile": "bfloat16",
            "dweight_accumulator": "float32",
            "saved_loss_dx_dweight": "float32 host arrays",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
        "setup": setup,
        "capability": capability,
        "details": details,
        "parity": parity,
        "result_tensors": str(args.output_tensors.resolve()),
        "timing_note": (
            "Forward, backward, and combined are independent warmed distributions; "
            "their medians must not be added. Raw GPU-event and synchronized host-wall "
            "samples are retained. Fixture transfer, compilation, and grad clearing "
            "are reported or performed separately."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 2 if parity is not None and not parity["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
