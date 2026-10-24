#!/usr/bin/env python3
"""PATCH 0089E consecutive Transformer-block CuPy/Torch benchmark."""

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
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VALID_DEPTHS = (1, 2, 4, 8, 20)
LOGICAL_PARAMETERS_PER_LAYER = 9
TORCH_LEAVES_PER_LAYER = 7
RELATIVE_L2_MAX = 0.02
COSINE_MIN = 0.999
RESULT_SCHEMA = "0089E-block-stack-result-v1"


def validate_depth(depth: int) -> int:
    depth = int(depth)
    if depth not in VALID_DEPTHS:
        raise ValueError(f"depth must be one of {VALID_DEPTHS}, got {depth}")
    return depth


def logical_gradient_count(depth: int) -> int:
    return validate_depth(depth) * LOGICAL_PARAMETERS_PER_LAYER


def torch_leaf_count(depth: int) -> int:
    return 1 + validate_depth(depth) * TORCH_LEAVES_PER_LAYER


def result_tensor_names(depth: int) -> tuple[str, ...]:
    depth = validate_depth(depth)
    reference = importlib.import_module("benchmarks.reference_0089b_torch_block")
    if depth == 1:
        return tuple(reference.RESULT_TENSOR_NAMES)
    if depth != 2:
        return ()
    gradient_suffixes = tuple(reference.RESULT_TENSOR_NAMES[2:])
    names = ["output", "d_input", "boundary_output", "boundary_grad"]
    for layer_index in range(2):
        names.extend(f"layer{layer_index}_{name}" for name in gradient_suffixes)
    return tuple(names)


def _validate_io_contract(
    depth: int,
    reference_tensors: Path | None,
    output_tensors: Path | None,
) -> None:
    depth = validate_depth(depth)
    if depth <= 2 and output_tensors is None:
        raise ValueError("depth 1 and 2 require --output-tensors")
    if depth > 2 and reference_tensors is not None:
        raise ValueError("reference tensors are unsupported for timing-only depths > 2")
    if depth > 2 and output_tensors is not None:
        raise ValueError("full tensor export is unsupported for timing-only depths > 2")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cupy", "torch-compile", "torch-eager"),
        required=True,
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=VALID_DEPTHS, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--output-tensors", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--memory-limit-fraction", type=float, default=0.90)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    if not 0.0 < args.memory_limit_fraction <= 1.0:
        parser.error("memory-limit-fraction must be in (0, 1]")
    try:
        _validate_io_contract(
            args.depth, args.reference_tensors, args.output_tensors
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


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


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("timing summary requires at least one sample")
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


def _complete_error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"parity shape mismatch: actual {actual.shape}, expected {expected.shape}"
        )
    actual_f = np.asarray(actual, dtype=np.float32)
    expected_f = np.asarray(expected, dtype=np.float32)
    difference = actual_f - expected_f
    flat_difference = difference.reshape(-1)
    worst_flat = int(np.argmax(np.abs(flat_difference))) if flat_difference.size else 0
    actual_norm = float(np.linalg.norm(actual_f.reshape(-1)))
    expected_norm = float(np.linalg.norm(expected_f.reshape(-1)))
    difference_norm = float(np.linalg.norm(flat_difference))
    denominator = actual_norm * expected_norm
    cosine = (
        float(np.dot(actual_f.reshape(-1), expected_f.reshape(-1)) / denominator)
        if denominator
        else (1.0 if difference_norm == 0.0 else 0.0)
    )
    return {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "relative_l2": (
            difference_norm / expected_norm
            if expected_norm
            else (0.0 if difference_norm == 0.0 else float("inf"))
        ),
        "cosine": cosine,
        "bias": float(np.mean(difference)) if difference.size else 0.0,
        "worst_pair": {
            "flat_index": worst_flat,
            "index": (
                list(np.unravel_index(worst_flat, actual_f.shape))
                if actual_f.size
                else []
            ),
            "actual": float(actual_f.reshape(-1)[worst_flat]) if actual_f.size else 0.0,
            "reference": (
                float(expected_f.reshape(-1)[worst_flat]) if expected_f.size else 0.0
            ),
        },
    }


def _parity_report(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> dict[str, Any]:
    names = tuple(str(name) for name in names)
    missing_actual = [name for name in names if name not in actual]
    missing_expected = [name for name in names if name not in expected]
    extra_actual = sorted(set(actual) - set(names))
    extra_expected = sorted(set(expected) - set(names))
    if missing_actual or missing_expected or extra_actual or extra_expected:
        raise ValueError(
            "parity tensor contract mismatch: "
            f"missing_actual={missing_actual}, missing_expected={missing_expected}, "
            f"extra_actual={extra_actual}, extra_expected={extra_expected}"
        )
    metrics = {
        name: _complete_error_metrics(actual[name], expected[name]) for name in names
    }
    accepted = all(
        np.isfinite(metric["relative_l2"])
        and np.isfinite(metric["cosine"])
        and metric["relative_l2"] <= RELATIVE_L2_MAX
        and metric["cosine"] >= COSINE_MIN
        for metric in metrics.values()
    )
    worst_name = max(names, key=lambda name: metrics[name]["relative_l2"])
    return {
        "metrics": metrics,
        "worst_pair": {"tensor": worst_name, **metrics[worst_name]["worst_pair"]},
        "tolerances_declared_before_timing": {
            "relative_l2_max": RELATIVE_L2_MAX,
            "cosine_min": COSINE_MIN,
        },
        "accepted": bool(accepted),
    }


def _save_tensors(
    path: Path,
    names: Sequence[str],
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    if set(tensors) != set(names):
        raise ValueError("saved tensor keys do not exactly match the declared contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata), sort_keys=True))
    }
    payload.update(
        {name: np.asarray(tensors[name], dtype=np.float32) for name in names}
    )
    np.savez(path, **payload)


def _load_tensors(
    path: Path, names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files) - {"metadata_json"}
        if available != set(names):
            raise ValueError(
                f"reference tensor names {sorted(available)} != {sorted(names)}"
            )
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


def _compare_to_file(
    tensors: Mapping[str, np.ndarray],
    reference_path: Path | None,
    names: Sequence[str],
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    metadata, expected = _load_tensors(reference_path, names)
    report = _parity_report(tensors, expected, names)
    report["reference"] = str(reference_path.resolve())
    report["reference_metadata"] = metadata
    return report


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _assert_cross_layer_ranges_disjoint(
    layer_ranges: Sequence[Sequence[tuple[int, int, str]]], kind: str
) -> None:
    for left_index, left_ranges in enumerate(layer_ranges):
        for right_index in range(left_index + 1, len(layer_ranges)):
            for left_start, left_end, left_name in left_ranges:
                for right_start, right_end, right_name in layer_ranges[right_index]:
                    if _ranges_overlap(
                        (left_start, left_end), (right_start, right_end)
                    ):
                        raise RuntimeError(
                            f"{kind} storage aliases across layers {left_index} and "
                            f"{right_index}: {left_name} overlaps {right_name}"
                        )


def _parameter_records(reference, arrays) -> tuple[tuple[str, str], ...]:
    records = tuple(
        (name, str(reference.PARAMETER_KEYS[name]))
        for name in reference.PARAMETER_NAMES
    )
    missing = [key for _, key in records if key not in arrays]
    if missing:
        raise KeyError(f"fixture is missing parameter arrays: {missing}")
    if len(records) != LOGICAL_PARAMETERS_PER_LAYER:
        raise ValueError("0089E requires the exact nine-parameter 0089B block")
    return records


def _parameter_suffix(name: str) -> str:
    return name.removeprefix("block.")


def _cupy_selected_parameters(block, records):
    by_name = {parameter.name: parameter for parameter in block.parameters()}
    selected = []
    for fixture_name, fixture_key in records:
        parameter = by_name.get(fixture_name)
        if parameter is None:
            suffix = _parameter_suffix(fixture_name)
            matches = [
                candidate
                for name, candidate in by_name.items()
                if name.endswith(suffix)
            ]
            if len(matches) == 1:
                parameter = matches[0]
        if parameter is None:
            raise KeyError(f"cannot map fixture parameter {fixture_name!r}")
        selected.append((fixture_name, fixture_key, parameter))
    if len({id(parameter) for _, _, parameter in selected}) != len(records):
        raise RuntimeError("one CuPy layer does not own nine unique Parameter objects")
    return selected


def _cupy_range(array, label: str) -> tuple[int, int, str]:
    start = int(array.data.ptr)
    return start, start + int(array.nbytes), label


def _torch_range(tensor, label: str) -> tuple[int, int, str]:
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel() * tensor.element_size()), label


def _check_cupy_storage(selected_layers) -> dict[str, Any]:
    parameter_ids = [
        id(parameter)
        for selected in selected_layers
        for _, _, parameter in selected
    ]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("CuPy Parameter objects are shared across layers")
    data_ranges = []
    grad_ranges = []
    for layer_index, selected in enumerate(selected_layers):
        data_ranges.append(
            [
                _cupy_range(parameter.data, f"layer{layer_index}.{name}.data")
                for name, _, parameter in selected
            ]
        )
        grad_ranges.append(
            [
                _cupy_range(parameter.grad, f"layer{layer_index}.{name}.grad")
                for name, _, parameter in selected
            ]
        )
    _assert_cross_layer_ranges_disjoint(data_ranges, "CuPy parameter")
    _assert_cross_layer_ranges_disjoint(grad_ranges, "CuPy gradient")
    return {
        "checked": True,
        "layers": len(selected_layers),
        "logical_parameters_per_layer": LOGICAL_PARAMETERS_PER_LAYER,
        "cross_layer_parameter_ranges_disjoint": True,
        "cross_layer_gradient_ranges_disjoint": True,
        "packed_qkv_gradient_views_within_layer_allowed": True,
    }


def _check_torch_storage(layer_leaves) -> dict[str, Any]:
    ranges = []
    tensor_ids = []
    for layer_index, leaves in enumerate(layer_leaves):
        tensor_ids.extend(id(value) for value in leaves)
        ranges.append(
            [
                _torch_range(value, f"layer{layer_index}.leaf{leaf_index}")
                for leaf_index, value in enumerate(leaves)
            ]
        )
    if len(tensor_ids) != len(set(tensor_ids)):
        raise RuntimeError("Torch leaf objects are shared across layers")
    _assert_cross_layer_ranges_disjoint(ranges, "Torch parameter")
    return {
        "checked": True,
        "layers": len(layer_leaves),
        "torch_leaves_per_layer": TORCH_LEAVES_PER_LAYER,
        "logical_parameters_per_layer": LOGICAL_PARAMETERS_PER_LAYER,
        "cross_layer_leaf_ranges_disjoint": True,
    }


def _zero_cupy_grads(blocks) -> None:
    for block in blocks:
        block.zero_grad()


def _cupy_forward_stack(blocks, x, position_ids):
    value = x
    caches = []
    boundary = None
    for layer_index, block in enumerate(blocks):
        value, cache = block.forward(
            value, return_cache=True, position_ids=position_ids
        )
        caches.append(cache)
        if len(blocks) == 2 and layer_index == 0:
            boundary = value
    return value, caches, boundary


def _cupy_backward_stack(blocks, d_out, caches):
    gradient = d_out
    boundary_gradient = None
    for layer_index in range(len(blocks) - 1, -1, -1):
        gradient = blocks[layer_index].backward(gradient, caches[layer_index])
        if len(blocks) == 2 and layer_index == 1:
            boundary_gradient = gradient
    return gradient, boundary_gradient


def _validate_cupy_paths_and_dtypes(
    cp, blocks, selected_layers, output, d_input, caches
) -> dict[str, Any]:
    bf16_name = str(cp.dtype("bfloat16"))
    fp32_name = str(cp.dtype("float32"))
    failures = []
    for layer_index, (block, selected, cache) in enumerate(
        zip(blocks, selected_layers, caches)
    ):
        if str(cache["residual1"].dtype) != fp32_name:
            failures.append(f"layer {layer_index} residual1 is not FP32")
        if str(cache["residual2"].dtype) != fp32_name:
            failures.append(f"layer {layer_index} residual2 is not FP32")
        norm2_cache = cache["norm2_cache"]
        if norm2_cache.get("fused_bf16_residual") is not True:
            failures.append(f"layer {layer_index} did not use 0089C fusion")
        attention_cache = cache["attn_cache"]
        pattern_caches = attention_cache.get("pattern_caches", ())
        if len(pattern_caches) != 1 or pattern_caches[0].get("kind") != "triton_mixed_0088a":
            failures.append(f"layer {layer_index} did not use strict Triton mixed attention")
        if str(attention_cache["x"].dtype) != bf16_name:
            failures.append(f"layer {layer_index} attention branch input is not BF16")
        for name, _, parameter in selected:
            if str(parameter.data.dtype) != bf16_name:
                failures.append(f"layer {layer_index} {name} data is not BF16")
            if str(parameter.grad.dtype) != fp32_name:
                failures.append(f"layer {layer_index} {name} gradient is not FP32")
    if str(output.dtype) != fp32_name:
        failures.append("stack output is not FP32")
    if str(d_input.dtype) != fp32_name:
        failures.append("stack input gradient is not FP32")
    finite = bool(cp.all(cp.isfinite(output)).item()) and bool(
        cp.all(cp.isfinite(d_input)).item()
    )
    for selected in selected_layers:
        finite = finite and all(
            bool(cp.all(cp.isfinite(parameter.grad)).item())
            for _, _, parameter in selected
        )
    if not finite:
        failures.append("non-finite output or gradient detected")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "finite": True,
        "residual_dtype": "float32",
        "parameter_dtype": "bfloat16",
        "parameter_gradient_dtype": "float32",
        "strict_triton_mixed_attention_layers": len(blocks),
        "fused_0089c_residual_rmsnorm_layers": len(blocks),
    }


def _time_cupy_combined(cp, blocks, x, d_out, position_ids, warmup, repeats):
    stream = cp.cuda.get_current_stream()

    def one(measure: bool):
        _zero_cupy_grads(blocks)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        output, caches, _ = _cupy_forward_stack(blocks, x, position_ids)
        d_input, _ = _cupy_backward_stack(blocks, d_out, caches)
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(cp.cuda.get_elapsed_time(start, end)),
            (wall_end - wall_start) / 1_000_000.0,
            output,
            d_input,
            caches,
        )

    for _ in range(warmup):
        one(False)
    device_samples = []
    wall_samples = []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        device_ms, wall_ms, output, d_input, caches = measured
        device_samples.append(device_ms)
        wall_samples.append(wall_ms)
        del output, d_input, caches
    return {
        "gpu_event": _summary(device_samples),
        "host_wall": _summary(wall_samples),
    }


def _cupy_export_tensors(
    cp, depth, output, d_input, boundary, boundary_gradient, selected_layers
):
    names = result_tensor_names(depth)
    if depth == 1:
        tensors = {
            names[0]: cp.asnumpy(output),
            names[1]: cp.asnumpy(d_input),
        }
        for result_name, (_, _, parameter) in zip(names[2:], selected_layers[0]):
            tensors[result_name] = cp.asnumpy(parameter.grad)
        return tensors
    tensors = {
        "output": cp.asnumpy(output),
        "d_input": cp.asnumpy(d_input),
        "boundary_output": cp.asnumpy(boundary),
        "boundary_grad": cp.asnumpy(boundary_gradient),
    }
    suffixes = tuple(
        importlib.import_module(
            "benchmarks.reference_0089b_torch_block"
        ).RESULT_TENSOR_NAMES[2:]
    )
    for layer_index, selected in enumerate(selected_layers):
        for suffix, (_, _, parameter) in zip(suffixes, selected):
            tensors[f"layer{layer_index}_{suffix}"] = cp.asnumpy(parameter.grad)
    if set(tensors) != set(names):
        raise RuntimeError("CuPy depth-2 export did not satisfy the tensor contract")
    return tensors


def _memory_gate(
    *, total_bytes: int, observed_bytes: int, limit_fraction: float, semantics: str
) -> dict[str, Any]:
    fraction = float(observed_bytes) / max(int(total_bytes), 1)
    return {
        "limit_fraction": float(limit_fraction),
        "observed_fraction": fraction,
        "observed_bytes": int(observed_bytes),
        "device_total_bytes": int(total_bytes),
        "semantics": semantics,
        "accepted": bool(fraction <= limit_fraction),
    }


def _run_cupy(args, manifest, arrays, reference):
    os.environ["MINI_LLM_BACKEND"] = "cupy"
    os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = "triton"
    os.environ["MINI_LLM_TRAIN_ATTN_STRICT"] = "1"
    os.environ["MINI_LLM_BF16_MIXED_CONTEXT"] = "0"
    os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] = "0"
    os.environ["MINI_LLM_FUSED_BF16_RMSNORM"] = "1"
    os.environ["MINI_LLM_FUSED_BF16_RMSNORM_STRICT"] = "1"
    os.environ["MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM"] = "1"

    import cupy as cp
    from mini_llm.backend import RandomStream
    from mini_llm.blocks.transformer_block import TransformerBlock
    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    setup_start = time.perf_counter_ns()
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    runtime_profile = apply_runtime_defaults(
        "gpu-fast",
        backend_name="cupy",
        config=config,
        training=True,
        precision="bf16-mixed",
    )
    records = _parameter_records(reference, arrays)
    attention_layers = config.attention_layers
    if attention_layers is None:
        raise ValueError("0089E production config must define per-layer attention")
    rng = RandomStream(0)
    blocks = []
    selected_layers = []
    for layer_index in range(args.depth):
        block = TransformerBlock(
            d_model=config.d_model,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            top_k=config.top_k,
            input_std=config.init_std,
            output_std=config.residual_init_std,
            rope_base=config.rope_base,
            rng=rng,
            eps=config.rms_eps,
            name="block",
            dtype="bfloat16",
            attention_config=attention_layers[layer_index],
            residual_gates=False,
            ffn_mode=config.ffn_mode,
        )
        blocks.append(block)
        selected_layers.append(_cupy_selected_parameters(block, records))

    device_sources = {}
    for _, fixture_key in records:
        host = np.asarray(arrays[fixture_key])
        source = cp.asarray(host)
        if host.dtype == np.uint16:
            source = source.view(cp.dtype("bfloat16"))
        else:
            source = source.astype(cp.dtype("bfloat16"), copy=False)
        device_sources[fixture_key] = source
    for block, selected in zip(blocks, selected_layers):
        for _, fixture_key, parameter in selected:
            source = device_sources[fixture_key]
            if source.shape != parameter.data.shape:
                raise ValueError(
                    f"fixture {fixture_key} shape {source.shape} != {parameter.data.shape}"
                )
            parameter.data[...] = source
        block.refresh_compute_buffers()
    del device_sources

    storage = _check_cupy_storage(selected_layers)
    x = cp.asarray(arrays["x"], dtype=cp.float32)
    d_out = cp.asarray(arrays["d_out"], dtype=cp.float32)
    position_ids = cp.asarray(arrays["position_ids"])
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    fixture_setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0

    pool = cp.get_default_memory_pool()
    baseline_pool_used = int(pool.used_bytes())
    baseline_pool_reserved = int(pool.total_bytes())
    _zero_cupy_grads(blocks)
    first_start = time.perf_counter_ns()
    output, caches, boundary = _cupy_forward_stack(blocks, x, position_ids)
    d_input, boundary_gradient = _cupy_backward_stack(blocks, d_out, caches)
    stream.synchronize()
    first_fwd_bwd_ms = (time.perf_counter_ns() - first_start) / 1_000_000.0
    health = _validate_cupy_paths_and_dtypes(
        cp, blocks, selected_layers, output, d_input, caches
    )
    free_after, total_memory = cp.cuda.runtime.memGetInfo()
    observed = max(
        int(pool.total_bytes()), int(total_memory - free_after)
    )
    gate = _memory_gate(
        total_bytes=int(total_memory),
        observed_bytes=observed,
        limit_fraction=args.memory_limit_fraction,
        semantics=(
            "max(CuPy pool high-water, device used after one live complete F+B); "
            "the latter includes non-process device use"
        ),
    )
    memory = {
        "baseline_pool_used_bytes": baseline_pool_used,
        "baseline_pool_reserved_bytes": baseline_pool_reserved,
        "post_preflight_pool_used_bytes": int(pool.used_bytes()),
        "peak_pool_reserved_bytes": int(pool.total_bytes()),
        "device_free_after_preflight_bytes": int(free_after),
        "device_total_bytes": int(total_memory),
        "gate": gate,
    }
    del output, d_input, caches, boundary, boundary_gradient

    timings = None
    tensors = None
    if gate["accepted"]:
        timings = {
            "combined": _time_cupy_combined(
                cp,
                blocks,
                x,
                d_out,
                position_ids,
                args.warmup,
                args.repeats,
            )
        }
        if args.depth <= 2:
            _zero_cupy_grads(blocks)
            output, caches, boundary = _cupy_forward_stack(
                blocks, x, position_ids
            )
            d_input, boundary_gradient = _cupy_backward_stack(
                blocks, d_out, caches
            )
            stream.synchronize()
            tensors = _cupy_export_tensors(
                cp,
                args.depth,
                output,
                d_input,
                boundary,
                boundary_gradient,
                selected_layers,
            )
            del output, d_input, caches, boundary, boundary_gradient

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    capability = {
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device": (
            device_name.decode() if isinstance(device_name, bytes) else str(device_name)
        ),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
    }
    details = {
        "fixture_transfer_and_stack_setup_ms": fixture_setup_ms,
        "first_fwd_bwd_ms": first_fwd_bwd_ms,
        "runtime_profile": {
            "requested": runtime_profile.requested,
            "resolved": runtime_profile.resolved,
            "applied": list(runtime_profile.applied),
        },
        "storage_independence": storage,
        "health": health,
        "environment_contract": {
            name: os.environ[name]
            for name in (
                "MINI_LLM_BACKEND",
                "MINI_LLM_TRAIN_ATTN_BACKEND",
                "MINI_LLM_TRAIN_ATTN_STRICT",
                "MINI_LLM_BF16_MIXED_CONTEXT",
                "MINI_LLM_FUSED_EXPERT_GEMM",
                "MINI_LLM_FUSED_BF16_RMSNORM",
                "MINI_LLM_FUSED_BF16_RMSNORM_STRICT",
                "MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM",
            )
        },
        "cache_structure": "one forward cache per layer; reverse-order backward",
        "gradient_clearing": "outside the timed CUDA event",
    }
    return timings, memory, tensors, capability, details


def _torch_from_fixture(torch, value, *, dtype, requires_grad=False):
    host = np.array(value, copy=True)
    tensor = torch.from_numpy(host)
    if host.dtype == np.uint16 and dtype is torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    tensor = tensor.to(device="cuda", dtype=dtype)
    return tensor.detach().requires_grad_(requires_grad)


def _torch_layer_leaves(torch, arrays, reference, depth):
    base = {
        name: _torch_from_fixture(
            torch,
            arrays[reference.PARAMETER_KEYS[name]],
            dtype=torch.bfloat16,
        )
        for name in reference.PARAMETER_NAMES
    }

    def by_suffix(suffix: str):
        matches = [value for name, value in base.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"expected one fixture parameter ending in {suffix!r}")
        return matches[0]

    packed_base = torch.cat(
        (by_suffix("attention.Wq"), by_suffix("attention.Wk"), by_suffix("attention.Wv")),
        dim=-1,
    )
    templates = (
        by_suffix("norm1.gamma"),
        packed_base,
        by_suffix("attention.Wo"),
        by_suffix("norm2.gamma"),
        by_suffix("W_gate"),
        by_suffix("W_up"),
        by_suffix("W_down"),
    )
    layers = [
        tuple(value.detach().clone().requires_grad_(True) for value in templates)
        for _ in range(depth)
    ]
    return layers


def _make_torch_stack_forward(block_forward, depth):
    depth = validate_depth(depth)

    def forward(x, *flat_layer_leaves):
        value = x
        boundary = None
        for layer_index in range(depth):
            offset = layer_index * TORCH_LEAVES_PER_LAYER
            layer = flat_layer_leaves[offset : offset + TORCH_LEAVES_PER_LAYER]
            value = block_forward(value, *layer)
            if depth == 2 and layer_index == 0:
                boundary = value
        if depth == 2:
            return value, boundary
        return value

    return forward


def _split_torch_output(depth, result):
    if depth == 2:
        output, boundary = result
        return output, boundary
    return result, None


def _zero_torch_grads(x, flat_layer_leaves) -> None:
    x.grad = None
    for leaf in flat_layer_leaves:
        leaf.grad = None


def _torch_export_gradients(torch, layer_leaves):
    exported = []
    q_width = 1024
    k_width = 256
    for layer_index, leaves in enumerate(layer_leaves):
        if any(value.grad is None for value in leaves):
            raise RuntimeError(f"Torch layer {layer_index} has a missing leaf gradient")
        packed_gradient = leaves[1].grad
        exported.extend(
            (
                leaves[0].grad.float(),
                packed_gradient[:, :q_width].float(),
                packed_gradient[:, q_width : q_width + k_width].float(),
                packed_gradient[:, q_width + k_width :].float(),
                leaves[2].grad.float(),
                leaves[3].grad.float(),
                leaves[4].grad.float(),
                leaves[5].grad.float(),
                leaves[6].grad.float(),
            )
        )
    if len(exported) != logical_gradient_count(len(layer_leaves)):
        raise RuntimeError("Torch logical gradient export count is incorrect")
    return exported


def _torch_health(
    torch, x, output, boundary, boundary_gradient, layer_leaves, exported
):
    failures = []
    if output.dtype is not torch.float32:
        failures.append("stack output is not FP32")
    if x.grad is None or x.grad.dtype is not torch.float32:
        failures.append("stack input gradient is not FP32")
    if boundary is not None and boundary.dtype is not torch.float32:
        failures.append("inter-block activation is not FP32")
    if boundary is not None and (
        boundary_gradient is None or boundary_gradient.dtype is not torch.float32
    ):
        failures.append("inter-block gradient is not FP32")
    for layer_index, leaves in enumerate(layer_leaves):
        if any(value.dtype is not torch.bfloat16 for value in leaves):
            failures.append(f"layer {layer_index} has a non-BF16 parameter leaf")
        if any(value.grad is None or value.grad.dtype is not torch.bfloat16 for value in leaves):
            failures.append(f"layer {layer_index} raw parameter gradients are not BF16")
    finite = bool(torch.isfinite(output).all().item())
    finite = finite and x.grad is not None and bool(torch.isfinite(x.grad).all().item())
    finite = finite and all(bool(torch.isfinite(value).all().item()) for value in exported)
    if not finite:
        failures.append("non-finite output or gradient detected")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "finite": True,
        "residual_dtype": "torch.float32",
        "raw_parameter_gradient_dtype": "torch.bfloat16",
        "exported_parameter_gradient_dtype": "torch.float32",
        "parameter_dtype": "torch.bfloat16",
    }


def _torch_boundary_gradient(
    torch, block_forward, boundary, d_out, layer_leaves
):
    if boundary is None:
        return None
    if boundary.grad is not None:
        return boundary.grad
    probe_input = boundary.detach().requires_grad_(True)
    probe_output = block_forward(probe_input, *layer_leaves[1])
    return torch.autograd.grad(
        probe_output, probe_input, d_out, retain_graph=False, create_graph=False
    )[0]


def _execute_torch_combined(torch, forward, depth, x, d_out, layer_leaves):
    flat = tuple(value for layer in layer_leaves for value in layer)
    result = forward(x, *flat)
    output, boundary = _split_torch_output(depth, result)
    if boundary is not None:
        boundary.retain_grad()
    output.backward(d_out)
    exported = _torch_export_gradients(torch, layer_leaves)
    return output, boundary, exported


def _time_torch_combined(
    torch, forward, depth, x, d_out, layer_leaves, warmup, repeats
):
    flat = tuple(value for layer in layer_leaves for value in layer)

    def one(measure: bool):
        _zero_torch_grads(x, flat)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output, boundary, exported = _execute_torch_combined(
            torch, forward, depth, x, d_out, layer_leaves
        )
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(start.elapsed_time(end)),
            (wall_end - wall_start) / 1_000_000.0,
            output,
            boundary,
            exported,
        )

    for _ in range(warmup):
        one(False)
    device_samples = []
    wall_samples = []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        device_ms, wall_ms, output, boundary, exported = measured
        device_samples.append(device_ms)
        wall_samples.append(wall_ms)
        del output, boundary, exported
    return {
        "gpu_event": _summary(device_samples),
        "host_wall": _summary(wall_samples),
    }


def _torch_export_tensors(
    torch, depth, output, boundary, boundary_gradient, x, layer_leaves, exported
):
    names = result_tensor_names(depth)
    tensors = {
        "output": output.detach().float().cpu().numpy(),
        "d_input": x.grad.detach().float().cpu().numpy(),
    }
    if depth == 2:
        tensors["boundary_output"] = boundary.detach().float().cpu().numpy()
        tensors["boundary_grad"] = (
            boundary_gradient.detach().float().cpu().numpy()
        )
        suffixes = tuple(
            importlib.import_module(
                "benchmarks.reference_0089b_torch_block"
            ).RESULT_TENSOR_NAMES[2:]
        )
        cursor = 0
        for layer_index in range(2):
            for suffix in suffixes:
                tensors[f"layer{layer_index}_{suffix}"] = (
                    exported[cursor].detach().cpu().numpy()
                )
                cursor += 1
    else:
        for result_name, gradient in zip(names[2:], exported):
            tensors[result_name] = gradient.detach().cpu().numpy()
    if set(tensors) != set(names):
        raise RuntimeError("Torch export did not satisfy the tensor contract")
    return tensors


def _run_torch(args, manifest, arrays, reference):
    torch = importlib.import_module("torch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    setup_start = time.perf_counter_ns()
    x = _torch_from_fixture(
        torch, arrays["x"], dtype=torch.float32, requires_grad=True
    )
    d_out = _torch_from_fixture(torch, arrays["d_out"], dtype=torch.float32)
    position_ids = _torch_from_fixture(
        torch, arrays["position_ids"], dtype=None
    )
    layer_leaves = _torch_layer_leaves(torch, arrays, reference, args.depth)
    flat = tuple(value for layer in layer_leaves for value in layer)
    if 1 + len(flat) != torch_leaf_count(args.depth):
        raise RuntimeError("Torch stack leaf count is incorrect")
    storage = _check_torch_storage(layer_leaves)
    fixture_setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0

    mask_start = time.perf_counter_ns()
    block_mask = reference.build_block_mask(torch, int(x.shape[1]))
    torch.cuda.synchronize()
    mask_build_ms = (time.perf_counter_ns() - mask_start) / 1_000_000.0
    block_forward = reference.build_torch_forward(torch, block_mask, position_ids)
    eager_forward = _make_torch_stack_forward(block_forward, args.depth)
    compiled = args.backend == "torch-compile"
    forward = eager_forward
    if compiled:
        forward = torch.compile(
            eager_forward,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )

    _zero_torch_grads(x, flat)
    compile_start = time.perf_counter_ns()
    output, boundary, exported = _execute_torch_combined(
        torch, forward, args.depth, x, d_out, layer_leaves
    )
    torch.cuda.synchronize()
    first_fwd_bwd_ms = (time.perf_counter_ns() - compile_start) / 1_000_000.0
    boundary_gradient = _torch_boundary_gradient(
        torch, block_forward, boundary, d_out, layer_leaves
    )
    health = _torch_health(
        torch, x, output, boundary, boundary_gradient, layer_leaves, exported
    )
    del output, boundary, boundary_gradient, exported

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    _zero_torch_grads(x, flat)
    memory_output, memory_boundary, memory_exports = _execute_torch_combined(
        torch, forward, args.depth, x, d_out, layer_leaves
    )
    torch.cuda.synchronize()
    free_after, total_memory = torch.cuda.mem_get_info()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    observed = max(peak_reserved, int(total_memory - free_after))
    gate = _memory_gate(
        total_bytes=int(total_memory),
        observed_bytes=observed,
        limit_fraction=args.memory_limit_fraction,
        semantics=(
            "max(Torch peak reserved, device used after one live complete F+B); "
            "the latter includes non-process device use"
        ),
    )
    memory = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_free_after_preflight_bytes": int(free_after),
        "device_total_bytes": int(total_memory),
        "gate": gate,
    }
    del memory_output, memory_boundary, memory_exports

    timings = None
    tensors = None
    if gate["accepted"]:
        timings = {
            "combined": _time_torch_combined(
                torch,
                forward,
                args.depth,
                x,
                d_out,
                layer_leaves,
                args.warmup,
                args.repeats,
            )
        }
        if args.depth <= 2:
            _zero_torch_grads(x, flat)
            output, boundary, exported = _execute_torch_combined(
                torch, forward, args.depth, x, d_out, layer_leaves
            )
            boundary_gradient = _torch_boundary_gradient(
                torch, block_forward, boundary, d_out, layer_leaves
            )
            torch.cuda.synchronize()
            tensors = _torch_export_tensors(
                torch,
                args.depth,
                output,
                boundary,
                boundary_gradient,
                x,
                layer_leaves,
                exported,
            )
            del output, boundary, boundary_gradient, exported

    try:
        triton_version = importlib.import_module("triton").__version__
    except (ImportError, AttributeError):
        triton_version = None
    capability = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    details = {
        "fixture_transfer_and_stack_setup_ms": fixture_setup_ms,
        "block_mask_build_ms": mask_build_ms,
        "first_fwd_bwd_ms": first_fwd_bwd_ms,
        "compiled": compiled,
        "fullgraph_requested": compiled,
        "dynamic_requested": False if compiled else None,
        "compile_mode": args.compile_mode if compiled else None,
        "whole_fixed_depth_chain_compiled": compiled,
        "per_block_compilation": False,
        "boundary_gradient_probe": (
            "untimed eager single-block VJP only when AOTAutograd returns the "
            "auxiliary boundary without a retained .grad"
            if args.depth == 2 else None
        ),
        "storage_independence": storage,
        "health": health,
        "precision_controls": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "gradient_clearing": "outside the timed CUDA event",
        "timed_gradient_export": (
            "all BF16 logical parameter gradients converted to FP32 inside timing"
        ),
    }
    return timings, memory, tensors, capability, details


def _timing_per_block(timings, depth):
    if timings is None:
        return None
    combined = timings["combined"]
    return {
        "gpu_event_median_ms_per_block": (
            float(combined["gpu_event"]["median_ms"]) / depth
        ),
        "host_wall_median_ms_per_block": (
            float(combined["host_wall"]["median_ms"]) / depth
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reference = importlib.import_module("benchmarks.reference_0089b_torch_block")
    manifest, arrays = reference.load_fixture(args.fixture)
    names = result_tensor_names(args.depth)
    boundary = reference.boundary_contract(sequence_length=int(manifest["seq"]))

    if args.backend == "cupy":
        timings, memory, tensors, capability, details = _run_cupy(
            args, manifest, arrays, reference
        )
    else:
        timings, memory, tensors, capability, details = _run_torch(
            args, manifest, arrays, reference
        )

    parity = None
    result_path = None
    if args.depth <= 2 and memory["gate"]["accepted"]:
        if tensors is None or args.output_tensors is None:
            raise RuntimeError("correctness depth completed without required tensors")
        metadata = {
            "schema_version": "0089E-block-stack-tensors-v1",
            "patch": "0089E",
            "backend": args.backend,
            "depth": args.depth,
            "fixture_sha256": manifest.get("sha256"),
            "result_tensor_names": list(names),
        }
        _save_tensors(args.output_tensors, names, tensors, metadata)
        result_path = str(args.output_tensors.resolve())
        parity = _compare_to_file(tensors, args.reference_tensors, names)

    if not memory["gate"]["accepted"]:
        status = "invalid_memory_gate"
    elif parity is not None and not parity["accepted"]:
        status = "invalid_numerical_parity"
    else:
        status = "valid"

    report = {
        "schema_version": RESULT_SCHEMA,
        "patch": "0089E",
        "scope": (
            "fixed-depth consecutive production Transformer blocks, complete forward "
            "plus full numerical backward, no optimizer or activation checkpointing"
        ),
        "backend": args.backend,
        "depth": args.depth,
        "status": status,
        "fixture_path": str(args.fixture.resolve()),
        "fixture": _jsonable(manifest),
        "mask_boundary": _jsonable(boundary),
        "tensor_contract": {
            "mode": "correctness" if args.depth <= 2 else "timing_only",
            "names": list(names),
            "count": len(names),
            "logical_parameter_gradients": logical_gradient_count(args.depth),
            "torch_leaf_count_including_x": torch_leaf_count(args.depth),
        },
        "dtype_contract": {
            "residuals_and_residual_gradients": "float32",
            "branch_and_parameter_storage": "bfloat16",
            "saved_parameter_gradients": "float32",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
        "combined_ms_per_block": _timing_per_block(timings, args.depth),
        "setup": details,
        "memory": memory,
        "capability": capability,
        "parity": parity,
        "result_tensors": result_path,
        "timing_note": (
            "Complete F+B only. Gradient clearing is outside timing. Torch BF16 "
            "logical parameter-gradient conversion to FP32 is inside timing. Raw "
            "GPU-event and synchronized host-wall samples are retained."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if status == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
