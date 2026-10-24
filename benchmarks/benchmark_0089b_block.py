#!/usr/bin/env python3
"""0089B complete production Transformer-block forward/backward benchmark."""

from __future__ import annotations

import argparse
import importlib
import inspect
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
    return importlib.import_module("benchmarks.reference_0089b_torch_block")


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


def _invoke_named(function: Callable[..., Any], context: Mapping[str, Any]) -> Any:
    """Call a reference helper using its descriptive argument names."""
    signature = inspect.signature(function)
    positional = []
    keywords = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name not in context:
            if parameter.default is not parameter.empty:
                continue
            raise TypeError(
                f"cannot call {function.__name__}: unsupported required helper "
                f"argument {parameter.name!r}"
            )
        value = context[parameter.name]
        if parameter.kind is parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[parameter.name] = value
    return function(*positional, **keywords)


def _parameter_records(
    parameter_names: Sequence[str],
    parameter_keys: Mapping[str, str] | Sequence[str],
    arrays: Mapping[str, Any],
) -> list[tuple[str, str]]:
    names = [str(name) for name in parameter_names]
    if len(names) != 9:
        raise ValueError(f"0089B requires exactly nine parameter names, got {len(names)}")
    if isinstance(parameter_keys, Mapping):
        records = []
        inverse = {str(value): str(key) for key, value in parameter_keys.items()}
        for name in names:
            candidates = [name]
            if name in parameter_keys:
                candidates.insert(0, str(parameter_keys[name]))
            if name in inverse:
                candidates.insert(0, inverse[name])
            fixture_key = next((key for key in candidates if key in arrays), None)
            if fixture_key is None:
                raise KeyError(
                    f"no fixture array for parameter {name!r}; tried {candidates}"
                )
            records.append((name, fixture_key))
        return records
    keys = [str(key) for key in parameter_keys]
    if len(keys) != len(names):
        raise ValueError("PARAMETER_KEYS and PARAMETER_NAMES must have equal length")
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise KeyError(f"fixture is missing parameter arrays: {missing}")
    return list(zip(names, keys))


def _result_names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if len(result) != 11:
        raise ValueError(
            "RESULT_TENSOR_NAMES must name output, d_input, and nine parameter gradients"
        )
    return result


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
            name: np.array(archive[name], dtype=np.float32, copy=True) for name in names
        }
    return metadata, tensors


def _complete_error_metrics(
    helper: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
    actual: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(
            f"parity shape mismatch: actual {actual.shape}, reference {reference.shape}"
        )
    actual_f = np.asarray(actual, dtype=np.float32)
    reference_f = np.asarray(reference, dtype=np.float32)
    difference = actual_f - reference_f
    flat_difference = difference.reshape(-1)
    worst_flat = int(np.argmax(np.abs(flat_difference))) if flat_difference.size else 0
    actual_norm = float(np.linalg.norm(actual_f.reshape(-1)))
    reference_norm = float(np.linalg.norm(reference_f.reshape(-1)))
    difference_norm = float(np.linalg.norm(flat_difference))
    denominator = actual_norm * reference_norm
    cosine = (
        float(np.dot(actual_f.reshape(-1), reference_f.reshape(-1)) / denominator)
        if denominator
        else (1.0 if difference_norm == 0.0 else 0.0)
    )
    calculated = {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "relative_l2": difference_norm / reference_norm
        if reference_norm
        else (0.0 if difference_norm == 0.0 else float("inf")),
        "cosine": cosine,
        "bias": float(np.mean(difference)) if difference.size else 0.0,
        "worst_pair": {
            "flat_index": worst_flat,
            "index": list(np.unravel_index(worst_flat, actual_f.shape))
            if actual_f.size
            else [],
            "actual": float(actual_f.reshape(-1)[worst_flat]) if actual_f.size else 0.0,
            "reference": float(reference_f.reshape(-1)[worst_flat])
            if reference_f.size
            else 0.0,
        },
    }
    supplied = dict(helper(actual_f, reference_f))
    aliases = {
        "rel_l2": "relative_l2",
        "relative_L2": "relative_l2",
        "cosine_similarity": "cosine",
        "mean_bias": "bias",
    }
    for old, new in aliases.items():
        if old in supplied and new not in supplied:
            supplied[new] = supplied[old]
    calculated.update(supplied)
    for key in ("max_abs", "rmse", "relative_l2", "cosine", "bias"):
        calculated[key] = float(calculated[key])
    calculated["worst_pair"] = _jsonable(calculated["worst_pair"])
    return calculated


def _compare(
    tensors: Mapping[str, np.ndarray],
    reference_path: Path | None,
    names: Sequence[str],
    error_metrics: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    metadata, reference = _load_result_tensors(reference_path, names)
    metrics = {
        name: _complete_error_metrics(error_metrics, tensors[name], reference[name])
        for name in names
    }
    accepted = all(
        metric["relative_l2"] <= _RELATIVE_L2_MAX
        and metric["cosine"] >= _COSINE_MIN
        for metric in metrics.values()
    )
    worst_name = max(names, key=lambda name: metrics[name]["relative_l2"])
    return {
        "reference": str(reference_path.resolve()),
        "reference_metadata": metadata,
        "metrics": metrics,
        "worst_pair": {
            "tensor": worst_name,
            **metrics[worst_name]["worst_pair"],
        },
        "tolerances_declared_before_timing": {
            "relative_l2_max": _RELATIVE_L2_MAX,
            "cosine_min": _COSINE_MIN,
        },
        "accepted": bool(accepted),
    }


def _time_cupy_forward(call: Callable[[], Any], warmup: int, repeats: int):
    import cupy as cp

    for _ in range(warmup):
        value = call()
        del value
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        value = call()
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        device_samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        wall_samples.append((wall_end - wall_start) / 1_000_000.0)
        del value
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _time_cupy_backward(block, x, d_out, position_ids, warmup: int, repeats: int):
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    def one(measure: bool):
        _, cache = block.forward(x, return_cache=True, position_ids=position_ids)
        stream.synchronize()
        block.zero_grad()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        d_input = block.backward(d_out, cache)
        end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(cp.cuda.get_elapsed_time(start, end)),
            (wall_end - wall_start) / 1_000_000.0,
            d_input,
        )

    for _ in range(warmup):
        one(False)
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        event_ms, wall_ms, d_input = measured
        device_samples.append(event_ms)
        wall_samples.append(wall_ms)
        del d_input
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _time_cupy_combined(block, x, d_out, position_ids, warmup: int, repeats: int):
    import cupy as cp

    stream = cp.cuda.get_current_stream()

    def one(measure: bool):
        block.zero_grad()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        output, cache = block.forward(x, return_cache=True, position_ids=position_ids)
        d_input = block.backward(d_out, cache)
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
        )

    for _ in range(warmup):
        one(False)
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        event_ms, wall_ms, output, d_input = measured
        device_samples.append(event_ms)
        wall_samples.append(wall_ms)
        del output, d_input
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _match_production_parameters(block, records: Sequence[tuple[str, str]]):
    by_name = {parameter.name: parameter for parameter in block.parameters()}
    if len(by_name) != 9:
        raise ValueError(
            f"production block has {len(by_name)} parameters; 0089B requires nine"
        )
    selected = []
    for requested_name, fixture_key in records:
        parameter = by_name.get(requested_name)
        if parameter is None:
            suffix_matches = [
                candidate
                for name, candidate in by_name.items()
                if name.endswith(f".{requested_name}")
            ]
            if len(suffix_matches) == 1:
                parameter = suffix_matches[0]
        if parameter is None:
            raise KeyError(
                f"fixture parameter name {requested_name!r} does not match production names "
                f"{sorted(by_name)}"
            )
        selected.append((requested_name, fixture_key, parameter))
    if len({id(parameter) for _, _, parameter in selected}) != 9:
        raise ValueError("PARAMETER_NAMES did not resolve to nine unique production parameters")
    return selected


def _run_cupy(args, manifest, arrays, records, result_names):
    os.environ["MINI_LLM_BACKEND"] = "cupy"
    os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = "triton"
    os.environ["MINI_LLM_TRAIN_ATTN_STRICT"] = "1"
    os.environ["MINI_LLM_BF16_MIXED_CONTEXT"] = "0"
    os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] = "0"

    import cupy as cp
    from mini_llm.backend import RandomStream
    from mini_llm.blocks.transformer_block import TransformerBlock
    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    runtime_profile = apply_runtime_defaults(
        "gpu-fast",
        backend_name="cupy",
        config=config,
        training=True,
        precision="bf16-mixed",
    )
    if config.attention_layers is None:
        raise ValueError("0089B production config must define per-layer attention")
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
        rng=RandomStream(0),
        eps=config.rms_eps,
        name="block",
        dtype="bfloat16",
        attention_config=config.attention_layers[0],
        residual_gates=False,
        ffn_mode=config.ffn_mode,
    )
    selected = _match_production_parameters(block, records)
    for _, fixture_key, parameter in selected:
        host = np.asarray(arrays[fixture_key])
        value = cp.asarray(host)
        if host.dtype == np.uint16:
            value = value.view(cp.dtype("bfloat16"))
        else:
            value = value.astype(cp.dtype("bfloat16"), copy=False)
        if value.shape != parameter.data.shape:
            raise ValueError(
                f"shape mismatch for {parameter.name}: fixture {value.shape}, "
                f"production {parameter.data.shape}"
            )
        parameter.data[...] = value
    block.refresh_compute_buffers()

    x = cp.asarray(arrays["x"], dtype=cp.float32)
    d_out = cp.asarray(arrays["d_out"], dtype=cp.float32)
    position_ids = cp.asarray(arrays["position_ids"])
    stream = cp.cuda.get_current_stream()
    stream.synchronize()

    timings = {
        "forward": _time_cupy_forward(
            lambda: block.forward(x, return_cache=True, position_ids=position_ids),
            args.warmup,
            args.repeats,
        ),
        "backward": _time_cupy_backward(
            block, x, d_out, position_ids, args.warmup, args.repeats
        ),
        "combined": _time_cupy_combined(
            block, x, d_out, position_ids, args.warmup, args.repeats
        ),
    }

    pool = cp.get_default_memory_pool()
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    baseline_used = int(pool.used_bytes())
    baseline_reserved = int(pool.total_bytes())
    block.zero_grad()
    memory_output, memory_cache = block.forward(
        x, return_cache=True, position_ids=position_ids
    )
    memory_d_input = block.backward(d_out, memory_cache)
    stream.synchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    memory = {
        "baseline_pool_used_bytes": baseline_used,
        "baseline_pool_reserved_bytes": baseline_reserved,
        "post_combined_pool_used_bytes": int(pool.used_bytes()),
        "peak_pool_reserved_bytes": int(pool.total_bytes()),
        "device_total_bytes": int(total_memory),
        "device_used_before_bytes": int(total_memory - free_before),
        "device_used_after_bytes": int(total_memory - free_after),
        "peak_semantics": (
            "CuPy pool total_bytes is the process allocator high-water mark after the "
            "measured workload; CUDA does not expose a resettable CuPy allocation peak"
        ),
    }
    del memory_output, memory_cache, memory_d_input

    block.zero_grad()
    output, cache = block.forward(x, return_cache=True, position_ids=position_ids)
    d_input = block.backward(d_out, cache)
    stream.synchronize()
    tensors = {
        result_names[0]: cp.asnumpy(output),
        result_names[1]: cp.asnumpy(d_input),
    }
    for result_name, (_, _, parameter) in zip(result_names[2:], selected):
        tensors[result_name] = cp.asnumpy(parameter.grad)

    if args.cuda_profiler_capture:
        block.zero_grad()
        cp.cuda.runtime.profilerStart()
        cp.cuda.nvtx.RangePush("0089B_CUPY_BLOCK_FWD_BWD")
        try:
            profile_output, profile_cache = block.forward(
                x, return_cache=True, position_ids=position_ids
            )
            profile_d_input = block.backward(d_out, profile_cache)
            stream.synchronize()
        finally:
            cp.cuda.nvtx.RangePop()
            cp.cuda.runtime.profilerStop()
        del profile_output, profile_cache, profile_d_input

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    capability = {
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
    }
    details = {
        "runtime_profile": {
            "requested": runtime_profile.requested,
            "resolved": runtime_profile.resolved,
            "applied": list(runtime_profile.applied),
        },
        "environment_contract": {
            name: os.environ[name]
            for name in (
                "MINI_LLM_BACKEND",
                "MINI_LLM_TRAIN_ATTN_BACKEND",
                "MINI_LLM_TRAIN_ATTN_STRICT",
                "MINI_LLM_BF16_MIXED_CONTEXT",
                "MINI_LLM_FUSED_EXPERT_GEMM",
            )
        },
        "production_parameter_names": [parameter.name for _, _, parameter in selected],
        "backward_setup": "forward cache and zero_grad are outside the timed event",
        "combined_setup": "zero_grad is outside the timed event; forward and backward are inside",
    }
    return timings, memory, tensors, capability, details


def _as_torch_tensor(torch, value, *, dtype=None, requires_grad=False):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cuda", dtype=dtype)
    else:
        host = np.array(value, copy=True)
        tensor = torch.from_numpy(host)
        if host.dtype == np.uint16 and dtype is torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        tensor = tensor.to(device="cuda", dtype=dtype)
    return tensor.requires_grad_(requires_grad)


def _torch_fixture_values(torch, helper, manifest, arrays, records):
    context = {"torch": torch, "manifest": manifest, "arrays": arrays, "fixture": arrays}
    supplied = _invoke_named(helper, context)
    supplied_map = supplied if isinstance(supplied, Mapping) else {}
    parameter_map = supplied_map.get("parameters", supplied_map)

    def value_for(primary: str, fixture_key: str | None = None):
        for source in (supplied_map, parameter_map, arrays):
            if primary in source:
                return source[primary]
            if fixture_key is not None and fixture_key in source:
                return source[fixture_key]
        raise KeyError(f"torch fixture tensors do not contain {primary!r}")

    x = _as_torch_tensor(torch, value_for("x"), dtype=torch.float32, requires_grad=True)
    d_out = _as_torch_tensor(torch, value_for("d_out"), dtype=torch.float32)
    position_ids = _as_torch_tensor(torch, value_for("position_ids"))
    parameters = {
        name: _as_torch_tensor(
            torch,
            value_for(name, fixture_key),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for name, fixture_key in records
    }
    return x, d_out, position_ids, parameters


def _parameter_by_suffix(parameters: Mapping[str, Any], suffix: str):
    matches = [value for name, value in parameters.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"expected one parameter ending in {suffix!r}, found {len(matches)}"
        )
    return matches[0]


def _torch_leaves(torch, x, parameters):
    norm1 = _parameter_by_suffix(parameters, "norm1.gamma")
    wq = _parameter_by_suffix(parameters, "attention.Wq")
    wk = _parameter_by_suffix(parameters, "attention.Wk")
    wv = _parameter_by_suffix(parameters, "attention.Wv")
    wo = _parameter_by_suffix(parameters, "attention.Wo")
    norm2 = _parameter_by_suffix(parameters, "norm2.gamma")
    w_gate = _parameter_by_suffix(parameters, "W_gate")
    w_up = _parameter_by_suffix(parameters, "W_up")
    w_down = _parameter_by_suffix(parameters, "W_down")
    packed_qkv = torch.cat((wq, wk, wv), dim=-1).detach().requires_grad_(True)
    leaves = (x, norm1, packed_qkv, wo, norm2, w_gate, w_up, w_down)
    components = {
        "norm1": norm1,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "norm2": norm2,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "wq_width": int(wq.shape[-1]),
        "wk_width": int(wk.shape[-1]),
    }
    return leaves, components


def _make_forward_adapter(built_forward, leaves, components):
    if isinstance(built_forward, tuple):
        built_forward = built_forward[0]
    if not callable(built_forward):
        raise TypeError("build_torch_forward must return a callable")
    signature = inspect.signature(built_forward)
    positional_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind is parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    named_values = {
        "x": leaves[0],
        "norm1_gamma": leaves[1],
        "gamma1": leaves[1],
        "wqkv": leaves[2],
        "w_qkv": leaves[2],
        "packed_wqkv": leaves[2],
        "wo": leaves[3],
        "w_o": leaves[3],
        "norm2_gamma": leaves[4],
        "gamma2": leaves[4],
        "w_gate": leaves[5],
        "wgate": leaves[5],
        "w_up": leaves[6],
        "wup": leaves[6],
        "w_down": leaves[7],
        "wdown": leaves[7],
    }
    if has_varargs or len(positional_parameters) == len(leaves):
        def forward(*current_leaves):
            value = built_forward(*current_leaves)
            return value[0] if isinstance(value, tuple) else value

        return forward
    if all(parameter.name in named_values for parameter in positional_parameters):
        positions = [
            list(named_values).index(parameter.name) for parameter in positional_parameters
        ]
        ordered_names = list(named_values)

        def forward(*current_leaves):
            current = dict(named_values)
            for name, value in zip(
                ("x", "norm1_gamma", "wqkv", "wo", "norm2_gamma", "w_gate", "w_up", "w_down"),
                current_leaves,
            ):
                current[name] = value
            arguments = [current[ordered_names[index]] for index in positions]
            value = built_forward(*arguments)
            return value[0] if isinstance(value, tuple) else value

        return forward
    if len(positional_parameters) == 1:
        def forward(*current_leaves):
            value = built_forward(current_leaves[0])
            return value[0] if isinstance(value, tuple) else value

        return forward
    raise TypeError(
        "unsupported torch forward signature; expected one x argument or the eight "
        "0089B leaves"
    )


def _zero_torch_grads(leaves) -> None:
    for leaf in leaves:
        leaf.grad = None


def _torch_gradient_exports(torch, leaves, components, parameters):
    if any(leaf.grad is None for leaf in leaves):
        raise RuntimeError("torch backward did not populate every benchmark leaf gradient")
    q_end = components["wq_width"]
    k_end = q_end + components["wk_width"]
    packed = leaves[2].grad
    by_tensor_id = {
        id(components["norm1"]): leaves[1].grad.float(),
        id(components["wo"]): leaves[3].grad.float(),
        id(components["norm2"]): leaves[4].grad.float(),
        id(components["w_gate"]): leaves[5].grad.float(),
        id(components["w_up"]): leaves[6].grad.float(),
        id(components["w_down"]): leaves[7].grad.float(),
        id(components["wq"]): packed[:, :q_end].float(),
        id(components["wk"]): packed[:, q_end:k_end].float(),
        id(components["wv"]): packed[:, k_end:].float(),
    }
    return [by_tensor_id[id(parameter)] for parameter in parameters.values()]


def _time_torch_forward(torch, forward, leaves, warmup: int, repeats: int):
    for _ in range(warmup):
        output = forward(*leaves)
        del output
    torch.cuda.synchronize()
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output = forward(*leaves)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        device_samples.append(float(start.elapsed_time(end)))
        wall_samples.append((wall_end - wall_start) / 1_000_000.0)
        del output
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _time_torch_backward(
    torch, forward, leaves, d_out, components, parameters, warmup: int, repeats: int
):
    def one(measure: bool):
        output = forward(*leaves)
        torch.cuda.synchronize()
        _zero_torch_grads(leaves)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output.backward(d_out)
        exported = _torch_gradient_exports(torch, leaves, components, parameters)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(start.elapsed_time(end)),
            (wall_end - wall_start) / 1_000_000.0,
            exported,
        )

    for _ in range(warmup):
        one(False)
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        event_ms, wall_ms, exported = measured
        device_samples.append(event_ms)
        wall_samples.append(wall_ms)
        del exported
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _time_torch_combined(
    torch, forward, leaves, d_out, components, parameters, warmup: int, repeats: int
):
    def one(measure: bool):
        _zero_torch_grads(leaves)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output = forward(*leaves)
        output.backward(d_out)
        exported = _torch_gradient_exports(torch, leaves, components, parameters)
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        if not measure:
            return None
        return (
            float(start.elapsed_time(end)),
            (wall_end - wall_start) / 1_000_000.0,
            output,
            exported,
        )

    for _ in range(warmup):
        one(False)
    device_samples, wall_samples = [], []
    for _ in range(repeats):
        measured = one(True)
        assert measured is not None
        event_ms, wall_ms, output, exported = measured
        device_samples.append(event_ms)
        wall_samples.append(wall_ms)
        del output, exported
    return {"gpu_event": _summary(device_samples), "host_wall": _summary(wall_samples)}


def _run_torch(args, manifest, arrays, records, result_names, reference):
    torch = importlib.import_module("torch")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    x, d_out, position_ids, parameters = _torch_fixture_values(
        torch, reference.torch_fixture_tensors, manifest, arrays, records
    )
    leaves, components = _torch_leaves(torch, x, parameters)
    helper_context = {
        "torch": torch,
        "manifest": manifest,
        "arrays": arrays,
        "fixture": arrays,
        "position_ids": position_ids,
        "seq": int(x.shape[1]),
        "sequence_length": int(x.shape[1]),
        "batch": int(x.shape[0]),
    }
    mask_start = time.perf_counter_ns()
    block_mask = _invoke_named(reference.build_block_mask, helper_context)
    torch.cuda.synchronize()
    mask_build_ms = (time.perf_counter_ns() - mask_start) / 1_000_000.0

    helper_context.update(
        {
            "block_mask": block_mask,
            "mask": block_mask,
            "x": leaves[0],
            "norm1_gamma": leaves[1],
            "packed_wqkv": leaves[2],
            "wqkv": leaves[2],
            "wo": leaves[3],
            "norm2_gamma": leaves[4],
            "w_gate": leaves[5],
            "w_up": leaves[6],
            "w_down": leaves[7],
        }
    )
    built_forward = _invoke_named(reference.build_torch_forward, helper_context)
    eager_forward: Callable[..., Any] = _make_forward_adapter(
        built_forward, leaves, components
    )
    compiled = args.backend == "torch-compile"
    forward: Callable[..., Any] = eager_forward
    compile_first_fwd_bwd_ms = None
    if compiled:
        forward = torch.compile(
            eager_forward,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
        _zero_torch_grads(leaves)
        compile_start = time.perf_counter_ns()
        compile_output = forward(*leaves)
        compile_output.backward(d_out)
        compile_exports = _torch_gradient_exports(
            torch, leaves, components, parameters
        )
        torch.cuda.synchronize()
        compile_first_fwd_bwd_ms = (
            time.perf_counter_ns() - compile_start
        ) / 1_000_000.0
        del compile_output, compile_exports

    timings = {
        "forward": _time_torch_forward(
            torch, forward, leaves, args.warmup, args.repeats
        ),
        "backward": _time_torch_backward(
            torch,
            forward,
            leaves,
            d_out,
            components,
            parameters,
            args.warmup,
            args.repeats,
        ),
        "combined": _time_torch_combined(
            torch,
            forward,
            leaves,
            d_out,
            components,
            parameters,
            args.warmup,
            args.repeats,
        ),
    }

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    _zero_torch_grads(leaves)
    memory_output = forward(*leaves)
    memory_output.backward(d_out)
    memory_exports = _torch_gradient_exports(torch, leaves, components, parameters)
    torch.cuda.synchronize()
    memory = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del memory_output, memory_exports

    _zero_torch_grads(leaves)
    output = forward(*leaves)
    output.backward(d_out)
    exported = _torch_gradient_exports(torch, leaves, components, parameters)
    torch.cuda.synchronize()
    tensors = {
        result_names[0]: output.detach().float().cpu().numpy(),
        result_names[1]: leaves[0].grad.detach().float().cpu().numpy(),
    }
    for result_name, gradient in zip(result_names[2:], exported):
        tensors[result_name] = gradient.detach().cpu().numpy()

    if args.cuda_profiler_capture:
        _zero_torch_grads(leaves)
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("0089B_TORCH_BLOCK_FWD_BWD"):
                profile_output = forward(*leaves)
                profile_output.backward(d_out)
                profile_exports = _torch_gradient_exports(
                    torch, leaves, components, parameters
                )
                torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        del profile_output, profile_exports

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
    details = {
        "compiled": compiled,
        "fullgraph_requested": compiled,
        "compile_mode": args.compile_mode if compiled else None,
        "compile_first_fwd_bwd_ms": compile_first_fwd_bwd_ms,
        "block_mask_build_ms": mask_build_ms,
        "torch_leaf_dtypes": {
            "x": str(leaves[0].dtype),
            "norm1_gamma": str(leaves[1].dtype),
            "packed_wqkv": str(leaves[2].dtype),
            "wo": str(leaves[3].dtype),
            "norm2_gamma": str(leaves[4].dtype),
            "w_gate": str(leaves[5].dtype),
            "w_up": str(leaves[6].dtype),
            "w_down": str(leaves[7].dtype),
        },
        "precision_controls": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "backward_setup": (
            "forward graph/cache and grad clearing are outside the timed event; "
            "backward and BF16 parameter-grad FP32 conversions are inside"
        ),
        "combined_setup": (
            "grad clearing is outside the timed event; forward, backward, and "
            "BF16 parameter-grad FP32 conversions are inside"
        ),
        "packed_qkv_gradient_export": "split into Wq, Wk, and Wv in fixture parameter order",
    }
    return timings, memory, tensors, capability, details


def main() -> int:
    args = _parse_args()
    reference = _reference_module()
    required = (
        "PARAMETER_NAMES",
        "PARAMETER_KEYS",
        "RESULT_TENSOR_NAMES",
        "load_fixture",
        "error_metrics",
        "boundary_contract",
        "build_block_mask",
        "build_torch_forward",
        "torch_fixture_tensors",
    )
    missing = [name for name in required if not hasattr(reference, name)]
    if missing:
        raise ImportError(f"reference_0089b_torch_block is missing helpers: {missing}")

    manifest, arrays = reference.load_fixture(args.fixture)
    records = _parameter_records(reference.PARAMETER_NAMES, reference.PARAMETER_KEYS, arrays)
    result_names = _result_names(reference.RESULT_TENSOR_NAMES)
    boundary = _invoke_named(
        reference.boundary_contract,
        {
            "manifest": manifest,
            "arrays": arrays,
            "fixture": arrays,
            "position_ids": arrays["position_ids"],
            "seq": int(np.asarray(arrays["x"]).shape[1]),
            "sequence_length": int(np.asarray(arrays["x"]).shape[1]),
        },
    )

    if args.backend == "cupy":
        timings, memory, tensors, capability, details = _run_cupy(
            args, manifest, arrays, records, result_names
        )
    else:
        timings, memory, tensors, capability, details = _run_torch(
            args, manifest, arrays, records, result_names, reference
        )

    metadata = {
        "patch": "0089B",
        "backend": args.backend,
        "fixture_sha256": manifest.get("sha256"),
        "parameter_names": list(reference.PARAMETER_NAMES),
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
        "schema_version": "0089B-production-block-result-v1",
        "patch": "0089B",
        "scope": (
            "complete production TransformerBlock (B,T,1024), forward plus full "
            "numerical backward including input and all nine parameter gradients"
        ),
        "backend": args.backend,
        "status": "valid" if parity is None or parity["accepted"] else "invalid_numerical_parity",
        "fixture_path": str(args.fixture.resolve()),
        "fixture": _jsonable(manifest),
        "mask_boundary": _jsonable(boundary),
        "dtype_contract": {
            "input_x": "float32",
            "upstream_d_out": "float32",
            "parameters": "byte-exact bfloat16 fixture bits",
            "saved_output_d_input_and_parameter_gradients": "float32",
            "torch_qkv": "one packed bfloat16 leaf, split to Wq/Wk/Wv gradients on export",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
        "memory": memory,
        "capability": capability,
        "details": details,
        "parity": parity,
        "result_tensors": str(args.output_tensors.resolve()),
        "timing_note": (
            "Each phase reports its own raw GPU-event and synchronized host-wall "
            "samples and distribution. Forward, backward, and combined medians are "
            "independent measurements and must not be added."
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
