"""0106N bounded CuPy correctness smoke, NOT a throughput benchmark.

MINI_LLM_BACKEND=cupy .venv/bin/python benchmarks/benchmark_0106n_recurrent_gpu.py \
    --output /tmp/0106n.json [--model /path/model.gguf --layer 0 --tokens 4]

CPU fixtures/oracles run in an isolated NumPy subprocess, never by switching the
parent backend. Importing this module requires only NumPy, not CUDA or torch.
Synthetic fixtures reuse 0106M factories and independent FP64 oracles. Actual
checkpoint mode reads one recurrent layer with native CPU packed projections;
no expert bank or complete model is dequantized. NPZ files never use pickle.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 256 * 1024**2
CASES = ("delta-default", "delta-map", "conv", "norm-silu", "norm-sigmoid",
         "mixer-dense", "mixer-packed", "model")


def read_npz(path):
    """Return owned host arrays and a JSON manifest; reject object payloads."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"].item()))
    return manifest, arrays


def create_cpu_fixture(path, case="mixer-dense", *, dtype="float32", tokens=4,
                       model=None, layer=0, timeout=90):
    """Export references without importing CPU-only test modules in this process."""
    command = [sys.executable, str(Path(__file__).resolve()), "--write-cpu-fixture",
               str(path), "--case", case, "--dtype", dtype, "--tokens", str(tokens),
               "--layer", str(layer)]
    if model is not None:
        command += ["--model", str(model)]
    env = dict(os.environ, MINI_LLM_BACKEND="numpy", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"CPU fixture export failed ({completed.returncode}):\n"
                           f"{completed.stdout}\n{completed.stderr}")
    return read_npz(path)


def _state_arrays(state):
    if hasattr(state, "conv"):
        return {"history": state.conv.history, "memory": state.delta.memory}
    if hasattr(state, "history"):
        return {"history": state.history}
    if hasattr(state, "memory"):
        return {"memory": state.memory}
    return {}


def _save_state(arrays, prefix, state):
    for name, value in _state_arrays(state).items():
        arrays[prefix + name] = value.copy()


def _model_budget(index, spec, layer, batch):
    """Header-only conservative weight + state + workspace bound, before upload."""
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS
    weights = 0
    headers = []
    for suffix, shape in spec.tensor_shapes().items():
        tensor = index.tensors[f"blk.{layer}." + suffix]
        kind = tensor.tensor_type.name
        elements = int(np.prod(shape))
        size = (elements // BLOCK_FORMATS[kind][0] * BLOCK_FORMATS[kind][1]
                if kind in BLOCK_FORMATS else elements * 4)
        weights += size
        headers.append(dict(name=tensor.name, shape=shape, quant_type=kind, bytes=size))
    state = batch * ((spec.conv_kernel_size - 1) * spec.qkv_channels +
                     spec.n_value_heads * spec.key_dim * spec.value_dim) * 4
    # Includes independent copies, recurrence intermediates, and projection scratch.
    bound = weights + 12 * state + 32 * 1024**2
    if bound >= BUDGET:
        raise MemoryError(f"Layer weights/state/workspace bound {bound} >= {BUDGET}")
    return dict(weight_bytes=weights, state_bytes=state, conservative_bytes=bound,
                headers=headers)


def _write_cpu_fixture(args):
    sys.path.insert(0, str(ROOT))
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != "numpy":
        raise RuntimeError("Fixture writer requires an isolated NumPy process")
    arrays = {}
    rng = np.random.default_rng(10614)
    manifest = dict(case=args.case, dtype=args.dtype, tokens=args.tokens, layer=args.layer,
                    model=str(args.model) if args.model else None, seed=10614)
    reference = None
    if args.case.startswith("delta"):
        functions = runpy.run_path(str(ROOT / "tests/test_0106m_gated_delta.py"))
        mapping = (2, 0, 2, 1, 0) if args.case == "delta-map" else None
        heads = (3, 5) if mapping else (2, 6)
        op, inputs, initial = functions["fixture"](np.dtype(args.dtype), heads, mapping)
        manifest.update(kind="delta", config=dict(n_key_heads=heads[0],
                        n_value_heads=heads[1], key_dim=3, value_dim=5,
                        head_map=op.head_map, state_dtype=args.dtype))
        reference = lambda values, state: functions["oracle"](
            *values, state.memory, op.head_map, op.query_scale)
    elif args.case == "conv":
        from mini_llm.ops.causal_conv_inference import CausalDepthwiseConv1DInference
        functions = runpy.run_path(str(ROOT / "tests/test_0106m_causal_conv.py"))
        config = dict(channels=4, kernel_size=4, activation="silu", compute_dtype=args.dtype)
        op = CausalDepthwiseConv1DInference(**config)
        weight = rng.normal(0, .2, (4, 4)).astype(args.dtype)
        bias = rng.normal(0, .1, 4).astype(args.dtype)
        op.set_weights(weight, bias)
        inputs = [rng.normal(0, .4, (2, 9, 4)).astype(args.dtype)]
        initial = op.create_state(2)
        initial.history[...] = rng.normal(0, .3, initial.history.shape)
        initial.length = 3
        arrays.update(weight=weight, bias=bias)
        manifest.update(kind="conv", config=config)
        def reference(values, state):
            value = functions["_reference"](values[0], weight, bias, "silu", state.history)
            raw = np.concatenate((state.history, values[0]), axis=1)
            return value, raw[:, -(config["kernel_size"] - 1):]
    elif args.case.startswith("norm"):
        from mini_llm.ops.gated_norm_inference import GatedRMSNormInference
        config = dict(width=5, gate_activation=args.case.split("-", 1)[1],
                      compute_dtype="float64" if args.dtype == "float64" else "float32")
        op = GatedRMSNormInference(**config)
        arrays["weight"] = np.linspace(.3, 1.5, 5).astype(args.dtype)
        op.set_weights(arrays["weight"])
        inputs = [rng.normal(0, .3, (2, 9, 4, 5)).astype(args.dtype),
                  rng.normal(0, .8, (2, 9, 4, 5)).astype(args.dtype)]
        # Include near-zero rows and large gates without overflowing the oracle.
        inputs[0][0, 0] = 1e-5
        inputs[1][0, 0, 0] = np.array([-30, -1, 0, 1, 30], dtype=args.dtype)
        x, gate = (value.astype(np.float64) for value in inputs)
        ref = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6)
        activation = np.exp(-np.logaddexp(0, -gate))
        if config["gate_activation"] == "silu":
            activation *= gate
        arrays["ref_output"] = (ref * arrays["weight"].astype(np.float64) * activation).astype(args.dtype)
        arrays["cpu_output"] = op.forward(*inputs)
        manifest.update(kind="norm", config=config)
        initial = None
    else:
        from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
        from mini_llm.external.qwen3next_mixer import (Qwen3NextMixerSpec,
                                                     Qwen3NextRecurrentMixerInference)
        functions = None
        if args.case == "model":
            if args.model is None:
                raise ValueError("--case model requires --model")
            index = Qwen3NextGGUFIndex.open(args.model)
            layer, batch = args.layer, 1
        else:
            functions = runpy.run_path(str(ROOT / "tests/test_0106m_qwen3next_mixer.py"))
            index, weights, _ = functions["_fixture"](packed=args.case == "mixer-packed")
            layer, batch = 0, 2
        spec = Qwen3NextMixerSpec.from_index(index)
        manifest["budget"] = _model_budget(index, spec, layer, batch)
        manifest.update(kind="mixer", spec=asdict(spec), layer=layer,
                        cpu_packed_backend="native" if args.case == "model" else "reference")
        if functions is not None:
            fields = {key: value.contents(0) for key, value in index.reader.fields.items()}
            tensors = []
            for n, tensor in enumerate(index.reader.tensors):
                key = f"payload_{n}"
                arrays[key] = np.asarray(tensor.data).copy()
                tensors.append(dict(name=tensor.name, shape=list(tensor.shape),
                                    kind=tensor.tensor_type.name, payload=key))
            manifest["index"] = dict(fields=fields, tensors=tensors)
        op = Qwen3NextRecurrentMixerInference.from_index(
            index, layer, manifest["cpu_packed_backend"])
        prefix = rng.normal(0, .2, (batch, 3, spec.d_model)).astype(np.float32)
        inputs = [rng.normal(0, .2, (batch, args.tokens, spec.d_model)).astype(np.float32)]
        initial = op.create_state(batch)
        op.forward(prefix, initial)
        if functions is not None:
            reference = lambda values, state: functions["_reference"](
                values[0], weights, spec, state.conv.history, state.delta.memory)
    for n, value in enumerate(inputs):
        arrays[f"input_{n}"] = value
    manifest["input_count"] = len(inputs)
    if initial is not None:
        manifest["initial_length"] = initial.length
        _save_state(arrays, "initial_", initial)
        doubled = [np.concatenate((value, value), axis=1) for value in inputs]
        state = initial.copy()
        begin = time.perf_counter()
        arrays["cpu_output"] = op.forward(*doubled, state)
        manifest["cpu_wall_ms"] = (time.perf_counter() - begin) * 1000
        _save_state(arrays, "cpu_", state)
        manifest["final_length"] = state.length
        fresh = initial.copy()
        fresh.reset()
        arrays["reset_output"] = op.forward(*inputs, fresh)
        _save_state(arrays, "reset_", fresh)
        if reference is not None:
            result = reference(doubled, initial)
            arrays["ref_output"] = result[0]
            for name, value in zip(_state_arrays(state), result[1:]):
                arrays["ref_" + name] = value
        else:
            manifest["reference_note"] = "Native CPU thin mixer; no independent full-checkpoint FP64 graph"
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez(args.write_cpu_fixture, **arrays)


class _Field:
    def __init__(self, value):
        self.value = value

    def contents(self, index):
        return self.value


def index_from_manifest(manifest, arrays):
    """Reconstruct real GGUF index headers/payloads, not model mathematics."""
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    if manifest["model"] is not None:
        return Qwen3NextGGUFIndex.open(manifest["model"])
    source = manifest["index"]
    reader = SimpleNamespace(fields={key: _Field(value) for key, value in source["fields"].items()},
                             tensors=[SimpleNamespace(name=tensor["name"], shape=tensor["shape"],
                                      tensor_type=SimpleNamespace(name=tensor["kind"]),
                                      data=arrays[tensor["payload"]]) for tensor in source["tensors"]])
    return Qwen3NextGGUFIndex(reader)


def numerical_metrics(actual, expected, *, rtol, atol, path):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, path=path, actual_shape=actual.shape,
                    expected_shape=expected.shape, dtype=str(actual.dtype), rtol=rtol, atol=atol)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = atol + rtol * np.abs(expected.astype(np.float64))
    ratio = error / np.maximum(tolerance, np.finfo(np.float64).tiny)
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    index = np.unravel_index(int(np.argmax(np.nan_to_num(ratio, nan=np.inf))), error.shape) if error.size else ()
    def number(value):
        value = float(value)
        return value if np.isfinite(value) else str(value)
    return dict(passed=bool(finite and np.all(error <= tolerance)), path=path,
                dtype=str(actual.dtype), reference_dtype=str(expected.dtype), rtol=rtol, atol=atol,
                max_abs_error=number(error.max()) if error.size else 0,
                max_tolerance_ratio=number(ratio.max()) if error.size else 0,
                worst_index=[int(i) for i in index],
                actual=number(actual[index]) if error.size else None,
                reference=number(expected[index]) if error.size else None,
                absolute_error=number(error[index]) if error.size else 0,
                relative_error=number(error[index] / max(abs(float(expected[index])), np.finfo(np.float64).tiny)) if error.size else 0)


@contextmanager
def bounded_gpu():
    """Private bounded allocator; never size a cache from available desktop VRAM."""
    import cupy as cp
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=BUDGET)
    try:
        with cp.cuda.using_allocator(pool.malloc):
            yield cp, pool
    finally:
        pool.free_all_blocks()


def validate_gpu(manifest, arrays, output, *, non_default=True):
    """Save metrics/API failures BEFORE asserting. Stop at the first failed check."""
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != "cupy":
        raise RuntimeError("GPU validation requires MINI_LLM_BACKEND=cupy")
    output = Path(output)
    report = dict(patch="0106N", case=manifest["case"], backend=BACKEND_NAME,
                  scope="single recurrent mixer/generic ops; correctness and timing smoke only",
                  manifest={key: value for key, value in manifest.items() if key != "index"},
                  budget_bytes=BUDGET, comparisons={}, timings={}, passed=False,
                  timing_warning="CUDA event and synchronized wall timings are smoke-only, not throughput")
    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    half = manifest["dtype"] == "float16"
    tol = ((.002, .0005) if half else
           ((3e-4, 5e-6) if manifest["kind"] == "mixer" else
            ((2e-14, 2e-14) if manifest["dtype"] == "float64" else (2e-6, 2e-6))))
    def check(name, actual, expected, tolerance=None):
        import cupy as cp
        host = cp.asnumpy(actual) if isinstance(actual, cp.ndarray) else np.asarray(actual)
        rtol, atol = tolerance or tol
        metric = numerical_metrics(host, expected, rtol=rtol, atol=atol, path=name)
        report["comparisons"][name] = metric
        if not metric["passed"]:
            np.savez(output.with_suffix(".failure.npz"), actual=host, expected=expected)
            save()
            raise AssertionError(json.dumps(metric))
    try:
        with bounded_gpu() as (cp, pool):
            properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            report["device"] = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
            stream = cp.cuda.Stream(non_blocking=True) if non_default else cp.cuda.Stream.null
            report["non_default_stream"] = bool(stream.ptr)
            with stream:
                if cp.cuda.get_current_stream().ptr != stream.ptr:
                    raise AssertionError("Requested stream is not current")
                kind = manifest["kind"]
                if kind == "mixer":
                    from mini_llm.external.qwen3next_mixer import Qwen3NextRecurrentMixerInference
                    index = index_from_manifest(manifest, arrays)
                    from mini_llm.external.qwen3next_mixer import Qwen3NextMixerSpec
                    report["budget_estimate"] = _model_budget(index, Qwen3NextMixerSpec.from_index(index), manifest["layer"], arrays["input_0"].shape[0])
                    op = Qwen3NextRecurrentMixerInference.from_index(index, manifest["layer"])
                elif kind == "delta":
                    from mini_llm.ops.gated_delta_inference import GatedDeltaRuleInference
                    op = GatedDeltaRuleInference(**manifest["config"])
                elif kind == "conv":
                    from mini_llm.ops.causal_conv_inference import CausalDepthwiseConv1DInference
                    op = CausalDepthwiseConv1DInference(**manifest["config"])
                    op.set_weights(cp.asarray(arrays["weight"]), cp.asarray(arrays["bias"]))
                else:
                    from mini_llm.ops.gated_norm_inference import GatedRMSNormInference
                    op = GatedRMSNormInference(**manifest["config"])
                    op.set_weights(cp.asarray(arrays["weight"]))
                inputs = [cp.asarray(arrays[f"input_{n}"]) for n in range(manifest["input_count"])]
                if not all(isinstance(value, cp.ndarray) for value in inputs):
                    raise AssertionError("Validation inputs must be real CuPy arrays")
                initial = None if kind == "norm" else op.create_state(inputs[0].shape[0])
                if initial is not None:
                    for name, value in _state_arrays(initial).items():
                        value[...] = cp.asarray(arrays["initial_" + name])
                    if kind == "mixer":
                        initial.conv.length = initial.delta.length = manifest["initial_length"]
                    else:
                        initial.length = manifest["initial_length"]
                doubled = [cp.concatenate((value, value), axis=1) for value in inputs]
                modes = ("full", "chunks", "steps", "repeat") if initial is not None else ("full", "repeat")
                for mode in modes:
                    state = initial.copy() if initial is not None else None
                    storage = {name: value.data.ptr for name, value in _state_arrays(state).items()}
                    # Compile/warm on an independent state, outside event timing.
                    if mode == "full":
                        op.forward(*inputs, initial.copy()) if initial is not None else op.forward(*inputs)
                    stream.synchronize()
                    begin, end = cp.cuda.Event(), cp.cuda.Event()
                    wall = time.perf_counter()
                    begin.record(stream)
                    if kind == "norm":
                        actual = op.forward(*inputs)
                    elif mode == "full":
                        actual = op.forward(*doubled, state)
                    elif mode == "repeat":
                        actual = cp.concatenate([op.forward(*inputs, state) for _ in range(2)], axis=1)
                    elif mode == "steps":
                        actual = cp.stack([op.step(*(value[:, t] for value in doubled), state)
                                           for t in range(doubled[0].shape[1])], axis=1)
                    else:
                        n = inputs[0].shape[1]
                        boundaries = (0, 1, 1, n, n + 1, 2*n)
                        actual = cp.concatenate([op.forward(*(value[:, a:b] for value in doubled), state)
                                                 for a, b in zip(boundaries, boundaries[1:])], axis=1)
                    end.record(stream)
                    end.synchronize()
                    report["timings"][mode] = dict(cuda_event_ms=float(cp.cuda.get_elapsed_time(begin, end)),
                                                       wall_ms=(time.perf_counter() - wall)*1000)
                    if not isinstance(actual, cp.ndarray) or str(actual.dtype) != manifest["dtype"]:
                        raise AssertionError(f"{mode}: output type/dtype {type(actual)} {actual.dtype}")
                    check(f"{mode}/cpu/output", actual, arrays["cpu_output"])
                    if "ref_output" in arrays:
                        check(f"{mode}/fp64/output", actual, arrays["ref_output"])
                        check("cpu/fp64/output", arrays["cpu_output"], arrays["ref_output"])
                    if state is not None:
                        if state.length != manifest["final_length"]:
                            raise AssertionError(f"{mode}/length {state.length} != {manifest['final_length']}")
                        for name, value in _state_arrays(state).items():
                            if value.data.ptr != storage[name]:
                                raise AssertionError(f"{mode}/{name}: storage identity changed")
                            check(f"{mode}/cpu/{name}", value, arrays["cpu_" + name])
                            if "ref_" + name in arrays:
                                check(f"{mode}/fp64/{name}", value, arrays["ref_" + name])
                        fork = state.copy()
                        state.reset()
                        if state.length != 0:
                            raise AssertionError(f"{mode}/reset length {state.length}")
                        for name, value in _state_arrays(state).items():
                            if value.data.ptr != storage[name] or value.data.ptr == _state_arrays(fork)[name].data.ptr:
                                raise AssertionError(f"{mode}/{name}: reset/copy storage contract")
                            check(f"{mode}/reset/{name}", value, np.zeros(value.shape, value.dtype), (0, 0))
                            check(f"{mode}/fork/{name}", _state_arrays(fork)[name], arrays["cpu_" + name])
                            check(f"{mode}/initial/{name}", _state_arrays(initial)[name], arrays["initial_" + name], (0, 0))
                        reset_actual = op.forward(*inputs, state)
                        check(f"{mode}/reset/output", reset_actual, arrays["reset_output"])
                        for name, value in _state_arrays(state).items():
                            check(f"{mode}/reset_final/{name}", value, arrays["reset_" + name])
                        del fork, state
                stream.synchronize()
                report["pool_peak_reserved_bytes"] = int(pool.total_bytes())
                report["pool_limit_bytes"] = int(pool.get_limit())
            report["passed"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-cpu-fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, default="mixer-dense", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.write_cpu_fixture:
        _write_cpu_fixture(args)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.model is not None and (args.layer != 0 or args.tokens != 4):
        parser.error("Actual-model smoke is restricted to --layer 0 --tokens 4")
    sys.path.insert(0, str(ROOT))
    if os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "cupy":
        parser.error("Use MINI_LLM_BACKEND=cupy; importing this module is CPU-safe")
    result = dict(patch="0106N", passed=False, runs=[])
    try:
        with tempfile.TemporaryDirectory(prefix="0106n-") as directory:
            cases = ("model",) if args.model else CASES[:-1]
            for case in cases:
                fixture = Path(directory) / (case + ".npz")
                manifest, arrays = create_cpu_fixture(fixture, case, tokens=args.tokens,
                                                      model=args.model, layer=args.layer)
                detail = args.output.with_name(args.output.stem + "." + case + ".json")
                result["runs"].append(validate_gpu(manifest, arrays, detail))
        result["passed"] = True
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
