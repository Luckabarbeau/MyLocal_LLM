"""Bounded Qwen3-Next attention CuPy correctness/timing smoke, not throughput.

MINI_LLM_BACKEND=cupy .venv/bin/python benchmarks/benchmark_0106n_attention_gpu.py \
    --output /tmp/0106n-attention.json [--model /path/model.gguf --layer 3 --tokens 4]

Synthetic default validates dense and Q8 fixtures with implicit and explicit /
prepared batch positions. CPU-only fixture factories and FP64 references run
only in isolated NumPy subprocesses. Actual-model mode reads one attention
layer, using native CPU packed projections; no expert/vocabulary decode, torch,
or whole-model dequantization. Import and --help require no CUDA initialization.
"""

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_HELPERS = ROOT / "benchmarks/benchmark_0106n_recurrent_gpu.py"
_spec = importlib.util.spec_from_file_location("_recurrent_gpu_helpers_0106n", _HELPERS)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
read_npz = _helpers.read_npz
index_from_manifest = _helpers.index_from_manifest
numerical_metrics = _helpers.numerical_metrics
bounded_gpu = _helpers.bounded_gpu
BUDGET = _helpers.BUDGET
OUTPUT_TOL = (3e-5, 6e-6)
KV_TOL = (2e-5, 3e-6)


def create_cpu_fixture(path, *, packed=False, positions="implicit", tokens=4,
                       model=None, layer=3, timeout=90):
    command = [sys.executable, str(Path(__file__).resolve()), "--write-cpu-fixture", str(path),
               "--positions", positions, "--tokens", str(tokens), "--layer", str(layer)]
    if packed:
        command.append("--packed")
    if model is not None:
        command += ["--model", str(model)]
    env = dict(os.environ, MINI_LLM_BACKEND="numpy", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Attention CPU export failed ({result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return read_npz(path)


def weight_state_budget(index, spec, layer, batch, capacity, tokens):
    """Use headers only; account for packed residency, KV forks and workspaces."""
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS
    headers, weights = [], 0
    for suffix, shape in spec.tensor_shapes().items():
        tensor = index.tensors[f"blk.{layer}." + suffix]
        kind = tensor.tensor_type.name
        elements = int(np.prod(shape))
        size = (elements // BLOCK_FORMATS[kind][0] * BLOCK_FORMATS[kind][1]
                if kind in BLOCK_FORMATS else elements * 4)
        weights += size
        headers.append(dict(name=tensor.name, shape=shape, kind=kind, device_bytes=size))
    state = batch * spec.n_kv_heads * capacity * (spec.key_dim + spec.value_dim) * 4
    activation = batch * max(tokens, 2) * (2 * spec.d_model +
                 10 * spec.n_q_heads * spec.key_dim + 4 * spec.n_kv_heads * spec.key_dim) * 4
    scores = 8 * batch * spec.n_q_heads * max(tokens, 2) * capacity * 4
    bound = weights + 6 * state + activation + scores + 4 * 1024**2
    if bound >= BUDGET:
        raise MemoryError(f"Attention header-based GPU bound {bound} >= {BUDGET}")
    return dict(weight_bytes=weights, state_bytes=state, conservative_bytes=bound,
                headers=headers, note="CUDA context/library overhead is not pool storage")


def _write_cpu_fixture(args):
    sys.path.insert(0, str(ROOT))
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.external.qwen3next_attention import (Qwen3NextAttentionInference,
                                                       Qwen3NextAttentionSpec)
    if BACKEND_NAME != "numpy":
        raise RuntimeError("CPU exporter requires an isolated NumPy backend")
    arrays = {}
    functions = None
    if args.model is None:
        functions = runpy.run_path(str(ROOT / "tests/test_0106n_qwen3next_attention.py"))
        index, weights, reads = functions["_fixture"](packed=args.packed)
        layer, batch = 3, 2
    else:
        from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
        index = Qwen3NextGGUFIndex.open(args.model)
        layer, batch = args.layer, 1
    spec = Qwen3NextAttentionSpec.from_index(index)
    capacity = 2 + args.tokens + 2
    manifest = dict(case="model" if args.model else "packed" if args.packed else "dense",
                    model=str(args.model.resolve()) if args.model else None,
                    layer=layer, dtype="float32", batch=batch, prefix_tokens=2,
                    tokens=args.tokens, capacity=capacity, position_mode=args.positions,
                    seed=10615, spec=asdict(spec),
                    cpu_packed_backend="native" if args.model else "reference")
    manifest["budget"] = weight_state_budget(index, spec, layer, batch, capacity, args.tokens)
    if functions is not None:
        requested = {f"blk.{layer}." + suffix for suffix in spec.tensor_shapes()}
        # Dummy expert tensors have data=None. Serialize headers and an empty
        # sentinel WITHOUT touching tensor.data, including its observable getter.
        arrays["unused_payload"] = np.empty(0, np.uint8)
        tensors = []
        for n, tensor in enumerate(index.reader.tensors):
            payload = "unused_payload"
            if tensor.name in requested:
                payload = f"payload_{n}"
                arrays[payload] = np.asarray(tensor.data).copy()
            tensors.append(dict(name=tensor.name, shape=list(tensor.shape),
                                kind=tensor.tensor_type.name, payload=payload))
        manifest["index"] = dict(fields={name: field.contents(0)
                                  for name, field in index.reader.fields.items()}, tensors=tensors)
    op = Qwen3NextAttentionInference.from_index(index, layer, manifest["cpu_packed_backend"])
    rng = np.random.default_rng(manifest["seed"])
    arrays["prefix"] = rng.normal(0, .4, (batch, 2, spec.d_model)).astype(np.float32)
    arrays["hidden"] = rng.normal(0, .4, (batch, args.tokens, spec.d_model)).astype(np.float32)
    prefix_positions = positions = None
    if args.positions != "implicit":
        # Nonmonotonic, batch-dependent positions test chronology independently
        # from rotary IDs, with small angles within the existing FP64 tolerance.
        prefix_positions = np.array([[9, 1], [3, 12]], np.int64)[:batch]
        patterns = np.array([[19, 2, 7, 0], [4, 11, 0, 8]], np.int64)[:batch]
        positions = np.tile(patterns, (1, (args.tokens + 3)//4))[:, :args.tokens].copy()
        arrays.update(prefix_positions=prefix_positions, positions=positions)
    state = op.create_state(batch, capacity)
    begin = time.perf_counter()
    arrays["cpu_prefix_output"] = op.forward(arrays["prefix"], state, position_ids=prefix_positions)
    manifest["cpu_prefix_wall_ms"] = (time.perf_counter() - begin)*1000
    for name in ("k_cache", "v_cache"):
        arrays["cpu_prefix_" + name] = getattr(state, name).copy()
    begin = time.perf_counter()
    arrays["cpu_output"] = op.forward(arrays["hidden"], state, position_ids=positions)
    manifest["cpu_full_wall_ms"] = (time.perf_counter() - begin)*1000
    for name in ("k_cache", "v_cache"):
        arrays["cpu_" + name] = getattr(state, name).copy()
    state.reset()  # GenerationState deliberately retains backing cache contents.
    arrays["cpu_reset_output"] = op.forward(arrays["hidden"], state, position_ids=positions)
    for name in ("k_cache", "v_cache"):
        arrays["cpu_reset_" + name] = getattr(state, name).copy()
    if functions is not None:
        arrays["ref_prefix_output"], ref_k, ref_v = functions["_reference"](
            arrays["prefix"], weights, spec, positions=prefix_positions)
        arrays.update(ref_prefix_k_cache=ref_k, ref_prefix_v_cache=ref_v)
        arrays["ref_output"], arrays["ref_k_cache"], arrays["ref_v_cache"] = functions["_reference"](
            arrays["hidden"], weights, spec, prefix=(ref_k, ref_v), positions=positions)
        arrays["ref_reset_output"], arrays["ref_reset_k_cache"], arrays["ref_reset_v_cache"] = functions["_reference"](
            arrays["hidden"], weights, spec, positions=positions)
        if not set(reads) <= requested:
            raise AssertionError(f"Unrequested fixture payloads read: {set(reads) - requested}")
        manifest["payload_reads"] = sorted(set(reads))
    else:
        manifest["reference_note"] = "Native CPU thin attention branch; no independent whole-checkpoint FP64 oracle"
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez(args.write_cpu_fixture, **arrays)


def validate_gpu(manifest, arrays, output):
    """Fail fast with saved stage/value diagnostics; never edit production math."""
    sys.path.insert(0, str(ROOT))
    output = Path(output)
    report = dict(patch="0106N-attention", passed=False, backend="cupy",
                  manifest={key: value for key, value in manifest.items() if key != "index"},
                  budget_bytes=BUDGET, comparisons={}, timings={},
                  sources=["mini_llm/external/qwen3next_attention.py",
                           "mini_llm/external/gguf_projection.py",
                           "mini_llm/ops/projected_attention_inference.py",
                           "mini_llm/ops/rotary_inference.py"],
                  timing_warning="CUDA event and synchronized wall timings are correctness smoke only, not throughput")
    stage = "setup"
    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def require(condition, message):
        if not condition:
            raise AssertionError(f"{stage}: {message}")
    def check(name, actual, expected, tolerance=OUTPUT_TOL):
        import cupy as cp
        host = cp.asnumpy(actual) if isinstance(actual, cp.ndarray) else np.asarray(actual)
        metric = numerical_metrics(host, expected, rtol=tolerance[0], atol=tolerance[1], path=name)
        error = np.abs(host.astype(np.float64) - expected.astype(np.float64)) if host.shape == expected.shape else np.array([])
        if error.size:
            relative = error / np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
            max_relative = float(relative.max())
            metric["max_relative_error"] = max_relative if np.isfinite(max_relative) else str(max_relative)
        report["comparisons"][name] = metric
        if not metric["passed"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output.with_suffix(".failure.npz"), actual=host, expected=expected)
            save()
            raise AssertionError(json.dumps(metric))
    def verify_state(name, state, cpu_prefix, ref_prefix, length):
        require(state.length == length, f"length {state.length} != {length}")
        for cache in ("k_cache", "v_cache"):
            value = getattr(state, cache)
            check(f"{name}/cpu/{cache}", value, arrays[cpu_prefix + cache], KV_TOL)
            if ref_prefix + cache in arrays:
                check(f"{name}/fp64/{cache}", value[:, :, :length], arrays[ref_prefix + cache], KV_TOL)
    try:
        from mini_llm.backend import BACKEND_NAME
        require(BACKEND_NAME == "cupy", "Use MINI_LLM_BACKEND=cupy")
        from mini_llm.external.qwen3next_attention import (Qwen3NextAttentionInference,
                                                           Qwen3NextAttentionSpec)
        from mini_llm.ops.rope import prepare_rope_positions
        with bounded_gpu() as (cp, pool):
            properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = properties["name"]
            report["device"] = name.decode() if isinstance(name, bytes) else name
            stream = cp.cuda.Stream(non_blocking=True)
            report["non_default_stream"] = bool(stream.ptr)
            with stream:
                require(cp.cuda.get_current_stream().ptr == stream.ptr, "Non-default stream is not current")
                index = index_from_manifest(manifest, arrays)
                spec = Qwen3NextAttentionSpec.from_index(index)
                report["budget_estimate"] = weight_state_budget(index, spec, manifest["layer"],
                    manifest["batch"], manifest["capacity"], manifest["tokens"])
                op = Qwen3NextAttentionInference.from_index(index, manifest["layer"])
                prefix, hidden = cp.asarray(arrays["prefix"]), cp.asarray(arrays["hidden"])
                def position_slice(key, a=0, b=None):
                    if key not in arrays:
                        return None
                    positions = cp.asarray(arrays[key][:, a:b])
                    if manifest["position_mode"] == "prepared":
                        return prepare_rope_positions(positions, d_head=spec.rotary_dim,
                                                      base=spec.rope_base, dtype="float32")
                    return positions
                def timed(name, function):
                    stream.synchronize()
                    begin, end = cp.cuda.Event(), cp.cuda.Event()
                    wall = time.perf_counter()
                    begin.record(stream)
                    result = function()
                    end.record(stream)
                    end.synchronize()
                    report["timings"][name] = dict(cuda_event_ms=float(cp.cuda.get_elapsed_time(begin, end)),
                                                     wall_ms=(time.perf_counter() - wall)*1000)
                    return result
                initial = op.create_state(manifest["batch"], manifest["capacity"])
                stage = "prefix/from_zero"
                prefix_output = timed("prefix_cold", lambda: op.forward(prefix, initial,
                                                     position_ids=position_slice("prefix_positions")))
                check("prefix/cpu/output", prefix_output, arrays["cpu_prefix_output"])
                if "ref_prefix_output" in arrays:
                    check("prefix/fp64/output", prefix_output, arrays["ref_prefix_output"])
                    check("cpu/prefix/fp64/output", arrays["cpu_prefix_output"], arrays["ref_prefix_output"])
                verify_state("prefix", initial, "cpu_prefix_", "ref_prefix_", 2)
                for mode in ("full", "chunks", "steps"):
                    stage = mode
                    state = op.attention.copy_state(initial)
                    pointers = (state.k_cache.data.ptr, state.v_cache.data.ptr)
                    require(all(a != b for a, b in zip(pointers, (initial.k_cache.data.ptr, initial.v_cache.data.ptr))),
                            "copy_state aliased initial KV storage")
                    # Explicit empty forward is a no-op even with prepared factors.
                    empty = op.forward(hidden[:, :0], state, position_ids=position_slice("positions", 0, 0))
                    require(isinstance(empty, cp.ndarray) and empty.shape == hidden[:, :0].shape
                            and empty.dtype == cp.float32, "Invalid empty output")
                    verify_state(mode + "/empty", state, "cpu_prefix_", "ref_prefix_", 2)
                    positions = position_slice("positions")
                    n = manifest["tokens"]
                    if mode == "full":
                        run = lambda: op.forward(hidden, state, position_ids=positions)
                    elif mode == "chunks":
                        def run():
                            edges = (0, min(2, n), min(2, n), n)
                            return cp.concatenate([op.forward(hidden[:, a:b], state,
                                position_ids=position_slice("positions", a, b))
                                for a, b in zip(edges, edges[1:])], axis=1)
                    else:
                        def run():
                            return cp.stack([op.step(hidden[:, t], state,
                                position_ids=position_slice("positions", t, t + 1)) for t in range(n)], axis=1)
                    actual = timed(mode, run)
                    require(isinstance(actual, cp.ndarray) and actual.dtype == cp.float32,
                            "Output is not a real FP32 CuPy array")
                    check(f"{mode}/cpu/output", actual, arrays["cpu_output"])
                    if "ref_output" in arrays:
                        check(f"{mode}/fp64/output", actual, arrays["ref_output"])
                        check("cpu/fp64/output", arrays["cpu_output"], arrays["ref_output"])
                    verify_state(mode, state, "cpu_", "ref_", 2 + n)
                    require(pointers == (state.k_cache.data.ptr, state.v_cache.data.ptr), "Forward replaced KV storage")
                    fork = op.attention.copy_state(state)
                    require(fork.k_cache.data.ptr != state.k_cache.data.ptr and
                            fork.v_cache.data.ptr != state.v_cache.data.ptr, "Fork aliased KV storage")
                    stage = mode + "/reset"
                    state.reset()
                    require(state.length == 0 and pointers == (state.k_cache.data.ptr, state.v_cache.data.ptr),
                            "Reset length/storage contract")
                    # Reset does NOT zero GenerationState caches. Old contents
                    # must remain intact but must not enter subsequent attention.
                    for cache in ("k_cache", "v_cache"):
                        check(f"{mode}/reset_retained/{cache}", getattr(state, cache), cp.asnumpy(getattr(fork, cache)), (0, 0))
                    reset_output = op.forward(hidden, state, position_ids=positions)
                    check(f"{mode}/reset/cpu/output", reset_output, arrays["cpu_reset_output"])
                    if "ref_reset_output" in arrays:
                        check(f"{mode}/reset/fp64/output", reset_output, arrays["ref_reset_output"])
                    verify_state(mode + "/reset", state, "cpu_reset_", "ref_reset_", n)
                    verify_state(mode + "/fork", fork, "cpu_", "ref_", 2 + n)
                    verify_state(mode + "/initial_preserved", initial, "cpu_prefix_", "ref_prefix_", 2)
                    del state, fork
                check("inputs/prefix_unchanged", prefix, arrays["prefix"], (0, 0))
                check("inputs/hidden_unchanged", hidden, arrays["hidden"], (0, 0))
                stream.synchronize()
                report["pool_peak_reserved_bytes"] = int(pool.total_bytes())
                report["pool_limit_bytes"] = int(pool.get_limit())
                require(pool.total_bytes() < BUDGET, "Allocation pool exceeded budget")
            report["passed"] = True
    except Exception as exc:
        report["affected_stage"] = stage
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-cpu-fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--packed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--positions", choices=("implicit", "explicit", "prepared"),
                        default="implicit", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.model and (args.layer != 3 or args.tokens != 4):
        parser.error("Actual-model smoke is restricted to --layer 3 --tokens 4")
    if args.write_cpu_fixture:
        _write_cpu_fixture(args)
        return
    if args.output is None:
        parser.error("--output is required")
    if os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "cupy":
        parser.error("Use MINI_LLM_BACKEND=cupy; import and --help remain CPU-safe")
    result = dict(patch="0106N-attention", passed=False, runs=[])
    try:
        with tempfile.TemporaryDirectory(prefix="0106n-attention-") as directory:
            cases = [(False, "implicit")] if args.model else [
                (packed, positions) for packed in (False, True)
                for positions in ("implicit", "explicit", "prepared")]
            for packed, positions in cases:
                case = ("model" if args.model else "packed" if packed else "dense") + "-" + positions
                manifest, arrays = create_cpu_fixture(Path(directory) / (case + ".npz"),
                    packed=packed, positions=positions, tokens=args.tokens, model=args.model, layer=args.layer)
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
