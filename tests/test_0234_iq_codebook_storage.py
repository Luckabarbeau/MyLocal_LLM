"""Only codebook storage changes: exact constant/global parity and bounded timing.

Real benchmark (no full model, warm identical resident weights):
MINI_LLM_BACKEND=cupy MINI_LLM_IQ_STORAGE_REAL_BENCHMARK=1
MINI_LLM_IQ_STORAGE_REPORT=docs/qwen4exp/0234_iq_codebook_storage_v1.json
.venv/bin/python -m pytest -q -s tests/test_0234_iq_codebook_storage.py
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import iq_packed_gpu as iq
from mini_llm.ops import grouped_quantized_experts as grouped
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0228_iq_packed_gpu import exhaustive_blocks
from test_0231_grouped_iq_gpu import ACTUAL, IQ, handles, sequential


ENV = "MINI_LLM_IQ_GPU_CODEBOOK_STORAGE"
CACHE_EVIDENCE = {}


@pytest.fixture(scope="module")
def cp():
    if BACKEND_NAME != "cupy":
        pytest.skip("requires MINI_LLM_BACKEND=cupy")
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def bits_equal(cp, left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    dtype = cp.uint32 if left.dtype == cp.float32 else cp.uint16
    np.testing.assert_array_equal(cp.asnumpy(left.view(dtype)), cp.asnumpy(right.view(dtype)))


def test_storage_source_only_changes_five_qualifiers():
    constant = iq._cuda_source("constant")
    global_source = iq._cuda_source("global")
    assert constant.count("__device__ __constant__") == 5
    assert global_source.count("__device__ const ") == 5
    assert global_source.replace("__device__ const ", "__device__ __constant__ ") == constant
    assert iq._cuda_source("constant") is constant
    assert iq._cuda_source("global") is global_source
    with pytest.raises(ValueError, match="storage"):
        iq._cuda_source("GLOBAL")


@pytest.mark.parametrize("mode", ["", "GLOBAL", " constant", "global ", "shared", "0"])
def test_strict_rejection_before_cuda(mode, monkeypatch):
    class NoCUDA:
        def __getattr__(self, name):
            pytest.fail(f"Invalid mode touched CUDA: {name}")
    monkeypatch.setenv(ENV, mode)
    monkeypatch.setattr(iq, "cp", NoCUDA())
    with pytest.raises(ValueError, match="constant or global"):
        iq._module(0, "float32")
    with pytest.raises(ValueError, match="constant or global"):
        iq._grouped_module(0)
    # CPU/unsupported-format metadata must not consult the runtime or mode.
    assert not iq.supports_type("Q8_0")
    assert not iq.supports_type("unsupported")
    assert iq.supports_type("IQ3_S")


def test_cpu_import_and_unsupported_metadata_do_not_initialize_cuda():
    code = '''
import cupy as cp
calls = []
def forbidden(*args, **kwargs):
    calls.append(True)
    raise AssertionError("CUDA runtime initialized by CPU metadata")
cp.cuda.Device = forbidden
cp.RawModule = forbidden
cp.cuda.runtime.getDeviceCount = forbidden
cp.cuda.runtime.getDevice = forbidden
from mini_llm.ops import iq_packed_gpu as iq
assert not iq.supports_type("UNKNOWN")
assert not iq.supports_type("Q8_0")
assert iq.supports_type("IQ2_S")
assert not calls
print("CPU metadata: no CUDA initialization")
'''
    env = dict(os.environ, MINI_LLM_BACKEND="numpy")
    env[ENV] = "intentionally-invalid-but-unused"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "no CUDA initialization" in result.stdout


def test_mode_keyed_caches_and_materialization_count(cp, monkeypatch):
    from cupy.cuda import compiler
    iq._cuda_source.cache_clear()
    iq._cached_module.cache_clear()
    iq._cached_grouped_module.cache_clear()
    original_module, original_compile = cp.RawModule, compiler.compile_using_nvrtc
    objects, nvrtc_calls, materializations = [], [], []
    def raw_module(*args, **kwargs):
        obj = original_module(*args, **kwargs)
        objects.append(obj)
        return obj
    def compile_nvrtc(*args, **kwargs):
        nvrtc_calls.append(True)
        return original_compile(*args, **kwargs)
    monkeypatch.setattr(cp, "RawModule", raw_module)
    monkeypatch.setattr(compiler, "compile_using_nvrtc", compile_nvrtc)
    modules = {}
    for mode in ("constant", "global", "constant", "global"):
        monkeypatch.setenv(ENV, mode)
        for dtype in ("float32", "float16"):
            module = iq._module(cp.cuda.Device().id, dtype)
            key = (mode, dtype)
            if key in modules:
                assert module is modules[key]
            else:
                modules[key] = module
                t = time.perf_counter()
                module.get_function("packed_linear")
                materializations.append(dict(mode=mode, family="single", dtype=dtype,
                    first_get_function_ms=(time.perf_counter() - t) * 1000))
        module = iq._grouped_module(cp.cuda.Device().id)
        key = (mode, "grouped")
        if key in modules:
            assert module is modules[key]
        else:
            modules[key] = module
            t = time.perf_counter()
            module.get_function("iq_grouped_gate_up")
            materializations.append(dict(mode=mode, family="grouped", dtype="float32",
                first_get_function_ms=(time.perf_counter() - t) * 1000))
    assert len(objects) == 6  # device * (two dtypes + grouped) * two modes.
    assert iq._cuda_source.cache_info().misses == 2
    assert iq._cached_module.cache_info().misses == 4
    assert iq._cached_grouped_module.cache_info().misses == 2
    assert modules["constant", "float32"] is not modules["global", "float32"]
    attrs = {mode: modules[mode, "grouped"].get_function("iq_grouped_gate_up").attributes
             for mode in ("constant", "global")}
    assert attrs["constant"]["const_size_bytes"] == 11408
    assert attrs["global"]["const_size_bytes"] == 0
    monkeypatch.delenv(ENV)
    assert iq._codebook_storage() == "constant"
    assert iq._module(cp.cuda.Device().id, "float32") is modules["constant", "float32"]
    CACHE_EVIDENCE.update(module_objects_created=len(objects),
        nvrtc_compile_calls_observed=len(nvrtc_calls), materializations=materializations,
        source_cache=iq._cuda_source.cache_info()._asdict(),
        single_cache=iq._cached_module.cache_info()._asdict(),
        grouped_cache=iq._cached_grouped_module.cache_info()._asdict(),
        kernel_attributes=attrs,
        note="First module materialization includes load/JIT; CuPy disk cache may avoid NVRTC. "
             "Source misses count generated strings, not compilation; module keys include device/dtype/storage.")


@pytest.mark.parametrize("fmt", IQ)
def test_all_lane_encodings_bitexact(cp, fmt, monkeypatch):
    raw = exhaustive_blocks(fmt)
    qk = iq.BLOCK_FORMATS[fmt][0]
    host = PackedGGUFWeight(raw, (len(raw), qk), fmt)
    weight = packed_to_device(host)
    x = cp.eye(qk, dtype=cp.float32)
    monkeypatch.setenv(ENV, "constant")
    reference = iq.linear(x, weight)
    monkeypatch.setenv(ENV, "global")
    got = iq.linear(x, weight)
    bits_equal(cp, got, reference)


@pytest.mark.parametrize("fmt", IQ)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("shape", [(2, 256), (2, 3, 768)])
def test_gemv_and_fused_bitexact_current_stream(cp, fmt, dtype, shape, monkeypatch):
    _, experts = handles((fmt, IQ[(IQ.index(fmt) + 1) % 6], "IQ4_NL"), count=1,
                         d=shape[-1], f=256)
    handle = experts[0]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    outputs = {}
    with stream:
        x = cp.asarray(np.random.default_rng(234).normal(0, .05, shape).astype(dtype))
        for mode in ("constant", "global", "constant"):
            monkeypatch.setenv(ENV, mode)
            linear = iq.linear(x, handle.gate)
            fused = iq.swiglu(x, handle.gate, handle.up)
            outputs[mode] = (linear * cp.float32(.75), fused * cp.float32(.75))
    stream.synchronize()
    for ref, got in zip(outputs["constant"], outputs["global"]):
        bits_equal(cp, got, ref)


@pytest.mark.parametrize("formats", [(fmt, fmt, fmt) for fmt in IQ] + ACTUAL)
@pytest.mark.parametrize("route_kind", ["contiguous", "reversed", "broadcast"])
def test_grouped_bitexact_current_stream_slots(cp, formats, route_kind, monkeypatch):
    _, experts = handles(formats, count=2, d=768, f=256)
    experts = [experts[0], experts[1], experts[0]]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    outputs = {}
    with stream:
        x = cp.full((1, 768), .0625, cp.float32)
        routes = cp.asarray(np.array([.75, -.25, .5], np.float32))
        if route_kind == "reversed":
            routes = routes[::-1]
        elif route_kind == "broadcast":
            routes = cp.broadcast_to(routes[:1], (3,))
        for mode in ("constant", "global", "constant"):
            monkeypatch.setenv(ENV, mode)
            outputs[mode] = grouped.grouped_quantized_experts_forward(
                x, experts, routes, routing_slots=(2, 0, 1)) + cp.float32(.25)
    stream.synchronize()
    bits_equal(cp, outputs["global"], outputs["constant"])


def test_no_per_call_codebook_transfer(cp, monkeypatch):
    _, experts = handles(ACTUAL[0], count=2)
    x, routes = cp.ones((1, 256), cp.float32), cp.ones((2,), cp.float32)
    original_asarray = cp.asarray
    transfers = []
    def transfer(value, *args, **kwargs):
        transfers.append((value.shape, value.dtype))
        return original_asarray(value, *args, **kwargs)
    monkeypatch.setattr(cp, "asarray", transfer)
    for mode in ("constant", "global", "constant", "global"):
        monkeypatch.setenv(ENV, mode)
        iq.linear(x, experts[0].gate)
        iq.swiglu(x, experts[0].gate, experts[0].up)
        iq.grouped(x, experts, routes)
    assert transfers == [((2, 3), np.dtype("uint64"))] * 4


def measured_call(cp, call):
    first, last = cp.cuda.Event(), cp.cuda.Event()
    start = time.perf_counter()
    first.record()
    call()
    last.record()
    last.synchronize()
    return cp.cuda.get_elapsed_time(first, last), (time.perf_counter() - start) * 1000


@pytest.mark.skipif(os.environ.get("MINI_LLM_IQ_STORAGE_REAL_BENCHMARK") != "1",
                    reason="opt-in bounded real storage benchmark")
def test_real_seven_triples_top10_storage_benchmark(cp, monkeypatch):
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    model = Path("/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S")
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    index = Qwen4ExpGGUFIndex.open(model, model_id="0234")
    original_rows = index.store.rows
    logical_read_bytes = 0
    max_read_bytes = 0
    def bounded_rows(name, first, last, *, outer=()):
        nonlocal logical_read_bytes, max_read_bytes
        size = (last - first) * index.store.tensors[name].row_bytes
        assert len(outer) == 1 and size < 4 * 1024**2
        logical_read_bytes += size
        max_read_bytes = max(max_read_bytes, size)
        return original_rows(name, first, last, outer=outer)
    monkeypatch.setattr(index.store, "rows", bounded_rows)
    def forbidden(*args, **kwargs):
        pytest.fail("No dense expert decode or expert-bank access")
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    monkeypatch.setattr(index.store, "packed_weight", forbidden)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
    records, seen = [], set()
    def guard():
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - baseline < 2 * 1024**3
    try:
        guard()
        x = cp.asarray(np.random.default_rng(234).normal(0, .1, (1, index.d_model)).astype(np.float32))
        ids = np.linspace(0, index.n_experts - 1, 10, dtype=int)
        routes = cp.asarray(np.arange(1, 11, dtype=np.float32) / np.float32(55))
        for layer in range(index.n_layers):
            signature = tuple(index.descriptors[f"blk.{layer}.ffn_{p}_exps.weight"].quant_type
                              for p in ("gate", "up", "down"))
            if signature in seen:
                continue
            seen.add(signature)
            experts = []
            for e in ids:
                weights = index.provider(ExpertKey(index.model_id, layer, int(e)))
                experts.append(PackedExpertFFNInference(*(packed_to_device(w) for w in weights)))
                guard()
            calls = {"sequential": lambda: sequential(cp, x, experts, routes),
                     "grouped": lambda: grouped.grouped_quantized_experts_forward(x, experts, routes)}
            reference = None
            # Materialize both storage modules and warm all kernels before timing.
            for mode in ("constant", "global"):
                monkeypatch.setenv(ENV, mode)
                for call in calls.values():
                    result = call()
                    if reference is None:
                        reference = result
                    else:
                        bits_equal(cp, result, reference)
                cp.cuda.get_current_stream().synchronize()
            samples = {kind: {mode: [] for mode in ("constant", "global")} for kind in calls}
            # Balanced ABBA/BAAB blocks avoid always favoring the later mode.
            for round_id in range(10):
                modes = ("constant", "global", "global", "constant") if round_id % 2 == 0 else (
                    "global", "constant", "constant", "global")
                for mode in modes:
                    monkeypatch.setenv(ENV, mode)
                    for kind in (tuple(calls) if round_id % 2 == 0 else tuple(reversed(calls))):
                        samples[kind][mode].append(measured_call(cp, calls[kind]))
            measurements = {}
            for kind, modes in samples.items():
                measurements[kind] = {}
                for mode, values in modes.items():
                    array = np.asarray(values)
                    measurements[kind][mode] = dict(event_ms=float(np.median(array[:, 0])),
                        event_p10_ms=float(np.percentile(array[:, 0], 10)),
                        event_p90_ms=float(np.percentile(array[:, 0], 90)),
                        synchronized_wall_ms=float(np.median(array[:, 1])), samples=len(values))
                measurements[kind]["global_event_speedup"] = (
                    measurements[kind]["constant"]["event_ms"] / measurements[kind]["global"]["event_ms"])
                measurements[kind]["global_wall_speedup"] = (
                    measurements[kind]["constant"]["synchronized_wall_ms"] /
                    measurements[kind]["global"]["synchronized_wall_ms"])
            record = dict(layer=layer, formats=list(signature), expert_ids=ids.tolist(),
                          bit_exact=True, max_absolute_error=0.0, relative_l2=0.0,
                          measurements=measurements)
            records.append(record)
            print(json.dumps(record), flush=True)
            guard()
        assert len(records) == 7
        assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
        ratios = [r["measurements"]["grouped"]["global_event_speedup"] for r in records]
        wall_ratios = [r["measurements"]["grouped"]["global_wall_speedup"] for r in records]
        significant = float(np.median(ratios)) >= 1.10 and min(ratios) >= 1.0 and min(wall_ratios) >= 1.0
        report = dict(scope="codebook storage only; real packed top10, one layer per actual format triple; no model forward",
            model_path=str(model), input="identical synthetic FP32, seed234 std0.1", shape=list(x.shape),
            routing="ten evenly spaced expert IDs in ascending order; synthetic normalized routes",
            gpu=cp.cuda.runtime.getDeviceProperties(x.device.id)["name"].decode(),
            cupy_version=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            records=records, passed=True, default_storage="constant", cache_evidence=CACHE_EVIDENCE,
            cache_at_end=dict(source=iq._cuda_source.cache_info()._asdict(),
                single=iq._cached_module.cache_info()._asdict(), grouped=iq._cached_grouped_module.cache_info()._asdict()),
            significant_improvement=significant, significance_rule="median grouped CUDA-event speedup >=1.10x; "
                "no triple regresses in median event or synchronized wall timing",
            recommendation=("Recommend opt-in global for main benchmark storage-only A/B; leave default constant pending broader evidence"
                if significant else "Reject significant-cost-improvement hypothesis for this workload; keep global opt-in, default constant"),
            interpretation="Storage-only timing can support the serialization hypothesis but does not establish a hardware cause without profiling",
            timing_notes="20 warm samples/mode/path in balanced ABBA/BAAB order; excludes weight upload and JIT; "
                "includes grouped pointer-table upload; event interval includes host launch gaps; synchronized wall reported separately",
            logical_mmap_read_bytes=logical_read_bytes, max_single_mmap_read_bytes=max_read_bytes,
            additional_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - baseline,
            checkpoint_size_mtime_unchanged=True)
        output = os.environ.get("MINI_LLM_IQ_STORAGE_REPORT")
        if output:
            path = Path(output)
            assert not path.exists(), "Refuse to overwrite versioned report"
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        index.store.close()
