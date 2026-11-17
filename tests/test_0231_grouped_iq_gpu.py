"""Grouped IQ: exact FP32 graph, three launches, stream/stride/slot contracts.

Opt-in bounded real top10 timing:
MINI_LLM_BACKEND=cupy MINI_LLM_GROUPED_IQ_REAL_BENCHMARK=1
MINI_LLM_GROUPED_IQ_REPORT=docs/qwen4exp/0231_grouped_iq_gpu_v1.json
.venv/bin/python -m pytest -q -s tests/test_0231_grouped_iq_gpu.py
"""
import itertools
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import iq_packed_gpu as iq
from mini_llm.ops import grouped_quantized_experts as grouped
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0228_iq_gpu_integration import IQ, packed, forbidden


ACTUAL = [("IQ3_XXS", "IQ3_XXS", "IQ4_NL"), ("IQ2_S", "IQ2_S", "Q2_0"),
          ("IQ2_S", "IQ2_S", "IQ4_NL"), ("IQ3_XXS", "IQ3_XXS", "Q2_0"),
          ("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q2_0"),
          ("IQ4_XS", "IQ4_XS", "IQ4_NL")]


@pytest.fixture(scope="module")
def cp():
    if BACKEND_NAME != "cupy":
        pytest.skip("requires MINI_LLM_BACKEND=cupy")
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def handles(formats, count=3, d=256, f=256):
    hosts, devices = [], []
    for e in range(count):
        weights = tuple(packed(fmt, n, k, 2310 + e * 3 + p)
                        for p, (fmt, n, k) in enumerate(zip(formats, (f, f, d), (d, d, f))))
        hosts.append(weights)
        devices.append(PackedExpertFFNInference(*(packed_to_device(w) for w in weights)))
    return hosts, devices


def sequential(cp, x, experts, routes, slots=None):
    out = cp.zeros_like(x)
    for e, handle in enumerate(experts):
        slot = e if slots is None else slots[e]
        # Keep the multiplication and ordered addition as distinct operations.
        term = handle.forward(x) * routes[slot]
        out += term
    return out


def dense_oracle(x, weights, routes):
    out = np.zeros_like(x)
    for ws, route in zip(weights, routes):
        g, u, down = [w.dequantize_reference().astype(np.float64) for w in ws]
        ag, au = x.astype(np.float64) @ g.T, x.astype(np.float64) @ u.T
        h = ((ag / (1 + np.exp(-ag))) * au).astype(np.float32)
        y = (h.astype(np.float64) @ down.T).astype(np.float32)
        out += np.multiply(y, route, dtype=np.float32)
    return out


@pytest.mark.parametrize("formats", list(itertools.product(IQ, repeat=3)))
def test_all_iq_projection_triples_exact_sequential(cp, formats, monkeypatch):
    _, experts = handles(formats, count=2)
    x = cp.asarray(np.random.default_rng(231).normal(0, .1, (1, 256)).astype(np.float32))
    routes = cp.asarray(np.array([.625, -.375], np.float32))
    expected = sequential(cp, x, experts, routes)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    assert grouped.supports_grouped_quantized_experts(x, experts, routes)
    got = grouped.grouped_quantized_experts_forward(x, experts, routes)
    assert got.shape == (1, 256) and got.dtype == cp.float32
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))


@pytest.mark.parametrize("formats", ACTUAL)
@pytest.mark.parametrize("block_k", [256, 512, 1024])
def test_top10_cpu_oracle_and_tail_tiles(cp, formats, block_k):
    hosts, experts = handles(formats, count=10, d=768, f=256)
    xh = np.random.default_rng(232).normal(0, .1, (1, 768)).astype(np.float32)
    rh = np.linspace(-.2, .5, 10, dtype=np.float32)
    expected = dense_oracle(xh, hosts, rh)
    x, routes = cp.asarray(xh), cp.asarray(rh)
    got = grouped.grouped_quantized_experts_forward(x, experts, routes, block_k=block_k)
    reference = cp.zeros_like(x)
    # Use the same requested tile size in the independent verified kernels.
    for handle, route in zip(experts, routes):
        hidden = iq.swiglu(x, handle.gate, handle.up, block_k=block_k)
        term = iq.linear(hidden, handle.down, block_k=block_k) * route
        reference += term
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(reference))
    # FP64 projections and FP32 projection/reduction trees differ by roundoff;
    # cancellation in the ordered expert sum makes pointwise relative error
    # ill-conditioned near zero. Keep exact FP32 graph parity above and bound
    # both normwise error and absolute error relative to the output scale.
    delta = cp.asnumpy(got).astype(np.float64) - expected.astype(np.float64)
    assert np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30) < 2e-6
    assert np.abs(delta).max() <= 32 * np.finfo(np.float32).eps * max(float(np.abs(expected).max()), 1e-30)


@pytest.mark.parametrize("route_kind", ["contiguous", "sliced", "reversed", "broadcast"])
@pytest.mark.parametrize("slots", [None, (2, 0, 1), (1, 1, 0)])
def test_strides_slots_duplicates_and_current_stream(cp, route_kind, slots):
    _, experts = handles(("IQ2_S", "IQ3_S", "IQ4_XS"), count=2)
    experts = [experts[0], experts[1], experts[0]]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x = cp.full((1, 256), .125, cp.float32)
        routes = cp.asarray(np.array([.75, -.25, .5], np.float32))
        if route_kind == "sliced":
            routes = cp.asarray(np.array([.75, 99, -.25, 99, .5, 99], np.float32))[::2]
        elif route_kind == "reversed":
            routes = routes[::-1]
        elif route_kind == "broadcast":
            routes = cp.broadcast_to(routes[:1], (3,))
        expected = sequential(cp, x, experts, routes, slots)
        got = grouped.grouped_quantized_experts_forward(x, experts, routes, routing_slots=slots)
        consumed = got * cp.float32(1.5)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(consumed), cp.asnumpy(expected) * np.float32(1.5))


def test_ordered_rounding_not_tree_reduction(cp):
    _, experts = handles(("IQ3_S", "IQ3_XXS", "IQ4_NL"), count=1)
    experts *= 3
    x = cp.full((1, 256), .25, cp.float32)
    routes = cp.asarray(np.array([2**25, -(2**25), 1], np.float32))
    expected = sequential(cp, x, experts, routes)
    alternative = sequential(cp, x, experts, routes, (0, 2, 1))
    assert np.any(cp.asnumpy(expected) != cp.asnumpy(alternative))
    got = grouped.grouped_quantized_experts_forward(x, experts, routes)
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))


def test_exactly_three_launches_bounded_scratch_table_and_no_decode(cp, monkeypatch):
    _, experts = handles(ACTUAL[0], count=3, d=768, f=256)
    x, routes = cp.ones((1, 768), cp.float32), cp.ones((3,), cp.float32)
    expected = sequential(cp, x, experts, routes)
    module = iq._grouped_module(x.device.id)
    calls, allocations, transfers = [], [], []
    class Module:
        def get_function(self, name):
            kernel = module.get_function(name)
            def launch(grid, block, args):
                calls.append((name, grid, block))
                return kernel(grid, block, args)
            return launch
    empty, asarray = cp.empty, cp.asarray
    def tracked_empty(shape, dtype):
        allocations.append((shape, np.dtype(dtype)))
        return empty(shape, dtype)
    def tracked_asarray(value, *args, **kwargs):
        transfers.append((value.shape, value.dtype))
        return asarray(value, *args, **kwargs)
    monkeypatch.setattr(iq, "_grouped_module", lambda device: Module())
    monkeypatch.setattr(cp, "empty", tracked_empty)
    monkeypatch.setattr(cp, "asarray", tracked_asarray)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    monkeypatch.setattr(grouped, "_launch", forbidden)
    assert grouped.supports_grouped_quantized_experts(x, experts, routes)
    assert not calls and not allocations and not transfers
    got = grouped.grouped_quantized_experts_forward(x, experts, routes)
    assert transfers == [((3, 3), np.dtype("uint64"))]
    assert allocations == [((3, 256), np.dtype("float32")),
                           ((3, 768), np.dtype("float32")), ((1, 768), np.dtype("float32"))]
    assert calls == [("iq_grouped_gate_up", (3 * 256,), (256,)),
                     ("iq_grouped_down_weighted", (3 * 768,), (256,)),
                     ("iq_grouped_ordered_reduce", (3,), (256,))]
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))


@pytest.mark.parametrize("case", ["hybrid", "heterogeneous", "batch", "precision", "slots", "tile"])
def test_unsupported_metadata_only(cp, case, monkeypatch):
    _, experts = handles(ACTUAL[0], count=2)
    x, routes = cp.ones((1, 256), cp.float32), cp.ones((2,), cp.float32)
    kwargs = {}
    if case == "hybrid":
        down = packed_to_device(packed("Q8_0", 256, 256))
        experts = [PackedExpertFFNInference(h.gate, h.up, down) for h in experts]
    elif case == "heterogeneous":
        experts[1] = handles(ACTUAL[1], count=1)[1][0]
    elif case == "batch":
        x = cp.ones((2, 256), cp.float32)
    elif case == "precision":
        x = x.astype(cp.float16)
    elif case == "slots":
        kwargs["routing_slots"] = (0, 2)
    elif case == "tile":
        kwargs["block_k"] = 128
    monkeypatch.setattr(cp, "asarray", forbidden)
    monkeypatch.setattr(cp, "empty", forbidden)
    assert not grouped.supports_grouped_quantized_experts(x, experts, routes, **kwargs)
    with pytest.raises(ValueError, match="Unsupported grouped"):
        grouped.grouped_quantized_experts_forward(x, experts, routes, **kwargs)


def test_iq_grouped_does_not_require_triton(cp, monkeypatch):
    from mini_llm.ops import quantized_linear as ql
    _, experts = handles(ACTUAL[1], count=1)
    x, routes = cp.ones((1, 256), cp.float32), cp.ones((1,), cp.float32)
    expected = sequential(cp, x, experts, routes)
    monkeypatch.setattr(ql, "triton", None)
    assert grouped.supports_grouped_quantized_experts(x, experts, routes)
    np.testing.assert_array_equal(cp.asnumpy(iq.grouped(x, experts, routes)), cp.asnumpy(expected))


def _measure(cp, call, repeats=20):
    call()
    cp.cuda.get_current_stream().synchronize()
    events, wall = [], []
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        t = time.perf_counter()
        start.record()
        call()
        stop.record()
        stop.synchronize()
        events.append(cp.cuda.get_elapsed_time(start, stop))
        wall.append((time.perf_counter() - t) * 1000)
    return {"event_ms": float(np.median(events)), "synchronized_wall_ms": float(np.median(wall)),
            "repeats": repeats}


@pytest.mark.skipif(os.environ.get("MINI_LLM_GROUPED_IQ_REAL_BENCHMARK") != "1",
                    reason="opt-in bounded real packed top10 benchmark")
def test_real_top10_microbenchmark(cp):
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    model = Path("/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S")
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    index = Qwen4ExpGGUFIndex.open(model, model_id="0231")
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
    records, seen = [], set()
    def guard():
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - baseline < 2 * 1024**3
    try:
        x = cp.asarray(np.random.default_rng(231).normal(0, .1, (1, index.d_model)).astype(np.float32))
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
                assert sum(w.nbytes for w in weights) < 4 * 1024**2
                experts.append(PackedExpertFFNInference(*(packed_to_device(w) for w in weights)))
                guard()
            def seq():
                return sequential(cp, x, experts, routes)
            def group():
                return grouped.grouped_quantized_experts_forward(x, experts, routes)
            a, b = cp.asnumpy(seq()), cp.asnumpy(group())
            delta = b.astype(np.float64) - a.astype(np.float64)
            relative = float(np.linalg.norm(delta) / max(np.linalg.norm(a.astype(np.float64)), 1e-30))
            assert relative < 2e-6
            np.testing.assert_array_equal(a, b)
            seq_times, group_times = _measure(cp, seq), _measure(cp, group)
            record = dict(layer=layer, formats=list(signature), expert_ids=ids.tolist(),
                          max_absolute_error=float(np.abs(delta).max()), relative_l2=relative,
                          sequential=seq_times, grouped=group_times,
                          event_speedup=seq_times["event_ms"] / group_times["event_ms"])
            records.append(record)
            print(json.dumps(record), flush=True)
            guard()
        assert len(records) == 7
        assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
        attributes = iq._grouped_module(x.device.id).get_function("iq_grouped_gate_up").attributes
        report = dict(scope="real packed top10 experts, one representative layer per format triple; not full model",
                      input="fixed synthetic FP32, seed231 std0.1", shape=list(x.shape),
                      route_selection="ten evenly spaced IDs, ascending; synthetic normalized routes",
                      gpu=cp.cuda.runtime.getDeviceProperties(x.device.id)["name"].decode(),
                      cupy_version=cp.__version__, records=records, passed=True,
                      grouped_compute_launches=3, sequential_projection_launches=20,
                      timing_notes="warm median, excludes expert upload and JIT; grouped includes pointer-table upload",
                      lookup_tables="unchanged canonical __constant__ tables; no unmeasured storage optimization",
                      codebook_bytes=sum(getattr(iq._books, n).nbytes for n in
                          ("iq2s_grid", "iq3s_grid", "iq3xxs_grid", "ksigns_iq2xs", "kvalues_iq4nl")),
                      gate_kernel_attributes=attributes,
                      additional_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - baseline,
                      checkpoint_size_mtime_unchanged=True)
        output = os.environ.get("MINI_LLM_GROUPED_IQ_REPORT")
        if output:
            path = Path(output)
            assert not path.exists(), "Refuse to overwrite versioned benchmark report"
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        index.store.close()
