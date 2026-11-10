"""Observational invariants for 0106I; no placement or arithmetic changes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.routing_analysis import analyze_trace
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference


def profiler(mode="detailed", **kwargs):
    return RoutingTrace(model_id="fixture", quantization="F32", expert_mapping="identity-v1",
                        workload="synthetic", mode=mode, **kwargs)


def run_layer(trace=None, workers=0, calls=1):
    rng = np.random.default_rng(10609)
    d, f = 8, 12
    bank = {ExpertKey("fixture", 2, i): tuple(
        rng.normal(0, .1, size=s).astype(np.float32)
        for s in ((d, f), (d, f), (f, d))) for i in range(3)}
    executor = HeterogeneousExpertExecutor(lambda key: bank[key], model_id="fixture",
                                         d_model=d, d_ff=f, cpu_workers=workers,
                                         routing_profiler=trace)
    layer = ExpertsInference(d, f, 3, expert_executor=executor, layer_id=2)
    x = xp.asarray(rng.normal(0, .1, size=(1, 2, d)).astype(np.float32))
    ids = xp.asarray([[[2, 0], [1, 2]]], dtype=xp.int32)
    mix = xp.asarray([[[.7, .3], [.6, .4]]], dtype=xp.float32)
    try:
        results = [asnumpy(layer.forward(x, mix, ids)) for _ in range(calls)]
        # A direct call after a scoped layer must not inherit its dispatch id.
        if trace is not None and trace.mode == "detailed":
            executor.forward(2, 0, x.reshape(-1, d))
        return results
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ["summary", "detailed"])
@pytest.mark.parametrize("workers", [0, 2])
def test_trace_is_observational_with_correct_assignments(mode, workers):
    trace = profiler(mode)
    expected = run_layer(workers=workers, calls=2)
    actual = run_layer(trace, workers, calls=2)
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first, second)
    snapshot = trace.snapshot()
    assert snapshot["tokens_by_layer"] == {"2": 4}
    assert snapshot["frequencies"] == [
        dict(layer_id=2, expert_id=0, selections=2),
        dict(layer_id=2, expert_id=1, selections=2),
        dict(layer_id=2, expert_id=2, selections=4)]
    assert snapshot["counters"]["routed_assignments"] == 8
    assert snapshot["counters"]["cpu_experts"] == (7 if mode == "detailed" else 6)
    assert snapshot["counters"]["cache_requests"] == 0
    assert snapshot["counters"]["transferred_bytes"] == 0
    assert snapshot["timings_ms"]["cpu_compute_host_ms"] > 0
    if mode == "detailed":
        assert [r["token_start"] for r in snapshot["routes"]] == [0, 2]
        assert snapshot["routes"][0]["expert_ids"] == [[2, 0], [1, 2]]
        np.testing.assert_allclose(snapshot["routes"][0]["routing_weights"], [[.7, .3], [.6, .4]])
        assert snapshot["executions"][-1]["dispatch_id"] is None
        assert [r["rows"] for r in snapshot["executions"][:3]] == [1, 1, 2]
        assert analyze_trace(snapshot, capacities=(0, 1, 3))["identity"] == snapshot["identity"]
    else:
        assert snapshot["routes"] == snapshot["executions"] == []


def test_summary_never_copies_route_ids_or_weights(monkeypatch):
    import mini_llm.diagnostics.routing_trace as module
    def forbidden(*args, **kwargs):
        raise AssertionError("summary must reuse existing host plan counts")
    monkeypatch.setattr(module, "asnumpy", forbidden)
    run_layer(profiler("summary"), workers=2)


def test_bounded_records_and_snapshot_ownership(tmp_path):
    trace = profiler(max_records=2)
    run_layer(trace, workers=2, calls=4)
    snap = trace.snapshot()
    assert len(snap["routes"]) == len(snap["executions"]) == 2
    assert snap["counters"]["dropped_route_records"] == 2
    assert snap["counters"]["dropped_execution_records"] == 11
    assert snap["counters"]["routed_assignments"] == 16
    snap["routes"][0]["expert_ids"][0][0] = 999
    snap["executions"][0]["timings_ms"].clear()
    assert trace.snapshot()["routes"][0]["expert_ids"][0][0] != 999
    path = tmp_path / "trace.json"
    trace.write_json(path)
    assert json.loads(path.read_text())["schema_version"] == 1


def test_cache_decisions_bytes_and_evictions_are_not_arithmetic_changes():
    trace = profiler()
    cache = AdaptiveExpertCache(32, promote_after=1, replacement_margin=1)
    first, second = ExpertKey("fixture", 0, 0), ExpertKey("fixture", 1, 0)
    for key in (first, first, second):
        ticket = trace.new_expert(key.layer_id, key.expert_id, 1)
        before, residents = dict(cache.counters), set(cache.resident)
        expert = cache.request(key, 32, object)
        ticket.selected(key, expert, cache, before, residents, cache_enabled=True)
    snap = trace.snapshot()
    assert [e["path"] for e in snap["executions"]] == ["gpu_promote", "gpu_hit", "cpu"]
    # The colder second expert is not uploaded solely because it missed.
    assert snap["counters"]["transferred_bytes"] == 32
    assert snap["counters"]["cache_misses"] == 2
    assert snap["counters"]["promoted_experts"] == 1
    for _ in range(3):
        before, residents = dict(cache.counters), set(cache.resident)
        ticket = trace.new_expert(1, 0, 1)
        expert = cache.request(second, 32, object)
        ticket.selected(second, expert, cache, before, residents, cache_enabled=True)
    assert any(e["evicted_experts"] == [dict(layer_id=0, expert_id=0)]
               for e in trace.snapshot()["executions"])


def test_cuda_timers_poll_without_sync_and_are_bounded(monkeypatch):
    import mini_llm.diagnostics.routing_trace as module
    events = []
    class Event:
        def __init__(self):
            self.done = False
            events.append(self)
        def record(self, stream):
            pass
        def query(self):
            return self.done
        def synchronize(self):
            raise AssertionError("profiling must never synchronize")
    monkeypatch.setattr(module, "BACKEND_NAME", "cupy")
    monkeypatch.setattr(module, "xp", SimpleNamespace(cuda=SimpleNamespace(
        Event=Event, get_current_stream=lambda: None, get_elapsed_time=lambda a, b: 2.5)))
    trace = profiler(max_pending_events=1)
    ticket = trace.new_expert(0, 0, 1)
    for _ in range(2):
        with ticket.span("gpu_compute", gpu=True):
            pass
    snap = trace.snapshot()
    assert snap["pending_gpu_timers"] == 1
    assert snap["counters"]["dropped_gpu_timers"] == 1
    assert "gpu_compute_gpu_ms" not in snap["timings_ms"]
    for event in events:
        event.done = True
    snap = trace.snapshot()
    assert snap["pending_gpu_timers"] == 0
    assert snap["timings_ms"]["gpu_compute_gpu_ms"] == 2.5


@pytest.mark.parametrize("kwargs", [dict(mode="unknown"), dict(max_records=0),
                                     dict(max_pending_events=-1), dict(max_records=True)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        profiler(**kwargs)


def test_trace_model_identity_must_match_executor():
    with pytest.raises(ValueError, match="identity"):
        HeterogeneousExpertExecutor(lambda _: None, model_id="other", d_model=8, d_ff=8,
                                    routing_profiler=profiler())


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CuPy GPU")
def test_detailed_mixed_gpu_cpu_trace_and_deferred_events():
    trace = profiler()
    rng = np.random.default_rng(106091)
    d = f = 32
    bank = {ExpertKey("fixture", 2, i): tuple(
        rng.normal(0, .02, size=(d, f)).astype(np.float32) for _ in range(3))
        for i in range(2)}
    mixed = HeterogeneousExpertExecutor(lambda k: bank[k], model_id="fixture", d_model=d,
        d_ff=f, compute_dtype="float32", cpu_workers=2, routing_profiler=trace,
        device_budget_bytes=3*d*f*4, promote_after=2)
    reference = HeterogeneousExpertExecutor(lambda k: bank[k], model_id="fixture", d_model=d,
        d_ff=f, compute_dtype="float32")
    x = xp.asarray(rng.normal(0, .1, size=(1, 1, d)).astype(np.float32))
    ids = xp.asarray([[[0, 1]]], dtype=xp.int32)
    weights = xp.asarray([[[.6, .4]]], dtype=xp.float32)
    try:
        for _ in range(2):
            mixed.forward(2, 0, x.reshape(-1, d))
        actual = ExpertsInference(d, f, 2, expert_executor=mixed, layer_id=2).forward(x, weights, ids)
        expected = ExpertsInference(d, f, 2, expert_executor=reference, layer_id=2).forward(x, weights, ids)
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=1e-4, atol=1e-7)
        mixed.close()
        xp.cuda.get_current_stream().synchronize()
        assert mixed.cache._busy_keys() == set()
        assert mixed.cache.statistics()["inflight_dispatches"] == 0
        snap = trace.snapshot()
        assert snap["pending_gpu_timers"] == 0
        assert snap["counters"]["cache_hits"] >= 1
        assert snap["counters"]["cpu_experts"] >= 1
        assert snap["timings_ms"]["gpu_compute_gpu_ms"] >= 0
        assert snap["timings_ms"]["activation_d2h_gpu_ms"] >= 0
        assert snap["counters"]["transferred_bytes"] == 3*d*f*4
    finally:
        mixed.close()
        reference.close()
