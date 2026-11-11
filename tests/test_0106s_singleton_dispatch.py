"""0106S: bounded managed singleton dispatch contracts, no model fixtures."""

from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
import mini_llm.ops.experts_inference as dispatch


class FakeExecutor:
    """Mixed ready arrays/Futures, with deterministic scheduling observations."""

    def __init__(self, asynchronous=True, values=(4e20, -8e20, 6.0),
                 fail_submit=None, fail_resolve=(), malformed=None):
        self.overlaps_cpu_gpu = asynchronous
        self.values = values
        self.fail_submit = fail_submit
        self.fail_resolve = fail_resolve
        self.malformed = malformed
        self.events = []
        self.inputs = []
        self.pending = {}
        self.protected = False

    @contextmanager
    def dispatch_scope(self, layer, experts):
        assert not self.protected
        self.events.append(("enter", layer, tuple(experts)))
        self.protected = True
        try:
            yield
        finally:
            self.protected = False
            self.events.append(("exit", layer))

    def _output(self, layer, expert, x):
        assert self.protected and layer == 7
        self.inputs.append(x)
        shape = (1, 2) if expert == self.malformed else x.shape
        return xp.full(shape, self.values[expert], dtype=x.dtype)

    def forward(self, layer, expert, x):
        self.events.append(("forward", expert))
        return self._output(layer, expert, x)

    def submit(self, layer, expert, x):
        self.events.append(("submit", expert))
        if expert == self.fail_submit:
            raise RuntimeError("submit failure")
        output = self._output(layer, expert, x)
        if expert % 2 == 0:
            work = Future()
            work.set_result(output)
        else:
            work = output
        self.pending[id(work)] = expert
        return work

    def resolve(self, work):
        assert self.protected
        expert = self.pending[id(work)]
        self.events.append(("resolve", expert))
        if expert in self.fail_resolve:
            raise RuntimeError(f"resolve failure {expert}")
        return work.result() if isinstance(work, Future) else work


def layer(executor):
    return ExpertsInference(3, 4, 3, expert_executor=executor, layer_id=7)


def inputs():
    return (xp.ones((1, 1, 3), dtype=xp.float32),
            xp.asarray([[[.5, .25, .125]]], dtype=xp.float32),
            xp.asarray([[[2, 0, 1]]], dtype=xp.int32))


def forbidden(*args, **kwargs):
    raise AssertionError("singleton dispatch used a general routing operation")


class BasicViewsOnly(np.ndarray):
    """Reject advanced input gathers and output indexed scatters."""

    @staticmethod
    def _check(index):
        indices = index if isinstance(index, tuple) else (index,)
        assert all(isinstance(i, (slice, int)) or i is None or i is Ellipsis
                   for i in indices), "advanced indexing is forbidden"

    def __getitem__(self, index):
        self._check(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        self._check(index)
        return super().__setitem__(index, value)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy view instrumentation")
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cpu_order_shared_input_basic_weight_views_and_no_scatter(monkeypatch, asynchronous):
    x, weights, ids = inputs()
    x = x.view(BasicViewsOnly)
    weights = weights.view(BasicViewsOnly)
    readbacks = []
    original_asnumpy = dispatch.asnumpy

    def read_ids(value):
        assert value is ids
        readbacks.append(value.shape)
        return original_asnumpy(value)

    monkeypatch.setattr(dispatch, "asnumpy", read_ids)
    monkeypatch.setattr(dispatch.RoutingPlan, "from_router_outputs", forbidden)
    monkeypatch.setattr(dispatch, "xp", SimpleNamespace(
        zeros=lambda shape, dtype: np.zeros(shape, dtype).view(BasicViewsOnly)))
    executor = FakeExecutor(asynchronous)
    result = layer(executor).forward(x, weights, ids)
    # In expert order: 1e20 - 1e20 + 3 == 3. Router-slot order gives zero.
    np.testing.assert_array_equal(np.asarray(result), np.full((1, 1, 3), 3, np.float32))
    assert readbacks == [(1, 1, 3)]
    assert len(executor.inputs) == 3
    assert all(value is executor.inputs[0] for value in executor.inputs)
    assert np.shares_memory(executor.inputs[0], x)
    assert executor.inputs[0].shape == (1, 3)
    assert executor.events[0] == ("enter", 7, (0, 1, 2))
    assert executor.events[-1] == ("exit", 7)
    operation = "submit" if asynchronous else "forward"
    assert [event for event in executor.events if event[0] == operation] == [
        (operation, 0), (operation, 1), (operation, 2)]
    if asynchronous:
        assert executor.events[1:-1] == [
            ("submit", 0), ("submit", 1), ("submit", 2),
            ("resolve", 0), ("resolve", 1), ("resolve", 2)]
    assert not executor.protected


@pytest.mark.parametrize("failure", ["submit", "resolve", "accumulate"])
def test_async_errors_drain_every_started_job_before_scope_exit(failure):
    kwargs = {"values": (1, 2, 3)}
    if failure == "submit":
        kwargs.update(fail_submit=2, fail_resolve=(0, 1))
        error, message, resolved = RuntimeError, "submit failure", [0, 1]
    elif failure == "resolve":
        kwargs.update(fail_resolve=(0, 1))
        error, message, resolved = RuntimeError, "resolve failure 0", [0, 1, 2]
    else:
        kwargs.update(malformed=0)
        error, message, resolved = ValueError, "broadcast", [0, 1, 2]
    executor = FakeExecutor(**kwargs)
    with pytest.raises(error, match=message):
        layer(executor).forward(*inputs())
    assert [event[1] for event in executor.events if event[0] == "resolve"] == resolved
    assert executor.events[-1] == ("exit", 7)
    assert not executor.protected


def test_sync_error_releases_dispatch_scope():
    executor = FakeExecutor(False, values=(1, 2, 3), malformed=0)
    with pytest.raises(ValueError, match="broadcast"):
        layer(executor).forward(*inputs())
    assert executor.events == [("enter", 7, (0, 1, 2)), ("forward", 0), ("exit", 7)]
    assert not executor.protected


@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_ids_fall_back_to_existing_general_semantics(monkeypatch, asynchronous):
    x, weights, _ = inputs()
    ids = xp.asarray([[[2, 0, 2]]], dtype=xp.int32)
    original = dispatch.RoutingPlan.from_router_outputs
    plans = []

    def general(*args, **kwargs):
        plan = original(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(dispatch.RoutingPlan, "from_router_outputs", general)
    executor = FakeExecutor(asynchronous, values=(1, 2, 3))
    actual = layer(executor).forward(x, weights, ids)
    native = ExpertsInference(3, 4, 3)
    native.experts = [SimpleNamespace(forward=lambda value, e=e:
                      xp.full(value.shape, (1, 2, 3)[e], dtype=value.dtype))
                      for e in range(3)]
    expected = native.forward(x, weights, ids)
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    assert len(plans) == 2
    assert plans[0].get_assignment_count(2) == 2
    assert [value.shape[0] for value in executor.inputs] == [1, 2]
    assert executor.events[0] == ("enter", 7, (0, 2))


@pytest.mark.parametrize("managed", [False, True])
def test_native_singleton_and_managed_prefill_do_not_read_singleton_ids(monkeypatch, managed):
    monkeypatch.setattr(dispatch, "asnumpy", forbidden)
    x, weights, ids = inputs()
    if managed:
        x, weights, ids = (xp.repeat(value, 2, axis=1) for value in (x, weights, ids))
        model = layer(FakeExecutor(values=(1, 2, 3)))
    else:
        model = ExpertsInference(3, 4, 3)
        model.experts = [SimpleNamespace(forward=lambda value: value) for _ in range(3)]
    assert model.forward(x, weights, ids).shape == x.shape


@pytest.mark.parametrize("mode", ["summary", "detailed"])
@pytest.mark.parametrize("workers", [0, 2])
def test_singleton_routing_trace_counts_identity_and_dispatch_ids(monkeypatch, mode, workers):
    import mini_llm.diagnostics.routing_trace as trace_module

    trace = RoutingTrace(model_id="singleton", quantization="F32",
                         expert_mapping="identity-v1", workload="tiny", mode=mode)
    rng = np.random.default_rng(10619)
    bank = {ExpertKey("singleton", 7, e): tuple(
        rng.normal(0, .1, size=s).astype(np.float32)
        for s in ((3, 4), (3, 4), (4, 3))) for e in range(3)}
    executor = HeterogeneousExpertExecutor(
        lambda key: bank[key], model_id="singleton", d_model=3, d_ff=4,
        compute_dtype="float32", cpu_workers=workers)
    model = layer(executor)
    x, weights, ids = inputs()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(dispatch._SingletonRoutingPlan, "from_router_outputs",
                          lambda *args: None)
            expected = asnumpy(model.forward(x, weights, ids))
        executor.routing_profiler = trace
        if mode == "summary":
            monkeypatch.setattr(trace_module, "asnumpy", forbidden)
        for _ in range(2):
            np.testing.assert_array_equal(asnumpy(model.forward(x, weights, ids)), expected)
        executor.forward(7, 0, x.reshape(1, 3))
        if BACKEND_NAME == "cupy":
            xp.cuda.get_current_stream().synchronize()
        snap = trace.snapshot()
        assert snap["identity"] == dict(model_id="singleton", quantization="F32",
                                        expert_mapping="identity-v1", workload="tiny")
        assert snap["tokens_by_layer"] == {"7": 2}
        assert snap["frequencies"] == [dict(layer_id=7, expert_id=e, selections=2)
                                        for e in range(3)]
        assert snap["counters"]["layer_dispatches"] == 2
        assert snap["counters"]["routed_assignments"] == 6
        assert snap["counters"]["cpu_experts"] == 7
        assert snap["counters"]["cpu_assignments"] == 7
        assert snap["counters"]["cache_requests"] == 0
        assert snap["counters"]["transferred_bytes"] == 0
        if mode == "detailed":
            assert [r["token_start"] for r in snap["routes"]] == [0, 1]
            assert snap["routes"][0]["expert_ids"] == [[2, 0, 1]]
            assert snap["routes"][0]["routing_weights"] == [[.5, .25, .125]]
            assert [e["dispatch_id"] for e in snap["executions"]] == [0]*3 + [1]*3 + [None]
            assert [e["expert_id"] for e in snap["executions"]] == [0, 1, 2]*2 + [0]
            assert all(e["rows"] == 1 for e in snap["executions"])
        else:
            assert snap["routes"] == snap["executions"] == []
    finally:
        executor.close()


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CuPy/CUDA required")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dtype,weight_dtype", [
    ("float16", "float16"), ("float16", "float32"),
    ("float32", "float32"), ("float64", "float64")])
def test_cuda_exact_general_first_row_parity_without_gpu_sort(monkeypatch, asynchronous,
                                                            dtype, weight_dtype):
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        x = xp.asarray([[[.1, -.2, .3]]], dtype=dtype)
        weights = xp.asarray([[[.471, .317, .212]]], dtype=weight_dtype)
        ids = xp.asarray([[[2, 0, 1]]], dtype=xp.int32)
        executor = FakeExecutor(asynchronous, values=(.12345, -1.2345, 2.3456))
        model = layer(executor)
        expected = model.forward(*(xp.repeat(v, 2, axis=1) for v in (x, weights, ids)))
        executor.inputs.clear()
        with monkeypatch.context() as patch:
            patch.setattr(xp, "argsort", forbidden)
            patch.setattr(xp, "bincount", forbidden)
            patch.setattr(dispatch.RoutingPlan, "from_router_outputs", forbidden)
            actual = model.forward(x, weights, ids)
        assert all(v is executor.inputs[0] for v in executor.inputs)
        assert executor.inputs[0].data.ptr == x.data.ptr
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected[:, :1]))
        assert actual.dtype == x.dtype


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CuPy/CUDA required")
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cuda_packed_executor_general_parity_on_nondefault_stream(workers, dtype):
    pytest.importorskip("triton")
    from mini_llm.ops.quantized_weight import PackedGGUFWeight

    rng = np.random.default_rng(10620)
    d = 32

    def packed():
        raw = rng.integers(0, 256, size=(d*d//32, 18), dtype=np.uint8)
        raw[:, :2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
        return PackedGGUFWeight(raw.reshape(-1), (d, d), "Q4_0")

    bank = {ExpertKey("packed-singleton", 7, e): tuple(packed() for _ in range(3))
            for e in range(3)}
    budget = sum(w.nbytes for expert in bank.values() for w in expert)
    executor = HeterogeneousExpertExecutor(
        lambda key: bank[key], model_id="packed-singleton", d_model=d, d_ff=d,
        compute_dtype=dtype, cpu_workers=workers, device_budget_bytes=budget,
        promote_after=1)
    model = ExpertsInference(d, d, 3, expert_executor=executor, layer_id=7)
    stream = xp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            x = xp.asarray(rng.normal(0, .1, size=(1, 1, d)), dtype=dtype)
            weights = xp.asarray([[[.47, .32, .21]]], dtype=dtype)
            ids = xp.asarray([[[2, 0, 1]]], dtype=xp.int32)
            for e in range(3):
                executor.forward(7, e, x.reshape(1, d))
            assert len(executor.cache.resident) == 3
            before = dict(executor.cache.counters)
            expected = model.forward(*(xp.repeat(v, 2, axis=1) for v in (x, weights, ids)))
            actual = model.forward(x, weights, ids)
            stream.synchronize()
            np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected[:, :1]))
        assert executor.cache.counters["hits"] - before.get("hits", 0) == 6
        assert executor.cache.counters["promotions"] == 3
        assert executor.counters["cpu_submitted"] == 0
        assert executor.cache._busy_keys() == set()
        assert executor._dispatch_depth == 0
    finally:
        executor.close()
