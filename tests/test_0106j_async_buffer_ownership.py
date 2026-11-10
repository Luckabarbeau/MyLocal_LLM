"""0106J caller-stream ownership checks and bounded real-CUDA parity stress."""

from concurrent.futures import Future
from contextlib import nullcontext
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.benchmark_0106j_async_parity import (
    ATOL, DEVICE_BUDGET_BYTES, MODES, RTOL, build_fixture, reference, run_case,
)
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops import expert_residency as residency_module
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor


class _ImmediatePool:
    """Return a completed Future, but execute on a genuinely different thread."""

    def __init__(self):
        self.calls = []
        self.closed = False

    def submit(self, function, *args):
        assert not self.closed
        self.calls.append((function, args))
        future = Future()

        def invoke():
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "fake CPU worker did not complete"
        return future

    def shutdown(self, wait=True):
        assert wait is True
        self.closed = True


@pytest.mark.parametrize("caller_kind", ["default", "non-default"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("traced", [False, True])
def test_cpu_fake_caller_allocates_output_before_ready_event(monkeypatch, caller_kind, dtype, traced):
    """Reject worker-default-stream allocation without requiring/importing CuPy."""
    caller_thread = threading.get_ident()
    local = threading.local()
    log, allocations, tickets = [], [], []

    def record(name, *details):
        log.append((name, threading.get_ident(), get_current_stream(), details))

    class Stream:
        def __init__(self, non_blocking=False, *, name=None):
            self.name = name or "worker-copy"
            self.non_blocking = non_blocking

        def __enter__(self):
            self.previous = getattr(local, "stream", None)
            local.stream = self
            return self

        def __exit__(self, *args):
            local.stream = self.previous

        def wait_event(self, event):
            assert event.recorded
            assert allocations, "output must exist before worker waits for ready"
            record("wait_ready", event)

        def synchronize(self):
            assert threading.get_ident() != caller_thread, "no new caller synchronization"
            assert self.non_blocking
            record("worker_sync")

    default = Stream(name="default")
    caller = default if caller_kind == "default" else Stream(non_blocking=True, name="caller")
    Stream.null = default

    def get_current_stream():
        return getattr(local, "stream", None) or default

    class Event:
        def __init__(self, disable_timing=False):
            assert disable_timing
            self.recorded = False

        def record(self, stream):
            assert threading.get_ident() == caller_thread
            assert stream is caller
            assert len(allocations) == 1, "allocate output on caller stream BEFORE ready.record()"
            self.recorded = True
            record("record_ready", self)

        def synchronize(self):
            raise AssertionError("ready event must not be synchronized")

    class Device:
        def __init__(self, device_id):
            assert device_id == 7

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Array:
        def __init__(self, data):
            self.data = data
            self.shape, self.dtype = data.shape, data.dtype
            self.size, self.nbytes = data.size, data.nbytes

        def get(self, *, out, stream, blocking):
            assert not blocking and stream is get_current_stream()
            assert stream.non_blocking and threading.get_ident() != caller_thread
            record("d2h")
            np.copyto(out, self.data)

        def set(self, data, *, stream):
            assert stream is get_current_stream() and stream.non_blocking
            assert threading.get_ident() != caller_thread
            record("h2d", self)
            np.copyto(self.data, data)

    def allocate(shape, dtype):
        assert threading.get_ident() == caller_thread, "worker must not allocate GPU output"
        assert get_current_stream() is caller, "allocation must belong to the caller stream"
        output = Array(np.empty(shape, dtype=dtype))
        allocations.append(output)
        record("allocate_output", output)
        return output

    class Ticket:
        def __init__(self):
            self.transfers = []

        def span(self, name, **kwargs):
            return nullcontext()

        def selected(self, *args, **kwargs):
            assert kwargs["cache_enabled"] is False

        def activation_bytes(self, direction, nbytes):
            self.transfers.append((direction, nbytes))

    def new_expert(layer_id, expert_id, rows):
        assert (layer_id, expert_id, rows) == (3, 2, 2)
        ticket = Ticket()
        tickets.append(ticket)
        return ticket

    profiler = SimpleNamespace(identity={"model_id": "ownership"}, new_expert=new_expert)
    weights = tuple(np.eye(8, dtype=np.float32) * scale for scale in (.1, .2, .3))
    # Construct the real executor/pool before monkeypatching backend globals.
    executor = HeterogeneousExpertExecutor(lambda key: weights, model_id="ownership",
        d_model=8, d_ff=8, cpu_workers=2, routing_profiler=profiler if traced else None)
    executor._pool.shutdown(wait=True)
    pool = _ImmediatePool()
    executor._pool = pool
    original_worker = executor._copy_compute_copy
    received = []

    def worker(key, provided_weights, x, ready_event, device_id, gpu_y, trace=None):
        assert threading.get_ident() != caller_thread
        assert key == ExpertKey("ownership", 3, 2)
        assert all(a is b for a, b in zip(provided_weights, weights))
        assert x is inputs and device_id == 7 and ready_event.recorded
        assert gpu_y is allocations[0]
        assert trace is (tickets[0] if traced else None)
        received.append(gpu_y)
        record("worker_entry", gpu_y)
        return original_worker(key, provided_weights, x, ready_event, device_id, gpu_y, trace)

    fake_xp = SimpleNamespace(empty=allocate, cuda=SimpleNamespace(
        Device=Device, Event=Event, Stream=Stream, get_current_stream=get_current_stream,
        alloc_pinned_memory=lambda size: bytearray(size),
        runtime=SimpleNamespace(getDevice=lambda: 7),
    ))
    monkeypatch.setattr(residency_module, "BACKEND_NAME", "cupy")
    monkeypatch.setattr(residency_module, "xp", fake_xp)
    monkeypatch.setattr(executor, "_copy_compute_copy", worker)
    inputs = Array(np.linspace(-.5, .5, 16, dtype=dtype).reshape(2, 8))
    try:
        with caller:
            pending = executor.submit(3, 2, inputs)
            assert isinstance(pending, Future), "retain submit/resolve Future semantics"
            output = executor.resolve(pending)
        assert output is allocations[0] is received[0]
        assert output.shape == inputs.shape and output.dtype == inputs.dtype
        expected = executor._cpu_forward_numpy(ExpertKey("ownership", 3, 2), weights, inputs.data)
        np.testing.assert_array_equal(output.data, expected)
        names = [item[0] for item in log]
        assert names == ["allocate_output", "record_ready", "worker_entry", "wait_ready",
                         "d2h", "worker_sync", "h2d", "worker_sync"]
        assert len(allocations) == len(pool.calls) == 1
        assert log[0][1] == log[1][1] == caller_thread
        assert log[0][2] is log[1][2] is caller
        assert all(item[1] != caller_thread for item in log[2:])
        assert executor.counters["cpu_submitted"] == 1
        if traced:
            assert tickets[0].transfers == [("d2h", inputs.nbytes), ("h2d", output.nbytes)]
    finally:
        executor.close()
    assert pool.closed and executor._pool is None


@pytest.mark.gpu
@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires MINI_LLM_BACKEND=cupy")
def test_gpu_async_decode_matches_cpu_oracle_all_modes_and_streams():
    """128-token allocator stress, including fixed placement with zero evictions.

    Two repeats use the exact same fixture. The 15-second host deadline never
    inserts a token synchronization; each run copies outputs only at its end.
    GPU caches are capped at 1 MiB; no large allocator-pressure allocations.
    """
    try:
        if xp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA GPU")
    except xp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    fixture = build_fixture()
    expected = reference(fixture)
    assert fixture.inputs.shape == (128, 1, 1, 256)
    assert fixture.inputs.dtype == np.float32
    assert expected.shape == (128, 1, 256) and expected.dtype == np.float32
    deadline = time.monotonic() + 15.0
    for policy in ("fixed", "dynamic"):
        for caller_stream in ("default", "non-default"):
            for repeat in range(2):
                baseline = None
                for mode in MODES:
                    context = f"{policy=}, {caller_stream=}, {repeat=}, {mode=}"
                    output, report = run_case(fixture, mode=mode, workers=2,
                        residency=policy, caller_stream=caller_stream, deadline=deadline)
                    assert output.shape == expected.shape, context
                    assert output.dtype == expected.dtype, context
                    np.testing.assert_allclose(output, expected, rtol=RTOL, atol=ATOL,
                                               err_msg=context + "; CPU dequant/SwiGLU oracle")
                    if baseline is None:
                        baseline = output
                    np.testing.assert_allclose(output, baseline, rtol=RTOL, atol=ATOL,
                                               err_msg=context + "; disabled mode")
                    cache = report["gpu_cache_after"]
                    assert cache["device_budget_bytes"] == DEVICE_BUDGET_BYTES
                    assert 0 < cache["resident_bytes"] <= DEVICE_BUDGET_BYTES, context
                    execution = report["execution"]["execution"]
                    assert execution.get("cpu_submitted", 0) > 0, context
                    assert execution.get("gpu_experts", 0) > 0, context
                    if policy == "fixed":
                        assert cache["resident_experts"] == 9, context
                        assert cache.get("promotions", 0) == 9, context
                        assert cache.get("evictions", 0) == 0, context
                    if mode != "disabled":
                        assert report["trace"]["tokens_by_layer"] == {"0": 128}, context
    assert time.monotonic() <= deadline, "GPU regression exceeded 15 seconds"
