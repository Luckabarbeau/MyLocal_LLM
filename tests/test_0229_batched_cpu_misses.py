"""Bounded singleton CPU-miss batching; run separately with numpy and cupy."""
import threading
from concurrent.futures import Future

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference


D = 16
IDS = [3, 0, 2, 1]
ROUTES = np.array([.125, .25, .5, .125], np.float32)


def make_executor(**kwargs):
    rng = np.random.default_rng(229)
    bank = {e: tuple(rng.normal(0, .1, (D, D)).astype(np.float32)
                     for _ in range(3)) for e in IDS}
    executor = HeterogeneousExpertExecutor(
        lambda key: bank[key.expert_id], model_id="0229", d_model=D, d_ff=D,
        compute_dtype="float32", cpu_workers=4, **kwargs)
    return executor


def dispatch(executor, x=None):
    if x is None:
        x = xp.ones((1, 1, D), dtype=xp.float32)
    return ExpertsInference(D, D, 4, expert_executor=executor).forward(
        x, xp.asarray(ROUTES.reshape(1, 1, 4)),
        xp.asarray(np.array(IDS, np.int64).reshape(1, 1, 4)))


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("resident", [(), (1, 3), (0, 1, 2, 3)])
def test_dense_mixed_exact_order_once_and_current_stream(enabled, resident, monkeypatch):
    """Actual CuPy residency/copies on GPU; deterministic CPU reference on NumPy."""
    executor = make_executor(batch_cpu_decode=enabled, promote_after=2,
                             device_budget_bytes=4 * 3 * D * D * 4)
    xh = np.linspace(-.5, .7, D, dtype=np.float32).reshape(1, D)
    stream = xp.cuda.Stream(non_blocking=True) if BACKEND_NAME == "cupy" else None
    from contextlib import nullcontext
    try:
        with stream if stream is not None else nullcontext():
            x = xp.asarray(xh)
            if BACKEND_NAME == "cupy":
                for e in resident:
                    executor.forward(0, e, x)
                    executor.forward(0, e, x)
                assert len(executor.cache.resident) == len(resident)
            # Build the exact reference from the same CPU/GPU tiers, not a
            # tolerance that might conceal reordered reduction or double weights.
            expected = xp.zeros_like(x)
            for e in sorted(IDS):
                key = ExpertKey("0229", 0, e)
                entry = executor.cache.resident.get(key)
                y = (entry[0].forward(x) if entry else xp.asarray(
                    executor._cpu_forward_numpy(key, executor._host_weights(key), xh)))
                expected += ROUTES[IDS.index(e)] * y
            prepares = []
            prepare = executor._prepare

            def tracked(layer, e, inputs):
                prepares.append(e)
                assert executor._dispatch_depth == 1
                return prepare(layer, e, inputs)

            monkeypatch.setattr(executor, "_prepare", tracked)
            seen = []
            compute = executor._run_cpu

            def tracked_cpu(key, weights, inputs, trace=None):
                if enabled:
                    assert not inputs.flags.writeable
                seen.append(inputs)
                return compute(key, weights, inputs, trace)

            monkeypatch.setattr(executor, "_run_cpu", tracked_cpu)
            # Produce on a non-default stream immediately before dispatch.
            produced = x + xp.float32(.25)
            produced -= xp.float32(.25)
            # Use its actual rounded values in both paths.
            produced[...] = x
            actual = dispatch(executor, produced.reshape(1, 1, D))
            dependent = actual * xp.float32(2)
        if stream is not None:
            stream.synchronize()
        np.testing.assert_array_equal(asnumpy(dependent).reshape(1, D), asnumpy(expected) * 2)
        assert prepares == sorted(IDS)
        if enabled:
            assert executor.counters["decode_batch_dispatches"] == 1
            assert len({id(a) for a in seen}) <= 1
            if BACKEND_NAME == "cupy":
                copies = int(len(resident) != 4)
                assert executor.counters["decode_activation_d2h"] == copies
                assert executor.counters["decode_results_h2d"] == copies
        else:
            assert not executor.counters["decode_batch_dispatches"]
    finally:
        executor.close()
        executor.cache.clear()


@pytest.mark.parametrize("failure", ["worker", "gpu", "submit"])
def test_exception_drains_every_started_future(failure, monkeypatch):
    executor = make_executor()
    prepare = executor._prepare
    compute = executor._run_cpu
    barrier = threading.Barrier(3, timeout=5)
    completed = []
    captured = []
    submit = executor._pool.submit

    def tracked_submit(*args, **kwargs):
        if failure == "submit" and len(captured) == 2:
            raise RuntimeError("injected submit")
        future = submit(*args, **kwargs)
        captured.append(future)
        return future

    class BrokenGPU:
        def forward(self, x):
            raise RuntimeError("injected gpu")

    def selection(layer, e, x):
        key, expert, weights, trace = prepare(layer, e, x)
        if failure == "gpu" and e == 3:
            return key, BrokenGPU(), None, trace
        return key, expert, weights, trace

    def work(key, weights, x, trace=None):
        try:
            # Three independently running jobs must start before any wait.
            if failure != "submit" and key.expert_id < 3:
                barrier.wait()
            if failure == "worker" and key.expert_id == 0:
                raise RuntimeError("injected worker")
            return compute(key, weights, x, trace)
        finally:
            completed.append(key.expert_id)

    monkeypatch.setattr(executor._pool, "submit", tracked_submit)
    monkeypatch.setattr(executor, "_prepare", selection)
    monkeypatch.setattr(executor, "_run_cpu", work)
    try:
        with pytest.raises(RuntimeError, match="injected " + failure):
            dispatch(executor)
        assert captured and all(isinstance(f, Future) and f.done() for f in captured)
        assert len(completed) == len(captured)
        assert executor._dispatch_depth == 0
        assert not executor.counters["decode_results_h2d"]
    finally:
        executor.close()


def test_cpu_jobs_overlap_resident_forward_and_sorted_fp32_combine(monkeypatch):
    executor = make_executor()
    prepare = executor._prepare
    started = threading.Event()
    release = threading.Event()
    values = {0: 1e20, 1: -1e20, 2: 3.0, 3: 4.0}
    visited = []

    class Resident:
        def forward(self, x):
            assert started.wait(5)
            visited.append(1)
            release.set()
            return xp.full_like(x, values[1])

    def selection(layer, e, x):
        key, expert, weights, trace = prepare(layer, e, x)
        return key, Resident() if e == 1 else None, weights, trace

    def work(key, weights, x, trace=None):
        assert not x.flags.writeable
        started.set()
        assert release.wait(5)
        return np.full_like(x, values[key.expert_id])

    monkeypatch.setattr(executor, "_prepare", selection)
    monkeypatch.setattr(executor, "_run_cpu", work)
    try:
        actual = dispatch(executor)
        expected = np.zeros((1, 1, D), np.float32)
        for e in sorted(IDS):
            expected += np.float32(values[e]) * ROUTES[IDS.index(e)]
        np.testing.assert_array_equal(asnumpy(actual), expected)
        assert visited == [1]
    finally:
        release.set()
        executor.close()


def test_ineligible_batch_is_observational(monkeypatch):
    executor = make_executor()
    x = xp.ones((1, D), xp.float32)
    routes = xp.asarray(ROUTES)

    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible call must not prepare any expert")

    monkeypatch.setattr(executor, "_prepare", forbidden)
    try:
        assert executor.try_forward_decode_batch(0, sorted(IDS), x, routes, IDS) is None
        with executor.dispatch_scope(0, sorted(IDS)):
            assert executor.try_forward_decode_batch(0, [0, 0], x, routes, IDS) is None
            assert executor.try_forward_decode_batch(0, IDS, x, routes, IDS) is None
            assert executor.try_forward_decode_batch(0, sorted(IDS), x.astype(xp.float16), routes, IDS) is None
            executor.routing_profiler = object()
            assert executor.try_forward_decode_batch(0, sorted(IDS), x, routes, IDS) is None
    finally:
        executor.close()
