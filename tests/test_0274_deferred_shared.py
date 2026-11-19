"""CPU-publication scheduling seam; arithmetic, callback count and failure drain."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.experts_inference import ExpertsInference


def test_generic_submit_callback_precedes_wait_and_drains():
    events = []
    executor = SimpleNamespace(
        overlaps_cpu_gpu=True,
        dispatch_scope=lambda *a: nullcontext(),
        submit=lambda layer, e, x: events.append(('submit', e)) or x.copy(),
        resolve=lambda work: events.append(('resolve', None)) or work)
    experts = ExpertsInference(4, 4, 2, expert_executor=executor)
    x = xp.ones((1, 1, 4), xp.float32)
    weights = xp.full((1, 1, 2), .5, xp.float32)
    ids = xp.asarray([[[0, 1]]], dtype=xp.int64)
    def callback():
        events.append(('callback', None))
    np.testing.assert_array_equal(asnumpy(experts.forward(x, weights, ids,
        on_cpu_submitted=callback)), asnumpy(x))
    assert [e[0] for e in events] == ['submit', 'submit', 'callback', 'resolve', 'resolve']
    events.clear()
    def fail():
        callback()
        raise RuntimeError('shared failure')
    with pytest.raises(RuntimeError, match='shared failure'):
        experts.forward(x, weights, ids, on_cpu_submitted=fail)
    assert [e[0] for e in events] == ['submit', 'submit', 'callback', 'resolve', 'resolve']


def test_unmanaged_callback_once():
    experts = ExpertsInference(4, 4, 2)
    for e in experts.experts:
        e.set_weight_arrays(xp.eye(4, dtype=xp.float32), xp.eye(4, dtype=xp.float32),
                            xp.eye(4, dtype=xp.float32))
    x = xp.ones((1, 1, 4), xp.float32)
    weights = xp.full((1, 1, 2), .5, xp.float32)
    ids = xp.asarray([[[0, 1]]], dtype=xp.int64)
    events = []
    expected = experts.forward(x, weights, ids)
    actual = experts.forward(x, weights, ids, on_cpu_submitted=lambda: events.append('callback'))
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    assert events == ['callback']


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA residency')
@pytest.mark.parametrize('hits', [0, 3, 10])
def test_device_plan_callback_and_failure(selected_library, hits):
    from test_0268_device_expert_plan import managed
    executor = managed(selected_library)
    experts = ExpertsInference(256, 320, 512, expert_executor=executor)
    x = xp.ones((1, 1, 256), xp.float32)
    weights = xp.full((1, 1, 10), .1, xp.float32)
    ids = xp.arange(10, dtype=xp.int64).reshape(1, 1, 10)
    try:
        for e in range(hits):
            executor._prepare(0, e, x.reshape(1, 256))
        executor.cache.set_admission_budget(0)
        expected = experts.forward(x, weights, ids)
        calls = []
        def callback():
            calls.append(executor.counters['cpu_submitted'])
            assert (executor._ggml_row_pool._pending is not None) == (hits != 10)
        actual = experts.forward(x, weights, ids, on_cpu_submitted=callback)
        np.testing.assert_array_equal(actual.get(), expected.get())
        assert len(calls) == 1
        def fail():
            raise RuntimeError('deferred shared failure')
        with pytest.raises(RuntimeError, match='deferred shared failure'):
            experts.forward(x, weights, ids, on_cpu_submitted=fail)
        assert executor._ggml_row_pool._pending is None
        # A failed callback must not poison persistent slabs or the pool.
        np.testing.assert_array_equal(experts.forward(x, weights, ids).get(), expected.get())
    finally:
        executor.close()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA shared stream')
@pytest.mark.parametrize('graph', [False, True])
def test_deferred_shared_exact_and_order(graph):
    from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
    from mini_llm.ops.inference_matmul import matmul_inference
    rng = np.random.default_rng(274)
    def rand(shape):
        return xp.asarray(rng.normal(0, .1, shape).astype(np.float32))
    def project(w):
        return lambda x: matmul_inference(x.reshape(1, -1), w).reshape(-1)
    events = []
    def routed(x, weights, ids, *, on_cpu_submitted=None):
        events.append('submit')
        if on_cpu_submitted:
            on_cpu_submitted()
        events.append('wait')
        return x * xp.float32(.5)
    router = SimpleNamespace(forward=lambda x: (xp.ones((1, 1, 1), xp.float32),
                                               xp.zeros((1, 1, 1), xp.int64)))
    args = (router, SimpleNamespace(forward=routed), project(rand((64, 96))),
            project(rand((64, 96))), project(rand((96, 64))), rand((64,)))
    control = Qwen4ExpMoE(*args, overlap_shared=True, shared_graph_decode=graph)
    candidate = Qwen4ExpMoE(*args, overlap_shared=True, shared_graph_decode=graph, defer_shared=True)
    original = candidate._run_shared
    def shared(x):
        events.append('shared')
        return original(x)
    candidate._run_shared = shared
    for _ in range(8):
        x = rand((64,))
        expected = control(x)
        events.clear()
        actual = candidate(x)
        assert events == ['submit', 'shared', 'wait']
        np.testing.assert_array_equal(actual.get(), expected.get())
    with pytest.raises(ValueError, match='requires overlap_shared'):
        Qwen4ExpMoE(*args, defer_shared=True)


# Explicit import makes the pinned-library fixture available without global conftest changes.
from test_0256_ggml_packed_cpu import selected_library
