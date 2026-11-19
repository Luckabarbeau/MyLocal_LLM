"""Dynamic routes/input and captured native host-node lifecycle validation."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from test_0256_ggml_packed_cpu import selected_library

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA host-node graphs')


def test_captured_handoff_dynamic_routes_and_inputs(selected_library, monkeypatch):
    from test_0268_device_expert_plan import managed, oracle
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_HANDOFF', '1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH', '1')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'warp')
    executor = managed(selected_library)
    rng = np.random.default_rng(276)
    saved = []
    try:
        for e in range(10):
            executor._prepare(0, e, xp.ones((1, 256), xp.float32))
        executor.cache.set_admission_budget(0)
        for step in range(24):
            ids = rng.choice(24, 10, replace=False).tolist()
            routes = rng.uniform(.01, .1, 10).astype(np.float32)
            x = xp.asarray(rng.normal(0, .1, (1, 256)).astype(np.float32))
            expected = oracle(executor, x, ids, routes)
            actual = executor.try_forward_device_plan(0, x, xp.asarray(routes),
                xp.asarray(ids, dtype=xp.int64), 512)
            saved.append((actual, expected))
        for actual, expected in saved:
            np.testing.assert_array_equal(actual.get(), expected.get())
        stats = executor.statistics()
        assert stats['execution']['native_handoff_dispatches'] == 24
        graph = executor._device_expert_plans[0].native_handoff.graph
        assert graph.statistics()['calls'] == 24
        assert graph.statistics()['pool_reserved_bytes'] <= 1024**2
    finally:
        executor.close()
    assert not executor.cache._protected
