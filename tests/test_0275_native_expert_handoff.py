"""Stream-ordered native CPU handoff versus retained identical arithmetic."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.expert_residency import ExpertKey
from test_0256_ggml_packed_cpu import selected_library

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Native CUDA host-node handoff')


@pytest.fixture(autouse=True)
def native_mode(monkeypatch):
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_HANDOFF', '1')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'warp')


@pytest.mark.parametrize('hits', [0, 3, 10])
def test_native_handoff_exact_retained_outputs_and_statistics(selected_library, hits):
    from test_0268_device_expert_plan import managed, oracle
    executor = managed(selected_library)
    rng = np.random.default_rng(275)
    x = xp.asarray(rng.normal(0, .1, (1, 256)).astype(np.float32))
    ids = [19, 0, 511, 4, 7, 11, 3, 1, 13, 9]
    routes = np.linspace(.01, .1, 10, dtype=np.float32)
    try:
        for e in ids[:hits]:
            executor._prepare(0, e, x)
        executor.cache.set_admission_budget(0)
        expected = oracle(executor, x, ids, routes)
        before = executor.statistics()['execution'].get('native_handoff_dispatches', 0)
        saved = []
        for _ in range(12):
            output = executor.try_forward_device_plan(0, x, xp.asarray(routes),
                                                     xp.asarray(ids, dtype=xp.int64), 512)
            saved.append(output)
        # One final read fences the dependency chain, not every layer dispatch.
        xp.cuda.get_current_stream().synchronize()
        for output in saved:
            np.testing.assert_array_equal(output.get(), expected.get())
        stats = executor.statistics()
        assert stats['execution']['native_handoff_dispatches'] - before == 12
        assert stats['execution']['cpu_submitted'] == 12 * (10-hits)
        assert stats['device']['protected_experts'] == hits
        executor.statistics()  # Counter collection is idempotent.
        assert executor.counters['native_handoff_dispatches'] == 12
        plan = executor._device_expert_plans[0]
        assert plan.native_handoff is not None
        # A placement change retires callbacks and releases every fixed lease.
        executor.cache.set_admission_budget(None)
        assert executor.try_forward_device_plan(0, x, xp.asarray(routes),
            xp.asarray(ids, dtype=xp.int64), 512) is None
        assert not executor.cache._protected
    finally:
        executor.close()


def test_callback_exception_and_pending_close(selected_library):
    from test_0268_device_expert_plan import managed
    executor = managed(selected_library)
    x = xp.ones((1, 256), xp.float32)
    routes, ids = xp.full(10, .1, xp.float32), xp.arange(10, dtype=xp.int64)
    try:
        executor._prepare(0, 0, x)
        executor.cache.set_admission_budget(0)
        def fail():
            raise RuntimeError('shared launch failure')
        with pytest.raises(RuntimeError, match='shared launch failure'):
            executor.try_forward_device_plan(0, x, routes, ids, 512, on_cpu_submitted=fail)
        # Close is deliberately called without a manual CUDA fence.
    finally:
        executor.close()
    assert not executor.cache._protected


def test_invalid_routes_are_explicit_not_stale(selected_library):
    from test_0268_device_expert_plan import managed
    executor = managed(selected_library)
    x = xp.ones((1, 256), xp.float32)
    routes, ids = xp.full(10, .1, xp.float32), xp.arange(10, dtype=xp.int64)
    executor.cache.set_admission_budget(0)
    try:
        executor.try_forward_device_plan(0, x, routes, ids, 512)
        bad = ids.copy()
        bad[1] = bad[0]
        output = executor.try_forward_device_plan(0, x, routes, bad, 512)
        assert not np.isfinite(output.get()).all()
        with pytest.raises(RuntimeError, match='invalid/failed dispatches'):
            executor.statistics()
    finally:
        executor.close()
