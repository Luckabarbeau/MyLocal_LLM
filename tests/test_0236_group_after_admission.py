"""Group complete newly admitted bundles without recounting or skipping routes."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_linear import PackedExpertFFNInference
from test_0231_grouped_iq_gpu import ACTUAL
from test_0228_iq_gpu_integration import packed

GPU = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual packed CuPy required')
IDS = [9, 0, 7, 2, 5, 1, 8, 3, 6, 4]
ROUTES = np.linspace(.03, .17, 10, dtype=np.float32)
ROUTES /= ROUTES.sum()


def fixture(formats, enabled, bank, budget, **kwargs):
    return HeterogeneousExpertExecutor(lambda key: bank[key.expert_id], model_id='0236',
        d_model=256, d_ff=256, device_budget_bytes=budget, promote_after=1,
        compute_dtype='float32', cpu_workers=2, cpu_packed_backend='native',
        group_after_admission=enabled, **kwargs)


def dispatch(executor, x, ids=IDS):
    return ExpertsInference(256, 256, 512, expert_executor=executor).forward(
        x.reshape(1, 1, 256), xp.asarray(ROUTES.reshape(1, 1, 10)),
        xp.asarray(np.asarray(ids, np.int64).reshape(1, 1, 10)))


@GPU
@pytest.mark.parametrize('formats', ACTUAL)
def test_cold_admission_group_exact_and_requests_counted_once(formats, monkeypatch):
    bank = {e: tuple(packed(f, 256, 256, 2360+e*3+p)
                     for p, f in enumerate(formats)) for e in range(10)}
    budget = sum(w.nbytes for ws in bank.values() for w in ws)
    old = fixture(formats, False, bank, budget)
    new = fixture(formats, True, bank, budget)
    try:
        x = xp.asarray(np.random.default_rng(236).normal(0, .1, (1, 256)).astype(np.float32))
        expected = asnumpy(dispatch(old, x)).copy()
        def forbidden(*args):
            pytest.fail('newly admitted all-GPU experts must use the grouped kernels')
        monkeypatch.setattr(PackedExpertFFNInference, 'forward', forbidden)
        got = dispatch(new, x)
        np.testing.assert_array_equal(asnumpy(got), expected)
        stats = new.statistics()
        assert stats['device']['requests'] == 10
        assert stats['device']['promotions'] == stats['device']['resident_experts'] == 10
        assert stats['execution']['gpu_experts'] == 10
        assert stats['execution']['admission_grouped_dispatches'] == 1
        assert stats['execution'].get('native_cpu_experts', 0) == 0
        assert stats['device']['resident_bytes'] <= budget
        np.testing.assert_array_equal(asnumpy(dispatch(new, x)), expected)
        assert new.cache.counters['requests'] == 20
        assert new.counters['admission_grouped_dispatches'] == 1
        assert new.counters['grouped_gpu_dispatches'] == 2
    finally:
        old.close(); new.close()
        old.cache.clear(); new.cache.clear()


@GPU
def test_budget_miss_preserves_mixed_fallback_and_nondefault_stream():
    formats = ACTUAL[0]
    bank = {e: tuple(packed(f, 256, 256, 2370+e*3+p)
                     for p, f in enumerate(formats)) for e in range(11)}
    budget = 10 * sum(w.nbytes for w in bank[0])
    old = fixture(formats, False, bank, budget)
    new = fixture(formats, True, bank, budget)
    stream = xp.cuda.Stream(non_blocking=True)
    try:
        xp.cuda.get_current_stream().synchronize()
        with stream:
            x = xp.asarray(np.random.default_rng(237).normal(0, .1, (1, 256)).astype(np.float32))
            for executor in (old, new):
                dispatch(executor, x)
                dispatch(executor, x)
            changed = list(IDS)
            changed[0] = 10
            expected = dispatch(old, x, changed)
            actual = dispatch(new, x, changed)
            dependent = actual * xp.float32(2)
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(dependent), asnumpy(expected) * 2)
        assert new.counters['native_cpu_experts'] == 1
        assert new.counters['admission_grouped_dispatches'] == 1
        assert new.cache.used_bytes <= budget
        assert not new.cache._protected
    finally:
        old.close(); new.close()
        old.cache.clear(); new.cache.clear()


@GPU
def test_group_failure_keeps_complete_sources_and_releases_protection(monkeypatch):
    from mini_llm.ops import grouped_quantized_experts as grouped
    formats = ACTUAL[0]
    bank = {e: tuple(packed(f, 256, 256, 2380+e*3+p)
                     for p, f in enumerate(formats)) for e in range(10)}
    budget = sum(w.nbytes for ws in bank.values() for w in ws)
    executor = fixture(formats, True, bank, budget)
    try:
        def fail(*args, **kwargs):
            raise RuntimeError('deliberate grouped failure')
        monkeypatch.setattr(grouped, 'grouped_quantized_experts_forward', fail)
        with pytest.raises(RuntimeError, match='deliberate grouped failure'):
            dispatch(executor, xp.ones((1, 256), dtype=xp.float32))
        assert not executor.cache._protected
        assert len(executor.cache.resident) == 10
        assert executor.cache.used_bytes == budget
        assert executor.cache.counters['requests'] == 10
    finally:
        executor.close(); executor.cache.clear()


def test_option_validation_and_cpu_mode_remains_cpu():
    formats = ACTUAL[0]
    bank = {e: tuple(packed(f, 256, 256, 2390+e*3+p)
                     for p, f in enumerate(formats)) for e in range(10)}
    with pytest.raises(TypeError, match='group_after_admission'):
        fixture(formats, 'true', bank, 0)
    if BACKEND_NAME == 'numpy':
        a = fixture(formats, False, bank, 0, execution_mode='cpu')
        b = fixture(formats, True, bank, 0, execution_mode='cpu')
        try:
            x = np.ones((1, 256), dtype=np.float32)*.01
            np.testing.assert_array_equal(dispatch(a, x), dispatch(b, x))
            assert not b.cache.resident
            assert b.counters.get('admission_grouped_dispatches', 0) == 0
        finally:
            a.close(); b.close()
