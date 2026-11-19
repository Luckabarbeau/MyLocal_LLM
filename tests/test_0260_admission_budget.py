"""Bound movement without changing selected experts or canonical ownership."""
import numpy as np
import pytest

from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey


def key(i):
    return ExpertKey('0260', 0, i)


@pytest.mark.parametrize('policy', ['sort', 'heap'])
def test_frozen_cache_retains_hits_and_observes_misses(policy):
    cache = AdaptiveExpertCache(2, promote_after=1, victim_policy=policy)
    cache.request(key(0), 1, lambda: 10)
    cache.request(key(1), 1, lambda: 11)
    cache.set_admission_budget(0)
    original = dict(cache.resident)
    def forbidden():
        raise AssertionError('Frozen placement must not materialize')
    for _ in range(200):
        assert cache.request(key(0), 1, forbidden) == 10
        assert cache.request(key(9), 1, forbidden) is None
    assert cache.resident == original and cache.used_bytes == 2
    assert cache._usage[key(9)].hits == 200
    assert cache.counters['admission_deferred'] == 200
    assert cache.counters['cpu_fallbacks'] == 200


def test_window_includes_hits_and_limits_attempts():
    cache = AdaptiveExpertCache(100, promote_after=1)
    cache.set_admission_budget(1, window_requests=4)
    materializations = []
    def request(i):
        return cache.request(key(i), 1, lambda: materializations.append(i) or i)
    assert request(0) == 0  # 1
    assert request(1) is None  # 2
    assert request(0) == 0  # 3; hit still advances the window
    assert request(2) is None  # 4
    assert request(3) == 3  # 5; second window
    assert materializations == [0, 3]
    cache.set_admission_budget(None)
    assert request(1) == 1
    assert request(2) == 2


def test_failed_attempt_consumes_budget_and_clear_resets_window():
    cache = AdaptiveExpertCache(10, promote_after=1)
    cache.set_admission_budget(1, window_requests=10)
    def oom():
        raise MemoryError('bounded test')
    assert cache.request(key(0), 1, oom) is None
    assert cache.request(key(1), 1, lambda: 1) is None
    assert cache.counters['allocation_failures'] == 1
    cache.clear()
    assert cache.request(key(1), 1, lambda: 1) == 1
    assert cache.statistics()['admission_budget'] == 1


def test_deferred_candidate_does_not_evict_or_bypass_inflight():
    cache = AdaptiveExpertCache(1, promote_after=1)
    cache.request(key(0), 1, lambda: 0)
    cache.set_admission_budget(0)
    for _ in range(100):
        assert cache.request(key(1), 1, lambda: 1) is None
    assert cache.resident[key(0)][0] == 0
    class Fence:
        def query(self):
            return False
    cache.record_inflight([key(0)], Fence())
    cache.set_admission_budget(1)
    assert cache.request(key(1), 1, lambda: 1) is None
    assert cache.resident[key(0)][0] == 0


@pytest.mark.parametrize('limit,window', [(-1, 480), (True, 480), (.5, 480), (1, 0), (1, True)])
def test_invalid_limits(limit, window):
    with pytest.raises(ValueError):
        AdaptiveExpertCache(1).set_admission_budget(limit, window_requests=window)


def test_complete_moe_fallback_under_frozen_placement():
    from mini_llm.backend import xp, BACKEND_NAME
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    rng = np.random.default_rng(260)
    weights = [tuple(rng.normal(0, .1, s).astype(np.float32)
                     for s in ((16, 32), (16, 32), (32, 16))) for _ in range(10)]
    x = rng.normal(0, .1, (1, 16)).astype(np.float32)
    route = np.linspace(.01, .1, 10, dtype=np.float32)
    expected = np.zeros_like(x)
    for p, (g, u, d) in zip(route, weights):
        gate = x @ g
        expected += p * (((gate / (1 + np.exp(-gate))) * (x @ u)) @ d)
    with_executor = HeterogeneousExpertExecutor(lambda k: weights[k.expert_id], model_id='0260',
        d_model=16, d_ff=32, compute_dtype='float32', execution_mode='cpu', cpu_workers=4)
    try:
        with_executor.cache.set_admission_budget(0)
        with with_executor.dispatch_scope(0, tuple(range(10))):
            actual = with_executor.try_forward_decode_batch(0, tuple(range(10)),
                xp.asarray(x), xp.asarray(route), tuple(range(10)))
        if BACKEND_NAME == 'cupy':
            actual = actual.get()
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=2e-5)
        assert with_executor.counters['cpu_submitted'] == 10
    finally:
        with_executor.close()
