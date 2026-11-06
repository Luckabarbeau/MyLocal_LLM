"""Reference CPU fallback, routing integration, residency/eviction policy."""

import numpy as np
import pytest

from mini_llm.ops.expert_residency import (
    AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor,
)
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.routing_plan import RoutingPlan


def test_cache_promotes_reused_experts_and_protects_hot_entries():
    cache = AdaptiveExpertCache(80, promote_after=2, score_half_life=8)
    a, b, c = (ExpertKey("model-a", 1, i) for i in range(3))
    def load(k):
        return {"key": k}
    assert cache.request(a, 40, lambda: load(a)) is None
    assert cache.request(a, 40, lambda: load(a)) == load(a)
    assert cache.request(b, 40, lambda: load(b)) is None
    assert cache.request(b, 40, lambda: load(b)) == load(b)
    for _ in range(9):
        assert cache.request(a, 40, lambda: 1) == load(a)
    for _ in range(12):
        cache.request(c, 40, lambda: load(c))
    assert c in cache.resident
    assert a in cache.resident or b in cache.resident  # 2-slot bounded cache
    assert cache.used_bytes == 80
    assert cache.statistics()["evictions"] >= 1
    assert cache.statistics()["hits"] >= 9
    cache.clear()
    assert cache.statistics()["resident_experts"] == 0
    assert cache.used_bytes == 0


def test_cache_handles_device_oom_and_rejects_invalid_request():
    cache = AdaptiveExpertCache(100, promote_after=1)
    key = ExpertKey("a", 0, 0)
    assert cache.request(key, 30, lambda: (_ for _ in ()).throw(MemoryError("oom"))) is None
    assert cache.statistics()["allocation_failures"] == 1
    assert not cache.resident
    with pytest.raises(ValueError):
        cache.request(key, 0, lambda: None)
    with pytest.raises(RuntimeError, match="programming error"):
        cache.request(key, 30, lambda: (_ for _ in ()).throw(RuntimeError("programming error")))


def test_executor_and_sparse_dispatch_match_existing_fully_resident_path():
    rng = np.random.default_rng(106)
    D, F, E = 8, 16, 6
    banks = {
        ExpertKey("external-a", 21, e): (
            rng.normal(size=(D, F)).astype(np.float32) * 0.1,
            rng.normal(size=(D, F)).astype(np.float32) * 0.1,
            rng.normal(size=(F, D)).astype(np.float32) * 0.1,
        ) for e in range(E)
    }
    provider_calls = []
    def host_provider(key):
        provider_calls.append(key)
        return banks[key]
    executor = HeterogeneousExpertExecutor(host_provider, model_id="external-a",
                                           d_model=D, d_ff=F, device_budget_bytes=0,
                                           compute_dtype="float32")
    managed = ExpertsInference(D, F, E, expert_executor=executor, layer_id=21)
    resident = ExpertsInference(D, F, E)
    for i, expert in enumerate(resident.experts):
        expert.set_weight_arrays(*banks[ExpertKey("external-a", 21, i)])
    x = rng.normal(size=(2, 3, D)).astype(np.float32)
    ids = np.array([[[1, 3], [0, 3], [1, 1]], [[0, 1], [5, 3], [1, 0]]])
    weights = np.array([[[.7, .3], [.4, .6], [.2, .8]],
                        [[.6, .4], [.7, .3], [.5, .5]]], np.float32)
    plan = RoutingPlan.from_router_outputs(ids, weights, 2, n_experts=E)
    assert plan.active_expert_ids() == (0, 1, 3, 5)
    np.testing.assert_allclose(managed.forward(x, weights, ids), resident.forward(x, weights, ids),
                               atol=2e-7, rtol=2e-6)
    assert set(provider_calls) == {ExpertKey("external-a", 21, i) for i in (0, 1, 3, 5)}
    assert len(managed.experts) == 0   # no upfront expert-weight allocations
    with pytest.raises(RuntimeError, match="host provider"):
        managed.set_weights(None)


def test_executor_detects_wrong_shapes_and_numerical_dtype():
    key = ExpertKey("m", 1, 2)
    bad = lambda _key: (np.ones((3, 4)), np.ones((4, 4)), np.ones((4, 3)))
    executor = HeterogeneousExpertExecutor(bad, model_id="m", d_model=4, d_ff=4)
    with pytest.raises(ValueError, match="weight"):
        executor.forward(1, 2, np.ones((2, 4), np.float16))
    good = lambda _key: (np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4)))
    executor = HeterogeneousExpertExecutor(good, model_id="m", d_model=4, d_ff=4)
    x = np.ones((2, 4), np.float16)
    result = executor.forward(1, 2, x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    with pytest.raises(ValueError, match="hidden"):
        executor.forward(1, 2, np.ones((2, 5), np.float16))
