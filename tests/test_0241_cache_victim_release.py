"""Victim sorting must not retain an evicted handle during materialization."""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor
from test_0238_packed_expert_slots import weights, stride


def test_eviction_releases_internal_lease_before_materializing_replacement():
    occupied = set()
    class Handle:
        def __init__(self, key):
            self.key = None
            if occupied:
                raise MemoryError('the old lease is still retained')
            occupied.add(key)
            self.key = key
        def __del__(self):
            if self.key is not None:
                occupied.remove(self.key)
    cache = AdaptiveExpertCache(10, promote_after=1, replacement_margin=1.0)
    a, b = ExpertKey('release', 0, 0), ExpertKey('release', 0, 1)
    assert cache.request(a, 10, lambda: Handle(a)) is not None
    assert occupied == {a}
    assert cache.request(b, 10, lambda: Handle(b)) is not None
    assert occupied == {b}
    assert a not in cache.resident and b in cache.resident
    assert cache.counters['allocation_failures'] == 0
    assert cache.used_bytes == 10
    cache.clear()
    assert not occupied


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual CuPy slot lease')
def test_real_slot_replacement_needs_no_false_cpu_fallback():
    bank = {0: weights(seed=2410), 1: weights(seed=2413)}
    slot = stride(bank[0])
    executor = HeterogeneousExpertExecutor(lambda key: bank[key.expert_id], model_id='release',
        d_model=256, d_ff=256, compute_dtype='float32', cpu_workers=2,
        cpu_packed_backend='native', promote_after=1, device_budget_bytes=slot,
        expert_slot_stride_bytes=slot)
    executor.cache.replacement_margin = 1.0
    try:
        x = xp.asarray(np.random.default_rng(241).normal(0, .1, (1, 256)).astype(np.float32))
        first = asnumpy(executor.forward(0, 0, x))
        assert np.isfinite(first).all()
        pointer = executor.cache.resident[ExpertKey('release', 0, 0)][0].gate.data.data.ptr
        result = asnumpy(executor.forward(0, 1, x))
        assert np.isfinite(result).all()
        key = ExpertKey('release', 0, 1)
        assert key in executor.cache.resident
        assert executor.cache.resident[key][0].gate.data.data.ptr == pointer
        assert executor.cache.counters['allocation_failures'] == 0
        assert executor.counters.get('native_cpu_experts', 0) == 0
        assert executor.statistics()['slots']['promotions'] == 2
    finally:
        executor.close(); executor.cache.clear()
