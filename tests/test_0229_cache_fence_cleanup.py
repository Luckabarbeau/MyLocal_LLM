"""Completed cache fences must be reaped without eviction or synchronization."""
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey


class DoneFence:
    def __init__(self, completed=True):
        self.done = completed

    def synchronize(self):
        pytest.fail("Fence cleanup must not synchronize")


class QueryFence:
    def __init__(self, completed=True):
        self.completed = completed

    def query(self):
        return self.completed

    def synchronize(self):
        pytest.fail("Fence cleanup must not synchronize")


class UnqueryableFence:
    def query(self):
        raise RuntimeError("Cannot query this event")

    def synchronize(self):
        pytest.fail("Fence cleanup must not synchronize")


def resident_cache(count=1):
    cache = AdaptiveExpertCache(1024, promote_after=1)
    keys = tuple(ExpertKey("fences", 0, i) for i in range(count))
    for key in keys:
        assert cache.request(key, 1, object) is not None
    return cache, keys


@pytest.mark.parametrize("fence_type", [DoneFence, QueryFence])
def test_completed_fences_do_not_accumulate_below_budget(fence_type):
    cache, keys = resident_cache()
    for _ in range(1000):
        event = fence_type()
        cache.record_inflight(keys, event)
        # The newly appended group is reaped by the next record operation.
        assert cache._inflight_groups == [(event, frozenset(keys))]
        assert cache.statistics()["inflight_dispatches"] == 1
    cache.record_inflight((), fence_type())
    assert cache.statistics()["inflight_dispatches"] == 0
    assert cache.used_bytes == 1
    assert not cache.counters["evictions"]


@pytest.mark.parametrize("fence_type", [DoneFence, QueryFence])
def test_pending_and_unknown_fences_keep_handles_protected(fence_type):
    cache, keys = resident_cache(4)
    pending = fence_type(False)
    unqueryable = UnqueryableFence()
    unknown = object()
    for key, event in zip(keys, [pending, unqueryable, unknown, DoneFence()]):
        cache.record_inflight((key,), event)
    for _ in range(32):
        cache.record_inflight((keys[3],), DoneFence())
        assert len(cache._inflight_groups) == 4
        assert cache.resident[keys[0]][1] == 1
    assert cache.evict_idle(0) == 1
    assert set(cache.resident) == set(keys[:3])
    if isinstance(pending, DoneFence):
        pending.done = True
    else:
        pending.completed = True
    cache.record_inflight((), DoneFence())
    assert cache._inflight_groups == [
        (unqueryable, frozenset((keys[1],))),
        (unknown, frozenset((keys[2],))),
    ]
    assert cache.evict_idle(0) == 1
    assert set(cache.resident) == set(keys[1:3])


@pytest.mark.gpu
def test_actual_cupy_completed_fences_bounded_across_token_boundaries():
    if BACKEND_NAME != "cupy":
        pytest.skip("Requires MINI_LLM_BACKEND=cupy")
    import cupy as cp

    cache, keys = resident_cache()
    stream = cp.cuda.Stream(non_blocking=True)
    for _ in range(64):
        event = cp.cuda.Event(disable_timing=True)
        event.record(stream)
        # Model the benchmark's synchronized token boundary. Cleanup itself
        # must only query the event, not introduce a synchronization.
        stream.synchronize()
        cache.record_inflight(keys, event)
        assert cache.statistics()["inflight_dispatches"] == 1
    cache.record_inflight((), cp.cuda.Event(disable_timing=True))
    assert cache.statistics()["inflight_dispatches"] == 0
    assert cache.used_bytes == 1
