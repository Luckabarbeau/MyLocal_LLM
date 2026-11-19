"""CPU-only placement tests; log near ties are deliberately not bit-exact.

Run this file as a script for the 2 x 3000 full-pressure CPU microbenchmark.
No model execution, device weights, or GPU are involved.
"""

import gc
import math
import random
import time
import weakref

import pytest

from mini_llm.ops.expert_residency import (
    AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor,
)
from mini_llm.ops.expert_victim_index import ExpertVictimIndex


def key(i):
    return ExpertKey("victims", 0, i)


def coherent(cache):
    assert cache.used_bytes == sum(size for _, size in cache.resident.values())
    assert cache.used_bytes <= cache.device_budget_bytes
    index = cache._victim_index
    if index is not None:
        assert set(index._live) == set(cache.resident)
        assert len(index._heap) <= 2 * len(cache.resident) + 64
        for entry in index._heap:
            assert all(isinstance(v, (int, float, ExpertKey)) for v in entry)


class Fence:
    def __init__(self, done=False, broken=False):
        self.done_value = done
        self.broken = broken
        self.synchronized = False

    def query(self):
        if self.broken:
            raise RuntimeError("unqueryable")
        return self.done_value

    def synchronize(self):
        self.synchronized = True
        self.done_value = True


def test_constructor_defaults_and_executor_forwarding():
    assert AdaptiveExpertCache(10).victim_policy == "sort"
    assert AdaptiveExpertCache(10)._victim_index is None
    with pytest.raises(ValueError, match="victim_policy"):
        AdaptiveExpertCache(10, victim_policy="other")
    for policy in ("sort", "heap"):
        executor = HeterogeneousExpertExecutor(
            lambda k: None, model_id="api", d_model=2, d_ff=3,
            expert_victim_policy=policy,
        )
        assert executor.cache.victim_policy == policy
        executor.close()


@pytest.mark.parametrize("half_life", [17, 193, 8192])
def test_random_representative_requests_match_sort(half_life):
    rng = random.Random(255 + half_life)
    sizes = [rng.randrange(7, 130) for _ in range(140)]
    caches = [AdaptiveExpertCache(1700, promote_after=2, score_half_life=half_life,
                                 victim_policy=p) for p in ("sort", "heap")]
    for step in range(6000):
        i = rng.randrange(140) if step % 3 else rng.randrange(25)
        # Vary requested sizes too: resident hits must retain their original size.
        size = sizes[i] + step % 5
        results = [c.request(key(i), size, lambda: i) for c in caches]
        assert results[0] == results[1]
        assert list(caches[0].resident) == list(caches[1].resident)
        assert caches[0].statistics() == caches[1].statistics()
        for c in caches:
            coherent(c)
    assert caches[1].counters["swap_evictions"] > 100


@pytest.mark.parametrize("policy", ["sort", "heap"])
def test_busy_restore_multivictim_and_failed_admission(policy):
    c = AdaptiveExpertCache(40, promote_after=1, score_half_life=8,
                            replacement_margin=1, victim_policy=policy)
    for i in range(4):
        c.request(key(i), 10, object)
    c.protect((key(0),))
    c.protect((key(0),))
    pending, unknown = Fence(), Fence(broken=True)
    c.record_inflight((key(1),), pending)
    c.record_inflight((key(2),), unknown)
    # Three busy entries mean the candidate list cannot free 20 bytes.
    assert c.request(key(4), 20, object) is None
    assert set(c.resident) == {key(i) for i in range(4)}
    coherent(c)
    c.unprotect((key(0),))
    assert c.request(key(4), 20, object) is None
    c.unprotect((key(0),))
    pending.done_value = True
    for _ in range(10):
        c.request(key(4), 20, object)
        if key(4) in c.resident:
            break
    assert key(4) in c.resident
    assert key(2) in c.resident
    assert c.counters["busy_victim_skips"] > 0
    coherent(c)
    c.clear()
    assert unknown.synchronized
    coherent(c)


@pytest.mark.parametrize("policy", ["sort", "heap"])
def test_exact_admission_threshold_and_oom(policy):
    c = AdaptiveExpertCache(10, promote_after=1, score_half_life=100,
                            victim_policy=policy)
    c.request(key(0), 10, object)
    for _ in range(5):
        c.request(key(0), 10, object)
    assert c.request(key(1), 10, object) is None
    assert key(0) in c.resident
    # Exact current score must still gate admission, not the log priority.
    assert c._usage[key(1)].score / 10 <= c.replacement_margin * c._effective_score(key(0)) / 10
    c.replacement_margin = 1
    for _ in range(20):
        c.request(key(2), 10, lambda: (_ for _ in ()).throw(MemoryError("oom")))
        if c.counters["allocation_failures"]:
            break
    assert c.counters["allocation_failures"] == 1
    assert not c.resident
    coherent(c)
    assert c.request(key(3), 10, object) is not None
    c.evict_idle(0)
    with pytest.raises(RuntimeError, match="programming"):
        c.request(key(4), 10, lambda: (_ for _ in ()).throw(RuntimeError("programming")))
    coherent(c)


@pytest.mark.parametrize("automatic", [False, True])
def test_idle_eviction_clear_and_reinsertion(automatic):
    c = AdaptiveExpertCache(80, promote_after=1, victim_policy="heap",
                            idle_eviction_requests=5 if automatic else 0)
    for i in range(8):
        c.request(key(i), 10, object)
    c.protect((key(7),))
    assert c.evict_idle(0, target_bytes=20) >= 1
    assert key(7) in c.resident
    coherent(c)
    c.unprotect((key(7),))
    c.evict_idle(0)
    assert not c._victim_index._live
    c.clear()
    assert not c._victim_index._heap
    assert not c._usage and c._step == 0
    assert c.request(key(7), 10, object) is not None
    coherent(c)


def test_generations_bound_repeated_updates_and_stable_ties():
    index = ExpertVictimIndex(8)
    # All densities and last steps equal: stable resident insertion order.
    for i in (3, 1, 2):
        index.update(key(i), 1, 10, 4)
    for _ in range(10000):
        index.update(key(1), 1, 10, 4)
        assert len(index._heap) <= 2 * len(index._live) + 64
    with index.victims() as victims:
        assert list(victims) == [key(3), key(1), key(2)]
    with pytest.raises(RuntimeError):
        with index.victims() as victims:
            assert next(victims) == key(3)
            raise RuntimeError("early exit")
    index.remove(key(3))
    index.update(key(3), 1, 10, 4)
    with index.victims() as victims:
        assert list(victims) == [key(1), key(2), key(3)]
    for i in (1, 2, 3):
        index.remove(key(i))
    assert len(index._heap) <= 64


def test_log_ranking_no_exponent_overflow_and_documented_near_ties():
    index = ExpertVictimIndex(1)
    index.update(key(0), 1e300, 10, 100000)
    index.update(key(1), 1e-300, 10, 100000)
    assert all(math.isfinite(entry[0]) for entry in index._heap)
    with index.victims() as victims:
        assert list(victims) == [key(1), key(0)]
    # The next representable score can round to the same log priority.
    # Heap policy promises deterministic ties, NOT bit-exact sort equivalence.
    index.clear()
    index.update(key(0), math.nextafter(1.0, 2.0), 1, 100000)
    index.update(key(1), 1.0, 1, 100000)
    assert index._live[key(0)][0] == index._live[key(1)][0]
    with index.victims() as victims:
        assert list(victims) == [key(0), key(1)]


def test_no_handles_retained_during_replacement_or_clear():
    c = AdaptiveExpertCache(10, promote_after=1, replacement_margin=1,
                            victim_policy="heap")

    class Handle:
        pass

    refs = []

    def load():
        assert not any(ref() is not None for ref in refs)
        handle = Handle()
        refs.append(weakref.ref(handle))
        return handle

    for i in range(200):
        assert c.request(key(i), 10, load) is not None
        coherent(c)
    c.clear()
    gc.collect()
    assert not any(ref() is not None for ref in refs)
    assert not c._victim_index._heap and not c._victim_index._live


def test_no_resident_scan_or_full_score_evaluations_on_normal_request():
    class NoScanDict(dict):
        def __iter__(self):
            pytest.fail("normal heap request scanned resident keys")

        def items(self):
            pytest.fail("normal heap request scanned resident items")

        def values(self):
            pytest.fail("normal heap request scanned resident values")

    c = AdaptiveExpertCache(500, promote_after=1, replacement_margin=1,
                            victim_policy="heap")
    for i in range(500):
        c.request(key(i), 1, object)
    c.resident = NoScanDict(c.resident)
    calls = []
    effective = c._effective_score

    def counted(k):
        calls.append(k)
        return effective(k)

    c._effective_score = counted
    assert c.request(key(0), 1, object) is not None
    assert calls == [key(0)]
    calls.clear()
    assert c.request(key(500), 1, object) is not None
    assert len(calls) == 2  # requested key plus selected victim, not 500 ranks


def benchmark():
    """Time only pressure requests, after filling 2419 resident scalar handles."""
    print("CPU-only: 2419 residents, 3000 full-pressure requests per round")
    for round_id in range(2):
        results = {}
        for policy in (("sort", "heap") if round_id == 0 else ("heap", "sort")):
            c = AdaptiveExpertCache(2419, promote_after=1, replacement_margin=1,
                                    victim_policy=policy)
            for i in range(2419):
                c.request(key(i), 1, lambda: 1)
            start = time.perf_counter()
            for i in range(2419, 5419):
                c.request(key(i), 1, lambda: 1)
            elapsed = time.perf_counter() - start
            results[policy] = elapsed
            coherent(c)
            assert c.counters["evictions"] == 3000
            print(f"round={round_id + 1} policy={policy} seconds={elapsed:.6f} "
                  f"evictions={c.counters['evictions']}")
        print(f"round={round_id + 1} speedup={results['sort'] / results['heap']:.2f}x")


if __name__ == "__main__":
    benchmark()
