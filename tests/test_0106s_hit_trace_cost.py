"""A traced cache hit must not enumerate the resident expert population."""

from types import SimpleNamespace

import pytest

import mini_llm.ops.expert_residency as residency

from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor


@pytest.mark.parametrize('mode', ['summary', 'detailed'])
@pytest.mark.parametrize('idle_eviction', [0, 1])
@pytest.mark.parametrize('hit', [False, True])
def test_hit_snapshot_is_constant_cost_but_miss_and_idle_keep_full_observation(mode, idle_eviction, hit, monkeypatch):
    class Residents(dict):
        scans = 0
        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    trace = RoutingTrace(model_id='trace-cost', quantization='F32',
                         expert_mapping='identity', workload='unit', mode=mode)
    executor = HeterogeneousExpertExecutor(lambda _: None, model_id='trace-cost',
        d_model=4, d_ff=4, device_budget_bytes=64,
        routing_profiler=trace, idle_eviction_requests=idle_eviction)
    monkeypatch.setattr(residency, 'BACKEND_NAME', 'cupy')
    key, other = (ExpertKey('trace-cost', 0, i) for i in range(2))
    handle = object()
    residents = Residents({other: (handle, 32)})
    if hit:
        residents[key] = (handle, 32)
    executor.cache.resident = residents
    # Exercise _prepare's observation independently of backend or allocation.
    # The ordinary hit leaves residency unchanged; the slow paths may evict.
    def select(layer, expert_id, x, ticket):
        if not hit or idle_eviction:
            del residents[other]
            executor.cache.counters['evictions'] += 1
        return key, handle if hit else None, None
    executor._select = select
    actual_key, actual_handle, _, _ = executor._prepare(0, 0, SimpleNamespace(shape=(1, 4)))
    assert actual_key == key and (actual_handle is handle) == hit
    assert residents.scans == (0 if hit and not idle_eviction else 1)
    snap = trace.snapshot()
    assert snap['counters'].get('evicted_experts', 0) == int(not hit or bool(idle_eviction))
    assert snap['counters']['gpu_hit_experts' if hit else 'cpu_experts'] == 1
    if mode == 'detailed':
        assert snap['executions'][0]['evicted_experts'] == (
            [{'layer_id': 0, 'expert_id': 1}] if not hit or idle_eviction else [])
    executor.close()
