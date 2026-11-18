"""Bounded expert mmap warmup and automatic GPU-cache budget contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import expert_host_prefetch as host
from test_0229_resident_budget import capacity, built_model


class Store:
    def __init__(self):
        self.arrays, self.tensors, self.reads = {}, {}, []
        for i, name in enumerate(('blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight',
                                  'blk.0.ffn_down_exps.weight', 'per_layer_token_embd.weight')):
            data = np.arange(12288, dtype=np.uint8)
            data.flags.writeable = False
            self.arrays[name] = data
            self.tensors[name] = SimpleNamespace(name=name, shard_index=0, data_offset=i*16384,
                                                nbytes=data.nbytes)
    def tensor_bytes(self, name):
        self.reads.append(name)
        assert name != 'per_layer_token_embd.weight', 'PLE must remain disk-backed'
        return self.arrays[name]


def test_warmup_uses_readonly_views_not_private_weight_copy(monkeypatch):
    store = Store()
    before = {k:v.tobytes() for k, v in store.arrays.items()}
    monkeypatch.setattr(host, 'memory_info', lambda: dict(MemTotal=1000000, MemAvailable=200000))
    report = host.warm_expert_pages(store, reserve_bytes=4096, chunk_bytes=4096)
    assert report['complete_expert_source']
    assert report['touched_bytes'] == report['total_expert_bytes'] == 3*12288
    assert report['resident_bytes_estimate'] == report['touched_bytes']
    assert report['private_weight_copy_bytes'] == report['pinned_expert_source_bytes'] == 0
    assert all(v.tobytes() == before[k] and not v.flags.writeable for k,v in store.arrays.items())
    assert all('_exps.weight' in name for name in store.reads)


@pytest.mark.parametrize('budget', [0, 1, 4097, 1000000])
def test_explicit_budget_capped_by_available_ram(monkeypatch, budget):
    store = Store()
    monkeypatch.setattr(host, 'memory_info', lambda: dict(MemTotal=1000000, MemAvailable=16384))
    report = host.warm_expert_pages(store, budget_bytes=budget, reserve_bytes=4096, chunk_bytes=4096)
    assert report['planned_bytes'] == min(budget, 12288)
    assert report['touched_bytes'] == report['planned_bytes']
    if not budget:
        assert not store.reads


def test_pressure_stops_warmup_but_keeps_demand_mapping_valid(monkeypatch):
    calls = []
    def memory():
        calls.append(True)
        return dict(MemTotal=1000000, MemAvailable=100000 if len(calls) <= 2 else 0)
    monkeypatch.setattr(host, 'memory_info', memory)
    store = Store()
    report = host.warm_expert_pages(store, reserve_bytes=4096, chunk_bytes=4096)
    assert report['touched_bytes'] == 4096
    assert report['stop_reason'] == 'system RAM reserve reached'
    assert store.arrays['blk.0.ffn_down_exps.weight'].size == 12288


@pytest.mark.parametrize('kwargs', [dict(budget_bytes=True), dict(budget_bytes=-1),
    dict(reserve_bytes=-1), dict(chunk_bytes=0), dict(chunk_bytes=65*1024**2)])
def test_bad_prefetch_budget_fails_before_payload_access(kwargs):
    store = Store()
    with pytest.raises(ValueError):
        host.warm_expert_pages(store, **kwargs)
    assert not store.reads


def test_auto_gpu_cache_uses_remaining_logical_budget(capacity):
    weights, memory, budgets = capacity
    fixed = dict(budgets, cache_bytes=0)
    base = weights.memory_plan(16, **fixed)
    auto = weights.memory_plan(16, **dict(budgets, cache_bytes='auto'))
    slack = base['headroom_bytes'] + base['state_bytes']
    assert auto['cache_bytes'] == budgets['device_budget_bytes']-base['device_reserved_bytes']-slack
    assert auto['device_reserved_bytes'] == budgets['device_budget_bytes']-slack
    assert auto['state_bytes'] == base['state_bytes']
    assert auto['headroom_bytes'] == base['headroom_bytes']


def test_auto_gpu_cache_respects_actual_free_memory(capacity):
    weights, memory, budgets = capacity
    base = weights.memory_plan(16, **dict(budgets, cache_bytes=0))
    slack = base['headroom_bytes'] + base['state_bytes']
    memory.free = base['device_reserved_bytes']+slack+12345
    auto = weights.memory_plan(16, **dict(budgets, cache_bytes='auto'))
    assert auto['cache_bytes'] == 12345
    assert auto['device_reserved_bytes'] == memory.free-slack
    memory.free = base['device_reserved_bytes']-1
    with pytest.raises(MemoryError, match='free device memory'):
        weights.memory_plan(16, **dict(budgets, cache_bytes='auto'))


def test_auto_cache_resolved_once_not_grown_on_replanning(capacity):
    weights, memory, budgets = capacity
    model = built_model(weights, dict(budgets, cache_bytes='auto'))
    approved = model.plan['cache_bytes']
    memory.free *= 2
    assert model.memory_plan(1, 16)['cache_bytes'] == approved


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual GPU slot preallocation')
def test_reserve_gpu_arena_is_idempotent_and_does_not_load_arbitrary_experts():
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    calls = []
    executor = HeterogeneousExpertExecutor(lambda key: calls.append(key), model_id='reserve',
        d_model=256, d_ff=256, compute_dtype='float32', device_budget_bytes=4*65536,
        expert_slot_stride_bytes=65536)
    try:
        assert executor.resident_device_bytes == 0
        first = executor.reserve_device_slots()
        pointer = executor._slot_arena._data.data.ptr
        second = executor.reserve_device_slots()
        assert first == second
        assert executor._slot_arena._data.data.ptr == pointer
        assert first['free_slots'] == first['capacity'] == 4
        assert executor.resident_device_bytes == 4*65536
        assert not executor.cache.resident and not calls
    finally:
        executor.close(); executor.cache.clear()


def test_real_checkpoint_bounded_host_prefetch_only():
    path = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S')
    if not path.exists():
        pytest.skip('local checkpoint absent')
    from mini_llm.external.gguf_store import MultiShardTensorStore
    with MultiShardTensorStore.open(path) as store:
        stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in store.paths]
        report = host.warm_expert_pages(store, budget_bytes=64*1024**2)
        assert report['touched_bytes'] <= 64*1024**2
        assert report['resident_bytes_estimate'] <= report['touched_bytes']
        assert report['private_weight_copy_bytes'] == 0
        assert stamps == [(p.stat().st_size, p.stat().st_mtime_ns) for p in store.paths]
