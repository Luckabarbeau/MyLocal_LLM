"""Resident-capacity accounting with mocked memGetInfo; no GPU allocations."""
from types import SimpleNamespace

import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.external import qwen4exp_model as runtime
from mini_llm.external import qwen4exp_weights as storage

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='Mocked NumPy-only planner tests')

BUDGET_KEYS = ('host_budget_bytes', 'static_device_budget_bytes', 'device_budget_bytes',
               'cache_bytes', 'headroom_bytes', 'staging_bytes', 'activation_bytes', 'cpu_workers')


@pytest.fixture
def capacity(monkeypatch):
    memory = SimpleNamespace(free=1 << 40, calls=0)

    def mem_info():
        memory.calls += 1
        return memory.free, 1 << 40

    monkeypatch.setattr(storage, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(runtime, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(storage, 'xp', SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=mem_info))))
    monkeypatch.setattr(storage, 'supports_packed_gpu_type', lambda kind: kind == 'packed')
    weights = storage.Qwen4ExpWeights.__new__(storage.Qwen4ExpWeights)
    weights._worker_capacity = 1  # Match constructor-owned snapshot in this mocked planner.
    weights.headers = SimpleNamespace(context=16, width=8, vocab=8, dense_bytes=1000,
        constructor_bytes=128, recurrent=(True, False), ple_layers=(),
        mixer=SimpleNamespace(conv_kernel_size=4, qkv_channels=8, n_value_heads=2, head_dim=4),
        attention=SimpleNamespace(n_kv_heads=1, head_dim=4, index_dim=4,
                                  block_size=4, n_q_heads=2, index_topk=4))
    weights.store = SimpleNamespace(tensors={
        'approved.weight': SimpleNamespace(quant_type='packed', nbytes=100),
        'cpu.weight': SimpleNamespace(quant_type='cpu', nbytes=200)})
    weights.index = SimpleNamespace(d_ff=8)
    weights.plan, weights.device_names = None, frozenset()
    budgets = dict(host_budget_bytes=1 << 20, static_device_budget_bytes=2000,
        device_budget_bytes=1 << 20, cache_bytes=2048, headroom_bytes=4096,
        staging_bytes=0, activation_bytes=0, cpu_workers=1)
    return weights, memory, budgets


def built_model(weights, budgets):
    plan = weights.memory_plan(16, **budgets)
    weights.approve(plan)
    # Only exercise session/planner APIs, not construction or GPU operators.
    mixer = SimpleNamespace(create_state=lambda length: None)
    blocks = [SimpleNamespace(recurrent=rec, mixer=mixer, ple=None) for rec in (True, False)]
    model = runtime.Qwen4ExpGGUFModel.from_components(width=8, vocab=8, blocks=blocks,
        final_hc=None, embedding=None, output=None, max_length=16)
    model.weights, model.plan = weights, plan
    model.expert_executor = SimpleNamespace(cache=SimpleNamespace(used_bytes=0))
    return model


def test_initial_physical_check_remains_strict(capacity):
    weights, memory, budgets = capacity
    plan = weights.memory_plan(16, **budgets)
    memory.free = plan['device_reserved_bytes'] - 1
    with pytest.raises(MemoryError, match=f"need {plan['device_reserved_bytes']}, free {memory.free}"):
        weights.memory_plan(16, **budgets)
    memory.free += 1
    assert weights.memory_plan(16, **budgets) == plan


@pytest.mark.parametrize('cache_used', [0, 512, 2048])
def test_postconstruction_checks_only_owned_remainder(capacity, cache_used):
    weights, memory, budgets = capacity
    model = built_model(weights, budgets)
    model.expert_executor.cache.used_bytes = cache_used
    remaining = model.plan['device_reserved_bytes'] - model.plan['device_static_bytes'] - cache_used
    memory.free = remaining - 1
    with pytest.raises(MemoryError, match=f'need {remaining}, free {memory.free}'):
        model.create_generation_state(1, 16)
    assert not model._states
    memory.free = remaining
    state = model.create_generation_state(1, 16)
    assert state.reserved_bytes == model.plan['state_bytes']
    replanned = model.memory_plan(1, 16)
    assert replanned == model.plan
    assert all(replanned[key] == budgets[key] for key in BUDGET_KEYS
               if key not in ('activation_bytes', 'staging_bytes'))


def test_live_sessions_credited_but_aggregate_reservation_unchanged(capacity):
    weights, memory, budgets = capacity
    model = built_model(weights, budgets)
    first = model.create_generation_state(1, 1)
    small = model.memory_plan(1, 1)
    model.expert_executor.cache.used_bytes = 512
    remaining = (small['device_reserved_bytes'] - model.plan['device_static_bytes'] -
                 512 - first.reserved_bytes)
    memory.free = remaining - 1
    with pytest.raises(MemoryError, match=f'need {remaining}, free {memory.free}'):
        model.create_generation_state(1, 1)
    memory.free = remaining
    second = model.create_generation_state(1, 1)
    assert sum(s.reserved_bytes for s in model._states) == 2 * small['state_bytes']
    assert second is not first
    memory.free = 1 << 40
    with pytest.raises(MemoryError, match='aggregate state reservation'):
        model.create_generation_state(1, 16)
    del first, second
    assert not model._states
    remaining = small['device_reserved_bytes'] - model.plan['device_static_bytes'] - 512
    memory.free = remaining - 1
    with pytest.raises(MemoryError, match=f'need {remaining}, free {memory.free}'):
        model.create_generation_state(1, 1)


def test_resident_credit_does_not_relax_logical_budget(capacity):
    weights, memory, budgets = capacity
    plan = weights.memory_plan(16, **budgets)
    budgets['device_budget_bytes'] = plan['device_reserved_bytes'] - 1
    with pytest.raises(MemoryError, match='device budget'):
        weights.memory_plan(16, resident_device_bytes=plan['device_reserved_bytes'], **budgets)


@pytest.mark.parametrize('resident', [-1, True, 1.5, 1 << 40])
def test_invalid_resident_credit(capacity, resident):
    weights, memory, budgets = capacity
    with pytest.raises(ValueError, match='resident_device_bytes'):
        weights.memory_plan(16, resident_device_bytes=resident, **budgets)


def test_replanning_cannot_expand_approved_placement(capacity, monkeypatch):
    weights, memory, budgets = capacity
    model = built_model(weights, budgets)
    monkeypatch.setattr(storage, 'supports_packed_gpu_type', lambda kind: True)
    replanned = model.memory_plan(1, 16)
    assert replanned == model.plan
    assert replanned['device_names'] == ('approved.weight',)


def test_numpy_model_does_not_credit_host_residency(capacity, monkeypatch):
    weights, memory, budgets = capacity
    monkeypatch.setattr(storage, 'BACKEND_NAME', 'numpy')
    monkeypatch.setattr(runtime, 'BACKEND_NAME', 'numpy')
    budgets['cache_bytes'] = 0
    model = built_model(weights, budgets)
    state = model.create_generation_state(1, 1)
    original = weights.memory_plan
    credits = []

    def tracked(*args, **kwargs):
        credits.append(kwargs['resident_device_bytes'])
        return original(*args, **kwargs)

    monkeypatch.setattr(weights, 'memory_plan', tracked)
    assert model.memory_plan(1, 1)['state_bytes'] == state.reserved_bytes
    assert credits == [0]
    assert memory.calls == 0


def test_orphaned_bound_state_credit_is_owned_deduplicated_and_not_live(capacity,monkeypatch):
    weights,memory,budgets=capacity
    model=built_model(weights,budgets)
    first=model.create_generation_state(1,16)
    history=SimpleNamespace(data=SimpleNamespace(ptr=1000),nbytes=96)
    delta=SimpleNamespace(data=SimpleNamespace(ptr=2000),nbytes=128)
    first.recurrent[0]=SimpleNamespace(conv=SimpleNamespace(history=history),
        delta=SimpleNamespace(memory=delta))
    graph=SimpleNamespace(bound_arrays=lambda:(history,delta))
    segment=SimpleNamespace(graph=graph,graph_statistics=lambda:dict(pool_reserved_bytes=0))
    model._pre_moe_segments={0:segment,1:segment}
    assert model._orphaned_bound_state_bytes()==0
    original=weights.memory_plan; credits=[]
    def tracked(*args,**kwargs):
        credits.append(kwargs['resident_device_bytes'])
        return original(*args,**kwargs)
    monkeypatch.setattr(weights,'memory_plan',tracked)
    assert model.memory_plan(1,16)==model.plan
    assert credits[-1]==model.plan['device_static_bytes']+first.reserved_bytes
    del first
    assert not model._states
    assert model._orphaned_bound_state_bytes()==224
    remaining=model.plan['device_reserved_bytes']-model.plan['device_static_bytes']-224
    memory.free=remaining-1
    with pytest.raises(MemoryError,match=f'need {remaining}, free {remaining-1}'):
        model.memory_plan(1,16)
    memory.free+=1
    assert model.memory_plan(1,16)==model.plan
    assert credits[-1]==model.plan['device_static_bytes']+224
