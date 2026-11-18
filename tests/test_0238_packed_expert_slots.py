"""Fixed packed slots: exact bytes, view leases, DMA publication and fallback."""
import gc
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.ops import packed_expert_slots as slots
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from test_0228_iq_gpu_integration import packed
from test_0231_grouped_iq_gpu import ACTUAL
from test_0229_resident_budget import capacity, built_model

GPU = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual CuPy required')


def weights(formats=ACTUAL[0], seed=238):
    return tuple(packed(fmt, 256, 256, seed+p) for p, fmt in enumerate(formats))


def stride(ws):
    return (sum(w.nbytes for w in ws)+255)//256*256


@GPU
@pytest.mark.parametrize('formats', ACTUAL)
def test_complete_slot_bytes_and_numerical_parity(formats):
    ws = weights(formats)
    step = stride(ws)
    arena = slots.PackedExpertSlotArena(2*step+255, step)
    expert = arena.materialize(ws)
    reference = PackedExpertFFNInference(*(packed_to_device(w) for w in ws))
    x = xp.asarray(np.random.default_rng(238).normal(0, .1, (1, 256)).astype(np.float32))
    for host, device in zip(ws, (expert.gate, expert.up, expert.down)):
        np.testing.assert_array_equal(asnumpy(device.data), host.data.reshape(-1))
    np.testing.assert_array_equal(asnumpy(expert.forward(x)), asnumpy(reference.forward(x)))
    stats = arena.statistics()
    assert stats['capacity'] == 2 and stats['leased_slots'] == 1
    assert stats['device_reserved_bytes'] == 2*step
    assert stats['transfer_bytes'] == sum(w.nbytes for w in ws)
    assert stats['asynchronous'] is False
    assert expert.gate.data.data.ptr == arena._data.data.ptr
    assert expert.up.data.data.ptr == expert.gate.data.data.ptr + ws[0].nbytes


@GPU
def test_projection_view_retains_lease_and_slot_address_is_reused():
    ws = weights()
    arena = slots.PackedExpertSlotArena(stride(ws), stride(ws))
    expert = arena.materialize(ws)
    pointer = expert.gate.data.data.ptr
    kept_view = expert.gate.data[::2]
    del expert
    gc.collect()
    assert arena.statistics()['free_slots'] == 0
    with pytest.raises(MemoryError, match='leased'):
        arena.materialize(ws)
    del kept_view
    gc.collect()
    assert arena.statistics()['free_slots'] == 1
    next_expert = arena.materialize(ws)
    assert next_expert.gate.data.data.ptr == pointer
    assert arena.statistics()['promotions'] == 2


@GPU
def test_cache_counts_padded_slots_and_held_view_forces_safe_cpu_fallback():
    bank = {0: weights(seed=238), 1: weights(seed=239)}
    step = stride(bank[0])
    executor = HeterogeneousExpertExecutor(lambda key: bank[key.expert_id], model_id='slots',
        d_model=256, d_ff=256, device_budget_bytes=step, expert_slot_stride_bytes=step,
        promote_after=1, compute_dtype='float32', cpu_packed_backend='native', cpu_workers=2)
    stream = xp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            x = xp.ones((1, 256), dtype=xp.float32)*.01
            first = executor.forward(0, 0, x)
        stream.synchronize()
        assert np.isfinite(asnumpy(first)).all()
        key = ExpertKey('slots', 0, 0)
        held = executor.cache.resident[key][0].gate.data[:]
        assert executor.cache.used_bytes == executor.resident_device_bytes == step
        assert executor.cache.evict_idle(0) == 1
        assert executor.cache.used_bytes == 0
        assert executor.resident_device_bytes == step  # Empty arena still owns VRAM.
        with stream:
            fallback = executor.forward(0, 1, x)
        stream.synchronize()
        expected = executor._cpu_forward_numpy(ExpertKey('slots', 0, 1), bank[1], asnumpy(x))
        np.testing.assert_array_equal(asnumpy(fallback), expected)
        assert executor.counters['native_cpu_experts'] == 2  # Fallback plus oracle.
        assert executor.cache.counters['allocation_failures'] == 1
        assert not executor.cache.resident
        del held
        gc.collect()
        with stream:
            resident = executor.forward(0, 1, x)
        stream.synchronize()
        np.testing.assert_allclose(asnumpy(resident), expected, atol=1e-7, rtol=2e-5)
        assert len(executor.cache.resident) == 1
        assert executor.statistics()['slots']['leased_slots'] == 1
    finally:
        executor.close(); executor.cache.clear()


@GPU
def test_copy_complete_before_any_projection_is_published(monkeypatch):
    ws = weights()
    arena = slots.PackedExpertSlotArena(stride(ws), stride(ws))
    stream = arena._stream
    completed = []
    def synchronize():
        stream.synchronize()
        completed.append(True)
    arena._stream = SimpleNamespace(ptr=stream.ptr, synchronize=synchronize)
    original = slots.DevicePackedGGUFWeight
    def validate(*args, **kwargs):
        assert completed and arena.promotions == 0
        return original(*args, **kwargs)
    monkeypatch.setattr(slots, 'DevicePackedGGUFWeight', validate)
    expert = arena.materialize(ws)
    assert len(completed) == 1 and arena.promotions == 1
    np.testing.assert_array_equal(asnumpy(expert.gate.data), ws[0].data.reshape(-1))


@GPU
def test_unknown_transfer_completion_quarantines_slot(monkeypatch):
    ws = weights()
    arena = slots.PackedExpertSlotArena(stride(ws), stride(ws))
    stream = arena._stream
    def fail():
        stream.synchronize()  # Actually retire test DMA; report completion unknown.
        raise RuntimeError('deliberate transfer fence failure')
    arena._stream = SimpleNamespace(ptr=stream.ptr, synchronize=fail)
    with pytest.raises(RuntimeError, match='transfer fence'):
        arena.materialize(ws)
    assert arena.statistics()['quarantined_slots'] == 1
    assert arena.statistics()['promotions'] == 0
    with pytest.raises(MemoryError, match='quarantined'):
        arena.materialize(ws)


@GPU
def test_invalid_layout_and_staging_budget_fail_before_publication():
    ws = weights()
    step = stride(ws)
    with pytest.raises(ValueError, match='aligned'):
        slots.PackedExpertSlotArena(step, step-1)
    with pytest.raises(MemoryError, match='staging'):
        slots.PackedExpertSlotArena(step, step, staging_budget_bytes=1)
    arena = slots.PackedExpertSlotArena(step, step)
    with pytest.raises(ValueError, match='shapes'):
        arena.materialize((ws[0], ws[1], packed('IQ4_NL', 32, 32)))
    assert arena.statistics()['free_slots'] == 1


def test_physical_arena_reservation_credited_separately_from_cached_handles(capacity):
    w, memory, budgets = capacity
    model = built_model(w, budgets)
    model.expert_executor.resident_device_bytes = 2048
    assert model.expert_executor.cache.used_bytes == 0
    remaining = model.plan['device_reserved_bytes'] - model.plan['device_static_bytes'] - 2048
    memory.free = remaining
    assert model.memory_plan(1, 16)['device_reserved_bytes'] == model.plan['device_reserved_bytes']
    memory.free -= 1
    with pytest.raises(MemoryError, match='free device memory'):
        model.memory_plan(1, 16)
