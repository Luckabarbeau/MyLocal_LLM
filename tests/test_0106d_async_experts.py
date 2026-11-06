"""0106D: bounded host dequantization plus overlapping managed-MoE dispatch."""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.expert_residency import (
    AdaptiveExpertCache, CPUDequantizedExpertCache,
    ExpertKey, HeterogeneousExpertExecutor,
)
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def make_packed_q4(n, k, seed=0):
    assert k % BLOCK_FORMATS["Q4_0"][0] == 0
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n * k // 32, 18), dtype=np.uint8)
    raw[:, 0:2] = np.frombuffer(np.float16(0.025).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(raw.reshape(-1), (n, k), "Q4_0")


def test_host_dequant_cache_reuses_and_evicts_in_lru_order(monkeypatch):
    cache = CPUDequantizedExpertCache(12_288)  # 3 * 32 * 32 * FP32
    keys = [ExpertKey("checkpoint", 2, i) for i in range(3)]
    weights = tuple(make_packed_q4(32, 32, seed=i) for i in range(3))
    calls = []
    old = PackedGGUFWeight.dequantize_reference

    def track(self):
        calls.append(self.quant_type)
        return old(self)

    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", track)
    first = cache.get(keys[0], weights)
    assert cache.get(keys[0], weights) is first
    assert len(calls) == 3
    assert not first[0].flags.writeable
    cache.get(keys[1], weights)
    assert len(calls) == 6
    assert cache.statistics()["evictions"] == 1
    cache.get(keys[0], weights)
    assert len(calls) == 9
    assert cache.statistics()["resident_experts"] == 1
    assert cache.statistics()["used_bytes"] == 12_288
    cache.clear()
    assert cache.statistics()["used_bytes"] == 0
    with pytest.raises(ValueError):
        CPUDequantizedExpertCache(-1)


def test_host_cache_dequantizes_only_on_first_cpu_expert_execution():
    d = f = 32
    key = ExpertKey("packed", 5, 1)
    w = (make_packed_q4(f, d, 1), make_packed_q4(f, d, 2),
         make_packed_q4(d, f, 3))
    executor = HeterogeneousExpertExecutor(
        lambda _: w, model_id="packed", d_model=d, d_ff=f,
        device_budget_bytes=0, host_dequant_budget_bytes=12_288,
    )
    x = np.random.default_rng(4).normal(0, .01, (2, d)).astype(np.float32)
    first = executor.forward(5, 1, x)
    second = executor.forward(5, 1, x)
    np.testing.assert_array_equal(first, second)
    assert executor.host_cache.statistics()["misses"] == 1
    assert executor.host_cache.statistics()["hits"] == 1
    assert executor.host_cache.used_bytes == 12_288
    executor.close()


def test_async_managed_dispatch_matches_synchronous_and_launches_before_wait():
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU event scheduling test uses the deterministic NumPy backend")
    rng = np.random.default_rng(10604)
    d, f, n = 16, 24, 4
    bank = {ExpertKey("test", 1, i): tuple(
        rng.normal(0, .035, size=shape).astype(np.float32)
        for shape in ((d, f), (d, f), (f, d))
    ) for i in range(n)}
    provider = lambda k: bank[k]
    baseline_executor = HeterogeneousExpertExecutor(provider, model_id="test",
        d_model=d, d_ff=f, compute_dtype="float32")
    async_executor = HeterogeneousExpertExecutor(provider, model_id="test",
        d_model=d, d_ff=f, compute_dtype="float32", cpu_workers=3)
    baseline = ExpertsInference(d, f, n, expert_executor=baseline_executor, layer_id=1)
    managed = ExpertsInference(d, f, n, expert_executor=async_executor, layer_id=1)
    x = rng.normal(0, .3, size=(1, 2, d)).astype(np.float32)
    ids = np.array([[[0, 1], [1, 2]]], dtype=np.int64)
    route = np.array([[[.8, .2], [.65, .35]]], dtype=np.float32)
    expected = baseline.forward(x, route, ids)

    # A synchronous per-expert loop cannot satisfy this barrier. The first
    # two selected CPU experts must start before any result is collected.
    barrier = threading.Barrier(2, timeout=5.0)
    actual_forward = async_executor._cpu_forward_numpy
    started = set()
    lock = threading.Lock()

    def blocked_forward(key, weights, inputs):
        with lock:
            started.add(key.expert_id)
        if key.expert_id in (0, 1):
            barrier.wait()
        return actual_forward(key, weights, inputs)

    async_executor._cpu_forward_numpy = blocked_forward
    try:
        with ThreadPoolExecutor(max_workers=1) as outer:
            result = outer.submit(managed.forward, x, route, ids)
            actual = result.result(timeout=10.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert {0, 1, 2}.issubset(started)
        assert async_executor.statistics()["execution"]["cpu_submitted"] == 3
    finally:
        async_executor.close()
        baseline_executor.close()


def test_async_exception_drains_other_submitted_cpu_workers():
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU error injection test")
    d = f = 4
    weights = tuple(np.eye(d, dtype=np.float32) for _ in range(3))
    executor = HeterogeneousExpertExecutor(lambda _: weights, model_id="m",
                                            d_model=d, d_ff=f, cpu_workers=2)
    original = executor._cpu_forward_numpy
    seen = []

    def compute(key, w, x):
        seen.append(key.expert_id)
        if key.expert_id == 0:
            raise RuntimeError("test worker error")
        return original(key, w, x)

    executor._cpu_forward_numpy = compute
    managed = ExpertsInference(d, f, 2, expert_executor=executor)
    x = np.ones((1, 1, d), np.float32)
    try:
        with pytest.raises(RuntimeError, match="test worker error"):
            managed.forward(x, np.array([[[.5, .5]]]), np.array([[[0, 1]]]))
        assert set(seen) == {0, 1}
    finally:
        executor.close()


def test_cuda_copy_worker_and_mixed_cpu_gpu_experts():
    if BACKEND_NAME != "cupy":
        pytest.skip("Requires CuPy GPU")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    # Keep resident GPU expert 0 hot; expert 1 uses the CPU fallback. The
    # test also validates event ordering across the two CUDA streams.
    rng = np.random.default_rng(10605)
    d, f = 32, 32
    bank = {ExpertKey("mixed", 2, i): tuple(
        rng.normal(0, .02, size=shape).astype(np.float32)
        for shape in ((d, f), (d, f), (f, d))
    ) for i in range(2)}
    # 3 matrices of d*f FP16 bytes: fits exactly one resident expert.
    executor = HeterogeneousExpertExecutor(lambda k: bank[k], model_id="mixed",
        d_model=d, d_ff=f, compute_dtype="float16", cpu_workers=2,
        device_budget_bytes=3*d*f*2, promote_after=2)
    reference = HeterogeneousExpertExecutor(lambda k: bank[k], model_id="mixed",
        d_model=d, d_ff=f, compute_dtype="float16")
    x_np = rng.normal(0, .2, (1, 2, d)).astype(np.float16)
    x = cp.asarray(x_np)
    try:
        # Warm expert 0 to residency; don't submit additional CPU experts.
        for _ in range(2):
            executor.forward(2, 0, x.reshape(-1, d))
        assert ExpertKey("mixed", 2, 0) in executor.cache.resident
        routing_ids = cp.asarray([[[0, 1], [1, 0]]], dtype=cp.int32)
        routing_weights = cp.asarray([[[.6, .4], [.3, .7]]], dtype=cp.float16)
        managed = ExpertsInference(d, f, 2, expert_executor=executor, layer_id=2)
        cpu_result = ExpertsInference(d, f, 2, expert_executor=reference,
                                      layer_id=2)
        got = cp.asnumpy(managed.forward(x, routing_weights, routing_ids))
        expected = cp.asnumpy(cpu_result.forward(x, routing_weights, routing_ids))
        np.testing.assert_allclose(got, expected, rtol=3e-2, atol=1e-3)
        assert executor.counters["cpu_submitted"] >= 1
        assert executor.counters["gpu_experts"] >= 1
    finally:
        executor.close()
        reference.close()
