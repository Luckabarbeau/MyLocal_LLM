"""0106E: on-demand packed CPU execution and safe idle GPU expert eviction."""

import numpy as np
import pytest

from mini_llm.ops.expert_residency import (
    AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor,
)
from mini_llm.ops.native_packed_cpu import packed_cpu_expert, packed_cpu_linear
from mini_llm.ops.quantized_linear import quantized_linear
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def packed(fmt, n, k, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    assert k % qk == 0
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n * k // qk, qb), dtype=np.uint8)
    if fmt == "Q3_K":
        raw[:, 108:110] = np.frombuffer(np.float16(.03125).tobytes(), dtype=np.uint8)
    else:
        raw[:, 0:2] = np.frombuffer(np.float16(.03125).tobytes(), dtype=np.uint8)
        if fmt in ("Q4_K", "Q5_K"):
            raw[:, 2:4] = np.frombuffer(np.float16(.01).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(raw.reshape(-1), (n, k), fmt)


@pytest.mark.parametrize("fmt", list(BLOCK_FORMATS))
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_native_cpu_linear_direct_packed_matches_reference(fmt, dtype):
    k = 512
    w = packed(fmt, 11, k, 42)
    x = np.random.default_rng(10).normal(0, .08, (2, 3, k)).astype(dtype)
    result = quantized_linear(x, w, cpu_packed_backend="native")
    expected = (np.asarray(x, np.float32).reshape(-1, k) @ w.dequantize_reference().T)
    np.testing.assert_allclose(result.reshape(-1, 11), expected.astype(dtype),
                               atol=1e-3 if dtype == np.float16 else 2e-4,
                               rtol=2e-3 if dtype == np.float16 else 3e-5)
    assert result.shape == (2, 3, 11)
    assert result.dtype == dtype
    np.testing.assert_allclose(packed_cpu_linear(x, w), result, rtol=0, atol=0)


@pytest.mark.parametrize("fmts", [
    ("Q3_K", "Q4_K", "Q3_K"),
    ("Q4_K", "Q3_K", "Q4_K"),
    ("Q8_0", "Q4_0", "Q8_0"),
])
def test_native_complete_expert_mixed_formats_matches_fp32_oracle(fmts, monkeypatch):
    d, ff = 256, 256
    weights = tuple(packed(fmt, rows, columns, 70 + i)
                    for i, (fmt, rows, columns) in enumerate(zip(
                        fmts, (ff, ff, d), (d, d, ff))))
    x = np.random.default_rng(15).normal(0, .025, (3, d)).astype(np.float32)
    g, u, down = (w.dequantize_reference() for w in weights)
    gg, uu = x @ g.T, x @ u.T
    expected = ((gg / (1.0 + np.exp(-gg))) * uu) @ down.T
    # Verify the native route does not call the reference decoder once.
    def reject_dequant(self):
        raise AssertionError("Unexpected full FP32 dequantization")
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", reject_dequant)
    got = packed_cpu_expert(x, weights)
    np.testing.assert_allclose(got, expected, atol=1e-2, rtol=2e-4)
    executor = HeterogeneousExpertExecutor(lambda _: weights, model_id="external",
        d_model=d, d_ff=ff, cpu_packed_backend="native", cpu_workers=2,
        host_dequant_budget_bytes=0)
    try:
        got_executor = executor.forward(3, 5, x)
        np.testing.assert_allclose(got_executor, expected, atol=1e-2, rtol=2e-4)
        assert executor.statistics()["execution"]["native_cpu_experts"] == 1
        assert executor.host_cache.used_bytes == 0
    finally:
        executor.close()


def test_native_execution_is_safe_for_concurrent_workers():
    from concurrent.futures import ThreadPoolExecutor
    d = ff = 256
    ws = (packed("Q4_K", ff, d, 101), packed("Q3_K", ff, d, 102),
          packed("Q8_0", d, ff, 103))
    x = np.random.default_rng(77).normal(0, .02, (1, d)).astype(np.float32)
    expected = packed_cpu_expert(x, ws)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: packed_cpu_expert(x, ws), range(12)))
    for output in results:
        np.testing.assert_array_equal(output, expected)


def test_backend_mode_validation():
    w = packed("Q4_0", 4, 32, 2)
    with pytest.raises(ValueError, match="cpu_packed_backend"):
        packed_cpu_linear(np.ones((1, 32), np.float32), w, backend="unknown")
    with pytest.raises(ValueError, match="cpu_packed_backend"):
        HeterogeneousExpertExecutor(lambda _: (), model_id="x", d_model=32,
                                    d_ff=32, cpu_packed_backend="broken")


class FakeEvent:
    def __init__(self):
        self.done = False
        self.syncs = 0
    def query(self):
        return self.done
    def synchronize(self):
        self.done = True
        self.syncs += 1


def test_adaptive_swapping_keeps_selected_and_inflight_experts_safe():
    a, b, c, d = (ExpertKey("80B", 7, i) for i in range(4))
    cache = AdaptiveExpertCache(120, promote_after=1, score_half_life=4,
                                replacement_margin=1.0)
    assert cache.request(a, 60, lambda: "A") == "A"
    assert cache.request(b, 60, lambda: "B") == "B"
    # While old GPU kernels read A and B, the incoming C must execute on CPU.
    event = FakeEvent()
    cache.record_inflight((a, b), event)
    for _ in range(6):
        assert cache.request(c, 60, lambda: "C") is None
    assert set(cache.resident) == {a, b}
    assert cache.statistics()["busy_victim_skips"] >= 1
    # The router may also protect a selected expert before its GPU launch.
    event.done = True
    cache.protect((a,))
    for _ in range(12):
        cache.request(c, 60, lambda: "C")
    assert a in cache.resident
    cache.unprotect((a,))
    # C's repeated requests eventually cause an actual swap.
    for _ in range(12):
        cache.request(c, 60, lambda: "C")
    assert c in cache.resident
    assert cache.used_bytes <= cache.device_budget_bytes
    assert cache.statistics()["swap_evictions"] >= 1
    # Explicit idle cleanup never needs a device-to-host copy.
    for _ in range(6):
        cache.request(d, 200, lambda: "oversized")
    count = cache.evict_idle(5)
    assert count >= 1
    assert cache.used_bytes <= 60


def test_idle_eviction_and_inflight_retention_lifecycle():
    cache = AdaptiveExpertCache(100, promote_after=1, idle_eviction_requests=3)
    x, y = ExpertKey("m", 0, 0), ExpertKey("m", 0, 1)
    assert cache.request(x, 50, lambda: "x") == "x"
    event = FakeEvent()
    cache.record_inflight((x,), event)
    for _ in range(6):
        assert cache.request(y, 200, lambda: None) is None
    assert x in cache.resident
    assert cache.evict_idle(0) == 0
    event.done = True
    assert cache.evict_idle(0) == 1
    assert cache.used_bytes == 0
    assert cache.statistics()["idle_evictions"] == 1
    cache.request(x, 50, lambda: "reloaded")
    pending = FakeEvent()
    cache.record_inflight((x,), pending)
    cache.clear()
    assert pending.syncs == 1
    assert cache.used_bytes == 0


def test_idle_policy_rejects_invalid_ranges_and_never_exceeds_budget():
    with pytest.raises(ValueError):
        AdaptiveExpertCache(16, idle_eviction_requests=-1)
    cache = AdaptiveExpertCache(95, promote_after=1)
    with pytest.raises(ValueError):
        cache.evict_idle(-3)
    with pytest.raises(ValueError):
        cache.evict_idle(2, target_bytes=-1)
    for i in range(20):
        key = ExpertKey("m", i // 4, i % 4)
        for _ in range(12):
            cache.request(key, 40, lambda i=i: str(i))
        assert cache.used_bytes <= 95
    assert cache.statistics()["requests"] == 240
    assert cache.statistics()["misses"] > 0


def test_native_cpu_accepts_memmapped_host_weights(tmp_path):
    w = packed("Q4_K", 12, 256, 199)
    file_path = tmp_path / "expert.q4k.bin"
    file_path.write_bytes(w.data.tobytes())
    mapped = np.memmap(file_path, dtype=np.uint8, mode="r")
    from_mmap = PackedGGUFWeight(mapped, w.shape, w.quant_type)
    x = np.random.default_rng(101).normal(0, .02, (2, 256)).astype(np.float32)
    np.testing.assert_allclose(packed_cpu_linear(x, from_mmap), packed_cpu_linear(x, w),
                               rtol=0, atol=0)
    assert isinstance(from_mmap.data.base, np.memmap) or np.shares_memory(from_mmap.data, mapped)


def test_dispatch_scope_protects_selected_keys_and_releases_after_error():
    executor = HeterogeneousExpertExecutor(lambda _: (), model_id="m",
        d_model=32, d_ff=32, device_budget_bytes=64, promote_after=1)
    a, b = ExpertKey("m", 9, 0), ExpertKey("m", 9, 1)
    assert executor.cache.request(a, 64, lambda: "A") == "A"
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            with executor.dispatch_scope(9, (0, 1)):
                assert executor.cache._protected[a] == 1
                assert executor.cache.request(b, 64, lambda: "B") is None
                raise RuntimeError("synthetic")
        assert not executor.cache._protected
        for _ in range(6):
            executor.cache.request(b, 64, lambda: "B")
        assert b in executor.cache.resident
    finally:
        executor.close()
