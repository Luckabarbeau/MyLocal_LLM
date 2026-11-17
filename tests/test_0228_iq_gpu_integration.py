"""Bounded public API/cache integration; synthetic packed experts only."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.quantized_linear import (
    SUPPORTED_GPU_PACKED_FORMATS, PackedExpertFFNInference, packed_to_device,
    quantized_linear, quantized_swiglu,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.native_packed_cpu import packed_cpu_expert
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.grouped_quantized_experts import (
    supports_grouped_quantized_experts, grouped_quantized_experts_forward,
)

IQ = ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0")
LEGACY = ("Q8_0", "Q4_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K")


@pytest.fixture(scope="module")
def cp():
    if BACKEND_NAME != "cupy":
        pytest.skip("Run with MINI_LLM_BACKEND=cupy")
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def packed(fmt, n=13, k=256, seed=228):
    qk, qb = BLOCK_FORMATS[fmt]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n * k // qk, qb), dtype=np.uint8)
    offset = {"Q3_K": 108, "Q6_K": 208}.get(fmt, 0)
    raw[:, offset:offset + 2] = np.frombuffer(np.float16(.002).tobytes(), np.uint8)
    if fmt in ("Q4_K", "Q5_K"):
        raw[:, 2:4] = np.frombuffer(np.float16(.001).tobytes(), np.uint8)
    return PackedGGUFWeight(raw, (n, k), fmt)


def swiglu_oracle(x, gate, up):
    g = x.astype(np.float64) @ gate.dequantize_reference().astype(np.float64).T
    u = x.astype(np.float64) @ up.dequantize_reference().astype(np.float64).T
    return ((g / (1 + np.exp(-g))) * u).astype(np.float32)


def forbidden(*args, **kwargs):
    pytest.fail("Integrated packed execution must not decode a full weight")


@pytest.mark.parametrize("fmt", IQ + LEGACY)
@pytest.mark.parametrize("shape", [(2, 256), (2, 3, 768)])
def test_public_linear(cp, fmt, shape, monkeypatch):
    host = packed(fmt, k=shape[-1])
    x = np.random.default_rng(229).normal(0, .04, shape).astype(np.float32)
    expected = x.astype(np.float64) @ host.dequantize_reference().astype(np.float64).T
    dev = packed_to_device(host)
    assert fmt in SUPPORTED_GPU_PACKED_FORMATS
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    got = quantized_linear(cp.asarray(x), dev)
    assert got.dtype == cp.float32 and got.shape == shape[:-1] + (13,)
    np.testing.assert_allclose(cp.asnumpy(got), expected, atol=2e-6, rtol=5e-5)


PAIRS = ([(g, u) for g in IQ for u in IQ]
         + [(g, u) for g in LEGACY for u in IQ]
         + [(g, u) for g in IQ for u in LEGACY])


@pytest.mark.parametrize("gate_fmt,up_fmt", PAIRS)
def test_public_swiglu_and_current_stream(cp, gate_fmt, up_fmt, monkeypatch):
    gate, up = packed(gate_fmt, k=768, seed=230), packed(up_fmt, k=768, seed=231)
    x = np.random.default_rng(232).normal(0, .03, (2, 3, 768)).astype(np.float32)
    expected = swiglu_oracle(x, gate, up)
    gd, ud = packed_to_device(gate), packed_to_device(up)
    cp.cuda.get_current_stream().synchronize()
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        xd = cp.empty(x.shape, cp.float32)
        xd.set(x, stream=stream)
        got = quantized_swiglu(xd, gd, ud)
        dependent = got * cp.float32(1.25) + cp.float32(.125)
    stream.synchronize()
    np.testing.assert_allclose(cp.asnumpy(dependent), expected * np.float32(1.25) + np.float32(.125),
                               atol=2e-6, rtol=5e-5)


@pytest.mark.parametrize("fmt", IQ)
@pytest.mark.parametrize("fused", [True, False])
def test_public_packed_expert(cp, fmt, fused, monkeypatch):
    weights = (packed(fmt, 256, 512, 233), packed("Q8_0", 256, 512, 234),
               packed(fmt, 512, 256, 235))
    x = np.random.default_rng(236).normal(0, .02, (2, 512)).astype(np.float32)
    expected = packed_cpu_expert(x, weights)
    handle = PackedExpertFFNInference(*(packed_to_device(w) for w in weights), fuse_gate_up=fused)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    got = handle.forward(cp.asarray(x))
    np.testing.assert_allclose(cp.asnumpy(got), expected, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("mode", ["cpu", "gpu", "mixed"])
@pytest.mark.parametrize("workers", [0, 2])
def test_executor_top10_misses_promotions_order_and_stream(cp, mode, workers, monkeypatch):
    ids = [511, 9, 400, 51, 0, 301, 127, 80, 500, 255]
    bank = {e: tuple(packed(IQ[(i + p) % len(IQ)], 256, 256, 240 + i * 3 + p)
                     for p in range(3)) for i, e in enumerate(ids)}
    xh = np.random.default_rng(270).normal(0, .05, (1, 256)).astype(np.float32)
    routes = np.arange(1, 11, dtype=np.float32) / np.float32(55)
    native = {e: packed_cpu_expert(xh, bank[e]) for e in ids}
    executor = HeterogeneousExpertExecutor(lambda key: bank[key.expert_id], model_id="0228",
        d_model=256, d_ff=256, compute_dtype="float32", execution_mode=mode,
        device_budget_bytes=sum(w.nbytes for ws in bank.values() for w in ws),
        promote_after=2, cpu_workers=workers, cpu_packed_backend="native")
    managed = ExpertsInference(256, 256, 512, expert_executor=executor)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            x = cp.asarray(xh)
            xd = x.reshape(1, 1, 256)
            rd = cp.asarray(routes.reshape(1, 1, 10))
            ed = cp.asarray(np.array(ids, np.int64).reshape(1, 1, 10))
            if mode == "mixed":
                for e in ids[:5]:
                    executor.forward(0, e, x)  # First use: native CPU fallback.
                    assert ExpertKey("0228", 0, e) not in executor.cache.resident
                    executor.forward(0, e, x)  # Second use: promote packed bytes.
                assert len(executor.cache.resident) == 5
            elif mode == "gpu":
                for e in ids:
                    executor.forward(0, e, x)
                assert len(executor.cache.resident) == 10
            # Exact ordered combine from the same execution tiers as the next
            # managed call; CPU/GPU numerical differences are checked separately.
            per_path = {}
            for e in ids:
                entry = executor.cache.resident.get(ExpertKey("0228", 0, e))
                per_path[e] = cp.asnumpy(entry[0].forward(x)) if entry else native[e]
            ordered = np.zeros_like(xh)
            native_sum = np.zeros_like(xh)
            for e in sorted(ids):
                route = routes[ids.index(e)]
                ordered += np.multiply(per_path[e], route, dtype=np.float32)
                native_sum += np.multiply(native[e], route, dtype=np.float32)
            first = managed.forward(xd, rd, ed)
            first_copy = first.copy()
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(first_copy).reshape(1, 256), ordered)
        np.testing.assert_allclose(cp.asnumpy(first).reshape(1, 256), native_sum, atol=2e-6, rtol=1e-4)
        if mode == "cpu":
            assert not executor.cache.resident
            assert executor.counters["native_cpu_experts"] >= 10
        elif mode == "mixed":
            assert len(executor.cache.resident) == 5  # Cold five used CPU on miss.
            assert executor.cache.counters["cpu_fallbacks"] >= 10
        with stream:
            second = managed.forward(xd, rd, ed)  # Cold five now promote.
            third = managed.forward(xd, rd, ed)
            observed = third + cp.float32(.25)  # Current-stream dependent read.
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(second), cp.asnumpy(third))
        np.testing.assert_allclose(cp.asnumpy(observed).reshape(1, 256), native_sum + np.float32(.25),
                                   atol=2e-6, rtol=1e-4)
        if mode != "cpu":
            assert len(executor.cache.resident) == 10
        assert executor.host_cache.used_bytes == 0 and not managed.experts
        assert not executor.counters["grouped_gpu_dispatches"]
    finally:
        executor.close()
        executor.cache.clear()


@pytest.mark.parametrize("fmt", IQ)
def test_grouped_iq_capability_is_metadata_only(cp, fmt, monkeypatch):
    weights = tuple(packed(fmt, 256, 256, 280 + p) for p in range(3))
    handle = PackedExpertFFNInference(*(packed_to_device(w) for w in weights))
    legacy_down = packed_to_device(packed("Q8_0", 256, 256, 284))
    mixed = PackedExpertFFNInference(handle.gate, handle.up, legacy_down)
    x, routes = cp.ones((1, 256), cp.float32), cp.ones((1,), cp.float32)
    monkeypatch.setattr(cp, "empty", forbidden)
    assert supports_grouped_quantized_experts(x, [handle], routes)
    assert not supports_grouped_quantized_experts(x, [mixed], routes)
    with pytest.raises(ValueError, match="unsupported packed quantization"):
        grouped_quantized_experts_forward(x, [mixed], routes)
