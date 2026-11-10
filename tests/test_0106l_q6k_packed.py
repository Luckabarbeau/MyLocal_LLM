"""0106L Q6_K lane layout, packed execution, and lazy Qwen expert parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.native_packed_cpu import packed_cpu_expert, packed_cpu_linear
from mini_llm.ops.quantized_linear import (
    PackedExpertFFNInference, packed_to_device, quantized_linear, quantized_swiglu,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from test_0106f_qwen3next_gguf import Reader
from test_0106e_cpu_packed_and_gpu_swaps import packed


def q6_weight(n, k, seed=10612):
    raw = np.random.default_rng(seed).integers(0, 256, (n * k // 256, 210), dtype=np.uint8)
    raw[:, 208:210] = np.frombuffer(np.float16(1 / 1024).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(raw.ravel(), (n, k), "Q6_K")


def test_q6_handpacked_all_lanes_and_signed_extremes():
    assert BLOCK_FORMATS["Q6_K"] == (256, 210)
    quants = np.tile(np.arange(-32, 32, dtype=np.int16), 4)
    scales = np.array([-128, -64, -32, -16, -1, 0, 1, 2,
                       3, 7, 16, 31, 63, 100, 126, 127], dtype=np.int8)
    raw = bytearray(210)
    # Independently encode the GGML four 32-lane groups in each 128-value half.
    for half in range(2):
        for lane in range(32):
            q = [int(v) + 32 for v in quants[half * 128 + lane::32][:4]]
            raw[half * 64 + lane] = q[0] & 15 | ((q[2] & 15) << 4)
            raw[half * 64 + lane + 32] = q[1] & 15 | ((q[3] & 15) << 4)
            raw[128 + half * 32 + lane] = sum((q[group] >> 4) << (2 * group)
                                             for group in range(4))
    raw[192:208] = scales.tobytes()
    raw[208:210] = np.float16(.03125).tobytes()
    weight = PackedGGUFWeight(np.frombuffer(raw, dtype=np.uint8), (1, 256), "Q6_K")
    expected = .03125 * np.repeat(scales.astype(np.float32), 16) * quants
    np.testing.assert_array_equal(weight.dequantize_reference().ravel(), expected)
    # Identity rows isolate every decoded coefficient in the native GEMV too.
    actual = packed_cpu_linear(np.eye(256, dtype=np.float32), weight)
    np.testing.assert_array_equal(actual.ravel(), expected)


def test_q6_reference_matches_independent_gguf_decoder():
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize
    w = q6_weight(7, 1024)
    independent = dequantize(w.data.reshape(7, -1), gguf.GGMLQuantizationType.Q6_K)
    np.testing.assert_array_equal(w.dequantize_reference(), independent)


def test_q6_mmap_outer_slice_is_zero_copy_and_validated(tmp_path):
    original = q6_weight(12, 512)
    path = tmp_path / "q6.bin"
    path.write_bytes(original.data.tobytes())
    backing = np.memmap(path, dtype=np.uint8, mode="r")
    bank = PackedGGUFWeight(backing, (3, 4, 512), "Q6_K")
    expert = bank.select_outer(1)
    assert expert.shape == (4, 512) and expert.nbytes == 4 * 2 * 210
    assert np.shares_memory(expert.data, backing)
    np.testing.assert_array_equal(expert.dequantize_reference(), bank.dequantize_reference()[1])
    with pytest.raises(ValueError, match="multiple of 256"):
        PackedGGUFWeight(np.zeros(210, dtype=np.uint8), (1, 255), "Q6_K")
    with pytest.raises(ValueError, match="expected"):
        PackedGGUFWeight(np.zeros(209, dtype=np.uint8), (1, 256), "Q6_K")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape", [(1, 512), (2, 3, 1024)])
def test_q6_native_cpu_gemv_matched_fp64(dtype, shape, monkeypatch):
    w = q6_weight(13, shape[-1])
    x = np.random.default_rng(12).normal(0, .04, shape).astype(dtype)
    expected = (x.astype(np.float64) @ w.dequantize_reference().astype(np.float64).T).astype(dtype)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference",
                        lambda _: pytest.fail("Native GEMV must not expand weights"))
    result = quantized_linear(x, w, cpu_packed_backend="native")
    np.testing.assert_allclose(result, expected, rtol=.003 if dtype == np.float16 else 2e-5,
                               atol=.002 if dtype == np.float16 else 2e-5)


@pytest.mark.parametrize("gate_fmt,up_fmt", [("Q6_K", "Q6_K"), ("Q4_K", "Q5_K")])
def test_q6_down_mixed_native_expert_and_executor(gate_fmt, up_fmt, monkeypatch):
    d = f = 256
    make = lambda fmt, seed: q6_weight(f, d, seed) if fmt == "Q6_K" else packed(fmt, f, d, seed)
    weights = (make(gate_fmt, 13), make(up_fmt, 14), q6_weight(d, f, 15))
    x = np.random.default_rng(16).normal(0, .001, (2, d)).astype(np.float32)
    expected = packed_swiglu_reference(x, *weights)["output"]
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference",
                        lambda _: pytest.fail("Native expert must not expand weights"))
    np.testing.assert_allclose(packed_cpu_expert(x, weights), expected, rtol=2e-4, atol=2e-5)
    executor = HeterogeneousExpertExecutor(lambda _: weights, model_id="q6", d_model=d,
        d_ff=f, cpu_packed_backend="native")
    try:
        np.testing.assert_allclose(asnumpy(executor.forward(0, 0, xp.asarray(x))), expected,
                                   rtol=2e-4, atol=2e-5)
    finally:
        executor.close()


@pytest.mark.parametrize("fused", [False, True])
def test_qwen_q6_provider_retains_original_packed_bytes(fused):
    reader = Reader(kind="Q6_K", fused=fused)
    index = Qwen3NextGGUFIndex(reader)
    report = index.report()
    assert report["unsupported_routed_formats"] == {}
    assert report["first_expert_projection_bytes"] == [256 * 210] * 3
    assert not report["full_model_decode_available"]
    weights = index.provider(ExpertKey("q6", 1, 2))
    assert all(w.quant_type == "Q6_K" and w.shape == (256, 256) for w in weights)
    for weight in weights:
        assert any(np.shares_memory(weight.data, tensor.data) for tensor in reader.tensors)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CuPy CUDA")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("block_k", [256, 512, 1024])
def test_gpu_q6_gemv_all_tiles_and_nondefault_stream(dtype, block_k):
    w = q6_weight(13, 768)
    x = np.random.default_rng(17).normal(0, .04, (2, 3, 768)).astype(dtype)
    expected = (x.astype(np.float64) @ w.dequantize_reference().astype(np.float64).T).astype(dtype)
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        actual = quantized_linear(xp.asarray(x), packed_to_device(w), block_k=block_k)
        host = actual.get(stream=stream)
    np.testing.assert_allclose(host, expected, rtol=.003 if dtype == np.float16 else 2e-5,
                               atol=.002 if dtype == np.float16 else 2e-5)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CuPy CUDA")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("formats", [("Q6_K", "Q6_K"), ("Q6_K", "Q4_K"), ("Q5_K", "Q6_K")])
def test_gpu_q6_mixed_fused_gate_up(dtype, formats):
    gate, up = [q6_weight(19, 512, 20 + i) if fmt == "Q6_K" else packed(fmt, 19, 512, 20 + i)
                for i, fmt in enumerate(formats)]
    x = np.random.default_rng(22).normal(0, .001, (2, 512)).astype(dtype)
    expected = packed_swiglu_reference(x, gate, up)["output"]
    actual = quantized_swiglu(xp.asarray(x), packed_to_device(gate), packed_to_device(up))
    np.testing.assert_allclose(asnumpy(actual), expected,
                               rtol=.005 if dtype == np.float16 else 2e-4,
                               atol=.002 if dtype == np.float16 else 2e-5)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CuPy CUDA")
def test_gpu_q6_resident_mixed_expert_never_dequantizes(monkeypatch):
    d = f = 256
    weights = (packed("Q4_K", f, d, 23), packed("Q5_K", f, d, 24), q6_weight(d, f, 25))
    x = np.random.default_rng(26).normal(0, .001, (2, d)).astype(np.float32)
    expected = packed_swiglu_reference(x, *weights)["output"]
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference",
                        lambda _: pytest.fail("Resident expert must remain packed"))
    executor = HeterogeneousExpertExecutor(lambda _: weights, model_id="q6", d_model=d,
        d_ff=f, compute_dtype="float32", promote_after=1,
        device_budget_bytes=sum(w.nbytes for w in weights))
    try:
        for _ in range(2):
            result = asnumpy(executor.forward(0, 0, xp.asarray(x)))
            np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-5)
        assert executor.cache.statistics()["hits"] == 1
        assert executor.cache.used_bytes == sum(w.nbytes for w in weights)
    finally:
        executor.cache.clear()
        executor.close()
