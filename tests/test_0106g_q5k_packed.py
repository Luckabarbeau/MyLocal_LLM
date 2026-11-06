"""0106G: Q5_K block-format, lazy Qwen expert and CPU/GPU parity tests."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.quantized_linear import quantized_linear, quantized_swiglu, packed_to_device
from mini_llm.ops.native_packed_cpu import packed_cpu_expert, packed_cpu_linear
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from test_0106f_qwen3next_gguf import Reader
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex


def _q5k(n, k, seed):
    assert k % 256 == 0
    raw = np.random.default_rng(seed).integers(0, 256, (n * k // 256, 176), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(.03125).tobytes(), dtype=np.uint8)
    raw[:, 2:4] = np.frombuffer(np.float16(.015625).tobytes(), dtype=np.uint8)
    return PackedGGUFWeight(raw.reshape(-1), (n, k), 'Q5_K')


def test_q5_k_layout_high_bits_and_minima():
    assert BLOCK_FORMATS['Q5_K'] == (256, 176)
    # Only group 0 is nonzero: 32 weights = (.5 * (low + high) - .25)
    raw = bytearray(176)
    raw[:2] = np.float16(.5).tobytes()
    raw[2:4] = np.float16(.25).tobytes()
    raw[4] = 1  # scale group 0 = 1
    raw[8] = 1  # minimum group 0 = 1
    raw[16:48] = bytes([1] * 32)  # high bit set only for group 0
    raw[48:176] = bytes([0x21] * 128)
    weight = PackedGGUFWeight(np.frombuffer(raw, dtype=np.uint8), (1, 256), 'Q5_K')
    decoded = weight.dequantize_reference().ravel()
    np.testing.assert_array_equal(decoded[:32], np.full(32, 8.25))
    np.testing.assert_array_equal(decoded[32:], np.zeros(224))

    # High bit is applied at a different bit location for the last group.
    raw[16:48] = bytes([0x80] * 32)
    raw[4:16] = bytes([0] * 12)
    raw[15] = 0x01  # scales[11]: Q5_K scale group 7 low nibble 1, min 0
    weight = PackedGGUFWeight(np.frombuffer(raw, dtype=np.uint8), (1, 256), 'Q5_K')
    decoded = weight.dequantize_reference().ravel()
    np.testing.assert_array_equal(decoded[:224], np.zeros(224))
    np.testing.assert_array_equal(decoded[224:], np.full(32, .5 * 18))


def test_q5_k_select_outer_still_zero_copy():
    original = _q5k(3 * 4, 512, 49)
    view3d = PackedGGUFWeight(original.data, (3, 4, 512), 'Q5_K')
    sample = view3d.select_outer(1)
    assert sample.shape == (4, 512)
    assert sample.nbytes == 4 * 2 * 176
    assert np.shares_memory(sample.data, original.data)


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_q5_native_cpu_gemv_matches_reference(dtype):
    w = _q5k(15, 512, 617)
    x = np.random.default_rng(16).normal(0, .075, size=(2, 3, 512)).astype(dtype)
    expected = quantized_linear(x, w, cpu_packed_backend='reference')
    native = quantized_linear(x, w, cpu_packed_backend='native')
    np.testing.assert_allclose(native, expected,
                               atol=0.002 if dtype == np.float16 else 0.0002,
                               rtol=0.002 if dtype == np.float16 else 0.00002)
    np.testing.assert_array_equal(native, packed_cpu_linear(x, w))


def test_q5_mixed_quantized_full_cpu_expert_avoids_dequantization(monkeypatch):
    # This reproduces the actual Qwen situation: Q5_K down and Q4_K gate/up.
    from test_0106e_cpu_packed_and_gpu_swaps import packed
    d = ff = 256
    weights = (packed('Q4_K', ff, d, 12), _q5k(ff, d, 13), _q5k(d, ff, 14))
    x = np.random.default_rng(15).normal(0, .02, (2, d)).astype(np.float32)
    g, u, down = [w.dequantize_reference() for w in weights]
    a, b = x @ g.T, x @ u.T
    expected = ((a / (1.0 + np.exp(-a))) * b) @ down.T
    def reject_dequant(self):
        raise AssertionError('Full FP32 matrix expansion is forbidden')
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', reject_dequant)
    got = packed_cpu_expert(x, weights)
    np.testing.assert_allclose(got, expected, atol=0.002, rtol=0.0002)
    executor = HeterogeneousExpertExecutor(lambda _: weights, model_id='qwen3next',
                                           d_model=d, d_ff=ff, device_budget_bytes=0,
                                           cpu_packed_backend='native')
    try:
        got2 = executor.forward(0, 0, x)
        np.testing.assert_allclose(got2, expected, atol=0.002, rtol=0.0002)
    finally:
        executor.close()


def test_qwen3next_q5_provider_accepts_real_style_down_expert():
    reader = Reader(kind='Q5_K', fused=True)
    index = Qwen3NextGGUFIndex(reader)
    report = index.report()
    assert not report['unsupported_routed_formats']
    assert report['first_expert_projection_bytes'] == [256 * 176, 256 * 176, 256 * 176]
    weights = index.expert_weights(0, 0)
    assert all(w.quant_type == 'Q5_K' for w in weights)
    assert all(np.shares_memory(w.data, reader.tensors[0].data) for w in weights[2:])


@pytest.mark.parametrize('shape', [(1, 512), (2, 3, 512)])
def test_gpu_q5_k_packed_gemv_parity(shape):
    if BACKEND_NAME != 'cupy':
        pytest.skip('GPU parity requires MINI_LLM_BACKEND=cupy')
    pytest.importorskip('triton')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA GPU')
    w = _q5k(19, 512, 77)
    x = np.random.default_rng(78).normal(0, .05, shape).astype(np.float32)
    expected = quantized_linear(x, w)
    for dtype in (cp.float16, cp.float32):
        y = quantized_linear(cp.asarray(x, dtype=dtype), packed_to_device(w))
        np.testing.assert_allclose(cp.asnumpy(y), expected,
                                   rtol=.01 if dtype == cp.float16 else .0001,
                                   atol=.04 if dtype == cp.float16 else .001)


def test_gpu_q5_k_fused_gate_up_parity():
    if BACKEND_NAME != 'cupy':
        pytest.skip('GPU parity requires MINI_LLM_BACKEND=cupy')
    pytest.importorskip('triton')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA GPU')
    gate, up = _q5k(17, 512, 700), _q5k(17, 512, 701)
    x = np.random.default_rng(702).normal(0, .03, (2, 512)).astype(np.float32)
    expected = quantized_swiglu(x, gate, up)
    actual = quantized_swiglu(cp.asarray(x), packed_to_device(gate), packed_to_device(up))
    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-3, rtol=3e-4)


def test_optional_gguf_q5_k_independent_parity():
    pytest.importorskip('gguf')
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
    w = _q5k(5, 256, 421)
    expected = dequantize(w.data.reshape(5, 176), GGMLQuantizationType.Q5_K).reshape(5, 256)
    np.testing.assert_array_equal(w.dequantize_reference(), expected)


def test_q5_k_independent_scalar_offsets_all_256_lanes():
    # Separately index each lane following the GGML block field offsets.
    w = _q5k(3, 512, 812)
    expected = w.dequantize_reference()
    for row in range(3):
        for col in range(512):
            b = (row * 2 + col // 256) * 176
            chunk = w.data[b:b + 176]
            i = col % 256
            group = i // 32
            d = float(np.frombuffer(chunk[0:2].tobytes(), dtype='<f2')[0])
            dm = float(np.frombuffer(chunk[2:4].tobytes(), dtype='<f2')[0])
            scb = chunk[4:16]
            if group < 4:
                scale, mn = int(scb[group]) & 63, int(scb[group + 4]) & 63
            else:
                scale = (int(scb[8 + group - 4]) & 15) | ((int(scb[group - 4]) >> 2) & 48)
                mn = (int(scb[8 + group - 4]) >> 4) | ((int(scb[group]) >> 2) & 48)
            qbyte = int(chunk[48 + (i // 64) * 32 + i % 32])
            low = (qbyte >> (((i % 64) // 32) * 4)) & 15
            high = (int(chunk[16 + i % 32]) >> group) & 1
            got = d * scale * (low | (high << 4)) - dm * mn
            assert got == float(expected[row, col]), (row, col, got, expected[row, col])


def test_gpu_mixed_q5_q4_fused_gate_up_parity():
    if BACKEND_NAME != 'cupy':
        pytest.skip('GPU parity requires MINI_LLM_BACKEND=cupy')
    pytest.importorskip('triton')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA GPU')
    from test_0106e_cpu_packed_and_gpu_swaps import packed
    gate, up = packed('Q4_K', 19, 512, 711), _q5k(19, 512, 712)
    x = np.random.default_rng(713).normal(0, .02, (2, 512)).astype(np.float32)
    expected = quantized_swiglu(x, gate, up)
    actual = quantized_swiglu(cp.asarray(x), packed_to_device(gate), packed_to_device(up))
    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-3, rtol=3e-4)
