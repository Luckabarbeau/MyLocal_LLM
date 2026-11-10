"""0106B native GGUF GEMV: CPU oracle and GPU packed-kernel parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.quantized_linear import quantized_linear, packed_to_device
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor


def _packed_weight(fmt, rows, cols, seed=106):
    qk, qb = BLOCK_FORMATS[fmt]
    assert cols % qk == 0
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows * cols // qk, qb), dtype=np.uint8)
    # Use representable, finite and varied scales to exercise the bit decoder.
    if fmt == "Q3_K":
        scales = np.array([0, 56], np.uint8)  # 0.5 float16
        raw[:, 108:110] = scales
    elif fmt == "Q6_K":
        raw[:, 208:210] = np.array([0, 56], np.uint8)
    elif fmt in ("Q4_K", "Q5_K"):
        raw[:, 0:2] = np.array([0, 56], np.uint8)
        raw[:, 2:4] = np.array([0, 52], np.uint8)  # 0.25 float16
    else:
        raw[:, 0:2] = np.array([0, 56], np.uint8)
    return PackedGGUFWeight(raw.reshape(-1), (rows, cols), fmt)


@pytest.mark.parametrize("fmt", list(BLOCK_FORMATS))
@pytest.mark.parametrize("shape", [(3, 512), (2, 3, 512)])
def test_cpu_quantized_linear_matches_reference_and_preserves_output_shape(fmt, shape):
    weight = _packed_weight(fmt, 7, shape[-1])
    rng = np.random.default_rng(103)
    x = rng.normal(size=shape).astype(np.float32)
    output = quantized_linear(x, weight)
    expected = x.reshape(-1, shape[-1]) @ weight.dequantize_reference().T
    assert output.shape == shape[:-1] + (7,)
    assert output.dtype == x.dtype
    np.testing.assert_allclose(output.reshape(-1, 7), expected, rtol=1e-6, atol=2e-5)
    np.testing.assert_array_equal(weight.data, weight.data.copy())


def test_shape_and_format_errors_prevent_expansions():
    w = _packed_weight("Q4_K", 2, 256)
    with pytest.raises(ValueError, match="Input K"):
        quantized_linear(np.zeros((2, 128), np.float32), w)
    with pytest.raises(ValueError, match="Expected x"):
        quantized_linear(np.zeros((256,), np.float32), w)
    with pytest.raises(TypeError, match="NumPy"):
        quantized_linear([[0] * 256], w)
    if BACKEND_NAME != "cupy":
        with pytest.raises(RuntimeError, match="cupy"):
            packed_to_device(w)


def test_packed_gguf_experts_cpu_fallback_and_cache_integration():
    # Real GGUF matrix orientation differs from the native dense FFN: [F,D].
    d_model, d_ff, layer_id = 32, 64, 2
    bank = {}
    for expert in range(3):
        gate = _packed_weight("Q4_0", d_ff, d_model, seed=101 + 3 * expert)
        up = _packed_weight("Q4_0", d_ff, d_model, seed=102 + 3 * expert)
        down = _packed_weight("Q4_0", d_model, d_ff, seed=103 + 3 * expert)
        bank[ExpertKey("coder", layer_id, expert)] = (gate, up, down)
    executor = HeterogeneousExpertExecutor(
        bank.__getitem__, model_id="coder", d_model=d_model, d_ff=d_ff,
        device_budget_bytes=0, compute_dtype="float32",
    )
    rng = np.random.default_rng(107)
    x = rng.normal(size=(2, d_model)).astype(np.float32)
    for expert in range(3):
        out = executor.forward(layer_id, expert, x)
        gate, up, down = bank[ExpertKey("coder", layer_id, expert)]
        g = x @ gate.dequantize_reference().T
        u = x @ up.dequantize_reference().T
        expected = ((g / (1 + np.exp(-g))) * u) @ down.dequantize_reference().T
        np.testing.assert_allclose(out, expected, atol=2e-4, rtol=2e-5)


def test_packed_gguf_expert_mixed_representations_rejected():
    w = _packed_weight("Q4_0", 32, 32)
    executor = HeterogeneousExpertExecutor(
        lambda key: (w, w, np.zeros((32, 32), dtype=np.float32)),
        model_id="m", d_model=32, d_ff=32,
    )
    with pytest.raises(ValueError, match="mixed"):
        executor.forward(0, 0, np.ones((1, 32), dtype=np.float32))


@pytest.mark.parametrize("fmt", list(BLOCK_FORMATS))
@pytest.mark.parametrize("block_k", [256, 512, 1024])
def test_triton_kernel_matches_cpu_oracle(fmt, block_k):
    # Requires a CUDA-capable environment to compile/run Triton. In standard
    # CI this test is skipped; on the user's RTX 5080 it is mandatory.
    if BACKEND_NAME != "cupy":
        pytest.skip("Set MINI_LLM_BACKEND=cupy to validate the real GPU kernel")
    triton = pytest.importorskip("triton")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    k = 512
    weight = _packed_weight(fmt, 13, k, seed=17)
    rng = np.random.default_rng(33)
    x = rng.normal(0, 0.2, size=(2, 3, k)).astype(np.float32)
    reference_weight = weight.dequantize_reference().astype(np.float64)
    qw = packed_to_device(weight)
    assert qw.nbytes == weight.nbytes
    for dtype in (cp.float16, cp.float32):
        # Compare the actual stored input, not the original FP32 random sample.
        expected = (x.astype(dtype).astype(np.float64) @ reference_weight.T).astype(dtype)
        cu_x = cp.asarray(x, dtype=dtype)
        got = quantized_linear(cu_x, qw, block_k=block_k)
        assert got.dtype == dtype
        np.testing.assert_allclose(cp.asnumpy(got), expected,
                                   rtol=3e-3 if dtype == cp.float16 else 2e-5,
                                   atol=2e-1 if dtype == cp.float16 else 2e-3)

@pytest.mark.parametrize("fmt", ["Q3_K", "Q4_K"])
def test_element_addressing_in_gpu_kernel_matches_reference(fmt):
    # Independent scalar transcription of the Triton block offsets. Checks
    # scale-field, nibble-group and high-mask addressing at all 256 positions.
    w = _packed_weight(fmt, 3, 512, seed=704)
    reference = w.dequantize_reference()
    qb = BLOCK_FORMATS[fmt][1]
    for row in range(3):
        for col in range(512):
            block = w.data[(row * 2 + col // 256) * qb:(row * 2 + col // 256 + 1) * qb]
            i = col % 256
            if fmt in ("Q4_K", "Q5_K"):
                d = np.frombuffer(block[0:2].tobytes(), dtype="<f2")[0]
                dm = np.frombuffer(block[2:4].tobytes(), dtype="<f2")[0]
                group = i // 32
                sb = int(block[4 + (group if group < 4 else group + 4)])
                mb = int(block[4 + (group + 4)])
                if group < 4:
                    sc, mn = sb & 63, mb & 63
                else:
                    hsb = int(block[group])
                    hmb = int(block[4 + group])
                    sc = (sb & 15) | ((hsb >> 2) & 48)
                    mn = (mb >> 4) | ((hmb >> 2) & 48)
                q = int(block[16 + (i // 64) * 32 + i % 32])
                v = (q >> ((i % 64) // 32 * 4)) & 15
                got = float(d) * sc * v - float(dm) * mn
            else:
                d = np.frombuffer(block[108:110].tobytes(), dtype="<f2")[0]
                q = int(block[32 + (i // 128) * 32 + i % 32])
                v = (q >> (2 * ((i % 128) // 32))) & 3
                mask = int(block[i % 32])
                signed = v - 4 * (1 - ((mask >> (i // 32)) & 1))
                sg = i // 16
                sb = int(block[96 + sg % 8])
                hb = int(block[104 + sg % 4])
                sc = ((sb >> (4 * (sg // 8))) & 15) | (((hb >> (2 * (sg // 4))) & 3) << 4)
                got = float(d) * (sc - 32) * signed
            assert got == float(reference[row, col]), (fmt, row, col, got, reference[row, col])

@pytest.mark.parametrize("fmt", ["Q3_K", "Q4_K"])
def test_resident_packed_expert_keeps_quantized_bytes_on_gpu(fmt, monkeypatch):
    if BACKEND_NAME != "cupy":
        pytest.skip("CuPy GPU required")
    pytest.importorskip("triton")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    key = ExpertKey("coder", 5, 7)
    d = f = 256
    gate = _packed_weight(fmt, f, d, seed=91)
    up = _packed_weight(fmt, f, d, seed=92)
    down = _packed_weight(fmt, d, f, seed=93)
    # Establish reference first; executing a GPU-cached expert afterwards must
    # never invoke the CPU dequantization machinery.
    xx = np.random.default_rng(111).normal(0, 0.01, (2, d)).astype(np.float32)
    # FP32 BLAS accumulation is not an oracle for cancellation-heavy entries.
    # Promote before accumulation and honor the stored FP32 hidden/output.
    expected = packed_swiglu_reference(xx, gate, up, down)["output"]
    provider_calls = []
    def provider(k):
        provider_calls.append(k)
        return gate, up, down
    executor = HeterogeneousExpertExecutor(
        provider, model_id="coder", d_model=d, d_ff=f,
        device_budget_bytes=gate.nbytes + up.nbytes + down.nbytes,
        promote_after=1, compute_dtype="float32",
    )
    def reject_dequant(self):
        raise AssertionError("GPU expert cache unexpectedly dequantized weights")
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", reject_dequant)
    for _ in range(2):
        got = cp.asnumpy(executor.forward(5, 7, cp.asarray(xx)))
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-3)
    assert provider_calls == [key]
    assert executor.cache.statistics()["resident_experts"] == 1
    assert executor.cache.used_bytes == gate.nbytes + up.nbytes + down.nbytes
