"""0106C fused packed GGUF gate/up GEMV + SwiGLU; CPU and CUDA parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.ops.quantized_linear import (
    PackedExpertFFNInference, packed_to_device, quantized_linear, quantized_swiglu,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def make_quant(fmt, n, k, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    assert k % qk == 0
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n * k // qk, qb), dtype=np.uint8)
    # Finite small scales avoid unstable random activation overflow.
    scale_bytes = np.float16(.025).tobytes()
    if fmt == "Q3_K":
        raw[:, 108:110] = np.frombuffer(scale_bytes, dtype=np.uint8)
    elif fmt == "Q6_K":
        raw[:, 208:210] = np.frombuffer(scale_bytes, dtype=np.uint8)
    elif fmt == "Q4_K":
        raw[:, :2] = np.frombuffer(scale_bytes, dtype=np.uint8)
        raw[:, 2:4] = np.frombuffer(np.float16(.0125).tobytes(), dtype=np.uint8)
    else:
        raw[:, :2] = np.frombuffer(scale_bytes, dtype=np.uint8)
    return PackedGGUFWeight(raw.reshape(-1), (n, k), fmt)


@pytest.mark.parametrize("gate_fmt,up_fmt", [
    ("Q4_K", "Q4_K"), ("Q3_K", "Q3_K"),
    ("Q3_K", "Q4_K"), ("Q8_0", "Q4_0"),
])
@pytest.mark.parametrize("shape", [(2, 256), (2, 3, 256)])
def test_cpu_fused_reference_matches_separate_projections(gate_fmt, up_fmt, shape):
    g = make_quant(gate_fmt, 17, 256, 10)
    u = make_quant(up_fmt, 17, 256, 11)
    rng = np.random.default_rng(12)
    x = rng.normal(0, .2, size=shape).astype(np.float32)
    y = quantized_swiglu(x, g, u)
    ag = quantized_linear(x, g)
    au = quantized_linear(x, u)
    expected = (ag / (1 + np.exp(-ag))) * au
    assert y.shape == shape[:-1] + (17,)
    assert y.dtype == x.dtype
    np.testing.assert_allclose(y, expected, atol=2e-6, rtol=2e-6)


def test_full_packed_expert_and_opt_out_cpu_reference():
    d = f = 256
    g = make_quant("Q3_K", f, d, 101)
    u = make_quant("Q4_K", f, d, 102)
    down = make_quant("Q4_0", d, f, 103)
    x = np.random.default_rng(104).normal(0, .01, (2, d)).astype(np.float32)
    fused = PackedExpertFFNInference(g, u, down, fuse_gate_up=True)
    legacy = PackedExpertFFNInference(g, u, down, fuse_gate_up=False)
    np.testing.assert_allclose(fused.forward(x), legacy.forward(x), rtol=1e-6, atol=1e-6)


def test_quantized_swiglu_rejects_invalid_shapes_and_host_types():
    g = make_quant("Q4_K", 7, 256, 8)
    u = make_quant("Q3_K", 7, 256, 9)
    x = np.ones((2, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="K"):
        quantized_swiglu(x[:, :128], g, u)
    with pytest.raises(ValueError, match="gate/up"):
        quantized_swiglu(x, g, make_quant("Q4_K", 8, 256, 10))
    with pytest.raises(ValueError, match="Expected x"):
        quantized_swiglu(x[0], g, u)
    with pytest.raises(TypeError, match="packed"):
        quantized_swiglu(x, g, np.ones((7, 256), dtype=np.float32))


@pytest.mark.parametrize("gate_fmt,up_fmt", [
    ("Q4_K", "Q4_K"), ("Q3_K", "Q3_K"),
    ("Q3_K", "Q4_K"), ("Q8_0", "Q4_0"),
])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_triton_fused_gate_up_matches_cpu(gate_fmt, up_fmt, dtype):
    if BACKEND_NAME != "cupy":
        pytest.skip("CUDA/CuPy required for Triton fused execution")
    pytest.importorskip("triton")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    d, f = 512, 256
    g = make_quant(gate_fmt, f, d, 171)
    u = make_quant(up_fmt, f, d, 172)
    x = np.random.default_rng(173).normal(0, .07, (2, 3, d)).astype(np.float32)
    # Match device input/output rounding; fused gate/up have no FP16 storage
    # boundary between their FP32 accumulators and the activation.
    expected = packed_swiglu_reference(x, g, u, storage_dtype=dtype)["output"]
    got = quantized_swiglu(cp.asarray(x, dtype=dtype), packed_to_device(g), packed_to_device(u))
    assert got.dtype == dtype
    assert got.shape == (2, 3, f)
    np.testing.assert_allclose(cp.asnumpy(got), expected, rtol=2e-2 if dtype == 'float16' else 3e-4,
                               atol=6e-3 if dtype == 'float16' else 2e-4)


@pytest.mark.parametrize("fmt", ["Q3_K", "Q4_K"])
def test_gpu_fused_expert_compares_to_three_quantized_gemvs(fmt):
    if BACKEND_NAME != "cupy":
        pytest.skip("CUDA/CuPy required")
    pytest.importorskip("triton")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    d = f = 256
    g = packed_to_device(make_quant(fmt, f, d, 221))
    u = packed_to_device(make_quant(fmt, f, d, 222))
    dn = packed_to_device(make_quant(fmt, d, f, 223))
    x = cp.asarray(np.random.default_rng(224).normal(0, .01, (2, d)).astype(np.float32))
    fused = PackedExpertFFNInference(g, u, dn, fuse_gate_up=True)
    unfused = PackedExpertFFNInference(g, u, dn, fuse_gate_up=False)
    np.testing.assert_allclose(cp.asnumpy(fused.forward(x)), cp.asnumpy(unfused.forward(x)),
                               rtol=1e-3, atol=3e-4)
