"""Synthetic-only Patch0213 CUDA packed IQ tests; no checkpoint loading.

Run: MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q
     tests/test_0228_iq_packed_gpu.py
CPU canonical decoders (independently checked against pinned GGML by 0202)
are the oracle. One-hot GEMVs expose every decoded lane without adding a
production full-weight decoder.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import iq_packed_gpu as gpu
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


FORMATS = ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0")


@pytest.fixture(scope="module")
def cp():
    module = pytest.importorskip("cupy")
    try:
        if not module.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA GPU")
    except module.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return module


def random_blocks(fmt, count, seed=228, *, finite_edges=False):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (count, BLOCK_FORMATS[fmt][1]), dtype=np.uint8)
    if finite_edges:
        bits = rng.integers(0, 65536, count, dtype=np.uint16)
        bits[(bits & 0x7c00) == 0x7c00] &= np.uint16(0xfbff)
        edges = np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff], dtype="<u2")
        bits[:min(count, edges.size)] = edges[:min(count, edges.size)]
        scales = bits.astype("<u2").view(np.uint8).reshape(-1, 2)
    else:
        scales = rng.uniform(-.012, .012, count).astype("<f2").view(np.uint8).reshape(-1, 2)
    raw[:, :2] = scales
    return raw


def weight(cp, fmt, n=11, k=768, seed=228):
    qk, _ = BLOCK_FORMATS[fmt]
    host = PackedGGUFWeight(random_blocks(fmt, n * k // qk, seed), (n, k), fmt)
    return host, device_weight(cp, host)


def device_weight(cp, host):
    return SimpleNamespace(data=cp.asarray(host.data), shape=host.shape, quant_type=host.quant_type)


def exhaustive_blocks(fmt):
    count = {"IQ2_S": 1024, "IQ3_S": 512, "IQ3_XXS": 256}.get(fmt, 256)
    raw = random_blocks(fmt, count + 256)
    raw[:, :2] = [0, 60]  # d = 1, exact dyadic lane values.
    # All codebook entries at all byte positions, with varied signs/scales.
    for i in range(count):
        if fmt == "IQ2_S":
            raw[i, 2:34] = i & 255
            raw[i, 66:74] = (i >> 8) * 85
        elif fmt == "IQ3_S":
            raw[i, 2:66] = i & 255
            raw[i, 66:74] = 255 if i >= 256 else 0
        elif fmt == "IQ3_XXS":
            raw[i, 2:66] = i
            aux = np.array([
                sum(((i + g + lane) & 127) << (7 * lane) for lane in range(4))
                | (((i + g) & 15) << 28) for g in range(8)], dtype="<u4")
            raw[i, 66:] = aux.view(np.uint8)
        else:
            raw[i, 2:] = i
    # Exhaust every sign/scale/high-bit/nibble/two-bit byte encoding at every
    # position. Separate codebook fixtures above exhaust the 9/10-bit indices.
    for i in range(256):
        raw[count + i, 2:] = i
    return raw


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("fixture", ["finite-half", "all-encodings"])
def test_every_lane_against_cpu_reference(cp, fmt, fixture):
    raw = (random_blocks(fmt, 257, finite_edges=True) if fixture == "finite-half"
           else exhaustive_blocks(fmt))
    qk, _ = BLOCK_FORMATS[fmt]
    host = PackedGGUFWeight(raw, (len(raw), qk), fmt)
    expected = host.dequantize_reference().T
    x = cp.eye(qk, dtype=cp.float32)
    got = gpu.linear(x, device_weight(cp, host))
    cp.cuda.get_current_stream().synchronize()
    # One-hot multiplication has no dot-product rounding error. This checks
    # all output lanes, including subnormals, signed scales and high bits.
    np.testing.assert_array_equal(cp.asnumpy(got), expected)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("shape", [(2, 256), (2, 3, 768)])
def test_linear_oracle(cp, fmt, dtype, shape):
    host, dev = weight(cp, fmt, k=shape[-1])
    x = np.random.default_rng(229).normal(0, .07, shape).astype(dtype)
    expected = (x.astype(np.float64) @ host.dequantize_reference().astype(np.float64).T).astype(dtype)
    got = gpu.linear(cp.asarray(x), dev)
    assert got.shape == shape[:-1] + (11,)
    assert got.dtype == x.dtype
    np.testing.assert_allclose(cp.asnumpy(got), expected,
                               atol=2e-4 if dtype == "float16" else 3e-6,
                               rtol=2e-3 if dtype == "float16" else 3e-5)


@pytest.mark.parametrize("gate_fmt", FORMATS)
@pytest.mark.parametrize("up_fmt", FORMATS)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_fused_mixed_gate_up(cp, gate_fmt, up_fmt, dtype):
    gh, gate = weight(cp, gate_fmt, seed=230)
    uh, up = weight(cp, up_fmt, seed=231)
    x = np.random.default_rng(232).normal(0, .04, (2, 3, 768)).astype(dtype)
    # Deliberately no FP16 projection storage boundary in the reference.
    g = x.astype(np.float64) @ gh.dequantize_reference().astype(np.float64).T
    u = x.astype(np.float64) @ uh.dequantize_reference().astype(np.float64).T
    expected = ((g / (1 + np.exp(-g))) * u).astype(dtype)
    got = gpu.swiglu(cp.asarray(x), gate, up)
    assert got.shape == (2, 3, 11)
    assert got.dtype == x.dtype
    np.testing.assert_allclose(cp.asnumpy(got), expected,
                               atol=1e-4 if dtype == "float16" else 3e-6,
                               rtol=3e-3 if dtype == "float16" else 5e-5)


@pytest.mark.parametrize("fmt", FORMATS)
def test_fused_expert_with_packed_down_projection(cp, fmt):
    gh, gate = weight(cp, fmt, n=256, k=512, seed=233)
    uh, up = weight(cp, "IQ3_XXS", n=256, k=512, seed=234)
    dh, down = weight(cp, "IQ4_XS", n=17, k=256, seed=235)
    x = np.random.default_rng(236).normal(0, .02, (2, 512)).astype(np.float32)
    g = x.astype(np.float64) @ gh.dequantize_reference().astype(np.float64).T
    u = x.astype(np.float64) @ uh.dequantize_reference().astype(np.float64).T
    h = ((g / (1 + np.exp(-g))) * u).astype(np.float32)
    expected = h.astype(np.float64) @ dh.dequantize_reference().astype(np.float64).T
    got = gpu.linear(gpu.swiglu(cp.asarray(x), gate, up), down)
    np.testing.assert_allclose(cp.asnumpy(got), expected, atol=3e-6, rtol=8e-5)


@pytest.mark.parametrize("block_k", [32, 64, 128, 256, 512, 1024])
def test_tiles_stream_and_no_host_decode_or_transfer(cp, monkeypatch, block_k):
    gh, gate = weight(cp, "IQ2_S", k=768, seed=237)
    uh, up = weight(cp, "IQ4_NL", k=768, seed=238)
    xh = np.random.default_rng(239).normal(0, .03, (2, 768)).astype(np.float32)
    g = xh.astype(np.float64) @ gh.dequantize_reference().astype(np.float64).T
    u = xh.astype(np.float64) @ uh.dequantize_reference().astype(np.float64).T
    expected = (g / (1 + np.exp(-g))) * u
    stream = cp.cuda.Stream(non_blocking=True)
    # Upload on the default stream before entering the independent stream.
    cp.cuda.get_current_stream().synchronize()
    with stream:
        x = cp.empty(xh.shape, cp.float32)
        x.set(xh, stream=stream)
        def forbidden(*args, **kwargs):
            raise AssertionError("packed execution must not decode on host or transfer tables")
        with monkeypatch.context() as patch:
            patch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
            patch.setattr(cp, "asarray", forbidden)
            patch.setattr(cp, "asnumpy", forbidden)
            y = gpu.linear(x, gate, block_k=block_k)
            h = gpu.swiglu(x, gate, up, block_k=block_k)
            # Dependent work must observe the kernels on the same stream.
            result = h + y * cp.float32(0.125)
    stream.synchronize()
    np.testing.assert_allclose(cp.asnumpy(y), g, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(cp.asnumpy(result), expected + g * .125, atol=3e-6, rtol=5e-5)


def test_support_contract():
    assert all(gpu.supports_type(fmt) for fmt in FORMATS)
    assert not any(gpu.supports_type(fmt) for fmt in ("Q8_0", "IQ2_XS", "iq2_s", None, 42))


def test_validation_and_empty_tokens(cp):
    _, dev = weight(cp, "IQ2_S", k=256)
    x = cp.ones((2, 256), cp.float32)
    with pytest.raises(TypeError, match="CuPy"):
        gpu.linear(np.ones((2, 256), np.float32), dev)
    for dtype in (cp.float64, cp.int32):
        with pytest.raises(TypeError, match="float32/float16"):
            gpu.linear(x.astype(dtype), dev)
    for bad in (x[0], x[:, ::2], x.reshape(1, 1, 2, 256)):
        with pytest.raises(ValueError, match="2D/3D"):
            gpu.linear(bad, dev)
    for tile in (0, 16, 33, 2048, True, 512.0):
        with pytest.raises(ValueError, match="block_k"):
            gpu.linear(x, dev, block_k=tile)
    with pytest.raises(ValueError, match="K"):
        gpu.linear(cp.ones((2, 128), cp.float32), dev)
    with pytest.raises(TypeError, match="packed device"):
        gpu.linear(x, object())
    with pytest.raises(NotImplementedError):
        gpu.linear(x, SimpleNamespace(data=dev.data, shape=dev.shape, quant_type="Q8_0"))
    for data, error, message in (
        (np.zeros(dev.data.size, np.uint8), TypeError, "CuPy uint8"),
        (dev.data.astype(cp.int8), TypeError, "CuPy uint8"),
        (dev.data[:-1], ValueError, "byte length"),
        (dev.data[::2], ValueError, "contiguous"),
    ):
        with pytest.raises(error, match=message):
            gpu.linear(x, SimpleNamespace(data=data, shape=dev.shape, quant_type=dev.quant_type))
    _, different = weight(cp, "IQ3_S", n=12, k=256)
    with pytest.raises(ValueError, match="gate/up"):
        gpu.swiglu(x, dev, different)
    for shape in ((0, 256), (2, 0, 256)):
        empty = cp.empty(shape, cp.float32)
        assert gpu.linear(empty, dev).shape == shape[:-1] + (11,)
        assert gpu.swiglu(empty, dev, dev).shape == shape[:-1] + (11,)
