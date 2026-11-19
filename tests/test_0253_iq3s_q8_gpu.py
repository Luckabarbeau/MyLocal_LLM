"""Patch0253 IQ3_S only; bounded synthetic and optional real packed slices.

MINI_LLM_BACKEND=cupy PATCH0253_GGUF=/path/to/model.gguf .venv/bin/python -m
pytest -q -s tests/test_0253_iq3s_q8_gpu.py
No full model construction or decoding. Real payload copies stay below 100 MB.
"""
from functools import lru_cache
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import iq3s_q8_gpu as gpu, q8_1_gpu as q8
from mini_llm.ops.quantized_weight import PackedGGUFWeight

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def cuda_required():
    if cp.cuda.runtime.is_hip or not cp.cuda.runtime.getDeviceCount():
        pytest.skip("NVIDIA CUDA device required")


_WRAPPER = r'''
extern "C" __global__ void reference_partials(const unsigned char* w,
        const block_q8_1* q, float* out, int n, int k) {
    int i = blockIdx.x*128 + threadIdx.x;
    if (i >= n*(k/32)) return;
    int row = i/(k/32), g = i%(k/32);
    out[i] = vec_dot_iq3_s_q8_1(w + (long long)row*(k/256)*110,
                               q + (g/8)*8, g/8, 2*(g%8));
}
// Test-only declared tree, not an extracted whole-engine scheduling kernel.
extern "C" __global__ void reference_rows(const unsigned char* w,
        const block_q8_1* q, float* out, int n, int k) {
    int row = blockIdx.x, l = threadIdx.x;
    q += (long long)blockIdx.y*(k/32);
    float v = 0.f;
    for (int g=l; g<k/32; g+=32)
        v += vec_dot_iq3_s_q8_1(w+(long long)row*(k/256)*110,
                                q+(g/8)*8, g/8, 2*(g%8));
    for (int off=16; off; off/=2) v += __shfl_down_sync(0xffffffff,v,off);
    if (!l) out[(long long)blockIdx.y*n+row] = v;
}
'''


@lru_cache(None)
def oracle(device):
    source = (Path(__file__).parent / "fixtures/strata_iq3s_q8_reference.cuh").read_text()
    return cp.RawModule(code=source + _WRAPPER,
                        options=("--std=c++17", "--fmad=false"),
                        name_expressions=("reference_partials", "reference_rows"))


def exact(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32),
                                  err_msg="FP32 graph bit tolerance=0; no finite differences")


def synthetic(n=13, k=1536, exhaustive=False):
    rng = np.random.default_rng(253)
    raw = rng.integers(0, 256, (n*k//256, 110), dtype=np.uint8)
    raw[:, :2] = rng.uniform(-.02, .02, len(raw)).astype('<f2').view(np.uint8).reshape(-1, 2)
    if exhaustive:
        assert k == 256 and n >= 768
        raw[:, :2] = np.array([1], '<f2').view(np.uint8)
        for i in range(512):
            raw[i, 2:66] = i & 255
            raw[i, 66:74] = 255 if i >= 256 else 0
        for i in range(256):
            raw[512+i, 2:] = i  # all signs, high bits, scale-nibble pairs
    return PackedGGUFWeight(raw, (n, k), "IQ3_S")


def device_weight(host):
    return SimpleNamespace(data=cp.asarray(host.data), shape=host.shape, quant_type="IQ3_S")


def partials(w, p, n, k, reference=False):
    out = cp.empty((n, k//32), cp.float32)
    module = oracle(p.device.id) if reference else gpu._module(p.device.id)
    name = "reference_partials" if reference else "partials"
    module.get_function(name)(((out.size+127)//128,), (128,),
                              (w, p, out, np.int32(n), np.int32(k)))
    return out


def row_reference(w, p, n, k):
    out = cp.empty(p.shape[:-1] + (n,), cp.float32)
    m = 1 if p.ndim == 1 else p.shape[0]
    if out.size:
        oracle(p.device.id).get_function("reference_rows")((n, m), (32,),
            (w, p, out, np.int32(n), np.int32(k)))
    return out


def decoded_fp64(host, p):
    # Independent per-weight D(Wq)D(Q8(x)); not an integer-dot graph replica.
    raw = cp.asnumpy(p).reshape(-1, 36)
    d = raw[:, :2].copy().view('<f2').astype(np.float64)
    a = (d * raw[:, 4:].copy().view(np.int8).astype(np.float64)).reshape(-1, host.shape[1])
    w = host.dequantize_reference().astype(np.float64)
    return a @ w.T, np.abs(a) @ np.abs(w).T


def verify(host, x, mode):
    dev = device_weight(host)
    p = q8.quantize_q8_1(x, arithmetic=mode)
    actual = gpu.linear(x, dev, arithmetic=mode)
    exact(actual, row_reference(dev.data, p, *host.shape))
    exact(actual, gpu.linear_q8_1(p, dev.data, host.shape))
    first = p if p.ndim == 1 else p[0]
    if first.size:
        exact(partials(dev.data, first, *host.shape),
              partials(dev.data, first, *host.shape, reference=True))
    baseline, magnitude = decoded_fp64(host, p)
    values = cp.asnumpy(actual).reshape(baseline.shape).astype(np.float64)
    # Dot boundaries plus lane-local and warp sums: forward rounding bound.
    bound = np.finfo(np.float32).eps * (host.shape[1]//1024 + 12) * magnitude + 1e-30
    err = np.abs(values-baseline)
    assert np.all(err <= bound), (err.max(), bound.max())
    rel = err / np.maximum(np.abs(baseline), 1e-30)
    print(f"{host.shape} {mode}: FP32-vs-FP64 max_abs={err.max():.9g}, "
          f"max_rel={rel.max():.9g}; source partial/row bit mismatches=0")
    return actual, baseline


@pytest.mark.parametrize("mode", ["native", "ieee"])
@pytest.mark.parametrize("shape", [(256,), (3, 768), (1, 1024), (4, 2304), (2, 8192)])
def test_synthetic_source_and_fp64(shape, mode):
    host = synthetic(k=shape[-1])
    x = cp.asarray(np.random.default_rng(2531).normal(size=shape).astype(np.float32))
    actual, baseline = verify(host, x, mode)
    # Rounding is expected; exact source parity above distinguishes it from a graph error.
    assert np.any(cp.asnumpy(actual).reshape(baseline.shape).astype(np.float64) != baseline)


@pytest.mark.parametrize("mode", ["native", "ieee"])
def test_all_encodings(mode):
    host = synthetic(n=768, k=256, exhaustive=True)
    x = np.resize(np.array([127, -127, .5, -.5, 1.5, -2.5, 31, -13], np.float32), (3, 256))
    x[1] = np.arange(256, dtype=np.float32)-128
    x[2] = np.roll(x[1], 7)
    verify(host, cp.asarray(x), mode)


@pytest.mark.parametrize("mode", ["native", "ieee"])
def test_half_edges_and_quantizer_inputs_identical(mode, monkeypatch):
    host = synthetic(n=16, k=256)
    raw = host.data.reshape(-1, 110)
    raw[:, :2] = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff], '<u2'), 16).view(np.uint8).reshape(-1, 2)
    x = np.zeros((4, 256), np.float32)
    x[0] = np.resize(np.array([1e20, 1, -1e20, -1], np.float32), 256)
    x[1] = np.arange(1, 257, dtype=np.uint32).view(np.float32)
    x[2] = -0.0
    x[3] = np.resize(np.array([127, .5, -.5, 1.5, -1.5], np.float32), 256)
    x = cp.asarray(x)
    original = q8.quantize_q8_1
    seen = []
    def spy(arg, **kwargs):
        assert arg is x
        assert kwargs["arithmetic"] == mode
        packed = original(arg, **kwargs)
        seen.append(packed)
        return packed
    monkeypatch.setattr(q8, "quantize_q8_1", spy)
    dev = device_weight(host)
    actual = gpu.linear(x, dev, arithmetic=mode)
    expected_p = original(x, arithmetic=mode)
    np.testing.assert_array_equal(cp.asnumpy(seen[0]), cp.asnumpy(expected_p))
    exact(actual, row_reference(dev.data, expected_p, *host.shape))
    exact(partials(dev.data, expected_p[3], *host.shape),
          partials(dev.data, expected_p[3], *host.shape, reference=True))


def test_each_activation_lane():
    host = synthetic(n=7, k=256)
    verify(host, cp.eye(256, dtype=cp.float32), "native")


@pytest.mark.parametrize("mode", ["native", "ieee"])
def test_real_packed_slice(mode):
    path = os.environ.get("PATCH0253_GGUF")
    if not path:
        pytest.skip("set PATCH0253_GGUF to enable bounded real IQ3_S slice")
    from mini_llm.external.gguf_store import _reader_api
    Reader, formats = _reader_api()
    reader = Reader(path, 'r')  # header only, no full tensor payload construction
    found = 0
    for field in reader.header_descriptors:
        dims, typ, offset = field.parts[-3:]
        if formats[int(typ[0])].name != "IQ3_S":
            continue
        k = int(dims[0])
        if k % 256 or len(dims) < 2:
            continue
        n = min(9, int(dims[1]))
        stride = k//256*110
        assert n*stride < 100_000_000 and n*k*8 < 100_000_000
        begin = int(reader.data_offset)+int(offset[0])
        raw = np.array(reader.data[begin:begin+n*stride], copy=True)
        host = PackedGGUFWeight(raw, (n, k), "IQ3_S")
        x = cp.asarray(np.random.default_rng(2532).normal(0, .1, (3, k)).astype(np.float32))
        print(f"real slice: {field.name}, {raw.nbytes} packed bytes")
        verify(host, x, mode)
        found += 1
        if found == 2:
            break
    assert found, "requested GGUF contains no block-aligned IQ3_S tensor"


def test_current_stream_ownership_and_bounds(monkeypatch):
    host = synthetic(n=9, k=1536)
    dev = device_weight(host)
    x = cp.ones((3, 1536), cp.float32)
    before_w, before_x = dev.data.copy(), x.copy()
    storage = cp.full(3*9+8, -123.25, cp.float32)
    out = storage[4:-4].reshape(3, 9)
    expected = gpu.linear(x, dev)
    # Warm both kernels before forbidding host interactions.
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("no host transfer, decode, or synchronization allowed")
    with monkeypatch.context() as m:
        m.setattr(cp, "asnumpy", forbidden)
        m.setattr(cp, "asarray", forbidden)
        m.setattr(cp.cuda.runtime, "deviceSynchronize", forbidden)
        m.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
        with stream:
            x.fill(1)
            assert gpu.linear(x, dev, out=out) is out
            dependent = out.copy()
    stream.synchronize()
    exact(dependent, expected)
    np.testing.assert_array_equal(cp.asnumpy(dev.data), cp.asnumpy(before_w))
    exact(x, before_x)
    np.testing.assert_array_equal(cp.asnumpy(storage[:4]), [-123.25]*4)
    np.testing.assert_array_equal(cp.asnumpy(storage[-4:]), [-123.25]*4)
    assert out.dtype == cp.float32


def test_device_weight_and_zero_rows():
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != "cupy":
        pytest.skip("DevicePackedGGUFWeight requires MINI_LLM_BACKEND=cupy")
    from mini_llm.ops.quantized_linear import DevicePackedGGUFWeight
    host = synthetic(n=5, k=256)
    dev = DevicePackedGGUFWeight(cp.asarray(host.data), host.shape, "IQ3_S")
    x = cp.ones(256, cp.float32)
    exact(gpu.linear(x, dev), gpu.linear(x.reshape(1, -1), dev).reshape(-1))
    assert gpu.linear(cp.empty((0, 256), cp.float32), dev).shape == (0, 5)


@pytest.mark.parametrize("case", ["dtype", "rank", "stride", "k", "shape", "bytes", "weight_dtype",
    "weight_stride", "format", "raw_shape", "out_dtype", "out_shape", "overlap_x",
    "overlap_w", "packed_dtype", "packed_bytes", "mode", "alignment"])
def test_validation(case):
    dev = device_weight(synthetic(n=8, k=256))
    x = cp.ones(256, cp.float32)
    with pytest.raises((TypeError, ValueError)):
        if case == "dtype": gpu.linear(x.astype(cp.float16), dev)
        elif case == "rank": gpu.linear(x.reshape(1, 1, -1), dev)
        elif case == "stride": gpu.linear(cp.ones(512, cp.float32)[::2], dev)
        elif case == "k": gpu.linear(x[:128], dev)
        elif case == "shape": gpu.linear(x, dev.data, (8, 255))
        elif case == "bytes": gpu.linear(x, dev.data[:-1], dev.shape)
        elif case == "weight_dtype": gpu.linear(x, dev.data.astype(cp.int8), dev.shape)
        elif case == "weight_stride": gpu.linear(x, cp.empty(dev.data.size*2, cp.uint8)[::2], dev.shape)
        elif case == "format": gpu.linear(x, SimpleNamespace(data=dev.data, shape=dev.shape, quant_type="IQ3_XXS"))
        elif case == "raw_shape": gpu.linear(x, dev.data)
        elif case == "out_dtype": gpu.linear(x, dev, out=cp.empty(8, cp.float16))
        elif case == "out_shape": gpu.linear(x, dev, out=cp.empty((1, 8), cp.float32))
        elif case == "overlap_x": gpu.linear(x, dev, out=x[:8])
        elif case == "overlap_w": gpu.linear(x, dev, out=dev.data[:32].view(cp.float32))
        elif case == "packed_dtype": gpu.linear_q8_1(cp.empty(288, cp.int8), dev)
        elif case == "packed_bytes": gpu.linear_q8_1(cp.empty(252, cp.uint8), dev)
        elif case == "mode": gpu.linear(x, dev, arithmetic="other")
        elif case == "alignment": gpu.linear(cp.empty(1025, cp.uint8)[1:].view(cp.float32), dev)


def test_device_cache_and_cross_device():
    device = cp.cuda.runtime.getDevice()
    assert gpu._module(device) is gpu._module(device)
    if cp.cuda.runtime.getDeviceCount() > 1:
        x = cp.ones(256, cp.float32)
        dev = device_weight(synthetic(n=2, k=256))
        with cp.cuda.Device((device+1) % cp.cuda.runtime.getDeviceCount()):
            with pytest.raises(ValueError, match="current device"):
                gpu.linear(x, dev)
