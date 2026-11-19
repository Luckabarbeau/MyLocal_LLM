"""Small, independent CUDA source oracle; no models, weights, counters or downloads.

Native oracle: pinned native_mmvq producer compiled with --use_fast_math.
IEEE oracle: identical pinned source without fast math. CPU equations are used
only on ordinary, unambiguous inputs, not as a substitute for CUDA division/FTZ.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from mini_llm.ops import q8_1_gpu as gpu

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def cuda_required():
    if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("actual NVIDIA CUDA device required")


@lru_cache(maxsize=None)
def oracle(device, mode):
    source = (Path(__file__).parent / "fixtures/strata_q8_1_reference.cuh").read_text()
    options = ("--std=c++17",) + (("--use_fast_math",) if mode == "native" else ())
    return cp.RawKernel(source, "native_quantize_q8_1_kernel", options=options)


def reference(x, mode):
    result = cp.empty(x.shape[:-1] + (x.shape[-1] // 32 * 36,), cp.uint8)
    oracle(x.device.id, mode)(((x.size + 255) // 256,), (256,),
                             (x, result, np.int32(x.size)))
    return result


def exact(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    different = np.flatnonzero(a.ravel() != b.ravel())
    if different.size:
        i = int(different[0])
        pytest.fail(f"byte tolerance=0: {different.size}/{a.size} mismatches; "
                    f"first block={i // 36}, byte={i % 36}, "
                    f"actual={a.ravel()[i]}, oracle={b.ravel()[i]}; "
                    "FP32 producer, half RN metadata, int8 codes; no finite differences")


def pathological(kind):
    x = np.zeros((9, 32), np.float32)
    if kind == "ties":
        x[:] = np.resize(np.array([.5, -.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5], np.float32), x.shape)
        x[:, -1] = 127
        x[1::3, :-1] = np.nextafter(x[1::3, :-1], np.float32(np.inf))
        x[2::3, :-1] = np.nextafter(x[2::3, :-1], np.float32(-np.inf))
    elif kind == "subnormals":
        bits = np.arange(1, 289, dtype=np.uint32).reshape(9, 32)
        x = bits.view(np.float32)
        x[1::2] *= -1
        x[3] = np.finfo(np.float32).tiny
        x[4, 0] = 127  # mixed ordinary and subnormal operands
        x[5] = np.float32(1e-36)  # d32 subnormal, stored d zero
    elif kind == "saturation":
        for i, a in enumerate([65504, 65519, 65520, 65504*127, 65504*128,
                               1e10, -1e10, 1e30, -1e30]):
            x[i] = np.linspace(-abs(a), abs(a), 32, dtype=np.float32)
    elif kind == "sumoverflow":
        for i, a in enumerate([2047, 2048, 65504, 65520, 1e7, -1e7, 1e30, -1e30, 1e38]):
            x[i] = a
    elif kind == "zeros":
        x[1::2] = -0.0
        x[2, ::2] = -0.0
    elif kind == "cancellation":
        x[:] = np.resize(np.array([1e20, 1, -1e20, -1], np.float32), x.shape)
        x[1] = np.roll(x[1], 1)
        x[2, :16] = 1e20
        x[2, 16:] = -1e20
        x[3] = np.resize(np.array([65504, -65504, .001, -.0001], np.float32), 32)
    return x


@pytest.mark.parametrize("mode", ["native", "ieee"])
@pytest.mark.parametrize("kind", ["ties", "subnormals", "saturation", "sumoverflow", "zeros", "cancellation"])
def test_pathological_oracle(kind, mode):
    x = cp.asarray(pathological(kind))
    before = x.copy()
    actual = gpu.quantize_q8_1(x, arithmetic=mode)
    exact(actual, reference(x, mode))
    np.testing.assert_array_equal(cp.asnumpy(x).view(np.uint32), cp.asnumpy(before).view(np.uint32))
    assert actual.dtype == cp.uint8 and actual.shape == (9, 36)


@pytest.mark.parametrize("mode", ["native", "ieee"])
@pytest.mark.parametrize("shape", [(32,), (96,), (3, 288), (2048, 128)])
def test_random_oracle(shape, mode):
    rng = np.random.default_rng(252)
    x = rng.normal(size=shape).astype(np.float32)
    x *= np.exp2(rng.integers(-30, 30, size=shape)).astype(np.float32)
    x = cp.asarray(x)
    exact(gpu.quantize_q8_1(x, arithmetic=mode), reference(x, mode))


@pytest.mark.parametrize("mode", ["native", "ieee"])
def test_near_rounding_boundaries(mode):
    rng = np.random.default_rng(2521)
    scale = rng.uniform(.0001, 10, (4096, 1)).astype(np.float32)
    codes = rng.integers(-126, 126, (4096, 32)).astype(np.float32) + np.float32(.5)
    x = codes * scale
    x[:, 0] = (scale * np.float32(127)).ravel()
    x[::3, 1:] = np.nextafter(x[::3, 1:], np.float32(np.inf))
    x[1::3, 1:] = np.nextafter(x[1::3, 1:], np.float32(-np.inf))
    x = cp.asarray(x)
    exact(gpu.quantize_q8_1(x, arithmetic=mode), reference(x, mode))


def cpu_equations(x):
    """IEEE CPU equations, only for cases far from division ambiguity."""
    blocks = x.reshape(-1, 32)
    a = np.max(np.abs(blocks), axis=1)
    d = np.minimum(a / np.float32(127), np.float32(65504))
    ratio = np.divide(blocks, d[:, None], out=np.zeros_like(blocks), where=d[:, None] != 0)
    # float64 addition avoids introducing an artificial FP32 rounding near .5.
    q = np.copysign(np.floor(np.abs(ratio.astype(np.float64)) + .5), ratio)
    q = np.clip(q, -127, 127).astype(np.int8)
    sums = blocks.copy()
    lanes = np.arange(32)
    for offset in (16, 8, 4, 2, 1):
        sums = sums + sums[:, lanes ^ offset]
    ds = np.stack((d, np.clip(sums[:, 0], -65504, 65504)), axis=1).astype(np.float16)
    packed = np.empty((len(blocks), 36), np.uint8)
    packed[:, :4] = ds.view(np.uint8)
    packed[:, 4:] = q.view(np.uint8)
    return packed


def test_cpu_equations_and_half_bits():
    x = np.zeros((4, 32), np.float32)
    x[0] = np.arange(-16, 16)
    x[0, -1] = 127
    x[1] = x[0] * np.float32(1.0003)  # nonrepresentable half scale
    x[2] = 65504
    expected = cpu_equations(x)
    for mode in ("native", "ieee"):
        np.testing.assert_array_equal(cp.asnumpy(gpu.quantize_q8_1(cp.asarray(x), arithmetic=mode)), expected)
    assert expected[2, 2:4].view(np.float16)[0] == 65504
    assert expected[3, :2].view(np.uint16)[0] == 0


def test_codes_use_fp32_scale_not_half_and_ties_away():
    x = np.zeros((2, 32), np.float32)
    x[:, 0] = np.float32(127 * 1.0003)
    x[:, 1] = np.float32(10.5 * 1.0002)  # below tie with d32, above tie with half d=1
    x[1, 0] = 127
    x[1, 1:7] = [.5, -.5, 1.5, -1.5, 2.5, -2.5]
    actual = cp.asnumpy(gpu.quantize_q8_1(cp.asarray(x)))
    assert actual[0, 5].view(np.int8) == 10
    np.testing.assert_array_equal(actual[1, 5:11].view(np.int8), [1, -1, 2, -2, 3, -3])
    assert actual[0, :2].view(np.float16)[0] == 1


@pytest.mark.parametrize("shape", [(96,), (7, 96), (0, 32)])
def test_dequantize_and_supplied_bounds(shape):
    x = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
    packed_shape = shape[:-1] + (shape[-1] // 32 * 36,)
    nbytes = int(np.prod(packed_shape))
    storage = cp.full(nbytes + 8, 0xA5, cp.uint8)
    out = storage[4:4+nbytes].reshape(packed_shape)
    assert gpu.quantize_q8_1(x, out=out) is out
    decoded = cp.empty_like(x)
    assert gpu.dequantize_q8_1(out, out=decoded) is decoded
    raw = cp.asnumpy(out).reshape(-1, 36)
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    q = raw[:, 4:].copy().view(np.int8).astype(np.float32)
    np.testing.assert_array_equal(cp.asnumpy(decoded), (d*q).reshape(shape))
    np.testing.assert_array_equal(cp.asnumpy(storage[:4]), [0xA5]*4)
    np.testing.assert_array_equal(cp.asnumpy(storage[-4:]), [0xA5]*4)


def test_current_stream_graph_and_no_synchronization(monkeypatch):
    x = cp.empty((3, 96), cp.float32)
    out = cp.empty((3, 108), cp.uint8)
    decoded = cp.empty_like(x)
    # Warm compile outside capture; capture also proves no runtime allocations
    # or host readback/synchronization when caller supplies buffers.
    gpu.quantize_q8_1(x, out=out)
    gpu.dequantize_q8_1(out, out=decoded)
    cp.cuda.get_current_stream().synchronize()
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected host readback/device synchronization")
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(cp.cuda.runtime, "deviceSynchronize", forbidden)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        x.fill(127)
        gpu.quantize_q8_1(x, out=out)
        gpu.dequantize_q8_1(out, out=decoded)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    assert bool(cp.all(decoded == 127))


@pytest.mark.parametrize("case", ["dtype", "rank", "stride", "k", "empty_k", "mode",
                                  "out_dtype", "out_size", "alignment", "overlap",
                                  "packed_size", "packed_dtype"])
def test_validation(case):
    x = cp.ones(32, cp.float32)
    with pytest.raises((TypeError, ValueError)):
        if case == "dtype": gpu.quantize_q8_1(x.astype(cp.float16))
        elif case == "rank": gpu.quantize_q8_1(x.reshape(1, 1, 32))
        elif case == "stride": gpu.quantize_q8_1(cp.ones(64, cp.float32)[::2])
        elif case == "k": gpu.quantize_q8_1(x[:31])
        elif case == "empty_k": gpu.quantize_q8_1(x[:0])
        elif case == "mode": gpu.quantize_q8_1(x, arithmetic="q8_0")
        elif case == "out_dtype": gpu.quantize_q8_1(x, out=cp.empty(36, cp.int8))
        elif case == "out_size": gpu.quantize_q8_1(x, out=cp.empty(72, cp.uint8))
        elif case == "alignment": gpu.quantize_q8_1(x, out=cp.empty(37, cp.uint8)[1:])
        elif case == "overlap": gpu.quantize_q8_1(x, out=x.view(cp.uint8)[:36])
        elif case == "packed_size": gpu.dequantize_q8_1(cp.empty(34, cp.uint8))
        elif case == "packed_dtype": gpu.dequantize_q8_1(cp.empty(36, cp.int8))


def test_per_device_cache():
    device = cp.cuda.runtime.getDevice()
    first = gpu._kernels(device, "native")
    assert gpu._kernels(device, "native") is first
    assert gpu._kernels(device, "ieee") is not first
    if cp.cuda.runtime.getDeviceCount() < 2:
        return  # Actual second-device execution cannot be checked on a one-GPU host.
    x = cp.ones(32, cp.float32)
    other = (device + 1) % cp.cuda.runtime.getDeviceCount()
    with cp.cuda.Device(other):
        with pytest.raises(ValueError, match="current device"):
            gpu.quantize_q8_1(x)
        local = cp.ones(32, cp.float32)
        exact(gpu.quantize_q8_1(local), reference(local, "native"))
        assert gpu._kernels(other, "native") is not first
