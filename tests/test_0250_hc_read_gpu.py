"""Synthetic three-kernel HC read parity, not parent performance benchmarks."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.ops import hyperconnection_read_gpu as gpu
from test_0232_hc_gpu_decode import HC, oracle, pair


pytestmark = pytest.mark.skipif(xp.__name__ != "cupy", reason="actual CuPy required")


def test_native_dispatch_and_readonly_collapse():
    _, hc, rng = pair(257, 33)
    hc.native_read = True
    h = xp.asarray(rng.normal(size=(4, 257)), dtype=xp.float32)
    expected = read(h, hc)
    actual = hc.read(h)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(asnumpy(a), asnumpy(b))
    hc.inject = None
    np.testing.assert_array_equal(asnumpy(hc.collapse(h)), asnumpy(read(h, hc, False)[0]))
    with pytest.raises(ValueError, match='separate'):
        HC(hc.gamma, hc.down, hc.up, epsilon=hc.epsilon,
           fast_decode=True, native_read=True, graph_decode=True)


def check(actual, expected, label):
    actual = asnumpy(actual)
    expected = np.asarray(asnumpy(expected), dtype=np.float64)
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    delta = actual.astype(np.float64) - expected
    error = float(np.max(np.abs(delta)))
    scale = float(np.max(np.abs(expected)))
    tiny = np.finfo(np.float32).tiny
    relative = error / max(scale, tiny)
    l2 = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), tiny))
    print(f"{label}: maxabs={error:.9g}, maxref={scale:.9g}, "
          f"normwise_rel={relative:.9g}, relativeL2={l2:.9g}")
    assert error <= 2e-6 * scale, (label, error, scale, relative, l2)


def read(h, hc, inject=True):
    return gpu.read_gpu(h, hc.gamma, hc.down, hc.up,
                        hc.inject if inject else None, hc.epsilon)


def bf16_expanded(a):
    # Round synthetic weights to BF16, then retain them as expanded FP32.
    bits = asnumpy(a).view(np.uint32)
    bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return xp.asarray(bits.view(np.float32))


@pytest.mark.parametrize("leading", [(), (1,), (1, 1)])
@pytest.mark.parametrize("d,rank,scale", [(1, 1, 1), (7, 3, .001),
                                          (257, 33, 1), (1024, 32, 1000),
                                          (2560, 320, 1)])
@pytest.mark.parametrize("inject", [False, True])
def test_parity(leading, d, rank, scale, inject):
    _, hc, rng = pair(d, rank, inject=inject, seed=250)
    if d == 2560:
        for name in ("gamma", "down", "up", "inject"):
            if getattr(hc, name) is not None:
                setattr(hc, name, bf16_expanded(getattr(hc, name)))
    h = xp.asarray(rng.normal(0, scale, leading + (4, d)), dtype=xp.float32)
    assert gpu.supports_read_gpu(h, hc.gamma, hc.down, hc.up, hc.inject, hc.epsilon)
    x, g = read(h, hc)
    xr, gr = hc.read(h)
    x64, g64 = oracle(asnumpy(h), hc)
    label = f"D={d} R={rank} leading={leading} inject={inject}"
    check(x, xr, label + " fast_decode mix")
    check(x, x64, label + " FP64 mix")
    if inject:
        check(g, gr, label + " fast_decode gains")
        check(g, g64, label + " FP64 gains")
    else:
        assert g is gr is g64 is None


def test_ownership_fresh_inputs_and_three_launches(monkeypatch):
    _, hc, rng = pair(513, 65)
    h = xp.asarray(rng.normal(size=(1, 4, 513)), dtype=xp.float32)
    arrays = (h, hc.gamma, hc.down, hc.up, hc.inject)
    saved = [asnumpy(a).copy() for a in arrays]
    kernels = gpu._kernels(h.device.id)
    calls = []

    def wrap(i):
        def launch(*args, **kwargs):
            calls.append(i)
            return kernels[i](*args, **kwargs)
        return launch

    monkeypatch.setattr(gpu, "_kernels", lambda device: tuple(wrap(i) for i in range(3)))
    x, g = read(h, hc)
    assert calls == [0, 1, 2]
    assert x.base is None and g.base is None
    assert not xp.shares_memory(x, g)
    for a, snapshot in zip(arrays, saved):
        assert not xp.shares_memory(a, x) and not xp.shares_memory(a, g)
        np.testing.assert_array_equal(asnumpy(a), snapshot)
    saved_x, saved_g = asnumpy(x).copy(), asnumpy(g).copy()
    h *= xp.float32(-.7)
    hc.gamma *= xp.float32(.9)
    x2, g2 = read(h, hc)
    check(x2, oracle(asnumpy(h), hc)[0], "fresh input/weight mix")
    check(g2, oracle(asnumpy(h), hc)[1], "fresh input/weight gains")
    np.testing.assert_array_equal(asnumpy(x), saved_x)
    np.testing.assert_array_equal(asnumpy(g), saved_g)
    assert not xp.shares_memory(x, x2) and not xp.shares_memory(g, g2)
    calls.clear()
    assert read(h, hc, inject=False)[1] is None
    assert calls == [0, 1, 2]


def test_nondefault_stream():
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        _, hc, rng = pair(513, 65)
        h = xp.asarray(rng.normal(size=(1, 1, 4, 513)), dtype=xp.float32)
        h *= xp.float32(-3)
        x, g = read(h, hc)
        consumed_x, consumed_g = x.copy(), g.copy()
        xr, gr = hc.read(h)
    stream.synchronize()
    check(consumed_x, xr, "nondefault mix")
    check(consumed_g, gr, "nondefault gains")


def test_zero_and_saturation():
    a = lambda v: xp.asarray(v, dtype=xp.float32)
    hc = HC(a(np.ones((4, 1))), a([[4000]*4]),
            a([[4000], [-4000], [4000], [-4000]]),
            a([[4000]*4, [-4000]*4]*2), epsilon=1e-5, fast_decode=True)
    for value in (0, 1, -1):
        h = xp.full((4, 1), value, dtype=xp.float32)
        x, g = read(h, hc)
        xr, gr = hc.read(h)
        x64, g64 = oracle(asnumpy(h), hc)
        check(x, xr, f"saturation {value} fast mix")
        check(g, gr, f"saturation {value} fast gains")
        check(x, x64, f"saturation {value} FP64 mix")
        check(g, g64, f"saturation {value} FP64 gains")


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("kind", ["strided", "dtype", "shape", "numpy"])
def test_invalid_arrays_rejected_before_launch(monkeypatch, index, kind):
    _, hc, _ = pair()
    args = [xp.ones((4, 7), dtype=xp.float32), hc.gamma, hc.down, hc.up, hc.inject]
    a = args[index]
    if kind == "strided":
        a = xp.ones(a.shape[:-1] + (a.shape[-1] * 2,), dtype=xp.float32)[..., ::2]
    elif kind == "dtype":
        a = a.astype(xp.float64)
    elif kind == "shape":
        a = xp.ones((2,) + a.shape, dtype=xp.float32)
    else:
        a = asnumpy(a)
    args[index] = a
    monkeypatch.setattr(gpu, "_kernels", lambda *a: pytest.fail("invalid input launched"))
    assert not gpu.supports_read_gpu(*args, hc.epsilon)
    with pytest.raises((ValueError, TypeError)):
        gpu.read_gpu(*args, hc.epsilon)


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf, 1e-50, 1e40])
def test_invalid_epsilon(epsilon):
    _, hc, _ = pair()
    h = xp.ones((4, 7), dtype=xp.float32)
    with pytest.raises(ValueError):
        gpu.read_gpu(h, hc.gamma, hc.down, hc.up, hc.inject, epsilon)


def test_device_scalar_epsilon_no_readback():
    _, hc, _ = pair()
    with pytest.raises(TypeError, match="host scalar"):
        gpu.read_gpu(xp.ones((4, 7), dtype=xp.float32), hc.gamma, hc.down,
                     hc.up, hc.inject, xp.asarray(1e-5, dtype=xp.float32))


def test_wrong_current_device():
    if xp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two devices required")
    _, hc, _ = pair()
    h = xp.ones((4, 7), dtype=xp.float32)
    with xp.cuda.Device((h.device.id + 1) % xp.cuda.runtime.getDeviceCount()):
        assert not gpu.supports_read_gpu(h, hc.gamma, hc.down, hc.up, hc.inject, hc.epsilon)
        with pytest.raises(ValueError, match="current device"):
            read(h, hc)
