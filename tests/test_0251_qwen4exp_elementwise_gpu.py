"""Exact CuPy expression parity; every tested array is at most 1M elements."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.ops import qwen4exp_elementwise_gpu as gpu


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(xp.__name__ != "cupy", reason="actual CuPy required")]
_MAX_ELEMENTS = 1_000_000


def sigmoid(v):
    z = xp.exp(-xp.abs(v))
    one = xp.float32(1)
    return xp.where(v >= 0, one / (one + z), z / (one + z))


def references(gate, up, routed, scalar, dt, ssm_a):
    return ((gate * sigmoid(gate)) * up,
            routed + sigmoid(scalar.reshape(())) * up,
            xp.exp(-xp.logaddexp(xp.float32(0), -gate)),
            ssm_a * xp.logaddexp(xp.float32(0), gate + dt))


def exact(actual, reference, label):
    assert actual.dtype == xp.float32 and actual.shape == reference.shape
    assert actual.size <= _MAX_ELEMENTS
    a, r = asnumpy(actual), asnumpy(reference)
    # Bitwise comparison also checks signed zero; NaN payloads are not a contract.
    same = (a.view(np.uint32) == r.view(np.uint32)) | (np.isnan(a) & np.isnan(r))
    if not np.all(same):
        index = int(np.flatnonzero(~same)[0])
        av, rv = float(a.ravel()[index]), float(r.ravel()[index])
        finite = np.isfinite(a) & np.isfinite(r)
        errors = np.abs(a[finite].astype(np.float64) - r[finite].astype(np.float64))
        relative = errors / np.maximum(np.abs(r[finite].astype(np.float64)),
                                      np.finfo(np.float32).tiny)
        pytest.fail(f"{label}: FP32 exact tolerance=0; index={index}, kernel={av!r}, "
                    f"reference={rv!r}, abs_error={abs(av-rv)!r}, "
                    f"max_abs={errors.max(initial=0)!r}, "
                    f"max_relative={relative.max(initial=0)!r}; no finite differences")


def exercise(gate, up, routed, scalar, dt, ssm_a):
    inputs = (gate, up, routed, scalar, dt, ssm_a)
    snapshots = [asnumpy(a).copy() for a in inputs]
    expected = references(*inputs)
    beta, decay = gpu.delta_coefficients(gate, gate, dt, ssm_a)
    actual = (gpu.shared_hidden(gate, up), gpu.combine_shared(routed, up, scalar),
              beta, decay)
    for name, a, r in zip(("hidden", "combined", "beta", "log_decay"), actual, expected):
        exact(a, r, name)
        assert a.device.id == gate.device.id
        for source in inputs:
            # CuPy 14.2 shares_memory falsely aliases independent 0-D arrays.
            # Disjoint allocation bounds are sufficient for these fresh outputs.
            assert not xp.may_share_memory(a, source)
    assert not xp.may_share_memory(beta, decay)
    for source, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(asnumpy(source), snapshot)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "reverse"])
def test_random_and_strides(layout):
    rng = np.random.default_rng(251)
    arrays = [xp.asarray(rng.normal(0, 12, (17, 38)), dtype=xp.float32)
              for _ in range(3)]
    if layout == "transpose":
        arrays = [a.T for a in arrays]
    elif layout == "slice":
        arrays = [a[:, ::2] for a in arrays]
    elif layout == "reverse":
        arrays = [a[::-1, ::-1] for a in arrays]
    heads = arrays[0].shape[-1]
    dt = xp.asarray(rng.normal(size=heads * 2), dtype=xp.float32)[::2]
    ssm_a = -xp.asarray(rng.uniform(.01, 5, heads * 2), dtype=xp.float32)[::-2]
    # A noncontiguous singleton must be passed without copying/readback.
    scalar = xp.asarray([99, -1.25, 88], dtype=xp.float32)[1:2:2]
    exercise(*arrays, scalar, dt, ssm_a)


@pytest.mark.parametrize("scalar", [-1000, -100, -1, -0.0, 0.0, 1, 100, 1000])
def test_sign_zero_and_saturation(scalar):
    values = [-np.inf, -1000, -104, -100, -90, -20, -1, -1e-7, -0.0,
              0.0, 1e-7, 1, 20, 90, 100, 104, 1000, np.inf, np.nan]
    gate = xp.asarray(values, dtype=xp.float32)
    up = xp.linspace(-2, 2, gate.size, dtype=xp.float32)
    exercise(gate, up, -up, xp.asarray(scalar, dtype=xp.float32),
             xp.asarray(0, dtype=xp.float32), xp.asarray(-1, dtype=xp.float32))


@pytest.mark.parametrize("shape", [(), (0,), (1,), (1, 1, 37), (2, 3, 37)])
def test_shapes_and_broadcast(shape):
    gate = xp.full(shape, .75, dtype=xp.float32)
    exercise(gate, gate, gate, xp.zeros((1, 1), dtype=xp.float32),
             xp.asarray(-.75, dtype=xp.float32), xp.asarray(-2, dtype=xp.float32))


def test_maximum_test_size():
    rng = np.random.default_rng(251)
    gate = xp.asarray(rng.uniform(-110, 110, _MAX_ELEMENTS), dtype=xp.float32)
    up = xp.asarray(rng.normal(size=_MAX_ELEMENTS), dtype=xp.float32)
    exercise(gate, up, -up, xp.asarray(.25, dtype=xp.float32),
             xp.asarray(-.125, dtype=xp.float32), xp.asarray(-.75, dtype=xp.float32))


def test_nondefault_stream_and_fresh_inputs():
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        gate = xp.arange(257, dtype=xp.float32) - xp.float32(128)
        up = gate * xp.float32(.03)
        scalar = xp.asarray(-.7, dtype=xp.float32)
        dt = xp.asarray(.25, dtype=xp.float32)
        ssm_a = xp.asarray(-.5, dtype=xp.float32)
        for factor in (1, -2):
            gate *= xp.float32(factor)
            expected = references(gate, up, up, scalar, dt, ssm_a)
            actual = (gpu.shared_hidden(gate, up).copy(),
                      gpu.combine_shared(up, up, scalar).copy(),
                      *[a.copy() for a in gpu.delta_coefficients(gate, gate, dt, ssm_a)])
            # Queue consumers on the same stream before any host validation.
            if factor == 1:
                first = actual, expected
    stream.synchronize()
    for a, r in zip(*first):
        exact(a, r, "nondefault first call")
    for a, r in zip(actual, expected):
        exact(a, r, "nondefault fresh call")


_CALLS = [("shared_hidden", 2), ("combine_shared", 3), ("delta_coefficients", 4)]


@pytest.mark.parametrize("name,nargs", _CALLS)
@pytest.mark.parametrize("kind", ["numpy", "float16", "float64", "host_scalar"])
def test_type_errors_before_kernel(monkeypatch, name, nargs, kind):
    good = xp.ones(1, dtype=xp.float32)
    bad = {"numpy": np.ones(1, dtype=np.float32),
           "float16": good.astype(xp.float16), "float64": good.astype(xp.float64),
           "host_scalar": np.float32(1)}[kind]
    monkeypatch.setattr(gpu, "_kernels", lambda device: pytest.fail("invalid input launched"))
    for index in range(nargs):
        args = [good] * nargs
        args[index] = bad
        with pytest.raises(TypeError, match="CuPy float32"):
            getattr(gpu, name)(*args)


@pytest.mark.parametrize("name,shapes", [
    ("shared_hidden", [(2, 3), (3,)]),
    ("combine_shared", [(2, 3), (3,), ()]),
    ("combine_shared", [(2, 3), (2, 3), (2,)]),
    ("delta_coefficients", [(2, 3), (3,), (), ()]),
    ("delta_coefficients", [(2, 3), (2, 3), (4,), ()]),
    ("delta_coefficients", [(3,), (3,), (2, 3), ()]),
    ("delta_coefficients", [(3,), (3,), (), (2, 3)]),
])
def test_shape_errors_before_kernel(monkeypatch, name, shapes):
    arrays = [xp.ones(s, dtype=xp.float32) for s in shapes]
    monkeypatch.setattr(gpu, "_kernels", lambda device: pytest.fail("invalid input launched"))
    with pytest.raises(ValueError):
        getattr(gpu, name)(*arrays)


def test_backend_required(monkeypatch):
    monkeypatch.setattr(gpu, "xp", np)
    for name, nargs in _CALLS:
        with pytest.raises(TypeError, match="CuPy backend"):
            getattr(gpu, name)(*[np.ones(1, dtype=np.float32)] * nargs)


def test_cached_single_kernel_per_call_and_no_array_helpers(monkeypatch):
    device = xp.cuda.runtime.getDevice()
    kernels = gpu._kernels(device)
    assert gpu._kernels(device) is kernels
    constructor = xp.ElementwiseKernel
    options = []

    def constructed(*args, **kwargs):
        options.append(kwargs.get('options'))
        return constructor(*args, **kwargs)

    with monkeypatch.context() as construction:
        construction.setattr(xp, 'ElementwiseKernel', constructed)
        gpu._kernels.__wrapped__(device)
    assert options == [("--fmad=false",)] * 3
    a = xp.ones(17, dtype=xp.float32)
    scalar = xp.ones((), dtype=xp.float32)
    calls = []

    def wrapper(index):
        def launch(*args):
            calls.append(index)
            return kernels[index](*args)
        return launch

    monkeypatch.setattr(gpu, "_kernels", lambda device: tuple(wrapper(i) for i in range(3)))

    def forbidden(*args, **kwargs):
        pytest.fail("runtime attempted a host readback, extra array, or CuPy expression")

    for name in ("asnumpy", "asarray", "empty", "empty_like", "isfinite", "exp", "logaddexp", "where"):
        monkeypatch.setattr(xp, name, forbidden)
    gpu.shared_hidden(a, a)
    gpu.combine_shared(a, a, scalar)
    beta, decay = gpu.delta_coefficients(a, a, scalar, scalar)
    assert beta.shape == decay.shape == a.shape
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("name,nargs", _CALLS)
def test_wrong_current_device_before_kernel(monkeypatch, name, nargs):
    a = xp.ones(1, dtype=xp.float32)
    monkeypatch.setattr(xp.cuda.runtime, "getDevice", lambda: a.device.id + 1)
    monkeypatch.setattr(gpu, "_kernels", lambda device: pytest.fail("invalid device launched"))
    for index in range(nargs):
        with pytest.raises(ValueError, match="current device"):
            getattr(gpu, name)(*[a] * nargs)


def test_actual_second_device_when_available():
    if xp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("second GPU required for actual mixed-device validation")
    current = xp.cuda.runtime.getDevice()
    a = xp.ones(1, dtype=xp.float32)
    with xp.cuda.Device((current + 1) % xp.cuda.runtime.getDeviceCount()):
        b = xp.ones(1, dtype=xp.float32)
        other_kernels = gpu._kernels(xp.cuda.runtime.getDevice())
    assert other_kernels is not gpu._kernels(current)
    for name, nargs in _CALLS:
        for index in range(nargs):
            args = [a] * nargs
            args[index] = b
            with pytest.raises(ValueError, match="current device"):
                getattr(gpu, name)(*args)
