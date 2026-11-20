"""Standalone operator tests; GPU cases are for the parent's isolated run only.

FP32 acceptance: rtol=2e-5, atol=4e-6 against an independent FP64 graph.
D=256 tree sums have ~8 rounding levels rather than 256 serial additions;
softmax/weighted sums and part merging add further FP32 rounding. The absolute
floor covers cancellation/near-zero outputs, where elementwise relative error
is not meaningful. These budgets are deliberately tested, not claimed measured
results. Metrics include max absolute and relative L2 error for old/new/FP64.
"""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import qsa_cooperative_gpu as cooperative


ATOL = 4e-6
RTOL = 2e-5


def oracle_fp64(q, qg, k, v, keys, values, n, scale):
    """Independent NumPy FP64 graph over the actual FP32 input values."""
    q, qg, k, v, keys, values = [
        np.asarray(a, dtype=np.float64) for a in (q, qg, k, v, keys, values)
    ]
    heads, dim = q.shape
    result = []
    for h in range(heads):
        kh = h // (heads // k.shape[0])
        all_k = np.concatenate((keys[:n-1, kh], k[kh:kh+1]), axis=0)
        all_v = np.concatenate((values[:n-1, kh], v[kh:kh+1]), axis=0)
        logits = np.sum(all_k * q[h], axis=1) * float(np.float32(scale))
        weights = np.exp(logits - np.max(logits))
        context = np.sum(weights[:, None] * all_v, axis=0) / np.sum(weights)
        gate = qg[h, dim:]
        z = np.exp(-np.abs(gate))
        result.append(context * np.where(gate >= 0, 1 / (1 + z), z / (1 + z)))
    return np.asarray(result).ravel()


def metrics(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    delta = actual - expected
    return dict(max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)))


class _Array:
    """Metadata only: CPU validation must never inspect contents or copy arrays."""

    def __init__(self, shape, dtype=np.float32, device=0):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.device = np.dtype(dtype), SimpleNamespace(id=device)


def metadata():
    return [_Array(s) for s in ((4, 256), (4, 512), (2, 256), (2, 256),
                                (140, 2, 256), (140, 2, 256))]


def test_optional_dependencies_not_needed_for_module_import(monkeypatch):
    import sys

    # Load this file directly so no other mini_llm module can import CuPy/Torch.
    spec = importlib.util.spec_from_file_location("_qsa_cpu_import", cooperative.__file__)
    module = importlib.util.module_from_spec(spec)
    for dependency in ("cupy", "triton", "triton.language", "torch"):
        monkeypatch.setitem(sys.modules, dependency, None)
    spec.loader.exec_module(module)
    assert module.triton is None
    assert callable(module.qsa_cooperative_gpu)


@pytest.mark.parametrize("n", [1, 2, 77, 141, 256, 435, 1024, 2051, 8192])
def test_metadata_context_bound_and_uncommitted_capacity(n):
    arrays = metadata()
    arrays[-2:] = [_Array((n-1, 2, 256)), _Array((n-1, 2, 256))]
    assert cooperative._validate(SimpleNamespace(ndarray=_Array), *arrays, n, .37) == (
        n, float(np.float32(.37)))


@pytest.mark.parametrize("kind,exc,match", [
    ("n0", ValueError, "n must"), ("n8193", ValueError, "n must"),
    ("nfloat", TypeError, "integer"), ("nbool", TypeError, "integer"),
    ("dtype", TypeError, "FP32"), ("device", ValueError, "same device"),
    ("shape", ValueError, "shapes"), ("groups", ValueError, "grouped heads"),
    ("capacity", ValueError, "capacity"), ("nan_scale", ValueError, "scale"),
    ("overflow_scale", ValueError, "scale"), ("host", TypeError, "CuPy"),
])
def test_invalid_metadata_rejected_before_launch(kind, exc, match):
    arrays, n, scale = metadata(), 141, .37
    if kind == "n0": n = 0
    if kind == "n8193": n = 8193
    if kind == "nfloat": n = 2.5
    if kind == "nbool": n = True
    if kind == "dtype": arrays[0] = _Array((4, 256), np.float16)
    if kind == "device": arrays[-1] = _Array((140, 2, 256), device=1)
    if kind == "shape": arrays[1] = _Array((4, 256))
    if kind == "groups": arrays[2:4] = [_Array((3, 256)), _Array((3, 256))]
    if kind == "capacity": arrays[-1] = _Array((139, 2, 256))
    if kind == "nan_scale": scale = float("nan")
    if kind == "overflow_scale": scale = 1e100
    if kind == "host": arrays[0] = np.zeros((4, 256), np.float32)
    with pytest.raises(exc, match=match):
        cooperative._validate(SimpleNamespace(ndarray=_Array), *arrays, n, scale)


def test_compiled_kernel_cache_device_and_explicit_current_stream(monkeypatch):
    calls, warmups = [], []

    class Compiled:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append((grid, args, kwargs))
            return run

    class Kernel:
        def warmup(self, *args, **kwargs):
            warmups.append((args, kwargs))
            return Compiled()

    device, stream = [0], SimpleNamespace(ptr=12345)
    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: device[0]),
        get_current_stream=lambda: stream))
    monkeypatch.setattr(cooperative, "_KERNELS", {})
    kernel = Kernel()
    for d, s, arg in [(0, 12345, 77), (0, 54321, 78), (1, 999, 77)]:
        device[0], stream.ptr = d, s
        cooperative._launch(cp, kernel, (24, 2), (arg,), (256, 2))
    assert len(warmups) == 2
    assert len(cooperative._KERNELS) == 2
    assert [c[2]["stream"] for c in calls] == [12345, 54321, 999]
    assert [c[1] for c in calls] == [(77,), (78,), (77,)]
    assert all(c[0] == (24, 2, 1) for c in calls)
    assert warmups[0][1] == dict(grid=(24, 2), num_warps=4, num_stages=1,
                                 enable_fp_fusion=False)


@pytest.fixture
def gpu(monkeypatch):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("triton")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy GPU unavailable: {exc}")
    if not count:
        pytest.skip("CuPy GPU unavailable")
    old = importlib.import_module("mini_llm.ops.qsa_decode_gpu")
    # The old operator imports backend.xp, whose CPU default need not change.
    monkeypatch.setattr(old, "xp", cp)
    return cp, old.qsa_decode_gpu


def gpu_case(cp, n, *, heads=24, kv=2, dim=256, strided=False, reverse=False, capacity=None):
    rng = np.random.default_rng(314 + n)
    capacity = n + 3 if capacity is None else capacity
    bases = []

    def array(shape):
        backing_shape = tuple(2*s for s in shape) if strided else shape
        base = cp.asarray(rng.normal(size=backing_shape).astype(np.float32))
        bases.append(base)
        view = base[tuple(slice(None, None, 2) for _ in shape)] if strided else base
        return view[tuple(slice(None, None, -1) for _ in shape)] if reverse else view

    q, qg = array((heads, dim)), array((heads, 2*dim))
    k, v = array((kv, dim)), array((kv, dim))
    keys, values = array((capacity, kv, dim)), array((capacity, kv, dim))
    keys[n-1:], values[n-1:] = cp.nan, cp.nan
    qg[:, :dim] = cp.nan  # Query is already q; this half must never enter gating.
    return (q, qg, k, v, keys, values), bases


def check_case(cp, old, inputs, bases, n, scale, record_property):
    before = [cp.asnumpy(a).view(np.uint32).copy() for a in bases]
    host = [cp.asnumpy(a) for a in inputs]
    expected = oracle_fp64(*host, n, scale)
    current_stream, current_device = cp.cuda.get_current_stream().ptr, cp.cuda.runtime.getDevice()
    actual = cooperative.qsa_cooperative_gpu(*inputs, n, scale)
    # Same-stream consumer, before any test synchronization.
    consumer = actual.copy()
    serial = old(*inputs, n, scale)
    assert cp.cuda.get_current_stream().ptr == current_stream
    assert cp.cuda.runtime.getDevice() == current_device
    new_h, old_h = cp.asnumpy(consumer), cp.asnumpy(serial)
    assert actual.dtype == cp.float32 and actual.shape == (inputs[0].size,)
    assert np.isfinite(new_h).all()
    for label, a, b in (("new_fp64", new_h, expected), ("old_fp64", old_h, expected),
                        ("new_old", new_h, old_h)):
        row = metrics(a, b)
        record_property(label, row)
        print(f"n={n} {label}: {row}")
    np.testing.assert_allclose(new_h, expected, atol=ATOL, rtol=RTOL)
    # A serial FP32 dot is less accurate than a tree reduction. Give its
    # accumulated rounding a separate budget, never use it as the FP64 oracle.
    np.testing.assert_allclose(old_h, expected, atol=8e-6, rtol=4e-5)
    np.testing.assert_allclose(new_h, old_h, atol=12e-6, rtol=6e-5)
    for base, snapshot in zip(bases, before):
        np.testing.assert_array_equal(cp.asnumpy(base).view(np.uint32), snapshot)
    return new_h


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 2, 77, 141, 256, 435, 1024, 2051, 8192])
def test_fp32_full_context_against_serial_and_fp64(gpu, n, record_property):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = gpu_case(cp, n)
        check_case(cp, old, inputs, bases, n, 1/16, record_property)


@pytest.mark.gpu
@pytest.mark.parametrize("n,reverse,dim", [(1, False, 256), (77, True, 10),
                                           (141, False, 256), (435, True, 256),
                                           (2051, False, 256)])
def test_arbitrary_strides_two_groups_nondefault_stream(gpu, n, reverse, dim, record_property):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = gpu_case(cp, n, heads=4, kv=2, dim=dim,
                                 strided=True, reverse=reverse)
        check_case(cp, old, inputs, bases, n, .0625, record_property)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 141, 435])
def test_extreme_logits_and_stable_second_half_gates(gpu, n, record_property):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = gpu_case(cp, n, heads=4, kv=2)
        q, qg, k, v, keys, values = inputs
        q[:] = 0
        q[:, 0] = 1
        keys[:n-1] = 0
        keys[:n-1, :, 0] = -2000
        if n > 1:
            keys[0, :, 0] = 2000
        k[:] = 0
        k[:, 0] = 2000  # Two equal, enormous logits: average past[0] and current.
        gates = np.resize(np.array([-1000, -100, -20, 0, 20, 100, 1000,
                                    -np.inf, np.inf], np.float32), 256)
        qg[:, 256:] = cp.asarray(gates)
        check_case(cp, old, inputs, bases, n, 1., record_property)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 2, 141])
def test_no_uncommitted_cache_slot_is_required(gpu, n, record_property):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = gpu_case(cp, n, heads=4, capacity=n-1)
        check_case(cp, old, inputs, bases, n, .0625, record_property)


@pytest.mark.gpu
def test_zero_strides_and_inflight_stream_producer(gpu, record_property):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = gpu_case(cp, 141, heads=4)
        q, qg, k, v, keys, values = inputs
        q = cp.broadcast_to(q[:1], q.shape)
        inputs = (q, qg, k, v, keys, values)
        check_case(cp, old, inputs, bases, 141, .0625, record_property)
        # No host reads between producer, both operator launches, and consumer.
        qg[:, 256:] = 0
        v[:] = 7
        values[:140] = 7
        output = cooperative.qsa_cooperative_gpu(*inputs, 141, .0625).copy()
        np.testing.assert_allclose(cp.asnumpy(output), 3.5, atol=4e-6, rtol=RTOL)


@pytest.mark.gpu
def test_other_current_device_is_restored(gpu, record_property):
    cp, _ = gpu
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two GPUs required for device restoration")
    original = cp.cuda.runtime.getDevice()
    target = (original + 1) % cp.cuda.runtime.getDeviceCount()
    with cp.cuda.Device(target), cp.cuda.Stream(non_blocking=True):
        inputs, _ = gpu_case(cp, 77, heads=4)
        expected = oracle_fp64(*[cp.asnumpy(a) for a in inputs], 77, .0625)
        target_stream = cp.cuda.get_current_stream().ptr
        with cp.cuda.Device(original):
            original_stream = cp.cuda.get_current_stream().ptr
            output = cooperative.qsa_cooperative_gpu(*inputs, 77, .0625)
            assert cp.cuda.runtime.getDevice() == original
            assert cp.cuda.get_current_stream().ptr == original_stream
        assert cp.cuda.get_current_stream().ptr == target_stream
        np.testing.assert_allclose(cp.asnumpy(output), expected, atol=ATOL, rtol=RTOL)
