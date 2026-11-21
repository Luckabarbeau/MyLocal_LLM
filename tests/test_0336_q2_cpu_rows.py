"""Same GGML Q8_0 bytes, Strata singleton reduction; no model/checkpoint load.

Run native tests only through scripts/run_0290_isolated_benchmark.py.
The parent bridge must include q2_cpu_rows.h after q2_cpu_simd.h.
"""
import ctypes as C

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import (
    GGMLPackedCPUExecutor, GGMLPackedCPURowPool, _load_bridge,
)
from test_0256_ggml_packed_cpu import selected_library, packed, DOT


@pytest.fixture(scope='module')
def bridge(selected_library):
    lib = _load_bridge()
    lib.mylocal_q2_rows_available.argtypes = []
    lib.mylocal_q2_rows_available.restype = C.c_int
    lib.mylocal_q2_rows_test.argtypes = [
        C.c_int, C.c_int, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p,
    ]
    lib.mylocal_q2_rows_test.restype = C.c_int
    lib.mylocal_q2_rows_exact_test.argtypes = lib.mylocal_q2_rows_test.argtypes
    lib.mylocal_q2_rows_exact_test.restype = C.c_int
    return lib


def synthetic(k, n, padding, seed):
    rng = np.random.default_rng(seed)
    stride = k // 64 * 18 + padding
    # Deliberately unaligned buffers and odd padded row strides.
    owner = rng.integers(0, 256, n * stride + 1, np.uint8)
    weights = owner[1:].reshape(n, stride)
    qowner = rng.integers(0, 256, k // 32 * 34 + 1, np.uint8)
    q8 = qowner[1:].reshape(-1, 34)
    blocks = weights[:, :k // 64 * 18].reshape(n, k // 64, 18).copy()
    for raw in (blocks.reshape(-1, 18), q8):
        bits = rng.integers(0, 65536, len(raw), np.uint16)
        bits[(bits & 0x7c00) == 0x7c00] &= np.uint16(0x83ff)
        # Explicit signed normal/subnormal scales, signed zeros and max finite.
        edges = np.array([1, 0x8001, 0x400, 0x8400, 0, 0x8000,
                          0x3c00, 0xbc00, 0x7bff, 0xfbff], dtype='<u2')
        bits[:min(len(bits), len(edges))] = edges[:min(len(bits), len(edges))]
        raw[:, :2] = bits.astype('<u2').view(np.uint8).reshape(-1, 2)
    weights[:, :k // 64 * 18] = blocks.reshape(n, -1)
    q8[:, 2] = 128  # signed -128, including the maddubs saturation boundary case
    return weights, q8, stride


def fp64_baseline(weights, q8, k):
    n = len(weights)
    blocks = weights[:, :k // 64 * 18].copy().reshape(n, k // 64, 18)
    d = blocks[:, :, :2].copy().view('<f2').reshape(n, -1).astype(np.float64)
    codes = ((blocks[:, :, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3)
    w = (codes.astype(np.float64) - 1).reshape(n, -1) * np.repeat(d, 64, axis=1)
    scale = q8[:, :2].copy().view('<f2').reshape(-1).astype(np.float64)
    x = (q8[:, 2:].copy().view(np.int8).astype(np.float64) * scale[:, None]).reshape(-1)
    terms = w * x
    return terms.sum(axis=1), np.abs(terms).sum(axis=1)


def assert_roundoff(actual, expected, contribution, k):
    # FP32 FMA lane sums plus scalar correction; cancellation is bounded by
    # absolute contributions, not by the potentially near-zero final dot.
    tolerance = (k // 64 + 8) * 4 * np.finfo(np.float32).eps * contribution + 1e-20
    error = np.abs(actual.astype(np.float64) - expected)
    assert np.isfinite(actual).all()
    assert np.all(error <= tolerance), (error, tolerance, expected, actual)


@pytest.mark.parametrize('k', [64, 128, 192, 320, 640, 768, 2560])
@pytest.mark.parametrize('padding', [0, 1, 37])
def test_identical_q8_fp64_and_strides(bridge, k, padding):
    weights, q8, stride = synthetic(k, 19, padding, 336 + k + padding)
    before = weights.copy(), q8.copy()
    expected, contribution = fp64_baseline(weights, q8, k)
    output = np.full(21, np.nan, np.float32)
    status = bridge.mylocal_q2_rows_test(19, k, weights.ctypes.data, stride,
                                        q8.ctypes.data, output[1:].ctypes.data)
    if not bridge.mylocal_q2_rows_available():
        assert status == 2
        assert np.isnan(output).all()
        return
    assert status == 0
    assert np.isnan(output[[0, -1]]).all()
    assert_roundoff(output[1:-1], expected, contribution, k)
    np.testing.assert_array_equal(weights, before[0])
    np.testing.assert_array_equal(q8, before[1])
    # Also compare the old reduction on the exact same Q8 bytes, not a fresh
    # quantization of an FP32 vector. Neither reduction is required bit-exact.
    dot = DOT(C.cast(bridge.mylocal_q2_avx2_dot, C.c_void_p).value)
    baseline = np.empty(19, np.float32)
    for r in range(19):
        dot(k, baseline[r:].ctypes.data, 0, weights[r:].ctypes.data, 0,
            q8.ctypes.data, 0, 1)
    assert_roundoff(baseline, expected, contribution, k)


@pytest.mark.parametrize('n,k,stride', [(0, 64, 18), (-1, 64, 18),
                                      (1, 0, 18), (1, 63, 18), (1, 64, 17)])
def test_diagnostic_invalid_dimensions(bridge, n, k, stride):
    buf = np.zeros(128, np.uint8)
    out = np.full(1, 123, np.float32)
    status = bridge.mylocal_q2_rows_test(n, k, buf.ctypes.data, stride,
                                        buf.ctypes.data, out.ctypes.data)
    assert status == (1 if bridge.mylocal_q2_rows_available() else 2)
    assert out[0] == 123


@pytest.mark.parametrize('missing', [0, 1, 2])
def test_diagnostic_null_pointers(bridge, missing):
    buf = np.zeros(128, np.uint8)
    pointers = [buf.ctypes.data] * 3
    pointers[missing] = None
    status = bridge.mylocal_q2_rows_test(1, 64, pointers[0], 18, pointers[1], pointers[2])
    assert status == (1 if bridge.mylocal_q2_rows_available() else 2)


def test_marker_unique_and_dot_compatible(bridge):
    marker = C.cast(bridge.mylocal_q2_rows_marker, C.c_void_p).value
    original = C.cast(bridge.mylocal_q2_avx2_dot, C.c_void_p).value
    assert marker != original
    if not bridge.mylocal_q2_rows_available():
        pytest.skip('AVX2/F16C/FMA unavailable')
    w, q8, _ = synthetic(640, 1, 0, 337)
    out = np.empty(2, np.float32)
    for i, fn in enumerate((marker, original)):
        DOT(fn)(640, out[i:].ctypes.data, 0, w.ctypes.data, 0, q8.ctypes.data, 0, 1)
    np.testing.assert_array_equal(out[:1].view(np.uint32), out[1:].view(np.uint32))


@pytest.mark.parametrize('workers', [1, 4])
def test_pool_integration(selected_library, workers):
    baseline = GGMLPackedCPUExecutor(selected_library, q2_simd=True, q2_rows=False)
    if not baseline._bridge.mylocal_q2_rows_available():
        pytest.skip('AVX2/F16C/FMA unavailable')
    optimized = GGMLPackedCPUExecutor(selected_library, q2_rows=True)
    bank = [(packed('IQ3_S', 320, 256, 400+i),
             packed('IQ3_S', 320, 256, 410+i),
             packed('Q2_0' if i % 2 == 0 else 'IQ4_NL', 256, 320, 420+i))
            for i in range(4)]
    rng = np.random.default_rng(338)
    with GGMLPackedCPURowPool(optimized, 256, 320, workers,
                             scheduler='tasks', host_help=True) as pool:
        for epoch in range(6):
            x = rng.normal(0, .1, (1, 256)).astype(np.float32)
            selected = bank[:1 + epoch % 4]
            expected = np.concatenate([baseline.expert(x, ws) for ws in selected])
            # Direct expert/linear paths still use the marker's original dot.
            direct = np.concatenate([optimized.expert(x, ws) for ws in selected])
            np.testing.assert_array_equal(direct, expected)
            got = pool.submit(x, selected).result()
            np.testing.assert_allclose(got, expected, atol=2e-7, rtol=3e-5)


@pytest.mark.parametrize('k', [64, 128, 192, 320, 640, 768, 2560])
@pytest.mark.parametrize('padding', [0, 1, 37])
@pytest.mark.parametrize('n', [1, 4, 7, 19])
def test_exact_arbitrary_q8_bitwise(bridge, k, padding, n):
    dot = DOT(C.cast(bridge.mylocal_q2_avx2_dot, C.c_void_p).value)
    rng = np.random.default_rng(3360 + k + padding + n)
    for trial in range(12):
        w, q8, stride = synthetic(k, n, padding, int(rng.integers(0, 2**31)))
        if trial == 1:
            q8[:, 2:] = 128  # all -128; no signed-abs dot workaround allowed
        elif trial == 2:
            q8[:, 2:] = 0
        elif trial == 3:
            q8[:, 2:] = 127
        before_w, before_q8 = w.copy(), q8.copy()
        out = np.full(n + 2, np.nan, np.float32)
        status = bridge.mylocal_q2_rows_exact_test(
            n, k, w.ctypes.data, stride, q8.ctypes.data, out[1:].ctypes.data)
        if not bridge.mylocal_q2_rows_available():
            assert status == 2
            assert np.isnan(out).all()
            return
        assert status == 0
        expected = np.empty(n, np.float32)
        for r in range(n):
            dot(k, expected[r:].ctypes.data, 0, w[r:].ctypes.data, 0,
                q8.ctypes.data, 0, 1)
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(out[1:-1].view(np.uint32), expected.view(np.uint32))
        assert np.isnan(out[[0, -1]]).all()
        np.testing.assert_array_equal(w, before_w)
        np.testing.assert_array_equal(q8, before_q8)


def test_exact_marker_unique_and_dot_compatible(bridge):
    pointers = [C.cast(getattr(bridge, name), C.c_void_p).value for name in (
        'mylocal_q2_rows_exact_marker', 'mylocal_q2_rows_marker', 'mylocal_q2_avx2_dot')]
    assert len(set(pointers)) == 3
    if not bridge.mylocal_q2_rows_available():
        pytest.skip('AVX2/F16C/FMA unavailable')
    w, q8, _ = synthetic(2560, 1, 0, 3361)
    out = np.empty(3, np.float32)
    for i, pointer in enumerate(pointers):
        DOT(pointer)(2560, out[i:].ctypes.data, 0, w.ctypes.data, 0, q8.ctypes.data, 0, 1)
    np.testing.assert_array_equal(out.view(np.uint32), np.repeat(out[2:].view(np.uint32), 3))


@pytest.mark.parametrize('workers', [1, 4])
def test_exact_pool_integration(selected_library, workers):
    baseline = GGMLPackedCPUExecutor(selected_library, q2_simd=True, q2_rows=False,
                                    q2_rows_exact=False)
    if not baseline._bridge.mylocal_q2_rows_available():
        pytest.skip('AVX2/F16C/FMA unavailable')
    optimized = GGMLPackedCPUExecutor(selected_library, q2_rows=False, q2_rows_exact=True)
    bank = [(packed('IQ3_S', 320, 256, 500+i),
             packed('IQ3_S', 320, 256, 510+i),
             packed('Q2_0' if i % 2 == 0 else 'IQ4_NL', 256, 320, 520+i))
            for i in range(4)]
    rng = np.random.default_rng(3362)
    with GGMLPackedCPURowPool(optimized, 256, 320, workers,
                             scheduler='tasks', host_help=True) as pool:
        for epoch in range(8):
            x = rng.normal(0, .1, (1, 256)).astype(np.float32)
            selected = bank[:1 + epoch % 4]
            expected = np.concatenate([baseline.expert(x, ws) for ws in selected])
            direct = np.concatenate([optimized.expert(x, ws) for ws in selected])
            got = pool.submit(x, selected).result()
            np.testing.assert_array_equal(direct.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
