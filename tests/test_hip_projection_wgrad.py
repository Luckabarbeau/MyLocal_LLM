"""Disjoint HIP projection wgrad accuracy, ABI, and stream contracts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mini_llm.ops.hip_projection_wgrad import _split_gradient, projection_wgrad

pytestmark = pytest.mark.gpu


@pytest.fixture
def hip():
    if not cp.cuda.runtime.is_hip:
        pytest.skip("requires HIP CuPy; CUDA rejection is tested separately")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("requires an AMD GPU")


def _check(actual, reference, label):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diff = np.abs(actual - reference)
    absolute = float(diff.max(initial=0))
    relative = float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-300))
    worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
    detail = (
        f"{label}: max_abs={absolute:.9g}, relative_l2={relative:.9g}, "
        f"worst_actual={actual[worst]:.12g}, worst_reference={reference[worst]:.12g}, "
        f"dtype=float32, eps={np.finfo(np.float32).eps}, atol=rtol=1e-5; "
        "finite_difference_step=not_applicable"
    )
    print(detail)
    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5, err_msg=detail)


@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 5, 9), (31, 17, 13), (128, 32, 64)])
@pytest.mark.parametrize("gradient_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("strided", [False, True])
def test_fp32_and_fp64_oracles(hip, shape, gradient_dtype, strided):
    t, din, dout = shape
    rng = np.random.default_rng(107)
    step = 2 if strided else 1
    a = cp.asarray(rng.normal(0, 0.1, (t * step, din * step)), dtype=cp.float32)
    g = cp.asarray(rng.normal(0, 0.1, (t * step, dout * step)), dtype=cp.float32)
    a = a.astype("bfloat16")[::step, ::step]
    g = g.astype(gradient_dtype)[::step, ::step]
    if strided:
        a, g = a[::-1], g[::-1]
    a_host = cp.asnumpy(a.astype(cp.float32))
    g_host = cp.asnumpy(g.astype(cp.float32))
    result = projection_wgrad(a, g)
    assert result.dtype == cp.float32
    assert result.shape == (din, dout)
    assert result.flags.c_contiguous
    actual = cp.asnumpy(result)
    _check(actual, cp.asnumpy(a.astype(cp.float32).T @ g.astype(cp.float32)),
           f"{shape}/{gradient_dtype}/strided={strided}/cupy_fp32")
    _check(actual, a_host.astype(np.float64).T @ g_host.astype(np.float64),
           f"{shape}/{gradient_dtype}/strided={strided}/numpy_fp64")
    np.testing.assert_array_equal(cp.asnumpy(a.astype(cp.float32)), a_host)
    np.testing.assert_array_equal(cp.asnumpy(g.astype(cp.float32)), g_host)


@pytest.mark.parametrize("multiscale", [False, True])
def test_three_part_reconstruction(hip, multiscale):
    rng = np.random.default_rng(107)
    values = rng.normal(size=(257, 31)).astype(np.float32)
    if multiscale:
        values = np.ldexp(values, rng.integers(-80, 81, size=values.shape)).astype(np.float32)
    g = cp.asarray(values)
    parts = _split_gradient(g)
    assert all(part.dtype == cp.dtype("bfloat16") for part in parts)
    host_parts = [cp.asnumpy(part.astype(cp.float32)) for part in parts]
    # Round once after summing the exact represented components. A sequential
    # FP32 high+mid+low expression itself introduces additional rounding.
    reconstruction = sum(part.astype(np.float64) for part in host_parts).astype(np.float32)
    mismatches = np.flatnonzero(reconstruction != values)
    absolute = np.abs(reconstruction.astype(np.float64) - values.astype(np.float64))
    relative = absolute / np.maximum(np.abs(values.astype(np.float64)), 1e-300)
    print(f"reconstruction multiscale={multiscale}: mismatches={mismatches.size}, "
          f"max_abs={absolute.max():.9g}, max_relative={relative.max():.9g}, "
          f"dtype=float32, eps={np.finfo(np.float32).eps}")
    if mismatches.size:
        index = int(mismatches[0])
        pytest.fail(f"reconstruction original={values.flat[index]!r}, "
                    f"actual={reconstruction.flat[index]!r}, "
                    f"parts={[part.flat[index] for part in host_parts]}, "
                    f"absolute={absolute.flat[index]}, relative={relative.flat[index]}, "
                    f"dtype=float32, eps={np.finfo(np.float32).eps}, tolerance=exact")


def test_fp32_residuals_are_not_discarded(hip):
    a = cp.ones((9, 5), dtype=cp.float32).astype("bfloat16")
    g = cp.full((9, 7), 1.003, dtype=cp.float32)
    actual = cp.asnumpy(projection_wgrad(a, g))
    oracle = np.full((5, 7), 9 * float(np.float32(1.003)), dtype=np.float64)
    _check(actual, oracle, "residual_sensitive")
    naive = cp.asnumpy(a.astype(cp.float32).T @ g.astype("bfloat16").astype(cp.float32))
    assert np.max(np.abs(naive - oracle)) > 1e-2


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32"])
def test_activation_dtype_errors(hip, dtype):
    with pytest.raises(TypeError, match="activation.*bfloat16"):
        projection_wgrad(cp.zeros((3, 5), dtype=dtype), cp.zeros((3, 7), dtype=cp.float32))


@pytest.mark.parametrize("dtype", ["float16", "float64", "int32"])
def test_gradient_dtype_errors(hip, dtype):
    with pytest.raises(TypeError, match="gradient.*bfloat16 or float32"):
        projection_wgrad(cp.zeros((3, 5), dtype="bfloat16"), cp.zeros((3, 7), dtype=dtype))


def test_shape_and_array_errors(hip):
    a = cp.zeros((3, 5), dtype="bfloat16")
    with pytest.raises(ValueError, match="same T"):
        projection_wgrad(a, cp.zeros((4, 7), dtype=cp.float32))
    with pytest.raises(ValueError, match="rank-2"):
        projection_wgrad(a, cp.zeros((3, 1, 7), dtype=cp.float32))
    with pytest.raises(TypeError, match="CuPy"):
        projection_wgrad(np.zeros((3, 5), dtype=np.float32), cp.zeros((3, 7), dtype=cp.float32))


@pytest.mark.parametrize("shape", [(0, 5, 7), (3, 0, 7), (3, 5, 0)])
def test_empty_dimensions(hip, shape):
    t, din, dout = shape
    actual = projection_wgrad(cp.empty((t, din), dtype="bfloat16"),
                              cp.empty((t, dout), dtype=cp.float32))
    np.testing.assert_array_equal(cp.asnumpy(actual), np.zeros((din, dout), dtype=np.float32))


@pytest.mark.parametrize("gradient_dtype,gemm_count", [("bfloat16", 1), ("float32", 3)])
def test_gemm_abi_and_pointer_mode_restoration(hip, monkeypatch, gradient_dtype, gemm_count):
    from cupy_backends.cuda.libs import cublas

    a = cp.ones((7, 5), dtype="bfloat16")
    g = cp.ones((7, 9), dtype=gradient_dtype)
    handle = cp.cuda.device.get_cublas_handle()
    original_mode = cublas.getPointerMode(handle)
    original_gemm = cublas.gemmEx
    calls = []

    def checked_gemm(*args):
        assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_HOST
        assert args[1:6] == (0, 1, 9, 5, 7)
        assert (args[8], args[9], args[11], args[12]) == (14, 9, 14, 5)
        assert (args[15], args[16], args[17]) == (0, 9, 0)
        beta = np.ctypeslib.as_array(__import__("ctypes").cast(
            args[13], __import__("ctypes").POINTER(__import__("ctypes").c_float)), shape=(1,))[0]
        assert beta == (0 if not calls else 1)
        calls.append(args)
        return original_gemm(*args)

    monkeypatch.setattr(cublas, "gemmEx", checked_gemm)
    try:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        actual = projection_wgrad(a, g)
        assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_DEVICE
        assert len(calls) == gemm_count
        assert len({args[14] for args in calls}) == 1
        np.testing.assert_array_equal(cp.asnumpy(actual), np.full((5, 9), 7, dtype=np.float32))
    finally:
        cublas.setPointerMode(handle, original_mode)


def test_pointer_mode_restored_after_gemm_error(hip, monkeypatch):
    from cupy_backends.cuda.libs import cublas

    a = cp.ones((3, 5), dtype="bfloat16")
    g = cp.ones((3, 7), dtype=cp.float32)
    handle = cp.cuda.device.get_cublas_handle()
    original_mode = cublas.getPointerMode(handle)

    def fail(*args):
        raise RuntimeError("injected GEMM error")

    monkeypatch.setattr(cublas, "gemmEx", fail)
    try:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        with pytest.raises(RuntimeError, match="injected"):
            projection_wgrad(a, g)
        assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_DEVICE
    finally:
        cublas.setPointerMode(handle, original_mode)


@pytest.mark.parametrize("gradient_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("nondefault", [False, True])
def test_current_stream_ordering(hip, gradient_dtype, nondefault):
    from cupy_backends.cuda.libs import cublas

    rng = np.random.default_rng(108)
    ah = rng.normal(0, 0.1, (513, 17)).astype(np.float32)
    gh = rng.normal(0, 0.1, (513, 13)).astype(np.float32)
    stream = cp.cuda.Stream(non_blocking=True) if nondefault else cp.cuda.Stream.null
    decoy = cp.cuda.Stream(non_blocking=True)
    handle = cp.cuda.device.get_cublas_handle()
    with stream:
        a = cp.asarray(ah).astype("bfloat16")
        g = cp.asarray(gh).astype(gradient_dtype)
        # Deliberately stale borrowed-handle stream; helper must bind current.
        cublas.setStream(handle, decoy.ptr)
        result = projection_wgrad(a, g)
        assert cublas.getStream(handle) == stream.ptr
        consumed = result * cp.float32(2)
        ready = cp.cuda.Event()
        ready.record()
    with cp.cuda.Stream(non_blocking=True) as consumer:
        consumer.wait_event(ready)
        copied = consumed.copy()
        consumer.synchronize()
    ah = cp.asnumpy(a.astype(cp.float32)).astype(np.float64)
    gh = cp.asnumpy(g.astype(cp.float32)).astype(np.float64)
    _check(cp.asnumpy(copied), 2 * (ah.T @ gh),
           f"stream/nondefault={nondefault}/{gradient_dtype}")


def test_cuda_is_rejected(monkeypatch):
    # Also runs on CUDA; on HIP it checks dispatch without invoking a CUDA GEMM.
    monkeypatch.setattr(cp.cuda.runtime, "is_hip", False)
    with pytest.raises(RuntimeError, match="requires HIP.*CUDA"):
        projection_wgrad(None, None)
