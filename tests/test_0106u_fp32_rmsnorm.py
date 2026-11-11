"""Focused FP32 inference RMSNorm contracts (small tensors; no model loads)."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops import rmsnorm_inference as impl
from mini_llm.ops.rmsnorm_inference import RMSNormInference


RTOL = ATOL = 2e-6  # Existing FP32 inference tolerances, including tree reduction.


def _fp32_graph(x, gamma, eps):
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        mean_sq = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
        inv_rms = np.float32(1) / np.sqrt(mean_sq + np.float32(eps))
        return (x * inv_rms) * gamma


def _fp64_reference(x, gamma, eps):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * gamma


@pytest.fixture(scope="module")
def compare():
    maxima = {"fp32": [0.0, 0.0], "fp64": [0.0, 0.0]}

    def check(actual, expected, reference="fp32"):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        finite = np.isfinite(actual) & np.isfinite(expected)
        if np.any(finite):
            a = actual[finite].astype(np.float64)
            b = expected[finite].astype(np.float64)
            error = np.abs(a - b)
            maxima[reference][0] = max(maxima[reference][0], float(error.max()))
            nonzero = b != 0
            if np.any(nonzero):
                maxima[reference][1] = max(
                    maxima[reference][1], float((error[nonzero] / np.abs(b[nonzero])).max())
                )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)

    yield check
    for reference, (absolute, relative) in maxima.items():
        print(f"0106U {BACKEND_NAME} vs {reference}: max_abs={absolute:.9g}, "
              f"max_rel_nonzero={relative:.9g}")


@pytest.fixture
def make_layer(monkeypatch):
    monkeypatch.delenv("MINI_LLM_DISABLE_FUSED_RMSNORM", raising=False)

    def make(x, gamma, eps=1e-6):
        layer = RMSNormInference(x.shape[-1], eps=eps)
        layer.set_weights(gamma)
        if BACKEND_NAME == "cupy":
            assert layer._can_use_fused_fp32(x)

            def unexpected_reference(_):
                pytest.fail("contiguous FP32 input unexpectedly used the reference path")

            monkeypatch.setattr(layer, "_forward_reference", unexpected_reference)
        else:
            assert not layer._can_use_fused_fp32(x)
        return layer

    return make


@pytest.mark.parametrize("width", [3, 32, 128, 2048])
@pytest.mark.parametrize("leading", [(), (1,), (2, 3)])
def test_fp32_shapes_reference_and_no_mutation(width, leading, make_layer, compare):
    rng = np.random.default_rng(106)
    x_host = rng.normal(size=leading + (width,)).astype(np.float32)
    gamma_host = rng.normal(size=width).astype(np.float32)
    x, gamma = xp.asarray(x_host), xp.asarray(gamma_host)
    layer = make_layer(x, gamma)
    assert layer.gamma is gamma
    y = layer.forward(x)
    assert y.shape == x.shape and y.dtype == xp.float32
    assert y is not x and y is not gamma
    actual = asnumpy(y)
    compare(actual, _fp32_graph(x_host, gamma_host, layer.eps))
    compare(actual, _fp64_reference(x_host, gamma_host, layer.eps), "fp64")
    np.testing.assert_array_equal(asnumpy(x), x_host)
    np.testing.assert_array_equal(asnumpy(gamma), gamma_host)


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-6, 0.25, 1e6])
def test_zero_tiny_large_and_epsilon(eps, make_layer, compare):
    # Each row has one magnitude so finite FP32 squares stay safely representable.
    scales = np.array([0, 1e-20, 1e-10, 1, 1e10, 1e18], dtype=np.float32)
    x_host = scales[:, None] * np.linspace(-1, 1, 32, dtype=np.float32)[None, :]
    gamma_host = np.linspace(-2, 2, 32, dtype=np.float32)
    x, gamma = xp.asarray(x_host), xp.asarray(gamma_host)
    layer = make_layer(x, gamma, eps)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        actual = asnumpy(layer.forward(x))
    expected = _fp32_graph(x_host, gamma_host, eps)
    if eps == 0 and BACKEND_NAME == "cupy":
        # CuPy flushes subnormal squares here, unlike NumPy. Preserve the
        # existing device graph's infinities rather than requiring gradual underflow.
        expected[1:2] = asnumpy(RMSNormInference._forward_reference(layer, x[1:2]))
    compare(actual, expected)
    # FP32 subnormal-square rounding is material at eps=0: do not replace its
    # computational graph by FP64 mathematics in that case.
    if eps > 0:
        compare(actual, _fp64_reference(x_host, gamma_host, eps), "fp64")
    np.testing.assert_array_equal(asnumpy(x), x_host)
    np.testing.assert_array_equal(asnumpy(gamma), gamma_host)


@pytest.mark.parametrize("eps", [0.0, 1e-6, -1.0])
def test_fp32_overflow_nonfinite_and_operation_order(eps, make_layer, compare):
    maximum = np.finfo(np.float32).max
    x_host = np.array([
        [1e20, -1e20, 1], [maximum, 1, -maximum], [np.inf, 0, 1],
        [np.nan, 1, 2], [0, 0, 0], [1, -1, 1],
    ], dtype=np.float32)
    # Multiplying x by gamma *before* normalizing would overflow even the last row.
    gamma_host = np.array([maximum, -maximum, 0], dtype=np.float32)
    x, gamma = xp.asarray(x_host), xp.asarray(gamma_host)
    layer = make_layer(x, gamma, eps)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        actual = asnumpy(layer.forward(x))
    compare(actual, _fp32_graph(x_host, gamma_host, eps))
    np.testing.assert_array_equal(asnumpy(x), x_host)
    np.testing.assert_array_equal(asnumpy(gamma), gamma_host)


@pytest.mark.parametrize("case", [
    "strided_x", "reversed_x", "transposed_x", "fortran_x", "strided_gamma",
    "broadcast_gamma", "float64_x", "float64_gamma", "float16_gamma",
    "empty_rows", "wrong_d_model",
])
def test_unusual_inputs_use_unchanged_fallback(case, monkeypatch):
    monkeypatch.delenv("MINI_LLM_DISABLE_FUSED_RMSNORM", raising=False)
    x = xp.asarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10)
    gamma = xp.asarray(np.linspace(-1, 1, 4, dtype=np.float32))
    d_model = 4
    if case == "strided_x":
        x, gamma, d_model = x[..., ::2], gamma[::2].copy(), 2
    elif case == "reversed_x":
        x = x[..., ::-1]
    elif case == "transposed_x":
        x = x.transpose(1, 0, 2)
    elif case == "fortran_x":
        x = xp.asfortranarray(x)
    elif case == "strided_gamma":
        gamma = xp.arange(8, dtype=xp.float32)[::2]
    elif case == "broadcast_gamma":
        gamma = gamma[None, :]
    elif case == "float64_x":
        x = x.astype(xp.float64)
    elif case == "float64_gamma":
        gamma = gamma.astype(xp.float64)
    elif case == "float16_gamma":
        gamma = gamma.astype(xp.float16)
    elif case == "empty_rows":
        x = x[:0]
    elif case == "wrong_d_model":
        d_model = 5
    layer = RMSNormInference(d_model)
    layer.set_weights(gamma)
    assert not layer._can_use_fused_fp32(x)
    expected = layer._forward_reference(x)
    x_before, gamma_before = asnumpy(x).copy(), asnumpy(gamma).copy()

    def unexpected_kernel(_):
        pytest.fail("unusual input dispatched to the FP32 kernel")

    monkeypatch.setattr(layer, "_forward_fused_fp32", unexpected_kernel)
    actual = layer.forward(x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    np.testing.assert_array_equal(asnumpy(x), x_before)
    np.testing.assert_array_equal(asnumpy(gamma), gamma_before)


@pytest.mark.parametrize("optout", ["1", "true", "YES", "On"])
def test_existing_optout_disables_fp32_kernel(optout, monkeypatch):
    monkeypatch.setenv("MINI_LLM_DISABLE_FUSED_RMSNORM", optout)
    x = xp.ones((2, 3, 32), dtype=xp.float32)
    layer = RMSNormInference(32)
    layer.set_weights(xp.ones(32, dtype=xp.float32))
    assert not layer._fused_fp32_enabled
    assert not layer._can_use_fused_fp32(x)

    def unexpected_compile():
        pytest.fail("optout attempted to compile the FP32 kernel")

    monkeypatch.setattr(impl, "_get_fused_fp32_rmsnorm_kernel", unexpected_compile)
    np.testing.assert_array_equal(asnumpy(layer.forward(x)), asnumpy(layer._forward_reference(x)))


@pytest.mark.gpu
@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy current stream")
def test_fp32_uses_current_nondefault_stream(make_layer, monkeypatch, compare):
    stream = xp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(107)
    x_host = rng.normal(size=(2, 3, 128)).astype(np.float32)
    gamma_host = rng.normal(size=128).astype(np.float32)
    kernel = impl._get_fused_fp32_rmsnorm_kernel()
    launches = []

    def checked_launch(*args, **kwargs):
        assert xp.cuda.get_current_stream().ptr == stream.ptr
        assert kwargs.get("stream") is None  # RawKernel must inherit the current stream.
        launches.append(1)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(impl, "_get_fused_fp32_rmsnorm_kernel", lambda: checked_launch)
    with stream:
        # Input production, normalization, and consumption all share this stream.
        x = xp.asarray(x_host) * xp.float32(2)
        gamma = xp.asarray(gamma_host)
        layer = make_layer(x, gamma)
        y = layer.forward(x)
        consumed = y * xp.float32(0.5)
    stream.synchronize()  # Tests may wait; the inference implementation must not.
    assert launches == [1]
    compare(asnumpy(consumed), _fp32_graph(x_host * np.float32(2), gamma_host, layer.eps) * 0.5)
    compare(asnumpy(consumed), _fp64_reference(x_host * np.float32(2), gamma_host, layer.eps) * 0.5, "fp64")
