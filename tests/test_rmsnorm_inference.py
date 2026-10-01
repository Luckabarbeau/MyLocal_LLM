"""Tests for inference-only RMSNorm, including the fused CuPy FP16 path."""

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, asnumpy, synchronize
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.rmsnorm_inference import RMSNormInference


def _reference_rmsnorm(x, gamma, eps):
    x32 = np.asarray(x, dtype=np.float32)
    gamma32 = np.asarray(gamma, dtype=np.float32)
    mean_sq = np.mean(x32 * x32, axis=-1, keepdims=True)
    return x32 * (1.0 / np.sqrt(mean_sq + eps)) * gamma32


def test_rmsnorm_inference_float32_matches_reference():
    rng = np.random.default_rng(123)
    x_np = rng.normal(size=(2, 3, 16)).astype(np.float32)
    gamma_np = rng.normal(loc=1.0, scale=0.1, size=(16,)).astype(np.float32)

    layer = RMSNormInference(16, eps=1e-6, dtype="float32")
    layer.set_weights(xp.asarray(gamma_np))
    y = layer.forward(xp.asarray(x_np))

    expected = _reference_rmsnorm(x_np, gamma_np, 1e-6)
    np.testing.assert_allclose(asnumpy(y), expected, rtol=2e-6, atol=2e-6)


def test_rmsnorm_inference_float16_matches_fp32_reference():
    rng = np.random.default_rng(456)
    x_np = rng.normal(size=(4, 7, 512)).astype(np.float16)
    gamma_np = rng.normal(loc=1.0, scale=0.1, size=(512,)).astype(np.float16)

    layer = RMSNormInference(512, eps=1e-6, dtype="float16")
    layer.set_weights(xp.asarray(gamma_np))
    y = layer.forward(xp.asarray(x_np))
    synchronize()

    y_np = asnumpy(y)
    expected = _reference_rmsnorm(x_np, gamma_np, 1e-6).astype(np.float16)

    assert y_np.dtype == np.float16
    assert y_np.shape == x_np.shape
    # Both paths use FP32 arithmetic followed by one FP16 output rounding.
    # The fused tree reduction can differ by a few FP32 ulps from np.mean.
    np.testing.assert_allclose(y_np, expected, rtol=2e-3, atol=2e-3)


def test_inference_rmsnorm_matches_training_forward_float16():
    rng = np.random.default_rng(654)
    x = xp.asarray(rng.normal(size=(2, 5, 512)).astype(np.float16))
    gamma = xp.asarray(
        rng.normal(loc=1.0, scale=0.1, size=(512,)).astype(np.float16)
    )

    training = RMSNorm(512, eps=1e-6, dtype="float16")
    training.gamma.data[...] = gamma
    inference = RMSNormInference(512, eps=1e-6, dtype="float16")
    inference.set_weights(training.gamma.data)

    y_training, _ = training.forward(x)
    y_inference = inference.forward(x)
    synchronize()

    np.testing.assert_allclose(
        asnumpy(y_inference),
        asnumpy(y_training),
        rtol=2e-3,
        atol=2e-3,
    )


def test_rmsnorm_inference_fused_and_reference_agree_on_cupy():
    if BACKEND_NAME != "cupy":
        return

    rng = np.random.default_rng(789)
    x = xp.asarray(rng.normal(size=(3, 5, 512)).astype(np.float16))
    gamma = xp.asarray(rng.normal(loc=1.0, scale=0.1, size=(512,)).astype(np.float16))

    fused = RMSNormInference(512, eps=1e-6, dtype="float16")
    fused.set_weights(gamma)

    reference = RMSNormInference(512, eps=1e-6, dtype="float16")
    reference.set_weights(gamma)
    reference._fused_fp16_enabled = False

    y_fused = fused.forward(x)
    y_reference = reference.forward(x)
    synchronize()

    np.testing.assert_allclose(
        asnumpy(y_fused),
        asnumpy(y_reference),
        rtol=2e-3,
        atol=2e-3,
    )
