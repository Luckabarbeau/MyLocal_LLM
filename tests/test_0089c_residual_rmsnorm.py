"""PATCH 0089C contracts for training residual/RMSNorm boundary fusion."""

from __future__ import annotations

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.rmsnorm import RMSNorm


def _relative_l2(actual, expected):
    actual_f64 = np.asarray(actual, dtype=np.float64)
    expected_f64 = np.asarray(expected, dtype=np.float64)
    difference = actual_f64 - expected_f64
    denominator = np.linalg.norm(expected_f64.reshape(-1))
    return float(
        np.linalg.norm(difference.reshape(-1)) / max(float(denominator), 1e-300)
    )


def test_residual_compute_fallback_preserves_reference_contract(monkeypatch):
    monkeypatch.delenv("MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM", raising=False)
    generator = np.random.default_rng(891)
    residual = xp.asarray(generator.normal(size=(2, 3, 8)), dtype=xp.float32)
    branch = xp.asarray(generator.normal(size=(2, 3, 8)), dtype=xp.float32)
    dy = xp.asarray(generator.normal(size=(2, 3, 8)), dtype=xp.float32)

    reference = RMSNorm(8, eps=1e-6, name="0089c.reference", dtype="float32")
    candidate = RMSNorm(8, eps=1e-6, name="0089c.candidate", dtype="float32")
    gamma = xp.asarray(generator.normal(loc=1.0, scale=0.1, size=(8,)), dtype=xp.float32)
    reference.gamma.data[...] = gamma
    candidate.gamma.data[...] = gamma

    expected_residual = residual + branch
    expected_output, expected_cache = reference.forward_compute(
        expected_residual, xp.float32
    )
    expected_dx = reference.backward_residual(dy, expected_cache)

    actual_residual, actual_output, actual_cache = candidate.forward_residual_compute(
        residual, branch, xp.float32
    )
    actual_dx = candidate.backward_residual(dy, actual_cache)

    np.testing.assert_array_equal(asnumpy(actual_residual), asnumpy(expected_residual))
    np.testing.assert_array_equal(asnumpy(actual_output), asnumpy(expected_output))
    np.testing.assert_array_equal(asnumpy(actual_cache["x"]), asnumpy(actual_residual))
    np.testing.assert_array_equal(
        asnumpy(actual_cache["inv_rms"]), asnumpy(expected_cache["inv_rms"])
    )
    np.testing.assert_array_equal(asnumpy(actual_dx), asnumpy(expected_dx))
    np.testing.assert_array_equal(
        asnumpy(candidate.gamma.grad), asnumpy(reference.gamma.grad)
    )


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy BF16")
def test_fused_bf16_residual_rmsnorm_matches_existing_forward_backward(monkeypatch):
    generator = np.random.default_rng(892)
    residual = xp.asarray(
        generator.normal(size=(3, 71, 1024)), dtype=xp.float32
    )
    branch = xp.asarray(
        generator.normal(size=(3, 71, 1024)), dtype=xp.float32
    ).astype("bfloat16")
    dy = xp.asarray(
        generator.normal(size=(3, 71, 1024)), dtype=xp.float32
    ).astype("bfloat16")
    gamma = xp.asarray(
        generator.normal(loc=1.0, scale=0.1, size=(1024,)), dtype=xp.float32
    ).astype("bfloat16")

    monkeypatch.setenv("MINI_LLM_FUSED_BF16_RMSNORM", "1")
    monkeypatch.setenv("MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM", "0")
    reference = RMSNorm(1024, eps=1e-6, name="0089c.reference", dtype="bfloat16")
    monkeypatch.setenv("MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM", "1")
    candidate = RMSNorm(1024, eps=1e-6, name="0089c.candidate", dtype="bfloat16")
    reference.gamma.data[...] = gamma
    candidate.gamma.data[...] = gamma

    expected_residual = residual + branch.astype(xp.float32)
    expected_output, expected_cache = reference.forward_compute(
        expected_residual, xp.dtype("bfloat16")
    )
    expected_dx = reference.backward_residual(dy, expected_cache)

    assert candidate.can_forward_residual_compute(
        residual, branch, xp.dtype("bfloat16")
    )
    actual_residual, actual_output, actual_cache = candidate.forward_residual_compute(
        residual, branch, xp.dtype("bfloat16")
    )
    actual_dx = candidate.backward_residual(dy, actual_cache)
    xp.cuda.get_current_stream().synchronize()

    actual_residual_host = asnumpy(actual_residual)
    expected_residual_host = asnumpy(expected_residual)
    np.testing.assert_array_equal(actual_residual_host, expected_residual_host)
    np.testing.assert_array_equal(asnumpy(actual_output), asnumpy(expected_output))
    np.testing.assert_allclose(
        asnumpy(actual_cache["inv_rms"]),
        asnumpy(expected_cache["inv_rms"]),
        rtol=2e-6,
        atol=2e-7,
    )
    assert actual_cache["x"] is actual_residual
    assert actual_cache["original_dtype"] == xp.float32
    assert actual_cache["fused_bf16_residual"] is True
    assert _relative_l2(asnumpy(actual_dx), asnumpy(expected_dx)) <= 2e-6
    assert _relative_l2(
        asnumpy(candidate.gamma.grad), asnumpy(reference.gamma.grad)
    ) <= 2e-6
