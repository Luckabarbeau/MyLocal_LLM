"""PATCH 0089C contracts for BF16 expert input-gradient GEMM accumulation."""

from __future__ import annotations

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream, asnumpy, xp
from mini_llm.ops.experts import ExpertFFN


def _expert(dtype):
    return ExpertFFN(
        d_model=32,
        d_ff=64,
        input_std=0.03,
        output_std=0.02,
        rng=RandomStream(89),
        name="0089c.expert",
        dtype=dtype,
    )


def _run(expert, x, dy, monkeypatch, enabled):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", "0")
    monkeypatch.setenv("MINI_LLM_EXPERT_DX_ACCUMULATE", str(int(enabled)))
    expert.zero_grad()
    output, cache = expert.forward(x)
    d_input = expert.backward(dy, cache)
    if BACKEND_NAME == "cupy":
        xp.cuda.get_current_stream().synchronize()
    return (
        asnumpy(output).copy(),
        asnumpy(d_input).copy(),
        [asnumpy(parameter.grad).copy() for parameter in expert.parameters()],
    )


def _relative_l2(actual, expected):
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = np.linalg.norm(np.asarray(expected, dtype=np.float64).reshape(-1))
    return float(np.linalg.norm(difference.reshape(-1)) / max(float(denominator), 1e-300))


def test_non_cupy_or_non_bf16_path_is_unchanged(monkeypatch):
    expert = _expert("float32")
    generator = np.random.default_rng(90)
    x = xp.asarray(generator.normal(size=(17, 32)), dtype=xp.float32)
    dy = xp.asarray(generator.normal(size=(17, 32)), dtype=xp.float32)
    reference = _run(expert, x, dy, monkeypatch, False)
    candidate = _run(expert, x, dy, monkeypatch, True)
    np.testing.assert_array_equal(candidate[0], reference[0])
    np.testing.assert_array_equal(candidate[1], reference[1])
    for actual, expected in zip(candidate[2], reference[2]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy BF16")
def test_bf16_matmul_into_beta_accumulates(monkeypatch):
    del monkeypatch
    from mini_llm.ops.cublas_fp16_gemm import (
        bf16_matmul_into,
        prepare_fp16_gemm_context,
    )

    generator = np.random.default_rng(91)
    a = xp.asarray(generator.normal(size=(19, 37)), dtype=xp.float32).astype("bfloat16")
    b = xp.asarray(generator.normal(size=(37, 23)), dtype=xp.float32).astype("bfloat16")
    c = xp.asarray(generator.normal(size=(19, 37)), dtype=xp.float32).astype("bfloat16")
    d = xp.asarray(generator.normal(size=(37, 23)), dtype=xp.float32).astype("bfloat16")
    expected = a @ b + c @ d
    actual = xp.empty_like(expected)
    context = prepare_fp16_gemm_context(xp.cuda.get_current_stream())
    bf16_matmul_into(a, b, actual, context=context)
    bf16_matmul_into(c, d, actual, context=context, beta=1.0)
    xp.cuda.get_current_stream().synchronize()
    actual_host = asnumpy(actual).astype(np.float32)
    expected_host = asnumpy(expected).astype(np.float32)
    assert _relative_l2(actual_host, expected_host) <= 0.01


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy BF16")
def test_expert_bf16_accumulated_dx_matches_existing_path(monkeypatch):
    expert = _expert("bfloat16")
    generator = np.random.default_rng(92)
    x = xp.asarray(generator.normal(size=(71, 32)), dtype=xp.float32).astype("bfloat16")
    dy = xp.asarray(generator.normal(size=(71, 32)), dtype=xp.float32).astype("bfloat16")
    reference = _run(expert, x, dy, monkeypatch, False)
    candidate = _run(expert, x, dy, monkeypatch, True)

    np.testing.assert_array_equal(candidate[0], reference[0])
    assert _relative_l2(candidate[1], reference[1]) <= 0.01
    for actual, expected in zip(candidate[2], reference[2]):
        np.testing.assert_array_equal(actual, expected)
