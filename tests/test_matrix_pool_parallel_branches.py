"""Regression tests for Model B direct shared-input local/global attention."""

from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import ModelConfig
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention


def _attention(monkeypatch, mode):
    # Branch topology is architectural state carried by the configuration, not
    # an ambient environment knob: the operator reads config.attention_branch_mode.
    monkeypatch.delenv("MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES", raising=False)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    config = replace(
        base, block_size=4, local_window=4, long_local_window=8,
        long_local_heads=1, summaries_per_block=2, global_heads=2,
        global_head_dim=4, max_blocks=8, global_query_chunk=4,
        attention_branch_mode="parallel" if mode else "legacy",
    )
    return MatrixPoolDualResolutionAttention(
        d_model=16, n_q_heads=4, n_kv_heads=2, d_head=4,
        input_std=0.08, output_std=0.06, rng=RandomStream(4301),
        dtype="float64", config=config,
    )


def _fixture():
    rng = np.random.default_rng(4302)
    x = rng.normal(scale=0.35, size=(1, 9, 16))
    residual = rng.normal(scale=0.5, size=x.shape)
    dy = rng.normal(size=x.shape)
    metadata = build_dual_resolution_metadata(
        np.arange(x.shape[1], dtype=np.int64)[None, :],
        eos_token_id=99, block_size=4,
    )
    return x, residual, dy, metadata


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU float64 numerical test")
def test_parallel_global_is_independent_of_residual_and_legacy_is_not(monkeypatch):
    x, residual, _dy, metadata = _fixture()
    parallel = _attention(monkeypatch, True)
    assert parallel.parallel_branches
    p1 = parallel.forward(
        x, residual_input=residual, dual_resolution_metadata=metadata,
        return_cache=False,
    )
    p2 = parallel.forward(
        x, residual_input=residual * 9 + 1,
        dual_resolution_metadata=metadata, return_cache=False,
    )
    np.testing.assert_array_equal(p1, p2)

    legacy = _attention(monkeypatch, False)
    assert not legacy.parallel_branches
    s1 = legacy.forward(
        x, residual_input=residual, dual_resolution_metadata=metadata,
        return_cache=False,
    )
    s2 = legacy.forward(
        x, residual_input=residual * 9 + 1,
        dual_resolution_metadata=metadata, return_cache=False,
    )
    assert np.max(np.abs(s1 - s2)) > 1e-8


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU float64 numerical test")
def test_parallel_backward_matches_input_finite_differences(monkeypatch):
    x, residual, dy, metadata = _fixture()
    operator = _attention(monkeypatch, True)
    output, cache = operator.forward(
        x, residual_input=residual, dual_resolution_metadata=metadata,
    )
    operator.zero_grad()
    dx, dr = operator.backward(dy, cache)
    np.testing.assert_array_equal(dr, np.zeros_like(dr))
    assert np.linalg.norm(operator.W_global_q.grad) > 0
    assert np.linalg.norm(operator.W_pool_score.grad) > 0

    eps = 2e-6
    for index in [(0, 0, 0), (0, 3, 7), (0, 5, 3), (0, 7, 15)]:
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        pos = operator.forward(
            plus, residual_input=residual, dual_resolution_metadata=metadata,
            return_cache=False,
        )
        neg = operator.forward(
            minus, residual_input=residual, dual_resolution_metadata=metadata,
            return_cache=False,
        )
        numeric = np.sum((pos - neg) * dy) / (2 * eps)
        np.testing.assert_allclose(dx[index], numeric, rtol=3e-4, atol=1e-5)
