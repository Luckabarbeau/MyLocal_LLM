"""0079O direct indexed K/V backward: reference invariance and GPU parity.

GPU tests require MINI_LLM_BACKEND=cupy and a CUDA BF16-capable device.
"""

import os

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, resolve_dtype, xp
from mini_llm.ops.attention_selection import KeySelectionPlan
from mini_llm.ops.indexed_attention import (
    _direct_indexed_bwd_enabled,
    indexed_attention_backward,
    indexed_attention_forward,
)


def _example(seed=79):
    rng = np.random.default_rng(seed)
    q = xp.asarray(rng.normal(size=(2, 4, 4, 8)), dtype="float64")
    k = xp.asarray(rng.normal(size=(2, 7, 2, 8)), dtype="float64")
    v = xp.asarray(rng.normal(size=(2, 7, 2, 8)), dtype="float64")
    indices = xp.asarray(rng.integers(0, 7, size=(2, 1, 4, 5)), dtype=xp.int64)
    indices[:, :, :, 1] = indices[:, :, :, 0]  # deliberate repeated keys
    valid = xp.asarray(rng.random(size=indices.shape) > 0.22, dtype=bool)
    valid[:, :, 0, :] = False  # fully masked rows
    plan = KeySelectionPlan(indices, valid)
    dcontext = xp.asarray(rng.normal(size=q.shape), dtype="float64")
    return q, k, v, plan, dcontext


def test_0079o_opt_in_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MINI_LLM_DIRECT_INDEXED_BWD", raising=False)
    assert not _direct_indexed_bwd_enabled()


def test_0079o_non_cuda_reference_remains_identical_with_flag(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU fallback test")
    q, k, v, plan, dc = _example()
    _, cache = indexed_attention_forward(q, k, v, plan, query_chunk_size=4)
    monkeypatch.delenv("MINI_LLM_DIRECT_INDEXED_BWD", raising=False)
    reference = indexed_attention_backward(dc, cache)
    monkeypatch.setenv("MINI_LLM_DIRECT_INDEXED_BWD", "1")
    result = indexed_attention_backward(dc, cache)
    for expected, actual in zip(reference[:3], result[:3]):
        np.testing.assert_array_equal(actual, expected)
    assert result[3] is None


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CUDA GPU required")
@pytest.mark.parametrize("logit_bias", [False, True])
@pytest.mark.parametrize("query_chunk", [1, 4])
def test_0079o_direct_cuda_bf16_matches_reference(
    monkeypatch, logit_bias, query_chunk,
):
    rng = np.random.default_rng(790)
    bf16 = resolve_dtype("bfloat16")
    q = xp.asarray(rng.normal(size=(2, 4, 4, 64)), dtype=bf16)
    k = xp.asarray(rng.normal(size=(2, 16, 2, 64)), dtype=bf16)
    v = xp.asarray(rng.normal(size=(2, 16, 2, 64)), dtype=bf16)
    idx = np.broadcast_to(
        rng.integers(0, 16, size=(2, 1, 4, 9)), (2, 4, 4, 9)
    ).copy()
    idx[..., 2] = idx[..., 1]  # test repeated key accumulation
    valid = rng.random(size=idx.shape) > 0.2
    valid[:, :, 0, :] = False  # all-invalid rows
    bias = (
        xp.asarray(rng.normal(size=idx.shape) * 0.2, dtype=xp.float32)
        if logit_bias else None
    )
    plan = KeySelectionPlan(
        xp.asarray(idx, dtype=xp.int64), xp.asarray(valid), bias,
    )
    kv_map = xp.asarray([1, 0, 1, 0], dtype=xp.int64)
    _, cache = indexed_attention_forward(
        q, k, v, plan, kv_head_indices=kv_map,
        query_chunk_size=query_chunk,
    )
    dc = xp.asarray(rng.normal(size=q.shape), dtype=xp.float32)
    monkeypatch.setenv("MINI_LLM_DIRECT_INDEXED_BWD", "0")
    expected = indexed_attention_backward(dc, cache)
    monkeypatch.setenv("MINI_LLM_DIRECT_INDEXED_BWD", "1")
    actual = indexed_attention_backward(dc, cache)
    xp.cuda.Stream.null.synchronize()
    for name, e, a in zip(("dq", "dk", "dv"), expected[:3], actual[:3]):
        np.testing.assert_allclose(
            xp.asnumpy(a), xp.asnumpy(e), rtol=1e-3, atol=4e-4,
            err_msg=name,
        )
    if logit_bias:
        np.testing.assert_allclose(
            xp.asnumpy(actual[3]), xp.asnumpy(expected[3]), rtol=0, atol=0,
        )
    else:
        assert actual[3] is expected[3] is None


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CUDA GPU required")
def test_0079o_large_key_selection_does_not_change_gradients(monkeypatch):
    rng = np.random.default_rng(791)
    bf16 = resolve_dtype("bfloat16")
    batch, q_length, n_heads, n_kv_heads, d_head, keys = 3, 4, 2, 1, 64, 256
    q = xp.asarray(rng.normal(size=(batch, q_length, n_heads, d_head)), dtype=bf16)
    k = xp.asarray(rng.normal(size=(batch, keys, n_kv_heads, d_head)), dtype=bf16)
    v = xp.asarray(rng.normal(size=(batch, keys, n_kv_heads, d_head)), dtype=bf16)
    idx = xp.broadcast_to(
        xp.arange(keys, dtype=xp.int64)[None, None, None, :],
        (batch, n_heads, q_length, keys),
    )
    valid = xp.ones(idx.shape, dtype=bool)
    plan = KeySelectionPlan(idx, valid)
    _, cache = indexed_attention_forward(q, k, v, plan, query_chunk_size=4)
    dc = xp.asarray(rng.normal(size=q.shape), dtype=xp.float32)
    monkeypatch.setenv("MINI_LLM_DIRECT_INDEXED_BWD", "0")
    ref = indexed_attention_backward(dc, cache)
    monkeypatch.setenv("MINI_LLM_DIRECT_INDEXED_BWD", "1")
    opt = indexed_attention_backward(dc, cache)
    for expected, actual in zip(ref[:3], opt[:3]):
        np.testing.assert_allclose(
            xp.asnumpy(actual), xp.asnumpy(expected), rtol=2e-3, atol=5e-4,
        )
