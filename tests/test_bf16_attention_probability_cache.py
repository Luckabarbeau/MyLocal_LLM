import numpy as np
import ml_dtypes

from mini_llm.backend import xp
from mini_llm.ops.indexed_attention import (
    block_retrieval_attention_backward,
    block_retrieval_attention_forward,
    dilated_attention_backward,
    dilated_attention_forward,
    global_sparse_attention_backward,
    global_sparse_attention_forward,
    local_window_attention_backward,
    local_window_attention_forward,
)


def _bf16_arrays(seed=10, seq=8, hq=4, hkv=2, dh=3):
    rng = np.random.default_rng(seed)
    dtype = ml_dtypes.bfloat16
    q = xp.asarray(rng.normal(size=(1, seq, hq, dh)), dtype=dtype)
    k = xp.asarray(rng.normal(size=(1, seq, hkv, dh)), dtype=dtype)
    v = xp.asarray(rng.normal(size=(1, seq, hkv, dh)), dtype=dtype)
    return q, k, v


def _assert_bf16(x):
    assert str(np.dtype(x.dtype)).lower() == "bfloat16"


def _assert_finite(*arrays):
    for array in arrays:
        assert bool(xp.all(xp.isfinite(array)))


def test_bf16_local_attention_caches_probabilities_in_bf16():
    q, k, v = _bf16_arrays(seed=11)
    out, cache = local_window_attention_forward(
        q, k, v, window=4, kv_head_indices=(0, 0, 1, 1), query_chunk_size=3
    )
    for *_, group_probs in cache["chunks"]:
        for _, _, probs in group_probs:
            _assert_bf16(probs)
    dq, dk, dv = local_window_attention_backward(xp.ones_like(out), cache)
    _assert_finite(out, dq, dk, dv)


def test_bf16_dilated_attention_caches_probabilities_in_bf16():
    q, k, v = _bf16_arrays(seed=12, hq=2, hkv=1)
    out, cache = dilated_attention_forward(
        q, k, v, window=8, dilation=2, offset=0, kv_head_indices=(0, 0),
        query_chunk_size=3,
    )
    for chunk in cache["chunks"]:
        probs = chunk[6]
        if probs is not None:
            _assert_bf16(probs)
    dq, dk, dv = dilated_attention_backward(xp.ones_like(out), cache)
    _assert_finite(out, dq, dk, dv)


def test_bf16_global_attention_caches_probabilities_in_bf16():
    q, k, v = _bf16_arrays(seed=13, hq=2, hkv=1)
    out, cache = global_sparse_attention_forward(
        q, k, v, stride=2, offset=0, include_current=True,
        kv_head_indices=(0, 0),
    )
    _assert_bf16(cache["probs"])
    dq, dk, dv = global_sparse_attention_backward(xp.ones_like(out), cache)
    _assert_finite(out, dq, dk, dv)


def test_bf16_block_retrieval_caches_probabilities_in_bf16():
    q, k, v = _bf16_arrays(seed=14, hq=2, hkv=1)
    selected = xp.asarray(
        [[[[0, 1], [1, 0]], [[0, 1], [1, 0]]]], dtype=xp.int64
    )
    weights = xp.asarray(
        [[[[0.4, 0.6], [0.7, 0.3]], [[0.55, 0.45], [0.2, 0.8]]]],
        dtype=ml_dtypes.bfloat16,
    )
    route_starts = xp.asarray([4, 6], dtype=xp.int64)
    out, cache = block_retrieval_attention_forward(
        q, k, v, selected, weights, route_starts,
        block_size=2, routing_stride=2, kv_head_indices=(0, 0),
        weight_mode="logit_bias", return_cache=True,
    )
    _assert_bf16(cache["probs"])
    dq, dk, dv, dw = block_retrieval_attention_backward(xp.ones_like(out), cache)
    _assert_finite(out, dq, dk, dv, dw)
