"""0085I: avoid needless attention-head copies and unused router gradient."""

import numpy as np

from mini_llm.backend import RandomStream
from mini_llm.config import AttentionLayerConfig, LocalAttentionConfig
from mini_llm.ops.attention import GQAAttention


def test_contiguous_head_selection_is_view_and_sparse_selection_preserves_order():
    rng = np.random.default_rng(801)
    q = rng.normal(size=(2, 7, 8, 16)).astype(np.float32)

    contiguous = GQAAttention._head_selection((1, 2, 3, 4, 5))
    assert contiguous == slice(1, 6)
    assert np.shares_memory(q, q[:, :, contiguous, :])
    np.testing.assert_array_equal(q[:, :, contiguous, :], q[:, :, (1, 2, 3, 4, 5), :])

    noncontiguous = GQAAttention._head_selection((1, 3, 5))
    assert noncontiguous == (1, 3, 5)
    np.testing.assert_array_equal(q[:, :, noncontiguous, :], q[:, :, (1, 3, 5), :])


def test_no_retrieval_returns_no_router_gradient_and_layer_gradients_are_finite():
    cfg = AttentionLayerConfig(heads=(
        LocalAttentionConfig(window=64),
        LocalAttentionConfig(window=8),
        LocalAttentionConfig(window=8),
        LocalAttentionConfig(window=8),
    ))
    attn = GQAAttention(
        d_model=32, n_q_heads=4, n_kv_heads=2, d_head=8,
        input_std=0.03, output_std=0.03, rng=RandomStream(802),
        dtype="float32", attention_config=cfg,
    )
    rng = np.random.default_rng(803)
    x = rng.normal(size=(2, 19, 32)).astype(np.float32)
    dout = rng.normal(size=x.shape).astype(np.float32)
    y, cache = attn.forward(x, return_cache=True)
    assert y.shape == x.shape

    dq, dk, dv, drouter = attn._mixed_context_backward(
        rng.normal(size=cache["q"].shape).astype(np.float32), cache
    )
    assert drouter is None
    assert dq.shape == cache["q"].shape
    assert dk.shape == cache["k"].shape
    assert dv.shape == cache["v"].shape

    attn.zero_grad()
    dx = attn.backward(dout, cache)
    assert dx.shape == x.shape
    assert np.isfinite(dx).all()
    assert all(np.isfinite(param.grad).all() for param in attn.parameters())
