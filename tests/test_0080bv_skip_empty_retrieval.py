"""0080BV no-route bypass must preserve exact attention outputs/gradients."""
import numpy as np
import pytest

from mini_llm.backend import RandomStream, xp
from mini_llm.config import (
    AttentionLayerConfig, ContextRouterConfig,
    LocalAttentionConfig, RetrievalAttentionConfig,
)
from mini_llm.ops.attention import GQAAttention, _has_eligible_retrieval_route


def _attention():
    config = ContextRouterConfig(
        history_block_size=2, routing_stride=2, query_window=2,
        router_dim=3, top_k_blocks=2, exclude_recent_tokens=2,
        query_pooling="learned", history_pooling="mean", num_queries=1,
    )
    attention = GQAAttention(
        d_model=4, n_q_heads=2, n_kv_heads=1, d_head=2,
        input_std=0.15, output_std=0.12, rng=RandomStream(8120),
        dtype="float64",
        attention_config=AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=8),
            RetrievalAttentionConfig(config, group="far"),
        )),
    )
    return attention, config


def _run(attention, x, dy, monkeypatch, enabled):
    monkeypatch.setenv("MINI_LLM_SKIP_EMPTY_RETRIEVAL", "1" if enabled else "0")
    attention.zero_grad()
    out, cache = attention.forward(x, return_cache=True)
    routing = cache["routing"]["far"]
    dx = attention.backward(dy, cache)
    grads = tuple(np.asarray(param.grad).copy() for param in attention.parameters())
    return np.asarray(out).copy(), np.asarray(dx).copy(), grads, routing


@pytest.mark.parametrize("length", [1, 2, 4, 6])
def test_empty_routes_match_reference_forward_backward(monkeypatch, length):
    attention, config = _attention()
    assert not _has_eligible_retrieval_route(length, config)
    rng = np.random.default_rng(8121 + length)
    x = xp.asarray(rng.normal(size=(2, length, 4)), dtype="float64")
    dy = xp.asarray(rng.normal(size=x.shape), dtype="float64")
    off = _run(attention, x, dy, monkeypatch, enabled=False)
    on = _run(attention, x, dy, monkeypatch, enabled=True)
    np.testing.assert_array_equal(on[0], off[0])
    np.testing.assert_array_equal(on[1], off[1])
    for grad_off, grad_on in zip(off[2], on[2]):
        np.testing.assert_array_equal(grad_off, grad_on)
    assert tuple(on[3]["weights"].shape) == tuple(off[3]["weights"].shape)
    np.testing.assert_array_equal(np.asarray(on[3]["selected_blocks"]),
                                  np.asarray(off[3]["selected_blocks"]))
    np.testing.assert_array_equal(np.asarray(on[3]["route_starts"]),
                                  np.asarray(off[3]["route_starts"]))


def test_skip_does_not_call_retrieval_module_on_no_route(monkeypatch):
    attention, _ = _attention()
    module = attention._retrieval_groups[0]["module"]

    def unreachable(*args, **kwargs):
        raise AssertionError("retrieval should have been bypassed")

    monkeypatch.setattr(module, "forward", unreachable)
    monkeypatch.setenv("MINI_LLM_SKIP_EMPTY_RETRIEVAL", "1")
    x = xp.asarray(np.ones((1, 6, 4)), dtype="float64")
    out = attention.forward(x, return_cache=False)
    assert out.shape == x.shape


@pytest.mark.parametrize("length", [7, 9, 14])
def test_eligible_routes_still_match_reference(monkeypatch, length):
    attention, config = _attention()
    assert _has_eligible_retrieval_route(length, config)
    rng = np.random.default_rng(8150 + length)
    x = xp.asarray(rng.normal(size=(1, length, 4)), dtype="float64")
    dy = xp.asarray(rng.normal(size=x.shape), dtype="float64")
    off = _run(attention, x, dy, monkeypatch, enabled=False)
    on = _run(attention, x, dy, monkeypatch, enabled=True)
    np.testing.assert_array_equal(on[0], off[0])
    np.testing.assert_array_equal(on[1], off[1])
    for grad_off, grad_on in zip(off[2], on[2]):
        np.testing.assert_array_equal(grad_off, grad_on)


def test_first_eligible_route_matches_stride_alignment():
    config = ContextRouterConfig(
        history_block_size=128, routing_stride=128,
        query_window=512, router_dim=32,
        top_k_blocks=4, exclude_recent_tokens=1024,
    )
    assert not _has_eligible_retrieval_route(1536, config)
    assert _has_eligible_retrieval_route(1537, config)
    assert _has_eligible_retrieval_route(4096, config)
