import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import (
    build_weighted_block_retrieval_plan,
    weighted_block_retrieval_bias_backward,
)
from mini_llm.ops.indexed_attention import (
    indexed_attention_forward,
    indexed_attention_backward,
    block_retrieval_attention_forward,
    block_retrieval_attention_backward,
)


def _fixture(shared_router=False):
    rng = np.random.default_rng(900)
    q = xp.asarray(rng.normal(size=(1, 12, 2, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    route_starts = xp.asarray([6, 8, 10], dtype=xp.int64)
    if shared_router:
        selected = xp.asarray([[[[0, 1]], [[0, 2]], [[1, 3]]]], dtype=xp.int64)
        weights = xp.asarray(
            [[[[0.35, 0.65]], [[0.6, 0.4]], [[0.55, 0.45]]]],
            dtype="float64",
        )
    else:
        selected = xp.asarray(
            [[
                [[0, 1], [1, 0]],
                [[0, 2], [2, 1]],
                [[1, 3], [3, 0]],
            ]],
            dtype=xp.int64,
        )
        weights = xp.asarray(
            [[
                [[0.35, 0.65], [0.7, 0.3]],
                [[0.6, 0.4], [0.45, 0.55]],
                [[0.55, 0.45], [0.2, 0.8]],
            ]],
            dtype="float64",
        )
    return q, k, v, selected, weights, route_starts


def _generic(q, k, v, selected, weights, route_starts):
    plan, plan_cache = build_weighted_block_retrieval_plan(
        selected,
        weights,
        route_starts,
        seq_len=q.shape[1],
        block_size=2,
        routing_stride=2,
        exclude_recent_tokens=2,
        weight_scale=0.7,
        weight_eps=1e-8,
    )
    out, cache = indexed_attention_forward(
        q, k, v, plan, kv_head_indices=(0, 0), return_cache=True
    )
    return out, cache, plan_cache


def test_block_retrieval_matches_generic_indexed_forward_backward():
    q, k, v, selected, weights, route_starts = _fixture(False)
    generic, generic_cache, plan_cache = _generic(
        q, k, v, selected, weights, route_starts
    )
    specialized, cache = block_retrieval_attention_forward(
        q,
        k,
        v,
        selected,
        weights,
        route_starts,
        block_size=2,
        routing_stride=2,
        kv_head_indices=(0, 0),
        weight_mode="logit_bias",
        weight_scale=0.7,
        weight_eps=1e-8,
        return_cache=True,
    )
    np.testing.assert_allclose(np.asarray(specialized), np.asarray(generic), rtol=1e-12, atol=1e-12)

    dout = xp.asarray(np.random.default_rng(901).normal(size=q.shape), dtype="float64")
    gdq, gdk, gdv, gdbias = indexed_attention_backward(dout, generic_cache)
    gdw = weighted_block_retrieval_bias_backward(gdbias, plan_cache)
    sdq, sdk, sdv, sdw = block_retrieval_attention_backward(dout, cache)
    np.testing.assert_allclose(np.asarray(sdq), np.asarray(gdq), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdk), np.asarray(gdk), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdv), np.asarray(gdv), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdw), np.asarray(gdw), rtol=1e-11, atol=1e-11)


def test_block_retrieval_shared_router_head_matches_generic():
    q, k, v, selected, weights, route_starts = _fixture(True)
    generic, generic_cache, plan_cache = _generic(
        q, k, v, selected, weights, route_starts
    )
    specialized, cache = block_retrieval_attention_forward(
        q,
        k,
        v,
        selected,
        weights,
        route_starts,
        block_size=2,
        routing_stride=2,
        kv_head_indices=(0, 0),
        weight_mode="logit_bias",
        weight_scale=0.7,
        weight_eps=1e-8,
        return_cache=True,
    )
    np.testing.assert_allclose(np.asarray(specialized), np.asarray(generic), rtol=1e-12, atol=1e-12)

    dout = xp.asarray(np.random.default_rng(902).normal(size=q.shape), dtype="float64")
    gdq, gdk, gdv, gdbias = indexed_attention_backward(dout, generic_cache)
    gdw = weighted_block_retrieval_bias_backward(gdbias, plan_cache)
    sdq, sdk, sdv, sdw = block_retrieval_attention_backward(dout, cache)
    np.testing.assert_allclose(np.asarray(sdq), np.asarray(gdq), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdk), np.asarray(gdk), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdv), np.asarray(gdv), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(sdw), np.asarray(gdw), rtol=1e-11, atol=1e-11)


def test_block_retrieval_none_mode_has_zero_router_weight_gradient():
    q, k, v, selected, weights, route_starts = _fixture(False)
    out, cache = block_retrieval_attention_forward(
        q,
        k,
        v,
        selected,
        weights,
        route_starts,
        block_size=2,
        routing_stride=2,
        kv_head_indices=(0, 0),
        weight_mode="none",
        return_cache=True,
    )
    _, _, _, dweights = block_retrieval_attention_backward(xp.ones_like(out), cache)
    np.testing.assert_array_equal(np.asarray(dweights), np.zeros_like(np.asarray(dweights)))
