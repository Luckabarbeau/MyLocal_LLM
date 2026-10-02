import dataclasses

import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.config import (
    AttentionLayerConfig,
    ContextRouterConfig,
    LocalAttentionConfig,
    ModelConfig,
    RetrievalAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.attention import GQAAttention


def _router_config(num_queries=1):
    return ContextRouterConfig(
        history_block_size=2,
        routing_stride=2,
        query_window=2,
        router_dim=3,
        top_k_blocks=2,
        exclude_recent_tokens=2,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=num_queries,
        router_weight_mode="logit_bias",
    )


def _copy_base_parameters(source, target):
    for source_param, target_param in zip(source.parameters()[:4], target.parameters()[:4]):
        target_param.data[...] = source_param.data


def test_all_local_full_window_matches_legacy_dense_gqa():
    legacy = GQAAttention(
        d_model=4,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=2,
        input_std=0.15,
        output_std=0.12,
        rng=RandomStream(300),
        dtype="float64",
    )
    mixed = GQAAttention(
        d_model=4,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=2,
        input_std=0.15,
        output_std=0.12,
        rng=RandomStream(301),
        dtype="float64",
        attention_config=AttentionLayerConfig(
            heads=(LocalAttentionConfig(window=64), LocalAttentionConfig(window=64))
        ),
    )
    _copy_base_parameters(legacy, mixed)

    x = xp.asarray(np.random.default_rng(302).normal(size=(2, 7, 4)), dtype="float64")
    coeff = xp.asarray(np.random.default_rng(303).normal(size=x.shape), dtype="float64")

    y_legacy, cache_legacy = legacy.forward(x)
    y_mixed, cache_mixed = mixed.forward(x)
    np.testing.assert_allclose(np.asarray(y_mixed), np.asarray(y_legacy), rtol=2e-12, atol=2e-12)

    legacy.zero_grad()
    mixed.zero_grad()
    dx_legacy = legacy.backward(coeff, cache_legacy)
    dx_mixed = mixed.backward(coeff, cache_mixed)
    np.testing.assert_allclose(np.asarray(dx_mixed), np.asarray(dx_legacy), rtol=3e-12, atol=3e-12)
    for p_legacy, p_mixed in zip(legacy.parameters(), mixed.parameters()):
        np.testing.assert_allclose(
            np.asarray(p_mixed.grad), np.asarray(p_legacy.grad), rtol=3e-12, atol=3e-12
        )


def test_mixed_local_retrieval_attention_input_directional_derivative():
    layer_config = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=3),
            RetrievalAttentionConfig(_router_config(), group="far"),
        )
    )
    attention = GQAAttention(
        d_model=4,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=2,
        input_std=0.18,
        output_std=0.13,
        rng=RandomStream(304),
        dtype="float64",
        attention_config=layer_config,
    )
    x = xp.asarray(np.random.default_rng(305).normal(size=(1, 12, 4)), dtype="float64")
    coeff = xp.asarray(np.random.default_rng(306).normal(size=x.shape), dtype="float64")
    direction = xp.asarray(np.random.default_rng(307).normal(size=x.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))

    y, cache = attention.forward(x)
    selected_ref = np.asarray(cache["routing"]["far"]["selected_blocks"]).copy()
    attention.zero_grad()
    dx = attention.backward(coeff, cache)
    analytical = float(xp.sum(dx * direction))

    def objective(x_value):
        y_value, value_cache = attention.forward(x_value)
        np.testing.assert_array_equal(
            np.asarray(value_cache["routing"]["far"]["selected_blocks"]),
            selected_ref,
        )
        return float(xp.sum(y_value * coeff))

    eps = 1e-6
    finite_difference = (
        objective(x + eps * direction) - objective(x - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=5e-5, atol=2e-8)

    router = attention._retrieval_groups[0]["module"].router
    assert bool(xp.any(router.W_query.grad != 0))
    assert bool(xp.any(attention.Wq.grad != 0))
    assert bool(xp.any(attention.Wk.grad != 0))
    assert bool(xp.any(attention.Wv.grad != 0))


def test_noncontiguous_retrieval_heads_share_one_router_group():
    router_config = _router_config(num_queries=2)
    layer_config = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=4),
            RetrievalAttentionConfig(router_config, group="far"),
            LocalAttentionConfig(window=6),
            RetrievalAttentionConfig(router_config, group="far"),
        )
    )
    attention = GQAAttention(
        d_model=8,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=2,
        input_std=0.12,
        output_std=0.1,
        rng=RandomStream(308),
        dtype="float64",
        attention_config=layer_config,
    )
    assert len(attention._retrieval_groups) == 1
    assert attention._retrieval_groups[0]["head_indices"] == (1, 3)

    x = xp.asarray(np.random.default_rng(309).normal(size=(1, 12, 8)), dtype="float64")
    y, cache = attention.forward(x)
    assert y.shape == x.shape
    assert cache["routing"]["far"]["weights"].shape[2] == 2

    attention.zero_grad()
    dx = attention.backward(xp.ones_like(y), cache)
    assert dx.shape == x.shape
    assert bool(xp.all(xp.isfinite(dx)))


def test_attention_model_config_round_trips_through_dataclasses_dict():
    router_config = _router_config()
    layer = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=4),
            RetrievalAttentionConfig(router_config, group="far"),
        )
    )
    original = ModelConfig(
        tokenizer_vocab_size=32,
        context_length=16,
        n_layers=1,
        d_model=4,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=2,
        n_experts=2,
        top_k=1,
        d_ff=8,
        attention_layers=(layer,),
    )
    restored = ModelConfig(**dataclasses.asdict(original))
    assert restored == original


def test_decoder_model_runs_mixed_attention_forward_backward():
    router_config = _router_config()
    layer = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=4),
            RetrievalAttentionConfig(router_config, group="far"),
        )
    )
    config = ModelConfig(
        tokenizer_vocab_size=32,
        context_length=16,
        n_layers=1,
        d_model=4,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=2,
        n_experts=2,
        top_k=1,
        d_ff=8,
        dtype="float64",
        attention_layers=(layer,),
    )
    model = DecoderLanguageModel(config, rng_seed=310)
    token_ids = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=xp.int64)
    targets = xp.asarray([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=xp.int64)

    logits, cache = model.forward(token_ids)
    loss, loss_cache = model.compute_loss(logits, targets)
    assert bool(xp.isfinite(loss))

    model.zero_grad()
    dlogits = model.backward_loss(loss_cache)
    model.backward(dlogits, cache)

    router = model.blocks[0].attention._retrieval_groups[0]["module"].router
    assert bool(xp.any(router.W_query.grad != 0))
    assert bool(xp.all(xp.isfinite(router.W_query.grad)))
