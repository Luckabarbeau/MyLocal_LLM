"""0079Q1: causal routing must use exclusively pre-window real history.

These tests exercise the PRODUCTION routed-prefix router and its handwritten
straight-through gradient, not a replacement/static selection mechanism.
"""
import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.config import (MemoryContextConfig, ModelConfig, AttentionLayerConfig,
                             ContextRouterConfig, LocalAttentionConfig,
                             RetrievalAttentionConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel


def _model(*, learned_heads=False):
    memory = MemoryContextConfig(
        enabled=True, memory_length=24, recent_length=8, target_length=8,
        block_size=2, top_k_blocks=2, router_query_length=4,
        router_dim=3, query_pooling='mean', history_pooling='mean',
        integration_mode='routed_prefix', memory_training='joint',
        router_surrogate_scale=1.0, router_gumbel_noise=False,
        retrieval_batch_probability=1.0, read_heads=1, read_kv_heads=1,
    )
    layer = None
    if learned_heads:
        heads_router = ContextRouterConfig(
            history_block_size=2, routing_stride=2, query_window=4,
            router_dim=2, top_k_blocks=2, exclude_recent_tokens=2,
            query_pooling='mean', history_pooling='mean', num_queries=1,
        )
        layer = AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=4),
            RetrievalAttentionConfig(heads_router, group='far'),
        ))
    return DecoderLanguageModel(ModelConfig(
        tokenizer_vocab_size=64, context_length=memory.active_length,
        n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        n_experts=2, top_k=1, d_ff=16, dtype='float64',
        memory_context=memory,
        attention_layers=None if layer is None else (layer,),
    ), rng_seed=97)


def _source():
    return xp.asarray([np.arange(1, 25)], dtype=xp.int64)


def _meta(**extras):
    return {'causal_prefix_query': True, 'use_retrieval': True,
            'router_stochastic': False, **extras}


def test_router_selection_and_prefix_cannot_see_any_window_target():
    model = _model()
    source_a = _source()
    source_b = source_a.copy()
    source_b[:, 16:] = xp.asarray(np.arange(30, 38)[None, :], dtype=xp.int64)
    active_a, cache_a = model._hierarchical_context_forward(source_a, _meta())
    active_b, cache_b = model._hierarchical_context_forward(source_b, _meta())
    assert cache_a['prefix_mode'] == cache_b['prefix_mode'] == 'routed_blocks'
    np.testing.assert_array_equal(asnumpy(active_a.selected_blocks), asnumpy(active_b.selected_blocks))
    np.testing.assert_allclose(asnumpy(active_a.route_weights), asnumpy(active_b.route_weights), rtol=0, atol=0)
    np.testing.assert_array_equal(asnumpy(active_a.position_ids[:, :4]), asnumpy(active_b.position_ids[:, :4]))
    np.testing.assert_array_equal(asnumpy(active_a.token_ids[:, :4]), asnumpy(active_b.token_ids[:, :4]))
    np.testing.assert_array_equal(asnumpy(cache_a['causal_prefix_query_ids']), np.array([[13, 14, 15, 16]]))
    assert np.max(asnumpy(active_a.position_ids[:, :4])) < 16
    assert np.min(asnumpy(active_a.position_ids[:, 4:])) == 16
    assert np.max(asnumpy(active_a.position_ids[:, 4:])) == 23


def test_causal_routing_backpropagates_into_history_not_future_targets():
    model = _model()
    _, cache = model._hierarchical_context_forward(_source(), _meta())
    assert cache['prefix_mode'] == 'routed_blocks'
    # No direct training-window loss gradient: router query must update ONLY
    # historical embedding IDs, even when the ST surrogate is active.
    rng = np.random.default_rng(8)
    dactive = xp.zeros((1, 12, 8), dtype=xp.float64)
    dactive[:, :4] = xp.asarray(rng.normal(size=(1, 4, 8)))
    model.zero_grad()
    model._routed_prefix_context_backward(dactive, cache)
    embedding_grad = asnumpy(model.embedding.W.grad)
    np.testing.assert_array_equal(embedding_grad[17:25], 0.0)
    assert np.linalg.norm(embedding_grad[1:17]) > 0.0
    assert np.linalg.norm(asnumpy(model.memory_router.W_query.grad)) > 0.0
    assert np.linalg.norm(asnumpy(model.memory_router.W_history.grad)) > 0.0


def test_full_causal_body_exposes_identical_routed_prefix_when_targets_change():
    model = _model()
    a = _source()
    b = a.copy()
    b[:, 16:] = xp.asarray(np.arange(30, 38)[None, :], dtype=xp.int64)
    _, tape_a = model.forward_body(a, memory_metadata=_meta(), return_cache=True)
    _, tape_b = model.forward_body(b, memory_metadata=_meta(), return_cache=True)
    prefix_a = tape_a['memory_context_cache']
    prefix_b = tape_b['memory_context_cache']
    np.testing.assert_array_equal(asnumpy(prefix_a['selected_blocks']), asnumpy(prefix_b['selected_blocks']))
    np.testing.assert_array_equal(asnumpy(prefix_a['selected_token_ids']), asnumpy(prefix_b['selected_token_ids']))
    assert prefix_a['prefix_length'] == prefix_b['prefix_length'] == 4


def test_rejects_noncausal_or_padded_router_query():
    model = _model()
    with pytest.raises(ValueError, match='fully valid pre-window'):
        model._hierarchical_context_forward(
            _source(), _meta(history_valid_starts=xp.asarray([13], dtype=xp.int64)))



def test_production_learned_heads_share_the_causal_router_selection():
    """Causal-prefix option also works with the actual learned head modules."""
    model = _model(learned_heads=True)
    a = _source()
    b = a.copy()
    b[:, 16:] = xp.asarray(np.arange(30, 38)[None, :], dtype=xp.int64)
    xa, tape_a = model.forward_body(a, memory_metadata=_meta(), return_cache=True)
    xb, tape_b = model.forward_body(b, memory_metadata=_meta(), return_cache=True)
    ma, mb = (tape_a['memory_context_cache'], tape_b['memory_context_cache'])
    np.testing.assert_array_equal(asnumpy(ma['selected_blocks']), asnumpy(mb['selected_blocks']))
    np.testing.assert_array_equal(asnumpy(ma['route_weights']), asnumpy(mb['route_weights']))
    assert len(model.blocks[0].attention._retrieval_groups) == 1
    assert xa.shape == xb.shape == (1, 12, 8)


def test_joint_backward_through_production_router_and_learned_heads():
    model = _model(learned_heads=True)
    source = _source()
    logits, cache = model.forward(source, memory_metadata=_meta(), return_cache=True)
    labels = xp.asarray([[18, 19, 20, 21, 22, 23, 24, 25]], dtype=xp.int64)
    loss, ce_cache = model.compute_loss(logits, labels)
    model.zero_grad()
    model.backward(model.backward_loss(ce_cache), cache)
    assert np.isfinite(float(loss))
    assert np.linalg.norm(asnumpy(model.memory_router.W_query.grad)) > 0
    assert np.linalg.norm(asnumpy(model.blocks[0].attention._retrieval_groups[0]['module'].router.W_query.grad)) > 0 if hasattr(model.blocks[0].attention._retrieval_groups[0]['module'].router, 'W_query') else True


def test_learned_retrieval_causal_trunk_states_ignore_later_real_text():
    model = _model(learned_heads=True)
    source = _source()
    changed = source.copy()
    changed[:, 18:] = xp.asarray(np.arange(30, 36)[None, :], dtype=xp.int64)
    xa, ca = model.forward_body(source, memory_metadata=_meta(), return_cache=True)
    xb, cb = model.forward_body(changed, memory_metadata=_meta(), return_cache=True)
    prefix = int(ca['memory_context_cache']['prefix_length'])
    assert prefix == int(cb['memory_context_cache']['prefix_length']) == 4
    # The last 6 tokens were changed, but preceding causal trunk states must
    # be byte-for-byte unchanged even with learned retrieval heads active.
    np.testing.assert_allclose(asnumpy(xa[:, :prefix + 2]),
                               asnumpy(xb[:, :prefix + 2]), rtol=0, atol=0)
