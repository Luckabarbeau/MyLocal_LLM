"""0079Q2: production routed-prefix -> one P2 shared-resident causal cache.

The production TOP-LEVEL learned router is active. Static-head provisional
attention is a temporary test configuration; Q3 retains a hard guard on the
production model's per-layer learned retrieval heads.
"""
from dataclasses import replace
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import (
    ModelConfig, MemoryContextConfig, AttentionLayerConfig, LocalAttentionConfig,
    GlobalSparseAttentionConfig, ContextRouterConfig, RetrievalAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer


def make_trainer(*, k=4, sparse=False, learned=False):
    memory = MemoryContextConfig(
        enabled=True, memory_length=48, recent_length=16, target_length=16,
        block_size=4, top_k_blocks=2, router_query_length=8,
        router_dim=4, query_pooling='mean', history_pooling='mean',
        integration_mode='routed_prefix', memory_training='joint',
        router_surrogate_scale=1., router_gumbel_noise=False,
        retrieval_batch_probability=1.0, read_heads=1, read_kv_heads=1,
    )
    layer = None
    if sparse:
        layer = AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=8),
            GlobalSparseAttentionConfig(stride=3, offset=0),
        ))
    if learned:
        router = ContextRouterConfig(history_block_size=2, routing_stride=8,
                                      query_window=4, router_dim=2, top_k_blocks=2,
                                      exclude_recent_tokens=2, query_pooling='mean',
                                      history_pooling='mean', num_queries=1)
        layer = AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=8), RetrievalAttentionConfig(router),
        ))
    cfg = ModelConfig(tokenizer_vocab_size=100, context_length=memory.active_length,
                      n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1,
                      d_head=8, n_experts=2, top_k=1, d_ff=32,
                      memory_context=memory, dtype='float64',
                      attention_layers=None if layer is None else (layer,layer))
    model = DecoderLanguageModel(cfg, rng_seed=98, dtype='float64')
    return WindowRefinementTrainer(model, RefinementReferenceConfig(
        block_size=k, enable_shared_context_training=True,
        compact_recurrent_tape=True,
    ), rng_seed=95)


def _source(batch=1):
    return xp.asarray(np.mod(np.arange(batch * 48).reshape(batch, 48) * 7 + 3, 97),
                      dtype=xp.int64)


def _evaluate(trainer, source, resident, *, k=4):
    trainer.zero_grad()
    metrics = trainer.train_routed_window(
        source, group_size=2, compact_history=True, compact_active_blocks=True,
        resident_history=resident, max_history_keys=32,
        max_window_tokens=16, max_active_queries=32,
        memory_metadata={'use_retrieval': True, 'router_stochastic': False,
                         'history_valid_starts': xp.zeros((source.shape[0],), dtype=xp.int64)},
    )
    return metrics, [(p.name, asnumpy(p.grad).copy()) for p in trainer.parameters()]


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('k', [4, 8])
def test_real_routed_encoder_resident_matches_physical_kv_and_every_gradient(sparse, k):
    trainer = make_trainer(k=k, sparse=sparse)
    source = _source(2)
    ref, ref_grads = _evaluate(trainer, source, False, k=k)
    opt, opt_grads = _evaluate(trainer, source, True, k=k)
    for key in ('total_loss', 'initial_ce', 'commitment_ce'):
        np.testing.assert_allclose(float(ref[key]), float(opt[key]), rtol=1e-10, atol=1e-10)
    for key in ('position_counts', 'initial_position_ce_sums', 'commitment_position_ce_sums'):
        np.testing.assert_allclose(asnumpy(ref[key]), asnumpy(opt[key]), rtol=1e-10, atol=1e-10)
    assert len(ref_grads) == len(opt_grads)
    for (a, x), (b, y) in zip(ref_grads, opt_grads):
        assert a == b
        np.testing.assert_allclose(x, y, rtol=1e-7, atol=1e-7, err_msg=a)
    assert opt['routed_prefix_history']
    assert opt['routed_prefix_slots'] == 8
    assert opt['routed_prefix_mode'] == 'routed_blocks'
    assert opt['resident_history']
    assert opt['supervised_tokens'] == 32
    assert opt['refinement_passes'] == (k+1) * ((16//k + 1)//2)  # sequential groups, not documents
    assert opt['avoided_group_history_copy_bytes'] > 0
    assert np.linalg.norm(asnumpy(trainer.model.memory_router.W_query.grad)) > 0
    assert np.linalg.norm(asnumpy(trainer.model.memory_router.W_history.grad)) > 0


def test_causal_routed_block_does_not_read_later_real_targets():
    trainer = make_trainer(sparse=False)
    model = trainer.model
    source = _source()
    changed = source.copy()
    changed[:, -9:] = (changed[:, -9:] + 14) % 99
    assert np.all(asnumpy(source[:, -16:-12]) == asnumpy(changed[:, -16:-12]))
    meta = {'causal_prefix_query':True, 'use_retrieval':True, 'router_stochastic':False}
    _, a = model.forward_body(source, memory_metadata=meta, return_cache=True)
    _, b = model.forward_body(changed, memory_metadata=meta, return_cache=True)
    assert a['memory_context_cache']['prefix_length'] == 8
    for layer_a, layer_b in zip(a['block_caches'], b['block_caches']):
        for key in ('k', 'v'):
            np.testing.assert_allclose(
                asnumpy(layer_a['attn_cache'][key][:, :12]),
                asnumpy(layer_b['attn_cache'][key][:, :12]), rtol=0, atol=0)
    np.testing.assert_array_equal(asnumpy(a['memory_context_cache']['selected_blocks']),
                                  asnumpy(b['memory_context_cache']['selected_blocks']))


def test_rejects_noncausal_metadata_invalid_source_and_retrieval_heads():
    trainer = make_trainer()
    source = _source()
    opts = dict(group_size=1, compact_history=True, compact_active_blocks=True,
                resident_history=True, max_history_keys=32, max_window_tokens=16)
    with pytest.raises(ValueError, match='causal_prefix_query'):
        trainer.train_routed_window(source, memory_metadata={
            'causal_prefix_query':False, 'history_valid_starts': xp.zeros((1,), dtype=xp.int64)}, **opts)
    with pytest.raises(ValueError, match='source IDs'):
        trainer.train_routed_window(source[:, :-1], **opts)
    with pytest.raises(ValueError, match='fully valid working window'):
        trainer.train_routed_window(source, valid_token_mask=xp.asarray([[True]*15 + [False]]), **opts)
    with pytest.raises(ValueError, match='history_valid_starts'):
        trainer.train_routed_window(source, **opts)
    # Q3 enables learned per-layer heads, while retaining the resident-K/V
    # requirement tested in test_predictive_refinement_0079q3.py.
    assert make_trainer(learned=True).model.blocks[0].attention._retrieval_groups


def test_compressed_trunk_key_plans_are_strictly_block_causal():
    """2K routed slots are older than all K-token working-window targets."""
    from mini_llm.predictive_refinement.compact_history import pack_group_history
    from mini_llm.ops.attention_selection import KeySelectionPlan
    starts = (0, 8)             # positions relative to the 4K working window
    routed_prefix = 8           # tiny test analog of production 2048
    masked = xp.ones((2, 2, 4), dtype=xp.bool_)
    layouts = ((dict(kind='dense', window=None, dilation=None, offset=0, stride=None),),)
    history, valid, plans = pack_group_history(
        tuple(routed_prefix + start for start in starts), 2, 4, masked, layouts,
        max_history_keys=24)
    hist = asnumpy(history)
    selection = asnumpy(plans[0][0].key_indices)
    allowed = asnumpy(plans[0][0].valid_mask)
    for document in range(2):
        for j, start in enumerate(starts):
            row = document * len(starts) + j
            cutoff = routed_prefix + start
            real_slots = selection[row][allowed[row] & (selection[row] < hist.shape[1])]
            physical_history = hist[row, real_slots]
            assert np.all(physical_history < cutoff)
            # At the start of the window we can already read the full prefix.
            if start == 0:
                np.testing.assert_array_equal(np.unique(physical_history), np.arange(routed_prefix))
            else:
                np.testing.assert_array_equal(np.unique(physical_history), np.arange(cutoff))


@pytest.mark.skipif(__import__('mini_llm.backend', fromlist=['BACKEND_NAME']).BACKEND_NAME != 'cupy',
                    reason='requires CUDA/BF16 runtime')
def test_cuda_routed_window_resident_vs_gathered_gradient_parity():
    # Use the original small routed memory + learned router, but leave active
    # retrieval heads to Q3. This is the first RTX integration parity check.
    trainer = make_trainer(k=8, sparse=True)
    from mini_llm.backend import resolve_dtype
    dtype = resolve_dtype('bfloat16')
    # A separately seeded BF16 model is required; changing parameters'
    # storage dtype in-place would invalidate packed compute buffers.
    cfg = replace(trainer.model.config, dtype='bfloat16')
    model = DecoderLanguageModel(cfg, rng_seed=98, dtype='bfloat16')
    trainer = WindowRefinementTrainer(model, RefinementReferenceConfig(
        block_size=8, enable_shared_context_training=True,
        compact_recurrent_tape=True), rng_seed=95)
    source = _source(1)
    ref, ref_grads = _evaluate(trainer, source, False, k=8)
    got, got_grads = _evaluate(trainer, source, True, k=8)
    xp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(float(ref['commitment_ce']),float(got['commitment_ce']),
                               atol=1e-2,rtol=1e-2)
    for (_, expected), (name, actual) in zip(ref_grads, got_grads):
        np.testing.assert_allclose(expected, actual, atol=5e-2, rtol=5e-2,
                                   err_msg=name)


def test_provisional_rope_starts_at_absolute_64k_source_offsets(monkeypatch):
    trainer = make_trainer(k=8)
    model = trainer.model
    observed = []
    original = model.forward_shared_context_block
    def capture(embeddings, context_kv, **kwargs):
        observed.extend(asnumpy(kwargs['block_position_starts']).tolist())
        return original(embeddings, context_kv, **kwargs)
    monkeypatch.setattr(model, 'forward_shared_context_block', capture)
    trainer.train_routed_window(
        _source(1), memory_metadata={
            'history_valid_starts': xp.asarray([0], dtype=xp.int64),
            'router_stochastic': False,
        }, group_size=2, compact_history=True, compact_active_blocks=True,
        resident_history=True, max_history_keys=32, max_window_tokens=16,
        max_active_queries=16, compute_gradients=False,
    )
    assert len(observed) == 2 * 9
    assert set(observed) == {32, 40}  # real working-window source positions
