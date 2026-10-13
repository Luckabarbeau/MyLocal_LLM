"""0079P1 recurrent vocabulary tape lifetime; exact reference comparisons."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.config import AttentionLayerConfig, LocalAttentionConfig, DilatedAttentionConfig


def make_reference(*, compact=False, sparse=False, shared=True, k=4,
                   dtype='float64', feedback='full'):
    cfg = ModelConfig(tokenizer_vocab_size=48, context_length=64,
                      n_layers=2, d_model=16, n_q_heads=2,
                      n_kv_heads=1, d_head=8, d_ff=32,
                      n_experts=2, top_k=1, dtype=dtype)
    if sparse:
        heads = (LocalAttentionConfig(window=9),
                 DilatedAttentionConfig(window=32, dilation=4))
        ac = AttentionLayerConfig(heads=heads)
        cfg = replace(cfg, attention_layers=(ac,) * cfg.n_layers)
    reference_cfg = RefinementReferenceConfig(
        block_size=k, enable_shared_context_training=shared,
        enable_compact_static_attention=sparse,
        feedback_mode=feedback, compact_recurrent_tape=compact,
    )
    return DenseRefinementTrainingReference(
        DecoderLanguageModel(cfg, rng_seed=237, dtype=dtype),
        reference_cfg, rng_seed=238,
    )


def gradients(ref, tape):
    ref.zero_grad()
    ref.backward(tape)
    return {p.name: asnumpy(p.grad).copy() for p in ref.parameters()}


@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('feedback', ['full', 'unknown'])
def test_tape_parity_loss_and_all_gradients(shared, sparse, k, feedback):
    # Sparse shared/ordinary and the K=1 special case must all work.
    ref = make_reference(sparse=sparse, shared=shared, k=k, feedback=feedback)
    ctx = xp.asarray([[3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])
    tgt = xp.asarray(np.arange(2*k).reshape(2,k) % 20 + 14)
    valid = xp.ones((2,k), dtype=xp.bool_)
    if k > 1:
        valid[1, -1] = False
        tgt[1, -1] = -100
    ref_metrics, tape_ref = ref.forward(ctx, tgt, valid_target_mask=valid)
    ref_grads = gradients(ref, tape_ref)
    ref.config = replace(ref.config, compact_recurrent_tape=True)
    compact_metrics, tape_compact = ref.forward(ctx, tgt, valid_target_mask=valid)
    for metric in ('initial_ce', 'commitment_ce', 'total_loss'):
        np.testing.assert_allclose(float(ref_metrics[metric]), float(compact_metrics[metric]),
                                   atol=1e-11, rtol=1e-11)
    assert len(tape_compact['passes']) == k + 1
    assert all(record.logits is None for record in tape_compact['passes'])
    assert tape_compact['passes'][0].loss_gradient.shape == (2,k,48)
    assert all(record.loss_gradient.shape == (2,48) for record in tape_compact['passes'][1:])
    if k > 1:
        assert compact_metrics['recurrent_vocab_tape_bytes'] < ref_metrics['recurrent_vocab_tape_bytes']
    compact_grads = gradients(ref, tape_compact)
    for p in ref.parameters():
        np.testing.assert_allclose(ref_grads[p.name], compact_grads[p.name],
                                   atol=1e-10, rtol=1e-10, err_msg=p.name)
    assert all(record.model_cache is None and record.loss_gradient is None
               for record in tape_compact['passes'])


@pytest.mark.parametrize('sparse', [False, True])
def test_window_equivalence_and_source_accounting(sparse):
    ref = make_reference(sparse=sparse, shared=True, k=4)
    trainer = WindowRefinementTrainer(ref.model, ref.config, rng_seed=238)
    # Share the exact state-embedding initialization from reference.
    ids = xp.asarray([[3,4,5,6,7,8,9,10,11,12,13,14,15],
                      [14,13,12,11,10,9,8,7,6,5,-100,-100,-100]])
    valid = xp.asarray([[True]*13, [True]*10 + [False]*3])
    def run(compact):
        trainer.config = replace(trainer.config, compact_recurrent_tape=compact)
        trainer.zero_grad()
        metrics = trainer.train_window(ids, prefix_length=5, group_size=2,
                                       valid_token_mask=valid, record_row_ce=True,
                                       compact_active_blocks=True, compute_gradients=True)
        return metrics, {p.name: asnumpy(p.grad).copy() for p in trainer.parameters()}
    a, ga = run(False)
    b, gb = run(True)
    for key in ('total_loss', 'commitment_ce', 'initial_ce'):
        np.testing.assert_allclose(float(a[key]), float(b[key]), atol=1e-11)
    for key in ('row_commitment_ce_sums', 'position_counts',
                'commitment_position_ce_sums', 'initial_position_ce_sums'):
        np.testing.assert_allclose(asnumpy(a[key]), asnumpy(b[key]), atol=1e-11)
    assert b['recurrent_vocab_tape_peak_bytes'] < a['recurrent_vocab_tape_peak_bytes']
    for name in ga:
        np.testing.assert_allclose(ga[name], gb[name], atol=1e-10, rtol=1e-10, err_msg=name)


def test_forward_only_is_unchanged_and_does_not_allocate_tape():
    ref = make_reference(k=4)
    ctx = xp.asarray([[1,2,3]])
    tgt = xp.asarray([[4,5,6,7]])
    base, tape = ref.forward(ctx,tgt,record_tape=False)
    ref.config = replace(ref.config, compact_recurrent_tape=True)
    new, new_tape = ref.forward(ctx,tgt,record_tape=False)
    assert tape is new_tape is None
    assert base['recurrent_vocab_tape_bytes'] == new['recurrent_vocab_tape_bytes'] == 0
    np.testing.assert_allclose(float(base['total_loss']),float(new['total_loss']))


def test_reference_tape_is_still_fully_inspectable():
    ref=make_reference(k=3)
    metrics,tape=ref.forward(xp.asarray([[1,2,3]]),xp.asarray([[4,5,6]]))
    assert all(p.logits is not None and p.loss_gradient.shape == p.logits.shape
               for p in tape['passes'])
    assert all(p.model_cache is not None for p in tape['passes'])
    assert metrics['compact_recurrent_tape'] is False


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA/CuPy runtime required')
def test_gpu_bfloat16_compact_tape_gradient_parity():
    # BF16 compares within kernel and FP32 scatter rounding tolerances.
    from mini_llm.backend import validate_bfloat16_backend
    validate_bfloat16_backend()
    ref=make_reference(shared=True,k=4,dtype='bfloat16')
    ctx=xp.asarray([[1,2,3],[3,4,5]])
    tgt=xp.asarray([[4,5,6,7],[6,7,8,9]])
    a,tape=ref.forward(ctx,tgt)
    ga=gradients(ref,tape)
    ref.config=replace(ref.config, compact_recurrent_tape=True)
    b,tape=ref.forward(ctx,tgt)
    gb=gradients(ref,tape)
    np.testing.assert_allclose(float(a['commitment_ce']),float(b['commitment_ce']),rtol=2e-4,atol=2e-4)
    for name in ga:
        np.testing.assert_allclose(ga[name],gb[name],rtol=3e-3,atol=3e-3,err_msg=name)
