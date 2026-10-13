"""0080O3 audit: diagnostics, checkpoint options and CSV log parity."""
import csv
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.decoder import _cached_per_token_nll
from mini_llm.block_proposal.objective import BlockValidationAccumulator
from mini_llm.ops import loss as loss_ops
from test_block_proposal_0080o2_hybrid_muon import _hybrid
from test_block_proposal_0080n_production import _docs, _close

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='audit CPU reference')


def test_stable_bf16_ce_diagnostics_use_actual_row_losses(monkeypatch):
    # Make the CPU float32 path traverse the BF16 CE implementation to test
    # its bookkeeping without claiming GPU BF16 numerical verification.
    monkeypatch.setattr(loss_ops, 'is_bfloat16_dtype', lambda dtype: True)
    monkeypatch.delenv('MINI_LLM_INPLACE_BF16_CE', raising=False)
    logits = np.array([[[0., -120.], [0., -6.]]], dtype=np.float32)
    labels = np.array([[1, 1]], dtype=np.int64)
    valid = np.array([[True, True]])
    ce, cache = loss_ops.cross_entropy_forward(logits, labels, loss_mask=valid,
                  return_device_loss=True, normalizer_count=2,
                  return_row_losses=True)
    by_position = _cached_per_token_nll(cache, labels, valid)
    np.testing.assert_allclose(by_position, [[120., np.log1p(np.exp(6.))]],
                               atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(by_position.sum()/2, ce, atol=1e-6, rtol=1e-6)
    # AR callers do not request row losses and retain their original cache
    # shape and allocation behavior.
    ordinary_loss, ordinary_cache = loss_ops.cross_entropy_forward(
        logits, labels, loss_mask=valid, return_device_loss=True,
        normalizer_count=2)
    assert 'row_losses' not in ordinary_cache
    np.testing.assert_allclose(ordinary_loss, ce, atol=0, rtol=0)
    # The ordinary reconstruction from a clipped probability incorrectly
    # reports ~69 rather than 120 for the first target.
    assert by_position[0, 0] > 100
    assert loss_ops.cross_entropy_backward(cache).shape == logits.shape


def test_group_size_is_an_exact_resume_contract_field(tmp_path, capsys):
    shard = _docs(tmp_path)
    left = _hybrid(shard)
    right = _hybrid(shard)
    try:
        saved = left.block_proposal_resume_contract()
        assert saved['block_group_size'] == 2
        right.training_objective.block_group_size = 1
        with pytest.raises(ValueError, match='block_group_size'):
            right.validate_block_proposal_resume_contract(
                {'block_proposal_resume_contract': saved})
        # Existing 0080N/O1/O2 checkpoints did not record this dimension.
        legacy = dict(saved)
        legacy.pop('block_group_size')
        right.validate_block_proposal_resume_contract(
            {'block_proposal_resume_contract': legacy})
        assert 'unverified' in capsys.readouterr().out
    finally:
        _close(left)
        _close(right)


def _validation_example(k=8):
    acc = BlockValidationAccumulator(k)
    counts = np.ones(k, dtype=np.float64)
    acc.add(dict(unique_supervised_tokens=k,
                 final_ce=2., proposal_ce=3.,
                 position_counts=counts,
                 final_position_ce_sums=2*counts,
                 proposal_position_ce_sums=3*counts))
    return acc.summary()


def test_two_stage_validation_csv_keeps_horizons_and_does_not_change_ar_schema(tmp_path):
    shard = _docs(tmp_path)
    trainer = _hybrid(shard)
    try:
        trainer.log_file = tmp_path / 'train.csv'
        base = dict(step=1,source='source',schema_version=1,mixture_manifest_sha256='',
                    tokenizer_sha256='',memory_mode='joint',ce_nats_per_target=2.,
                    perplexity=7.4,loss_nll_sum=16.,supervised_token_weight=8,
                    supervised_tokens=8,target_capacity_tokens=8,target_fill=1.,
                    validation_batches=1,fixed_seed=42,sample_sha256='123')
        result = dict(base, **_validation_example())
        trainer._log_val_source_losses({'source': result})
        path = tmp_path / 'train.val_sources.block_proposal.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and 'h8_final_ce' in rows[0]
        assert rows[0]['h8_final_ce'] == '2.0'
        representative = dict(result, sampling_mode='representative_token_sampled',
                    sampled_document_count=1,unique_sampled_documents=1,
                    sampled_document_length_mean=32., sampled_document_length_median=32.,
                    sampled_document_length_p10=32.,sampled_document_length_p90=32.,
                    short_document_fraction=0., valid_history_mean=24.,
                    document_selection_fallback_count=0)
        trainer._log_representative_source_losses({'source': representative})
        path = tmp_path / 'train.val_sources_representative.block_proposal.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and 'h1_proposal_ce' in rows[0]
        # AR retains the original schemas and file names.
        trainer.training_objective = None
        trainer._log_val_source_losses({'source': base})
        with (tmp_path / 'train.val_sources.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and 'h1_proposal_ce' not in rows[0]
    finally:
        _close(trainer)


def test_0080_rejects_silently_ignored_activation_checkpoint(tmp_path, monkeypatch):
    shard = _docs(tmp_path)
    monkeypatch.setenv('MINI_LLM_ACTIVATION_CHECKPOINT', 'block')
    with pytest.raises(ValueError, match='does not currently support'):
        _hybrid(shard)


def test_end_to_end_gradient_route_includes_both_moe_and_memory_routers():
    from dataclasses import replace
    from test_block_proposal_0080l_baselines import _configuration, _window
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.block_proposal.training import SinglePassBlockTrainer
    model = DecoderLanguageModel(replace(_configuration(), top_k=2),
                                 rng_seed=8080, dtype='float64')
    net = TwoStageBlockModel(
        HorizonProposalHead(model, 8, hidden_width=12, rng_seed=8081),
        BlockPrefixDecoder(model, 8, width=12, n_layers=1, n_heads=3,
                           rng_seed=8082))
    trainer = SinglePassBlockTrainer(net)
    src, targets, valid, meta = _window()
    trainer.zero_grad()
    result = trainer.train_window(src, targets, valid_target_mask=valid,
                                  memory_metadata=meta, block_group_size=2)
    assert (result['main_forward_passes'], result['main_backward_passes']) == (1, 1)
    grads = {p.name: np.asarray(p.grad) for p in trainer.parameters()}
    expected = ('embedding.W', 'output_proj.W',
                'memory_router.W_query', 'memory_router.W_history',
                'blocks.0.attention.retrieval.retrieval.router.W_query',
                'blocks.1.attention.retrieval.retrieval.router.W_history',
                'blocks.0.moe.router.W_router', 'blocks.1.moe.router.W_router',
                'block_proposal.horizon_context', 'block_proposal.horizon_back',
                'block_decoder.plan.W', 'block_decoder.tokens.W',
                'block_decoder.layer0.attn.Wq', 'block_decoder.layer0.attn.Wk',
                'block_decoder.layer0.ffn.W_gate',
                'block_decoder.correction.W')
    for name in expected:
        assert np.any(grads[name] != 0.), name
        assert np.all(np.isfinite(grads[name])), name
    # Proposal position one never uses the learned horizon offset.
    assert np.all(grads['block_proposal.horizon_embedding'][0] == 0)


def test_decoder_ff_multiplier_cli_and_checkpoint_are_consistent():
    from test_block_proposal_0080l_baselines import _make
    from test_block_proposal_0080n_production import _args
    from train_model import build_training_objective
    model = _make(8).model
    kwargs = dict(objective='block-proposal', block_proposal_k=8,
                  block_proposal_head_width=12,block_proposal_decoder_width=12,
                  block_proposal_decoder_layers=1,block_proposal_decoder_heads=3,
                  block_proposal_decoder_ff_multiplier=3)
    first = build_training_objective(model, _args(**kwargs))
    assert first.checkpoint_metadata()['decoder_ff_width'] == 36
    state = dict(prediction_objective='block-proposal',
                 architecture=first.checkpoint_metadata())
    restored = build_training_objective(model, _args(), checkpoint_state=state)
    assert restored.checkpoint_metadata() == state['architecture']
    with pytest.raises(ValueError, match='decoder_ff_multiplier'):
        build_training_objective(model, _args(block_proposal_decoder_ff_multiplier=4),
                                 checkpoint_state=state)
    with pytest.raises(ValueError, match='multiplier must be positive'):
        build_training_objective(model, _args(**dict(kwargs,
                  block_proposal_decoder_ff_multiplier=0)))
    with pytest.raises(ValueError, match='positive lambda'):
        build_training_objective(model, _args(**dict(kwargs,block_proposal_lambda=0)))
