"""0080N production CLI, fixed validation, routing provenance and resume."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.block_proposal.objective import BlockProposalObjective, BlockValidationAccumulator
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.train_extended import ExtendedTrainer
from test_block_proposal_0080l_baselines import _make, _window
from train_model import build_training_objective

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='0080N CPU reference only')


def _args(**options):
    keys = ('objective', 'block_proposal_k', 'block_proposal_head_width',
            'block_proposal_residual_scale', 'block_proposal_decoder_width',
            'block_proposal_decoder_layers', 'block_proposal_decoder_heads',
            'block_proposal_lambda', 'block_proposal_group_size')
    result = {key: None for key in keys}
    result['block_proposal_group_size'] = 2
    result.update(options)
    return SimpleNamespace(**result)


def _close(trainer):
    for cache in (trainer.shards, trainer.val_shards):
        for value in cache.values():
            trainer._close_mapped_shard(value)
        cache.clear()


def _docs(tmp_path):
    docs = [list(range(1, 45)) + [100], [60,61,62,100],
            list(range(10, 65)) + [100], list(range(21,52)) + [100]]
    path = tmp_path / 'packed.bin'
    np.asarray([v for doc in docs for v in doc], dtype='<u2').tofile(path)
    return path


def _trainer(path, *, checkpoint_dir=None, mixed=False):
    single = _make(8)
    kw = dict(model=single.model, training_objective=BlockProposalObjective(single.net, block_group_size=2),
              train_shard_paths=[str(path)], val_shard_paths=[str(path)],
              batch_size=2, seq_length=24, grad_accum_steps=2,
              gradient_accumulation_weighting='supervised_tokens',
              total_steps=10, warmup_steps=0, peak_lr=1e-4, grad_clip=10.,
              eos_token_id=100, rng_seed=101, checkpoint_dir=checkpoint_dir)
    if mixed:
        kw.update(train_source_shards={'a':[str(path)], 'b':[str(path)]},
                  val_source_shards={'a':[str(path)], 'b':[str(path)]},
                  source_weights={'a':.3, 'b':.7}, val_fixed_mixed_steps=3,
                  val_per_source_steps=2)
    return ExtendedTrainer(**kw)


def test_cli_defaults_and_checkpoint_objective_contract():
    original = _make(8)
    assert build_training_objective(original.model, _args()) is None
    candidate = build_training_objective(original.model, _args(
        objective='block-proposal', block_proposal_k=8,
        block_proposal_head_width=12, block_proposal_decoder_width=12,
        block_proposal_decoder_layers=1, block_proposal_decoder_heads=3))
    saved = {'prediction_objective':'block-proposal', 'architecture': candidate.checkpoint_metadata()}
    restored = build_training_objective(original.model, _args(), checkpoint_state=saved)
    assert restored.checkpoint_metadata() == candidate.checkpoint_metadata()
    with pytest.raises(ValueError, match='checkpoint option mismatch'):
        build_training_objective(original.model, _args(block_proposal_k=4), checkpoint_state=saved)
    with pytest.raises(ValueError, match='prediction objective'):
        build_training_objective(original.model, _args(objective='ar'), checkpoint_state=saved)
    with pytest.raises(ValueError, match='prediction objective'):
        build_training_objective(original.model, _args(objective='block-proposal'),
                                 checkpoint_state={'prediction_objective':'ar'})


def test_cpu_horizon_validation_uses_unique_token_denominator():
    original = _make(8)
    src, y, mask, metadata = _window()
    result = original.train_window(src,y,valid_target_mask=mask,
                                   memory_metadata=metadata,compute_gradients=False,
                                   block_group_size=2)
    acc = BlockValidationAccumulator(8)
    acc.add(result); acc.add(result)
    summary = acc.summary()
    assert summary['block_unique_supervised_tokens'] == int(mask.sum()) * 2
    assert sum(summary[f'h{i}_count'] for i in range(1,9)) == mask.sum()*2
    np.testing.assert_allclose(summary['proposal_ce_nats_per_target'], result['proposal_ce'])
    np.testing.assert_allclose(summary['final_ce_nats_per_target'], result['final_ce'])
    for i in range(1,9):
        assert summary[f'h{i}_count'] >= 0
        if summary[f'h{i}_count']:
            np.testing.assert_allclose(summary[f'h{i}_decoder_gain'],
                summary[f'h{i}_proposal_ce'] - summary[f'h{i}_final_ce'])


def test_fixed_production_validation_and_routing_accounting(tmp_path):
    shard = _docs(tmp_path)
    trainer = _trainer(shard, mixed=True)
    try:
        loss, norm = trainer.train_step()
        assert np.isfinite(loss) and norm > 0
        assert trainer._last_step_routing_modes['microbatches'] == 2
        assert trainer._last_step_routing_modes['samples'] == 4
        assert trainer._last_step_routing_modes['supervised_token_weight'] == trainer._last_step_supervised_tokens
        assert trainer.sample_trace_microbatches == 2
        assert trainer.sample_trace_sha256 != '00'*32
        state = trainer.train_rng.bit_generator.state
        before = json.dumps(state, sort_keys=True)
        fixed = trainer.compute_val_loss_fixed_mixed()
        assert fixed == trainer.compute_val_loss_fixed_mixed()
        assert json.dumps(trainer.train_rng.bit_generator.state, sort_keys=True) == before
        assert fixed['block_unique_supervised_tokens'] == fixed['supervised_token_weight']
        np.testing.assert_allclose(fixed['ce_nats_per_target'], fixed['final_ce_nats_per_target'])
        assert sum(fixed[f'h{i}_count'] for i in range(1,9)) == fixed['block_unique_supervised_tokens']
        by_source = trainer.compute_val_loss_by_source()
        assert set(by_source) == {'a','b'}
        assert all('h8_decoder_gain' in metric for metric in by_source.values())
    finally:
        _close(trainer)


def test_exact_cpu_joint_model_optimizer_and_sample_trace_resume(tmp_path):
    shard = _docs(tmp_path)
    directory = str(tmp_path / 'checkpoint')
    live = _trainer(shard, checkpoint_dir=directory)
    recovered = _trainer(shard)
    try:
        live.train_step()
        live.save()
        state = load_two_stage_checkpoint(directory, recovered.training_objective.net,
                                          optimizer=recovered.optimizer)
        assert state['supervised_tokens'] == int(live._last_step_supervised_tokens)
        assert state['block_proposal_resume_contract'] == live.block_proposal_resume_contract()
        recovered.validate_block_proposal_resume_contract(state)
        recovered.scheduler.total_steps += 1
        with pytest.raises(ValueError, match='training contract mismatch'):
            recovered.validate_block_proposal_resume_contract(state)
        recovered.scheduler.total_steps -= 1
        assert state['data_provenance']['sample_trace_sha256'] == live.sample_trace_sha256
        recovered.step = state['step']
        recovered.tokens_processed = state['tokens_processed']
        recovered.current_train_shard_idx = state['current_train_shard_idx']
        recovered.train_rng.bit_generator.state = state['train_rng_state']
        recovered.val_rng.bit_generator.state = state['val_rng_state']
        recovered.sample_trace_sha256 = state['data_provenance']['sample_trace_sha256']
        recovered.sample_trace_microbatches = state['data_provenance']['sample_trace_microbatches']
        recovered._block_proposal_supervised_tokens = state['supervised_tokens']
        loss_a, norm_a = live.train_step()
        loss_b, norm_b = recovered.train_step()
        np.testing.assert_allclose([loss_a, norm_a],[loss_b,norm_b],rtol=0,atol=0)
        assert live.sample_trace_sha256 == recovered.sample_trace_sha256
        assert live._block_proposal_supervised_tokens == recovered._block_proposal_supervised_tokens
        assert live.optimizer.step_index == recovered.optimizer.step_index == 2
        for a,b in zip(live.optimizer.parameters, recovered.optimizer.parameters):
            np.testing.assert_array_equal(asnumpy(a.data), asnumpy(b.data),err_msg=a.name)
    finally:
        _close(live);_close(recovered)
