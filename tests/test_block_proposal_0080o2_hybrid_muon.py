"""0080O2: joint block proposal with the existing hybrid Muon/AdamW optimizer."""
import json
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.block_proposal.objective import BlockProposalObjective
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.train_extended import ExtendedTrainer
from test_block_proposal_0080l_baselines import _make
from test_block_proposal_0080n_production import _docs, _close

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='fixed CPU hybrid reference')


def _hybrid(shard, *, checkpoint_dir=None, momentum=0.95):
    setup = _make(8)
    return ExtendedTrainer(
        setup.model, [str(shard)], [str(shard)],
        training_objective=BlockProposalObjective(setup.net, block_group_size=2),
        batch_size=2, seq_length=24, grad_accum_steps=2,
        gradient_accumulation_weighting='supervised_tokens',
        total_steps=10, warmup_steps=0, peak_lr=1e-4, grad_clip=10.,
        eos_token_id=100, rng_seed=101, checkpoint_dir=checkpoint_dir,
        optimizer_kind='hybrid-muon', muon_momentum=momentum,
        muon_ns_steps=3, muon_matmul_precision='fp32')


def _state(opt):
    # Compare copies: checkpoint loading is allowed to consume dictionary entries.
    return {k: {n: np.asarray(asnumpy(v)).copy() for n, v in values.items()}
            if isinstance(values, dict) else values
            for k, values in opt.checkpoint_state().items()}


def test_joint_parameter_partition_and_training(tmp_path):
    shard = _docs(tmp_path)
    trainer = _hybrid(shard)
    try:
        opt = trainer.optimizer
        assert isinstance(opt, HybridMuonAdamW)
        assert opt.muon_names == {p.name for p in opt.muon.parameters}
        assert set(p.name for p in opt.parameters) == {
            p.name for p in trainer.training_objective.parameters()}
        assert len(opt.parameters) == len({p.name for p in opt.parameters})
        expected_muon = {'blocks.0.moe.experts.expert_1.W_gate',
                         'block_proposal.horizon_context',
                         'block_proposal.horizon_back',
                         'block_decoder.plan.W',
                         'block_decoder.layer0.attn.Wq',
                         'block_decoder.layer0.ffn.W_gate',
                         'block_decoder.correction.W'}
        assert expected_muon <= opt.muon_names
        adam_names = {p.name for p in opt.adam.parameters}
        assert {'embedding.W', 'output_proj.W',
                'block_proposal.horizon_embedding',
                'block_decoder.plan_positions',
                'block_decoder.dec_positions',
                'block_decoder.final_norm.gamma'} <= adam_names
        assert not opt.muon_names & adam_names
        before = {p.name: np.asarray(asnumpy(p.data)).copy() for p in opt.parameters}
        loss, norm = trainer.train_step()
        assert np.isfinite(loss) and np.isfinite(norm)
        assert opt.step_index == 1
        for group in ('block_proposal.horizon_back', 'block_decoder.correction.W',
                      'blocks.0.moe.experts.expert_1.W_gate', 'output_proj.W'):
            assert not np.array_equal(before[group], asnumpy(next(
                p.data for p in opt.parameters if p.name == group))), group
        assert trainer.block_proposal_resume_contract()['optimizer_kind'] == 'hybrid-muon'
    finally:
        _close(trainer)


def test_exact_joint_hybrid_checkpoint_resume(tmp_path):
    shard = _docs(tmp_path)
    checkpoint = str(tmp_path / 'hybrid-ckpt')
    live = _hybrid(shard, checkpoint_dir=checkpoint)
    resumed = _hybrid(shard)
    try:
        live.train_step()
        live.save()
        saved = load_two_stage_checkpoint(checkpoint, resumed.training_objective.net,
                                          optimizer=resumed.optimizer)
        assert saved['optimizer_kind'] == 'hybrid-muon'
        assert saved['muon_hyperparameters']['group_names'] == sorted(live.optimizer.muon_names)
        resumed.validate_block_proposal_resume_contract(saved)
        for field in ('step', 'tokens_processed', 'current_train_shard_idx'):
            setattr(resumed, field, saved[field])
        resumed.train_rng.bit_generator.state = saved['train_rng_state']
        resumed.val_rng.bit_generator.state = saved['val_rng_state']
        resumed._block_proposal_supervised_tokens = saved['supervised_tokens']
        resumed.sample_trace_sha256 = saved['data_provenance']['sample_trace_sha256']
        resumed.sample_trace_microbatches = saved['data_provenance']['sample_trace_microbatches']
        assert live.optimizer.muon_names == resumed.optimizer.muon_names
        loss_a, norm_a = live.train_step()
        loss_b, norm_b = resumed.train_step()
        np.testing.assert_array_equal([loss_a, norm_a], [loss_b, norm_b])
        assert live.sample_trace_sha256 == resumed.sample_trace_sha256
        for name, left in _state(live.optimizer).items():
            right = _state(resumed.optimizer)[name]
            if isinstance(left, dict):
                assert left.keys() == right.keys()
                for p_name, v in left.items():
                    np.testing.assert_array_equal(v, right[p_name], err_msg=f'{name}:{p_name}')
            else:
                assert left == right
        for a, b in zip(live.optimizer.parameters, resumed.optimizer.parameters):
            np.testing.assert_array_equal(asnumpy(a.data), asnumpy(b.data), err_msg=a.name)
    finally:
        _close(live)
        _close(resumed)


def test_hybrid_checkpoint_rejects_changed_momentum_and_wrong_optimizer(tmp_path):
    shard = _docs(tmp_path)
    checkpoint = str(tmp_path / 'hybrid-ckpt')
    live = _hybrid(shard, checkpoint_dir=checkpoint)
    changed = _hybrid(shard, momentum=.92)
    try:
        live.train_step()
        live.save()
        with pytest.raises(ValueError, match='incompatible Muon optimizer hyperparameters'):
            load_two_stage_checkpoint(checkpoint, changed.training_objective.net,
                                      optimizer=changed.optimizer)
    finally:
        _close(live)
        _close(changed)
