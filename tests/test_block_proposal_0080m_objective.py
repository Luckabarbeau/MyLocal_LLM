"""0080M production optimizer/validation integration, CPU reference parity."""
from pathlib import Path
import json
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.block_proposal.objective import BlockProposalObjective
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.optim.grad_clip import clip_grad_global_norm
from mini_llm.train_extended import ExtendedTrainer
from test_block_proposal_0080l_baselines import _make, _window, _slice_window

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='0080M CPU numerical parity')


def _make_production(tmp_path, k=8, accumulation=1, weighting='supervised_tokens',
                     *, group_size=2, checkpoint_dir=None, loss_scale=1.0):
    original = _make(k)
    objective = BlockProposalObjective(original.net, block_group_size=group_size)
    trainer = ExtendedTrainer(
        model=original.model, training_objective=objective,
        train_shard_paths=[str(tmp_path / 'dummy.bin')],
        val_shard_paths=[str(tmp_path / 'dummy.bin')],
        batch_size=2 if accumulation == 1 else 1,
        seq_length=24, grad_accum_steps=accumulation,
        gradient_accumulation_weighting=weighting,
        warmup_steps=0, total_steps=10, peak_lr=1e-4,
        grad_clip=10000., weight_decay=.1, eos_token_id=100,
        checkpoint_dir=checkpoint_dir,
        rng_seed=42, loss_scale=loss_scale,
    )
    # Regression fixture deliberately supplies an already frozen routing
    # policy; separate test below verifies real production stochastic policy.
    trainer._configure_routed_prefix_metadata = lambda metadata, *, training: metadata
    return original, objective, trainer


def _feed(trainer, *, accumulation):
    samples = [_window()] if accumulation == 1 else [_slice_window(0), _slice_window(1)]
    queue = iter(samples)
    def get_batch(return_metadata=False):
        assert return_metadata
        src, tgt, mask, md = next(queue)
        trainer._last_train_batch_source = 'fixture'
        return src, tgt, dict(md, target_loss_mask=mask.astype(np.float32))
    trainer.get_train_batch = get_batch
    return samples


def _params(params):
    return {p.name: asnumpy(p.data).copy() for p in params}


@pytest.mark.parametrize('k', [1, 8])
@pytest.mark.parametrize('accumulation', [1, 2])
@pytest.mark.parametrize('loss_scale', [1.0, 32.0])
def test_production_objective_legacy_forward_gradient_optimizer_parity(tmp_path, k, accumulation, loss_scale):
    legacy = _make(k)
    _, objective, trainer = _make_production(tmp_path, k, accumulation, loss_scale=loss_scale)
    samples = _feed(trainer, accumulation=accumulation)

    # Legacy 0080 objective, normalized by *unique valid targets* across the
    # step, paired with the SAME production AdamW and scheduler choices.
    legacy.zero_grad()
    nvalid = sum(int(sample[2].sum()) for sample in samples)
    reference_loss = 0.
    for source, targets, mask, md in samples:
        weight = float(mask.sum()) / nvalid
        metric = legacy.train_window(source, targets,
            valid_target_mask=mask, memory_metadata=md,
            block_group_size=2, grad_scale=weight)
        reference_loss += metric['total_loss'] * weight
    pre_update_grads = {p.name: asnumpy(p.grad).copy() for p in legacy.parameters()}
    # Capture gradients at the real optimizer boundary: production clears
    # its gradient buffers immediately after the update.
    captured = {}
    optimizer_step = trainer.optimizer.step
    def instrumented_step(*args, **kwargs):
        captured.update({p.name: asnumpy(p.grad).copy()
                         for p in trainer.optimizer.parameters})
        return optimizer_step(*args, **kwargs)
    trainer.optimizer.step = instrumented_step
    loss, grad_norm = trainer.train_step()
    np.testing.assert_allclose(loss, reference_loss, rtol=2e-7, atol=1e-6)
    assert grad_norm > 0
    for p in objective.parameters():
        np.testing.assert_allclose(pre_update_grads[p.name], captured[p.name],
                                   rtol=2e-6, atol=1e-7, err_msg=p.name)
    assert [id(p) for p in trainer.optimizer.parameters] == [id(p) for p in objective.parameters()]
    assert len(trainer.optimizer.parameters) == len({id(p) for p in objective.parameters()})
    assert trainer.optimizer.step_index == 1

    # Compare the matched legacy optimizer update after the production step.
    from mini_llm.optim.adamw import AdamW
    baseline_opt = AdamW(legacy.parameters(), lr=1e-4, weight_decay=.1)
    lr = trainer.scheduler(0)
    clip_grad_global_norm(legacy.parameters(), max_norm=10000.)
    baseline_opt.step(lr=lr)
    for p, q in zip(legacy.parameters(), objective.parameters()):
        np.testing.assert_allclose(p.data, q.data, rtol=2e-7, atol=1e-8, err_msg=p.name)


def test_evaluation_uses_original_production_validation_wrapper(tmp_path):
    legacy, obj, trainer = _make_production(tmp_path)
    src, labels, mask, meta = _window()
    result, scored, used = trainer._validation_batch_loss(
        src, labels, dict(meta, target_loss_mask=mask.astype(np.float32)))
    reference = legacy.train_window(src, labels, valid_target_mask=mask,
                                    memory_metadata=meta, compute_gradients=False,
                                    block_group_size=2)
    np.testing.assert_allclose(result, reference['final_ce'], rtol=0, atol=1e-12)
    assert scored.shape == labels.shape and int(used.sum()) == 13


def test_production_routing_policy_and_forbidden_future_query(tmp_path):
    _, objective, trainer = _make_production(tmp_path)
    src, y, mask, metadata = _window()
    trainer._configure_routed_prefix_metadata = ExtendedTrainer._configure_routed_prefix_metadata.__get__(trainer)
    policy = trainer._configure_routed_prefix_metadata(metadata, training=True)
    assert policy['router_stochastic'] is True
    assert isinstance(policy['router_gumbel_seed'], int)
    assert policy['router_temperature'] > 0
    prepared = objective.prepare_metadata(policy)
    assert prepared['causal_prefix_query'] is True
    assert prepared['causal_masked_target_refinement'] is True
    with pytest.raises(ValueError, match='noncausal'):
        objective.evaluate_batch(src, y, mask, dict(policy, causal_prefix_query=False))
    with pytest.raises(ValueError, match='binary'):
        objective.evaluate_batch(src, y, np.full(mask.shape, .5), metadata)


def test_production_save_records_full_joint_parameters(tmp_path):
    legacy, obj, trainer = _make_production(tmp_path, checkpoint_dir=str(tmp_path / 'ckpt'))
    _feed(trainer, accumulation=1)
    trainer.train_step()
    trainer.save()
    with (tmp_path / 'ckpt' / 'training_state.json').open(encoding='utf8') as handle:
        state = json.load(handle)
    assert state['architecture'] == obj.checkpoint_metadata()
    assert state['prediction_objective'] == 'block-proposal'
    assert state['optimizer_step'] == 1
    assert len(trainer.optimizer.parameters) == len(obj.parameters())
    # Legacy 0080 checkpoint loader remains usable at compatible settings.
    recovered = _make(8)
    from mini_llm.optim.adamw import AdamW
    fresh_opt = AdamW(recovered.parameters(), lr=trainer.optimizer.lr,
                      weight_decay=trainer.optimizer.weight_decay)
    loaded = load_two_stage_checkpoint(tmp_path / 'ckpt', recovered.net,
                                       optimizer=fresh_opt)
    assert loaded['step'] == 1
    for p, q in zip(recovered.parameters(), obj.parameters()):
        np.testing.assert_allclose(p.data, q.data, rtol=0, atol=0, err_msg=p.name)


def test_rejects_inconsistent_model(tmp_path):
    legacy = _make(8)
    other = _make(8)
    with pytest.raises(ValueError, match='bound to model'):
        ExtendedTrainer(model=other.model, training_objective=BlockProposalObjective(legacy.net),
                        train_shard_paths=[], val_shard_paths=[], seq_length=24,
                        eos_token_id=100)


def test_real_production_document_first_batch_and_joint_optimizer(tmp_path):
    # Exercise actual production sampler, history-valid-start masks, EOS
    # boundaries, production route policy, and scheduler with no mock batch.
    docs = [list(range(1, 45)) + [100], [60, 61, 62, 100],
            list(range(10, 65)) + [100], list(range(21, 52)) + [100]]
    shard = tmp_path / 'packed.bin'
    np.asarray([v for doc in docs for v in doc], dtype='<u2').tofile(shard)
    trainers = []
    for _ in range(2):
        legacy = _make(8)
        objective = BlockProposalObjective(legacy.net, block_group_size=2)
        trainers.append(ExtendedTrainer(
            legacy.model, [str(shard)], [str(shard)], batch_size=2,
            seq_length=24, grad_accum_steps=2, warmup_steps=0,
            total_steps=10, peak_lr=1e-4, grad_clip=10.,
            gradient_accumulation_weighting='supervised_tokens',
            eos_token_id=100, rng_seed=101, training_objective=objective,
        ))
    try:
        # Production's sampling law and optimizer state are deterministic
        # without a separate 0080-specific source sampler.
        for trainer in trainers:
            loss, grad_norm = trainer.train_step()
            assert np.isfinite(loss) and np.isfinite(grad_norm)
            assert 0 < trainer._last_step_supervised_tokens <= 32
            assert (trainer._last_step_block_proposal_metrics['unique_supervised_tokens']
                    == int(trainer._last_step_supervised_tokens))
            assert trainer.optimizer.step_index == 1
        assert trainers[0].step == trainers[1].step == 1
        assert (trainers[0]._last_step_supervised_tokens
                == trainers[1]._last_step_supervised_tokens)
        for x, y in zip(trainers[0].optimizer.parameters, trainers[1].optimizer.parameters):
            np.testing.assert_allclose(x.data, y.data, rtol=0, atol=0, err_msg=x.name)
    finally:
        # Windows needs memory-map handles released before TemporaryDirectory cleanup.
        for trainer in trainers:
            for cache in (trainer.shards, trainer.val_shards):
                for mapped in cache.values():
                    trainer._close_mapped_shard(mapped)
                cache.clear()


def test_requires_unique_supervised_token_accumulation(tmp_path):
    original = _make(8)
    with pytest.raises(ValueError, match='supervised_tokens'):
        ExtendedTrainer(
            model=original.model,
            training_objective=BlockProposalObjective(original.net),
            train_shard_paths=[], val_shard_paths=[],
            seq_length=24, eos_token_id=100,
            gradient_accumulation_weighting='microbatch',
        )
