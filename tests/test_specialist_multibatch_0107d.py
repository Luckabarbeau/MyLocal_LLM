"""0107D: document-first multirow correctness and specialist training parity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.specialists.document_sampler import SpecialistDocumentSampler
from mini_llm.specialists.training_run import SpecialistTrainingRun
from test_specialist_training_run_0105e import prepare


def _sampler(original, *, batch_size, seed=42):
    return SpecialistDocumentSampler(
        original.root, context_length=32, batch_size=batch_size,
        eos_token_id=63, block_size=4, seed=seed, shape_bucket='fine')


def _mixed_length_batch(sampler):
    # The fixture includes short and long documents (21 and 35 pairs).
    for _ in range(30):
        inputs, targets, meta = sampler.next_batch()
        if len(set(meta['target_valid_lengths'].tolist())) > 1:
            return inputs, targets, meta
    pytest.fail('did not sample both short and long documents')


@pytest.mark.parametrize('batch_size', [1, 2, 3, 5, 8])
def test_arbitrary_batch_size_has_rowwise_masks_and_causal_document_metadata(tmp_path, batch_size):
    _, _, reference, _ = prepare(tmp_path)
    sampler = _sampler(reference, batch_size=batch_size)
    for _ in range(3):
        inputs, targets, meta = sampler.next_batch()
        b, t = inputs.shape
        assert b == batch_size
        assert targets.shape == (b, t)
        assert 0 < t <= sampler.context_length
        assert meta['target_loss_mask'].shape == (b, t)
        dual = meta['dual_resolution_metadata']
        assert dual['document_ids'].shape == (b, t)
        assert dual['position_ids'].shape == (b, t)
        assert dual['block_indices'].shape[0] == b
        assert dual['block_valid'].shape[0] == b
        for row, n in enumerate(meta['target_valid_lengths']):
            n = int(n)
            assert 0 < n <= t
            assert np.all(meta['target_loss_mask'][row, :n] == 1)
            assert not np.any(meta['target_loss_mask'][row, n:])
            assert np.all(inputs[row, n:] == 63)  # EOS padding, never supervised.
            assert np.all(dual['document_ids'][row, :n] == dual['document_ids'][row, 0])
            assert np.array_equal(dual['position_ids'][row, :n], np.arange(n))
            for indices, valid in zip(dual['block_indices'][row], dual['block_valid'][row]):
                if valid:
                    assert indices[-1] < n
        assert meta['source_name'] in sampler.names


@pytest.mark.parametrize('batch_size', [2, 3, 5])
def test_batched_paired_ce_matches_valid_weighted_individual_rows(tmp_path, batch_size):
    _, composition, reference, _ = prepare(tmp_path)
    inputs, targets, meta = _mixed_length_batch(_sampler(reference, batch_size=batch_size))
    batch_ce, batch_base_ce = composition.loss(
        inputs, targets, loss_mask=meta['target_loss_mask'],
        dual_resolution_metadata=meta['dual_resolution_metadata'],
        return_cache=False, return_foundation_ce=True)
    summed_ce = summed_base = 0.0
    for row, n in enumerate(meta['target_valid_lengths']):
        n = int(n)
        one_input = inputs[row:row + 1, :n]
        one_target = targets[row:row + 1, :n]
        one_mask = meta['target_loss_mask'][row:row + 1, :n]
        one_dual = build_dual_resolution_metadata(one_input, 63, 4)
        corrected, baseline = composition.loss(
            one_input, one_target, loss_mask=one_mask,
            dual_resolution_metadata=one_dual,
            return_cache=False, return_foundation_ce=True)
        summed_ce += corrected * n
        summed_base += baseline * n
    valid = meta['target_loss_mask'].sum()
    assert batch_ce == pytest.approx(summed_ce / valid, abs=2e-5)
    assert batch_base_ce == pytest.approx(summed_base / valid, abs=2e-5)


@pytest.mark.parametrize('batch_size', [2, 3])
@pytest.mark.parametrize('checkpoint', [True, False])
def test_multibatch_gradients_equal_valid_token_weighted_single_rows(tmp_path, batch_size, checkpoint):
    _, composition, reference, _ = prepare(tmp_path)
    inputs, targets, meta = _mixed_length_batch(_sampler(reference, batch_size=batch_size))
    # Exercise gradients inside transformer blocks, not only the zero-initialized
    # output projection (which would otherwise block deeper gradients).
    projection = composition.specialist.correction_proj.W
    projection.data[...] = np.eye(projection.data.shape[0], dtype=projection.data.dtype) * 0.003
    params = composition.parameters()
    total_valid = int(meta['target_loss_mask'].sum())
    composition.zero_grad()
    loss, cache = composition.loss(
        inputs, targets, loss_mask=meta['target_loss_mask'],
        dual_resolution_metadata=meta['dual_resolution_metadata'],
        return_cache=True, activation_checkpoint=checkpoint)
    composition.backward(cache, grad_scale=total_valid)
    full_grads = [None if p.grad is None else p.grad.copy() for p in params]

    composition.zero_grad()
    for row, n in enumerate(meta['target_valid_lengths']):
        n = int(n)
        one_input = inputs[row:row + 1, :n]
        one_target = targets[row:row + 1, :n]
        one_mask = meta['target_loss_mask'][row:row + 1, :n]
        one_dual = build_dual_resolution_metadata(one_input, 63, 4)
        one_loss, cache = composition.loss(
            one_input, one_target, loss_mask=one_mask,
            dual_resolution_metadata=one_dual, return_cache=True,
            activation_checkpoint=checkpoint)
        composition.backward(cache, grad_scale=n)
    assert any(np.any(g != 0) for g in full_grads if g is not None)
    for p, full in zip(params, full_grads):
        if full is not None:
            np.testing.assert_allclose(p.grad, full, rtol=2e-4, atol=2e-5,
                                       err_msg=p.name)


@pytest.mark.parametrize('batch_size', [2, 3, 5])
def test_multirow_train_logs_and_deterministic_resume(tmp_path, batch_size):
    foundation, composition, original, cfg = prepare(tmp_path)
    sampler = _sampler(original, batch_size=batch_size)
    run = SpecialistTrainingRun(
        composition, sampler, config=cfg, checkpoint_dir=tmp_path/'specialist')
    frozen = [p.data.copy() for p in foundation.model.parameters()]
    report = run.train_step(source_metrics=True, foundation_train_ce=True,
                            measure_step_time=True)
    assert report['supervised_tokens_step'] == sum(
        record['supervised_tokens'] for record in report['per_source'].values())
    assert report['foundation_ce'] == pytest.approx(report['loss'], abs=1e-6)
    assert report['capacity_tokens_step'] == batch_size * 32 * cfg.grad_accum_steps
    assert report['executed_tokens_step'] <= report['capacity_tokens_step']
    assert sum(v['samples'] for v in report['per_source'].values()) == batch_size * cfg.grad_accum_steps
    assert sum(v['valid_source_tokens'] for v in report['per_source'].values()) == report['supervised_tokens_step']
    assert all(p.grad is None for p in foundation.model.parameters())
    for expected, p in zip(frozen, foundation.model.parameters()):
        np.testing.assert_array_equal(expected, p.data)
    run.save_checkpoint()
    resumed = SpecialistTrainingRun.resume(
        tmp_path/'specialist', foundation, _sampler(original, batch_size=batch_size, seed=77),
        config=cfg)
    a = run.train_step(source_metrics=True, foundation_train_ce=True)
    b = resumed.train_step(source_metrics=True, foundation_train_ce=True)
    assert a['supervised_tokens'] == b['supervised_tokens']
    assert a['loss'] == pytest.approx(b['loss'], abs=1e-7)
    assert a['foundation_ce'] == pytest.approx(b['foundation_ce'], abs=1e-7)
    for x, y in zip(run.parameters, resumed.parameters):
        np.testing.assert_array_equal(x.data, y.data)
    wrong_sampler = _sampler(original, batch_size=batch_size + 1)
    with pytest.raises(ValueError, match='batch_size'):
        SpecialistTrainingRun.resume(tmp_path/'specialist', foundation, wrong_sampler,
                                     config=cfg)


def test_batch_size_validation_from_cli_and_sampler(tmp_path):
    from scripts.train_0105_specialist import arguments, validate_paths
    _, _, original, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match='positive'):
        _sampler(original, batch_size=0)
    with pytest.raises(ValueError, match='positive'):
        _sampler(original, batch_size=-3)
    foundation = tmp_path/'foundation'
    for bad in (0, -4):
        args = arguments([
            '--foundation-checkpoint', str(foundation),
            '--mixed-shard-root', str(original.root),
            '--specialist-checkpoint', str(tmp_path/'new_specialist'),
            '--batch-size', str(bad),
        ])
        with pytest.raises(ValueError, match='batch size must be positive'):
            validate_paths(args)
