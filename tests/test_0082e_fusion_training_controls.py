"""0082E: regression tests for opt-in stability controls and early stopping."""
import numpy as np
import pytest

from examples.train_probability_fusion_0082a import (
    _clip_grad_norm_, _should_stop_early, main,
)
from mini_llm.backend import asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def test_global_gradient_norm_clip_scales_only_grads():
    net = ProbabilityResidualCombiner(FusionConfig(vocab_size=13, seed=82))
    params = net.parameters()
    for p in params:
        p.grad[...] = 2.0
    before = [asnumpy(p.data).copy() for p in params]
    norm = _clip_grad_norm_(params, 1.0)
    assert norm > 1.0
    observed = np.sqrt(sum(float(np.sum(asnumpy(p.grad) ** 2)) for p in params))
    assert observed == pytest.approx(1.0, rel=2e-5)
    for p, baseline in zip(params, before):
        np.testing.assert_array_equal(asnumpy(p.data), baseline)
    assert _clip_grad_norm_(params, 0.0) is None


def test_early_stop_patience():
    assert not _should_stop_early(500, 0)
    assert not _should_stop_early(2, 3)
    assert _should_stop_early(3, 3)
    assert _should_stop_early(4, 3)


def test_training_early_stops_and_saves_best(tmp_path, capsys):
    rng = np.random.default_rng(8282)
    vocab = 17
    for split, size in (('train', 24), ('val', 12)):
        base = rng.normal(size=(size, vocab)).astype(np.float16)
        for member in (0, 1):
            np.save(tmp_path/f'{split}_{member}.npy', base + 0.02 * member)
        np.save(tmp_path/f'{split}_target.npy',
                rng.integers(0, vocab, size=size, dtype=np.int64))
    output = tmp_path/'best_fusion.npz'
    main([
        '--member', f'0:{tmp_path}/train_0.npy',
        '--member', f'1:{tmp_path}/train_1.npy',
        '--targets', str(tmp_path/'train_target.npy'),
        '--val-member', f'0:{tmp_path}/val_0.npy',
        '--val-member', f'1:{tmp_path}/val_1.npy',
        '--val-targets', str(tmp_path/'val_target.npy'),
        '--batch-rows', '4', '--steps', '4',
        '--sample-all-rows', '--grad-clip', '1', '--weight-decay', '0.01',
        '--early-stop-patience', '2', '--eval-interval', '1',
        '--log-interval', '2', '--output', str(output),
    ])
    assert output.exists()
    output_text = capsys.readouterr().out
    assert 'fusion_best ' in output_text
    assert 'fusion_best_final ' in output_text
    assert 'train_gain_mean=' in output_text
    assert 'grad_norm=' in output_text
    assert 'saved=' in output_text


@pytest.mark.parametrize('extra', [('--early-stop-patience', '-1'),
                                   ('--grad-clip', '-2'),
                                   ('--weight-decay', '-0.01')])
def test_invalid_training_controls_rejected(extra):
    with pytest.raises(SystemExit):
        main(['--member', '0:no_file', '--targets', 'not_found',
              '--output', 'never.npz'] + list(extra))


def test_early_stop_requires_validation():
    with pytest.raises(SystemExit):
        main(['--member', '0:no_file', '--targets', 'not_found',
              '--output', 'never.npz', '--early-stop-patience', '3'])


def test_early_stop_stops_at_patience_and_keeps_best(tmp_path, capsys, monkeypatch):
    from examples import train_probability_fusion_0082a as trainer
    rng = np.random.default_rng(8283)
    for member in (0, 1):
        np.save(tmp_path/f'{member}.npy', rng.normal(size=(8, 13)).astype(np.float16))
    np.save(tmp_path/'targets.npy', rng.integers(0, 13, size=8, dtype=np.int64))
    score_sequence = iter((4.0, 3.5, 3.6, 3.7))
    evaluations = []

    def mock_evaluate(*args, **kwargs):
        value = next(score_sequence)
        evaluations.append(value)
        return value, 4.1, 4.0, np.array([4.2, 4.3])

    monkeypatch.setattr(trainer, '_evaluate', mock_evaluate)
    path = tmp_path/'best.npz'
    trainer.main([
        '--member', f'0:{tmp_path}/0.npy',
        '--member', f'1:{tmp_path}/1.npy',
        '--targets', str(tmp_path/'targets.npy'),
        '--val-member', f'0:{tmp_path}/0.npy',
        '--val-member', f'1:{tmp_path}/1.npy',
        '--val-targets', str(tmp_path/'targets.npy'),
        '--output', str(path), '--steps', '100', '--batch-rows', '2',
        '--early-stop-patience', '2', '--eval-interval', '1',
    ])
    assert evaluations == [4.0, 3.5, 3.6, 3.7]
    assert path.exists()
    text = capsys.readouterr().out
    assert 'fusion_best step=2 val_ce=3.50000' in text
    assert 'fusion_early_stop step=4 best_step=2' in text
    assert 'fusion_best_final step=2 val_ce=3.50000' in text
