"""0079J: forward-only grouped validation, first-token science and ablation."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer


def make_trainer(*, feedback='full', block_size=3):
    cfg = ModelConfig(tokenizer_vocab_size=48, context_length=32, n_layers=1,
                      d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                      d_ff=32, n_experts=1, top_k=1, dtype='float64')
    return WindowRefinementTrainer(
        DecoderLanguageModel(cfg, rng_seed=19, dtype='float64'),
        RefinementReferenceConfig(block_size=block_size, feedback_mode=feedback,
                                  enable_shared_context_training=True), rng_seed=20)


def sample():
    tokens = xp.asarray([[4, 5, 6, 7, 8, 9, 10, 11, 12],
                         [14, 15, 16, 17, 18, -100, -100, -100, -100]])
    valid = xp.asarray([[True] * 9, [True] * 5 + [False] * 4])
    return tokens, valid


@pytest.mark.parametrize('group', [1, 2, 3])
def test_forward_only_evaluation_matches_training_without_changing_gradients(group):
    ref = make_trainer()
    tokens, valid = sample()
    ref.zero_grad()
    full = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid, group_size=group)
    before = [asnumpy(p.grad).copy() for p in ref.parameters()]
    nocache = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                               group_size=group, compute_gradients=False)
    for key in ('commitment_ce', 'initial_ce', 'total_loss'):
        np.testing.assert_allclose(float(full[key]), float(nocache[key]), rtol=1e-12)
    for key in ('initial_position_ce_sums', 'commitment_position_ce_sums', 'position_counts'):
        np.testing.assert_allclose(asnumpy(full[key]), asnumpy(nocache[key]), atol=1e-12)
    assert nocache['supervised_tokens'] == 10
    assert int(asnumpy(nocache['position_counts']).sum()) == 10
    for original, param in zip(before, ref.parameters()):
        np.testing.assert_array_equal(original, asnumpy(param.grad))


def test_position_one_improvement_is_exclusively_self_conditioning():
    ref = make_trainer()
    tokens, valid = sample()
    a = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                         group_size=3, compute_gradients=False)
    changed = tokens.copy()
    # Change FUTURE targets of the first block without changing its first
    # target or the causal prefix; its pass-0 and pass-1 position-1 logits
    # may not depend on those changed targets.
    changed[0, 3:5] = xp.asarray([40, 41])
    b = ref.train_window(changed, prefix_length=2, valid_token_mask=valid,
                         group_size=3, compute_gradients=False)
    # Changing future tokens of an earlier block changes the context used by
    # later blocks, so isolate first target block to compare raw pass 0/1 CE.
    base_metrics, _ = ref.forward(tokens[:, :2], tokens[:, 2:5], record_tape=False)
    altered_metrics, _ = ref.forward(changed[:, :2], changed[:, 2:5], record_tape=False)
    assert float(base_metrics['initial_position_ce_sums'][0]) == pytest.approx(
        float(altered_metrics['initial_position_ce_sums'][0]))
    assert float(base_metrics['commitment_position_ce_sums'][0]) == pytest.approx(
        float(altered_metrics['commitment_position_ce_sums'][0]))


def test_unknown_feedback_ablation_has_full_backward_and_distinct_metadata(tmp_path):
    from mini_llm.predictive_refinement.checkpoint import (
        _architecture_metadata, save_refinement_checkpoint, load_refinement_checkpoint,
    )
    normal, control = make_trainer(), make_trainer(feedback='unknown')
    tokens, valid = sample()
    normal.zero_grad()
    measured = normal.train_window(tokens, prefix_length=2, valid_token_mask=valid)
    control.zero_grad()
    abl = control.train_window(tokens, prefix_length=2, valid_token_mask=valid)
    assert float(abl['total_loss']) > 0
    assert np.isfinite(float(abl['total_loss']))
    assert any(np.any(asnumpy(p.grad)) for p in control.parameters())
    assert _architecture_metadata(control)['feedback_type'] == 'unknown_position_control'
    save_refinement_checkpoint(tmp_path, normal)
    with pytest.raises(ValueError, match='feedback_type'):
        load_refinement_checkpoint(tmp_path, control)
    assert float(measured['commitment_ce']) != float(abl['commitment_ce'])


def test_fixed_windows_dont_advance_train_rng_and_are_reproducible():
    from examples.train_predictive_refinement_text import _fixed_validation_windows, _evaluate, parse_args
    class Dummy:
        def sample(self, rng, *, batch_size, prefix_length, block_size, window_blocks):
            length = prefix_length + block_size * window_blocks
            arr = rng.integers(2, 45, size=(batch_size, length))
            return arr, np.ones_like(arr, dtype=bool), prefix_length, {'stub': batch_size}
    args = parse_args(['--shard-dir', '.', '--val-batches', '3', '--batch-size', '2',
                       '--prefix-length', '2', '--block-size', '3', '--window-blocks', '2'])
    val = Dummy()
    first = _fixed_validation_windows(val, args)
    second = _fixed_validation_windows(val, args)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a[0], b[0])
    ref = make_trainer()
    rng = np.random.default_rng(8)
    state_before = rng.bit_generator.state.copy()
    a = _evaluate(ref, val, rng, args, fixed_windows=first)
    b = _evaluate(ref, val, rng, args, fixed_windows=second)
    assert rng.bit_generator.state == state_before
    assert a['commitment_ce'] == pytest.approx(b['commitment_ce'])
    assert a['first_position_refinement_gain'] == pytest.approx(b['first_position_refinement_gain'])
    assert a['supervised_tokens'] == 36
    assert len(a['commit_ce_by_pos']) == 3


def test_ablation_args_and_profile_validation():
    from examples.train_predictive_refinement_text import parse_args
    a = parse_args(['--shard-dir', '.', '--feedback-mode', 'unknown',
                    '--refinement-group-size', '8', '--profile-steps', '2'])
    assert a.feedback_mode == 'unknown' and a.refinement_group_size == 8
    with pytest.raises(SystemExit):
        parse_args(['--shard-dir', '.', '--profile-steps', '-1'])
    with pytest.raises(ValueError):
        RefinementReferenceConfig(feedback_mode='not-a-mode')


def test_65k_resume_evaluation_is_read_only_and_logs_position_metrics(tmp_path, monkeypatch, capsys):
    import json
    from examples import train_predictive_refinement_text as script
    docs = np.asarray([4, 5, 6, 1, 7, 8, 9, 10, 1], dtype=np.uint16)
    for split in ('train', 'val'):
        docs.tofile(tmp_path / f'{split}_shard_0000.bin')
    (tmp_path / 'tokenizer.json').write_text('{"stub": true}')
    class FakeTokenizer:
        def __len__(self):
            return 65280
    monkeypatch.setattr(script, '_load_tokenizer', lambda path: (FakeTokenizer(), 1))
    ckpt = tmp_path / 'ckpt'
    args = ['--shard-dir', str(tmp_path), '--checkpoint-dir', str(ckpt),
            '--steps', '1', '--batch-size', '1', '--grad-accum', '1',
            '--prefix-length', '2', '--block-size', '2', '--window-blocks', '2',
            '--refinement-group-size', '2', '--layers', '1', '--d-model', '16',
            '--n-q-heads', '2', '--n-kv-heads', '1', '--d-ff', '32',
            '--experts', '1', '--log-every', '1', '--save-every', '1',
            '--val-every', '1', '--val-batches', '2', '--generate', '0']
    script.main(args)
    state_before = (ckpt / 'training_state.json').read_bytes()
    metrics_file = tmp_path / 'evaluation.jsonl'
    script.main(args + ['--resume', str(ckpt), '--eval-only',
                        '--val-batches', '3', '--metrics-jsonl', str(metrics_file)])
    assert (ckpt / 'training_state.json').read_bytes() == state_before
    output = capsys.readouterr().out
    assert 'evaluation checkpoint_step=1' in output
    assert 'gain_first=' in output and 'gain_pos=' in output
    metrics = json.loads(metrics_file.read_text().splitlines()[-1])
    assert metrics['kind'] == 'validation'
    assert metrics['supervised_tokens'] > 0
    assert len(metrics['initial_ce_by_pos']) == 2
