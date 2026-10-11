"""0074B: optimizer counterfactuals, clipping, lifecycle, equivalence."""

import csv
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.diagnostics.adamw_update import (
    counterfactual_direction, group_update_rows,
)
from mini_llm.diagnostics.gradient_noise import GradientNoiseMonitor
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm
from mini_llm.parameter import Parameter
from test_gradient_noise import _make_trainer


def _param(shape=(3, 4), *, decay=False):
    return Parameter(xp.zeros(shape, dtype="float32"),
                     name="blocks.0.attention.Wq", decay=decay)


def test_counterfactual_matches_actual_adamw_with_moments_and_local_age():
    p = _param()
    opt = AdamW([p], lr=0.01, beta1=0.9, beta2=0.95, eps=1e-7,
                weight_decay=0.0)
    rng = np.random.default_rng(74)
    opt.m[0][...] = rng.normal(size=p.shape).astype('float32') * 0.15
    opt.v[0][...] = np.abs(rng.normal(size=p.shape).astype('float32')) * 0.3
    opt.step_index = 15
    opt.parameter_birth_steps[0] = 5
    g = rng.normal(size=p.shape).astype('float32') * 0.1
    p.grad[...] = g
    m_prior = opt.m[0].copy()
    v_prior = opt.v[0].copy()
    before = opt.master_weights[0].copy()
    expected = counterfactual_direction(
        g, m_prior, v_prior, beta1=opt.beta1, beta2=opt.beta2,
        eps=opt.eps, learning_rate=opt.lr,
        local_step=opt.step_index + 1 - opt.parameter_birth_steps[0])
    np.testing.assert_array_equal(m_prior, opt.m[0])
    np.testing.assert_array_equal(v_prior, opt.v[0])
    opt.step()
    np.testing.assert_allclose(before - opt.master_weights[0], expected, rtol=3e-5, atol=1e-8)


def test_full_model_clipping_is_used_not_slice_clipping(tmp_path):
    p = _param((2, 2))
    other = Parameter(np.zeros((2, 2), 'float32'), name="extra_large", decay=False)
    opt = AdamW([p, other], lr=0.01, weight_decay=0.0)
    monitor = GradientNoiseMonitor(
        [p, other], accum_steps=4, accum_points="2,4",
        categories="attention", interval=1, output=str(tmp_path / "noise.csv"),
        adamw_optimizer=opt, adamw_output=str(tmp_path / "adamw.csv"),
        grad_clip=1.0)
    assert monitor.begin(0)
    for i in range(4):
        p.grad[...] += 0.25
        other.grad[...] += 25.0
        monitor.capture(i + 1)
    assert not (tmp_path / 'adamw.csv').exists()
    rows = monitor.finish()
    assert len(monitor.adamw_rows) == 2
    at4 = monitor.adamw_rows[-1]
    assert at4["global_gradient_norm"] > 25
    assert at4["global_clip_scale"] < 0.04
    assert at4["heldout_direction_cosine"] is None
    assert monitor.adamw_rows[0]["heldout_direction_cosine"] == pytest.approx(1)
    # The monitor has not touched either gradient or the optimizer state.
    np.testing.assert_array_equal(p.grad, np.ones((2, 2)))
    np.testing.assert_array_equal(other.grad, np.full((2, 2), 100.0))
    assert opt.step_index == 0
    # Match trainer normalization before global gradient clipping.
    p.grad[...] /= 4
    other.grad[...] /= 4
    norm, actual_clip, finite = clip_grad_global_norm(opt.parameters, max_norm=1.)
    assert finite and actual_clip == pytest.approx(at4['global_clip_scale'])
    before = opt.master_weights[0].copy()
    opt.step()
    actual = before - opt.master_weights[0]
    # Single sampled parameter group includes entire p; its hypothetical
    # full-window Adam direction must match the optimizer's actual update.
    assert np.linalg.norm(actual) == pytest.approx(at4['adamw_update_norm'], rel=1e-5)
    monitor.commit(rows)
    with (tmp_path / 'adamw.csv').open(newline='') as f:
        csvrows = list(csv.DictReader(f))
    assert len(csvrows) == 2


def test_disjoint_suffix_quality_and_no_zero_denominator():
    p = _param((2, 2))
    opt = AdamW([p], lr=0.01, weight_decay=0.0)
    snapshots = {2: np.array([[1., 0.], [0., 0.]], np.float32),
                 4: np.array([[1., 3.], [0., 0.]], np.float32)}
    rows = group_update_rows(
        step=1, label='attention_early', name=p.name, points=(2, 4),
        snapshots=snapshots, m0=opt.m[0], v0=opt.v[0],
        optimizer=opt, optimizer_index=0,
        clip_norms={2: 1., 4: np.sqrt(10.)}, grad_clip=10.,
        loss_scale=1., token_snapshots={2: 8, 4: 16}, learning_rate=.01)
    assert rows[0]['heldout_gradient_cosine'] == pytest.approx(0.)
    assert rows[0]['heldout_direction_cosine'] == pytest.approx(0.)
    assert rows[0]['heldout_first_order_gain'] == pytest.approx(0.)
    assert rows[1]['heldout_direction_cosine'] is None
    assert rows[1]['adamw_update_cosine_to_reference'] == pytest.approx(1)
    assert rows[1]['adamw_relative_update_difference'] == pytest.approx(0)


def test_second_moment_changes_update_even_if_momentum_dominates():
    g = np.array([1., -1.], 'float32')
    m = np.array([1., 1.], 'float32')
    v = np.array([.1, 10.], 'float32')
    u = counterfactual_direction(g, m, v, beta1=.9, beta2=.95,
                                 eps=1e-8, learning_rate=.001, local_step=10)
    assert u[0] > u[1]
    u_flat_v = counterfactual_direction(g, m, np.array([.1, .1], 'float32'),
        beta1=.9, beta2=.95, eps=1e-8, learning_rate=.001, local_step=10)
    assert u[1] < u_flat_v[1]  # second-moment preconditioning matters
    np.testing.assert_array_equal(m, np.ones(2, 'float32'))
    np.testing.assert_array_equal(v, np.array([.1, 10.], 'float32'))


def test_end_to_end_74b_preserves_training_state_and_csv(tmp_path):
    plain = _make_trainer(tmp_path, False)
    monitored = _make_trainer(tmp_path, True)
    monitored.gradient_noise.adamw_optimizer = monitored.optimizer
    monitored.gradient_noise.adamw_output = tmp_path / 'adamw.csv'
    monitored.gradient_noise.grad_clip = monitored.grad_clip
    for _ in range(2):
        la, ga = plain.train_step()
        lb, gb = monitored.train_step()
        assert (la, ga) == pytest.approx((lb, gb), rel=0, abs=1e-6)
    assert plain.train_rng.bit_generator.state == monitored.train_rng.bit_generator.state
    assert plain.optimizer.step_index == monitored.optimizer.step_index
    for a, b in zip(plain.optimizer.parameters, monitored.optimizer.parameters):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.grad, b.grad)
    for state in ('m', 'v', 'master_weights'):
        for a, b in zip(getattr(plain.optimizer, state), getattr(monitored.optimizer, state)):
            np.testing.assert_array_equal(a, b)
    with (tmp_path / 'adamw.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {int(r['step']) for r in rows} == {1, 2}
    assert all(int(r['reference_accumulation']) == 8 for r in rows)
    assert all(r['heldout_direction_cosine'] == '' for r in rows if r['accumulation_length'] == '8')


def test_74b_enables_74a_observer_without_extra_flag(tmp_path):
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.config import ModelConfig
    from mini_llm.train_extended import ExtendedTrainer
    data = tmp_path / "train_shard_00000.bin"
    np.random.default_rng(777).integers(0, 256, size=2048, dtype=np.uint16).tofile(data)
    trainer = ExtendedTrainer(
        DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=74, dtype="float32"),
        train_shard_paths=[str(data)], val_shard_paths=[str(data)],
        batch_size=1, seq_length=16, grad_accum_steps=4,
        total_steps=10, warmup_steps=2, adamw_update_monitor=True,
        gradient_noise_accum_points="2,4", gradient_noise_interval=1,
        gradient_noise_output=str(tmp_path / 'gradient.csv'),
        adamw_update_output=str(tmp_path / 'updates.csv'))
    assert trainer.gradient_noise is not None
    assert trainer.gradient_noise.adamw_optimizer is trainer.optimizer
    trainer.train_step()
    assert (tmp_path / 'updates.csv').exists()
    assert (tmp_path / 'gradient.csv').exists()


def test_loss_scale_cancels_in_counterfactual():
    p = _param((2, 2))
    opt = AdamW([p], lr=0.001)
    base = {2: np.full((2, 2), 2., dtype='float32'),
            4: np.full((2, 2), 4., dtype='float32')}
    common = dict(
        step=1, label='attention_early', name=p.name, points=(2, 4),
        m0=opt.m[0], v0=opt.v[0], optimizer=opt, optimizer_index=0,
        grad_clip=0.5, token_snapshots={2: 4, 4: 8}, learning_rate=.001)
    a = group_update_rows(snapshots=base,
        clip_norms={2: 4., 4: 8.}, loss_scale=1., **common)
    b = group_update_rows(
        snapshots={n: x * 128 for n, x in base.items()},
        clip_norms={2: 4. * 128, 4: 8. * 128}, loss_scale=128., **common)
    for ra, rb in zip(a, b):
        for k in ('global_clip_scale', 'adamw_update_norm',
                  'adamw_update_cosine_to_reference', 'innovation_norm'):
            assert ra[k] == pytest.approx(rb[k])


def test_skipped_cycle_never_commits_adamw(tmp_path):
    p = _param()
    opt = AdamW([p])
    mon = GradientNoiseMonitor([p], accum_steps=4, accum_points="2,4",
        categories='attention', interval=1, output=str(tmp_path/'noise.csv'),
        adamw_optimizer=opt, adamw_output=str(tmp_path/'adamw.csv'))
    assert mon.begin(0)
    for i in range(4):
        p.grad += 1.
        mon.capture(i+1)
    _ = mon.finish()
    assert mon.adamw_rows and not (tmp_path/'adamw.csv').exists()
    mon.begin(0)  # aborted / skipped step: rows must be discarded
    assert not mon.adamw_rows
    assert not (tmp_path/'adamw.csv').exists()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy GPU unavailable')
def test_gpu_offload_moment_snapshot(tmp_path):
    import cupy
    p = Parameter(cupy.zeros((8, 8), dtype='float32'),
                  name='blocks.0.attention.Wq')
    # Requires MINI_LLM_OPTIMIZER_OFFLOAD=full or moments on the target machine.
    opt = AdamW([p])
    mon = GradientNoiseMonitor([p], accum_steps=4, accum_points='2,4',
        categories='attention', interval=1, output=str(tmp_path/'noise.csv'),
        adamw_optimizer=opt, adamw_output=str(tmp_path/'adamw.csv'))
    assert mon.begin(0)
    for i in range(4):
        p.grad += 1
        mon.capture(i+1)
    assert len(mon.finish()) == 2
    assert all(np.isfinite(r['adamw_update_norm']) for r in mon.adamw_rows)
