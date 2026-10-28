"""0076A hybrid Muon/AdamW functional, numerical, and checkpoint tests."""
import sys
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.hybrid_muon import (
    HybridMuonAdamW, newton_schulz_orthogonalize, muon_eligible,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.checkpoint import save_checkpoint, load_checkpoint
from mini_llm.train_extended import ExtendedTrainer
import train_model


def param(name, dims, seed=11):
    rng = np.random.default_rng(seed)
    return Parameter(xp.asarray(rng.normal(size=dims).astype('float32')), name=name)


def np_value(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


@pytest.mark.parametrize('dims', [(5, 7), (7, 5), (8, 8)])
def test_newton_schulz_finite_and_shapes(dims):
    arr = xp.asarray(np.random.default_rng(23).normal(size=dims), dtype='float32')
    update = newton_schulz_orthogonalize(arr, steps=5)
    assert update.shape == arr.shape
    assert np.all(np.isfinite(np_value(update)))
    assert np.linalg.norm(np_value(update)) > 0
    zero = newton_schulz_orthogonalize(xp.zeros_like(arr))
    np.testing.assert_array_equal(np_value(zero), np.zeros(dims))


def test_pinned_host_buffer_allocation_padding(monkeypatch):
    """CuPy's pinned allocator may expose an allocation larger than requested.

    The original 0076A implementation reshaped the entire buffer and failed
    for the 768x768 matrices used by the 500M model when allocation was padded
    to 4 MiB. This test does not need a GPU: simulate the same buffer behavior.
    """
    from types import SimpleNamespace
    from mini_llm.optim.hybrid_muon import _MuonMatrices

    requested = []

    def padded_alloc(nbytes):
        requested.append(nbytes)
        padded = 1 << (nbytes - 1).bit_length()
        return bytearray(padded)

    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        cuda=SimpleNamespace(alloc_pinned_memory=padded_alloc)
    ))
    matrix = param('blocks.0.attention.Wq', (768, 768))
    state = object.__new__(_MuonMatrices)
    state._pinned_handles = []
    state.offload_host_bytes = 0

    moment = state._pinned(matrix, zeros=True)
    master = state._pinned(matrix, zeros=False)

    assert moment.shape == (768, 768)
    assert master.shape == (768, 768)
    assert moment.dtype == np.float32
    assert requested == [768 * 768 * 4] * 2
    assert all(len(buf) == 4 * 1024 * 1024 for buf in state._pinned_handles)
    assert state.offload_host_bytes == 2 * 768 * 768 * 4
    assert np.count_nonzero(moment) == 0
    master[...] = 1.25
    assert master[-1, -1] == 1.25


def test_parameter_grouping_real_decoder():
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=8, dtype='float32')
    opt = HybridMuonAdamW(model.optimization_parameters(), lr=2e-4,
                          beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
    eligible = [p.name for p in opt.muon.parameters]
    adam = [p.name for p in opt.adam.parameters]
    assert len(eligible) == 13
    assert 'blocks.0.attention.Wq' in eligible
    assert 'blocks.0.moe.experts.expert_1.W_gate' in eligible
    assert 'blocks.0.moe.router.W_router' in adam
    assert 'embedding.W' in adam and 'output_proj.W' in adam
    assert 'blocks.0.norm1.gamma' in adam
    assert len(set(eligible + adam)) == len(model.optimization_parameters())
    assert muon_eligible(param('blocks.1.attention.Wq', (8, 8)))
    assert not muon_eligible(param('blocks.1.moe.router.W_router', (8, 8)))
    assert not muon_eligible(param('output_proj.W', (8, 8)))


def test_adam_subset_equivalent_to_standalone_adamw():
    adam1 = param('embedding.W', (4, 6))
    muon1 = param('blocks.0.attention.Wq', (5, 6))
    adam2 = param('embedding.W', (4, 6))
    hybrid = HybridMuonAdamW([adam1, muon1], lr=1.5e-4,
                             beta1=.93, beta2=.97, eps=1e-8, weight_decay=.03)
    pure = AdamW([adam2], lr=1.5e-4, beta1=.93, beta2=.97,
                 eps=1e-8, weight_decay=.03)
    for i in range(4):
        rng = np.random.default_rng(i)
        g = xp.asarray(rng.normal(size=adam1.data.shape), dtype='float32')
        adam1.grad[...] = g
        adam2.grad[...] = g
        muon1.grad[...] = .01
        hybrid.step()
        pure.step()
        np.testing.assert_array_equal(np_value(adam1.data), np_value(adam2.data))
        np.testing.assert_array_equal(np_value(hybrid.adam.m[0]), np_value(pure.m[0]))
        np.testing.assert_array_equal(np_value(hybrid.adam.v[0]), np_value(pure.v[0]))


def test_named_subset_freezes_weights_moments_and_adam_age():
    old_muon = param('blocks.0.attention.Wq', (5, 7), seed=31)
    old_adam = param('embedding.W', (4, 6), seed=32)
    new_history = param('blocks.0.full_history.Wq', (5, 7), seed=33)
    ref_muon = param('blocks.0.attention.Wq', (5, 7), seed=31)
    ref_adam = param('embedding.W', (4, 6), seed=32)
    opt = HybridMuonAdamW(
        [old_muon, old_adam, new_history], lr=2e-4,
        beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1,
    )
    ref = HybridMuonAdamW(
        [ref_muon, ref_adam], lr=2e-4,
        beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1,
    )

    for value, reference in ((old_muon, ref_muon), (old_adam, ref_adam)):
        value.grad[...] = .01
        reference.grad[...] = .01
    new_history.grad[...] = .01
    opt.step()
    ref.step()

    frozen_weight = [np_value(p.data).copy() for p in (old_muon, old_adam)]
    frozen_muon_m = np_value(opt.muon.m[0]).copy()
    frozen_adam_m = np_value(opt.adam.m[0]).copy()
    frozen_adam_v = np_value(opt.adam.v[0]).copy()
    active_name = {new_history.name}
    for gradient in (.02, .03):
        old_muon.grad[...] = gradient
        old_adam.grad[...] = gradient
        new_history.grad[...] = gradient
        opt.step(active_parameter_names=active_name)

    np.testing.assert_array_equal(np_value(old_muon.data), frozen_weight[0])
    np.testing.assert_array_equal(np_value(old_adam.data), frozen_weight[1])
    np.testing.assert_array_equal(np_value(opt.muon.m[0]), frozen_muon_m)
    np.testing.assert_array_equal(np_value(opt.adam.m[0]), frozen_adam_m)
    np.testing.assert_array_equal(np_value(opt.adam.v[0]), frozen_adam_v)
    assert opt.adam.parameter_birth_steps[0] == 2
    assert opt.step_index == 3

    for value, reference in ((old_muon, ref_muon), (old_adam, ref_adam)):
        value.grad[...] = .04
        reference.grad[...] = .04
    new_history.grad[...] = .04
    opt.step()
    ref.step()
    np.testing.assert_array_equal(np_value(old_muon.data), np_value(ref_muon.data))
    np.testing.assert_array_equal(np_value(old_adam.data), np_value(ref_adam.data))
    np.testing.assert_array_equal(np_value(opt.muon.m[0]), np_value(ref.muon.m[0]))
    np.testing.assert_array_equal(np_value(opt.adam.m[0]), np_value(ref.adam.m[0]))
    np.testing.assert_array_equal(np_value(opt.adam.v[0]), np_value(ref.adam.v[0]))


def test_full_history_only_parameter_boundary():
    old = param('blocks.0.attention.Wq', (5, 7), seed=41)
    history = param('blocks.0.full_history.Wq', (5, 7), seed=42)
    trainer = object.__new__(ExtendedTrainer)
    trainer.optimization_parameters = [old, history]
    trainer.full_history_only_parameters = [history]
    trainer.full_history_only_start_step = 16_900
    trainer.full_history_only_until_step = 17_000

    trainer.step = 16_900
    assert trainer._optimizer_step_parameters() == [history]
    trainer.step = 16_999
    assert trainer._optimizer_step_parameters() == [history]
    trainer.step = 17_000
    assert trainer._optimizer_step_parameters() is trainer.optimization_parameters


def test_muon_momentum_and_nesterov_full_float32():
    p = param('blocks.0.attention.Wq', (5, 7))
    opt = HybridMuonAdamW([p], lr=1e-3, beta1=.9, beta2=.95,
                          eps=1e-8, weight_decay=.1, muon_momentum=.8)
    first = p.data.copy()
    g = xp.asarray(np.random.default_rng(123).normal(size=p.data.shape), dtype='float32')
    p.grad[...] = g
    opt.step()
    m = .2 * np_value(g)
    np.testing.assert_allclose(np_value(opt.muon.m[0]), m, rtol=1e-6, atol=1e-7)
    from mini_llm.optim.hybrid_muon import newton_schulz_orthogonalize
    direction = newton_schulz_orthogonalize(xp.asarray(.8*m + .2*np_value(g)))
    expected = np_value(first) * (1 - 1e-3*.1) - 1e-3 * np_value(direction) * .2 * np.sqrt(7)
    np.testing.assert_allclose(np_value(p.data), expected, rtol=2e-6, atol=1e-7)


def test_hybrid_save_reload_exact_continuation(tmp_path):
    p = [param('blocks.0.attention.Wq', (6, 6)), param('embedding.W', (6, 6), seed=12)]
    q = [param('blocks.0.attention.Wq', (6, 6)), param('embedding.W', (6, 6), seed=12)]
    a = HybridMuonAdamW(p, lr=2e-4, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
    b = HybridMuonAdamW(q, lr=2e-4, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
    for t in range(3):
        for x, y in zip(p, q):
            x.grad[...] = .01*(t+1)
            y.grad[...] = .01*(t+1)
        a.step()
        if t == 0:
            save_checkpoint(tmp_path, {x.name:x.data for x in p}, a.checkpoint_state(),
                            {'optimizer_kind':'hybrid-muon'})
            params, state, training = load_checkpoint(tmp_path)
            assert training['optimizer_kind'] == 'hybrid-muon'
            for x in q:
                x.data[...] = params[x.name]
            b.restore_checkpoint(state, from_hybrid=True)
        else:
            b.step()
    for x, y in zip(p, q):
        np.testing.assert_array_equal(np_value(x.data), np_value(y.data))
    np.testing.assert_array_equal(np_value(a.muon.m[0]), np_value(b.muon.m[0]))
    np.testing.assert_array_equal(np_value(a.adam.v[0]), np_value(b.adam.v[0]))


def test_legacy_adam_checkpoint_transition_preserves_adam_moments():
    p = [param('blocks.0.attention.Wq', (5, 5)), param('embedding.W', (6, 4))]
    old = AdamW(p, lr=3e-4)
    for x in p: x.grad[...] = .07
    old.step()
    q = [param('blocks.0.attention.Wq', (5, 5)), param('embedding.W', (6, 4))]
    new = HybridMuonAdamW(q, lr=3e-4, beta1=.9, beta2=.95,
                          eps=1e-8, weight_decay=.1)
    state = {
        'step': old.step_index,
        'master_weights': {x.name: old.master_weights[i].copy() for i, x in enumerate(p)},
        'm': {x.name: old.m[i].copy() for i,x in enumerate(p)},
        'v': {x.name: old.v[i].copy() for i,x in enumerate(p)},
    }
    counts = new.restore_checkpoint(state, from_hybrid=False)
    assert counts == (2, 1, 0)
    np.testing.assert_array_equal(np_value(new.muon.m[0]), np.zeros((5,5)))
    np.testing.assert_array_equal(np_value(new.adam.m[0]), np_value(old.m[1]))
    np.testing.assert_array_equal(np_value(new.adam.v[0]), np_value(old.v[1]))


def test_progressive_add_parameters_keeps_state():
    p = param('blocks.0.attention.Wq', (6, 6))
    opt = HybridMuonAdamW([p], lr=1e-3, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
    p.grad[...] = .3
    opt.step()
    stored = np_value(opt.muon.m[0]).copy()
    new = param('blocks.1.attention.Wq', (7, 5))
    adam = param('blocks.1.norm1.gamma', (7,))
    assert opt.add_parameters([new, adam]) == 2
    np.testing.assert_array_equal(np_value(opt.muon.m[0]), stored)
    assert opt.adam.parameter_birth_steps[0] == 1
    assert opt.step_index == 1


def test_training_small_model_saves_kind_and_state(tmp_path):
    shard = tmp_path / 'train.bin'
    (np.arange(4096, dtype='uint16') % 256).tofile(shard)
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=42, dtype='float32')
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)], batch_size=1, seq_length=16,
        total_steps=2, warmup_steps=0, optimizer_kind='hybrid-muon',
        checkpoint_dir=str(tmp_path / 'ckpt'))
    trainer.train_step()
    trainer.save()
    _, state, meta = load_checkpoint(tmp_path / 'ckpt')
    assert meta['optimizer_kind'] == 'hybrid-muon'
    assert len(state['muon_m']) == len(trainer.optimizer.muon.parameters)
    assert len(state['v']) == len(trainer.optimizer.adam.parameters)


def test_cli_options(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train_model.py', '--optimizer', 'hybrid-muon',
                                     '--muon-momentum', '0.96', '--muon-ns-steps', '4',
                                     '--full-history-only-steps', '100'])
    args = train_model.parse_args()
    assert args.optimizer == 'hybrid-muon'
    assert args.muon_momentum == .96 and args.muon_ns_steps == 4
    assert args.full_history_only_steps == 100
    monkeypatch.setattr(sys, 'argv', ['train_model.py'])
    assert train_model.parse_args().optimizer is None


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy and a CUDA GPU')
def test_gpu_forward_and_optimizer():
    a = param('blocks.0.attention.Wq', (128, 256))
    b = param('embedding.W', (256, 128))
    hybrid = HybridMuonAdamW([a,b], lr=3e-4, beta1=.9, beta2=.95,
                             eps=1e-8, weight_decay=.1)
    a.grad[...] = .02
    b.grad[...] = .01
    hybrid.step()
    assert np.isfinite(np_value(a.data)).all()
    assert np.isfinite(np_value(b.data)).all()
