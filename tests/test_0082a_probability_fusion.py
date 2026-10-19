"""0082A: arithmetic baseline, identity semantics, derivatives and CLI."""
import numpy as np
import pytest

from mini_llm.backend import asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def _logits(rows=2, members=3, vocab=9, seed=82):
    return np.random.default_rng(seed).normal(size=(rows, members, vocab)).astype(np.float32)


def _arithmetic(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    return np.mean(p, axis=1)


@pytest.mark.parametrize("members", (1, 2, 3, 5, 10))
def test_initialization_exactly_matches_arithmetic_mean(members):
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=13, max_models=12))
    raw = _logits(members=members, vocab=13)
    actual = asnumpy(c.forward(raw, np.arange(members)))
    np.testing.assert_allclose(actual, _arithmetic(raw), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=2e-7)
    assert np.isfinite(actual).all()


def test_math_feature_normalized_geometric_and_difference():
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=4))
    z = _logits(members=3, vocab=7)
    _, cache = c.forward(z, [0, 1, 2], return_cache=True)
    shifted = z - z.max(axis=-1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    geom_logits = np.mean(lp, axis=1)
    geom = np.exp(geom_logits-geom_logits.max(axis=-1, keepdims=True))
    geom /= geom.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.exp(asnumpy(cache['log_pg'])), geom, rtol=2e-6)
    np.testing.assert_allclose(asnumpy(cache['difference']),
                               np.log(geom)-np.log(_arithmetic(z)), atol=5e-7)


def test_order_independent_when_identity_follows_distribution():
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=11, max_models=6, seed=41))
    c.output_weight.data[:] = np.arange(c.config.decoder_width, dtype=np.float32) * 0.1
    z = _logits(members=3, vocab=11)
    a = asnumpy(c.forward(z, [4, 1, 2]))
    b = asnumpy(c.forward(z[:, [2, 0, 1]], [2, 4, 1]))
    np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)


def test_identity_changes_correction_when_not_reordered_with_distribution():
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=9, max_models=6, seed=43))
    c.output_weight.data[:] = np.linspace(-0.5, 0.5, c.config.decoder_width, dtype=np.float32)
    c.w_identity.data[:] = 2.0
    c.model_embedding.data[1] *= 8.0
    z = _logits(members=2)
    a = asnumpy(c.forward(z, [0, 1]))
    b = asnumpy(c.forward(z, [1, 0]))
    assert np.max(np.abs(a-b)) > 1e-7


def test_cross_token_correction_responds_to_alternatives():
    """Token 0's logits are fixed; elsewhere changes the moment and correction."""
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=4, seed=44))
    c.output_weight.data[:] = np.linspace(0.05, 0.2, c.config.decoder_width)
    c.w_context.data[:] = 1.0
    c.w_log_ratio.data[:] = 0.0
    c.w_entropy.data[:] = 0.0
    z = np.array([[[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]], np.float32)
    changed = z.copy()
    changed[0, :, 1] = 2
    changed[0, :, 2] = -2
    # Because logits are normalized, P(v=0) will change.  Check that the
    # cross-token feature itself changes beyond the direct token evidence.
    _, old = c.forward(z, [0, 1], return_cache=True)
    _, new = c.forward(changed, [0, 1], return_cache=True)
    assert np.max(np.abs(asnumpy(old['token_context']-new['token_context']))) > 1e-6


def test_unrestricted_residual_and_no_single_member_special_case():
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=5, max_models=3))
    z = _logits(rows=1, members=1, vocab=5)
    before = asnumpy(c.forward(z, [1]))
    c.output_bias.data[:] = np.asarray([0, 0, 0, 30, 0], np.float32)
    after = asnumpy(c.forward(z, [1]))
    assert after[0, 3] > 0.999999
    assert after[0, 3] > before[0, 3]


def test_loss_gradients_finite_difference_all_representative_paths():
    conf = FusionConfig(vocab_size=5, max_models=4, embed_dim=3,
                        member_width=3, decoder_width=4, seed=35)
    c = ProbabilityResidualCombiner(conf)
    c.output_weight.data[:] = np.array([0.15, -0.1, 0.08, 0.03], np.float32)
    z = _logits(rows=2, members=2, vocab=5)
    targets = np.array([1, 4])
    c.zero_grad()
    loss, _ = c.loss_and_backward(z, [2, 0], targets)
    assert np.isfinite(loss)
    for name, index in (
        ('output_weight', (1,)),
        ('output_bias', (1,)),
        ('fusion_weight', (2, 2)),
        ('fusion_bias', (1,)),
        ('w_log_ratio', (1,)),
        ('w_identity', (1,)),
        ('w_context', (1,)),
        ('w_entropy', (1,)),
        ('member_bias', (1,)),
        ('token_embedding', (2, 1)),
        ('model_embedding', (2, 1)),
    ):
        param = c.parameters_by_name[name]
        grad = float(asnumpy(param.grad)[index])
        x0 = float(asnumpy(param.data)[index])
        h = 3e-3
        def evaluate(value):
            param.data[index] = value
            p = asnumpy(c.forward(z, [2, 0]))
            return -float(np.log(p[np.arange(2), targets]).mean())
        numeric = (evaluate(x0+h)-evaluate(x0-h))/(2*h)
        param.data[index] = x0
        np.testing.assert_allclose(grad, numeric, rtol=4e-2, atol=6e-5,
                                   err_msg=f'gradient mismatch for {name}')


def test_checkpoint_roundtrip_and_bad_inputs(tmp_path):
    c = ProbabilityResidualCombiner(FusionConfig(vocab_size=6, max_models=3))
    c.output_bias.data[2] = 2.0
    z = _logits(members=2, vocab=6)
    path = tmp_path / 'combiner.npz'
    c.save(path)
    restored = ProbabilityResidualCombiner.load(path)
    np.testing.assert_array_equal(asnumpy(c.forward(z, [0, 2])),
                                  asnumpy(restored.forward(z, [0, 2])))
    with pytest.raises(ValueError, match='unique'):
        c.forward(z, [0, 0])
    with pytest.raises(ValueError, match='range'):
        c.forward(z, [0, 4])
    with pytest.raises(ValueError, match='finite'):
        z[0, 0, 0] = np.nan
        c.forward(z, [0, 2])


def test_small_training_script_with_memmapped_logits(tmp_path):
    from examples.train_probability_fusion_0082a import main
    rng = np.random.default_rng(82)
    for mode in ('train', 'val'):
        n = 12 if mode == 'train' else 6
        for mid in (0, 3):
            np.save(tmp_path / f'{mode}_{mid}.npy',
                    rng.normal(size=(n, 7)).astype(np.float32))
        np.save(tmp_path / f'{mode}_targets.npy', rng.integers(0, 7, size=n))
    out = tmp_path / 'learned.npz'
    main(['--member', f'0:{tmp_path}/train_0.npy',
          '--member', f'3:{tmp_path}/train_3.npy',
          '--targets', str(tmp_path / 'train_targets.npy'),
          '--val-member', f'3:{tmp_path}/val_3.npy',
          '--val-member', f'0:{tmp_path}/val_0.npy',
          '--val-targets', str(tmp_path / 'val_targets.npy'),
          '--max-models', '5', '--steps', '3', '--batch-rows', '2',
          '--eval-interval', '2', '--log-interval', '1', '--output', str(out)])
    assert out.is_file()
    assert ProbabilityResidualCombiner.load(out).config.vocab_size == 7


def test_exporter_real_checkpoint_alignment(tmp_path):
    import dataclasses
    import json
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from examples.export_probability_fusion_logits_0082a import main as export
    config = ModelConfig(tokenizer_vocab_size=11, context_length=4, n_layers=1,
                         d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
                         n_experts=1, top_k=1, d_ff=16, ffn_mode='dense')
    for i in (0, 1):
        ckpt = tmp_path / f'ckpt{i}'
        ckpt.mkdir()
        (ckpt / 'config.json').write_text(json.dumps(dataclasses.asdict(config)))
        (ckpt / 'tokenizer.json').write_text('{"tokenizer-test": true}')
        model = DecoderLanguageModel(config, rng_seed=30 + i, dtype='float32')
        save_checkpoint(ckpt, {p.name: p.data for p in model.parameters()})
    windows = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], dtype=np.int64)
    np.save(tmp_path / 'windows.npy', windows)
    export(['--member', f'0:{tmp_path}/ckpt0',
            '--member', f'1:{tmp_path}/ckpt1',
            '--windows', str(tmp_path / 'windows.npy'),
            '--output-dir', str(tmp_path / 'out'),
            '--context-length', '4'])
    for idx in (0, 1):
        exported = np.load(tmp_path / 'out' / f'model_{idx}_logits.npy')
        assert exported.shape == (8, 11)
        assert np.isfinite(exported).all()
    np.testing.assert_array_equal(np.load(tmp_path / 'out/targets.npy'),
                                  windows[:, 1:].reshape(-1))
    manifest = json.loads((tmp_path / 'out/metadata.json').read_text())
    assert manifest['rows'] == 8
    assert manifest['models'][0]['id'] == 0
