"""0082H: size controls and decoupled teacher context semantics."""
import numpy as np
import pytest

from mini_llm.backend import asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def _raw():
    return np.random.default_rng(188).normal(size=(3, 2, 7)).astype(np.float32)


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_decoder_depth_preserves_zero_residual_at_initialization(depth):
    m = ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=3, decoder_layers=depth))
    raw = _raw()
    z = raw - raw.max(axis=-1, keepdims=True)
    p = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    np.testing.assert_allclose(asnumpy(m.forward(raw, [2, 0])), p.mean(axis=1), rtol=1e-6, atol=1e-7)
    assert sum(int(p.data.size) for p in m.parameters()) > 0


def test_increasing_capacity_increases_parameters_and_roundtrips(tmp_path):
    small = ProbabilityResidualCombiner(FusionConfig(vocab_size=9, max_models=10))
    large = ProbabilityResidualCombiner(FusionConfig(
        vocab_size=9, max_models=10, embed_dim=16, member_width=12,
        decoder_width=20, decoder_layers=3))
    assert sum(int(p.data.size) for p in large.parameters()) > sum(int(p.data.size) for p in small.parameters())
    large.output_weight.data[:] = 0.07
    before = asnumpy(large.forward(np.zeros((2, 3, 9), np.float32), [0, 2, 3]))
    ckpt = tmp_path / 'fusion.npz'
    large.save(ckpt)
    restored = ProbabilityResidualCombiner.load(ckpt)
    assert restored.config.decoder_layers == 3
    np.testing.assert_allclose(asnumpy(restored.forward(np.zeros((2, 3, 9), np.float32), [0, 2, 3])), before)


def test_deep_decoder_gradient_against_finite_difference():
    conf = FusionConfig(vocab_size=7, max_models=3, embed_dim=4,
                        member_width=5, decoder_width=6, decoder_layers=3, seed=11)
    m = ProbabilityResidualCombiner(conf)
    m.output_weight.data[:] = np.asarray([.05, -.08, .04, .03, -.02, .09], np.float32)
    z = _raw()
    target = np.asarray([3, 1, 4])
    m.zero_grad()
    loss, baseline = m.loss_and_backward(z, [0, 2], target)
    assert np.isfinite(loss) and np.isfinite(baseline)
    for key, idx in [('decoder_hidden_weight_1', (2, 4)),
                     ('decoder_hidden_bias_2', (3,)),
                     ('fusion_weight', (1, 2)),
                     ('token_embedding', (2, 1)),
                     ('output_weight', (3,))]:
        param = m.parameters_by_name[key]
        analytic = float(asnumpy(param.grad)[idx])
        old = float(asnumpy(param.data)[idx])
        eps = 2e-3
        def compute(v):
            param.data[idx] = v
            probs = asnumpy(m.forward(z, [0, 2]))
            return -float(np.log(probs[np.arange(len(target)), target]).mean())
        finite_diff = (compute(old + eps) - compute(old - eps)) / (2 * eps)
        param.data[idx] = old
        np.testing.assert_allclose(analytic, finite_diff, rtol=.07, atol=8e-5, err_msg=key)


def test_default_depth_preserves_0082g_parameter_names():
    m = ProbabilityResidualCombiner(FusionConfig(vocab_size=9))
    assert not any('decoder_hidden' in key for key in m.parameters_by_name)


def test_backward_compatible_streaming_config_dictionary():
    from examples.train_probability_fusion_streaming_0082f import FusionConfig as StreamingConfig
    old = dict(vocab_size=7, max_models=10, embed_dim=16, member_width=8,
               decoder_width=24, seed=82)
    assert StreamingConfig(**old).__dict__ == FusionConfig(vocab_size=7).__dict__
