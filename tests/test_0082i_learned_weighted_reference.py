"""0082I: learn model-identity-aware global mixture while retaining residual."""
import json

import numpy as np
import pytest

from mini_llm.backend import asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from examples.train_probability_fusion_0082a import _evaluate
from examples.train_probability_fusion_streaming_0082f import _comparison_fields


def make_logits(rows=3, members=3, vocab=7, seed=12):
    return np.random.default_rng(seed).normal(size=(rows, members, vocab)).astype(np.float32)


def member_probs(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize('members', [1, 2, 4, 10])
def test_zero_reference_and_zero_residual_exact_arithmetic(members):
    z = make_logits(members=members)
    ids = list(range(members))[::-1]
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=z.shape[-1], max_models=10))
    weights = asnumpy(model.member_weights(ids))
    np.testing.assert_allclose(weights, np.ones(members) / members, atol=1e-7)
    np.testing.assert_allclose(asnumpy(model.forward(z, ids)), member_probs(z).mean(axis=1), atol=1e-6)


def test_weighted_skip_connection_is_not_selection_and_keeps_residual_unrestricted():
    z = make_logits(rows=2, members=3)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=z.shape[-1], max_models=10))
    scores = np.array([2.0, -1.0, 0.7], np.float32)
    model.reference_logits.data[[5, 0, 8]] = scores
    weight = np.exp(scores - scores.max()); weight /= weight.sum()
    expected = (member_probs(z) * weight[None, :, None]).sum(axis=1)
    np.testing.assert_allclose(asnumpy(model.forward(z, [5, 0, 8])), expected, atol=1e-6)
    # Correctly labeled pair reordering is invariant.
    np.testing.assert_allclose(asnumpy(model.forward(z[:, [2, 0, 1]], [8, 5, 0])), expected, atol=1e-6)
    # Swapping only identities changes the weighted reference.
    other = asnumpy(model.forward(z, [0, 5, 8]))
    assert abs(other - expected).max() > 1e-4
    model.output_bias.data[3] = 30.0
    assert asnumpy(model.forward(z, [5, 0, 8]))[:, 3].min() > 0.999


def test_analytic_reference_weight_gradients_finite_differences():
    z = make_logits(rows=4, members=3, vocab=9)
    targets = np.array([0, 2, 5, 4], np.int64)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=9, max_models=10))
    model.reference_logits.data[[8, 3, 5]] = [0.9, -0.4, 0.3]
    model.output_weight.data[:] = np.linspace(-0.08, 0.1, model.config.decoder_width)
    model.zero_grad()
    loss, arithmetic, weighted = model.loss_and_backward(z, [8, 3, 5], targets,
                                                          return_weighted=True)
    assert np.isfinite(loss) and np.isfinite(arithmetic) and np.isfinite(weighted)
    grads = asnumpy(model.reference_logits.grad)
    assert np.all(grads[[0, 1, 2, 4, 6, 7, 9]] == 0.0)
    assert abs(grads[[8, 3, 5]].sum()) < 1e-6
    for model_id in [8, 3, 5]:
        orig = float(asnumpy(model.reference_logits.data)[model_id])
        h = 1e-3
        def ce(score):
            model.reference_logits.data[model_id] = score
            probs = asnumpy(model.forward(z, [8, 3, 5]))
            return -np.log(probs[np.arange(len(targets)), targets]).mean()
        finite = (ce(orig + h) - ce(orig - h)) / (2 * h)
        model.reference_logits.data[model_id] = orig
        np.testing.assert_allclose(grads[model_id], finite, rtol=1e-2, atol=2e-4)


def test_weights_can_learn_to_trust_a_dominant_model():
    # A clearly better teacher: 90% on the correct token versus 10%.
    probs0 = np.array([.05, .9, .05], np.float32)
    probs1 = np.array([.45, .1, .45], np.float32)
    z = np.stack([np.log(probs0), np.log(probs1)], axis=0)[None, :, :]
    z = np.repeat(z, 4, axis=0)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=3, max_models=3))
    assert asnumpy(model.member_weights([0, 1]))[0] == pytest.approx(.5)
    for _ in range(30):
        model.zero_grad()
        model.loss_and_backward(z, [0, 1], [1]*4)
        # Optimize only the mixture; the residual stays zero for this test.
        model.reference_logits.data -= 0.3 * model.reference_logits.grad
    w = asnumpy(model.member_weights([0, 1]))
    assert w[0] > .8
    assert w[1] < .2


def test_legacy_checkpoint_can_be_loaded_with_equal_reference(tmp_path):
    z = make_logits(rows=2, members=2)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=10))
    state = model.state_dict()
    state.pop('reference_logits')
    payload = {**state, '__config_json__': np.asarray(json.dumps(model.config.__dict__))}
    path = tmp_path / 'legacy.npz'
    np.savez(path, **payload)
    reloaded = ProbabilityResidualCombiner.load(path)
    assert np.all(asnumpy(reloaded.reference_logits.data) == 0)
    np.testing.assert_allclose(asnumpy(model.forward(z, [0, 1])),
                               asnumpy(reloaded.forward(z, [0, 1])), atol=1e-7)


def test_reference_eval_and_gain_metrics():
    logits = make_logits(rows=11, members=2, vocab=9)
    targets = np.random.default_rng(9).integers(0, 9, size=11)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=9, max_models=10))
    model.reference_logits.data[5] = 1.8
    args = ([5, 2], [logits[:, 0], logits[:, 1]], targets)
    fusion, arithmetic, geometric, members, weighted = _evaluate(
        model, args, 3, return_weighted=True)
    assert weighted == pytest.approx(fusion, abs=1e-6)
    assert np.isfinite([arithmetic, geometric, *members]).all()
    fields = _comparison_fields(fusion, arithmetic, geometric, args[0], members,
                                weighted_ce=weighted)
    assert 'gain_weighted_best_member=' in fields
    assert 'gain_fusion_best_reference=' in fields
