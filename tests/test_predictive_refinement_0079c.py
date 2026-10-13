"""0079C full-gradient predictive-refinement training CPU reference tests."""

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)


def make_reference(k=3, aux=0.1):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=15, dtype="float64")
    return DenseRefinementTrainingReference(
        model,
        RefinementReferenceConfig(block_size=k, initial_loss_weight=aux),
        rng_seed=8,
    )


def sample(k=3):
    context = xp.asarray([[11, 15, 21], [3, 5, 8]], dtype="int64")
    targets = xp.asarray([[19, 13, 7], [22, 8, 9]], dtype="int64")
    return context, targets[:, :k]


def test_loss_shapes_gradients_and_actual_number_of_passes():
    ref = make_reference()
    ctx, targets = sample()
    metrics, tape = ref.forward(ctx, targets)
    assert metrics["full_model_passes"] == 4
    assert metrics["supervised_tokens"] == 6
    assert len(metrics["position_ce"]) == 3
    assert np.isfinite(float(metrics["commitment_ce"]))
    assert np.isfinite(float(metrics["initial_ce"]))
    assert np.isfinite(float(metrics["total_loss"]))
    ref.zero_grad()
    ref.backward(tape)
    for p in ref.parameters():
        assert np.isfinite(asnumpy(p.grad)).all(), p.name
    assert np.linalg.norm(asnumpy(ref.states.unknown.grad)) > 0
    assert np.linalg.norm(asnumpy(ref.states.soft.grad)) > 0
    assert np.linalg.norm(asnumpy(ref.states.fixed.grad)) > 0
    assert np.linalg.norm(asnumpy(ref.model.embedding.W.grad)) > 0


def test_no_early_target_leakage_and_legitimate_prefix_teacher_forcing():
    ref = make_reference()
    context, targets = sample()
    alternate = targets.copy()
    alternate[:, 0] = 99
    one, tape_one = ref.forward(context, targets)
    other, tape_other = ref.forward(context, alternate)
    # Initial predictions and first refined commitment must be identical.
    np.testing.assert_array_equal(
        asnumpy(tape_one["passes"][0].logits), asnumpy(tape_other["passes"][0].logits)
    )
    np.testing.assert_array_equal(
        asnumpy(tape_one["passes"][1].logits), asnumpy(tape_other["passes"][1].logits)
    )
    # Subsequent pass is legitimately influenced by the revealed prefix.
    assert not np.allclose(
        asnumpy(tape_one["passes"][2].logits), asnumpy(tape_other["passes"][2].logits)
    )
    assert float(one["initial_ce"]) != float(other["initial_ce"])


def test_padding_targets_do_not_contribute_to_loss_or_gradient():
    ref = make_reference(k=3)
    ctx, targets = sample()
    valid = xp.asarray([[True, True, False], [True, False, False]])
    targets_sentinel = targets.copy()
    targets_sentinel[~valid] = -100
    metrics, tape = ref.forward(ctx, targets_sentinel, valid_target_mask=valid)
    ref.zero_grad()
    ref.backward(tape)
    grads = [asnumpy(p.grad).copy() for p in ref.parameters()]
    different = targets.copy()
    different[~valid] = 245
    metrics2, tape2 = ref.forward(ctx, different, valid_target_mask=valid)
    ref.zero_grad()
    ref.backward(tape2)
    np.testing.assert_allclose(float(metrics["total_loss"]), float(metrics2["total_loss"]))
    for a, p in zip(grads, ref.parameters()):
        np.testing.assert_allclose(a, asnumpy(p.grad), atol=1e-11, rtol=1e-11)
    assert metrics["supervised_tokens"] == 3


def test_two_passes_gradient_reaches_initial_unknown_without_auxiliary_loss():
    ref = make_reference(k=2, aux=0.0)
    ctx, targets = sample(k=2)
    metrics, tape = ref.forward(ctx, targets)
    ref.zero_grad()
    ref.backward(tape)
    assert np.linalg.norm(asnumpy(ref.states.unknown.grad)) > 1e-12
    assert np.linalg.norm(asnumpy(ref.model.output_proj.W.grad)) > 1e-12


def test_full_unrolled_loss_gradient_finite_difference():
    ref = make_reference(k=2, aux=0.2)
    ctx, targets = sample(k=2)
    metrics, tape = ref.forward(ctx, targets)
    ref.zero_grad()
    ref.backward(tape)
    selected = [
        (ref.states.unknown, (0,)),
        (ref.states.soft, (0,)),
        (ref.model.embedding.W, (13, 1)),
        (ref.model.output_proj.W, (3, 13)),
        (ref.model.blocks[0].attention.Wq, (0, 0)),
    ]
    eps = 2e-5
    for parameter, index in selected:
        analytic = float(parameter.grad[index])
        parameter.data[index] += eps
        plus = float(ref.forward(ctx, targets)[0]["total_loss"])
        parameter.data[index] -= 2 * eps
        minus = float(ref.forward(ctx, targets)[0]["total_loss"])
        parameter.data[index] += eps
        numerical = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(
            analytic, numerical, rtol=1e-2, atol=1e-5,
            err_msg=f"unrolled BPTT mismatch for {parameter.name}{index}",
        )


def test_k1_remains_two_full_model_passes():
    ref = make_reference(k=1)
    ctx, target = sample(k=1)
    metrics, tape = ref.forward(ctx, target)
    assert metrics["full_model_passes"] == 2
    assert len(tape["passes"]) == 2
    ref.zero_grad()
    ref.backward(tape)
    assert np.linalg.norm(asnumpy(ref.states.soft.grad)) > 0


def test_invalid_mask_and_shapes():
    ref = make_reference()
    ctx, targets = sample()
    with pytest.raises(ValueError):
        ref.forward(ctx, targets, valid_target_mask=xp.asarray([[True, False, True], [True, True, True]]))
    with pytest.raises(ValueError):
        ref.forward(ctx, targets[:, :2])


def test_tiny_sequence_can_overfit_under_full_bptt():
    from mini_llm.optim.adamw import AdamW
    ref = make_reference(k=2)
    ctx = xp.asarray([[1, 2, 3], [1, 2, 3]])
    target = xp.asarray([[4, 5], [4, 5]])
    opt = AdamW(ref.parameters(), lr=0.01, weight_decay=0.0)
    initial = float(ref.forward(ctx, target)[0]["total_loss"])
    for _ in range(20):
        ref.zero_grad()
        _, tape = ref.forward(ctx, target)
        ref.backward(tape)
        opt.step()
    final = float(ref.forward(ctx, target)[0]["total_loss"])
    assert final < 0.8 * initial, (initial, final)
