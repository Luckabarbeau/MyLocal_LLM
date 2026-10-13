"""0079A numerical tests; intentionally tiny and backend-agnostic."""

import numpy as np
import pytest

from mini_llm.backend import xp, RandomStream, asnumpy
from mini_llm.parameter import Parameter
from mini_llm.predictive_refinement.soft_embeddings import (
    SoftTokenExpectation, FuturePositionEmbeddings, stable_softmax, softmax_vjp,
)


def _np(x):
    return asnumpy(x)


def test_one_hot_and_uniform_expectations():
    W = Parameter(xp.asarray(np.arange(20).reshape(5, 4), dtype="float64"))
    op = SoftTokenExpectation(W)
    logits = xp.asarray([[[-1000.0, -1000.0, 0.0, -1000.0, -1000.0]]])
    s, _ = op.forward(logits)
    np.testing.assert_allclose(_np(s[0, 0]), _np(W.data[2]))
    s, _ = op.forward(xp.zeros((3, 2, 5), dtype="float64"))
    np.testing.assert_allclose(_np(s), np.broadcast_to(_np(xp.mean(W.data, axis=0)), (3, 2, 4)))


def test_softmax_shape_normalization_and_temperature():
    logits = xp.asarray([[[900.0, 901.0, 899.0], [1.0, -2.0, 4.0]]], dtype="float32")
    q = stable_softmax(logits, temperature=0.8)
    assert q.shape == logits.shape
    assert np.isfinite(_np(q)).all()
    np.testing.assert_allclose(_np(q.sum(axis=-1)), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        stable_softmax(logits, temperature=0.0)
    with pytest.raises(ValueError):
        softmax_vjp(q, q[:, 0])


def test_superposition_logit_and_weight_gradients_fd():
    rng = np.random.default_rng(1)
    W = Parameter(xp.asarray(rng.normal(size=(7, 5)), dtype="float64"))
    op = SoftTokenExpectation(W)
    logits = xp.asarray(rng.normal(size=(2, 3, 7)), dtype="float64")
    upstream = xp.asarray(rng.normal(size=(2, 3, 5)), dtype="float64")
    s, cache = op.forward(logits, temperature=0.7)
    dl = op.backward(upstream, cache)
    analytic_l = _np(dl).copy()
    analytic_w = _np(W.grad).copy()
    eps = 1e-6

    def objective():
        out, _ = op.forward(logits, temperature=0.7)
        return float(xp.sum(out * upstream))

    for i in np.ndindex(*logits.shape):
        logits[i] += eps
        plus = objective()
        logits[i] -= 2 * eps
        minus = objective()
        logits[i] += eps
        np.testing.assert_allclose(analytic_l[i], (plus - minus) / (2 * eps), rtol=2e-5, atol=2e-7)
    for i in np.ndindex(*W.shape):
        W.data[i] += eps
        plus = objective()
        W.data[i] -= 2 * eps
        minus = objective()
        W.data[i] += eps
        np.testing.assert_allclose(analytic_w[i], (plus - minus) / (2 * eps), rtol=2e-5, atol=2e-7)


def test_shared_parameter_gradients_accumulate_across_passes():
    W = Parameter(xp.asarray(np.random.default_rng(7).normal(size=(9, 4)), dtype="float64"))
    op = SoftTokenExpectation(W)
    logits = xp.asarray(np.random.default_rng(9).normal(size=(1, 3, 9)), dtype="float64")
    dy = xp.ones((1, 3, 4), dtype="float64")
    _, a = op.forward(logits)
    _, b = op.forward(logits + 0.1)
    op.backward(dy, b)
    single = W.grad.copy()
    op.backward(dy, a)
    np.testing.assert_allclose(_np(W.grad), 2 * _np(single), atol=1e-12)


def test_future_type_states_and_gradient_accumulation():
    states = FuturePositionEmbeddings(4, RandomStream(3), dtype="float64")
    W = Parameter(xp.asarray(np.arange(28).reshape(7, 4), dtype="float64"))
    unknown = states.unknown_block(2, 3)
    assert unknown.shape == (2, 3, 4)
    states.backward_unknown(xp.ones_like(unknown))
    np.testing.assert_array_equal(_np(states.unknown.grad), np.full(4, 6))
    soft = states.soft_block(xp.ones((2, 3, 4)))
    assert soft.shape == unknown.shape
    states.backward_soft(xp.ones_like(soft))
    np.testing.assert_array_equal(_np(states.soft.grad), np.full(4, 6))
    ids = xp.asarray([[1, 1, 2], [2, 5, 6]])
    fixed = states.fixed_block(ids, W)
    np.testing.assert_allclose(_np(fixed), _np(W.data[ids]))
    states.backward_fixed(xp.ones_like(fixed), ids, W)
    np.testing.assert_array_equal(_np(states.fixed.grad), np.full(4, 6))
    np.testing.assert_array_equal(_np(W.grad[1]), np.full(4, 2))
    np.testing.assert_array_equal(_np(W.grad[2]), np.full(4, 2))


def test_float32_large_vocab_without_jacobian():
    rng = np.random.default_rng(4)
    W = Parameter(xp.asarray(rng.normal(size=(257, 12)), dtype="float32"))
    op = SoftTokenExpectation(W)
    logits = xp.asarray(rng.normal(size=(2, 4, 257)), dtype="float32")
    y, cache = op.forward(logits)
    g = op.backward(xp.ones_like(y), cache)
    assert g.shape == logits.shape
    assert np.isfinite(_np(g)).all()
    assert np.isfinite(_np(W.grad)).all()


def test_no_mutation_of_normal_embedding():
    from mini_llm.ops.embedding import Embedding
    emb = Embedding(8, 4, 0.02, RandomStream(10), dtype="float64")
    op = SoftTokenExpectation(emb.W)
    ids = xp.asarray([[1, 2]])
    original, original_cache = emb.forward(ids)
    _, cache = op.forward(xp.zeros((1, 2, 8)))
    op.backward(xp.ones((1, 2, 4)), cache)
    emb.backward(xp.ones_like(original), original_cache)
    np.testing.assert_allclose(_np(emb.W.grad[1]), _np(emb.W.grad[2]))
    assert np.linalg.norm(_np(emb.W.grad[1])) > 0
