import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.topk import (
    selected_topk_softmax_backward,
    selected_topk_softmax_forward,
)


def test_selected_topk_forward_matches_reference():
    logits = xp.asarray(
        [[0.2, 2.0, -1.0, 1.0], [3.0, 0.0, 2.0, 1.0]], dtype="float64"
    )
    weights, indices, _ = selected_topk_softmax_forward(logits, 2)

    np.testing.assert_array_equal(np.asarray(indices), np.array([[1, 3], [0, 2]]))
    selected = np.take_along_axis(np.asarray(logits), np.asarray(indices), axis=-1)
    selected -= selected.max(axis=-1, keepdims=True)
    ref = np.exp(selected)
    ref /= ref.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(weights), ref, rtol=1e-12, atol=1e-12)


def test_selected_topk_backward_directional_derivative_stable_selection():
    logits = xp.asarray(
        [[4.0, 1.5, -2.0, 0.25], [0.5, 3.0, 1.25, -1.0]], dtype="float64"
    )
    coeff = xp.asarray([[0.3, -0.7], [1.1, 0.2]], dtype="float64")
    direction = xp.asarray(
        [[0.2, -0.3, 0.1, 0.4], [-0.1, 0.25, 0.35, -0.2]], dtype="float64"
    )
    direction /= xp.sqrt(xp.sum(direction * direction))

    _, _, cache = selected_topk_softmax_forward(logits, 2)
    dlogits = selected_topk_softmax_backward(coeff, cache)
    analytical = float(xp.sum(dlogits * direction))

    def objective(z):
        weights, _, _ = selected_topk_softmax_forward(z, 2)
        return float(xp.sum(weights * coeff))

    eps = 1e-6
    finite_difference = (
        objective(logits + eps * direction) - objective(logits - eps * direction)
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        analytical, finite_difference, rtol=1e-6, atol=1e-8
    )
