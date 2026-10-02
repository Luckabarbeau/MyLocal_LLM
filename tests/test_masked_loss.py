import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward


def test_masked_cross_entropy_ignores_unselected_positions():
    logits = xp.asarray(
        [[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]], dtype="float32"
    )
    targets = xp.asarray([[0, 0, 1]], dtype="int64")
    mask = xp.asarray([[1.0, 0.0, 0.0]], dtype="float32")

    loss, cache = cross_entropy_forward(logits, targets, loss_mask=mask)
    expected = -np.log(np.exp(3.0) / (np.exp(3.0) + 1.0))
    assert np.isclose(loss, expected, atol=1e-6)

    grad = cross_entropy_backward(cache)
    grad_np = np.asarray(grad)
    assert np.allclose(grad_np[:, 1:, :], 0.0)
    assert not np.allclose(grad_np[:, 0, :], 0.0)


def test_masked_cross_entropy_rejects_empty_mask():
    logits = xp.zeros((1, 2, 3), dtype="float32")
    targets = xp.zeros((1, 2), dtype="int64")
    mask = xp.zeros((1, 2), dtype="float32")
    try:
        cross_entropy_forward(logits, targets, loss_mask=mask)
    except ValueError as exc:
        assert "select at least one" in str(exc)
    else:
        raise AssertionError("Expected an empty assistant mask to be rejected")
