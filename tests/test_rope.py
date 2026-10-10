import numpy as np
from mini_llm.backend import xp, asnumpy
from mini_llm.ops.rope import rope_forward, rope_backward, build_rope_matrix


def test_rope_matrix_is_orthogonal():
    R = asnumpy(build_rope_matrix(7, 8, dtype="float64"))
    assert np.max(np.abs(R.T @ R - np.eye(8))) < 1e-12


def test_rope_backward_direction():
    x = xp.asarray(np.random.default_rng(1).normal(size=(1,4,2,8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(2).normal(size=x.shape), dtype="float64")
    _, cache = rope_forward(x)
    dx = rope_backward(dy, cache)

    v = xp.asarray(np.random.default_rng(3).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        out, _ = rope_forward(z)
        return float(xp.sum(out*dy))

    fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    assert abs(fd-an) < 1e-7


def test_rope_explicit_batch_position_ids_match_reference_matrices():
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(2, 4, 1, 8))
    x = xp.asarray(x_np, dtype="float64")
    positions_np = np.asarray([[0, 2, 5, 9], [1, 4, 6, 11]], dtype=np.int64)
    positions = xp.asarray(positions_np)

    y, cache = rope_forward(x, position_ids=positions)
    y_np = asnumpy(y)
    for batch in range(2):
        for token in range(4):
            R = asnumpy(
                build_rope_matrix(
                    int(positions_np[batch, token]), 8, dtype="float64"
                )
            )
            expected = R @ x_np[batch, token, 0]
            assert np.max(np.abs(y_np[batch, token, 0] - expected)) < 5e-7

    dy = xp.asarray(rng.normal(size=x_np.shape), dtype="float64")
    dx = rope_backward(dy, cache)
    direction = xp.asarray(rng.normal(size=x_np.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))
    eps = 1e-6

    def objective(z):
        out, _ = rope_forward(z, position_ids=positions)
        return float(xp.sum(out * dy))

    fd = (objective(x + eps * direction) - objective(x - eps * direction)) / (2 * eps)
    analytic = float(xp.sum(dx * direction))
    assert abs(fd - analytic) < 1e-7
