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
