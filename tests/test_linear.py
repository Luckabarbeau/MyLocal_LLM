import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.linear import Linear


def test_linear_backward_direction():
    rng = RandomStream(1)
    layer = Linear(4, 3, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(2).normal(size=(2, 4)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(3).normal(size=(2, 3)), dtype="float64")

    _, cache = layer.forward(x)
    dx = layer.backward(dy, cache)

    v = xp.asarray(np.random.default_rng(4).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        return float(xp.sum((z @ layer.W.data) * dy))

    fd = (objective(x + eps*v) - objective(x - eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    assert abs(fd-an) < 1e-7
