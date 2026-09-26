import numpy as np
from mini_llm.backend import xp
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward


def test_cross_entropy_backward_direction():
    logits = xp.asarray(np.random.default_rng(21).normal(size=(2,3,5)), dtype="float64")
    targets = xp.asarray([[1,2,3],[0,4,1]], dtype="int64")
    _, cache = cross_entropy_forward(logits, targets)
    grad = cross_entropy_backward(cache)

    v = xp.asarray(np.random.default_rng(22).normal(size=logits.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    lp, _ = cross_entropy_forward(logits+eps*v, targets)
    lm, _ = cross_entropy_forward(logits-eps*v, targets)
    fd = (lp-lm)/(2*eps)
    an = float(xp.sum(grad*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 1e-7
