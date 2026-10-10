import numpy as np
from mini_llm.backend import xp
from mini_llm.ops.rmsnorm import RMSNorm


def test_rmsnorm_backward_direction():
    layer = RMSNorm(5, eps=1e-8, dtype="float64")
    x = xp.asarray(np.random.default_rng(1).normal(size=(2,3,5)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(2).normal(size=x.shape), dtype="float64")
    _, cache = layer.forward(x)
    dx = layer.backward(dy, cache)

    v = xp.asarray(np.random.default_rng(3).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        out, _ = layer.forward(z)
        return float(xp.sum(out*dy))

    fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 1e-7


def test_rmsnorm_output_dtype_hint_preserves_reference_semantics_without_fusion():
    """The optimization hint must not change the portable/reference path."""
    layer = RMSNorm(8, eps=1e-6, dtype="float32")
    x = xp.asarray(np.random.default_rng(7).normal(size=(2, 4, 8)), dtype="float32")
    y_default, cache_default = layer.forward(x)
    y_hint, cache_hint = layer.forward(x, output_dtype="float16")
    assert y_hint.dtype == y_default.dtype
    np.testing.assert_allclose(y_hint, y_default, rtol=0, atol=0)
    np.testing.assert_allclose(cache_hint["inv_rms"], cache_default["inv_rms"], rtol=0, atol=0)
