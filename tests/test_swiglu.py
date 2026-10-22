import numpy as np
from mini_llm.backend import xp, RandomStream, BACKEND_NAME, asnumpy, synchronize
from mini_llm.ops.swiglu import SwiGLU
from mini_llm.ops.experts import _fused_swiglu_forward


def test_fused_fp16_swiglu_matches_reference(monkeypatch):
    if BACKEND_NAME != "cupy":
        return
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU", "1")
    rng = np.random.default_rng(4)
    g_np = rng.normal(size=(3, 257)).astype(np.float16)
    u_np = rng.normal(size=(3, 257)).astype(np.float16)
    actual = _fused_swiglu_forward(xp.asarray(g_np), xp.asarray(u_np))
    synchronize()
    assert actual is not None
    expected = (
        g_np.astype(np.float32)
        / (1.0 + np.exp(-g_np.astype(np.float32)))
        * u_np.astype(np.float32)
    ).astype(np.float16)
    np.testing.assert_allclose(asnumpy(actual), expected, rtol=2e-3, atol=2e-3)


def test_swiglu_backward_direction():
    rng = RandomStream(5)
    layer = SwiGLU(4, 7, 0.1, 0.05, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(6).normal(size=(2,3,4)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(7).normal(size=(2,3,4)), dtype="float64")
    _, cache = layer.forward(x)
    dx = layer.backward(dy, cache)

    v = xp.asarray(np.random.default_rng(8).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        out, _ = layer.forward(z)
        return float(xp.sum(out*dy))

    fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 1e-6
