import numpy as np
from mini_llm.backend import xp, RandomStream, BACKEND_NAME, asnumpy, synchronize
from mini_llm.ops.swiglu import SwiGLU
from mini_llm.ops.experts import _fused_swiglu_forward, _fused_swiglu_backward


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


def test_fused_bf16_swiglu_even_and_odd_match_scalar_reference(monkeypatch):
    if BACKEND_NAME != "cupy":
        return
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU", "1")
    monkeypatch.setenv("MINI_LLM_SWIGLU_VEC2", "1")
    rng = np.random.default_rng(89)
    for shape in ((4, 258), (3, 257)):
        g_np = rng.normal(size=shape).astype(np.float32)
        u_np = rng.normal(size=shape).astype(np.float32)
        g = xp.asarray(g_np).astype("bfloat16")
        u = xp.asarray(u_np).astype("bfloat16")
        actual = _fused_swiglu_forward(g, u)
        synchronize()
        assert actual is not None
        g_rounded = asnumpy(g).astype(np.float32)
        u_rounded = asnumpy(u).astype(np.float32)
        expected = g_rounded / (1.0 + np.exp(-g_rounded)) * u_rounded
        expected = xp.asarray(expected).astype("bfloat16")
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))


def test_fused_bf16_swiglu_backward_contiguous_and_packed(monkeypatch):
    if BACKEND_NAME != "cupy":
        return
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU", "1")
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU_STRICT", "1")
    random = np.random.default_rng(107)
    for packed in (False, True):
        values = xp.asarray(random.normal(size=(3, 514)), dtype="bfloat16")
        g, u = values[:, :257], values[:, 257:]
        if not packed:
            g, u = xp.ascontiguousarray(g), xp.ascontiguousarray(u)
        dh = xp.asarray(random.normal(size=g.shape), dtype="bfloat16")
        h = _fused_swiglu_forward(g, u)
        result = _fused_swiglu_backward(dh, g, u)
        assert h is not None and result is not None
        dg, du = result
        assert h.dtype == dg.dtype == du.dtype == g.dtype
        assert h.flags.c_contiguous and dg.flags.c_contiguous and du.flags.c_contiguous
        gf, uf, dhf = [asnumpy(a.astype("float32")) for a in (g, u, dh)]
        sigmoid = 1.0 / (1.0 + np.exp(-gf))
        # G -> SiLU(G), H = SiLU(G) * U, then reverse both product paths.
        expected = (
            (dhf * uf) * (sigmoid + gf * sigmoid * (1.0 - sigmoid)),
            dhf * (gf * sigmoid),
        )
        for actual, reference in zip((dg, du), expected):
            reference = asnumpy(xp.asarray(reference).astype("bfloat16").astype("float32"))
            np.testing.assert_allclose(asnumpy(actual.astype("float32")), reference,
                                       rtol=0.008, atol=0.008)


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
