import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.attention import GQAAttention


def make_attention():
    rng = RandomStream(10)
    return GQAAttention(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        input_std=0.08, output_std=0.04, rng=rng, dtype="float64"
    )


def test_attention_is_causal():
    attn = make_attention()
    x = xp.asarray(np.random.default_rng(11).normal(size=(1,4,8)), dtype="float64")
    _, cache = attn.forward(x)
    p = cache["probs"]
    for t in range(4):
        if t + 1 < 4:
            assert float(xp.max(xp.abs(p[:,:,t,t+1:]))) == 0.0


def test_attention_input_backward_direction():
    attn = make_attention()
    x = xp.asarray(np.random.default_rng(12).normal(size=(1,3,8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(13).normal(size=(1,3,8)), dtype="float64")
    _, cache = attn.forward(x)
    attn.zero_grad()
    dx = attn.backward(dy, cache)

    v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        out, _ = attn.forward(z)
        return float(xp.sum(out*dy))

    fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 2e-6


def test_attention_wq_backward_direction():
    attn = make_attention()
    x = xp.asarray(np.random.default_rng(15).normal(size=(1,3,8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(16).normal(size=(1,3,8)), dtype="float64")
    _, cache = attn.forward(x)
    attn.zero_grad()
    attn.backward(dy, cache)
    analytic = attn.Wq.grad.copy()

    v = xp.asarray(np.random.default_rng(17).normal(size=attn.Wq.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6
    original = attn.Wq.data.copy()

    attn.Wq.data[...] = original + eps*v
    y_plus, _ = attn.forward(x)
    f_plus = float(xp.sum(y_plus*dy))
    attn.Wq.data[...] = original - eps*v
    y_minus, _ = attn.forward(x)
    f_minus = float(xp.sum(y_minus*dy))
    attn.Wq.data[...] = original

    fd = (f_plus-f_minus)/(2*eps)
    an = float(xp.sum(analytic*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 2e-6
