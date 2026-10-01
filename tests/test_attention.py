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


def test_fp16_attention_prescaling_avoids_score_overflow():
    """FP16 QK^T remains finite even when the unscaled score would overflow."""
    rng = RandomStream(20)
    attn = GQAAttention(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        input_std=0.01, output_std=0.01, rng=rng, dtype="float16"
    )

    # q_pre and k_pre remain finite (~800), but the original FP16 score path
    # would form O(1e6) logits and overflow before softmax.
    attn.Wq.data[...] = xp.asarray(1.0, dtype="float16")
    attn.Wk.data[...] = xp.asarray(1.0, dtype="float16")
    attn.Wv.data[...] = xp.asarray(1e-3, dtype="float16")
    attn.Wo.data[...] = xp.asarray(1e-3, dtype="float16")
    x = xp.full((1, 4, 8), 100.0, dtype="float16")

    y, cache = attn.forward(x)

    assert bool(xp.all(xp.isfinite(cache["probs"])))
    assert bool(xp.all(xp.isfinite(y)))
    row_sums = xp.sum(cache["probs"].astype("float32"), axis=-1)
    assert bool(xp.all(xp.abs(row_sums - 1.0) < 2e-3))


def test_scaled_softmax_matches_direct_softmax_in_safe_range():
    """Pre-scaled logits preserve the original softmax temperature."""
    from mini_llm.ops.attention import softmax_forward

    x = xp.asarray(
        [[[-1.5, 0.2, 1.1, 2.0], [0.3, -0.7, 0.1, 1.4]]],
        dtype="float32",
    )
    direct = softmax_forward(x, axis=-1)
    scaled = softmax_forward(x / 32.0, axis=-1, logit_multiplier=32.0)
    assert bool(xp.all(xp.abs(direct - scaled) < 1e-6))
