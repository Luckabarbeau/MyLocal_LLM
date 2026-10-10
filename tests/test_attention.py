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


def test_packed_qkv_matches_reference_forward_backward(monkeypatch):
    attn = make_attention()
    rng = np.random.default_rng(5200)
    x = xp.asarray(rng.normal(size=(2, 5, 8)), dtype="float64")
    dy = xp.asarray(rng.normal(size=(2, 5, 8)), dtype="float64")

    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "0")
    y_ref, cache_ref = attn.forward(x)
    attn.zero_grad()
    dx_ref = attn.backward(dy, cache_ref)
    grads_ref = [p.grad.copy() for p in (attn.Wq, attn.Wk, attn.Wv)]

    attn.zero_grad()
    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "1")
    attn.refresh_compute_buffers()
    y_fast, cache_fast = attn.forward(x)
    dx_fast = attn.backward(dy, cache_fast)
    grads_fast = [p.grad.copy() for p in (attn.Wq, attn.Wk, attn.Wv)]

    np.testing.assert_allclose(np.asarray(y_fast), np.asarray(y_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(dx_fast), np.asarray(dx_ref), rtol=1e-12, atol=1e-12)
    for got, ref in zip(grads_fast, grads_ref):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_packed_qkv_refresh_tracks_parameter_updates(monkeypatch):
    attn = make_attention()
    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "1")
    attn.refresh_compute_buffers()
    before = attn._packed_qkv_weight.copy()

    attn.Wq.data[...] += 0.25
    attn.refresh_compute_buffers()
    q_end = attn._q_width

    assert bool(xp.all(attn._packed_qkv_weight[:, :q_end] == attn.Wq.data))
    assert bool(xp.any(attn._packed_qkv_weight[:, :q_end] != before[:, :q_end]))


def test_packed_qkv_parameter_grads_are_views_of_one_buffer(monkeypatch):
    attn = make_attention()
    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "1")
    attn.refresh_compute_buffers()

    packed = np.asarray(attn._packed_qkv_grad)
    q_grad = np.asarray(attn.Wq.grad)
    k_grad = np.asarray(attn.Wk.grad)
    v_grad = np.asarray(attn.Wv.grad)
    assert np.shares_memory(q_grad, packed)
    assert np.shares_memory(k_grad, packed)
    assert np.shares_memory(v_grad, packed)

    q_end = attn._q_width
    k_end = q_end + attn._kv_width
    attn.Wq.grad[...] = 1.0
    attn.Wk.grad[...] = 2.0
    attn.Wv.grad[...] = 3.0
    np.testing.assert_array_equal(packed[:, :q_end], 1.0)
    np.testing.assert_array_equal(packed[:, q_end:k_end], 2.0)
    np.testing.assert_array_equal(packed[:, k_end:], 3.0)

    attn.zero_grad()
    assert float(np.max(np.abs(packed))) == 0.0


def test_attention_explicit_sequential_positions_match_implicit_path():
    attn = make_attention()
    x = xp.asarray(np.random.default_rng(5801).normal(size=(2, 4, 8)), dtype="float64")
    implicit, _ = attn.forward(x)
    positions = xp.asarray(np.tile(np.arange(4, dtype=np.int64), (2, 1)))
    explicit, _ = attn.forward(x, position_ids=positions)
    np.testing.assert_allclose(
        np.asarray(explicit), np.asarray(implicit), rtol=1e-12, atol=1e-12
    )


def test_attention_gapped_position_ids_backward_direction():
    attn = make_attention()
    rng = np.random.default_rng(5802)
    x = xp.asarray(rng.normal(size=(1, 4, 8)), dtype="float64")
    dy = xp.asarray(rng.normal(size=(1, 4, 8)), dtype="float64")
    positions = xp.asarray([[2, 7, 11, 19]], dtype=xp.int64)
    _, cache = attn.forward(x, position_ids=positions)
    attn.zero_grad()
    dx = attn.backward(dy, cache)

    direction = xp.asarray(rng.normal(size=x.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))
    eps = 1e-6

    def objective(z):
        out, _ = attn.forward(z, position_ids=positions)
        return float(xp.sum(out * dy))

    fd = (objective(x + eps * direction) - objective(x - eps * direction)) / (2 * eps)
    analytic = float(xp.sum(dx * direction))
    rel = abs(fd - analytic) / (abs(fd) + abs(analytic) + 1e-12)
    assert rel < 3e-6
