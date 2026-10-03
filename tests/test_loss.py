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


def test_cross_entropy_fp16_workspace_matches_reference():
    rng = np.random.default_rng(123)
    logits_np = rng.normal(size=(3, 4, 17)).astype(np.float16)
    targets_np = rng.integers(0, 17, size=(3, 4), dtype=np.int64)

    logits = xp.asarray(logits_np)
    targets = xp.asarray(targets_np)
    loss, cache = cross_entropy_forward(logits, targets)
    grad = cross_entropy_backward(cache)

    z = logits_np.astype(np.float32).reshape(-1, 17)
    t = targets_np.reshape(-1)
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shift)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    expected_loss = np.mean(
        np.log(np.sum(exp_z, axis=-1)) - z_shift[np.arange(len(t)), t]
    )
    expected_grad = probs
    expected_grad[np.arange(len(t)), t] -= 1.0
    expected_grad /= len(t)
    expected_grad = expected_grad.reshape(logits_np.shape).astype(np.float32)

    # Loss and dL/dlogits stay in FP32 even when model logits are FP16.
    # This avoids throwing away information before dynamic/static loss scaling.
    assert abs(loss - float(expected_loss)) < 1e-5
    assert grad.dtype == xp.float32
    np.testing.assert_allclose(
        np.asarray(grad), expected_grad, rtol=2e-5, atol=2e-7
    )


def test_cross_entropy_fp16_cache_reuses_single_probability_workspace():
    logits = xp.asarray(np.zeros((2, 3, 11), dtype=np.float16))
    targets = xp.asarray(np.zeros((2, 3), dtype=np.int64))
    _, cache = cross_entropy_forward(logits, targets)
    assert "probs_f32" in cache
    assert cache["probs_f32"].dtype == "float32"
    assert "shifted" not in cache
    assert "exp_logits" not in cache


def test_cross_entropy_fp16_backward_preserves_tiny_non_target_gradients():
    # Match the production microbatch scale: 8 * 512 = 4096 tokens. With a
    # 16k vocabulary, typical non-target CE gradients are ~1.5e-8, which
    # underflow if dL/dlogits is cast to FP16 before trainer loss scaling.
    n = 4096
    vocab = 16384
    logits = xp.asarray(np.zeros((n, 1, vocab), dtype=np.float16))
    targets = xp.asarray(np.zeros((n, 1), dtype=np.int64))

    _, cache = cross_entropy_forward(logits, targets)
    grad = cross_entropy_backward(cache)

    grad_np = np.asarray(grad)
    assert grad.dtype == xp.float32
    assert np.isfinite(grad_np).all()

    # A representative non-target gradient must survive and remain positive.
    expected_non_target = 1.0 / (vocab * n)
    assert grad_np[0, 0, 1] > 0.0
    np.testing.assert_allclose(
        grad_np[0, 0, 1], expected_non_target, rtol=2e-5, atol=1e-12
    )

    # Softmax cross-entropy gradients should sum to approximately zero per row.
    row_sums = grad_np[:, 0, :].sum(axis=-1)
    np.testing.assert_allclose(row_sums, 0.0, atol=2e-10)


def test_bfloat16_cross_entropy_uses_compact_bf16_cache_and_gradient():
    """BF16 keeps stable FP32 math but stores full-vocab state in BF16."""
    import ml_dtypes
    from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward

    logits = xp.zeros((2, 4, 16), dtype=ml_dtypes.bfloat16)
    targets = xp.zeros((2, 4), dtype="int64")
    loss, cache = cross_entropy_forward(logits, targets)
    grad = cross_entropy_backward(cache)

    assert "probs_bf16" not in cache  # backward consumes/reuses the cache buffer
    assert str(np.dtype(grad.dtype)).lower() == "bfloat16"
    assert bool(xp.all(xp.isfinite(grad)))
    assert loss > 0.0

    grad_f32 = np.asarray(grad, dtype=np.float32)
    expected_target = (1.0 / 16.0 - 1.0) / 8.0
    expected_other = (1.0 / 16.0) / 8.0
    np.testing.assert_allclose(grad_f32[0, 0, 0], expected_target, rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(grad_f32[0, 0, 1], expected_other, rtol=1e-2, atol=1e-4)


def test_bfloat16_cross_entropy_cache_is_half_fp32_size_before_backward():
    import ml_dtypes

    logits = xp.zeros((4, 8, 257), dtype=ml_dtypes.bfloat16)
    targets = xp.zeros((4, 8), dtype="int64")
    _, cache = cross_entropy_forward(logits, targets)

    assert "probs_bf16" in cache
    probs = cache["probs_bf16"]
    assert str(np.dtype(probs.dtype)).lower() == "bfloat16"
    assert probs.nbytes == logits.size * 2
    assert probs.nbytes * 2 == logits.size * 4


def test_bfloat16_cross_entropy_can_reuse_logits_storage(monkeypatch):
    """0055A in-place BF16 CE should avoid a second full-vocab BF16 buffer."""
    import ml_dtypes

    monkeypatch.setenv("MINI_LLM_INPLACE_BF16_CE", "1")
    logits = xp.asarray(
        np.random.default_rng(140).normal(size=(2, 3, 17)).astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    targets = xp.asarray([[1, 2, 3], [4, 5, 6]], dtype="int64")
    original_shape = logits.shape

    _, cache = cross_entropy_forward(logits, targets)
    probs = cache["probs_bf16"]
    assert cache.get("inplace_bf16_ce") is True
    assert probs.size == logits.size
    assert bool(xp.shares_memory(probs, logits))

    grad = cross_entropy_backward(cache)
    assert bool(xp.shares_memory(grad, logits))
    assert bool(xp.all(xp.isfinite(grad)))
