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
    expected_grad = expected_grad.reshape(logits_np.shape).astype(np.float16)

    # The public FP16 loss intentionally returns an FP16-rounded scalar.
    assert abs(loss - float(np.float16(expected_loss))) < 1e-3
    np.testing.assert_allclose(
        np.asarray(grad), expected_grad, rtol=2e-3, atol=2e-4
    )


def test_cross_entropy_fp16_cache_reuses_single_probability_workspace():
    logits = xp.asarray(np.zeros((2, 3, 11), dtype=np.float16))
    targets = xp.asarray(np.zeros((2, 3), dtype=np.int64))
    _, cache = cross_entropy_forward(logits, targets)
    assert "probs_f32" in cache
    assert cache["probs_f32"].dtype == "float32"
    assert "shifted" not in cache
    assert "exp_logits" not in cache
