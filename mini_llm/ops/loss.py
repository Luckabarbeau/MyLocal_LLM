from ..backend import xp, scalar


def cross_entropy_forward(logits, targets):
    """
    Cross entropy loss with mixed precision support.
    
    Performs log-sum-exp in float32 for numerical stability,
    especially important for float16 inputs where underflow is common.
    """
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

    if logits.dtype == "float16":
        # One FP32 workspace for the complete stable softmax/loss path.
        # Converting FP16 -> FP32 necessarily allocates once; after that we
        # reuse the same array in-place instead of materializing separate
        # shifted, exp_logits, and probs_f32 arrays (each can be hundreds of
        # MiB at production batch/vocab sizes).
        work = flat_logits.astype("float32", copy=True)

        rows = xp.arange(n)
        max_logit = xp.max(work, axis=-1, keepdims=True)
        work -= max_logit

        # Save only the target shifted logits before overwriting work with exp.
        # loss_i = log(sum_j exp(z_j-max_z)) - (z_target-max_z)
        target_shifted = work[rows, flat_targets].copy()

        xp.exp(work, out=work)
        normalizer = xp.sum(work, axis=-1, keepdims=True)
        loss_f32 = xp.mean(xp.log(normalizer[:, 0]) - target_shifted)

        # Reuse work as the FP32 probability cache needed by backward.
        xp.divide(work, normalizer, out=work)

        cache = {
            "probs_f32": work,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
        }

        return scalar(loss_f32.astype(logits.dtype)), cache
    else:
        # Float32 path - standard computation
        max_logit = xp.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_logit
        exp_logits = xp.exp(shifted)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

        rows = xp.arange(n)
        target_probs = probs[rows, flat_targets]
        loss = -xp.mean(xp.log(target_probs + 1e-30))

        return scalar(loss), {
            "probs": probs,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
        }


def cross_entropy_backward(cache):
    """
    Backward pass for cross entropy loss.
    
    Returns gradient in the same dtype as logits.
    """
    probs_f32 = cache.get("probs_f32", None)
    
    if probs_f32 is not None:
        # Float16 path - compute in float32, then cast
        probs = probs_f32.astype("float16", copy=False)
    else:
        probs = cache["probs"].copy()
    
    targets = cache["targets"]
    n = cache["n"]
    original_shape = cache["original_shape"]
    
    rows = xp.arange(n)
    probs[rows, targets] -= 1.0
    probs /= float(n)
    
    return probs.reshape(original_shape)
