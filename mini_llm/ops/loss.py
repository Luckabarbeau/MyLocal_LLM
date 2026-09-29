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
        # Convert to float32 for numerically stable computation
        logits_f32 = flat_logits.astype("float32", copy=False)
        
        # Log-sum-exp trick (stable in float32)
        max_logit = xp.max(logits_f32, axis=-1, keepdims=True)
        shifted = logits_f32 - max_logit
        exp_logits = xp.exp(shifted)
        logsumexp = max_logit.squeeze() + xp.log(xp.sum(exp_logits, axis=-1))
        
        # Get target logits
        rows = xp.arange(n)
        target_logits = logits_f32[rows, flat_targets]
        
        # Loss in float32
        loss_f32 = xp.mean(logsumexp - target_logits)
        
        # Compute probabilities for backward (in float32)
        probs_f32 = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)
        
        cache = {
            "probs_f32": probs_f32,
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
