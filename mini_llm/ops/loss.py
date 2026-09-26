from ..backend import xp, scalar


def cross_entropy_forward(logits, targets):
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

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
    probs = cache["probs"].copy()
    targets = cache["targets"]
    n = cache["n"]
    rows = xp.arange(n)
    probs[rows, targets] -= 1.0
    probs /= float(n)
    return probs.reshape(cache["original_shape"])
