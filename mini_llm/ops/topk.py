"""Reusable selected-Top-K softmax primitives.

The selected indices are treated as fixed during backward.  This is the same
local-gradient convention used by the MoE router: gradients flow through the
softmax weights of the selected entries, but not through the discrete Top-K
identity itself.
"""

from ..backend import xp, is_low_precision_dtype


def selected_topk_softmax_forward(logits, k, output_dtype=None):
    """Select the largest ``k`` logits and normalize only those entries.

    Args:
        logits: Tensor with candidate dimension on the last axis.
        k: Number of candidates to select.
        output_dtype: Optional dtype for the returned weights.  Selection and
            softmax are promoted to FP32 for low-precision logits.

    Returns:
        weights: Selected softmax weights, shape ``logits.shape[:-1] + (k,)``.
        indices: Selected candidate indices with the same leading shape.
        cache: Minimal cache consumed by :func:`selected_topk_softmax_backward`.
    """
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")

    n_candidates = int(logits.shape[-1])
    k = int(k)
    if not (1 <= k <= n_candidates):
        raise ValueError("k must satisfy 1 <= k <= logits.shape[-1]")

    logits_work = (
        logits.astype("float32", copy=False)
        if is_low_precision_dtype(logits.dtype)
        else logits
    )

    # argsort is intentionally retained rather than argpartition so ties are
    # handled deterministically in exactly the same way as the original MoE
    # router implementation.
    indices = xp.argsort(-logits_work, axis=-1)[..., :k]
    selected_logits = xp.take_along_axis(logits_work, indices, axis=-1)

    selected_max = xp.max(selected_logits, axis=-1, keepdims=True)
    exp_selected = xp.exp(selected_logits - selected_max)
    weights_work = exp_selected / xp.sum(exp_selected, axis=-1, keepdims=True)

    if output_dtype is not None and is_low_precision_dtype(output_dtype):
        weights = weights_work.astype(output_dtype, copy=False)
    else:
        weights = weights_work

    cache = {
        "indices": indices,
        "weights_work": weights_work,
        "logits_shape": logits.shape,
    }
    return weights, indices, cache


def selected_topk_softmax_backward(dweights, cache):
    """Backward through selected-Top-K softmax with fixed selected indices.

    The selected-softmax Jacobian is applied exactly, then the selected-logit
    gradients are scattered back into a full tensor.  Unselected logits have
    zero local gradient.
    """
    indices = cache["indices"]
    weights_work = cache["weights_work"]
    logits_shape = tuple(cache["logits_shape"])

    dweights_work = (
        dweights.astype("float32", copy=False)
        if is_low_precision_dtype(dweights.dtype)
        else dweights
    )
    if dweights_work.shape != weights_work.shape:
        raise ValueError(
            f"dweights shape {dweights_work.shape} does not match selected "
            f"weights shape {weights_work.shape}"
        )

    correction = xp.sum(weights_work * dweights_work, axis=-1, keepdims=True)
    dselected = weights_work * (dweights_work - correction)

    n_candidates = logits_shape[-1]
    leading = int(dselected.size // dselected.shape[-1])
    k = int(dselected.shape[-1])

    dlogits_flat = xp.zeros((leading, n_candidates), dtype=dselected.dtype)
    rows = xp.arange(leading)[:, None]
    dlogits_flat[rows, indices.reshape(leading, k)] = dselected.reshape(leading, k)
    return dlogits_flat.reshape(logits_shape)
