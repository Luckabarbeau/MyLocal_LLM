"""Full-resolution attention over explicitly selected key/value positions.

This module contains no routing policy.  It consumes a ``KeySelectionPlan`` and
performs ordinary scaled dot-product attention over the exact K/V tokens named
by that plan.  It therefore serves local/dilated/global/manual/learned sparse
patterns equally; only the key-visibility provider changes.
"""

import math

from ..backend import xp, is_bfloat16_dtype, is_low_precision_dtype
from .attention_selection import KeySelectionPlan


def _masked_softmax_forward(scores, valid_mask, logit_multiplier=1.0):
    """Stable softmax that returns exactly zero for rows with no valid keys."""
    work = (
        scores.astype("float32", copy=False)
        if is_low_precision_dtype(scores.dtype)
        else scores
    )
    valid_mask = valid_mask.astype(bool, copy=False)
    has_valid = xp.any(valid_mask, axis=-1, keepdims=True)
    neg_inf = xp.asarray(-xp.inf, dtype=work.dtype)
    masked = xp.where(valid_mask, work, neg_inf)
    row_max = xp.max(masked, axis=-1, keepdims=True)
    row_max = xp.where(has_valid, row_max, 0.0)
    shifted = work - row_max
    if logit_multiplier != 1.0:
        shifted = shifted * logit_multiplier
    exp_scores = xp.where(valid_mask, xp.exp(shifted), 0.0)
    denom = xp.sum(exp_scores, axis=-1, keepdims=True)
    safe_denom = xp.where(has_valid, denom, 1.0)
    return exp_scores / safe_denom


def _softmax_backward(dprobs, probs):
    correction = xp.sum(dprobs * probs, axis=-1, keepdims=True)
    return probs * (dprobs - correction)


def _default_kv_head_indices(n_q_heads, n_kv_heads):
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            "n_q_heads must be divisible by n_kv_heads when kv_head_indices "
            "is not supplied"
        )
    group_size = n_q_heads // n_kv_heads
    return xp.arange(n_q_heads, dtype=xp.int64) // group_size


def indexed_attention_forward(
    q,
    k,
    v,
    plan,
    kv_head_indices=None,
    scale=None,
    return_cache=True,
):
    """Attend over exact selected keys while retaining native GQA K/V storage.

    Args:
        q: ``(B,Tq,Hq,Dh)`` query tensor, normally after RoPE.
        k, v: ``(B,Tk,Hkv,Dh)`` native grouped-query K/V tensors.
        plan: :class:`KeySelectionPlan` containing exact token positions.
        kv_head_indices: Optional ``(Hq,)`` mapping from each query head to its
            source KV head.  This is important for heterogeneous subsets of
            query heads; when omitted, standard contiguous GQA grouping is used.
        scale: Optional score scale.  Defaults to ``1/sqrt(Dh)``.

    Returns:
        context: ``(B,Tq,Hq,Dh)``.
        cache: Explicit backward cache when ``return_cache`` is true.
    """
    if not isinstance(plan, KeySelectionPlan):
        raise TypeError("plan must be a KeySelectionPlan")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or k_d_head != d_head:
        raise ValueError("q/k batch or head width mismatch")

    plan.validate(batch, n_q_heads, query_length, key_length)
    if kv_head_indices is None:
        kv_head_indices = _default_kv_head_indices(n_q_heads, n_kv_heads)
    else:
        kv_head_indices = xp.asarray(kv_head_indices, dtype=xp.int64)
        if kv_head_indices.shape != (n_q_heads,):
            raise ValueError("kv_head_indices must have shape (n_q_heads,)")
        if bool(xp.any(kv_head_indices < 0)) or bool(
            xp.any(kv_head_indices >= n_kv_heads)
        ):
            raise ValueError("kv_head_indices contains an invalid KV head")

    key_indices = plan.key_indices
    valid_mask = plan.valid_mask
    if key_indices.shape[1] == 1 and n_q_heads != 1:
        key_indices = xp.broadcast_to(
            key_indices, (batch, n_q_heads, query_length, key_indices.shape[-1])
        )
        valid_mask = xp.broadcast_to(valid_mask, key_indices.shape)
        logit_bias = (
            None
            if plan.logit_bias is None
            else xp.broadcast_to(plan.logit_bias, key_indices.shape)
        )
    else:
        logit_bias = plan.logit_bias

    q_heads = q.transpose(0, 2, 1, 3)  # [B,Hq,Tq,Dh]
    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]
    k_selected = k[batch_ids, key_indices, kv_ids, :]
    v_selected = v[batch_ids, key_indices, kv_ids, :]

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)

    # Match the dense attention numerical policy.  FP16 pre-scales the score
    # product to avoid overflow; BF16 indexed products use FP32 because CuPy's
    # generic BF16 batched operations are not consistently supported.
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0
    bf16_attention = is_bfloat16_dtype(q.dtype)
    if bf16_attention:
        q_score = q_heads.astype("float32") * scale
        k_score = k_selected.astype("float32")
    else:
        q_score = q_heads * (scale * score_prescale)
        k_score = k_selected

    scores = xp.sum(q_score[..., None, :] * k_score, axis=-1)
    if logit_bias is not None:
        # ``scores`` is pre-scaled only on FP16.  Bias must be pre-scaled by
        # the same amount because softmax restores the original temperature.
        scores = scores + logit_bias.astype(scores.dtype, copy=False) * score_prescale

    probs = _masked_softmax_forward(
        scores, valid_mask, logit_multiplier=(1.0 / score_prescale)
    )

    if bf16_attention:
        probs_compute = probs
        v_compute = v_selected.astype("float32")
    else:
        probs_compute = (
            probs.astype(q.dtype, copy=False)
            if is_low_precision_dtype(q.dtype)
            else probs
        )
        v_compute = v_selected

    context_heads = xp.sum(probs_compute[..., None] * v_compute, axis=-2)
    context = context_heads.transpose(0, 2, 1, 3)

    if not return_cache:
        return context

    cache = {
        "q": q,
        "k": k,
        "v": v,
        "key_indices": key_indices,
        "valid_mask": valid_mask,
        "kv_head_indices": kv_head_indices,
        "probs": probs,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "has_logit_bias": logit_bias is not None,
    }
    return context, cache


def indexed_attention_backward(dcontext, cache):
    """Explicit backward for :func:`indexed_attention_forward`.

    Repeated token selections and shared GQA K/V heads are accumulated with
    scatter-add, so overlapping/manual retrieval plans are handled correctly.

    Returns:
        ``dq, dk, dv, dlogit_bias``.  ``dlogit_bias`` is ``None`` when no bias
        was supplied in the forward plan.  It has the *expanded* per-query-head
        shape used by the kernel; callers that broadcast one bias across heads
        should sum that gradient over the broadcasted head dimension.
    """
    q, k, v = cache["q"], cache["k"], cache["v"]
    key_indices = cache["key_indices"]
    valid_mask = cache["valid_mask"]
    kv_head_indices = cache["kv_head_indices"]
    probs = cache["probs"]
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]

    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    batch, query_length, n_q_heads, d_head = q.shape
    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]
    k_selected = k[batch_ids, key_indices, kv_ids, :]
    v_selected = v[batch_ids, key_indices, kv_ids, :]

    if bf16_attention:
        dcontext_compute = dcontext_heads.astype("float32")
        q_compute = q_heads.astype("float32")
        k_compute = k_selected.astype("float32")
        v_compute = v_selected.astype("float32")
        probs_compute = probs
    else:
        dcontext_compute = dcontext_heads
        q_compute = q_heads
        k_compute = k_selected
        v_compute = v_selected
        probs_compute = (
            probs.astype(q.dtype, copy=False)
            if is_low_precision_dtype(q.dtype)
            else probs
        )

    dprobs = xp.sum(dcontext_compute[..., None, :] * v_compute, axis=-1)
    dv_selected = probs_compute[..., None] * dcontext_compute[..., None, :]

    # probs is zero at every invalid key, hence the softmax Jacobian also gives
    # exactly zero score gradient there (including completely empty rows).
    dscores = _softmax_backward(dprobs, probs)
    dscores = xp.where(valid_mask, dscores, 0.0)

    if bf16_attention:
        dscores_compute = dscores.astype("float32", copy=False)
    else:
        dscores_compute = (
            dscores.astype(q.dtype, copy=False)
            if is_low_precision_dtype(q.dtype)
            else dscores
        )

    dq_heads = xp.sum(
        dscores_compute[..., None] * k_compute, axis=-2
    ) * scale
    dk_selected = dscores_compute[..., None] * q_compute[..., None, :] * scale

    # Scatter selected K/V gradients back to native [B,Tk,Hkv,Dh] storage.
    dk = xp.zeros(k.shape, dtype=dk_selected.dtype)
    dv = xp.zeros(v.shape, dtype=dv_selected.dtype)
    scatter_batch = xp.broadcast_to(batch_ids, key_indices.shape)
    scatter_kv = xp.broadcast_to(kv_ids, key_indices.shape)
    xp.add.at(dk, (scatter_batch, key_indices, scatter_kv), dk_selected)
    xp.add.at(dv, (scatter_batch, key_indices, scatter_kv), dv_selected)

    dq = dq_heads.transpose(0, 2, 1, 3)
    dlogit_bias = dscores if cache["has_logit_bias"] else None
    return dq, dk, dv, dlogit_bias
