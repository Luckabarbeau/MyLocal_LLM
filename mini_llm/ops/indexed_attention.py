"""Full-resolution attention over explicitly selected key/value positions.

This module contains no routing policy.  It consumes a ``KeySelectionPlan`` and
performs ordinary scaled dot-product attention over the exact K/V tokens named
by that plan.  It therefore serves local/dilated/global/manual/learned sparse
patterns equally; only the key-visibility provider changes.
"""

import math
import os

from ..backend import xp, is_bfloat16_dtype, is_low_precision_dtype
from .attention_selection import KeySelectionPlan


_DEFAULT_QUERY_CHUNK_SIZE = 128


def _resolve_query_chunk_size(query_length, query_chunk_size=None):
    """Resolve bounded query chunking for the indexed attention hot path.

    ``KeySelectionPlan`` can expose thousands of keys per query. Gathering all
    selected K/V vectors for a full long sequence would create a
    ``[B,H,T,K,D]`` tensor, which is far larger than the actual attention
    probability matrix. Chunking only the query axis keeps the mathematical
    result unchanged while bounding those transient gather/product buffers.

    The environment override is intentionally a runtime knob so GPU-memory
    experiments do not require changing model/checkpoint configuration.
    """
    if query_chunk_size is None:
        raw = os.environ.get("MINI_LLM_INDEXED_ATTN_QUERY_CHUNK")
        query_chunk_size = (
            _DEFAULT_QUERY_CHUNK_SIZE if raw is None else int(raw)
        )
    query_chunk_size = int(query_chunk_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    return min(int(query_length), query_chunk_size)


def _query_chunks(query_length, query_chunk_size):
    for start in range(0, int(query_length), int(query_chunk_size)):
        yield start, min(start + int(query_chunk_size), int(query_length))


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
    query_chunk_size=None,
):
    """Attend over exact selected keys while retaining native GQA K/V storage.

    The query dimension is evaluated in bounded chunks.  This is essential for
    long sparse contexts: materializing all selected K/V vectors at once would
    require ``O(B * Hq * Tq * Kvisible * Dh)`` temporary memory even though
    the persistent attention probabilities only require
    ``O(B * Hq * Tq * Kvisible)``.  Chunking changes neither the selected keys
    nor the softmax mathematics.

    Args:
        q: ``(B,Tq,Hq,Dh)`` query tensor, normally after RoPE.
        k, v: ``(B,Tk,Hkv,Dh)`` native grouped-query K/V tensors.
        plan: :class:`KeySelectionPlan` containing exact token positions.
        kv_head_indices: Optional ``(Hq,)`` mapping from each query head to its
            source KV head.  This is important for heterogeneous subsets of
            query heads; when omitted, standard contiguous GQA grouping is used.
        scale: Optional score scale.  Defaults to ``1/sqrt(Dh)``.
        query_chunk_size: Maximum number of query positions processed at once.
            Defaults to 128 and can be overridden at runtime with
            ``MINI_LLM_INDEXED_ATTN_QUERY_CHUNK``.

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

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)
    query_chunk_size = _resolve_query_chunk_size(
        query_length, query_chunk_size
    )

    q_heads = q.transpose(0, 2, 1, 3)  # [B,Hq,Tq,Dh]
    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]

    # Match the dense attention numerical policy. FP16 pre-scales the score
    # product to avoid overflow; BF16 indexed products run in FP32 because
    # generic CuPy BF16 batched operations are not consistently supported.
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0
    bf16_attention = is_bfloat16_dtype(q.dtype)
    probs_dtype = (
        xp.float32 if is_low_precision_dtype(q.dtype) else q.dtype
    )
    context_dtype = xp.float32 if bf16_attention else q.dtype
    probs = (
        xp.empty(key_indices.shape, dtype=probs_dtype) if return_cache else None
    )
    context_heads = xp.empty(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )

    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        chunk_indices = key_indices[:, :, q_start:q_end, :]
        chunk_valid = valid_mask[:, :, q_start:q_end, :]
        q_chunk = q_heads[:, :, q_start:q_end, :]

        # Gather only this query chunk.  For the 4k medium preset this changes
        # the largest local-head BF16 gather from [1,4,4096,1024,64] to
        # [1,4,chunk,1024,64].
        k_selected = k[batch_ids, chunk_indices, kv_ids, :]
        if bf16_attention:
            q_score = q_chunk.astype("float32") * scale
            k_score = k_selected.astype("float32")
        else:
            q_score = q_chunk * (scale * score_prescale)
            k_score = k_selected

        # Batched [1,D] @ [D,K] avoids materializing the old
        # q[...,None,:] * k_selected [B,H,Q,K,D] product.
        scores = xp.matmul(
            q_score[..., None, :], k_score.swapaxes(-1, -2)
        )[..., 0, :]
        if logit_bias is not None:
            chunk_bias = logit_bias[:, :, q_start:q_end, :]
            scores = scores + (
                chunk_bias.astype(scores.dtype, copy=False) * score_prescale
            )

        probs_chunk = _masked_softmax_forward(
            scores, chunk_valid, logit_multiplier=(1.0 / score_prescale)
        )
        if return_cache:
            probs[:, :, q_start:q_end, :] = probs_chunk

        # K is no longer needed for the forward chunk. Release references before
        # gathering V so peak memory is bounded by one selected-value tensor.
        del scores, k_score, k_selected

        v_selected = v[batch_ids, chunk_indices, kv_ids, :]
        if bf16_attention:
            probs_compute = probs_chunk
            v_compute = v_selected.astype("float32")
        else:
            probs_compute = (
                probs_chunk.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_chunk
            )
            v_compute = v_selected

        context_heads[:, :, q_start:q_end, :] = xp.matmul(
            probs_compute[..., None, :], v_compute
        )[..., 0, :]

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
        "query_chunk_size": query_chunk_size,
    }
    return context, cache

def indexed_attention_backward(dcontext, cache):
    """Explicit chunked backward for :func:`indexed_attention_forward`.

    Repeated token selections and shared GQA K/V heads are accumulated with
    scatter-add.  As in forward, selected K/V tensors are reconstructed only
    for a bounded query chunk so backward does not recreate a full
    ``[B,H,T,K,D]`` temporary.

    Returns:
        ``dq, dk, dv, dlogit_bias``.  ``dlogit_bias`` is ``None`` when no bias
        was supplied in forward.  It has the expanded per-query-head shape used
        by the kernel; callers that broadcast one bias across heads should sum
        that gradient over the broadcasted head dimension.
    """
    q, k, v = cache["q"], cache["k"], cache["v"]
    key_indices = cache["key_indices"]
    valid_mask = cache["valid_mask"]
    kv_head_indices = cache["kv_head_indices"]
    probs = cache["probs"]
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    query_chunk_size = cache.get(
        "query_chunk_size", _resolve_query_chunk_size(q.shape[1])
    )

    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    batch, query_length, n_q_heads, d_head = q.shape
    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]
    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)
    dlogit_bias = (
        xp.zeros(probs.shape, dtype=probs.dtype)
        if cache["has_logit_bias"]
        else None
    )

    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        chunk_indices = key_indices[:, :, q_start:q_end, :]
        chunk_valid = valid_mask[:, :, q_start:q_end, :]
        probs_chunk = probs[:, :, q_start:q_end, :]
        q_chunk = q_heads[:, :, q_start:q_end, :]
        dcontext_chunk = dcontext_heads[:, :, q_start:q_end, :]

        if bf16_attention:
            dcontext_compute = dcontext_chunk.astype("float32")
            q_compute = q_chunk.astype("float32")
            probs_compute = probs_chunk
        else:
            dcontext_compute = dcontext_chunk
            q_compute = q_chunk
            probs_compute = (
                probs_chunk.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_chunk
            )

        # V branch: dP and dV. Gather and release V before K is gathered so the
        # two largest selected-value buffers do not coexist.
        v_selected = v[batch_ids, chunk_indices, kv_ids, :]
        v_compute = (
            v_selected.astype("float32") if bf16_attention else v_selected
        )
        dprobs = xp.matmul(
            dcontext_compute[..., None, :], v_compute.swapaxes(-1, -2)
        )[..., 0, :]
        dv_selected = (
            probs_compute[..., None] * dcontext_compute[..., None, :]
        )

        scatter_batch = xp.broadcast_to(batch_ids, chunk_indices.shape)
        scatter_kv = xp.broadcast_to(kv_ids, chunk_indices.shape)
        xp.add.at(
            dv,
            (scatter_batch, chunk_indices, scatter_kv),
            dv_selected.astype(grad_dtype, copy=False),
        )
        del v_compute, v_selected, dv_selected

        # probs is zero at every invalid key, so the softmax Jacobian gives an
        # exactly zero score gradient there (including completely empty rows).
        dscores = _softmax_backward(dprobs, probs_chunk)
        dscores = xp.where(chunk_valid, dscores, 0.0)
        if dlogit_bias is not None:
            dlogit_bias[:, :, q_start:q_end, :] = dscores

        dscores_compute = (
            dscores.astype("float32", copy=False)
            if bf16_attention
            else (
                dscores.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else dscores
            )
        )

        # K/Q branch.  The Q gradient is another batched [1,K] @ [K,D]
        # product; only dK requires an explicit outer-product tensor, and that
        # tensor is bounded by the query chunk.
        k_selected = k[batch_ids, chunk_indices, kv_ids, :]
        k_compute = (
            k_selected.astype("float32") if bf16_attention else k_selected
        )
        dq_heads[:, :, q_start:q_end, :] = (
            xp.matmul(dscores_compute[..., None, :], k_compute)[..., 0, :]
            * scale
        )
        dk_selected = (
            dscores_compute[..., None] * q_compute[..., None, :] * scale
        )
        xp.add.at(
            dk,
            (scatter_batch, chunk_indices, scatter_kv),
            dk_selected.astype(grad_dtype, copy=False),
        )

    dq = dq_heads.transpose(0, 2, 1, 3)
    return dq, dk, dv, dlogit_bias

