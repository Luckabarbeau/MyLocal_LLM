"""0102B experimental Triton local GQA: 16 sliding heads, EOS-safe.

Adapted from triton_mixed_attention.py, with all 16 heads local (including head
zero) and compact per-token document-ID masking in forward and backward. The
first configurable number of query heads may use a longer local window.
CUDA/CuPy only, BF16, 16Q/4KV, D64, short window 512. Never imports PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised by capability probe
    raise RuntimeError(
        "the opt-in Triton attention backend requires triton>=3.8"
    ) from exc


N_Q_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = 64
LOCAL_WINDOW = 512
_LOG2E = 1.4426950408889634


class _Pointer:
    """Expose a CuPy allocation through Triton's minimal tensor protocol."""

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


@triton.jit
def _fwd_kernel(
    q,
    k,
    v,
    doc_ids,
    out,
    lse_log2,
    head_indices,
    q_sb: tl.constexpr,
    q_st: tl.constexpr,
    q_sh: tl.constexpr,
    k_sb: tl.constexpr,
    k_st: tl.constexpr,
    k_sh: tl.constexpr,
    v_sb: tl.constexpr,
    v_st: tl.constexpr,
    v_sh: tl.constexpr,
    doc_sb: tl.constexpr,
    doc_st: tl.constexpr,
    o_sb: tl.constexpr,
    o_st: tl.constexpr,
    o_sh: tl.constexpr,
    l_sb: tl.constexpr,
    l_st: tl.constexpr,
    l_sh: tl.constexpr,
    n_ctx: tl.constexpr,
    scale_log2: tl.constexpr,
    active_mask: tl.constexpr,
    head_begin: tl.constexpr,
    head_count: tl.constexpr,
    use_head_indices: tl.constexpr,
    window: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // head_count
    head_slot = pid_bh % head_count
    if use_head_indices:
        head = tl.load(head_indices + head_slot)
    else:
        head = head_begin + head_slot
    kv_head = head // 4
    enabled = True

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    q_mask = (offs_m[:, None] < n_ctx) & enabled
    query_docs = tl.load(doc_ids + batch * doc_sb + offs_m * doc_st,
                         mask=offs_m < n_ctx, other=-1)
    q_block = tl.load(
        q
        + batch * q_sb
        + offs_m[:, None] * q_st
        + head * q_sh
        + offs_d[None, :],
        mask=q_mask,
        other=0.0,
    )

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, 64), tl.float32)

    q_start = pid_m * block_m
    key_begin = tl.maximum(0, q_start - window + 1)
    key_begin = (key_begin // block_n) * block_n
    key_end = tl.minimum(n_ctx, q_start + block_m)

    for key_start in tl.range(key_begin, key_end, block_n, num_stages=2):
        key_offsets = key_start + offs_n
        key_mask = key_offsets[:, None] < n_ctx
        k_block = tl.load(
            k
            + batch * k_sb
            + key_offsets[:, None] * k_st
            + kv_head * k_sh
            + offs_d[None, :],
            mask=key_mask,
            other=0.0,
        )
        scores = tl.dot(q_block, tl.trans(k_block)) * scale_log2
        key_docs = tl.load(doc_ids + batch * doc_sb + key_offsets * doc_st,
                           mask=key_offsets < n_ctx, other=-2)
        valid = (
            (offs_m[:, None] < n_ctx)
            & (key_offsets[None, :] <= offs_m[:, None])
            & (key_offsets[None, :] >= offs_m[:, None] - window + 1)
            & (key_offsets[None, :] < n_ctx)
            & (query_docs[:, None] == key_docs[None, :])
            & enabled
        )
        scores = tl.where(valid, scores, -float("inf"))
        tile_has_values = tl.sum(valid.to(tl.int32), axis=1) > 0
        tile_max = tl.max(scores, axis=1)
        new_max = tl.where(
            tile_has_values, tl.maximum(row_max, tile_max), row_max
        )
        probabilities = tl.where(
            valid, tl.exp2(scores - new_max[:, None]), 0.0
        )
        alpha = tl.where(
            tile_has_values, tl.exp2(row_max - new_max), 1.0
        )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        acc *= alpha[:, None]

        v_block = tl.load(
            v
            + batch * v_sb
            + key_offsets[:, None] * v_st
            + kv_head * v_sh
            + offs_d[None, :],
            mask=key_mask,
            other=0.0,
        )
        acc = tl.dot(probabilities.to(tl.bfloat16), v_block, acc)
        row_max = new_max

    result = acc / row_sum[:, None]
    output_mask = (offs_m[:, None] < n_ctx) & enabled
    tl.store(
        out
        + batch * o_sb
        + offs_m[:, None] * o_st
        + head * o_sh
        + offs_d[None, :],
        result,
        mask=output_mask,
    )
    tl.store(
        lse_log2 + batch * l_sb + offs_m * l_st + head * l_sh,
        row_max + tl.log2(row_sum),
        mask=(offs_m < n_ctx) & enabled,
    )


@triton.jit
def _delta_kernel(
    out,
    d_out,
    delta,
    o_sb: tl.constexpr,
    o_st: tl.constexpr,
    o_sh: tl.constexpr,
    do_sb: tl.constexpr,
    do_st: tl.constexpr,
    do_sh: tl.constexpr,
    d_sb: tl.constexpr,
    d_st: tl.constexpr,
    d_sh: tl.constexpr,
    n_ctx: tl.constexpr,
    active_mask: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 16
    head = pid_bh % 16
    enabled = (active_mask & (1 << head)) != 0
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    mask = (offs_m[:, None] < n_ctx) & enabled
    output = tl.load(
        out
        + batch * o_sb
        + offs_m[:, None] * o_st
        + head * o_sh
        + offs_d[None, :],
        mask=mask,
        other=0.0,
    )
    upstream = tl.load(
        d_out
        + batch * do_sb
        + offs_m[:, None] * do_st
        + head * do_sh
        + offs_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    value = tl.sum(output * upstream, axis=1)
    tl.store(
        delta + batch * d_sb + offs_m * d_st + head * d_sh,
        value,
        mask=(offs_m < n_ctx) & enabled,
    )


@triton.jit
def _dq_kernel(
    q,
    k,
    v,
    doc_ids,
    d_out,
    lse_log2,
    delta,
    d_q,
    head_indices,
    q_sb: tl.constexpr,
    q_st: tl.constexpr,
    q_sh: tl.constexpr,
    k_sb: tl.constexpr,
    k_st: tl.constexpr,
    k_sh: tl.constexpr,
    v_sb: tl.constexpr,
    v_st: tl.constexpr,
    v_sh: tl.constexpr,
    doc_sb: tl.constexpr,
    doc_st: tl.constexpr,
    do_sb: tl.constexpr,
    do_st: tl.constexpr,
    do_sh: tl.constexpr,
    l_sb: tl.constexpr,
    l_st: tl.constexpr,
    l_sh: tl.constexpr,
    d_sb: tl.constexpr,
    d_st: tl.constexpr,
    d_sh: tl.constexpr,
    dq_sb: tl.constexpr,
    dq_st: tl.constexpr,
    dq_sh: tl.constexpr,
    n_ctx: tl.constexpr,
    scale: tl.constexpr,
    scale_log2: tl.constexpr,
    active_mask: tl.constexpr,
    head_begin: tl.constexpr,
    head_count: tl.constexpr,
    use_head_indices: tl.constexpr,
    window: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // head_count
    head_slot = pid_bh % head_count
    if use_head_indices:
        head = tl.load(head_indices + head_slot)
    else:
        head = head_begin + head_slot
    kv_head = head // 4
    enabled = True
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    row_mask = (offs_m[:, None] < n_ctx) & enabled

    query_docs = tl.load(doc_ids + batch * doc_sb + offs_m * doc_st,
                         mask=offs_m < n_ctx, other=-1)
    q_block = tl.load(
        q
        + batch * q_sb
        + offs_m[:, None] * q_st
        + head * q_sh
        + offs_d[None, :],
        mask=row_mask,
        other=0.0,
    )
    upstream = tl.load(
        d_out
        + batch * do_sb
        + offs_m[:, None] * do_st
        + head * do_sh
        + offs_d[None, :],
        mask=row_mask,
        other=0.0,
    ).to(tl.bfloat16)
    lse = tl.load(
        lse_log2 + batch * l_sb + offs_m * l_st + head * l_sh,
        mask=(offs_m < n_ctx) & enabled,
        other=0.0,
    )
    row_delta = tl.load(
        delta + batch * d_sb + offs_m * d_st + head * d_sh,
        mask=(offs_m < n_ctx) & enabled,
        other=0.0,
    )
    dq_acc = tl.zeros((block_m, 64), tl.float32)

    q_start = pid_m * block_m
    key_begin = tl.maximum(0, q_start - window + 1)
    key_begin = (key_begin // block_n) * block_n
    key_end = tl.minimum(n_ctx, q_start + block_m)
    for key_start in tl.range(key_begin, key_end, block_n, num_stages=2):
        key_offsets = key_start + offs_n
        key_mask = key_offsets[:, None] < n_ctx
        k_block = tl.load(
            k
            + batch * k_sb
            + key_offsets[:, None] * k_st
            + kv_head * k_sh
            + offs_d[None, :],
            mask=key_mask,
            other=0.0,
        )
        v_block = tl.load(
            v
            + batch * v_sb
            + key_offsets[:, None] * v_st
            + kv_head * v_sh
            + offs_d[None, :],
            mask=key_mask,
            other=0.0,
        )
        key_docs = tl.load(doc_ids + batch * doc_sb + key_offsets * doc_st,
                           mask=key_offsets < n_ctx, other=-2)
        valid = (
            (offs_m[:, None] < n_ctx)
            & (key_offsets[None, :] <= offs_m[:, None])
            & (key_offsets[None, :] >= offs_m[:, None] - window + 1)
            & (key_offsets[None, :] < n_ctx)
            & (query_docs[:, None] == key_docs[None, :])
            & enabled
        )
        scores = tl.dot(q_block, tl.trans(k_block)) * scale_log2
        probabilities = tl.exp2(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        d_probabilities = tl.dot(upstream, tl.trans(v_block))
        d_scores = probabilities * (d_probabilities - row_delta[:, None])
        dq_acc = tl.dot(d_scores.to(tl.bfloat16), k_block, dq_acc)

    dq_acc *= scale
    tl.store(
        d_q
        + batch * dq_sb
        + offs_m[:, None] * dq_st
        + head * dq_sh
        + offs_d[None, :],
        dq_acc,
        mask=row_mask,
    )


@triton.jit
def _dkdv_kernel(
    q,
    k,
    v,
    doc_ids,
    d_out,
    lse_log2,
    delta,
    d_k,
    d_v,
    q_sb: tl.constexpr,
    q_st: tl.constexpr,
    q_sh: tl.constexpr,
    k_sb: tl.constexpr,
    k_st: tl.constexpr,
    k_sh: tl.constexpr,
    v_sb: tl.constexpr,
    v_st: tl.constexpr,
    v_sh: tl.constexpr,
    doc_sb: tl.constexpr,
    doc_st: tl.constexpr,
    do_sb: tl.constexpr,
    do_st: tl.constexpr,
    do_sh: tl.constexpr,
    l_sb: tl.constexpr,
    l_st: tl.constexpr,
    l_sh: tl.constexpr,
    d_sb: tl.constexpr,
    d_st: tl.constexpr,
    d_sh: tl.constexpr,
    dk_sb: tl.constexpr,
    dk_st: tl.constexpr,
    dk_sh: tl.constexpr,
    dv_sb: tl.constexpr,
    dv_st: tl.constexpr,
    dv_sh: tl.constexpr,
    n_ctx: tl.constexpr,
    scale: tl.constexpr,
    scale_log2: tl.constexpr,
    active_mask: tl.constexpr,
    local_window: tl.constexpr,
    long_window: tl.constexpr,
    long_heads: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bkv = tl.program_id(1)
    batch = pid_bkv // 4
    kv_head = pid_bkv % 4
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m_base = tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    key_mask = offs_n[:, None] < n_ctx
    key_docs = tl.load(doc_ids + batch * doc_sb + offs_n * doc_st,
                       mask=offs_n < n_ctx, other=-1)
    k_block = tl.load(
        k
        + batch * k_sb
        + offs_n[:, None] * k_st
        + kv_head * k_sh
        + offs_d[None, :],
        mask=key_mask,
        other=0.0,
    )
    v_block = tl.load(
        v
        + batch * v_sb
        + offs_n[:, None] * v_st
        + kv_head * v_sh
        + offs_d[None, :],
        mask=key_mask,
        other=0.0,
    )
    dk_acc = tl.zeros((block_n, 64), tl.float32)
    dv_acc = tl.zeros((block_n, 64), tl.float32)

    for group_offset in tl.static_range(0, 4):
        original_head = kv_head * 4 + group_offset
        head_enabled = (active_mask & (1 << original_head)) != 0
        is_long_head = (original_head < long_heads).to(tl.int1)
        head_window = tl.where(is_long_head, long_window, local_window)
        query_end = tl.minimum(
            n_ctx, (pid_n + 1) * block_n + head_window - 1
        )
        query_begin = pid_n * block_n
        if head_enabled:
            for query_start in tl.range(query_begin, query_end, block_m, num_stages=2):
                offs_m = query_start + offs_m_base
                query_mask = offs_m[:, None] < n_ctx
                q_block = tl.load(
                    q
                    + batch * q_sb
                    + offs_m[:, None] * q_st
                    + original_head * q_sh
                    + offs_d[None, :],
                    mask=query_mask,
                    other=0.0,
                )
                upstream = tl.load(
                    d_out
                    + batch * do_sb
                    + offs_m[:, None] * do_st
                    + original_head * do_sh
                    + offs_d[None, :],
                    mask=query_mask,
                    other=0.0,
                ).to(tl.bfloat16)
                lse = tl.load(
                    lse_log2 + batch * l_sb + offs_m * l_st + original_head * l_sh,
                    mask=offs_m < n_ctx,
                    other=0.0,
                )
                row_delta = tl.load(
                    delta + batch * d_sb + offs_m * d_st + original_head * d_sh,
                    mask=offs_m < n_ctx,
                    other=0.0,
                )
                query_docs = tl.load(doc_ids + batch * doc_sb + offs_m * doc_st,
                                     mask=offs_m < n_ctx, other=-2)
                valid = (
                    (offs_n[:, None] < n_ctx)
                    & (offs_m[None, :] < n_ctx)
                    & (key_docs[:, None] == query_docs[None, :])
                    & (offs_n[:, None] <= offs_m[None, :])
                    & (offs_n[:, None] >= offs_m[None, :] - head_window + 1)
                )
                scores_t = tl.dot(k_block, tl.trans(q_block)) * scale_log2
                probabilities_t = tl.exp2(scores_t - lse[None, :])
                probabilities_t = tl.where(valid, probabilities_t, 0.0)
                dv_acc = tl.dot(probabilities_t.to(tl.bfloat16), upstream, dv_acc)
                d_probabilities_t = tl.dot(v_block, tl.trans(upstream))
                d_scores_t = probabilities_t * (
                    d_probabilities_t - row_delta[None, :]
                )
                dk_acc = tl.dot(d_scores_t.to(tl.bfloat16), q_block, dk_acc)

    tl.store(
        d_k
        + batch * dk_sb
        + offs_n[:, None] * dk_st
        + kv_head * dk_sh
        + offs_d[None, :],
        dk_acc * scale,
        mask=key_mask,
    )
    tl.store(
        d_v
        + batch * dv_sb
        + offs_n[:, None] * dv_st
        + kv_head * dv_sh
        + offs_d[None, :],
        dv_acc,
        mask=key_mask,
    )


def _element_strides(array):
    return tuple(int(value // array.dtype.itemsize) for value in array.strides)


def _ptr(array):
    dtype_name = str(array.dtype)
    if dtype_name == "bfloat16":
        dtype = tl.bfloat16
    elif dtype_name == "int32":
        dtype = tl.int32
    else:
        dtype = tl.float32
    return _Pointer(array, dtype)


@dataclass(frozen=True)
class LocalAttentionCache:
    q: object
    k: object
    v: object
    documents: object
    output: object
    lse_log2: object
    unused_head_indices: object
    scale: float
    window: int = LOCAL_WINDOW
    long_window: int = LOCAL_WINDOW
    long_heads: int = 0


def _validate(q, k, v, documents, *, window, long_window, long_heads, scale):
    import cupy as cp
    if not all(isinstance(array, cp.ndarray) for array in (q, k, v, documents)):
        raise TypeError("Triton local attention requires CuPy arrays")
    if q.ndim != 4 or q.shape[2:] != (16, 64):
        raise ValueError("Triton local Q must have shape [B,T,16,64]")
    if k.shape != v.shape or k.shape != q.shape[:2] + (4, 64):
        raise ValueError("Triton local K/V must have shape [B,T,4,64]")
    if documents.shape != q.shape[:2]:
        raise ValueError("document IDs must have shape [B,T]")
    if any(str(array.dtype) != "bfloat16" for array in (q, k, v)):
        raise ValueError("Triton local backend requires BF16 Q/K/V")
    if str(documents.dtype) != "int32":
        raise ValueError("Triton local backend requires int32 document IDs")
    if any(array.strides[-1] != array.dtype.itemsize for array in (q, k, v)):
        raise ValueError("contiguous Q/K/V head dimension required")
    if window != LOCAL_WINDOW:
        raise ValueError("Triton local backend requires short window 512")
    if not isinstance(long_window, (int, np.integer)) or long_window < LOCAL_WINDOW:
        raise ValueError("long_window must be a positive integer >= 512")
    if not isinstance(long_heads, (int, np.integer)) or not 0 <= long_heads <= N_Q_HEADS:
        raise ValueError("long_heads must be an integer in [0, 16]")
    if float(scale) != 0.125:
        raise ValueError("Triton local backend is specialized to scale 0.125")
    if q.shape[1] == 0:
        raise ValueError("empty sequence")


def local_attention_forward(
    q, k, v, document_ids, *, window=512, scale=0.125,
    long_window=None, long_heads=0,
):
    """BF16 EOS-safe local attention with optional long-window leading heads.

    ``window`` remains fixed at 512. ``long_window`` defaults to ``window``;
    query heads before ``long_heads`` use it. Returns FP32 [B,T,16,64]
    context without allocating T*T probabilities.
    """
    import cupy as cp
    if long_window is None:
        long_window = window
    docs = cp.asarray(document_ids, dtype=cp.int32)
    _validate(
        q, k, v, docs, window=window, long_window=long_window,
        long_heads=long_heads, scale=scale,
    )
    window = int(window)
    long_window = int(long_window)
    long_heads = int(long_heads)
    batch, length = (int(value) for value in q.shape[:2])
    out = cp.empty(q.shape, dtype=cp.float32)
    lse = cp.empty(q.shape[:3], dtype=cp.float32)
    heads = cp.empty((1,), dtype=cp.int32)  # unused: all heads are contiguous
    qs, ks, vs, ds, os, ls = map(
        _element_strides, (q, k, v, docs, out, lse)
    )
    head_groups = []
    if long_heads:
        head_groups.append((0, long_heads, long_window))
    if long_heads < N_Q_HEADS:
        head_groups.append((long_heads, N_Q_HEADS - long_heads, window))
    for head_begin, head_count, head_window in head_groups:
        _fwd_kernel[(triton.cdiv(length, 64), batch * head_count)](
            _ptr(q), _ptr(k), _ptr(v), _ptr(docs), _ptr(out), _ptr(lse), _ptr(heads),
            *qs[:3], *ks[:3], *vs[:3], *ds[:2], *os[:3], *ls[:3],
            n_ctx=length, scale_log2=float(scale) * _LOG2E,
            active_mask=0xFFFF, head_begin=head_begin, head_count=head_count,
            use_head_indices=False, window=head_window, block_m=64, block_n=64,
            num_warps=4, num_stages=3,
        )
    cache = LocalAttentionCache(
        q, k, v, docs, out, lse, heads, float(scale),
        window, long_window, long_heads,
    )
    return out, cache


def local_attention_backward(d_out, cache):
    """Return FP32 dQ, dK, dV; KV gradients sum all four query groups."""
    import cupy as cp
    q, k, v, docs = cache.q, cache.k, cache.v, cache.documents
    if d_out.shape != q.shape or str(d_out.dtype) != 'float32':
        raise ValueError('local Triton backward requires FP32 d_out with Q shape')
    batch, length = map(int, q.shape[:2])
    dq = cp.empty(q.shape, dtype=cp.float32)
    dk = cp.empty(k.shape, dtype=cp.float32)
    dv = cp.empty(v.shape, dtype=cp.float32)
    delta = cp.empty(q.shape[:3], dtype=cp.float32)
    qs, ks, vs, ds, os, dos, ls, dels, dqs, dks, dvs = map(
        _element_strides, (
            q, k, v, docs, cache.output, d_out, cache.lse_log2, delta, dq, dk, dv,
        )
    )
    grid = triton.cdiv(length, 64)
    _delta_kernel[(grid, batch * 16)](
        _ptr(cache.output), _ptr(d_out), _ptr(delta),
        *os[:3], *dos[:3], *dels[:3],
        n_ctx=length, active_mask=0xFFFF, block_m=64, num_warps=4,
    )
    head_groups = []
    if cache.long_heads:
        head_groups.append((0, cache.long_heads, cache.long_window))
    if cache.long_heads < N_Q_HEADS:
        head_groups.append((
            cache.long_heads, N_Q_HEADS - cache.long_heads, cache.window,
        ))
    for head_begin, head_count, head_window in head_groups:
        _dq_kernel[(grid, batch * head_count)](
            _ptr(q), _ptr(k), _ptr(v), _ptr(docs), _ptr(d_out),
            _ptr(cache.lse_log2), _ptr(delta), _ptr(dq),
            _ptr(cache.unused_head_indices),
            *qs[:3], *ks[:3], *vs[:3], *ds[:2], *dos[:3], *ls[:3],
            *dels[:3], *dqs[:3],
            n_ctx=length, scale=cache.scale, scale_log2=cache.scale * _LOG2E,
            active_mask=0xFFFF, head_begin=head_begin, head_count=head_count,
            use_head_indices=False, window=head_window,
            block_m=64, block_n=64, num_warps=4, num_stages=3,
        )
    _dkdv_kernel[(grid, batch * 4)](
        _ptr(q), _ptr(k), _ptr(v), _ptr(docs), _ptr(d_out),
        _ptr(cache.lse_log2), _ptr(delta), _ptr(dk), _ptr(dv),
        *qs[:3], *ks[:3], *vs[:3], *ds[:2], *dos[:3], *ls[:3], *dels[:3],
        *dks[:3], *dvs[:3],
        n_ctx=length, scale=cache.scale, scale_log2=cache.scale * _LOG2E,
        active_mask=0xFFFF, local_window=cache.window,
        long_window=cache.long_window, long_heads=cache.long_heads,
        block_m=64, block_n=64, num_warps=4, num_stages=3,
    )
    return dq, dk, dv
