"""Opt-in Triton mixed-attention kernels for the production geometry.

The implementation specializes FlashAttention-style forward and backward
traversals for BF16 Q/K/V, D=64, 16:4 GQA, one full causal head, and 512-token
local causal heads. It accepts CuPy allocations directly and has no Torch
dependency. Production dispatch remains opt-in and falls back before importing
this module when the geometry or optional Triton dependency is unavailable.
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
        valid = (
            (offs_m[:, None] < n_ctx)
            & (key_offsets[None, :] <= offs_m[:, None])
            & (key_offsets[None, :] >= offs_m[:, None] - window + 1)
            & (key_offsets[None, :] < n_ctx)
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
        valid = (
            (offs_m[:, None] < n_ctx)
            & (key_offsets[None, :] <= offs_m[:, None])
            & (key_offsets[None, :] >= offs_m[:, None] - window + 1)
            & (key_offsets[None, :] < n_ctx)
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
        is_full_head = original_head == 0
        query_end = tl.where(
            is_full_head,
            n_ctx,
            tl.minimum(n_ctx, (pid_n + 1) * block_n + local_window - 1),
        )
        head_window = tl.where(is_full_head, n_ctx, local_window)
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
                valid = (
                    (offs_n[:, None] < n_ctx)
                    & (offs_m[None, :] < n_ctx)
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


def _active_mask(active_local_heads: Iterable[int]) -> int:
    heads = tuple(int(head) for head in active_local_heads)
    if len(set(heads)) != len(heads):
        raise ValueError("active local heads must be unique")
    if any(head < 1 or head >= N_Q_HEADS for head in heads):
        raise ValueError("active local heads must be original heads in [1, 15]")
    mask = 1
    for head in heads:
        mask |= 1 << head
    return mask


def _validate(q, k, v, window):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must have shape [B,T,H,D]")
    if q.shape[2:] != (N_Q_HEADS, HEAD_DIM):
        raise ValueError("0088A requires Q with 16 heads and D=64")
    if k.shape != v.shape or k.shape[:2] != q.shape[:2]:
        raise ValueError("K/V must match Q batch and sequence dimensions")
    if k.shape[2:] != (N_KV_HEADS, HEAD_DIM):
        raise ValueError("0088A requires K/V with four heads and D=64")
    if any(str(array.dtype) != "bfloat16" for array in (q, k, v)):
        raise ValueError("0088A requires BF16 Q/K/V")
    if int(window) != LOCAL_WINDOW:
        raise ValueError("0088A is specialized to the inclusive 512-token window")
    if int(q.shape[1]) < 1:
        raise ValueError("sequence length must be positive")
    for name, array in (("Q", q), ("K", k), ("V", v)):
        if int(array.strides[-1]) != int(array.dtype.itemsize):
            raise ValueError(f"{name} requires contiguous head-dimension elements")


@dataclass(frozen=True)
class AttentionCache:
    q: object
    k: object
    v: object
    output: object
    lse_log2: object
    active_local_heads: object
    local_heads_indexed: bool
    active_mask: int
    window: int
    scale: float


def mixed_attention_forward(q, k, v, active_local_heads, *, window=512, scale=0.125):
    """Return FP32 mixed-attention output and the minimal backward cache."""
    import cupy as cp

    _validate(q, k, v, window)
    active_local_heads = tuple(int(head) for head in active_local_heads)
    active_mask = _active_mask(active_local_heads)
    local_head_indices = cp.asarray(active_local_heads, dtype=cp.int32)
    local_heads_indexed = active_local_heads != tuple(
        range(1, 1 + len(active_local_heads))
    )
    batch, length = map(int, q.shape[:2])
    output = cp.zeros(q.shape, dtype=cp.float32)
    lse_log2 = cp.zeros(q.shape[:3], dtype=cp.float32)
    q_s, k_s, v_s = map(_element_strides, (q, k, v))
    o_s, l_s = map(_element_strides, (output, lse_log2))
    scale_log2 = float(scale) * _LOG2E

    common = (
        _ptr(q), _ptr(k), _ptr(v), _ptr(output), _ptr(lse_log2),
        _ptr(local_head_indices), q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
        v_s[0], v_s[1], v_s[2], o_s[0], o_s[1], o_s[2],
        l_s[0], l_s[1], l_s[2],
    )
    grid_m = triton.cdiv(length, 64)
    _fwd_kernel[(grid_m, batch)](
        *common,
        n_ctx=length,
        scale_log2=scale_log2,
        active_mask=active_mask,
        head_begin=0,
        head_count=1,
        use_head_indices=False,
        window=length,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
    _fwd_kernel[(grid_m, batch * len(active_local_heads))](
        *common,
        n_ctx=length,
        scale_log2=scale_log2,
        active_mask=active_mask,
        head_begin=1,
        head_count=len(active_local_heads),
        use_head_indices=local_heads_indexed,
        window=LOCAL_WINDOW,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
    return output, AttentionCache(
        q=q,
        k=k,
        v=v,
        output=output,
        lse_log2=lse_log2,
        active_local_heads=local_head_indices,
        local_heads_indexed=local_heads_indexed,
        active_mask=active_mask,
        window=int(window),
        scale=float(scale),
    )


def mixed_attention_backward(d_out, cache: AttentionCache):
    """Return FP32 dQ/dK/dV with native shared-KV accumulation."""
    import cupy as cp

    q, k, v = cache.q, cache.k, cache.v
    if d_out.shape != q.shape or str(d_out.dtype) != "float32":
        raise ValueError("d_out must be FP32 and match Q shape")
    batch, length = map(int, q.shape[:2])
    d_q = cp.zeros(q.shape, dtype=cp.float32)
    d_k = cp.empty(k.shape, dtype=cp.float32)
    d_v = cp.empty(v.shape, dtype=cp.float32)
    delta = cp.zeros(q.shape[:3], dtype=cp.float32)

    q_s, k_s, v_s = map(_element_strides, (q, k, v))
    o_s = _element_strides(cache.output)
    do_s = _element_strides(d_out)
    l_s = _element_strides(cache.lse_log2)
    d_s = _element_strides(delta)
    dq_s, dk_s, dv_s = map(_element_strides, (d_q, d_k, d_v))
    grid_m = triton.cdiv(length, 64)

    _delta_kernel[(grid_m, batch * N_Q_HEADS)](
        _ptr(cache.output), _ptr(d_out), _ptr(delta),
        o_s[0], o_s[1], o_s[2], do_s[0], do_s[1], do_s[2],
        d_s[0], d_s[1], d_s[2],
        n_ctx=length,
        active_mask=cache.active_mask,
        block_m=64,
        num_warps=4,
    )

    dq_common = (
        _ptr(q), _ptr(k), _ptr(v), _ptr(d_out), _ptr(cache.lse_log2),
        _ptr(delta), _ptr(d_q), _ptr(cache.active_local_heads),
        q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
        v_s[0], v_s[1], v_s[2], do_s[0], do_s[1], do_s[2],
        l_s[0], l_s[1], l_s[2], d_s[0], d_s[1], d_s[2],
        dq_s[0], dq_s[1], dq_s[2],
    )
    _dq_kernel[(grid_m, batch)](
        *dq_common,
        n_ctx=length,
        scale=cache.scale,
        scale_log2=cache.scale * _LOG2E,
        active_mask=cache.active_mask,
        head_begin=0,
        head_count=1,
        use_head_indices=False,
        window=length,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
    local_head_count = int(cache.active_local_heads.size)
    _dq_kernel[(grid_m, batch * local_head_count)](
        *dq_common,
        n_ctx=length,
        scale=cache.scale,
        scale_log2=cache.scale * _LOG2E,
        active_mask=cache.active_mask,
        head_begin=1,
        head_count=local_head_count,
        use_head_indices=cache.local_heads_indexed,
        window=LOCAL_WINDOW,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )

    dkdv_common = (
        _ptr(q), _ptr(k), _ptr(v), _ptr(d_out), _ptr(cache.lse_log2),
        _ptr(delta), _ptr(d_k), _ptr(d_v),
        q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
        v_s[0], v_s[1], v_s[2], do_s[0], do_s[1], do_s[2],
        l_s[0], l_s[1], l_s[2], d_s[0], d_s[1], d_s[2],
        dk_s[0], dk_s[1], dk_s[2], dv_s[0], dv_s[1], dv_s[2],
    )
    _dkdv_kernel[(grid_m, batch * N_KV_HEADS)](
        *dkdv_common,
        n_ctx=length,
        scale=cache.scale,
        scale_log2=cache.scale * _LOG2E,
        active_mask=cache.active_mask,
        local_window=LOCAL_WINDOW,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
    return d_q, d_k, d_v


def mixed_attention_forward_backward(
    q, k, v, d_out, active_local_heads, *, window=512, scale=0.125
):
    output, cache = mixed_attention_forward(
        q, k, v, active_local_heads, window=window, scale=scale
    )
    gradients = mixed_attention_backward(d_out, cache)
    return output, gradients
