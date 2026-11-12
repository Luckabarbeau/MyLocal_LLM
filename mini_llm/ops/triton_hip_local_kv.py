"""HIP-only split-pass local dK/dV for the optimized local backward policy.

Reconstruct the existing BF16 attention backward graph, but give each query
head/key tile its own FP32 accumulator. Compile-time specialization removes the
unused derivative branch before register allocation. No atomics or FP32 dots.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _kv_kernel(
    q, k, v, documents, d_out, lse_log2, delta, partial,
    q_sb: tl.constexpr, q_st: tl.constexpr, q_sh: tl.constexpr,
    k_sb: tl.constexpr, k_st: tl.constexpr, k_sh: tl.constexpr,
    v_sb: tl.constexpr, v_st: tl.constexpr, v_sh: tl.constexpr,
    doc_sb: tl.constexpr, doc_st: tl.constexpr,
    do_sb: tl.constexpr, do_st: tl.constexpr, do_sh: tl.constexpr,
    l_sb: tl.constexpr, l_st: tl.constexpr, l_sh: tl.constexpr,
    d_sb: tl.constexpr, d_st: tl.constexpr, d_sh: tl.constexpr,
    n_ctx: tl.constexpr, scale: tl.constexpr, scale_log2: tl.constexpr,
    local_window: tl.constexpr, long_window: tl.constexpr,
    long_heads: tl.constexpr, skip_head: tl.constexpr,
    compute_dk: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 16
    head = batch_head % 16
    kv_head = head // 4
    ns = tile * block_n + tl.arange(0, block_n)
    ds = tl.arange(0, 64)
    acc = tl.zeros((block_n, 64), tl.float32)
    if head != skip_head:
        key_docs = tl.load(documents + batch * doc_sb + ns * doc_st,
                           mask=ns < n_ctx, other=-1)
        keys = tl.load(k + batch * k_sb + ns[:, None] * k_st
                       + kv_head * k_sh + ds[None, :],
                       mask=ns[:, None] < n_ctx, other=0.0)
        if compute_dk:
            values = tl.load(v + batch * v_sb + ns[:, None] * v_st
                             + kv_head * v_sh + ds[None, :],
                             mask=ns[:, None] < n_ctx, other=0.0)
        window = tl.where(head < long_heads, long_window, local_window)
        query_end = tl.minimum(n_ctx, (tile + 1) * block_n + window - 1)
        for query_start in tl.range(tile * block_n, query_end, block_m,
                                    num_stages=1):
            ms = query_start + tl.arange(0, block_m)
            queries = tl.load(q + batch * q_sb + ms[:, None] * q_st
                              + head * q_sh + ds[None, :],
                              mask=ms[:, None] < n_ctx, other=0.0)
            lse = tl.load(lse_log2 + batch * l_sb + ms * l_st + head * l_sh,
                          mask=ms < n_ctx, other=0.0)
            query_docs = tl.load(documents + batch * doc_sb + ms * doc_st,
                                 mask=ms < n_ctx, other=-2)
            valid = ((ns[:, None] < n_ctx) & (ms[None, :] < n_ctx)
                     & (key_docs[:, None] == query_docs[None, :])
                     & (ns[:, None] <= ms[None, :])
                     & (ns[:, None] >= ms[None, :] - window + 1))
            scores = tl.dot(keys, tl.trans(queries)) * scale_log2
            probabilities = tl.where(valid, tl.exp2(scores - lse[None, :]), 0.0)
            upstream = tl.load(d_out + batch * do_sb + ms[:, None] * do_st
                               + head * do_sh + ds[None, :],
                               mask=ms[:, None] < n_ctx, other=0.0).to(tl.bfloat16)
            if compute_dk:
                row_delta = tl.load(delta + batch * d_sb + ms * d_st + head * d_sh,
                                    mask=ms < n_ctx, other=0.0)
                d_probabilities = tl.dot(values, tl.trans(upstream))
                d_scores = probabilities * (d_probabilities - row_delta[None, :])
                acc = tl.dot(d_scores.to(tl.bfloat16), queries, acc)
            else:
                acc = tl.dot(probabilities.to(tl.bfloat16), upstream, acc)
    if compute_dk:
        acc = acc * scale
    destination = ((batch * n_ctx + ns[:, None]) * 16 + head) * 64 + ds[None, :]
    # Skipped heads still own their partial slice, and must write zeros.
    tl.store(partial + destination, acc, mask=ns[:, None] < n_ctx)


@triton.jit
def _reduce_kernel(partial_dk, partial_dv, dk, dv,
                   n_elements: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    source = (offsets // 256) * 1024 + ((offsets // 64) % 4) * 256 + offsets % 64
    dk_acc = tl.load(partial_dk + source, mask=mask, other=0.0)
    dv_acc = tl.load(partial_dv + source, mask=mask, other=0.0)
    for group in tl.static_range(1, 4):
        dk_acc += tl.load(partial_dk + source + group * 64, mask=mask, other=0.0)
        dv_acc += tl.load(partial_dv + source + group * 64, mask=mask, other=0.0)
    tl.store(dk + offsets, dk_acc, mask=mask)
    tl.store(dv + offsets, dv_acc, mask=mask)


def _launch_delta(output, d_out, delta, output_strides, upstream_strides,
                  delta_strides, length, batch, active_mask):
    # Runtime imports keep this independent module safe for future dispatch use.
    from .triton_dual_resolution_local import _delta_kernel, _ptr

    return _delta_kernel[(triton.cdiv(length, 64), batch * 16)](
        _ptr(output), _ptr(d_out), _ptr(delta),
        *output_strides[:3], *upstream_strides[:3], *delta_strides[:3],
        n_ctx=length, active_mask=active_mask, block_m=64, num_warps=4,
    )


def backward(d_out, cache, *, delta=None):
    """Return FP32 [B,T,4,64] dK/dV for an existing local-attention cache.

    Q/K/V may be packed views with arbitrary batch/token/head strides, provided
    the last dimension is contiguous, as required by the existing forward API.
    """
    import cupy as cp
    from .triton_dual_resolution_local import _element_strides, _ptr

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('split-pass local KV backward requires HIP')
    q, k, v, docs = cache.q, cache.k, cache.v, cache.documents
    if d_out.shape != q.shape or str(d_out.dtype) != 'float32':
        raise ValueError('local Triton backward requires FP32 d_out with Q shape')
    batch, length = map(int, q.shape[:2])
    reuse_delta = delta is not None
    if not reuse_delta:
        delta = cp.empty(q.shape[:3], dtype=cp.float32)
    partial_dk = cp.empty(q.shape, dtype=cp.float32)
    partial_dv = cp.empty(q.shape, dtype=cp.float32)
    dk = cp.empty(k.shape, dtype=cp.float32)
    dv = cp.empty(v.shape, dtype=cp.float32)
    qs, ks, vs, docs_s, os, dos, ls, dels = map(
        _element_strides,
        (q, k, v, docs, cache.output, d_out, cache.lse_log2, delta),
    )
    active_mask = 0xFFFF if cache.skip_head < 0 else 0xFFFF & ~(1 << cache.skip_head)
    if not reuse_delta:
        _launch_delta(cache.output, d_out, delta, os, dos, dels, length, batch, active_mask)
    for compute_dk, partial in ((True, partial_dk), (False, partial_dv)):
        _kv_kernel[(triton.cdiv(length, 32), batch * 16)](
            _ptr(q), _ptr(k), _ptr(v), _ptr(docs), _ptr(d_out),
            _ptr(cache.lse_log2), _ptr(delta), _ptr(partial),
            *qs[:3], *ks[:3], *vs[:3], *docs_s[:2], *dos[:3], *ls[:3], *dels[:3],
            n_ctx=length, scale=cache.scale,
            scale_log2=cache.scale * 1.4426950408889634,
            local_window=cache.window, long_window=cache.long_window,
            long_heads=cache.long_heads, skip_head=cache.skip_head,
            compute_dk=compute_dk, block_m=32, block_n=32,
            num_warps=4, num_stages=1,
        )
    n_elements = batch * length * 4 * 64
    _reduce_kernel[(triton.cdiv(n_elements, 256),)](
        _ptr(partial_dk), _ptr(partial_dv), _ptr(dk), _ptr(dv),
        n_elements=n_elements, block=256, num_warps=4,
    )
    return dk, dv
