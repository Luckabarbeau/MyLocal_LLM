"""Segmented BF16 HIP global/sparse dK/dV for optimized backward dispatch.

Each key tile/query head/query segment owns one FP32 partial per pass. Segment
and GQA reductions have a fixed order and use no atomics. The cache's original
mask predicate is authoritative; no bounded-history metadata assumptions are
used to prune the query range. FP32 Q/K/V and CUDA must use the caller's existing
fallback. This module does not compute dQ.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .triton_dual_resolution_dense import _LOG2E, _ptr, _strides, _validate, _visible


@triton.jit
def _delta(
    O, DO, DELTA,
    osb: tl.constexpr, ost: tl.constexpr, osh: tl.constexpr, osd: tl.constexpr,
    dosb: tl.constexpr, dost: tl.constexpr, dosh: tl.constexpr, dosd: tl.constexpr,
    NQ: tl.constexpr, HQ: tl.constexpr,
):
    m = tl.program_id(0) * 32 + tl.arange(0, 32)
    bh = tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    d = tl.arange(0, 64)
    o = tl.load(O + batch * osb + m[:, None] * ost + head * osh
                + d[None, :] * osd, mask=m[:, None] < NQ, other=0.)
    dout = tl.load(DO + batch * dosb + m[:, None] * dost + head * dosh
                   + d[None, :] * dosd, mask=m[:, None] < NQ, other=0.)
    delta = tl.sum(o.to(tl.float32) * dout.to(tl.float32), axis=1)
    tl.store(DELTA + (batch * NQ + m) * HQ + head, delta, mask=m < NQ)


@triton.jit
def _kv_pass(
    Q, K, V, QDOC, QPOS, KDOC, KEND, KVALID, DO, LSE, DELTA, PARTIAL,
    qsb: tl.constexpr, qst: tl.constexpr, qsh: tl.constexpr,
    ksb: tl.constexpr, kst: tl.constexpr, ksh: tl.constexpr,
    vsb: tl.constexpr, vst: tl.constexpr, vsh: tl.constexpr,
    dosb: tl.constexpr, dost: tl.constexpr, dosh: tl.constexpr, dosd: tl.constexpr,
    lsb: tl.constexpr, lst: tl.constexpr, lsh: tl.constexpr,
    qdsb: tl.constexpr, qdst: tl.constexpr, qpsb: tl.constexpr, qpst: tl.constexpr,
    kdsb: tl.constexpr, kdst: tl.constexpr, kesb: tl.constexpr, kest: tl.constexpr,
    kvsb: tl.constexpr, kvst: tl.constexpr,
    NQ: tl.constexpr, NK: tl.constexpr, HQ: tl.constexpr, GROUP: tl.constexpr,
    SEGMENTS: tl.constexpr, CHUNK: tl.constexpr, SCALE_LOG2: tl.constexpr,
    SPARSE_STRIDE: tl.constexpr, SPARSE_PHASE: tl.constexpr, COMPUTE_DK: tl.constexpr,
):
    kn = tl.program_id(0) * 32 + tl.arange(0, 32)
    bh, segment = tl.program_id(1), tl.program_id(2)
    batch, head = bh // HQ, bh % HQ
    kvhead = head // GROUP
    d = tl.arange(0, 64)
    k = tl.load(K + batch * ksb + kn[:, None] * kst + kvhead * ksh + d[None, :],
                mask=kn[:, None] < NK, other=0.)
    if COMPUTE_DK:
        v = tl.load(V + batch * vsb + kn[:, None] * vst + kvhead * vsh + d[None, :],
                    mask=kn[:, None] < NK, other=0.)
    kdoc = tl.load(KDOC + batch * kdsb + kn * kdst, mask=kn < NK, other=-2)
    if SPARSE_STRIDE > 0:
        kend = SPARSE_PHASE + kn * SPARSE_STRIDE - 1
        kval = tl.full((32,), 1, tl.int32)
    else:
        kend = tl.load(KEND + batch * kesb + kn * kest, mask=kn < NK, other=-2)
        kval = tl.load(KVALID + batch * kvsb + kn * kvst, mask=kn < NK, other=0)
    acc = tl.zeros((32, 64), tl.float32)
    for query_start in tl.range(segment * CHUNK, tl.minimum((segment + 1) * CHUNK, NQ),
                                32, num_stages=1):
        m = query_start + tl.arange(0, 32)
        qdoc = tl.load(QDOC + batch * qdsb + m * qdst, mask=m < NQ, other=-1)
        if SPARSE_STRIDE > 0:
            qpos = m
        else:
            qpos = tl.load(QPOS + batch * qpsb + m * qpst, mask=m < NQ, other=-1)
        valid = _visible(m, kn, qdoc, qpos, kdoc, kend, kval, NQ, NK, True)
        # Only the exact mask can justify skipping a tile, including packed EOS.
        if tl.sum(valid.to(tl.int32)) > 0:
            q = tl.load(Q + batch * qsb + m[:, None] * qst + head * qsh + d[None, :],
                        mask=m[:, None] < NQ, other=0.)
            lse = tl.load(LSE + batch * lsb + m * lst + head * lsh,
                          mask=m < NQ, other=0.)
            upstream = tl.load(DO + batch * dosb + m[:, None] * dost + head * dosh
                               + d[None, :] * dosd, mask=m[:, None] < NQ,
                               other=0.).to(tl.bfloat16)
            scores = tl.dot(k, tl.trans(q)) * SCALE_LOG2
            p = tl.where(tl.trans(valid), tl.exp2(scores - lse[None, :]), 0.)
            if COMPUTE_DK:
                delta = tl.load(DELTA + (batch * NQ + m) * HQ + head,
                                mask=m < NQ, other=0.)
                dp = tl.dot(v, tl.trans(upstream))
                ds = p * (dp - delta[None, :])
                acc = tl.dot(ds.to(tl.bfloat16), q, acc)
            else:
                acc = tl.dot(p.to(tl.bfloat16), upstream, acc)
    destination = (((bh * SEGMENTS + segment) * NK + kn[:, None]) * 64 + d[None, :])
    tl.store(PARTIAL + destination, acc, mask=kn[:, None] < NK)


@triton.jit
def _reduce(
    PK, PV, DK, DV,
    dksb: tl.constexpr, dkst: tl.constexpr, dksh: tl.constexpr, dksd: tl.constexpr,
    dvsb: tl.constexpr, dvst: tl.constexpr, dvsh: tl.constexpr, dvsd: tl.constexpr,
    ELEMENTS: tl.constexpr, NK: tl.constexpr, HK: tl.constexpr, GROUP: tl.constexpr,
    SEGMENTS: tl.constexpr, SCALE: tl.constexpr, ADD_DKDV: tl.constexpr,
):
    offset = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offset < ELEMENTS
    d = offset % 64
    head = (offset // 64) % HK
    key = (offset // (64 * HK)) % NK
    batch = offset // (64 * HK * NK)
    source = ((batch * HK * GROUP + head * GROUP) * SEGMENTS * NK + key) * 64 + d
    dkacc = tl.full((256,), 0., tl.float32)
    dvacc = tl.full((256,), 0., tl.float32)
    # Serial group-major, segment-minor accumulation is deterministic.
    for group in tl.range(0, GROUP, num_stages=1):
        for segment in tl.range(0, SEGMENTS, num_stages=1):
            address = source + (group * SEGMENTS + segment) * NK * 64
            dkacc += tl.load(PK + address, mask=mask, other=0.)
            dvacc += tl.load(PV + address, mask=mask, other=0.)
    dkacc *= SCALE
    kp = DK + batch * dksb + key * dkst + head * dksh + d * dksd
    vp = DV + batch * dvsb + key * dvst + head * dvsh + d * dvsd
    if ADD_DKDV:
        dkacc += tl.load(kp, mask=mask, other=0.)
        dvacc += tl.load(vp, mask=mask, other=0.)
    tl.store(kp, dkacc, mask=mask)
    tl.store(vp, dvacc, mask=mask)


def backward(dout, cache, *, dk=None, dv=None, add_dkdv=False):
    """Return FP32 dK,dV for a BF16 GLOBAL ``TiledAttentionCache`` on HIP.

    ``dout`` stays FP32 in storage; only dot operands are cast to BF16. Supplied
    destinations may have arbitrary non-overlapping strides (including the last
    dimension) and are returned by identity. Addition reads each destination
    once, after the complete deterministic reduction; omitted destinations start
    at zero. Overlapping/aliased destinations or inputs are not supported.

    Two private [B,HQ,min(16,ceil(NQ/512)),NK,64] FP32 arrays bound scratch's
    segment factor, not its overall size. The numerical association differs from
    the existing serial GQA/query accumulation, so equality is not bitwise with
    that reference. Repeated candidate calls are deterministic.
    """
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('split-pass dense KV backward requires AMD HIP; CUDA is unsupported')
    q, k, v = cache.q, cache.k, cache.v
    _validate(q, k, v, cache.scale)
    if str(q.dtype) != 'bfloat16':
        raise TypeError('split-pass dense KV backward supports BF16 Q/K/V only; use existing fallback')
    if not cache.global_mask:
        raise ValueError('split-pass dense KV backward requires a GLOBAL (including sparse) cache')
    if not isinstance(dout, cp.ndarray) or dout.shape != q.shape or str(dout.dtype) != 'float32':
        raise ValueError('dense KV backward requires FP32 dout with Q shape')
    for name, destination, source in (('dk', dk, k), ('dv', dv, v)):
        if destination is not None and (not isinstance(destination, cp.ndarray)
                or destination.shape != source.shape or str(destination.dtype) != 'float32'):
            raise ValueError(f'provided {name} must be matching FP32 storage')
    if dk is None:
        dk = cp.zeros(k.shape, dtype=cp.float32)
    if dv is None:
        dv = cp.zeros(v.shape, dtype=cp.float32)
    batch, nq, hq, _ = map(int, q.shape)
    nk, hk = map(int, k.shape[1:3])
    if not batch or not nq or not nk:
        if not add_dkdv:
            dk.fill(0.)
            dv.fill(0.)
        return dk, dv
    segments = min(16, triton.cdiv(nq, 512))
    chunk = triton.cdiv(nq, segments * 32) * 32
    delta = cp.empty(q.shape[:3], dtype=cp.float32)
    partial_k = cp.empty((batch, hq, segments, nk, 64), dtype=cp.float32)
    partial_v = cp.empty_like(partial_k)
    qs, ks, vs, os, dos, ls = map(_strides, (q, k, v, cache.output, dout, cache.lse))
    qds, qps, kds, kes, kvs = map(_strides, cache.metadata)
    _delta[(triton.cdiv(nq, 32), batch * hq)](
        _ptr(cache.output), _ptr(dout), _ptr(delta), *os, *dos,
        NQ=nq, HQ=hq, num_warps=4, num_stages=1,
    )
    for compute_dk, partial in ((True, partial_k), (False, partial_v)):
        _kv_pass[(triton.cdiv(nk, 32), batch * hq, segments)](
            *map(_ptr, (q, k, v, *cache.metadata, dout, cache.lse, delta, partial)),
            *qs[:3], *ks[:3], *vs[:3], *dos, *ls[:3],
            *qds[:2], *qps[:2], *kds[:2], *kes[:2], *kvs[:2],
            NQ=nq, NK=nk, HQ=hq, GROUP=hq // hk, SEGMENTS=segments, CHUNK=chunk,
            SCALE_LOG2=cache.scale * _LOG2E, SPARSE_STRIDE=cache.sparse_stride,
            SPARSE_PHASE=cache.sparse_phase, COMPUTE_DK=compute_dk,
            num_warps=4, num_stages=1,
        )
    elements = batch * nk * hk * 64
    _reduce[(triton.cdiv(elements, 256),)](
        *map(_ptr, (partial_k, partial_v, dk, dv)), *_strides(dk), *_strides(dv),
        ELEMENTS=elements, NK=nk, HK=hk, GROUP=hq // hk, SEGMENTS=segments,
        SCALE=cache.scale, ADD_DKDV=bool(add_dkdv), num_warps=4, num_stages=1,
    )
    return dk, dv
