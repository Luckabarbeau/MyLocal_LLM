"""Explicit Triton prompt-expert policy for Qwen4Exp chunked prefill.

Every active routed expert executes on tensor cores.  Packed GGUF expert bytes
stream host->device through a pinned multi-slot ring (GIL-free ctypes.memmove
fill, asynchronous H2D on a copy stream, prefetch one wave ahead) and dequantize
*on device* into FP16 tiles whose per-block scales are baked in.  The GEMMs are
plain FP16 tl.dot products accumulated in FP32; activations are FP16.

This is a distinct arithmetic policy, like the earlier ``stream`` policy: FP16
products with FP32 accumulation are not bitwise equivalent to the mixed
CPU/GPU decode arithmetic and no such equivalence is claimed.  Routing is
unchanged and every selected expert executes; accumulation visits expert IDs in
ascending order per token, matching the grouped reference's add order.

Wave structure: the expert-ID space is partitioned into ascending waves of at
most MINI_LLM_TRITON_WAVE experts.  Within a wave, token rows are bucketed by
expert (rows ascending per expert); tiles of MINI_LLM_TRITON_BM rows multiply
against that expert's FP16 weights.  A combine kernel adds each token's wave
contributions in ascending expert order.

MINI_LLM_TRITON_ARITHMETIC=fp16 (default) preserves that policy.
The opt-in native_int8 policy uses native per32 activation quantization and
integer products, then combines all waves in original router-rank order.
Its host grouping and pinned ring are intermediate scaffolding, not a
reproduction of Strata's staging/scheduling. Routes already include moe.scale;
shared-expert addition remains the caller's responsibility outside this slice.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os

import numpy as np

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from mini_llm.backend import BACKEND_NAME, asnumpy, xp

# Format IDs and byte layouts match mini_llm.ops.iq_packed_gpu.lane_weight.
_FORMAT_ID = {name: i for i, name in enumerate(
    ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0"))}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class _Ptr:
    """CuPy-array -> Triton pointer bridge (kernel args stay explicit)."""

    def __init__(self, array, dtype, byte_offset=0):
        # Keep the array referenced: _Ptr stores only the address, and dropping
        # the array would let the caching allocator recycle live kernel input.
        self._array = array
        self._ptr = int(array.data.ptr) + int(byte_offset)
        self.dtype = dtype

    def data_ptr(self):
        return self._ptr


def _ptr(array, element_offset=0):
    name = str(array.dtype)
    dtype = {
        "float32": tl.float32, "float16": tl.float16, "uint8": tl.uint8,
        "int32": tl.int32, "int8": tl.int8,
    }.get(name)
    if dtype is None:
        raise TypeError(f"unsupported Triton expert dtype: {name}")
    return _Ptr(array, dtype, int(element_offset) * int(array.dtype.itemsize))


_CODEBOOKS = None


def _codebooks():
    # Keep the CuPy arrays referenced by the returned pointers: _Ptr holds the
    # integer address only, and without this tuple the caching allocator would
    # recycle the tables while kernels still read them.
    global _CODEBOOKS
    if _CODEBOOKS is None:
        from .gguf_codebooks import (iq2s_grid, iq3s_grid, iq3xxs_grid,
                                     ksigns_iq2xs, kvalues_iq4nl)
        arrays = tuple(xp.asarray(np.ascontiguousarray(t.reshape(-1)))
                       for t in (iq2s_grid, iq3s_grid, iq3xxs_grid,
                                 ksigns_iq2xs, kvalues_iq4nl))
        _CODEBOOKS = (tuple(_ptr(a) for a in arrays), arrays)
    return _CODEBOOKS[0]


@triton.jit
def _dequant_kernel(W, OUT16, G2S, G3S, G3X, KSG, KV,
                    stride, N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
                    QB: tl.constexpr, FMT: tl.constexpr, TILE: tl.constexpr = 32):
    n = tl.program_id(0)
    kb = tl.program_id(1)
    slot = tl.program_id(2)
    rowbytes: tl.constexpr = (K // QK) * QB
    j = kb * TILE + tl.arange(0, TILE)
    B = W + slot.to(tl.int64) * stride + n.to(tl.int64) * rowbytes + (j // QK) * QB
    d = (tl.load(B).to(tl.uint16)
         | (tl.load(B + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    if FMT == 5:  # Q2_0: adjacent two-bit lanes.
        jj = j % 64
        v = (tl.load(B + 2 + jj // 4) >> (2 * (jj % 4))) & 3
        w = d * (v.to(tl.float32) - 1.0)
    elif FMT == 3:  # IQ4_NL: low half then high half nibbles.
        jj = j % 32
        nib = (tl.load(B + 2 + jj % 16) >> (4 * (jj // 16))) & 15
        w = d * tl.load(KV + nib).to(tl.float32)
    elif FMT == 4:  # IQ4_XS: signed six-bit scale per 32.
        g0 = (j % 256) // 32
        within = j % 32
        d2 = tl.load(B + 2).to(tl.uint32) | (tl.load(B + 3).to(tl.uint32) << 8)
        scale = ((tl.load(B + 4 + g0 // 2) >> (4 * (g0 % 2))) & 15) \
            | (((d2 >> (2 * g0)) & 3) << 4)
        nib = (tl.load(B + 8 + 16 * g0 + within % 16) >> (4 * (within // 16))) & 15
        w = d * (scale.to(tl.float32) - 32.0) * tl.load(KV + nib).to(tl.float32)
    else:
        g0 = (j % 256) // 32
        within = j % 32
        lane = within // 8
        sub = within % 8
        if FMT == 0:  # IQ2_S: 10-bit index, independent eight-bit signs.
            index = tl.load(B + 2 + 4 * g0 + lane).to(tl.uint32) \
                | ((tl.load(B + 66 + g0).to(tl.uint32) >> (2 * lane)) & 3) << 8
            value = tl.load(G2S + index * 8 + sub).to(tl.float32)
            scale = (tl.load(B + 74 + g0) >> (4 * (lane // 2))) & 15
            v = d * (0.5 + scale.to(tl.float32)) * 0.25 * value
            signs = tl.load(B + 34 + 4 * g0 + lane)
        elif FMT == 1:  # IQ3_S: two 9-bit grids per eight values.
            pos = 2 + 8 * g0 + 2 * lane + sub // 4
            index = tl.load(B + pos).to(tl.uint32) \
                | ((tl.load(B + 66 + g0).to(tl.uint32)
                    >> (2 * lane + sub // 4)) & 1) << 8
            value = tl.load(G3S + index * 4 + sub % 4).to(tl.float32)
            signs = tl.load(B + 74 + 4 * g0 + lane)
            scale = (tl.load(B + 106 + g0 // 2) >> (4 * (g0 % 2))) & 15
            v = d * (1.0 + 2.0 * scale.to(tl.float32)) * value
        else:  # IQ3_XXS: seven-bit parity signs and a high scale nibble.
            index = tl.load(B + 2 + 8 * g0 + 2 * lane + sub // 4).to(tl.uint32)
            value = tl.load(G3X + index * 4 + sub % 4).to(tl.float32)
            aux = (tl.load(B + 66 + 4 * g0).to(tl.uint32)
                   | (tl.load(B + 67 + 4 * g0).to(tl.uint32) << 8)
                   | (tl.load(B + 68 + 4 * g0).to(tl.uint32) << 16)
                   | (tl.load(B + 69 + 4 * g0).to(tl.uint32) << 24))
            signs = tl.load(KSG + ((aux >> (7 * lane.to(tl.uint32))) & 127))
            v = d * (0.5 + (aux >> 28).to(tl.float32)) * 0.5 * value
        neg = (signs.to(tl.uint32) >> sub.to(tl.uint32)) & 1
        w = tl.where(neg != 0, -v, v)
    tl.store(OUT16 + slot.to(tl.int64) * (N * K) + n.to(tl.int64) * K + j,
             w.to(tl.float16))


@triton.jit
def _gateup_kernel(A16, G16, U16, H, TOKROW, TSTART, TCNT, TSLOT,
                   N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(TSLOT + t)
    r0 = tl.load(TSTART + t)
    rc = tl.load(TCNT + t)
    rm = tl.arange(0, BM)
    rn = nb * BN + tl.arange(0, BN)
    mask_m = rm < rc
    tok = tl.load(TOKROW + r0 + rm, mask=mask_m, other=0)
    accg = tl.zeros((BM, BN), tl.float32)
    accu = tl.zeros((BM, BN), tl.float32)
    for kb in range(K // BK):
        rk = kb * BK + tl.arange(0, BK)
        a = tl.load(A16 + tok[:, None].to(tl.int64) * K + rk[None, :],
                    mask=mask_m[:, None], other=0.0)
        gw = tl.load(G16 + (slot.to(tl.int64) * N + rn[None, :]) * K + rk[:, None])
        uw = tl.load(U16 + (slot.to(tl.int64) * N + rn[None, :]) * K + rk[:, None])
        accg = tl.dot(a, gw, accg)
        accu = tl.dot(a, uw, accu)
    h = accg * (1.0 / (1.0 + tl.exp(-accg))) * accu
    tl.store(H + (r0 + rm)[:, None].to(tl.int64) * N + rn[None, :], h,
             mask=mask_m[:, None])


@triton.jit
def _down_kernel(H16, D16, VALUES, TSTART, TCNT, TSLOT, rbase,
                 N: tl.constexpr, K: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(TSLOT + t)
    r0 = tl.load(TSTART + t)
    rc = tl.load(TCNT + t)
    rm = tl.arange(0, BM)
    rn = nb * BN + tl.arange(0, BN)
    mask_m = rm < rc
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(K // BK):
        rk = kb * BK + tl.arange(0, BK)
        a = tl.load(H16 + (r0 - rbase + rm)[:, None].to(tl.int64) * K + rk[None, :],
                    mask=mask_m[:, None], other=0.0)
        dw = tl.load(D16 + (slot.to(tl.int64) * N + rn[None, :]) * K + rk[:, None])
        acc = tl.dot(a, dw, acc)
    tl.store(VALUES + (r0 + rm)[:, None].to(tl.int64) * N + rn[None, :], acc,
             mask=mask_m[:, None])


@triton.jit
def _combine_kernel(VALUES, OUT, TOKROW, TOKW, TOKMASK,
                    D: tl.constexpr, K: tl.constexpr, DC: tl.constexpr):
    t = tl.program_id(0)
    dc = tl.program_id(1) * DC + tl.arange(0, DC)
    acc = tl.load(OUT + t.to(tl.int64) * D + dc)
    for k in range(K):
        m = tl.load(TOKMASK + t * K + k)
        r = tl.load(TOKROW + t * K + k)
        w = tl.load(TOKW + t * K + k)
        acc += tl.where(m != 0, w, 0.0) \
            * tl.load(VALUES + r.to(tl.int64) * D + dc, mask=m != 0, other=0.0)
    tl.store(OUT + t.to(tl.int64) * D + dc, acc)


@triton.jit
def _combine_native_kernel(VALUES, OUT, TOKROW, TOKW,
                           D: tl.constexpr, K: tl.constexpr, DC: tl.constexpr):
    # Strata 712fec6b4297 src/prefill/kernels.cu:2335 moe_combine_kernel,
    # routed contribution only: rank-order fmaf(w[k], Dm[slot[k]], s).
    # MIT, Copyright (c) 2026 Niko1221 and the Strata contributors.
    # Shared expert and pre-applied moe.scale belong to the caller boundary.
    t = tl.program_id(0)
    dc = tl.program_id(1) * DC + tl.arange(0, DC)
    acc = tl.full((DC,), 0.0, tl.float32)
    for k in range(K):
        r = tl.load(TOKROW + t * K + k)
        w = tl.load(TOKW + t * K + k)
        value = tl.load(VALUES + r.to(tl.int64) * D + dc, dc < D, 0.0)
        acc = tl.fma(w, value, acc)
    tl.store(OUT + t.to(tl.int64) * D + dc, acc, dc < D)


class _StageRing:
    """Pinned multi-slot ring; fill on worker threads, H2D on a copy stream."""

    def __init__(self, seg_max, wave_experts, slots):
        self.seg = list(seg_max)
        self.wave_experts = int(wave_experts)
        self.offu = self.wave_experts * self.seg[0]
        self.offd = self.offu + self.wave_experts * self.seg[1]
        self.total = self.offd + self.wave_experts * self.seg[2]
        self.slots = int(slots)
        self.device = xp.empty(self.slots * self.total, xp.uint8)
        self.owner = xp.cuda.alloc_pinned_memory(self.slots * self.total)
        self.host = np.frombuffer(self.owner, np.uint8,
                                  count=self.slots * self.total)
        self.host_addr = int(self.host.ctypes.data)
        self.device_stream = xp.cuda.Stream(non_blocking=True)
        self.h2d = [xp.cuda.Event(disable_timing=True) for _ in range(self.slots)]
        self.filled = [False] * self.slots
        self.consumed = [xp.cuda.Event(disable_timing=True) for _ in range(self.slots)]
        self.device_used = [False] * self.slots

    def memmove(self, slot, src_addr, nbytes, dst_off):
        ctypes.memmove(ctypes.c_void_p(self.host_addr + slot * self.total + dst_off),
                       ctypes.c_void_p(src_addr), int(nbytes))

    def upload(self, slot):
        if self.device_used[slot]:
            self.device_stream.wait_event(self.consumed[slot])
        view = self.host[slot * self.total:(slot + 1) * self.total]
        self.device[slot * self.total:(slot + 1) * self.total].set(view,
                                                                   stream=self.device_stream)
        self.h2d[slot].record(self.device_stream)

    def record_consumed(self, slot, stream):
        self.consumed[slot].record(stream)
        self.device_used[slot] = True


def try_triton_prefill_experts(experts, x, routes, ids):
    """Return [1,T,D] FP32, or None when the policy is ineligible.

    Never mutates executor state before returning None.  After GPU work starts,
    errors drain the fill pool and fence the copy stream before propagating,
    mirroring mini_llm.ops.prefill_experts.
    """
    arithmetic = os.environ.get('MINI_LLM_TRITON_ARITHMETIC', 'fp16')
    if arithmetic not in ('fp16', 'native_int8'):
        raise ValueError('MINI_LLM_TRITON_ARITHMETIC must be fp16 or native_int8')
    native_int8 = arithmetic == 'native_int8'
    from .expert_residency import HeterogeneousExpertExecutor, ExpertKey
    from .quantized_weight import PackedGGUFWeight

    if triton is None or BACKEND_NAME != 'cupy':
        return None
    executor = getattr(experts, 'expert_executor', None)
    if (not isinstance(executor, HeterogeneousExpertExecutor)
            or executor.cache._admission_limit != 0
            or executor.cache.idle_eviction_requests
            or executor.routing_profiler is not None
            or executor.execution_mode != 'mixed'
            or x.ndim != 3 or x.shape[0] != 1 or x.dtype != xp.float32
            or routes.dtype != xp.float32 or ids.dtype != xp.int64):
        return None
    if xp.cuda.runtime.is_hip:
        return None
    tokens, width = x.shape[1:]
    topk = routes.shape[-1]
    if routes.shape != ids.shape or routes.shape[:2] != (1, tokens) \
            or not 1 <= topk <= 16 or width not in (2560,) or tokens < 1 or xp.cuda.runtime.is_hip:
        return None
    host_ids = np.ascontiguousarray(asnumpy(ids[0]))
    if (np.any(host_ids < 0) or np.any(host_ids >= experts.n_experts)
            or np.any(np.diff(np.sort(host_ids, axis=1), axis=1) == 0)):
        raise ValueError('Chunk routes must be unique valid experts per token')
    host_routes = np.ascontiguousarray(asnumpy(routes[0]))

    # Resolve all host source weights before any device work is submitted.
    active = np.nonzero(np.bincount(host_ids.ravel(),
                                    minlength=experts.n_experts))[0]
    weights = {}
    formats = None
    for expert_id in active:
        key = ExpertKey(executor.model_id, experts.layer_id, int(expert_id))
        trio = executor.cpu_source_weights(key)
        if not all(isinstance(w, PackedGGUFWeight) for w in trio):
            return None
        if formats is None:
            formats = tuple(w.quant_type for w in trio)
            if any(t not in _FORMAT_ID for t in formats):
                return None
        elif tuple(w.quant_type for w in trio) != formats:
            return None
        if not all(w.data.flags['C_CONTIGUOUS'] for w in trio):
            return None
        weights[int(expert_id)] = trio
    from .quantized_weight import BLOCK_FORMATS as bf
    gate0, up0, down0 = weights[int(active[0])]
    gate_n, gate_k = int(gate0.shape[0]), int(gate0.shape[1])
    up_n, up_k = int(up0.shape[0]), int(up0.shape[1])
    down_n, down_k = int(down0.shape[0]), int(down0.shape[1])
    if (gate_n != up_n or gate_k != up_k or down_n != width or down_k != gate_n
            or gate_k != width):
        return None

    if (gate_n % 64 or gate_k % 64 or down_n % 256 or down_k % 64
            or any(tuple(w.shape for w in trio) != tuple(w.shape for w in (gate0,up0,down0))
                   for trio in weights.values()) or tokens < 1
            or not all(a.flags.c_contiguous for a in (x,routes,ids))):
        return None
    if native_int8 and any(k % bf[fmt][0] for k, fmt in
                           zip((gate_k, up_k, down_k), formats)):
        return None
    wave_experts = _env_int('MINI_LLM_TRITON_WAVE', 32)
    bm = _env_int('MINI_LLM_TRITON_BM', 64)
    threads = _env_int('MINI_LLM_TRITON_THREADS', 6)
    if not 1 <= wave_experts <= 512 or bm not in (16,32,64,128) or not 1 <= threads <= 64:
        raise ValueError('Invalid Triton prefill wave/tile/worker configuration')

    # --- host grouping ------------------------------------------------------
    flat_ids = host_ids.ravel()
    flat_routes = host_routes.ravel()
    rows = flat_ids.size
    counts = np.bincount(flat_ids, minlength=experts.n_experts)
    starts = np.zeros(experts.n_experts + 1, np.int64)
    np.cumsum(counts, out=starts[1:])
    tok_seq = np.repeat(np.arange(tokens, dtype=np.int64), topk)
    order = np.lexsort((tok_seq, flat_ids))  # expert ascending, token ascending
    rowid = np.empty(rows, np.int32)
    rowid[order] = (starts[flat_ids[order]]
                    + np.arange(rows) - starts[flat_ids[order]]).astype(np.int32)
    token_of_row = tok_seq[order].astype(np.int32)
    order2 = np.lexsort((flat_ids, tok_seq))  # token ascending, expert ascending
    tok_row = (rowid if native_int8 else rowid[order2]).reshape(tokens, topk)
    tok_w = np.ascontiguousarray(
        (flat_routes if native_int8 else flat_routes[order2]).reshape(tokens, topk),
        np.float32)
    tok_id = flat_ids[order2].reshape(tokens, topk)

    active_set = set(int(e) for e in active)
    n_waves = (experts.n_experts + wave_experts - 1) // wave_experts
    tiles_start, tiles_cnt, tiles_slot, wave_tiles = [], [], [], []
    for w in range(n_waves):
        lo, hi = w * wave_experts, (w + 1) * wave_experts
        wave_tile0 = len(tiles_start)
        slot = 0
        for e in range(lo, hi):
            if int(e) not in active_set:
                continue
            base = int(starts[e])
            cnt = int(counts[e])
            for off in range(0, cnt, bm):
                tiles_start.append(base + off)
                tiles_cnt.append(min(bm, cnt - off))
                tiles_slot.append(slot)
            slot += 1
        wave_tiles.append((wave_tile0, len(tiles_start), lo, hi))
    n_active_wave_max = max(1, max(e1 - e0 for (_, _, e0, e1) in wave_tiles))

    seg_max = [0, 0, 0]
    for trio in weights.values():
        for m, w in enumerate(trio):
            seg_max[m] = max(seg_max[m], w.nbytes)
    ring = _StageRing(seg_max, wave_experts, slots=3)
    offg = 0

    # --- device buffers -----------------------------------------------------
    stream = xp.cuda.get_current_stream()
    stream.synchronize()  # complete any preceding native-service dispatch first
    if native_int8:
        from . import triton_native_prefill as native
        token_quantized = native.quantize_native(x[0])
    a16 = None if native_int8 else xp.ascontiguousarray(x[0].astype(xp.float16))
    hbuf = None if native_int8 else xp.empty((rows, gate_n), xp.float16)
    values = xp.empty((rows, width), xp.float32)
    out = xp.zeros((tokens, width), xp.float32)
    fused_dequant = os.environ.get('MINI_LLM_TRITON_FUSED_DEQUANT', '0') == '1'
    g16 = None if native_int8 or fused_dequant else xp.empty((wave_experts, gate_n * gate_k), xp.float16)
    u16 = None if native_int8 or fused_dequant else xp.empty((wave_experts, up_n * up_k), xp.float16)
    d16 = None if native_int8 or fused_dequant else xp.empty((wave_experts, down_n * down_k), xp.float16)
    dev_row = xp.asarray(token_of_row)
    dev_ts = xp.asarray(np.asarray(tiles_start, np.int32))
    dev_tc = xp.asarray(np.asarray(tiles_cnt, np.int32))
    dev_tslot = xp.asarray(np.asarray(tiles_slot, np.int32))
    dev_tokrow = xp.asarray(np.ascontiguousarray(tok_row, np.int32))
    dev_tokw = xp.asarray(tok_w)
    G2S, G3S, G3X, KSG, KV = _codebooks()
    wave_members = []
    for w, (_, _, lo, hi) in enumerate(wave_tiles):
        members, s = [], 0
        for e in range(lo, hi):
            if int(e) in active_set:
                members.append((s, weights[int(e)]))
                s += 1
        wave_members.append(members)
    # References retain mapping lifetimes, not physical page residency.

    def fill_part(wave_index, part):
        slot = wave_index % ring.slots
        if ring.filled[slot]:
            # A pinned slot may only be refilled once its H2D read completed.
            ring.h2d[slot].synchronize()
        members = wave_members[wave_index]
        for i in range(part, len(members), threads):
            s, trio = members[i]
            for m, w in enumerate(trio):
                ring.memmove(slot, int(w.data.ctypes.data), w.nbytes,
                             [offg, ring.offu, ring.offd][m] + s * seg_max[m])


    def submit_wave(wave_index):
        return [pool.submit(fill_part, wave_index, p) for p in range(threads)]

    # Optional orchestration experiment: shared workers avoid thread churn,
    # but production A/B has not established a throughput win over local pools.
    pool = (getattr(executor, '_pool', None)
            if os.environ.get('MINI_LLM_TRITON_REUSE_WORKERS','0') == '1' else None)
    owns_pool = pool is None
    if owns_pool:
        pool = ThreadPoolExecutor(max_workers=threads)
    counters = executor.counters
    pending = []
    try:
        with executor.dispatch_scope(experts.layer_id, tuple(int(e) for e in active)):
            pending = submit_wave(0)
            for w, (tw0, tw1, lo, hi) in enumerate(wave_tiles):
                for f in pending:
                    f.result()
                if w + 1 < n_waves:
                    pending = submit_wave(w + 1)
                slot = w % ring.slots
                ring.filled[slot] = True
                ring.upload(slot)
                stream.wait_event(ring.h2d[slot])
                e_w = len(wave_members[w])
                if not e_w:
                    continue
                # Reference retains materialized weights; fused mode decodes
                # only the matrix tiles consumed by the tensor-core products.
                for m, (stage_off, fp_out, fmt, weights_m) in enumerate((
                        (offg, g16, _FORMAT_ID[formats[0]], 0),
                        (ring.offu, u16, _FORMAT_ID[formats[1]], 1),
                        (ring.offd, d16, _FORMAT_ID[formats[2]], 2))):
                    if native_int8 or fused_dequant:
                        continue
                    qk, qb = bf[formats[m]]
                    N, K = ((gate_n, gate_k) if m == 0 else
                           (up_n, up_k) if m == 1 else (down_n, down_k))
                    tile = 128 if K % 128 == 0 else 32
                    _dequant_kernel[(N, K // tile, e_w)](
                        _ptr(ring.device, slot * ring.total + stage_off),
                        _ptr(fp_out), G2S, G3S, G3X, KSG, KV,
                        seg_max[m], N=N, K=K, QK=qk, QB=qb, FMT=fmt, TILE=tile)
                ntiles = tw1 - tw0
                if ntiles and native_int8:
                    # Slots are compact in active expert order, as fill_part.
                    wave_counts = np.ascontiguousarray(counts[lo:hi][counts[lo:hi] > 0], np.int32)
                    wave_offsets = np.zeros(e_w + 1, np.int32)
                    np.cumsum(wave_counts, out=wave_offsets[1:])
                    row0 = int(starts[lo])
                    row1 = row0 + int(wave_offsets[-1])
                    token_rows = dev_row[row0:row1]
                    aq = native.NativeQuantized(
                        xp.ascontiguousarray(token_quantized.codes[token_rows]),
                        xp.ascontiguousarray(token_quantized.scales[token_rows]), None)
                    dev_counts = xp.asarray(wave_counts)
                    dev_offsets = xp.asarray(wave_offsets)
                    base = slot * ring.total
                    products = []
                    for m, stage_off in enumerate((offg, ring.offu)):
                        staged = ring.device[base+stage_off:base+stage_off+e_w*seg_max[m]]
                        products.append(native.native_linear(
                            aq, staged, dev_counts, dev_offsets, N=gate_n, K=gate_k,
                            kind=formats[m], weight_stride=seg_max[m],
                            max_rows=int(wave_counts.max())))
                    hq = native.native_swiglu_quantize(*products)
                    staged = ring.device[base+ring.offd:base+ring.offd+e_w*seg_max[2]]
                    values[row0:row1] = native.native_linear(
                        hq, staged, dev_counts, dev_offsets, N=width, K=gate_n,
                        kind=formats[2], weight_stride=seg_max[2],
                        max_rows=int(wave_counts.max()))
                elif ntiles and fused_dequant:
                    from .triton_packed_prefill import gateup_packed, down_packed
                    base = slot * ring.total
                    gateup_packed[(ntiles, gate_n // 64)](
                        _ptr(a16), _ptr(ring.device,base), _ptr(ring.device,base+ring.offu),
                        _ptr(hbuf), _ptr(dev_row), _ptr(dev_ts,tw0), _ptr(dev_tc,tw0),
                        _ptr(dev_tslot,tw0), G2S,G3S,G3X,KSG,KV,seg_max[0],seg_max[1],
                        N=gate_n,K=gate_k,BM=bm,BN=64,BK=64,
                        GQK=bf[formats[0]][0],GQB=bf[formats[0]][1],GFMT=_FORMAT_ID[formats[0]],
                        UQK=bf[formats[1]][0],UQB=bf[formats[1]][1],UFMT=_FORMAT_ID[formats[1]])
                    down_packed[(ntiles,width//64)](
                        _ptr(hbuf),_ptr(ring.device,base+ring.offd),_ptr(values),
                        _ptr(dev_ts,tw0),_ptr(dev_tc,tw0),_ptr(dev_tslot,tw0),
                        G2S,G3S,G3X,KSG,KV,seg_max[2],N=width,K=gate_n,BM=bm,BN=64,BK=64,
                        DQK=bf[formats[2]][0],DQB=bf[formats[2]][1],DFMT=_FORMAT_ID[formats[2]])
                elif ntiles:
                    _gateup_kernel[(ntiles, (gate_n + 63) // 64)](
                        _ptr(a16), _ptr(g16), _ptr(u16), _ptr(hbuf),
                        _ptr(dev_row), _ptr(dev_ts, tw0), _ptr(dev_tc, tw0),
                        _ptr(dev_tslot, tw0), N=gate_n, K=gate_k, BM=bm, BN=64,
                        BK=64)
                    _down_kernel[(ntiles, (width + 63) // 64)](
                        _ptr(hbuf), _ptr(d16), _ptr(values),
                        _ptr(dev_ts, tw0), _ptr(dev_tc, tw0),
                        _ptr(dev_tslot, tw0), 0,
                        N=width, K=gate_n, BM=bm, BN=64, BK=64)
                ring.record_consumed(slot, stream)
                if ntiles and not native_int8:
                    mask = xp.asarray(np.ascontiguousarray(
                        (tok_id >= lo) & (tok_id < hi), np.uint8))
                    _combine_kernel[(tokens, width // 256)](
                        _ptr(values), _ptr(out), _ptr(dev_tokrow), _ptr(dev_tokw),
                        _ptr(mask), D=width, K=topk, DC=256)
                if ntiles:
                    counters['prefill_triton_rows'] += int(counts[lo:hi].sum())
                    counters['prefill_triton_waves'] += 1
                counters['prefill_triton_streamed_bytes'] += sum(
                    w.nbytes for _, trio in wave_members[w] for w in trio)
            if native_int8:
                native._launch(
                    _combine_native_kernel, (tokens, triton.cdiv(width, 256), 1),
                    (_ptr(values), _ptr(out), _ptr(dev_tokrow), _ptr(dev_tokw)),
                    dict(D=width, K=topk, DC=256), 4, xp)
                counters['prefill_triton_native_int8_layers'] += 1
            counters['prefill_triton_layers'] += 1
            counters['prefill_triton_experts'] += int(active.size)
            counters['prefill_routed_rows'] += rows
    finally:
        import sys
        error = None
        for f in pending:
            try:
                f.result()
            except Exception as exc:
                if error is None:
                    error = exc
        if owns_pool:
            pool.shutdown(wait=True)
        stream.synchronize()
        ring.device_stream.synchronize()
        if error is not None and sys.exc_info()[0] is None:
            raise error
    return out[None]