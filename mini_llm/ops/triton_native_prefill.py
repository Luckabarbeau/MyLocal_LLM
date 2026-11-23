"""Source-arithmetic INT8 prefill primitives (not an orchestration policy).

Reference: Strata 712fec6b4297, src/prefill/moe_fused_iq.cu,
quant_act_nat_kernel, convert, and native_kernel's normal non-F16/non-T75 path.
No Strata library, build, source-file access, or imports are required at runtime.

Inputs are contiguous CuPy CUDA arrays; launches use the current CuPy stream.
K must be divisible by 64 (and by the weight format's block size for linear).
Only finite activations/weights are supported. Arithmetic uses FP32 throughout,
RN dynamic division and RN-even conversion, with sequential 32-value scale
epilogues and CUDA's constant-reciprocal lowering for stored scales.
CUDA __expf is reproduced with mul.rn + ex2.approx.ftz, not libdevice exp.
This is an explicit long-chunk slice, including when used at 1K tokens; it
is not Strata's short-prompt policy. This does not reproduce optional F16,
Turing Q2_0, HIP arithmetic policies, or CUDA's physical warp/copy schedule.
Linear and fused gate/up use 64 feature columns and 64/128 routed-row tiles.

The physical layout is [rows, K/64, 80] uint8: 64 signed code bytes,
two FP32 scales, two FP32 code sums. Separate contiguous codes/scales are
also returned for a simple grouped-linear API. Thus storage is redundant, not
a zero-copy view of the original layout. H's otherwise unused sums are written
here too (the normal native_kernel H epilogue leaves them unspecified).

Portions derived from Strata, MIT License:
Copyright (c) 2026 Niko1221 and the Strata contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from dataclasses import dataclass

import triton
import triton.language as tl

from .triton_prefill_experts import _ptr


# (format ID, values per GGUF block, bytes per GGUF block)
_FORMATS = {
    'IQ2_S': (0, 256, 82), 'IQ3_S': (1, 256, 110),
    'IQ3_XXS': (2, 256, 98), 'IQ4_NL': (3, 32, 18),
    'IQ4_XS': (4, 256, 136), 'Q2_0': (5, 64, 18),
}
_TABLES = {}


@dataclass(frozen=True)
class NativeQuantized:
    """codes int8 [R,K], scales float32 [R,K/32], packed uint8 [R,K/64,80]."""

    codes: object
    scales: object
    packed: object


def _codebooks(cp):
    # Like the existing prefill codebooks, but device-keyed and independent of
    # MINI_LLM_BACKEND (which can still be numpy in a CuPy caller).
    device = cp.cuda.runtime.getDevice()
    if device not in _TABLES:
        from .gguf_codebooks import (
            iq2s_grid, iq3s_grid, iq3xxs_grid, ksigns_iq2xs, kvalues_iq4nl,
        )
        _TABLES[device] = tuple(cp.asarray(t.reshape(-1)) for t in (
            iq2s_grid, iq3s_grid, iq3xxs_grid, ksigns_iq2xs, kvalues_iq4nl))
    return tuple(_ptr(a) for a in _TABLES[device])


@triton.jit
def _div_rn(a, b):
    # tl.div_rn exists in current Triton, but PTX makes the contract explicit.
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                     args=[a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _expf(x):
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0f3fb8aa3b; ex2.approx.ftz.f32 $0, t; }",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _quantize(X, U, C, S, P, K: tl.constexpr, SWIGLU: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    i = tl.arange(0, 32)
    v = tl.load(X + row.to(tl.int64)*K + block*32 + i)
    if SWIGLU:
        u = tl.load(U + row.to(tl.int64)*K + block*32 + i)
        v = _div_rn(v, 1.0 + _expf(-v)) * u
    am = tl.max(tl.abs(v), 0)
    inv = tl.where(am > 0.0, _div_rn(127.0, am), 0.0)
    code = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;", constraints="=r,f", args=[v * inv],
        dtype=tl.int32, is_pure=True, pack=1)
    # CUDA's compiled constant division uses the rounded FP32 reciprocal.
    # Explicit div.rn differs by one ULP for some scales (covered by oracle).
    scale = am * 0.007874015718698502
    tl.store(C + row.to(tl.int64)*K + block*32 + i, code.to(tl.int8))
    tl.store(S + row.to(tl.int64)*(K//32) + block, scale)
    physical = P + (row.to(tl.int64)*(K//64) + block//2)*80
    tl.store(physical + (block % 2)*32 + i, code.to(tl.uint8))
    floats = tl.cast(physical + 64, tl.pointer_type(tl.float32))
    tl.store(floats + block % 2, scale)
    tl.store(floats + 2 + block % 2, tl.sum(code, 0).to(tl.float32))


@triton.jit
def _decode(W, G2S, G3S, G3X, KSG, KV, row, col, valid,
            K: tl.constexpr, QK: tl.constexpr, QB: tl.constexpr,
            FMT: tl.constexpr):
    # GGUF byte indexing follows triton_packed_prefill._weight_tile, but codes
    # and FP32 scales stay separate: no scaled-weight FP16 conversion.
    base = W + row.to(tl.int64)*(K//QK)*QB + (col//QK)*QB
    bits = (tl.load(base, valid, 0).to(tl.uint16) |
            (tl.load(base+1, valid, 0).to(tl.uint16) << 8))
    d = bits.to(tl.float16, bitcast=True).to(tl.float32)
    if FMT == 5:
        within = col % 64
        code = ((tl.load(base+2+within//4, valid, 0) >> (2*(within % 4))) & 3).to(tl.int32)-1
        scale = d
    elif FMT == 3:
        within = col % 32
        index = (tl.load(base+2+within % 16, valid, 0) >> (4*(within//16))) & 15
        code = tl.load(KV+index).to(tl.int32)
        scale = d
    elif FMT == 4:
        group = (col % 256)//32
        within = col % 32
        high = (tl.load(base+2, valid, 0).to(tl.uint32) |
                (tl.load(base+3, valid, 0).to(tl.uint32) << 8))
        factor = ((tl.load(base+4+group//2, valid, 0) >> (4*(group % 2))) & 15) | (((high >> (2*group)) & 3) << 4)
        index = (tl.load(base+8+16*group+within % 16, valid, 0) >> (4*(within//16))) & 15
        code = tl.load(KV+index).to(tl.int32)
        scale = d * (factor.to(tl.int32)-32).to(tl.float32)
    else:
        group = (col % 256)//32
        within = col % 32
        lane = within//8
        sub = within % 8
        if FMT == 0:
            index = tl.load(base+2+4*group+lane, valid, 0).to(tl.uint32) | (((tl.load(base+66+group, valid, 0).to(tl.uint32) >> (2*lane)) & 3) << 8)
            code = tl.load(G2S+index*8+sub).to(tl.int32)
            factor = (tl.load(base+74+group, valid, 0) >> (4*(lane//2))) & 15
            scale = (d * (2*factor.to(tl.int32)+1).to(tl.float32)) * 0.125
            signs = tl.load(base+34+4*group+lane, valid, 0)
        elif FMT == 1:
            index = tl.load(base+2+8*group+2*lane+sub//4, valid, 0).to(tl.uint32) | (((tl.load(base+66+group, valid, 0).to(tl.uint32) >> (2*lane+sub//4)) & 1) << 8)
            code = tl.load(G3S+index*4+sub % 4).to(tl.int32)
            signs = tl.load(base+74+4*group+lane, valid, 0)
            factor = (tl.load(base+106+group//2, valid, 0) >> (4*(group % 2))) & 15
            scale = d * (1+2*factor.to(tl.int32)).to(tl.float32)
        else:
            index = tl.load(base+2+8*group+2*lane+sub//4, valid, 0).to(tl.uint32)
            code = tl.load(G3X+index*4+sub % 4).to(tl.int32)
            aux = (tl.load(base+66+4*group, valid, 0).to(tl.uint32) |
                   (tl.load(base+67+4*group, valid, 0).to(tl.uint32) << 8) |
                   (tl.load(base+68+4*group, valid, 0).to(tl.uint32) << 16) |
                   (tl.load(base+69+4*group, valid, 0).to(tl.uint32) << 24))
            signs = tl.load(KSG+((aux >> (7*lane.to(tl.uint32))) & 127))
            scale = (d * (2*(aux >> 28)+1).to(tl.float32)) * 0.25
        code = tl.where(((signs.to(tl.uint32) >> sub.to(tl.uint32)) & 1) != 0, -code, code)
    return code.to(tl.int8), scale


@triton.jit
def _dotblock(a, dx, W, G2S, G3S, G3X, KSG, KV, n, col, acc,
              N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
              QB: tl.constexpr, FMT: tl.constexpr):
    w, sw = _decode(W, G2S, G3S, G3X, KSG, KV,
                    n[None, :], col[:, None], n[None, :] < N,
                    K, QK, QB, FMT)
    kk = tl.arange(0, 32)
    sw0 = tl.sum(tl.where(kk[:, None] == 0, sw, 0.0), 0)
    if FMT == 0:
        # Triton int8 dot requires K >= 32; zero-pad each per16 dot.
        a0 = tl.where(kk[None, :] < 16, a, 0).to(tl.int8)
        a1 = tl.where(kk[None, :] >= 16, a, 0).to(tl.int8)
        d0 = tl.dot(a0, w, out_dtype=tl.int32).to(tl.float32)
        d1 = tl.dot(a1, w, out_dtype=tl.int32).to(tl.float32)
        sw1 = tl.sum(tl.where(kk[:, None] == 16, sw, 0.0), 0)
        v = tl.fma(sw1[None, :], d1, sw0[None, :]*d0)
        acc = tl.fma(dx[:, None], v, acc)
    else:
        dot = tl.dot(a, w, out_dtype=tl.int32).to(tl.float32)
        p = sw0[None, :] * dx[:, None]
        acc = tl.fma(p, dot, acc)
    return acc


@triton.jit
def _linear(C, S, W, CNT, OFF, Y, G2S, G3S, G3X, KSG, KV,
            STRIDE: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            QK: tl.constexpr, QB: tl.constexpr, FMT: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    expert = tl.program_id(2)
    start = tl.load(OFF+expert)
    count = tl.load(CNT+expert)
    m = tl.program_id(0)*BM + tl.arange(0, BM)
    n = tl.program_id(1)*BN + tl.arange(0, BN)
    kk = tl.arange(0, 32)
    if tl.program_id(0)*BM < count:
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(K//32):
            col = block*32 + kk
            a = tl.load(C+(start+m[:, None]).to(tl.int64)*K+col[None, :],
                        m[:, None] < count, 0)
            dx = tl.load(S+(start+m).to(tl.int64)*(K//32)+block, m < count, 0)
            acc = _dotblock(a, dx, W+expert.to(tl.int64)*STRIDE,
                            G2S, G3S, G3X, KSG, KV, n, col, acc, N, K, QK, QB, FMT)
        tl.store(Y+(start+m[:, None]).to(tl.int64)*N+n[None, :], acc,
                 (m[:, None] < count) & (n[None, :] < N))


@triton.jit
def _gateup_quantize(C, S, G, U, CNT, OFF, HC, HS, HP, G2S, G3S, G3X, KSG, KV,
                     GSTRIDE: tl.constexpr, USTRIDE: tl.constexpr,
                     N: tl.constexpr, K: tl.constexpr,
                     GQK: tl.constexpr, GQB: tl.constexpr, GFMT: tl.constexpr,
                     UQK: tl.constexpr, UQB: tl.constexpr, UFMT: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
    # native_kernel's normal INT8 GU graph: keep both products in registers,
    # then reduce/quantize each 32-feature half without materializing G or U.
    expert = tl.program_id(2)
    start = tl.load(OFF+expert)
    count = tl.load(CNT+expert)
    m = tl.program_id(0)*BM + tl.arange(0, BM)
    feature = tl.arange(0, BN)
    n = tl.program_id(1)*BN + feature
    kk = tl.arange(0, 32)
    if tl.program_id(0)*BM < count:
        gate = tl.zeros((BM, BN), tl.float32)
        up = tl.zeros((BM, BN), tl.float32)
        for block in range(K//32):
            col = block*32 + kk
            a = tl.load(C+(start+m[:, None]).to(tl.int64)*K+col[None, :],
                        m[:, None] < count, 0)
            dx = tl.load(S+(start+m).to(tl.int64)*(K//32)+block, m < count, 0)
            gate = _dotblock(a, dx, G+expert.to(tl.int64)*GSTRIDE,
                             G2S, G3S, G3X, KSG, KV, n, col, gate, N, K, GQK, GQB, GFMT)
            up = _dotblock(a, dx, U+expert.to(tl.int64)*USTRIDE,
                           G2S, G3S, G3X, KSG, KV, n, col, up, N, K, UQK, UQB, UFMT)
        h = _div_rn(gate, 1.0 + _expf(-gate)) * up
        h32 = tl.reshape(h, (BM, BN//32, 32))
        am = tl.max(tl.abs(h32), 2)
        inv = tl.where(am > 0.0, _div_rn(127.0, am), 0.0)
        code32 = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;", constraints="=r,f",
            args=[h32 * inv[:, :, None]], dtype=tl.int32, is_pure=True, pack=1)
        code = tl.reshape(code32, (BM, BN))
        scale = am * 0.007874015718698502
        row = (start+m).to(tl.int64)
        valid = m < count
        tl.store(HC+row[:, None]*N+n[None, :], code.to(tl.int8), valid[:, None])
        half = tl.arange(0, BN//32)
        group = tl.program_id(1)*(BN//32)+half
        tl.store(HS+row[:, None]*(N//32)+group[None, :], scale, valid[:, None])
        # BN=64: each program exclusively owns one physical 80-byte H block
        # per routed row. Both halves' scales and optional code sums are set.
        physical = HP+(row*(N//64)+tl.program_id(1))*80
        tl.store(physical[:, None]+feature[None, :], code.to(tl.uint8), valid[:, None])
        floats = tl.cast(physical+64, tl.pointer_type(tl.float32))
        tl.store(floats[:, None]+half[None, :], scale, valid[:, None])
        tl.store(floats[:, None]+2+half[None, :],
                 tl.sum(code32, 2).to(tl.float32), valid[:, None])


def _array(cp, a, dtype, ndim, name):
    if not isinstance(a, cp.ndarray) or a.dtype != cp.dtype(dtype):
        raise TypeError(f'{name} must be a CuPy {dtype} array')
    if a.ndim != ndim or not a.flags.c_contiguous:
        raise ValueError(f'{name} must be contiguous with {ndim} dimensions')
    if a.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f'{name} must be on the current CUDA device')


def _launch(kernel, grid, args, constants, num_warps, cp):
    # JITFunction rejects stream= in Triton 3.8; CompiledKernel supports it.
    # Preserve CuPy stream ordering without mutating Triton's global driver.
    compiled = kernel.warmup(*args, **constants, grid=grid,
                             num_warps=num_warps, enable_fp_fusion=False)
    compiled[grid](*args, *constants.values(), stream=cp.cuda.get_current_stream().ptr)


def _quantized(x, u=None):
    import cupy as cp
    _array(cp, x, 'float32', 2, 'x')
    rows, k = x.shape
    if k <= 0 or k % 64:
        raise ValueError('K must be positive and divisible by 64')
    if u is not None:
        _array(cp, u, 'float32', 2, 'u')
        if u.shape != x.shape:
            raise ValueError('gate and up shapes must agree')
    result = NativeQuantized(cp.empty((rows, k), cp.int8),
                             cp.empty((rows, k//32), cp.float32),
                             cp.empty((rows, k//64, 80), cp.uint8))
    if rows:
        _launch(_quantize, (rows, k//32, 1),
                (_ptr(x), _ptr(x if u is None else u), _ptr(result.codes),
                 _ptr(result.scales), _ptr(result.packed)),
                dict(K=k, SWIGLU=u is not None), 1, cp)
    return result


def quantize_native(x):
    """Quantize grouped float32 [R,K] per32 with RN-even codes and RN scales."""
    return _quantized(x)


def native_swiglu_quantize(g, u):
    """Quantize (g / (1 + __expf(-g))) * u without FP16 intermediates."""
    return _quantized(g, u)


def native_linear(quantized, weights, counts, offsets, *, N, K, kind,
                  weight_stride, max_rows):
    """Return float32 [R,N] from contiguous routed rows and packed wave slots.

    counts: device int32 [E]; offsets: device int32 [E+1]. Expert/slot e owns
    rows offsets[e]:offsets[e]+counts[e] and weights at byte e*weight_stride.
    weights is a flat uint8 array (a staged wave subview is accepted). Rows of
    each weight matrix are packed with stride (K/QK)*QB, without row padding.
    max_rows is a host upper bound on every count, avoiding device-host sync.

    Caller invariants (not read back): offsets[0]==0, offsets[-1]==R,
    offsets[e+1]-offsets[e]==counts[e]>=0, counts[e]<=max_rows. For a sparse
    wave, pass the corresponding grouped row slice with rebased offsets.
    No routing, gather, scatter, weighting, or expert combination is performed.
    """
    import cupy as cp
    if kind not in _FORMATS:
        raise ValueError(f'unsupported native format: {kind}')
    fmt, qk, qb = _FORMATS[kind]
    if N <= 0 or K <= 0 or K % 64 or K % qk:
        raise ValueError('N must be positive; K must be a multiple of 64 and QK')
    if max_rows < 0 or weight_stride < N*(K//qk)*qb:
        raise ValueError('invalid max_rows or undersized weight_stride')
    if not isinstance(quantized, NativeQuantized):
        raise TypeError('quantized must be NativeQuantized')
    _array(cp, quantized.codes, 'int8', 2, 'codes')
    _array(cp, quantized.scales, 'float32', 2, 'scales')
    rows = quantized.codes.shape[0]
    if quantized.codes.shape[1] != K or quantized.scales.shape != (rows, K//32):
        raise ValueError('quantized dimensions disagree with K')
    _array(cp, weights, 'uint8', 1, 'weights')
    _array(cp, counts, 'int32', 1, 'counts')
    _array(cp, offsets, 'int32', 1, 'offsets')
    experts = counts.size
    if offsets.size != experts+1:
        raise ValueError('offsets must have E+1 entries')
    needed = (experts-1)*weight_stride + N*(K//qk)*qb if experts else 0
    if weights.size < needed:
        raise ValueError('weights buffer is too small')
    if rows and (not experts or not max_rows):
        raise ValueError('nonempty rows require experts and positive max_rows')
    out = cp.empty((rows, N), cp.float32)
    if rows:
        _launch(_linear, (triton.cdiv(max_rows, 16), triton.cdiv(N, 32), experts),
                (_ptr(quantized.codes), _ptr(quantized.scales), _ptr(weights),
                 _ptr(counts), _ptr(offsets), _ptr(out), *_codebooks(cp)),
                dict(STRIDE=weight_stride, N=N, K=K, QK=qk, QB=qb, FMT=fmt), 4, cp)
    return out
