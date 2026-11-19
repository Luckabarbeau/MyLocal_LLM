"""Patch0253 standalone IQ3_S x Q8_1 CUDA GEMV; no production dispatch.

W[N,K], K positive and divisible by 256; FP32 x[K] or x[M,K] -> FP32
[N] or [M,N]. Only IQ3_S (110 bytes/256) is supported. Activation packing
uses q8_1_gpu.quantize_q8_1 unchanged (native or IEEE producer arithmetic).
Finite inputs/scales are assumed, as in the pinned Strata dot routine.

Per 32: integer signed codebook dot, integer odd-nibble scale, then
mul_rn(mul_rn(float(weight_half), float(activation_half)), float(integer)).
No per-weight FP32 products, dense weight allocation, FMA, or use of Q8 s.
Reduction: warp lane l accumulates groups l,l+32,... in increasing order;
shuffle-down additions at offsets 16,8,4,2,1, lane zero writes the result.
This tree is explicit, not a claim of whole-engine Strata reduction parity.

Current CuPy stream; asynchronous, no host readback or synchronization.
Caller owns input/output storage and cross-stream ordering. Only compiled
kernels/codebook constants are cached per device. linear allocates Q8 scratch
and (unless supplied) output; linear_q8_1 only allocates output. No retained
weight pointers or precision policy changes. Intended for single/few rows,
not prefill GEMM; speed must be measured before any default integration.
"""
from functools import lru_cache
import operator

import numpy as np

from . import gguf_codebooks as _books
from . import q8_1_gpu as _q8

cp = _q8.cp
BLOCK_BYTES = 110
BLOCK_ELEMENTS = 256


@lru_cache(maxsize=1)
def _source():
    license_text = "\n".join("// " + line for line in _books.__doc__.splitlines())
    values = ",".join(str(int(v)) for v in _books.iq3s_grid.reshape(-1))
    return license_text + '\n#include <cuda_fp16.h>\n' + (
        f'__device__ __constant__ unsigned char grid[{_books.iq3s_grid.size}] = {{{values}}};\n'
    ) + r'''
__device__ __forceinline__ float partial(const unsigned char* w,
                                        const unsigned char* q, int group) {
    const unsigned char* b = w + (long long)(group / 8) * 110;
    const int g = group % 8;
    const signed char* a = (const signed char*)(q + (long long)group * 36 + 4);
    int dot = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int index = b[2 + 8*g + j] | (((b[66 + g] >> j) & 1) << 8);
        int signs = b[74 + 4*g + j/2] >> (4*(j%2));
        unsigned int codes = 0, acts = 0;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int v = grid[4*index + t];
            if ((signs >> t) & 1) v = -v;
            codes |= (unsigned int)(unsigned char)v << (8*t);
            acts |= (unsigned int)(unsigned char)a[4*j+t] << (8*t);
        }
        dot = __dp4a((int)codes, (int)acts, dot);
    }
    dot *= 1 + 2*((b[106 + g/2] >> (4*(g%2))) & 15);
    float wd = __half2float(__ushort_as_half((unsigned short)(b[0] | (b[1]<<8))));
    const unsigned char* h = q + (long long)group*36;
    float xd = __half2float(__ushort_as_half((unsigned short)(h[0] | (h[1]<<8))));
    return __fmul_rn(__fmul_rn(wd, xd), (float)dot);
}
extern "C" __global__ void gemv(const unsigned char* w, const unsigned char* q,
                                float* out, int n, int k) {
    int row = blockIdx.x*4 + threadIdx.x/32;
    int lane = threadIdx.x%32;
    if (row >= n) return;
    const unsigned char* wr = w + (long long)row*(k/256)*110;
    const unsigned char* qr = q + (long long)blockIdx.y*(k/32)*36;
    float sum = 0.f;
    for (int g = lane; g < k/32; g += 32)
        sum = __fadd_rn(sum, partial(wr, qr, g));
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    if (!lane) out[(long long)blockIdx.y*n + row] = sum;
}
// Diagnostic per-32 output, not used by linear or production dispatch.
extern "C" __global__ void partials(const unsigned char* w, const unsigned char* q,
                                    float* out, int n, int k) {
    int g = blockIdx.x*128 + threadIdx.x;
    if (g < n*(k/32))
        out[g] = partial(w + (long long)(g/(k/32))*(k/256)*110, q, g%(k/32));
}
'''


@lru_cache(maxsize=None)
def _module(device):
    return cp.RawModule(code=_source(), options=("--std=c++17", "--fmad=false"),
                        name_expressions=("gemv", "partials"))


def _weight(weight, shape):
    if hasattr(weight, "quant_type"):
        if weight.quant_type != "IQ3_S":
            raise ValueError("only IQ3_S weights are supported")
        if shape is not None:
            raise ValueError("shape is only for raw packed data")
        shape, weight = weight.shape, weight.data
    if shape is None:
        raise ValueError("raw packed data requires shape=(N,K)")
    shape = tuple(operator.index(v) for v in shape)
    if len(shape) != 2 or min(shape) <= 0 or shape[1] % 256:
        raise ValueError("weight shape must be positive [N,K], K divisible by 256")
    if max(shape) > 2147483647:
        raise ValueError("weight dimensions must fit INT32")
    if cp is None or not isinstance(weight, cp.ndarray) or weight.dtype != np.uint8:
        raise TypeError("weight must be CuPy uint8 packed data")
    if weight.device.id != cp.cuda.runtime.getDevice():
        raise ValueError("weight must be on the current device")
    if not weight.flags.c_contiguous or weight.ndim not in (1, 2):
        raise ValueError("weight must be contiguous 1D/2D")
    if weight.size != shape[0]*(shape[1]//256)*110:
        raise ValueError("incorrect IQ3_S packed byte length")
    return weight, shape


def linear_q8_1(packed, weight, shape=None, *, out=None):
    """Multiply already packed Q8_1 rows by IQ3_S W; packed is uint8.

    Accepts DevicePackedGGUFWeight (or the same data/shape/quant_type contract),
    or raw device bytes plus shape=(N,K). Supplied out must be exact FP32 shape,
    contiguous, aligned, on the current device, and disjoint from both inputs.
    Zero activation rows are allowed; dimensions/activation size fit INT32.
    """
    device = _q8._validate(packed, np.uint8, "packed")
    w, (n, k) = _weight(weight, shape)
    if packed.shape[-1] != (k//32)*36 or packed.size//36*32 > 2147483647:
        raise ValueError("packed Q8_1 row length must match weight K; size <= INT32_MAX")
    result = _q8._output(out, packed.shape[:-1] + (n,), np.float32, packed)
    if result.nbytes and result.data.ptr < w.data.ptr+w.nbytes and w.data.ptr < result.data.ptr+result.nbytes:
        raise ValueError("out must not overlap weight")
    if result.size:
        m = 1 if packed.ndim == 1 else packed.shape[0]
        if m > 65535:
            raise ValueError("at most 65535 activation rows")
        _module(device).get_function("gemv")(((n+3)//4, m), (128,),
                                             (w, packed, result, np.int32(n), np.int32(k)))
    return result


def linear(x, weight, shape=None, *, out=None, arithmetic="native"):
    """FP32 x[K] / x[M,K] @ packed IQ3_S W[N,K].T -> FP32.

    arithmetic selects the existing Q8_1 producer, not the dot/reduction graph.
    Input is quantized once per call and reused for every weight row.
    """
    _q8._validate(x, np.float32, "x")
    w, dims = _weight(weight, shape)
    if x.shape[-1] != dims[1]:
        raise ValueError("x K must match weight K")
    result = _q8._output(out, x.shape[:-1] + (dims[0],), np.float32, x)
    if result.nbytes and result.data.ptr < w.data.ptr+w.nbytes and w.data.ptr < result.data.ptr+result.nbytes:
        raise ValueError("out must not overlap weight")
    packed = _q8.quantize_q8_1(x, arithmetic=arithmetic)
    return linear_q8_1(packed, w, dims, out=result)
