"""Standalone CUDA Q8_1 activation codec (not Q8_0 or Q8_K).

Arithmetic follows Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf,
native_mmvq.cu:158-173 and q8_1_finite.hpp. For each 32 inputs:
A = max(abs(x)); d32 = finite_half_range(A / 127);
q = A == 0 ? 0 : clamp(round_away(x / d32), -127, 127).
Store half_rn(d32), half_rn(finite_half_range(XOR_sum(x))), then 32 int8.
Codes never use the rounded half scale. XOR offsets are 16,8,4,2,1.

``native`` uses --use_fast_math, including FTZ and approximate full-range
FP32 division, as documented by the pinned producer's header. ``ieee`` is
an explicit non-fast-math control, not a claim of native byte parity.
Inputs are assumed finite; no value readback, conversion, synchronization,
or hidden precision-policy integration. Nonfinite-input codes are unspecified
(as in the native float-to-int conversion). Kernels only are cached per device.
"""

from functools import lru_cache

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


BLOCK_BYTES = 36
BLOCK_ELEMENTS = 32
_SOURCE = r'''
#include <cuda_fp16.h>
struct Block { half d; half s; signed char qs[32]; };
static_assert(sizeof(Block) == 36, "Q8_1 stride");
__device__ float bounded(float x) {
    return fabsf(x) > 65504.f ? copysignf(65504.f, x) : x;
}
extern "C" __global__ void quantize(const float* x, Block* out, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = x[i], a = fabsf(v), s = v;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, offset, 32));
        s += __shfl_xor_sync(0xffffffff, s, offset, 32);
    }
    float d = bounded(a / 127.f);
    float q = a == 0.f ? 0.f : roundf(v / d);
    out[i / 32].qs[i % 32] = (signed char)(q > 127.f ? 127.f : q < -127.f ? -127.f : q);
    if (i % 32 == 0) {
        out[i / 32].d = __float2half_rn(bounded(d));
        out[i / 32].s = __float2half_rn(bounded(s));
    }
}
extern "C" __global__ void dequantize(const Block* x, float* out, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = __half2float(x[i / 32].d) * (float)x[i / 32].qs[i % 32];
}
'''


@lru_cache(maxsize=None)
def _kernels(device, arithmetic):
    options = ("--std=c++17",)
    if arithmetic == "native":
        options += ("--use_fast_math",)
    return (cp.RawKernel(_SOURCE, "quantize", options=options),
            cp.RawKernel(_SOURCE, "dequantize", options=("--std=c++17",)))


def _validate(a, dtype, name):
    if cp is None or not isinstance(a, cp.ndarray) or a.dtype != dtype:
        raise TypeError(f"{name} must be a CuPy {np.dtype(dtype)} array")
    if cp.cuda.runtime.is_hip:
        raise RuntimeError("Pinned native Q8_1 codec requires CUDA, not HIP")
    device = cp.cuda.runtime.getDevice()
    if a.device.id != device:
        raise ValueError(f"{name} must be on the current device")
    if a.ndim not in (1, 2) or not a.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous and 1D or 2D")
    if a.data.ptr % 4:
        raise ValueError(f"{name} must be 4-byte aligned")
    return device


def _output(out, shape, dtype, source):
    if out is None:
        return cp.empty(shape, dtype=dtype)
    _validate(out, dtype, "out")
    if out.shape != shape:
        raise ValueError(f"out must have exact shape {shape}")
    # Contiguous buffers: metadata-only overlap check, including differently typed views.
    if (out.nbytes and source.nbytes and out.data.ptr < source.data.ptr + source.nbytes
            and source.data.ptr < out.data.ptr + out.nbytes):
        raise ValueError("out must not overlap input")
    return out


def quantize_q8_1(x, *, out=None, arithmetic="native"):
    """Pack FP32 ``(K,)`` / ``(M,K)`` into uint8 ``(K/32*36,)`` / ``(M,K/32*36)``.

    K must be positive and divisible by 32; zero rows are allowed. Optional out
    has the exact packed shape/dtype and disjoint, 4-byte-aligned storage.
    Launches asynchronously on the current stream. Caller owns stream ordering
    and lifetimes, including supplied output storage. Maximum input: INT32_MAX
    elements. No scratch allocation; only the result is allocated without out.
    """
    device = _validate(x, np.float32, "x")
    if arithmetic not in ("native", "ieee"):
        raise ValueError("arithmetic must be 'native' or 'ieee'")
    if x.shape[-1] == 0 or x.shape[-1] % 32 or x.size > 2147483647:
        raise ValueError("K must be positive, divisible by 32; total size <= INT32_MAX")
    shape = x.shape[:-1] + (x.shape[-1] // 32 * 36,)
    result = _output(out, shape, np.uint8, x)
    if x.size:
        _kernels(device, arithmetic)[0](((x.size + 255) // 256,), (256,),
                                        (x, result, np.int32(x.size)))
    return result


def dequantize_q8_1(packed, *, out=None):
    """Decode stored half d times signed codes to FP32; s is not used.

    Input is contiguous 1D/2D uint8 with a positive multiple of 36 bytes per row.
    Output restores the corresponding 1D/2D FP32 rows; asynchronous, no scratch.
    This is lossy and intentionally does not reconstruct from stored s.
    """
    device = _validate(packed, np.uint8, "packed")
    if packed.shape[-1] == 0 or packed.shape[-1] % 36:
        raise ValueError("packed row byte count must be positive and divisible by 36")
    shape = packed.shape[:-1] + (packed.shape[-1] // 36 * 32,)
    n = packed.size // 36 * 32
    if n > 2147483647:
        raise ValueError("decoded size must be <= INT32_MAX")
    result = _output(out, shape, np.float32, packed)
    if n:
        _kernels(device, "native")[1](((n + 255) // 256,), (256,),
                                      (packed, result, np.int32(n)))
    return result
