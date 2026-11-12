"""Standalone HIP BF16-input, FP32-output projection weight gradient.

Used by the HIP Model B projection dispatch. Requires CuPy's gemmEx binding to
map CUDA_R_16BF (14) to HIPBLAS_R_16B. CUDA callers must use their own fallback.
"""

from __future__ import annotations

import numpy as np


def _split_gradient(gradient):
    """Three BF16 residual components, without reading any device values.

    This preserves FP32 significand precision in the ordinary finite training
    range, unlike a single BF16 cast. BF16 exponent limits still apply: extreme
    FP32 values can overflow the high part or underflow residual components.
    Reconstruction and GEMM accumulation are not promised to be bitwise FP32
    matmul equivalents.
    """
    import cupy as cp

    gradient = cp.ascontiguousarray(gradient, dtype=cp.float32)
    high, mid, low = (cp.empty(gradient.shape, dtype="bfloat16") for _ in range(3))
    if gradient.size == 0:
        return high, mid, low
    kernel = getattr(_split_gradient, "_kernel", None)
    if kernel is None:
        kernel = cp.RawKernel(r'''
        __device__ __forceinline__ unsigned short round_bf16(float value) {
            union { float f; unsigned int u; } bits;
            bits.f = value;
            // CuPy HIP astype preserves the sign/payload and quiets NaNs.
            if ((bits.u & 0x7fffffffu) > 0x7f800000u) {
                return (unsigned short)((bits.u >> 16) | 0x0040u);
            }
            return (unsigned short)((bits.u + 0x7fffu + ((bits.u >> 16) & 1u)) >> 16);
        }

        __device__ __forceinline__ float expand_bf16(unsigned short value) {
            union { float f; unsigned int u; } bits;
            bits.u = (unsigned int)value << 16;
            return bits.f;
        }

        extern "C" __global__ void split_gradient(
                const float* gradient, unsigned short* high,
                unsigned short* mid, unsigned short* low,
                unsigned long long count) {
            const unsigned long long index =
                (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float value = gradient[index];
            const unsigned short hi = round_bf16(value);
            const float residual = value - expand_bf16(hi);
            const unsigned short mi = round_bf16(residual);
            const float remaining = residual - expand_bf16(mi);
            high[index] = hi;
            mid[index] = mi;
            low[index] = round_bf16(remaining);
        }
        ''', "split_gradient", options=("--std=c++11",))
        _split_gradient._kernel = kernel
    kernel(((gradient.size + 255) // 256,), (256,),
           (gradient, high, mid, low, np.uint64(gradient.size)))
    return high, mid, low


def projection_wgrad(activation, gradient):
    """Return FP32 ``activation.T @ gradient`` with shape ``[Din, Dout]``.

    Inputs must be same-current-device HIP CuPy rank-2 arrays: BF16 activation
    ``[T, Din]`` and BF16 or FP32 gradient ``[T, Dout]``. Strided inputs are copied
    as needed. Work is enqueued on the current stream, without synchronization
    or device-to-host value reads. FP32 gradients use three BF16 residual parts;
    every GEMM writes/accumulates FP32, never a BF16 intermediate dot result.
    """
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("projection_wgrad requires HIP; CUDA is not supported")
    if not isinstance(activation, cp.ndarray) or not isinstance(gradient, cp.ndarray):
        raise TypeError("projection_wgrad inputs must be CuPy arrays")
    if activation.ndim != 2 or gradient.ndim != 2:
        raise ValueError("projection_wgrad inputs must be rank-2 [T, width] arrays")
    bf16 = cp.dtype("bfloat16")
    if activation.dtype != bf16:
        raise TypeError("projection_wgrad activation must have dtype bfloat16")
    if gradient.dtype not in (bf16, np.dtype(np.float32)):
        raise TypeError("projection_wgrad gradient must have dtype bfloat16 or float32")
    t, din = activation.shape
    tg, dout = gradient.shape
    if t != tg:
        raise ValueError("projection_wgrad inputs must have the same T dimension")
    if max(t, din, dout) > 2**31 - 1:
        raise ValueError("projection_wgrad dimensions exceed GEMM int32 limits")
    device_id = cp.cuda.runtime.getDevice()
    if activation.device.id != device_id or gradient.device.id != device_id:
        raise ValueError("projection_wgrad inputs must be on the current HIP device")
    if t == 0:
        return cp.zeros((din, dout), dtype=cp.float32)
    out = cp.empty((din, dout), dtype=cp.float32)
    if din == 0 or dout == 0:
        return out

    from cupy_backends.cuda.libs import cublas

    activation = cp.ascontiguousarray(activation)
    gradient = cp.ascontiguousarray(gradient)
    parts = (gradient,) if gradient.dtype == bf16 else _split_gradient(gradient)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    pointer_mode = cublas.getPointerMode(handle)
    alpha = np.array(1.0, dtype=np.float32)
    beta_zero = np.array(0.0, dtype=np.float32)
    beta_one = np.array(1.0, dtype=np.float32)
    try:
        if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        for index, part in enumerate(parts):
            beta = beta_zero if index == 0 else beta_one
            # Row-major C = A.T @ B becomes column-major C.T = B.T @ A.
            # HIP's installed legacy gemmEx expects a datatype (0), NOT the
            # CUDA-only CUBLAS_COMPUTE_32F enum (68), for FP32 computation.
            cublas.gemmEx(
                handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
                dout, din, t, alpha.ctypes.data,
                part.data.ptr, cp.cuda.runtime.CUDA_R_16BF, dout,
                activation.data.ptr, cp.cuda.runtime.CUDA_R_16BF, din,
                beta.ctypes.data, out.data.ptr, cp.cuda.runtime.CUDA_R_32F, dout,
                cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
            )
    finally:
        if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
            cublas.setPointerMode(handle, pointer_mode)
    return out
