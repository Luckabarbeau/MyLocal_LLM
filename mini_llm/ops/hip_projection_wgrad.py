"""HIP BF16-input, FP32-output projection weight gradient and its accumulation
helper for the tied output head.

Used by the Model B projection dispatch. Requires CuPy's gemmEx binding to map
CUDA_R_16BF (14) to HIPBLAS_R_16B. CUDA callers must use their own fallback.

``accumulate_transposed`` exists because a tied output head stores its FP32
gradient row-major ``[Vocab, D]`` while the fast ``gemmEx`` orientation produces
``[D, Vocab]``: the transposed combine must go through a tiled kernel, because a
strided elementwise add over 33.5M FP32 elements costs 27 ms (15 GB/s) against
the 31.9 ms of the surrounding GEMM, and even a strided *source* costs 10 ms.
"""

from __future__ import annotations

import os

import numpy as np

from ..backend import BACKEND_NAME


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


def _bf16_wgrad_gradient_enabled():
    """Round an FP32 gradient operand to BF16 and issue a single tensor-core GEMM.

    Off by default: the three-part decomposition preserves the FP32 gradient
    significand, while this policy rounds the operand (about half a BF16 ulp per
    element) exactly as the packed QKV and ``_linear_backward`` projection
    gradients already do. Gradient accumulation stays FP32 either way.
    """
    return os.environ.get(
        "MINI_LLM_HIP_WGRAD_BF16_GRADIENT", "0"
    ).strip().lower() in {"1", "true", "yes"}


def accumulate_transposed(destination, source):
    """``destination[i, j] += source[j, i]`` for same-device contiguous FP32.

    Tiling keeps both the read and the write coalesced. ``destination += source.T``
    (or ``destination.T += source``) leaves one side strided and collapses to a
    small fraction of memory bandwidth at Model B's ``[Vocab, D]`` shape, which
    is why this kernel exists rather than a BLAS/copy expression.

    Returns ``None``: the caller reads the result from ``destination``.
    """
    import cupy as cp

    if not isinstance(destination, cp.ndarray) or not isinstance(source, cp.ndarray):
        raise TypeError("accumulate_transposed requires CuPy arrays")
    if destination.ndim != 2 or source.ndim != 2:
        raise ValueError("accumulate_transposed requires rank-2 arrays")
    if not destination.flags.c_contiguous or not source.flags.c_contiguous:
        raise ValueError("accumulate_transposed requires C-contiguous arrays")
    if destination.dtype != cp.float32 or source.dtype != cp.float32:
        raise TypeError("accumulate_transposed requires float32 arrays")
    rows, cols = map(int, destination.shape)
    if tuple(reversed(destination.shape)) != tuple(map(int, source.shape)):
        raise ValueError("accumulate_transposed shapes must be [R, C] and [C, R]")
    device_id = cp.cuda.runtime.getDevice()
    if destination.device.id != device_id or source.device.id != device_id:
        raise ValueError("accumulate_transposed arrays must be on the current device")
    if rows == 0 or cols == 0:
        return None
    kernel = getattr(accumulate_transposed, "_kernel", None)
    if kernel is None:
        kernel = cp.RawKernel(r'''
        extern "C" __global__ void accumulate_transposed_f32(
                const float* source, float* destination,
                unsigned int rows, unsigned int cols) {
            // destination is [rows, cols], source is [cols, rows], both row-major.
            // A 16x16 tile with padded LDS banks makes both passes coalesced.
            __shared__ float tile[16][17];
            const unsigned int col0 = blockIdx.x * 16u;
            const unsigned int row0 = blockIdx.y * 16u;
            const unsigned int sx = col0 + threadIdx.y;
            const unsigned int sy = row0 + threadIdx.x;
            if (sx < cols && sy < rows) {
                tile[threadIdx.y][threadIdx.x] = source[sx * rows + sy];
            }
            __syncthreads();
            const unsigned int dx = row0 + threadIdx.y;
            const unsigned int dy = col0 + threadIdx.x;
            if (dx < rows && dy < cols) {
                destination[dx * cols + dy] += tile[threadIdx.x][threadIdx.y];
            }
        }
        ''', "accumulate_transposed_f32", options=("--std=c++11",))
        accumulate_transposed._kernel = kernel
    kernel(((cols + 15) // 16, (rows + 15) // 16), (16, 16),
           (source, destination, np.uint32(rows), np.uint32(cols)))
    return None


def projection_wgrad(activation, gradient, *, out=None, accumulate=False):
    """Return FP32 ``activation.T @ gradient`` with shape ``[Din, Dout]``.

    Inputs must be same-current-device HIP CuPy rank-2 arrays: BF16 activation
    ``[T, Din]`` and BF16 or FP32 gradient ``[T, Dout]``. Strided inputs are copied
    as needed. Work is enqueued on the current stream, without synchronization
    or device-to-host value reads. FP32 gradients use three BF16 residual parts,
    or one BF16 part when ``MINI_LLM_HIP_WGRAD_BF16_GRADIENT=1``; every GEMM
    writes/accumulates FP32, never a BF16 intermediate dot result.

    When ``out`` is supplied it must be a contiguous FP32 ``[Din, Dout]`` array on
    the same device. With ``accumulate=True`` every part adds into it (``beta=1``
    throughout); with ``accumulate=False`` the first part overwrites it, which is
    how a caller stages several chunks of a weight gradient in scratch space
    before combining it once with the parameter gradient.
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
        if out is None:
            return cp.zeros((din, dout), dtype=cp.float32)
        return out
    if out is None:
        out = cp.empty((din, dout), dtype=cp.float32)
        overwrite_output = True
    else:
        if not isinstance(out, cp.ndarray):
            raise TypeError("projection_wgrad out must be a CuPy array")
        if out.ndim != 2:
            raise ValueError("projection_wgrad out must be rank-2 [Din, Dout]")
        if out.shape != (din, dout):
            raise ValueError("projection_wgrad out must match [Din, Dout]")
        if out.dtype != cp.float32:
            raise TypeError("projection_wgrad out must have dtype float32")
        if not out.flags.c_contiguous:
            raise ValueError("projection_wgrad out must be C-contiguous")
        if out.device.id != device_id:
            raise ValueError("projection_wgrad out must be on the current HIP device")
        overwrite_output = not accumulate
    if din == 0 or dout == 0:
        return out

    from cupy_backends.cuda.libs import cublas

    activation = cp.ascontiguousarray(activation)
    gradient = cp.ascontiguousarray(gradient)
    if gradient.dtype == bf16:
        parts = (gradient,)
    elif _bf16_wgrad_gradient_enabled():
        parts = (gradient.astype(bf16),)
    else:
        parts = _split_gradient(gradient)
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
            beta = beta_zero if (index == 0 and overwrite_output) else beta_one
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
