"""Inference-only RMSNorm without backward caches.

The CuPy/float16 path uses a fused CUDA kernel.  Values stay in FP16 in
memory, while the sum-of-squares reduction and normalization arithmetic are
performed in FP32 registers/shared memory.  This avoids materializing full
FP32 copies of the activation tensor during autoregressive inference.
"""

import os
import numpy as _np

from mini_llm.backend import (
    xp, BACKEND_NAME, is_low_precision_dtype, is_bfloat16_dtype,
)


_FUSED_FP16_RMSNORM_KERNEL = None
_FUSED_RESIDUAL_RMSNORM_MODULE = None
_FUSED_RESIDUAL_RMSNORM_DISABLED = False

_FUSED_FP16_RMSNORM_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" __global__
void rmsnorm_fp16_fused(
    const half* __restrict__ x,
    const half* __restrict__ gamma,
    half* __restrict__ y,
    const int n_rows,
    const int d_model,
    const float eps)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int base = row * d_model;

    // Accumulate x^2 in FP32.  x itself is never expanded to an FP32 array.
    float sum_sq = 0.0f;
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const float value = __half2float(x[base + col]);
        sum_sq += value * value;
    }

    // One block handles one RMSNorm row.  The launch always uses 256 threads.
    __shared__ float partial[256];
    partial[tid] = sum_sq;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float mean_sq = partial[0] / (float)d_model;
        partial[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float inv_rms = partial[0];

    // Normalize and scale in FP32 registers, then write FP16 once.
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const float value = __half2float(x[base + col]);
        const float scale = __half2float(gamma[col]);
        y[base + col] = __float2half_rn(value * inv_rms * scale);
    }
}
"""


_FUSED_RESIDUAL_RMSNORM_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define REDUCE_RMS(INPUT_EXPR) \
    float sum_sq = 0.0f; \
    for (int col = tid; col < d_model; col += (int)blockDim.x) { \
        const float value = (INPUT_EXPR); \
        sum_sq += value * value; \
    } \
    for (int offset = 16; offset > 0; offset >>= 1) \
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
    __shared__ float warp_sums[8]; \
    const int lane = tid & 31; \
    const int warp = tid >> 5; \
    if (lane == 0) warp_sums[warp] = sum_sq; \
    __syncthreads(); \
    if (warp == 0) { \
        sum_sq = lane < 8 ? warp_sums[lane] : 0.0f; \
        for (int offset = 16; offset > 0; offset >>= 1) \
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
        if (lane == 0) warp_sums[0] = rsqrtf(sum_sq / (float)d_model + eps); \
    } \
    __syncthreads(); \
    const float inv_rms = warp_sums[0]

extern "C" __global__
void rmsnorm_f32_to_fp16(
    const float* __restrict__ x,
    const half* __restrict__ gamma,
    half* __restrict__ y,
    const int n_rows,
    const int d_model,
    const float eps)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (row >= n_rows) return;
    const int base = row * d_model;
    REDUCE_RMS(x[base + col]);
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        y[base + col] = __float2half_rn(
            x[base + col] * inv_rms * __half2float(gamma[col]));
    }
}

extern "C" __global__
void rmsnorm_f32_to_bf16(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ y,
    const int n_rows,
    const int d_model,
    const float eps)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (row >= n_rows) return;
    const int base = row * d_model;
    REDUCE_RMS(x[base + col]);
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        y[base + col] = __float2bfloat16_rn(
            x[base + col] * inv_rms * __bfloat162float(gamma[col]));
    }
}

extern "C" __global__
void residual_add_rmsnorm_f32_fp16(
    const float* __restrict__ residual,
    const half* __restrict__ branch,
    const half* __restrict__ gamma,
    float* __restrict__ residual_out,
    half* __restrict__ y,
    const int n_rows,
    const int d_model,
    const float alpha,
    const float eps)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (row >= n_rows) return;
    const int base = row * d_model;
    REDUCE_RMS(residual[base + col] + alpha * __half2float(branch[base + col]));
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const int index = base + col;
        const float value = residual[index] + alpha * __half2float(branch[index]);
        residual_out[index] = value;
        y[index] = __float2half_rn(value * inv_rms * __half2float(gamma[col]));
    }
}

extern "C" __global__
void residual_add_rmsnorm_f32_bf16(
    const float* __restrict__ residual,
    const __nv_bfloat16* __restrict__ branch,
    const __nv_bfloat16* __restrict__ gamma,
    float* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ y,
    const int n_rows,
    const int d_model,
    const float alpha,
    const float eps)
{
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (row >= n_rows) return;
    const int base = row * d_model;
    REDUCE_RMS(residual[base + col] + alpha * __bfloat162float(branch[base + col]));
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const int index = base + col;
        const float value = residual[index] + alpha * __bfloat162float(branch[index]);
        residual_out[index] = value;
        y[index] = __float2bfloat16_rn(
            value * inv_rms * __bfloat162float(gamma[col]));
    }
}
"""


def _get_fused_residual_rmsnorm_module():
    """Compile/cache FP32-residual to low-precision RMSNorm kernels."""
    global _FUSED_RESIDUAL_RMSNORM_MODULE, _FUSED_RESIDUAL_RMSNORM_DISABLED
    if BACKEND_NAME != "cupy" or _FUSED_RESIDUAL_RMSNORM_DISABLED:
        return None
    if _FUSED_RESIDUAL_RMSNORM_MODULE is not None:
        return _FUSED_RESIDUAL_RMSNORM_MODULE
    try:
        module = xp.RawModule(
            code=_FUSED_RESIDUAL_RMSNORM_SOURCE,
            options=("--std=c++14",),
            name_expressions=(
                "rmsnorm_f32_to_fp16", "rmsnorm_f32_to_bf16",
                "residual_add_rmsnorm_f32_fp16",
                "residual_add_rmsnorm_f32_bf16",
            ),
        )
        module.get_function("rmsnorm_f32_to_fp16")
        module.get_function("rmsnorm_f32_to_bf16")
        module.get_function("residual_add_rmsnorm_f32_fp16")
        module.get_function("residual_add_rmsnorm_f32_bf16")
        _FUSED_RESIDUAL_RMSNORM_MODULE = module
    except Exception:
        if os.environ.get("MINI_LLM_FUSED_RMSNORM_STRICT", "0").lower() in {
            "1", "true", "yes", "on",
        }:
            raise
        _FUSED_RESIDUAL_RMSNORM_DISABLED = True
        return None
    return _FUSED_RESIDUAL_RMSNORM_MODULE


def _get_fused_fp16_rmsnorm_kernel():
    """Compile/cache the CuPy FP16 RMSNorm kernel on first use."""
    global _FUSED_FP16_RMSNORM_KERNEL

    if BACKEND_NAME != "cupy":
        return None

    if _FUSED_FP16_RMSNORM_KERNEL is None:
        _FUSED_FP16_RMSNORM_KERNEL = xp.RawKernel(
            _FUSED_FP16_RMSNORM_SOURCE,
            "rmsnorm_fp16_fused",
            options=("-std=c++11",),
        )

    return _FUSED_FP16_RMSNORM_KERNEL


class RMSNormInference:
    """
    RMSNorm for inference - no cache needed for backward pass.

    Computes: y = gamma * x / sqrt(mean(x^2) + eps)

    On CuPy with FP16 input and FP16 gamma, the default path is a fused CUDA
    kernel that keeps tensors in FP16 memory but performs the reduction and
    normalization arithmetic in FP32.  The NumPy path, FP32 path, and unusual
    dtype/layout cases retain the reference implementation.
    """

    _FUSED_THREADS = 256

    def __init__(self, d_model, eps=1e-6, dtype="float32"):
        """
        Initialize RMSNorm for inference.

        Args:
            d_model: Model dimension
            eps: Numerical stability constant
            dtype: Data type
        """
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.dtype = dtype

        # Useful for A/B testing and as a portability escape hatch.  Training
        # RMSNorm is completely independent of this setting.
        disable_fused = os.environ.get("MINI_LLM_DISABLE_FUSED_RMSNORM", "0")
        self._fused_fp16_enabled = (
            BACKEND_NAME == "cupy"
            and disable_fused.lower() not in {"1", "true", "yes", "on"}
        )

        # Will be set externally from the training model.
        self.gamma = None

    def set_weights(self, gamma):
        """Set normalization weights (shared with the training model)."""
        if BACKEND_NAME == "cupy" and not hasattr(gamma, "__cuda_array_interface__"):
            self.gamma = xp.asarray(gamma)
        else:
            self.gamma = gamma

    def _can_use_fused_fp16(self, x):
        """Return True only when the fused kernel can preserve API semantics."""
        if not self._fused_fp16_enabled or BACKEND_NAME != "cupy":
            return False
        if self.gamma is None:
            return False
        if x.dtype != xp.float16 or self.gamma.dtype != xp.float16:
            return False
        if x.shape[-1] != self.d_model or self.gamma.shape != (self.d_model,):
            return False

        # RawKernel indexes rows as one contiguous [*, d_model] matrix.  Do
        # not silently copy unusual views; use the reference path instead.
        if not x.flags.c_contiguous or not self.gamma.flags.c_contiguous:
            return False
        return True

    def _forward_fused_fp16(self, x):
        """FP16 storage + FP32 accumulation RMSNorm in one CUDA kernel."""
        n_rows = int(x.size // self.d_model)
        y = xp.empty_like(x)

        kernel = _get_fused_fp16_rmsnorm_kernel()
        kernel(
            (n_rows,),
            (self._FUSED_THREADS,),
            (
                x,
                self.gamma,
                y,
                _np.int32(n_rows),
                _np.int32(self.d_model),
                _np.float32(self.eps),
            ),
        )
        return y

    def _can_use_fused_residual(self, x):
        if not self._fused_fp16_enabled or BACKEND_NAME != "cupy":
            return False
        if self.gamma is None or x.dtype != xp.float32:
            return False
        if self.gamma.dtype != xp.float16 and not is_bfloat16_dtype(self.gamma.dtype):
            return False
        if x.shape[-1] != self.d_model or self.gamma.shape != (self.d_model,):
            return False
        if not x.flags.c_contiguous or not self.gamma.flags.c_contiguous:
            return False
        return _get_fused_residual_rmsnorm_module() is not None

    def _forward_fused_residual(self, x):
        module = _get_fused_residual_rmsnorm_module()
        n_rows = int(x.size // self.d_model)
        y = xp.empty(x.shape, dtype=self.gamma.dtype)
        name = (
            "rmsnorm_f32_to_bf16"
            if is_bfloat16_dtype(self.gamma.dtype)
            else "rmsnorm_f32_to_fp16"
        )
        module.get_function(name)(
            (n_rows,),
            (self._FUSED_THREADS,),
            (
                x,
                self.gamma,
                y,
                _np.int32(n_rows),
                _np.int32(self.d_model),
                _np.float32(self.eps),
            ),
        )
        return y

    def forward_residual(self, x):
        """Normalize an FP32 residual stream directly into the branch dtype."""
        if self._can_use_fused_residual(x):
            return self._forward_fused_residual(x)
        output_dtype = self.gamma.dtype if self.gamma is not None else self.dtype
        return self.forward(x).astype(output_dtype, copy=False)

    def forward_residual_add(self, residual, branch, alpha=1.0):
        """Add a low-precision branch and normalize the updated FP32 residual."""
        can_fuse = (
            self._can_use_fused_residual(residual)
            and branch.size == residual.size
            and branch.dtype == self.gamma.dtype
            and branch.flags.c_contiguous
            and isinstance(alpha, (int, float, _np.number))
        )
        if can_fuse:
            n_rows = int(residual.size // self.d_model)
            residual_out = xp.empty_like(residual)
            y = xp.empty(residual.shape, dtype=self.gamma.dtype)
            suffix = "bf16" if is_bfloat16_dtype(self.gamma.dtype) else "fp16"
            _get_fused_residual_rmsnorm_module().get_function(
                f"residual_add_rmsnorm_f32_{suffix}"
            )(
                (n_rows,),
                (self._FUSED_THREADS,),
                (
                    residual,
                    branch,
                    self.gamma,
                    residual_out,
                    y,
                    _np.int32(n_rows),
                    _np.int32(self.d_model),
                    _np.float32(alpha),
                    _np.float32(self.eps),
                ),
            )
            return residual_out, y

        branch_residual = branch.astype(xp.float32, copy=False)
        if alpha is not None:
            branch_residual = branch_residual * alpha
        residual_out = residual + branch_residual
        return residual_out, self.forward_residual(residual_out)

    def _forward_reference(self, x):
        """Portable/reference implementation used outside the fused FP16 path."""
        input_dtype = x.dtype

        if is_low_precision_dtype(input_dtype):
            # Low-precision storage, FP32 reduction/normalization. This covers
            # both FP16 and BF16; only FP16 has a fused inference kernel today.
            x_f32 = x.astype("float32", copy=False)
            mean_sq = xp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            x_hat_f32 = x_f32 * inv_rms

            if BACKEND_NAME == "cupy" and not hasattr(self.gamma, "__cuda_array_interface__"):
                gamma_f32 = xp.asarray(self.gamma).astype("float32", copy=False)
            else:
                gamma_f32 = self.gamma.astype("float32", copy=False)

            y_f32 = x_hat_f32 * gamma_f32
            return y_f32.astype(input_dtype)

        mean_sq = xp.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
        x_hat = x * inv_rms

        if BACKEND_NAME == "cupy" and not hasattr(self.gamma, "__cuda_array_interface__"):
            return x_hat * xp.asarray(self.gamma)
        return x_hat * self.gamma

    def forward(self, x):
        """
        Forward pass through RMSNorm.

        Args:
            x: Input tensor [..., D]

        Returns:
            y: Output tensor with the same shape and dtype as ``x``.
        """
        if self._can_use_fused_fp16(x):
            return self._forward_fused_fp16(x)
        return self._forward_reference(x)
