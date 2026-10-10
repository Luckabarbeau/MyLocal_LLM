"""Training RMSNorm with optional fused BF16 mixed-precision CUDA kernels."""

import os
import numpy as np

from ..backend import (
    xp,
    BACKEND_NAME,
    is_low_precision_dtype,
    is_bfloat16_dtype,
)
from ..init import ones_parameter
from ..performance_profiler import PERFORMANCE_PROFILER


_FUSED_BF16_RMSNORM_MODULE = None
_FUSED_BF16_RMSNORM_DISABLED = False


def _env_enabled(name, default="0"):
    raw = os.environ.get(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def _get_fused_bf16_rmsnorm_module():
    """Compile/cache the training BF16 RMSNorm kernels on first use."""
    global _FUSED_BF16_RMSNORM_MODULE, _FUSED_BF16_RMSNORM_DISABLED

    if BACKEND_NAME != "cupy" or _FUSED_BF16_RMSNORM_DISABLED:
        return None
    if _FUSED_BF16_RMSNORM_MODULE is not None:
        return _FUSED_BF16_RMSNORM_MODULE

    code = r"""
    #include <cuda_bf16.h>

    // Production BF16 training keeps the transformer residual stream in FP32.
    // This kernel therefore reads FP32 residuals, accumulates RMS statistics in
    // FP32, multiplies the BF16 scale in FP32 registers, and writes one BF16
    // branch activation.  No full FP32 normalized/output tensor is materialized.
    extern "C" __global__
    void rmsnorm_f32_to_bf16_forward(
        const float* __restrict__ x,
        const __nv_bfloat16* __restrict__ gamma,
        __nv_bfloat16* __restrict__ y,
        float* __restrict__ inv_rms,
        int n_rows,
        int d_model,
        float eps)
    {
        const int row = (int)blockIdx.x;
        const int tid = (int)threadIdx.x;
        if (row >= n_rows) return;

        const long long base = (long long)row * d_model;
        float local = 0.0f;
        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float v = x[base + col];
            local += v * v;
        }

        __shared__ float partial[256];
        partial[tid] = local;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float mean_sq = partial[0] / (float)d_model;
            partial[0] = rsqrtf(mean_sq + eps);
            inv_rms[row] = partial[0];
        }
        __syncthreads();

        const float inv = partial[0];
        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float scale = __bfloat162float(gamma[col]);
            y[base + col] = __float2bfloat16_rn(x[base + col] * inv * scale);
        }
    }

    // dX for FP32 incoming branch gradients.  One block owns one RMSNorm row,
    // performs the projection reduction in FP32, then writes FP32 residual grads.
    extern "C" __global__
    void rmsnorm_bwd_dx_f32dy(
        const float* __restrict__ x,
        const float* __restrict__ dy,
        const __nv_bfloat16* __restrict__ gamma,
        const float* __restrict__ inv_rms,
        float* __restrict__ dx,
        int n_rows,
        int d_model)
    {
        const int row = (int)blockIdx.x;
        const int tid = (int)threadIdx.x;
        if (row >= n_rows) return;
        const long long base = (long long)row * d_model;

        float local = 0.0f;
        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float dxhat = dy[base + col] * __bfloat162float(gamma[col]);
            local += dxhat * x[base + col];
        }

        __shared__ float partial[256];
        partial[tid] = local;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        const float projection = partial[0];
        const float inv = inv_rms[row];
        const float coeff = inv * inv * inv * projection / (float)d_model;

        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float dxhat = dy[base + col] * __bfloat162float(gamma[col]);
            dx[base + col] = dxhat * inv - x[base + col] * coeff;
        }
    }

    // Same dX kernel for the final RMSNorm, whose incoming output-projection
    // gradient remains BF16 on the BF16 model path.
    extern "C" __global__
    void rmsnorm_bwd_dx_bf16dy(
        const float* __restrict__ x,
        const __nv_bfloat16* __restrict__ dy,
        const __nv_bfloat16* __restrict__ gamma,
        const float* __restrict__ inv_rms,
        float* __restrict__ dx,
        int n_rows,
        int d_model)
    {
        const int row = (int)blockIdx.x;
        const int tid = (int)threadIdx.x;
        if (row >= n_rows) return;
        const long long base = (long long)row * d_model;

        float local = 0.0f;
        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float dyf = __bfloat162float(dy[base + col]);
            float dxhat = dyf * __bfloat162float(gamma[col]);
            local += dxhat * x[base + col];
        }

        __shared__ float partial[256];
        partial[tid] = local;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        const float projection = partial[0];
        const float inv = inv_rms[row];
        const float coeff = inv * inv * inv * projection / (float)d_model;

        for (int col = tid; col < d_model; col += (int)blockDim.x) {
            float dyf = __bfloat162float(dy[base + col]);
            float dxhat = dyf * __bfloat162float(gamma[col]);
            dx[base + col] = dxhat * inv - x[base + col] * coeff;
        }
    }

    // dGamma uses a modest 2-D launch.  Each thread owns one feature column
    // and one row chunk, so loads across a warp are coalesced.  Only one FP32
    // atomic add per (column,chunk) is required instead of atomics per token.
    extern "C" __global__
    void rmsnorm_bwd_wgrad_f32dy(
        const float* __restrict__ x,
        const float* __restrict__ dy,
        const float* __restrict__ inv_rms,
        float* __restrict__ dgamma,
        int n_rows,
        int d_model,
        int row_chunks)
    {
        const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
        const int chunk = (int)blockIdx.y;
        if (col >= d_model || chunk >= row_chunks) return;

        const int rows_per_chunk = (n_rows + row_chunks - 1) / row_chunks;
        const int row_begin = chunk * rows_per_chunk;
        const int row_end = min(n_rows, row_begin + rows_per_chunk);
        float local = 0.0f;
        for (int row = row_begin; row < row_end; ++row) {
            long long idx = (long long)row * d_model + col;
            local += dy[idx] * x[idx] * inv_rms[row];
        }
        atomicAdd(dgamma + col, local);
    }

    extern "C" __global__
    void rmsnorm_bwd_wgrad_bf16dy(
        const float* __restrict__ x,
        const __nv_bfloat16* __restrict__ dy,
        const float* __restrict__ inv_rms,
        float* __restrict__ dgamma,
        int n_rows,
        int d_model,
        int row_chunks)
    {
        const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
        const int chunk = (int)blockIdx.y;
        if (col >= d_model || chunk >= row_chunks) return;

        const int rows_per_chunk = (n_rows + row_chunks - 1) / row_chunks;
        const int row_begin = chunk * rows_per_chunk;
        const int row_end = min(n_rows, row_begin + rows_per_chunk);
        float local = 0.0f;
        for (int row = row_begin; row < row_end; ++row) {
            long long idx = (long long)row * d_model + col;
            local += __bfloat162float(dy[idx]) * x[idx] * inv_rms[row];
        }
        atomicAdd(dgamma + col, local);
    }
    """

    names = (
        "rmsnorm_f32_to_bf16_forward",
        "rmsnorm_bwd_dx_f32dy",
        "rmsnorm_bwd_dx_bf16dy",
        "rmsnorm_bwd_wgrad_f32dy",
        "rmsnorm_bwd_wgrad_bf16dy",
    )
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++14",),
            name_expressions=names,
        )
        for name in names:
            module.get_function(name)
        _FUSED_BF16_RMSNORM_MODULE = module
    except Exception:
        if _env_enabled("MINI_LLM_FUSED_BF16_RMSNORM_STRICT"):
            raise
        _FUSED_BF16_RMSNORM_DISABLED = True
        return None

    return _FUSED_BF16_RMSNORM_MODULE


class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    y = gamma * x / sqrt(mean(x^2) + eps)

    The reference implementation performs numerically sensitive work in FP32.
    For BF16 training with the repository's FP32 residual stream, an opt-in
    fused CUDA path keeps x and dx in FP32, stores branch activations in BF16,
    and performs only row/statistical reductions in FP32 registers/shared memory.
    """

    _FUSED_THREADS = 256

    def __init__(self, d_model, eps=1e-6, name="rmsnorm", dtype="float32"):
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.gamma = ones_parameter((d_model,), f"{name}.gamma", dtype=dtype, decay=False)
        self._fused_bf16_enabled = (
            BACKEND_NAME == "cupy"
            and _env_enabled("MINI_LLM_FUSED_BF16_RMSNORM")
        )

    def parameters(self):
        return [self.gamma]

    def _can_use_fused_bf16_forward(self, x, output_dtype):
        if not self._fused_bf16_enabled or BACKEND_NAME != "cupy":
            return False
        if output_dtype is None or not is_bfloat16_dtype(output_dtype):
            return False
        if x.dtype != xp.float32 or not is_bfloat16_dtype(self.gamma.data.dtype):
            return False
        if x.shape[-1] != self.d_model or self.gamma.data.shape != (self.d_model,):
            return False
        if not x.flags.c_contiguous or not self.gamma.data.flags.c_contiguous:
            return False
        return _get_fused_bf16_rmsnorm_module() is not None

    def _forward_fused_bf16(self, x):
        module = _get_fused_bf16_rmsnorm_module()
        if module is None:
            return None
        n_rows = int(x.size // self.d_model)
        y = xp.empty(x.shape, dtype=self.gamma.data.dtype)
        inv_rms = xp.empty(x.shape[:-1] + (1,), dtype=xp.float32)
        with PERFORMANCE_PROFILER.rmsnorm_detail_scope("rmsnorm.fwd.kernel"):
            module.get_function("rmsnorm_f32_to_bf16_forward")(
                (n_rows,), (self._FUSED_THREADS,),
                (
                    x, self.gamma.data, y, inv_rms,
                    np.int32(n_rows), np.int32(self.d_model), np.float32(self.eps),
                ),
            )
        cache = {
            "x": x,
            "inv_rms": inv_rms,
            "original_dtype": x.dtype,
            "fused_bf16_residual": True,
        }
        return y, cache

    def forward_compute(self, x, output_dtype):
        """Forward with a requested downstream compute dtype when supported."""
        return self.forward(x, output_dtype=output_dtype)

    def forward(self, x, output_dtype=None):
        """
        Forward pass with mixed precision support.

        ``output_dtype`` is a performance hint used by the BF16 mixed-precision
        transformer: its residual stream is FP32 but the following attention/MoE
        GEMMs consume BF16.  The reference path intentionally ignores the hint so
        disabling the fused kernel preserves the historical implementation exactly.
        """
        input_dtype = x.dtype

        if self._can_use_fused_bf16_forward(x, output_dtype):
            fused = self._forward_fused_bf16(x)
            if fused is not None:
                return fused

        if is_low_precision_dtype(input_dtype):
            # Convert to float32 only for the numerically sensitive computation.
            x_f32 = x.astype("float32", copy=False)
            mean_sq = xp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            x_hat_f32 = x_f32 * inv_rms
            y_f32 = x_hat_f32 * self.gamma.data.astype("float32", copy=False)
            cache = {
                "x": x,
                "inv_rms": inv_rms,
                "original_dtype": input_dtype,
            }
            return y_f32.astype(input_dtype), cache
        else:
            mean_sq = xp.mean(x * x, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            x_hat = x * inv_rms
            y = x_hat * self.gamma.data
            cache = {
                "x": x,
                "inv_rms": inv_rms,
                "original_dtype": input_dtype,
            }
            return y, cache

    def _can_use_fused_bf16_backward(self, dy, cache):
        if not cache.get("fused_bf16_residual", False):
            return False
        if not self._fused_bf16_enabled or BACKEND_NAME != "cupy":
            return False
        if cache["x"].dtype != xp.float32:
            return False
        if dy.dtype != xp.float32 and not is_bfloat16_dtype(dy.dtype):
            return False
        if not is_bfloat16_dtype(self.gamma.data.dtype):
            return False
        if self.gamma.grad.dtype != xp.float32:
            return False
        if not cache["x"].flags.c_contiguous or not dy.flags.c_contiguous:
            return False
        return _get_fused_bf16_rmsnorm_module() is not None

    def _backward_fused_bf16(self, dy, cache):
        module = _get_fused_bf16_rmsnorm_module()
        if module is None:
            return None

        x = cache["x"]
        inv_rms = cache["inv_rms"]
        n_rows = int(x.size // self.d_model)
        dx = xp.empty_like(x)
        if dy.dtype == xp.float32:
            dx_name = "rmsnorm_bwd_dx_f32dy"
            wg_name = "rmsnorm_bwd_wgrad_f32dy"
        else:
            dx_name = "rmsnorm_bwd_dx_bf16dy"
            wg_name = "rmsnorm_bwd_wgrad_bf16dy"

        with PERFORMANCE_PROFILER.rmsnorm_detail_scope("rmsnorm.bwd.dx"):
            module.get_function(dx_name)(
                (n_rows,), (self._FUSED_THREADS,),
                (
                    x, dy, self.gamma.data, inv_rms, dx,
                    np.int32(n_rows), np.int32(self.d_model),
                ),
            )

        # 16-32 row chunks give enough independent blocks for d_model=512 while
        # keeping the number of FP32 atomics tiny relative to the token count.
        row_chunks = max(1, min(32, (n_rows + 511) // 512))
        col_blocks = (self.d_model + self._FUSED_THREADS - 1) // self._FUSED_THREADS
        with PERFORMANCE_PROFILER.rmsnorm_detail_scope("rmsnorm.bwd.wgrad"):
            module.get_function(wg_name)(
                (col_blocks, row_chunks), (self._FUSED_THREADS,),
                (
                    x, dy, inv_rms, self.gamma.grad,
                    np.int32(n_rows), np.int32(self.d_model), np.int32(row_chunks),
                ),
            )
        return dx

    def backward_residual(self, dy, cache):
        """Backward helper for an FP32 residual stream fed by BF16 branch grads.

        The fused path consumes BF16 ``dy`` directly.  If fusion is unavailable,
        :meth:`backward` promotes it before entering the reference FP32 math.
        """
        return self.backward(dy, cache)

    def backward(self, dy, cache):
        """Backward pass with FP32 reductions and residual-gradient accumulation."""
        input_dtype = cache["original_dtype"]

        fused_cache = cache.get("fused_bf16_residual", False)
        if self._can_use_fused_bf16_backward(dy, cache):
            fused = self._backward_fused_bf16(dy, cache)
            if fused is not None:
                return fused

        # A fused forward may still encounter an unusual non-contiguous/dtype
        # backward input.  Preserve the pre-0050 reference semantics in that
        # fallback by promoting the branch gradient before FP32 residual math.
        if fused_cache and dy.dtype != xp.float32:
            dy = dy.astype("float32", copy=False)

        if is_low_precision_dtype(input_dtype):
            x_f32 = cache["x"].astype("float32", copy=False)
            inv_rms = cache["inv_rms"]
            x_hat_f32 = x_f32 * inv_rms
            dy_f32 = dy.astype("float32", copy=False)
            d = x_f32.shape[-1]
            reduce_axes = tuple(range(dy.ndim - 1))
            self.gamma.grad += xp.sum(dy_f32 * x_hat_f32, axis=reduce_axes)
            dx_hat_f32 = dy_f32 * self.gamma.data.astype("float32", copy=False)
            projection = xp.sum(dx_hat_f32 * x_f32, axis=-1, keepdims=True)
            dx_f32 = dx_hat_f32 * inv_rms - x_f32 * (inv_rms ** 3) * projection / float(d)
            return dx_f32.astype(input_dtype)
        else:
            x = cache["x"]
            inv_rms = cache["inv_rms"]
            x_hat = x * inv_rms
            d = x.shape[-1]
            reduce_axes = tuple(range(dy.ndim - 1))
            self.gamma.grad += xp.sum(dy * x_hat, axis=reduce_axes)
            dx_hat = dy * self.gamma.data
            projection = xp.sum(dx_hat * x, axis=-1, keepdims=True)
            return dx_hat * inv_rms - x * (inv_rms ** 3) * projection / float(d)
