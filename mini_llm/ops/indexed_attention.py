"""Full-resolution attention over explicitly selected key/value positions.

This module contains no routing policy.  It consumes a ``KeySelectionPlan`` and
performs ordinary scaled dot-product attention over the exact K/V tokens named
by that plan.  It therefore serves local/dilated/global/manual/learned sparse
patterns equally; only the key-visibility provider changes.
"""

import math
import os

import numpy as np

from ..backend import xp, BACKEND_NAME, is_bfloat16_dtype, is_low_precision_dtype
from ..performance_profiler import local_detail_scope
from .attention_selection import KeySelectionPlan
from .cublas_grouped import (
    RowMajorGemmGroup,
    RowMajorGemmProblem,
    available as _cublas_grouped_available,
    grouped_bf16_gemm as _cublas_grouped_bf16_gemm,
    is_enabled as _cublas_grouped_local_enabled,
    local_cublas_mode as _cublas_local_mode,
    strict_enabled as _cublas_grouped_strict_enabled,
)


def _bf16_mixed_context_enabled(dtype):
    """Mirror the mixed-attention BF16 context-storage fast-path flag."""
    if not is_bfloat16_dtype(dtype):
        return False
    raw = os.environ.get("MINI_LLM_BF16_MIXED_CONTEXT", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


_DEFAULT_QUERY_CHUNK_SIZE = 128
_DEFAULT_LOCAL_QUERY_CHUNK_SIZE = 512
_DEFAULT_DILATED_QUERY_CHUNK_SIZE = 1024


_LOCAL_SOFTMAX_MODULE = None
_LOCAL_SOFTMAX_DISABLED = False

_LOCAL_BF16_PIPELINE_MODULE = None
_LOCAL_BF16_PIPELINE_DISABLED = False


_RETRIEVAL_SCATTER_MODULE = None
_RETRIEVAL_SCATTER_DISABLED = False

_RETRIEVAL_BF16_PIPELINE_MODULE = None
_RETRIEVAL_BF16_PIPELINE_DISABLED = False


def _bf16_retrieval_pipeline_enabled(dtype):
    """Use BF16 Tensor-Core GEMMs throughout block retrieval attention.

    The retrieval path needs a dedicated implementation because router weights
    enter attention as an additive log-probability bias.  The fast path keeps
    score/probability/backward matrices in BF16 storage, performs softmax and
    its Jacobian in FP32 inside CUDA, and accumulates final Q/K/V gradients
    into the existing FP32 buffers.
    """
    raw = os.environ.get(
        "MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE", "0"
    ).strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _direct_retrieval_scatter_enabled():
    """Use direct CUDA scatter kernels for block retrieval attention."""
    raw = os.environ.get("MINI_LLM_DIRECT_RETRIEVAL_SCATTER", "0").strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no"}


def _get_retrieval_scatter_module():
    global _RETRIEVAL_SCATTER_MODULE, _RETRIEVAL_SCATTER_DISABLED
    if not _direct_retrieval_scatter_enabled() or _RETRIEVAL_SCATTER_DISABLED:
        return None
    if _RETRIEVAL_SCATTER_MODULE is not None:
        return _RETRIEVAL_SCATTER_MODULE
    code = r"""
    extern "C" __global__
    void retrieval_route_scatter_f32(
        const float* src, float* dst,
        const long long* query_positions, const bool* query_valid,
        int batch, int routes, int heads, int stride, int d_head, int seq_len) {
        long long n = (long long)batch * routes * heads * stride * d_head;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += (long long)blockDim.x * gridDim.x) {
            long long t = idx;
            int d = (int)(t % d_head); t /= d_head;
            int s = (int)(t % stride); t /= stride;
            int h = (int)(t % heads); t /= heads;
            int r = (int)(t % routes); t /= routes;
            int b = (int)t;
            long long rs = (long long)r * stride + s;
            if (!query_valid[rs]) continue;
            long long qpos = query_positions[rs];
            if (qpos < 0 || qpos >= seq_len) continue;
            long long dst_idx = ((((long long)b * seq_len + qpos) * heads + h) * d_head + d);
            dst[dst_idx] = src[idx];
        }
    }

    extern "C" __global__
    void retrieval_kv_scatter_add_f32(
        const float* dk_src, const float* dv_src,
        float* dk_dst, float* dv_dst,
        const long long* key_indices, const long long* kv_map,
        int batch, int routes, int heads, int keys, int d_head,
        int seq_len, int n_kv_heads) {
        long long n = (long long)batch * routes * heads * keys * d_head;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += (long long)blockDim.x * gridDim.x) {
            long long t = idx;
            int d = (int)(t % d_head); t /= d_head;
            int kslot = (int)(t % keys); t /= keys;
            int h = (int)(t % heads); t /= heads;
            int r = (int)(t % routes); t /= routes;
            int b = (int)t;
            long long key_idx = key_indices[(((long long)b * routes + r) * heads + h) * keys + kslot];
            long long kvh = kv_map[h];
            if (key_idx < 0 || key_idx >= seq_len || kvh < 0 || kvh >= n_kv_heads) continue;
            long long dst_idx = ((((long long)b * seq_len + key_idx) * n_kv_heads + kvh) * d_head + d);
            atomicAdd(dk_dst + dst_idx, dk_src[idx]);
            atomicAdd(dv_dst + dst_idx, dv_src[idx]);
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=("retrieval_route_scatter_f32", "retrieval_kv_scatter_add_f32"),
        )
        module.get_function("retrieval_route_scatter_f32")
        module.get_function("retrieval_kv_scatter_add_f32")
        _RETRIEVAL_SCATTER_MODULE = module
    except Exception:
        strict = os.environ.get("MINI_LLM_DIRECT_RETRIEVAL_SCATTER_STRICT", "0").strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _RETRIEVAL_SCATTER_DISABLED = True
        return None
    return _RETRIEVAL_SCATTER_MODULE


def _retrieval_route_scatter_f32(src, dst, query_positions, query_valid):
    module = _get_retrieval_scatter_module()
    if module is None or src.dtype != xp.float32 or dst.dtype != xp.float32:
        return False
    if not src.flags.c_contiguous or not dst.flags.c_contiguous:
        return False
    if query_positions.dtype != xp.int64 or query_valid.dtype != xp.bool_:
        return False
    if not query_positions.flags.c_contiguous or not query_valid.flags.c_contiguous:
        return False
    if src.ndim != 5 or dst.ndim != 4:
        return False
    batch, routes, heads, stride, d_head = map(int, src.shape)
    if (int(dst.shape[0]), int(dst.shape[2]), int(dst.shape[3])) != (batch, heads, d_head):
        return False
    if tuple(query_positions.shape) != (routes, stride) or tuple(query_valid.shape) != (routes, stride):
        return False
    n = int(src.size)
    if n == 0:
        return True
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    module.get_function("retrieval_route_scatter_f32")(
        (blocks,), (threads,),
        (src, dst, query_positions, query_valid,
         np.int32(batch), np.int32(routes), np.int32(heads), np.int32(stride),
         np.int32(d_head), np.int32(dst.shape[1])),
    )
    return True


def _retrieval_kv_scatter_add_f32(dk_src, dv_src, dk_dst, dv_dst, key_indices, kv_map):
    module = _get_retrieval_scatter_module()
    arrays = (dk_src, dv_src, dk_dst, dv_dst)
    if module is None or any(a.dtype != xp.float32 for a in arrays):
        return False
    if any(not a.flags.c_contiguous for a in arrays):
        return False
    if key_indices.dtype != xp.int64 or kv_map.dtype != xp.int64:
        return False
    if not key_indices.flags.c_contiguous or not kv_map.flags.c_contiguous:
        return False
    if dk_src.shape != dv_src.shape or dk_dst.shape != dv_dst.shape:
        return False
    if dk_src.ndim != 5 or dk_dst.ndim != 4:
        return False
    batch, routes, heads, keys, d_head = map(int, dk_src.shape)
    if tuple(key_indices.shape) != (batch, routes, heads, keys) or tuple(kv_map.shape) != (heads,):
        return False
    if int(dk_dst.shape[0]) != batch or int(dk_dst.shape[3]) != d_head:
        return False
    n = int(dk_src.size)
    if n == 0:
        return True
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    module.get_function("retrieval_kv_scatter_add_f32")(
        (blocks,), (threads,),
        (dk_src, dv_src, dk_dst, dv_dst, key_indices, kv_map,
         np.int32(batch), np.int32(routes), np.int32(heads), np.int32(keys),
         np.int32(d_head), np.int32(dk_dst.shape[1]), np.int32(dk_dst.shape[2])),
    )
    return True



def _get_retrieval_bf16_pipeline_module():
    """Compile the block-retrieval BF16 Tensor-Core/softmax CUDA kernels."""
    global _RETRIEVAL_BF16_PIPELINE_MODULE, _RETRIEVAL_BF16_PIPELINE_DISABLED
    if BACKEND_NAME != "cupy" or _RETRIEVAL_BF16_PIPELINE_DISABLED:
        return None
    if _RETRIEVAL_BF16_PIPELINE_MODULE is not None:
        return _RETRIEVAL_BF16_PIPELINE_MODULE

    code = r"""
    #include <cuda_bf16.h>
    #include <mma.h>
    using namespace nvcuda;

    // All three GEMM kernels intentionally require dimensions divisible by 16.
    // The production 4k retrieval geometry is S=128, K=512, Dh=64, so no edge
    // tiles are needed and every block is a single warp-level Tensor-Core tile.
    extern "C" __global__
    void retrieval_bf16_gemm_nt(
        const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
        int batches, int M, int N, int K, float alpha) {
        int batch = (int)blockIdx.z;
        int m0 = (int)blockIdx.y * 16;
        int n0 = (int)blockIdx.x * 16;
        if (batch >= batches || m0 >= M || n0 >= N) return;

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        const __nv_bfloat16* Ab = A + (long long)batch * M * K;
        // B is physically [N,K] row-major.  Interpreting it as [K,N]
        // column-major gives B^T without an explicit transpose.
        const __nv_bfloat16* Bb = B + (long long)batch * N * K;
        for (int k0 = 0; k0 < K; k0 += 16) {
            wmma::load_matrix_sync(a_frag, Ab + (long long)m0 * K + k0, K);
            wmma::load_matrix_sync(b_frag, Bb + (long long)n0 * K + k0, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] *= alpha;
        __shared__ float tile[16 * 16];
        wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            int mi = i >> 4;
            int ni = i & 15;
            C[((long long)batch * M + (m0 + mi)) * N + (n0 + ni)] =
                __float2bfloat16_rn(tile[i]);
        }
    }

    extern "C" __global__
    void retrieval_bf16_gemm_nn(
        const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
        int batches, int M, int N, int K, float alpha) {
        int batch = (int)blockIdx.z;
        int m0 = (int)blockIdx.y * 16;
        int n0 = (int)blockIdx.x * 16;
        if (batch >= batches || m0 >= M || n0 >= N) return;

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        const __nv_bfloat16* Ab = A + (long long)batch * M * K;
        const __nv_bfloat16* Bb = B + (long long)batch * K * N;
        for (int k0 = 0; k0 < K; k0 += 16) {
            wmma::load_matrix_sync(a_frag, Ab + (long long)m0 * K + k0, K);
            wmma::load_matrix_sync(b_frag, Bb + (long long)k0 * N + n0, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] *= alpha;
        __shared__ float tile[16 * 16];
        wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            int mi = i >> 4;
            int ni = i & 15;
            C[((long long)batch * M + (m0 + mi)) * N + (n0 + ni)] =
                __float2bfloat16_rn(tile[i]);
        }
    }

    extern "C" __global__
    void retrieval_bf16_gemm_tn(
        const __nv_bfloat16* A_storage, const __nv_bfloat16* B,
        __nv_bfloat16* C, int batches, int M, int N, int K, float alpha) {
        int batch = (int)blockIdx.z;
        int m0 = (int)blockIdx.y * 16;
        int n0 = (int)blockIdx.x * 16;
        if (batch >= batches || m0 >= M || n0 >= N) return;

        // A_storage is physically [K,M] row-major.  The same bytes represent
        // A_storage^T=[M,K] in column-major order with leading dimension M.
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __nv_bfloat16, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        const __nv_bfloat16* Ab = A_storage + (long long)batch * K * M;
        const __nv_bfloat16* Bb = B + (long long)batch * K * N;
        for (int k0 = 0; k0 < K; k0 += 16) {
            wmma::load_matrix_sync(a_frag, Ab + (long long)k0 * M + m0, M);
            wmma::load_matrix_sync(b_frag, Bb + (long long)k0 * N + n0, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] *= alpha;
        __shared__ float tile[16 * 16];
        wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            int mi = i >> 4;
            int ni = i & 15;
            C[((long long)batch * M + (m0 + mi)) * N + (n0 + ni)] =
                __float2bfloat16_rn(tile[i]);
        }
    }

    extern "C" __global__
    void retrieval_softmax_fwd_bf16(
        const __nv_bfloat16* scores, const __nv_bfloat16* weights,
        const bool* query_valid, __nv_bfloat16* probs,
        long long rows, int routes, int heads, int stride, int keys,
        int router_heads, int selected_blocks, int block_size,
        float score_scale, int use_logit_bias, float weight_scale,
        float weight_eps) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        int s = (int)(row % stride);
        long long t = row / stride;
        int h = (int)(t % heads); t /= heads;
        int r = (int)(t % routes); t /= routes;
        int b = (int)t;
        const __nv_bfloat16* src = scores + row * keys;
        __nv_bfloat16* dst = probs + row * keys;
        if (!query_valid[(long long)r * stride + s]) {
            for (int j = threadIdx.x; j < keys; j += blockDim.x)
                dst[j] = __float2bfloat16_rn(0.0f);
            return;
        }
        int rh = router_heads == 1 ? 0 : h;
        const __nv_bfloat16* wr = weights
            + (((long long)b * routes + r) * router_heads + rh) * selected_blocks;
        extern __shared__ float sh[];
        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < keys; j += blockDim.x) {
            float value = __bfloat162float(src[j]) * score_scale;
            if (use_logit_bias) {
                int block = j / block_size;
                float w = __bfloat162float(wr[block]);
                value += weight_scale * logf(w + weight_eps);
            }
            local_max = fmaxf(local_max, value);
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (threadIdx.x < d) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + d]);
            __syncthreads();
        }
        float row_max = sh[0];
        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < keys; j += blockDim.x) {
            float value = __bfloat162float(src[j]) * score_scale;
            if (use_logit_bias) {
                int block = j / block_size;
                float w = __bfloat162float(wr[block]);
                value += weight_scale * logf(w + weight_eps);
            }
            local_sum += expf(value - row_max);
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (threadIdx.x < d) sh[threadIdx.x] += sh[threadIdx.x + d];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < keys; j += blockDim.x) {
            float value = __bfloat162float(src[j]) * score_scale;
            if (use_logit_bias) {
                int block = j / block_size;
                float w = __bfloat162float(wr[block]);
                value += weight_scale * logf(w + weight_eps);
            }
            dst[j] = __float2bfloat16_rn(expf(value - row_max) * inv);
        }
    }

    extern "C" __global__
    void retrieval_softmax_bwd_bf16(
        const __nv_bfloat16* dprobs, const __nv_bfloat16* probs,
        const bool* query_valid, __nv_bfloat16* dscores,
        long long rows, int routes, int heads, int stride, int keys) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        int s = (int)(row % stride);
        long long t = row / stride;
        (void)(t % heads); t /= heads;
        int r = (int)(t % routes);
        const __nv_bfloat16* dp = dprobs + row * keys;
        const __nv_bfloat16* p = probs + row * keys;
        __nv_bfloat16* ds = dscores + row * keys;
        if (!query_valid[(long long)r * stride + s]) {
            for (int j = threadIdx.x; j < keys; j += blockDim.x)
                ds[j] = __float2bfloat16_rn(0.0f);
            return;
        }
        extern __shared__ float sh[];
        float local = 0.0f;
        for (int j = threadIdx.x; j < keys; j += blockDim.x)
            local += __bfloat162float(dp[j]) * __bfloat162float(p[j]);
        sh[threadIdx.x] = local;
        __syncthreads();
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (threadIdx.x < d) sh[threadIdx.x] += sh[threadIdx.x + d];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < keys; j += blockDim.x) {
            float pf = __bfloat162float(p[j]);
            ds[j] = __float2bfloat16_rn(pf * (__bfloat162float(dp[j]) - correction));
        }
    }

    extern "C" __global__
    void retrieval_route_scatter_bf16_to_f32(
        const __nv_bfloat16* src, float* dst,
        const long long* query_positions, const bool* query_valid,
        int batch, int routes, int heads, int stride, int d_head, int seq_len) {
        long long n = (long long)batch * routes * heads * stride * d_head;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += (long long)blockDim.x * gridDim.x) {
            long long t = idx;
            int d = (int)(t % d_head); t /= d_head;
            int s = (int)(t % stride); t /= stride;
            int h = (int)(t % heads); t /= heads;
            int r = (int)(t % routes); t /= routes;
            int b = (int)t;
            long long rs = (long long)r * stride + s;
            if (!query_valid[rs]) continue;
            long long qpos = query_positions[rs];
            if (qpos < 0 || qpos >= seq_len) continue;
            long long out = ((((long long)b * seq_len + qpos) * heads + h) * d_head + d);
            dst[out] = __bfloat162float(src[idx]);
        }
    }

    extern "C" __global__
    void retrieval_kv_scatter_bf16_to_f32(
        const __nv_bfloat16* dk_src, const __nv_bfloat16* dv_src,
        float* dk_dst, float* dv_dst,
        const long long* key_indices, const long long* kv_map,
        int batch, int routes, int heads, int keys, int d_head,
        int seq_len, int n_kv_heads) {
        long long n = (long long)batch * routes * heads * keys * d_head;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += (long long)blockDim.x * gridDim.x) {
            long long t = idx;
            int d = (int)(t % d_head); t /= d_head;
            int kslot = (int)(t % keys); t /= keys;
            int h = (int)(t % heads); t /= heads;
            int r = (int)(t % routes); t /= routes;
            int b = (int)t;
            long long key_idx = key_indices[(((long long)b * routes + r) * heads + h) * keys + kslot];
            long long kvh = kv_map[h];
            if (key_idx < 0 || key_idx >= seq_len || kvh < 0 || kvh >= n_kv_heads) continue;
            long long out = ((((long long)b * seq_len + key_idx) * n_kv_heads + kvh) * d_head + d);
            atomicAdd(dk_dst + out, __bfloat162float(dk_src[idx]));
            atomicAdd(dv_dst + out, __bfloat162float(dv_src[idx]));
        }
    }

    extern "C" __global__
    void retrieval_bias_grad_bf16(
        const __nv_bfloat16* dscores, const __nv_bfloat16* weights,
        float* dweights, int batch, int routes, int heads, int stride,
        int router_heads, int selected_blocks, int block_size,
        float weight_scale, float weight_eps) {
        long long out_count = (long long)batch * routes * router_heads * selected_blocks;
        long long out = (long long)blockIdx.x;
        if (out >= out_count) return;
        long long t = out;
        int block = (int)(t % selected_blocks); t /= selected_blocks;
        int rh = (int)(t % router_heads); t /= router_heads;
        int r = (int)(t % routes); t /= routes;
        int b = (int)t;
        int keys = selected_blocks * block_size;
        float local = 0.0f;
        for (int h = (router_heads == 1 ? 0 : rh);
             h < heads;
             h += (router_heads == 1 ? 1 : heads)) {
            for (int s = 0; s < stride; ++s) {
                const __nv_bfloat16* row = dscores
                    + (((((long long)b * routes + r) * heads + h) * stride + s) * keys);
                for (int j = threadIdx.x; j < block_size; j += blockDim.x)
                    local += __bfloat162float(row[block * block_size + j]);
            }
        }
        extern __shared__ float sh[];
        sh[threadIdx.x] = local;
        __syncthreads();
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (threadIdx.x < d) sh[threadIdx.x] += sh[threadIdx.x + d];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float w = __bfloat162float(weights[out]);
            dweights[out] = sh[0] * (weight_scale / (w + weight_eps));
        }
    }
    """
    names = (
        "retrieval_bf16_gemm_nt",
        "retrieval_bf16_gemm_nn",
        "retrieval_bf16_gemm_tn",
        "retrieval_softmax_fwd_bf16",
        "retrieval_softmax_bwd_bf16",
        "retrieval_route_scatter_bf16_to_f32",
        "retrieval_kv_scatter_bf16_to_f32",
        "retrieval_bias_grad_bf16",
    )
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++14",),
            name_expressions=names,
        )
        for name in names:
            module.get_function(name)
        _RETRIEVAL_BF16_PIPELINE_MODULE = module
    except Exception:
        strict = os.environ.get(
            "MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE_STRICT", "0"
        ).strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _RETRIEVAL_BF16_PIPELINE_DISABLED = True
        return None
    return _RETRIEVAL_BF16_PIPELINE_MODULE


def _retrieval_bf16_shape_ok(*dims):
    return all(int(d) > 0 and int(d) % 16 == 0 for d in dims)


def _retrieval_bf16_gemm_nt(a, b, alpha=1.0):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(a.dtype) and is_bfloat16_dtype(b.dtype)):
        return None
    if a.ndim < 2 or b.ndim != a.ndim or a.shape[:-2] != b.shape[:-2]:
        return None
    M, K = map(int, a.shape[-2:])
    N, Kb = map(int, b.shape[-2:])
    if K != Kb or not _retrieval_bf16_shape_ok(M, N, K):
        return None
    a = xp.ascontiguousarray(a)
    b = xp.ascontiguousarray(b)
    batches = int(np.prod(a.shape[:-2], dtype=np.int64))
    out = xp.empty(a.shape[:-2] + (M, N), dtype=a.dtype)
    module.get_function("retrieval_bf16_gemm_nt")(
        ((N + 15) // 16, (M + 15) // 16, batches), (32,),
        (a, b, out, np.int32(batches), np.int32(M), np.int32(N), np.int32(K), np.float32(alpha)),
    )
    return out


def _retrieval_bf16_gemm_nn(a, b, alpha=1.0):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(a.dtype) and is_bfloat16_dtype(b.dtype)):
        return None
    if a.ndim < 2 or b.ndim != a.ndim or a.shape[:-2] != b.shape[:-2]:
        return None
    M, K = map(int, a.shape[-2:])
    Kb, N = map(int, b.shape[-2:])
    if K != Kb or not _retrieval_bf16_shape_ok(M, N, K):
        return None
    a = xp.ascontiguousarray(a)
    b = xp.ascontiguousarray(b)
    batches = int(np.prod(a.shape[:-2], dtype=np.int64))
    out = xp.empty(a.shape[:-2] + (M, N), dtype=a.dtype)
    module.get_function("retrieval_bf16_gemm_nn")(
        ((N + 15) // 16, (M + 15) // 16, batches), (32,),
        (a, b, out, np.int32(batches), np.int32(M), np.int32(N), np.int32(K), np.float32(alpha)),
    )
    return out


def _retrieval_bf16_gemm_tn(a_storage, b, alpha=1.0):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(a_storage.dtype) and is_bfloat16_dtype(b.dtype)):
        return None
    if a_storage.ndim < 2 or b.ndim != a_storage.ndim or a_storage.shape[:-2] != b.shape[:-2]:
        return None
    K, M = map(int, a_storage.shape[-2:])
    Kb, N = map(int, b.shape[-2:])
    if K != Kb or not _retrieval_bf16_shape_ok(M, N, K):
        return None
    a_storage = xp.ascontiguousarray(a_storage)
    b = xp.ascontiguousarray(b)
    batches = int(np.prod(a_storage.shape[:-2], dtype=np.int64))
    out = xp.empty(a_storage.shape[:-2] + (M, N), dtype=a_storage.dtype)
    module.get_function("retrieval_bf16_gemm_tn")(
        ((N + 15) // 16, (M + 15) // 16, batches), (32,),
        (a_storage, b, out, np.int32(batches), np.int32(M), np.int32(N), np.int32(K), np.float32(alpha)),
    )
    return out


def _retrieval_softmax_forward_bf16(
    scores, selected_weights, query_valid, n_router_heads, block_size,
    scale, weight_mode, weight_scale, weight_eps,
):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not is_bfloat16_dtype(scores.dtype):
        return None
    if not is_bfloat16_dtype(selected_weights.dtype):
        return None
    if scores.ndim != 5:
        return None
    batch, routes, heads, stride, keys = map(int, scores.shape)
    selected_blocks = int(selected_weights.shape[-1])
    if keys != selected_blocks * int(block_size):
        return None
    scores = xp.ascontiguousarray(scores)
    weights = xp.ascontiguousarray(selected_weights)
    query_valid = xp.ascontiguousarray(query_valid.astype(bool, copy=False))
    probs = xp.empty_like(scores)
    rows = batch * routes * heads * stride
    threads = 256
    module.get_function("retrieval_softmax_fwd_bf16")(
        (rows,), (threads,),
        (
            scores, weights, query_valid, probs, np.int64(rows),
            np.int32(routes), np.int32(heads), np.int32(stride), np.int32(keys),
            np.int32(n_router_heads), np.int32(selected_blocks), np.int32(block_size),
            np.float32(scale), np.int32(1 if weight_mode == "logit_bias" else 0),
            np.float32(weight_scale), np.float32(weight_eps),
        ),
        shared_mem=threads * 4,
    )
    return probs


def _retrieval_softmax_backward_bf16(dprobs, probs, query_valid):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(dprobs.dtype) and is_bfloat16_dtype(probs.dtype)):
        return None
    if dprobs.shape != probs.shape or probs.ndim != 5:
        return None
    batch, routes, heads, stride, keys = map(int, probs.shape)
    dprobs = xp.ascontiguousarray(dprobs)
    probs = xp.ascontiguousarray(probs)
    query_valid = xp.ascontiguousarray(query_valid.astype(bool, copy=False))
    dscores = xp.empty_like(probs)
    rows = batch * routes * heads * stride
    threads = 256
    module.get_function("retrieval_softmax_bwd_bf16")(
        (rows,), (threads,),
        (
            dprobs, probs, query_valid, dscores, np.int64(rows),
            np.int32(routes), np.int32(heads), np.int32(stride), np.int32(keys),
        ),
        shared_mem=threads * 4,
    )
    return dscores


def _retrieval_route_scatter_bf16_to_f32(src, dst, query_positions, query_valid):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not is_bfloat16_dtype(src.dtype) or dst.dtype != xp.float32:
        return False
    src = xp.ascontiguousarray(src)
    query_positions = xp.ascontiguousarray(query_positions)
    query_valid = xp.ascontiguousarray(query_valid.astype(bool, copy=False))
    batch, routes, heads, stride, d_head = map(int, src.shape)
    n = int(src.size)
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    module.get_function("retrieval_route_scatter_bf16_to_f32")(
        (blocks,), (threads,),
        (
            src, dst, query_positions, query_valid,
            np.int32(batch), np.int32(routes), np.int32(heads), np.int32(stride),
            np.int32(d_head), np.int32(dst.shape[1]),
        ),
    )
    return True


def _retrieval_kv_scatter_bf16_to_f32(
    dk_src, dv_src, dk_dst, dv_dst, key_indices, kv_map,
):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(dk_src.dtype) and is_bfloat16_dtype(dv_src.dtype)):
        return False
    if dk_dst.dtype != xp.float32 or dv_dst.dtype != xp.float32:
        return False
    dk_src = xp.ascontiguousarray(dk_src)
    dv_src = xp.ascontiguousarray(dv_src)
    key_indices = xp.ascontiguousarray(key_indices)
    kv_map = xp.ascontiguousarray(kv_map)
    batch, routes, heads, keys, d_head = map(int, dk_src.shape)
    n = int(dk_src.size)
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    module.get_function("retrieval_kv_scatter_bf16_to_f32")(
        (blocks,), (threads,),
        (
            dk_src, dv_src, dk_dst, dv_dst, key_indices, kv_map,
            np.int32(batch), np.int32(routes), np.int32(heads), np.int32(keys),
            np.int32(d_head), np.int32(dk_dst.shape[1]), np.int32(dk_dst.shape[2]),
        ),
    )
    return True


def _retrieval_bias_grad_bf16(
    dscores, selected_weights, n_router_heads, block_size, weight_scale, weight_eps,
):
    module = _get_retrieval_bf16_pipeline_module()
    if module is None or not (is_bfloat16_dtype(dscores.dtype) and is_bfloat16_dtype(selected_weights.dtype)):
        return None
    batch, routes, heads, stride, keys = map(int, dscores.shape)
    selected_blocks = int(selected_weights.shape[-1])
    if keys != selected_blocks * int(block_size):
        return None
    dscores = xp.ascontiguousarray(dscores)
    weights = xp.ascontiguousarray(selected_weights)
    out = xp.empty(selected_weights.shape, dtype=xp.float32)
    count = int(out.size)
    threads = 256
    module.get_function("retrieval_bias_grad_bf16")(
        (count,), (threads,),
        (
            dscores, weights, out,
            np.int32(batch), np.int32(routes), np.int32(heads), np.int32(stride),
            np.int32(n_router_heads), np.int32(selected_blocks), np.int32(block_size),
            np.float32(weight_scale), np.float32(weight_eps),
        ),
        shared_mem=threads * 4,
    )
    return out


def _fused_local_softmax_enabled():
    raw = os.environ.get("MINI_LLM_FUSED_LOCAL_SOFTMAX", "1").strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no"}


def _bf16_local_tensorcore_enabled(dtype):
    """Opt-in BF16 GEMM path for the large local-attention products.

    CuPy's generic N-D BF16 matmul has been unreliable on the target setup,
    while ordinary 2-D BF16 GEMM is validated at startup.  This path reshapes
    each batch/GQA group into large 2-D products so Q/K/V stay in BF16 storage
    and Tensor Cores can be used.  Softmax and its Jacobian remain FP32.

    The path is intentionally opt-in because the 2-D GEMM result is rounded to
    BF16 before promotion to FP32 softmax state, which is a small numerical
    change relative to the conservative all-FP32 attention-product path.
    """
    raw = os.environ.get("MINI_LLM_BF16_LOCAL_GEMM", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _get_local_softmax_module():
    """Lazily compile the CUDA causal softmax pair used by local attention.

    Compilation is deliberately lazy so NumPy/reference use never imports or
    requires CUDA.  If a particular CuPy/CUDA combination rejects the kernel,
    the code falls back to the existing vectorized implementation unless
    ``MINI_LLM_FUSED_LOCAL_SOFTMAX_STRICT=1`` is requested.
    """
    global _LOCAL_SOFTMAX_MODULE, _LOCAL_SOFTMAX_DISABLED
    if not _fused_local_softmax_enabled() or _LOCAL_SOFTMAX_DISABLED:
        return None
    if _LOCAL_SOFTMAX_MODULE is not None:
        return _LOCAL_SOFTMAX_MODULE
    code = r"""
    extern "C" __global__
    void local_causal_softmax_fwd(
        const float* scores, float* probs,
        int rows, int q_len, int k_len,
        int q_start, int key_start, int window, float logit_multiplier) {
        int row = blockIdx.x;
        if (row >= rows) return;
        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const float* s = scores + ((long long)row) * k_len;
        float* p = probs + ((long long)row) * k_len;
        extern __shared__ float sh[];

        // Avoid relying on host C/C++ headers in NVRTC.  CUDA device
        // intrinsics such as fmaxf/expf are available directly, while this
        // finite value is sufficient as the masked-softmax reduction seed.
        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1)
                local_max = fmaxf(local_max, s[j]);
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            float value = 0.0f;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                value = expf((s[j] - row_max) * logit_multiplier);
            }
            p[j] = value;
            local_sum += value;
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) p[j] *= inv;
    }

    extern "C" __global__
    void local_causal_softmax_bwd(
        const float* dprobs, const float* probs, float* dscores,
        int rows, int q_len, int k_len, int q_start, int key_start, int window) {
        int row = blockIdx.x;
        if (row >= rows) return;
        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const float* dp = dprobs + ((long long)row) * k_len;
        const float* p = probs + ((long long)row) * k_len;
        float* ds = dscores + ((long long)row) * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1)
                local_sum += dp[j] * p[j];
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            ds[j] = (k_pos <= q_pos && k_pos >= q_pos - window + 1)
                ? p[j] * (dp[j] - correction) : 0.0f;
        }
    }
    """
    try:
        # RawModule compilation is lazy: constructing the object can succeed
        # even when NVRTC compilation will fail later at get_function().
        # Resolve both kernels here so the advertised fallback actually catches
        # compiler/header/toolchain failures before entering the hot path.
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=("local_causal_softmax_fwd", "local_causal_softmax_bwd"),
        )
        module.get_function("local_causal_softmax_fwd")
        module.get_function("local_causal_softmax_bwd")
        _LOCAL_SOFTMAX_MODULE = module
    except Exception:
        strict = os.environ.get("MINI_LLM_FUSED_LOCAL_SOFTMAX_STRICT", "0").strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _LOCAL_SOFTMAX_DISABLED = True
        return None
    return _LOCAL_SOFTMAX_MODULE


def _local_causal_softmax_forward_cuda(scores, q_start, key_start, window, logit_multiplier):
    module = _get_local_softmax_module()
    if module is None or scores.dtype != xp.float32 or not scores.flags.c_contiguous:
        return None
    q_len = int(scores.shape[-2])
    k_len = int(scores.shape[-1])
    rows = int(scores.size // k_len)
    probs = xp.empty(scores.shape, dtype=xp.float32)
    threads = 256
    module.get_function("local_causal_softmax_fwd")(
        (rows,), (threads,),
        (
            scores,
            probs,
            np.int32(rows),
            np.int32(q_len),
            np.int32(k_len),
            np.int32(q_start),
            np.int32(key_start),
            np.int32(window),
            np.float32(logit_multiplier),
        ),
        shared_mem=threads * 4,
    )
    return probs


def _local_causal_softmax_backward_cuda(dprobs, probs, q_start, key_start, window):
    module = _get_local_softmax_module()
    if (
        module is None
        or dprobs.dtype != xp.float32
        or probs.dtype != xp.float32
        or not dprobs.flags.c_contiguous
        or not probs.flags.c_contiguous
    ):
        return None
    q_len = int(probs.shape[-2])
    k_len = int(probs.shape[-1])
    rows = int(probs.size // k_len)
    dscores = xp.empty(probs.shape, dtype=xp.float32)
    threads = 256
    module.get_function("local_causal_softmax_bwd")(
        (rows,), (threads,),
        (
            dprobs,
            probs,
            dscores,
            np.int32(rows),
            np.int32(q_len),
            np.int32(k_len),
            np.int32(q_start),
            np.int32(key_start),
            np.int32(window),
        ),
        shared_mem=threads * 4,
    )
    return dscores



def _fused_bf16_local_pipeline_enabled(dtype):
    """Use BF16-native local softmax I/O around Tensor-Core GEMMs.

    The ordinary fused local softmax keeps its interface in FP32.  With BF16
    Tensor-Core score/context GEMMs that forces several full attention-matrix
    casts per chunk.  This opt-in path performs those BF16<->FP32 conversions
    inside the softmax kernels instead, while max/sum/exp/Jacobian arithmetic
    remains FP32.
    """
    raw = os.environ.get("MINI_LLM_FUSED_BF16_LOCAL_PIPELINE", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _get_local_bf16_pipeline_module():
    global _LOCAL_BF16_PIPELINE_MODULE, _LOCAL_BF16_PIPELINE_DISABLED
    if BACKEND_NAME != "cupy" or _LOCAL_BF16_PIPELINE_DISABLED:
        return None
    if _LOCAL_BF16_PIPELINE_MODULE is not None:
        return _LOCAL_BF16_PIPELINE_MODULE

    code = r"""
    __device__ __forceinline__ float bf16_to_float(unsigned short x) {
        union { unsigned int u; float f; } v;
        v.u = ((unsigned int)x) << 16;
        return v.f;
    }

    __device__ __forceinline__ unsigned short float_to_bf16(float x) {
        union { unsigned int u; float f; } v;
        v.f = x;
        unsigned int bits = v.u;
        unsigned int lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return (unsigned short)(bits >> 16);
    }

    extern "C" __global__
    void local_causal_softmax_fwd_bf16(
        const unsigned short* scores,
        unsigned short* probs,
        int rows, int q_len, int k_len,
        int q_start, int key_start, int window, float scale) {
        int row = blockIdx.x;
        if (row >= rows) return;
        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const unsigned short* s = scores + ((long long)row) * k_len;
        unsigned short* p = probs + ((long long)row) * k_len;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(s[j]) * scale;
                local_max = fmaxf(local_max, value);
            }
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(s[j]) * scale;
                local_sum += expf(value - row_max);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;

        // Recompute exp here so no FP32 probability workspace is required.
        // Only the final normalized probability is rounded to BF16, matching
        // the existing context/cache boundary.
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            float out = 0.0f;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(s[j]) * scale;
                out = expf(value - row_max) * inv;
            }
            p[j] = float_to_bf16(out);
        }
    }

    extern "C" __global__
    void local_causal_softmax_bwd_bf16(
        const unsigned short* dprobs,
        const unsigned short* probs,
        unsigned short* dscores,
        int rows, int q_len, int k_len,
        int q_start, int key_start, int window) {
        int row = blockIdx.x;
        if (row >= rows) return;
        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const unsigned short* dp = dprobs + ((long long)row) * k_len;
        const unsigned short* p = probs + ((long long)row) * k_len;
        unsigned short* ds = dscores + ((long long)row) * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                local_sum += bf16_to_float(dp[j]) * bf16_to_float(p[j]);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            float out = 0.0f;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float pf = bf16_to_float(p[j]);
                out = pf * (bf16_to_float(dp[j]) - correction);
            }
            ds[j] = float_to_bf16(out);
        }
    }

    // 0053F: one launch covers the three static local-attention query chunks.
    // Each CUDA block still owns exactly one softmax row, so the numerical
    // reduction order is identical to the established per-chunk kernel.
    extern "C" __global__
    void local_causal_softmax_fwd3_bf16(
        const unsigned short* scores0, unsigned short* probs0,
        int rows0, int q_len0, int k_len0, int q_start0, int key_start0,
        const unsigned short* scores1, unsigned short* probs1,
        int rows1, int q_len1, int k_len1, int q_start1, int key_start1,
        const unsigned short* scores2, unsigned short* probs2,
        int rows2, int q_len2, int k_len2, int q_start2, int key_start2,
        int window, float scale) {
        int global_row = blockIdx.x;
        int total_rows = rows0 + rows1 + rows2;
        if (global_row >= total_rows) return;

        const unsigned short* scores;
        unsigned short* probs;
        int row, q_len, k_len, q_start, key_start;
        if (global_row < rows0) {
            scores = scores0; probs = probs0; row = global_row;
            q_len = q_len0; k_len = k_len0; q_start = q_start0; key_start = key_start0;
        } else if (global_row < rows0 + rows1) {
            scores = scores1; probs = probs1; row = global_row - rows0;
            q_len = q_len1; k_len = k_len1; q_start = q_start1; key_start = key_start1;
        } else {
            scores = scores2; probs = probs2; row = global_row - rows0 - rows1;
            q_len = q_len2; k_len = k_len2; q_start = q_start2; key_start = key_start2;
        }

        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const unsigned short* sp = scores + ((long long)row) * k_len;
        unsigned short* pp = probs + ((long long)row) * k_len;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(sp[j]) * scale;
                local_max = fmaxf(local_max, value);
            }
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(sp[j]) * scale;
                local_sum += expf(value - row_max);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            float out = 0.0f;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float value = bf16_to_float(sp[j]) * scale;
                out = expf(value - row_max) * inv;
            }
            pp[j] = float_to_bf16(out);
        }
    }

    extern "C" __global__
    void local_causal_softmax_bwd3_bf16(
        const unsigned short* dprobs0, const unsigned short* probs0, unsigned short* dscores0,
        int rows0, int q_len0, int k_len0, int q_start0, int key_start0,
        const unsigned short* dprobs1, const unsigned short* probs1, unsigned short* dscores1,
        int rows1, int q_len1, int k_len1, int q_start1, int key_start1,
        const unsigned short* dprobs2, const unsigned short* probs2, unsigned short* dscores2,
        int rows2, int q_len2, int k_len2, int q_start2, int key_start2,
        int window) {
        int global_row = blockIdx.x;
        int total_rows = rows0 + rows1 + rows2;
        if (global_row >= total_rows) return;

        const unsigned short* dprobs;
        const unsigned short* probs;
        unsigned short* dscores;
        int row, q_len, k_len, q_start, key_start;
        if (global_row < rows0) {
            dprobs = dprobs0; probs = probs0; dscores = dscores0; row = global_row;
            q_len = q_len0; k_len = k_len0; q_start = q_start0; key_start = key_start0;
        } else if (global_row < rows0 + rows1) {
            dprobs = dprobs1; probs = probs1; dscores = dscores1; row = global_row - rows0;
            q_len = q_len1; k_len = k_len1; q_start = q_start1; key_start = key_start1;
        } else {
            dprobs = dprobs2; probs = probs2; dscores = dscores2; row = global_row - rows0 - rows1;
            q_len = q_len2; k_len = k_len2; q_start = q_start2; key_start = key_start2;
        }

        int q_local = row % q_len;
        int q_pos = q_start + q_local;
        const unsigned short* dp = dprobs + ((long long)row) * k_len;
        const unsigned short* pp = probs + ((long long)row) * k_len;
        unsigned short* ds = dscores + ((long long)row) * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1)
                local_sum += bf16_to_float(dp[j]) * bf16_to_float(pp[j]);
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_pos = key_start + j;
            float out = 0.0f;
            if (k_pos <= q_pos && k_pos >= q_pos - window + 1) {
                float pf = bf16_to_float(pp[j]);
                out = pf * (bf16_to_float(dp[j]) - correction);
            }
            ds[j] = float_to_bf16(out);
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=(
                "local_causal_softmax_fwd_bf16",
                "local_causal_softmax_bwd_bf16",
                "local_causal_softmax_fwd3_bf16",
                "local_causal_softmax_bwd3_bf16",
            ),
        )
        module.get_function("local_causal_softmax_fwd_bf16")
        module.get_function("local_causal_softmax_bwd_bf16")
        module.get_function("local_causal_softmax_fwd3_bf16")
        module.get_function("local_causal_softmax_bwd3_bf16")
        _LOCAL_BF16_PIPELINE_MODULE = module
    except Exception:
        strict = os.environ.get(
            "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE_STRICT", "0"
        ).strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _LOCAL_BF16_PIPELINE_DISABLED = True
        return None
    return _LOCAL_BF16_PIPELINE_MODULE


def _local_causal_softmax_forward_bf16_cuda(
    scores_bf16, probs_bf16, q_start, key_start, window, scale
):
    if not _fused_bf16_local_pipeline_enabled(scores_bf16.dtype):
        return None
    module = _get_local_bf16_pipeline_module()
    if module is None:
        return None
    if scores_bf16.dtype != probs_bf16.dtype:
        return None
    if not is_bfloat16_dtype(scores_bf16.dtype):
        return None
    if scores_bf16.shape != probs_bf16.shape:
        return None
    if not scores_bf16.flags.c_contiguous or not probs_bf16.flags.c_contiguous:
        return None
    q_len = int(scores_bf16.shape[-2])
    k_len = int(scores_bf16.shape[-1])
    rows = int(scores_bf16.size // k_len)
    threads = 256
    module.get_function("local_causal_softmax_fwd_bf16")(
        (rows,),
        (threads,),
        (
            scores_bf16,
            probs_bf16,
            np.int32(rows),
            np.int32(q_len),
            np.int32(k_len),
            np.int32(q_start),
            np.int32(key_start),
            np.int32(window),
            np.float32(scale),
        ),
        shared_mem=threads * 4,
    )
    return probs_bf16


def _local_causal_softmax_backward_bf16_cuda(
    dprobs_bf16, probs_bf16, q_start, key_start, window
):
    if not _fused_bf16_local_pipeline_enabled(probs_bf16.dtype):
        return None
    module = _get_local_bf16_pipeline_module()
    if module is None:
        return None
    if dprobs_bf16.dtype != probs_bf16.dtype:
        return None
    if not is_bfloat16_dtype(probs_bf16.dtype):
        return None
    if dprobs_bf16.shape != probs_bf16.shape:
        return None
    if not dprobs_bf16.flags.c_contiguous or not probs_bf16.flags.c_contiguous:
        return None
    q_len = int(probs_bf16.shape[-2])
    k_len = int(probs_bf16.shape[-1])
    rows = int(probs_bf16.size // k_len)
    dscores_bf16 = xp.empty_like(probs_bf16)
    threads = 256
    module.get_function("local_causal_softmax_bwd_bf16")(
        (rows,),
        (threads,),
        (
            dprobs_bf16,
            probs_bf16,
            dscores_bf16,
            np.int32(rows),
            np.int32(q_len),
            np.int32(k_len),
            np.int32(q_start),
            np.int32(key_start),
            np.int32(window),
        ),
        shared_mem=threads * 4,
    )
    return dscores_bf16

def _multi_chunk_local_softmax_enabled():
    raw = os.environ.get(
        "MINI_LLM_FUSED_LOCAL_MULTI_CHUNK_SOFTMAX", "1"
    ).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _local_causal_softmax_forward3_bf16_cuda(chunks, window, scale):
    """0053F: process exactly three local chunks in one CUDA launch.

    The production 4096/1536 geometry and the GPU validator both have three
    chunks. Other geometries deliberately fall back to the established
    per-chunk kernels rather than changing their behavior.
    """
    if not _multi_chunk_local_softmax_enabled() or len(chunks) != 3:
        return False
    module = _get_local_bf16_pipeline_module()
    if module is None:
        return False
    args = []
    total_rows = 0
    for chunk in chunks:
        scores = chunk["scores"]
        probs = chunk["probs"]
        if (
            scores.dtype != probs.dtype
            or not is_bfloat16_dtype(scores.dtype)
            or scores.shape != probs.shape
            or not scores.flags.c_contiguous
            or not probs.flags.c_contiguous
        ):
            return False
        q_len = int(scores.shape[-2])
        k_len = int(scores.shape[-1])
        rows = int(scores.size // k_len)
        total_rows += rows
        args.extend((
            scores, probs,
            np.int32(rows), np.int32(q_len), np.int32(k_len),
            np.int32(chunk["q_start"]), np.int32(chunk["key_start"]),
        ))
    if total_rows <= 0:
        return True
    threads = 256
    args.extend((np.int32(window), np.float32(scale)))
    module.get_function("local_causal_softmax_fwd3_bf16")(
        (total_rows,), (threads,), tuple(args), shared_mem=threads * 4,
    )
    return True


def _local_causal_softmax_backward3_bf16_cuda(chunks, window):
    """0053F: process three local softmax Jacobian products in one launch."""
    if not _multi_chunk_local_softmax_enabled() or len(chunks) != 3:
        return False
    module = _get_local_bf16_pipeline_module()
    if module is None:
        return False
    args = []
    total_rows = 0
    allocated = []
    for chunk in chunks:
        dprobs = chunk["dprobs"]
        probs = chunk["probs"]
        if (
            dprobs.dtype != probs.dtype
            or not is_bfloat16_dtype(probs.dtype)
            or dprobs.shape != probs.shape
            or not dprobs.flags.c_contiguous
            or not probs.flags.c_contiguous
        ):
            return False
        dscores = xp.empty_like(probs)
        allocated.append(dscores)
        q_len = int(probs.shape[-2])
        k_len = int(probs.shape[-1])
        rows = int(probs.size // k_len)
        total_rows += rows
        args.extend((
            dprobs, probs, dscores,
            np.int32(rows), np.int32(q_len), np.int32(k_len),
            np.int32(chunk["q_start"]), np.int32(chunk["key_start"]),
        ))
    if total_rows <= 0:
        for chunk, dscores in zip(chunks, allocated):
            chunk["dscores"] = dscores
        return True
    threads = 256
    args.append(np.int32(window))
    module.get_function("local_causal_softmax_bwd3_bf16")(
        (total_rows,), (threads,), tuple(args), shared_mem=threads * 4,
    )
    for chunk, dscores in zip(chunks, allocated):
        chunk["dscores"] = dscores
    return True


def _resolve_query_chunk_size(query_length, query_chunk_size=None):
    """Resolve bounded query chunking for the indexed attention hot path.

    ``KeySelectionPlan`` can expose thousands of keys per query. Gathering all
    selected K/V vectors for a full long sequence would create a
    ``[B,H,T,K,D]`` tensor, which is far larger than the actual attention
    probability matrix. Chunking only the query axis keeps the mathematical
    result unchanged while bounding those transient gather/product buffers.

    The environment override is intentionally a runtime knob so GPU-memory
    experiments do not require changing model/checkpoint configuration.
    """
    if query_chunk_size is None:
        raw = os.environ.get("MINI_LLM_INDEXED_ATTN_QUERY_CHUNK")
        query_chunk_size = (
            _DEFAULT_QUERY_CHUNK_SIZE if raw is None else int(raw)
        )
    query_chunk_size = int(query_chunk_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    return min(int(query_length), query_chunk_size)






def _resolve_local_query_chunk_size(query_length, query_chunk_size=None):
    """Resolve the query chunk size for specialized local attention.

    The contiguous local kernel does not materialize per-query ``[Q,K,D]``
    gathers, so it can safely use a substantially larger chunk than generic
    indexed attention.  Larger chunks reduce Python/CUDA launch overhead and
    expose larger regular GEMMs.  Keep a separate runtime override because
    the best value depends on context/window size and available VRAM.
    """
    if query_chunk_size is None:
        raw = os.environ.get("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK")
        query_chunk_size = (
            _DEFAULT_LOCAL_QUERY_CHUNK_SIZE if raw is None else int(raw)
        )
    query_chunk_size = int(query_chunk_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    return min(int(query_length), query_chunk_size)


def _resolve_dilated_query_chunk_size(query_length, query_chunk_size=None):
    """Resolve the reduced-sequence chunk size for specialized dilation.

    Dilated attention is evaluated independently for each residue class, so
    ``query_length`` here is the length of one downsampled phase rather than
    the original token sequence.  A much larger chunk than generic indexed
    attention is therefore both memory-safe and important for GPU occupancy.
    """
    if query_chunk_size is None:
        raw = os.environ.get("MINI_LLM_DILATED_ATTN_QUERY_CHUNK")
        query_chunk_size = (
            _DEFAULT_DILATED_QUERY_CHUNK_SIZE if raw is None else int(raw)
        )
    query_chunk_size = int(query_chunk_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    return min(int(query_length), query_chunk_size)


def _query_chunks(query_length, query_chunk_size):
    for start in range(0, int(query_length), int(query_chunk_size)):
        yield start, min(start + int(query_chunk_size), int(query_length))




_INDEXED_SOFTMAX_MODULE = None
_INDEXED_SOFTMAX_DISABLED = False


def _fused_indexed_softmax_enabled():
    """Use fused CUDA masked-softmax kernels for indexed attention paths."""
    raw = os.environ.get("MINI_LLM_FUSED_INDEXED_SOFTMAX", "0").strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no"}


def _get_indexed_softmax_module():
    """Compile generic FP32 masked-softmax CUDA kernels lazily.

    The kernels accept a compact broadcast mask rather than materializing a
    score-sized boolean tensor.  Shapes up to five dimensions cover every
    indexed-attention layout currently used by the model:

      generic/dilated/global: [B,H,Q,K]
      block retrieval:        [B,R,H,Q,K]

    Softmax reductions stay FP32 and empty rows return exact zeros, matching
    :func:`_masked_softmax_forward`.
    """
    global _INDEXED_SOFTMAX_MODULE, _INDEXED_SOFTMAX_DISABLED
    if not _fused_indexed_softmax_enabled() or _INDEXED_SOFTMAX_DISABLED:
        return None
    if _INDEXED_SOFTMAX_MODULE is not None:
        return _INDEXED_SOFTMAX_MODULE

    code = r"""
    __device__ __forceinline__ long long mask_offset_5d(
        long long row, int col,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        int i3 = (int)(row % d3); row /= d3;
        int i2 = (int)(row % d2); row /= d2;
        int i1 = (int)(row % d1); row /= d1;
        int i0 = (int)row;
        long long s4 = 1;
        long long s3 = (long long)m4;
        long long s2 = (long long)m3 * s3;
        long long s1 = (long long)m2 * s2;
        long long s0 = (long long)m1 * s1;
        return (long long)(m0 == 1 ? 0 : i0) * s0
             + (long long)(m1 == 1 ? 0 : i1) * s1
             + (long long)(m2 == 1 ? 0 : i2) * s2
             + (long long)(m3 == 1 ? 0 : i3) * s3
             + (long long)(m4 == 1 ? 0 : col) * s4;
    }

    extern "C" __global__
    void indexed_masked_softmax_fwd_f32(
        const float* scores, const unsigned char* mask, float* probs,
        long long rows, int k_len, float logit_multiplier,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        const float* s = scores + row * k_len;
        float* p = probs + row * k_len;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        int any_valid = 0;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            if (mask[mo]) {
                local_max = fmaxf(local_max, s[j]);
                any_valid = 1;
            }
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        // Use the second half of shared memory for a validity reduction.
        float* sh_valid = sh + blockDim.x;
        sh_valid[threadIdx.x] = (float)any_valid;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh_valid[threadIdx.x] += sh_valid[threadIdx.x + stride];
            __syncthreads();
        }
        if (sh_valid[0] == 0.0f) {
            for (int j = threadIdx.x; j < k_len; j += blockDim.x) p[j] = 0.0f;
            return;
        }

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            float value = 0.0f;
            if (mask[mo]) value = expf((s[j] - row_max) * logit_multiplier);
            p[j] = value;
            local_sum += value;
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) p[j] *= inv;
    }

    extern "C" __global__
    void indexed_masked_softmax_bwd_f32(
        const float* dprobs, const float* probs,
        const unsigned char* mask, float* dscores,
        long long rows, int k_len,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        const float* dp = dprobs + row * k_len;
        const float* p = probs + row * k_len;
        float* ds = dscores + row * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            if (mask[mo]) local_sum += dp[j] * p[j];
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            ds[j] = mask[mo] ? p[j] * (dp[j] - correction) : 0.0f;
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=(
                "indexed_masked_softmax_fwd_f32",
                "indexed_masked_softmax_bwd_f32",
            ),
        )
        module.get_function("indexed_masked_softmax_fwd_f32")
        module.get_function("indexed_masked_softmax_bwd_f32")
        _INDEXED_SOFTMAX_MODULE = module
    except Exception:
        strict = os.environ.get(
            "MINI_LLM_FUSED_INDEXED_SOFTMAX_STRICT", "0"
        ).strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _INDEXED_SOFTMAX_DISABLED = True
        return None
    return _INDEXED_SOFTMAX_MODULE


def _indexed_softmax_layout(scores, valid_mask):
    """Return the compact 5-D broadcast descriptor used by CUDA kernels."""
    if scores.ndim < 2 or scores.ndim > 5:
        return None
    if valid_mask.ndim > scores.ndim:
        return None
    score_shape = (1,) * (5 - scores.ndim) + tuple(map(int, scores.shape))
    mask_shape_raw = (1,) * (scores.ndim - valid_mask.ndim) + tuple(
        map(int, valid_mask.shape)
    )
    mask_shape = (1,) * (5 - scores.ndim) + mask_shape_raw
    for d, m in zip(score_shape, mask_shape):
        if m not in (1, d):
            return None
    # Last axis is the key axis. A singleton is allowed for route-valid masks.
    return score_shape, mask_shape


def _masked_softmax_forward_cuda(scores, valid_mask, logit_multiplier=1.0):
    module = _get_indexed_softmax_module()
    if module is None or scores.dtype != xp.float32:
        return None
    if not scores.flags.c_contiguous:
        return None
    layout = _indexed_softmax_layout(scores, valid_mask)
    if layout is None:
        return None
    score_shape, mask_shape = layout
    mask = valid_mask.astype(bool, copy=False)
    if not mask.flags.c_contiguous:
        mask = xp.ascontiguousarray(mask)
    k_len = int(scores.shape[-1])
    rows = int(scores.size // k_len)
    if rows == 0 or k_len == 0:
        return xp.zeros_like(scores, dtype=xp.float32)
    probs = xp.empty_like(scores, dtype=xp.float32)
    threads = 256
    # One block per row. Current indexed layouts stay well below CUDA's 1-D
    # grid limit; retain a defensive fallback for pathological shapes.
    if rows > 2147483647:
        return None
    module.get_function("indexed_masked_softmax_fwd_f32")(
        (rows,), (threads,),
        (
            scores, mask, probs,
            np.int64(rows), np.int32(k_len), np.float32(logit_multiplier),
            np.int32(score_shape[1]), np.int32(score_shape[2]), np.int32(score_shape[3]),
            np.int32(mask_shape[0]), np.int32(mask_shape[1]), np.int32(mask_shape[2]),
            np.int32(mask_shape[3]), np.int32(mask_shape[4]),
        ),
        shared_mem=threads * 2 * 4,
    )
    return probs


def _masked_softmax_backward_cuda(dprobs, probs, valid_mask):
    module = _get_indexed_softmax_module()
    if module is None or dprobs.dtype != xp.float32 or probs.dtype != xp.float32:
        return None
    if dprobs.shape != probs.shape:
        return None
    if not dprobs.flags.c_contiguous or not probs.flags.c_contiguous:
        return None
    layout = _indexed_softmax_layout(probs, valid_mask)
    if layout is None:
        return None
    score_shape, mask_shape = layout
    mask = valid_mask.astype(bool, copy=False)
    if not mask.flags.c_contiguous:
        mask = xp.ascontiguousarray(mask)
    k_len = int(probs.shape[-1])
    rows = int(probs.size // k_len)
    if rows == 0 or k_len == 0:
        return xp.zeros_like(probs, dtype=xp.float32)
    dscores = xp.empty_like(probs, dtype=xp.float32)
    threads = 256
    if rows > 2147483647:
        return None
    module.get_function("indexed_masked_softmax_bwd_f32")(
        (rows,), (threads,),
        (
            dprobs, probs, mask, dscores,
            np.int64(rows), np.int32(k_len),
            np.int32(score_shape[1]), np.int32(score_shape[2]), np.int32(score_shape[3]),
            np.int32(mask_shape[0]), np.int32(mask_shape[1]), np.int32(mask_shape[2]),
            np.int32(mask_shape[3]), np.int32(mask_shape[4]),
        ),
        shared_mem=threads * 4,
    )
    return dscores


def _masked_softmax_backward(dprobs, probs, valid_mask):
    """Masked softmax Jacobian with an optional fused CUDA fast path."""
    fused = _masked_softmax_backward_cuda(dprobs, probs, valid_mask)
    if fused is not None:
        return fused
    dscores = _softmax_backward(dprobs, probs)
    return xp.where(valid_mask, dscores, 0.0)

def _masked_softmax_forward(scores, valid_mask, logit_multiplier=1.0):
    """Stable softmax that returns exactly zero for rows with no valid keys."""
    work = (
        scores.astype("float32", copy=False)
        if is_low_precision_dtype(scores.dtype)
        else scores
    )
    valid_mask = valid_mask.astype(bool, copy=False)
    fused = _masked_softmax_forward_cuda(work, valid_mask, logit_multiplier)
    if fused is not None:
        return fused
    has_valid = xp.any(valid_mask, axis=-1, keepdims=True)
    neg_inf = xp.asarray(-xp.inf, dtype=work.dtype)
    masked = xp.where(valid_mask, work, neg_inf)
    row_max = xp.max(masked, axis=-1, keepdims=True)
    row_max = xp.where(has_valid, row_max, 0.0)
    shifted = work - row_max
    if logit_multiplier != 1.0:
        shifted = shifted * logit_multiplier
    exp_scores = xp.where(valid_mask, xp.exp(shifted), 0.0)
    denom = xp.sum(exp_scores, axis=-1, keepdims=True)
    safe_denom = xp.where(has_valid, denom, 1.0)
    return exp_scores / safe_denom



_INDEXED_BF16_PIPELINE_MODULE = None
_INDEXED_BF16_PIPELINE_DISABLED = False


def _bf16_indexed_pipeline_enabled(dtype):
    """Use BF16 storage around regular indexed-attention Tensor-Core GEMMs.

    This path is intended for regular sparse patterns (currently dilated and
    global-sparse attention) where each attention product can be expressed as
    a small number of large 2-D BF16 GEMMs.  Softmax reductions remain FP32
    inside CUDA while score/probability/score-gradient matrices stay BF16.
    """
    raw = os.environ.get(
        "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE", "0"
    ).strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _get_indexed_bf16_pipeline_module():
    global _INDEXED_BF16_PIPELINE_MODULE, _INDEXED_BF16_PIPELINE_DISABLED
    if BACKEND_NAME != "cupy" or _INDEXED_BF16_PIPELINE_DISABLED:
        return None
    if _INDEXED_BF16_PIPELINE_MODULE is not None:
        return _INDEXED_BF16_PIPELINE_MODULE

    code = r"""
    __device__ __forceinline__ float bf16_to_float(unsigned short x) {
        union { unsigned int u; float f; } v;
        v.u = ((unsigned int)x) << 16;
        return v.f;
    }

    __device__ __forceinline__ unsigned short float_to_bf16(float x) {
        union { unsigned int u; float f; } v;
        v.f = x;
        unsigned int bits = v.u;
        unsigned int lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return (unsigned short)(bits >> 16);
    }

    __device__ __forceinline__ long long mask_offset_5d_bf16(
        long long row, int col,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        int i3 = (int)(row % d3); row /= d3;
        int i2 = (int)(row % d2); row /= d2;
        int i1 = (int)(row % d1); row /= d1;
        int i0 = (int)row;
        long long s4 = 1;
        long long s3 = (long long)m4;
        long long s2 = (long long)m3 * s3;
        long long s1 = (long long)m2 * s2;
        long long s0 = (long long)m1 * s1;
        return (long long)(m0 == 1 ? 0 : i0) * s0
             + (long long)(m1 == 1 ? 0 : i1) * s1
             + (long long)(m2 == 1 ? 0 : i2) * s2
             + (long long)(m3 == 1 ? 0 : i3) * s3
             + (long long)(m4 == 1 ? 0 : col);
    }

    extern "C" __global__
    void indexed_masked_softmax_fwd_bf16(
        const unsigned short* scores, const unsigned char* mask,
        unsigned short* probs,
        long long rows, int k_len, float score_scale,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        const unsigned short* s = scores + row * k_len;
        unsigned short* p = probs + row * k_len;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        int any_valid = 0;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d_bf16(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            if (mask[mo]) {
                local_max = fmaxf(local_max, bf16_to_float(s[j]) * score_scale);
                any_valid = 1;
            }
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float* sh_valid = sh + blockDim.x;
        sh_valid[threadIdx.x] = (float)any_valid;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh_valid[threadIdx.x] += sh_valid[threadIdx.x + stride];
            __syncthreads();
        }
        if (sh_valid[0] == 0.0f) {
            for (int j = threadIdx.x; j < k_len; j += blockDim.x)
                p[j] = (unsigned short)0;
            return;
        }

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d_bf16(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            if (mask[mo]) {
                float value = bf16_to_float(s[j]) * score_scale;
                local_sum += expf(value - row_max);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d_bf16(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            float out = 0.0f;
            if (mask[mo]) {
                float value = bf16_to_float(s[j]) * score_scale;
                out = expf(value - row_max) * inv;
            }
            p[j] = float_to_bf16(out);
        }
    }

    extern "C" __global__
    void indexed_masked_softmax_bwd_bf16(
        const unsigned short* dprobs, const unsigned short* probs,
        const unsigned char* mask, unsigned short* dscores,
        long long rows, int k_len,
        int d1, int d2, int d3,
        int m0, int m1, int m2, int m3, int m4) {
        long long row = (long long)blockIdx.x;
        if (row >= rows) return;
        const unsigned short* dp = dprobs + row * k_len;
        const unsigned short* p = probs + row * k_len;
        unsigned short* ds = dscores + row * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d_bf16(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            if (mask[mo])
                local_sum += bf16_to_float(dp[j]) * bf16_to_float(p[j]);
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            long long mo = mask_offset_5d_bf16(
                row, j, d1, d2, d3, m0, m1, m2, m3, m4);
            float out = 0.0f;
            if (mask[mo]) {
                float pf = bf16_to_float(p[j]);
                out = pf * (bf16_to_float(dp[j]) - correction);
            }
            ds[j] = float_to_bf16(out);
        }
    }

    extern "C" __global__
    void dilated_softmax_fwd4_inplace_bf16(
        unsigned short* buf0, long long rows0, int q_len0, int k_len0,
        int phase_start0, int key_start0, int alignment_shift0,
        unsigned short* buf1, long long rows1, int q_len1, int k_len1,
        int phase_start1, int key_start1, int alignment_shift1,
        unsigned short* buf2, long long rows2, int q_len2, int k_len2,
        int phase_start2, int key_start2, int alignment_shift2,
        unsigned short* buf3, long long rows3, int q_len3, int k_len3,
        int phase_start3, int key_start3, int alignment_shift3,
        int key_slots, float score_scale) {
        long long global_row = (long long)blockIdx.x;
        long long total_rows = rows0 + rows1 + rows2 + rows3;
        if (global_row >= total_rows) return;

        unsigned short* buf;
        long long row;
        int q_len, k_len, phase_start, key_start, alignment_shift;
        if (global_row < rows0) {
            buf = buf0; row = global_row;
            q_len = q_len0; k_len = k_len0;
            phase_start = phase_start0; key_start = key_start0;
            alignment_shift = alignment_shift0;
        } else if (global_row < rows0 + rows1) {
            buf = buf1; row = global_row - rows0;
            q_len = q_len1; k_len = k_len1;
            phase_start = phase_start1; key_start = key_start1;
            alignment_shift = alignment_shift1;
        } else if (global_row < rows0 + rows1 + rows2) {
            buf = buf2; row = global_row - rows0 - rows1;
            q_len = q_len2; k_len = k_len2;
            phase_start = phase_start2; key_start = key_start2;
            alignment_shift = alignment_shift2;
        } else {
            buf = buf3; row = global_row - rows0 - rows1 - rows2;
            q_len = q_len3; k_len = k_len3;
            phase_start = phase_start3; key_start = key_start3;
            alignment_shift = alignment_shift3;
        }

        int q_local = (int)(row % q_len);
        int q_phase = phase_start + q_local;
        int max_key = q_phase + alignment_shift;
        unsigned short* data = buf + row * k_len;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        int any_valid = 0;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_phase = key_start + j;
            if (k_phase <= max_key && k_phase >= max_key - key_slots + 1) {
                local_max = fmaxf(
                    local_max, bf16_to_float(data[j]) * score_scale
                );
                any_valid = 1;
            }
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(
                    sh[threadIdx.x], sh[threadIdx.x + stride]
                );
            __syncthreads();
        }
        float row_max = sh[0];

        float* sh_valid = sh + blockDim.x;
        sh_valid[threadIdx.x] = (float)any_valid;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh_valid[threadIdx.x] += sh_valid[threadIdx.x + stride];
            __syncthreads();
        }
        if (sh_valid[0] == 0.0f) {
            for (int j = threadIdx.x; j < k_len; j += blockDim.x)
                data[j] = (unsigned short)0;
            return;
        }

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_phase = key_start + j;
            if (k_phase <= max_key && k_phase >= max_key - key_slots + 1) {
                float value = bf16_to_float(data[j]) * score_scale;
                local_sum += expf(value - row_max);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_phase = key_start + j;
            float out = 0.0f;
            if (k_phase <= max_key && k_phase >= max_key - key_slots + 1) {
                float value = bf16_to_float(data[j]) * score_scale;
                out = expf(value - row_max) * inv;
            }
            data[j] = float_to_bf16(out);
        }
    }

    extern "C" __global__
    void dilated_softmax_bwd4_inplace_bf16(
        unsigned short* dprobs0, const unsigned short* probs0,
        long long rows0, int q_len0, int k_len0,
        int phase_start0, int key_start0, int alignment_shift0,
        unsigned short* dprobs1, const unsigned short* probs1,
        long long rows1, int q_len1, int k_len1,
        int phase_start1, int key_start1, int alignment_shift1,
        unsigned short* dprobs2, const unsigned short* probs2,
        long long rows2, int q_len2, int k_len2,
        int phase_start2, int key_start2, int alignment_shift2,
        unsigned short* dprobs3, const unsigned short* probs3,
        long long rows3, int q_len3, int k_len3,
        int phase_start3, int key_start3, int alignment_shift3,
        int key_slots) {
        long long global_row = (long long)blockIdx.x;
        long long total_rows = rows0 + rows1 + rows2 + rows3;
        if (global_row >= total_rows) return;

        unsigned short* dp;
        const unsigned short* p;
        long long row;
        int q_len, k_len, phase_start, key_start, alignment_shift;
        if (global_row < rows0) {
            dp = dprobs0; p = probs0; row = global_row;
            q_len = q_len0; k_len = k_len0;
            phase_start = phase_start0; key_start = key_start0;
            alignment_shift = alignment_shift0;
        } else if (global_row < rows0 + rows1) {
            dp = dprobs1; p = probs1; row = global_row - rows0;
            q_len = q_len1; k_len = k_len1;
            phase_start = phase_start1; key_start = key_start1;
            alignment_shift = alignment_shift1;
        } else if (global_row < rows0 + rows1 + rows2) {
            dp = dprobs2; p = probs2; row = global_row - rows0 - rows1;
            q_len = q_len2; k_len = k_len2;
            phase_start = phase_start2; key_start = key_start2;
            alignment_shift = alignment_shift2;
        } else {
            dp = dprobs3; p = probs3; row = global_row - rows0 - rows1 - rows2;
            q_len = q_len3; k_len = k_len3;
            phase_start = phase_start3; key_start = key_start3;
            alignment_shift = alignment_shift3;
        }

        int q_local = (int)(row % q_len);
        int q_phase = phase_start + q_local;
        int max_key = q_phase + alignment_shift;
        unsigned short* dp_row = dp + row * k_len;
        const unsigned short* p_row = p + row * k_len;
        extern __shared__ float sh[];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_phase = key_start + j;
            if (k_phase <= max_key && k_phase >= max_key - key_slots + 1)
                local_sum += (
                    bf16_to_float(dp_row[j]) * bf16_to_float(p_row[j])
                );
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float correction = sh[0];
        for (int j = threadIdx.x; j < k_len; j += blockDim.x) {
            int k_phase = key_start + j;
            float out = 0.0f;
            if (k_phase <= max_key && k_phase >= max_key - key_slots + 1) {
                float pf = bf16_to_float(p_row[j]);
                out = pf * (bf16_to_float(dp_row[j]) - correction);
            }
            dp_row[j] = float_to_bf16(out);
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=(
                "indexed_masked_softmax_fwd_bf16",
                "indexed_masked_softmax_bwd_bf16",
                "dilated_softmax_fwd4_inplace_bf16",
                "dilated_softmax_bwd4_inplace_bf16",
            ),
        )
        module.get_function("indexed_masked_softmax_fwd_bf16")
        module.get_function("indexed_masked_softmax_bwd_bf16")
        module.get_function("dilated_softmax_fwd4_inplace_bf16")
        module.get_function("dilated_softmax_bwd4_inplace_bf16")
        _INDEXED_BF16_PIPELINE_MODULE = module
    except Exception:
        strict = os.environ.get(
            "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE_STRICT", "0"
        ).strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _INDEXED_BF16_PIPELINE_DISABLED = True
        return None
    return _INDEXED_BF16_PIPELINE_MODULE


def _masked_softmax_forward_bf16_cuda(scores, valid_mask, score_scale):
    if not _bf16_indexed_pipeline_enabled(scores.dtype):
        return None
    module = _get_indexed_bf16_pipeline_module()
    if module is None or not scores.flags.c_contiguous:
        return None
    layout = _indexed_softmax_layout(scores, valid_mask)
    if layout is None:
        return None
    score_shape, mask_shape = layout
    mask = valid_mask.astype(bool, copy=False)
    if not mask.flags.c_contiguous:
        mask = xp.ascontiguousarray(mask)
    k_len = int(scores.shape[-1])
    rows = int(scores.size // k_len)
    if rows == 0 or k_len == 0:
        return xp.zeros_like(scores)
    if rows > 2147483647:
        return None
    probs = xp.empty_like(scores)
    threads = 256
    module.get_function("indexed_masked_softmax_fwd_bf16")(
        (rows,), (threads,),
        (
            scores, mask, probs,
            np.int64(rows), np.int32(k_len), np.float32(score_scale),
            np.int32(score_shape[1]), np.int32(score_shape[2]), np.int32(score_shape[3]),
            np.int32(mask_shape[0]), np.int32(mask_shape[1]), np.int32(mask_shape[2]),
            np.int32(mask_shape[3]), np.int32(mask_shape[4]),
        ),
        shared_mem=threads * 2 * 4,
    )
    return probs


def _masked_softmax_backward_bf16_cuda(dprobs, probs, valid_mask):
    if not _bf16_indexed_pipeline_enabled(probs.dtype):
        return None
    module = _get_indexed_bf16_pipeline_module()
    if module is None:
        return None
    if dprobs.dtype != probs.dtype or dprobs.shape != probs.shape:
        return None
    if not dprobs.flags.c_contiguous or not probs.flags.c_contiguous:
        return None
    layout = _indexed_softmax_layout(probs, valid_mask)
    if layout is None:
        return None
    score_shape, mask_shape = layout
    mask = valid_mask.astype(bool, copy=False)
    if not mask.flags.c_contiguous:
        mask = xp.ascontiguousarray(mask)
    k_len = int(probs.shape[-1])
    rows = int(probs.size // k_len)
    if rows == 0 or k_len == 0:
        return xp.zeros_like(probs)
    if rows > 2147483647:
        return None
    dscores = xp.empty_like(probs)
    threads = 256
    module.get_function("indexed_masked_softmax_bwd_bf16")(
        (rows,), (threads,),
        (
            dprobs, probs, mask, dscores,
            np.int64(rows), np.int32(k_len),
            np.int32(score_shape[1]), np.int32(score_shape[2]), np.int32(score_shape[3]),
            np.int32(mask_shape[0]), np.int32(mask_shape[1]), np.int32(mask_shape[2]),
            np.int32(mask_shape[3]), np.int32(mask_shape[4]),
        ),
        shared_mem=threads * 4,
    )
    return dscores


def _multi_chunk_dilated_softmax_enabled():
    """Enable 0054A four-phase in-place dilated softmax fusion."""
    raw = os.environ.get(
        "MINI_LLM_FUSED_DILATED_MULTI_CHUNK_SOFTMAX", "1"
    ).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _dilated_softmax_forward4_inplace_bf16_cuda(chunks, key_slots, score_scale):
    """0054A: normalize four dilated phase buffers in one in-place launch.

    The fixed-phase mask is reconstructed analytically from phase metadata, so
    no boolean ``valid`` matrices are allocated or retained for backward.
    Reusing each BF16 score allocation as its probability cache also removes
    the extra score->probability allocation at the softmax boundary.
    """
    if not _multi_chunk_dilated_softmax_enabled() or len(chunks) != 4:
        return False
    module = _get_indexed_bf16_pipeline_module()
    if module is None:
        return False

    args = []
    total_rows = 0
    for chunk in chunks:
        scores = chunk.get("scores")
        if (
            scores is None
            or not is_bfloat16_dtype(scores.dtype)
            or scores.ndim != 4
            or not scores.flags.c_contiguous
        ):
            return False
        q_len = int(scores.shape[-2])
        k_len = int(scores.shape[-1])
        rows = int(scores.size // k_len)
        if rows <= 0 or k_len <= 0:
            return False
        total_rows += rows
        args.extend((
            scores, np.int64(rows), np.int32(q_len), np.int32(k_len),
            np.int32(chunk["phase_start"]), np.int32(chunk["key_start"]),
            np.int32(chunk["alignment_shift"]),
        ))
    if total_rows > 2147483647:
        return False

    args.extend((np.int32(key_slots), np.float32(score_scale)))
    threads = 256
    module.get_function("dilated_softmax_fwd4_inplace_bf16")(
        (total_rows,), (threads,), tuple(args),
        shared_mem=threads * 2 * 4,
    )
    for chunk in chunks:
        chunk["probs"] = chunk.pop("scores")
        chunk["valid"] = None
    return True


def _dilated_softmax_backward4_inplace_bf16_cuda(chunks, key_slots):
    """0054A: four softmax Jacobian products in-place on dP buffers."""
    if not _multi_chunk_dilated_softmax_enabled() or len(chunks) != 4:
        return False
    module = _get_indexed_bf16_pipeline_module()
    if module is None:
        return False

    args = []
    total_rows = 0
    for chunk in chunks:
        dprobs = chunk.get("dprobs")
        probs = chunk.get("probs")
        if (
            dprobs is None
            or probs is None
            or dprobs.dtype != probs.dtype
            or not is_bfloat16_dtype(probs.dtype)
            or dprobs.shape != probs.shape
            or dprobs.ndim != 4
            or not dprobs.flags.c_contiguous
            or not probs.flags.c_contiguous
        ):
            return False
        q_len = int(probs.shape[-2])
        k_len = int(probs.shape[-1])
        rows = int(probs.size // k_len)
        if rows <= 0 or k_len <= 0:
            return False
        total_rows += rows
        args.extend((
            dprobs, probs, np.int64(rows), np.int32(q_len), np.int32(k_len),
            np.int32(chunk["phase_start"]), np.int32(chunk["key_start"]),
            np.int32(chunk["alignment_shift"]),
        ))
    if total_rows > 2147483647:
        return False

    args.append(np.int32(key_slots))
    threads = 256
    module.get_function("dilated_softmax_bwd4_inplace_bf16")(
        (total_rows,), (threads,), tuple(args),
        shared_mem=threads * 4,
    )
    for chunk in chunks:
        chunk["dscores"] = chunk.pop("dprobs")
    return True


def _softmax_backward(dprobs, probs):
    correction = xp.sum(dprobs * probs, axis=-1, keepdims=True)
    return probs * (dprobs - correction)


def _cache_probs_for_backward(probs, source_dtype):
    """Store BF16 attention probabilities compactly after FP32 compute.

    Softmax itself stays FP32.  Only the persistent backward cache is reduced
    to BF16, whose FP32-like exponent range avoids the underflow concern that
    makes the same transformation unsafe for FP16.
    """
    if is_bfloat16_dtype(source_dtype):
        return probs.astype(source_dtype, copy=False)
    return probs


def _restore_cached_probs(probs, bf16_attention):
    """Promote compact BF16 probability caches only for active backward work."""
    if bf16_attention and probs is not None:
        return probs.astype("float32")
    return probs


def _default_kv_head_indices(n_q_heads, n_kv_heads):
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            "n_q_heads must be divisible by n_kv_heads when kv_head_indices "
            "is not supplied"
        )
    group_size = n_q_heads // n_kv_heads
    return xp.arange(n_q_heads, dtype=xp.int64) // group_size


def indexed_attention_forward(
    q,
    k,
    v,
    plan,
    kv_head_indices=None,
    scale=None,
    return_cache=True,
    query_chunk_size=None,
):
    """Attend over exact selected keys while retaining native GQA K/V storage.

    The query dimension is evaluated in bounded chunks.  This is essential for
    long sparse contexts: materializing all selected K/V vectors at once would
    require ``O(B * Hq * Tq * Kvisible * Dh)`` temporary memory even though
    the persistent attention probabilities only require
    ``O(B * Hq * Tq * Kvisible)``.  Chunking changes neither the selected keys
    nor the softmax mathematics.

    Args:
        q: ``(B,Tq,Hq,Dh)`` query tensor, normally after RoPE.
        k, v: ``(B,Tk,Hkv,Dh)`` native grouped-query K/V tensors.
        plan: :class:`KeySelectionPlan` containing exact token positions.
        kv_head_indices: Optional ``(Hq,)`` mapping from each query head to its
            source KV head.  This is important for heterogeneous subsets of
            query heads; when omitted, standard contiguous GQA grouping is used.
        scale: Optional score scale.  Defaults to ``1/sqrt(Dh)``.
        query_chunk_size: Maximum number of query positions processed at once.
            Defaults to 128 and can be overridden at runtime with
            ``MINI_LLM_INDEXED_ATTN_QUERY_CHUNK``.

    Returns:
        context: ``(B,Tq,Hq,Dh)``.
        cache: Explicit backward cache when ``return_cache`` is true.
    """
    if not isinstance(plan, KeySelectionPlan):
        raise TypeError("plan must be a KeySelectionPlan")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or k_d_head != d_head:
        raise ValueError("q/k batch or head width mismatch")

    plan.validate(batch, n_q_heads, query_length, key_length)
    if kv_head_indices is None:
        kv_head_indices = _default_kv_head_indices(n_q_heads, n_kv_heads)
    else:
        kv_head_indices = xp.asarray(kv_head_indices, dtype=xp.int64)
        if kv_head_indices.shape != (n_q_heads,):
            raise ValueError("kv_head_indices must have shape (n_q_heads,)")
        if bool(xp.any(kv_head_indices < 0)) or bool(
            xp.any(kv_head_indices >= n_kv_heads)
        ):
            raise ValueError("kv_head_indices contains an invalid KV head")

    key_indices = plan.key_indices
    valid_mask = plan.valid_mask
    if key_indices.shape[1] == 1 and n_q_heads != 1:
        key_indices = xp.broadcast_to(
            key_indices, (batch, n_q_heads, query_length, key_indices.shape[-1])
        )
        valid_mask = xp.broadcast_to(valid_mask, key_indices.shape)
        logit_bias = (
            None
            if plan.logit_bias is None
            else xp.broadcast_to(plan.logit_bias, key_indices.shape)
        )
    else:
        logit_bias = plan.logit_bias

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)
    query_chunk_size = _resolve_query_chunk_size(
        query_length, query_chunk_size
    )

    q_heads = q.transpose(0, 2, 1, 3)  # [B,Hq,Tq,Dh]
    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]

    # Match the dense attention numerical policy. FP16 pre-scales the score
    # product to avoid overflow; BF16 indexed products run in FP32 because
    # generic CuPy BF16 batched operations are not consistently supported.
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0
    bf16_attention = is_bfloat16_dtype(q.dtype)
    probs_dtype = (
        xp.float32 if is_low_precision_dtype(q.dtype) else q.dtype
    )
    context_dtype = xp.float32 if bf16_attention else q.dtype
    probs = (
        xp.empty(key_indices.shape, dtype=probs_dtype) if return_cache else None
    )
    context_heads = xp.empty(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )

    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        chunk_indices = key_indices[:, :, q_start:q_end, :]
        chunk_valid = valid_mask[:, :, q_start:q_end, :]
        q_chunk = q_heads[:, :, q_start:q_end, :]

        # Gather only this query chunk.  For the 4k medium preset this changes
        # the largest local-head BF16 gather from [1,4,4096,1024,64] to
        # [1,4,chunk,1024,64].
        k_selected = k[batch_ids, chunk_indices, kv_ids, :]
        if bf16_attention:
            q_score = q_chunk.astype("float32") * scale
            k_score = k_selected.astype("float32")
        else:
            q_score = q_chunk * (scale * score_prescale)
            k_score = k_selected

        # Batched [1,D] @ [D,K] avoids materializing the old
        # q[...,None,:] * k_selected [B,H,Q,K,D] product.
        scores = xp.matmul(
            q_score[..., None, :], k_score.swapaxes(-1, -2)
        )[..., 0, :]
        if logit_bias is not None:
            chunk_bias = logit_bias[:, :, q_start:q_end, :]
            scores = scores + (
                chunk_bias.astype(scores.dtype, copy=False) * score_prescale
            )

        probs_chunk = _masked_softmax_forward(
            scores, chunk_valid, logit_multiplier=(1.0 / score_prescale)
        )
        if return_cache:
            probs[:, :, q_start:q_end, :] = probs_chunk

        # K is no longer needed for the forward chunk. Release references before
        # gathering V so peak memory is bounded by one selected-value tensor.
        del scores, k_score, k_selected

        v_selected = v[batch_ids, chunk_indices, kv_ids, :]
        if bf16_attention:
            probs_compute = probs_chunk
            v_compute = v_selected.astype("float32")
        else:
            probs_compute = (
                probs_chunk.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_chunk
            )
            v_compute = v_selected

        context_heads[:, :, q_start:q_end, :] = xp.matmul(
            probs_compute[..., None, :], v_compute
        )[..., 0, :]

    context = context_heads.transpose(0, 2, 1, 3)

    if not return_cache:
        return context

    cache = {
        "q": q,
        "k": k,
        "v": v,
        "key_indices": key_indices,
        "valid_mask": valid_mask,
        "kv_head_indices": kv_head_indices,
        "probs": probs,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "has_logit_bias": logit_bias is not None,
        "query_chunk_size": query_chunk_size,
    }
    return context, cache

def indexed_attention_backward(dcontext, cache):
    """Explicit chunked backward for :func:`indexed_attention_forward`.

    Repeated token selections and shared GQA K/V heads are accumulated with
    scatter-add.  As in forward, selected K/V tensors are reconstructed only
    for a bounded query chunk so backward does not recreate a full
    ``[B,H,T,K,D]`` temporary.

    Returns:
        ``dq, dk, dv, dlogit_bias``.  ``dlogit_bias`` is ``None`` when no bias
        was supplied in forward.  It has the expanded per-query-head shape used
        by the kernel; callers that broadcast one bias across heads should sum
        that gradient over the broadcasted head dimension.
    """
    q, k, v = cache["q"], cache["k"], cache["v"]
    key_indices = cache["key_indices"]
    valid_mask = cache["valid_mask"]
    kv_head_indices = cache["kv_head_indices"]
    probs = cache["probs"]
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    query_chunk_size = cache.get(
        "query_chunk_size", _resolve_query_chunk_size(q.shape[1])
    )

    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    batch, query_length, n_q_heads, d_head = q.shape
    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_head_indices[None, :, None, None]
    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)
    dlogit_bias = (
        xp.zeros(probs.shape, dtype=probs.dtype)
        if cache["has_logit_bias"]
        else None
    )

    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        chunk_indices = key_indices[:, :, q_start:q_end, :]
        chunk_valid = valid_mask[:, :, q_start:q_end, :]
        probs_chunk = probs[:, :, q_start:q_end, :]
        q_chunk = q_heads[:, :, q_start:q_end, :]
        dcontext_chunk = dcontext_heads[:, :, q_start:q_end, :]

        if bf16_attention:
            dcontext_compute = dcontext_chunk.astype("float32")
            q_compute = q_chunk.astype("float32")
            probs_compute = probs_chunk
        else:
            dcontext_compute = dcontext_chunk
            q_compute = q_chunk
            probs_compute = (
                probs_chunk.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_chunk
            )

        # V branch: dP and dV. Gather and release V before K is gathered so the
        # two largest selected-value buffers do not coexist.
        v_selected = v[batch_ids, chunk_indices, kv_ids, :]
        v_compute = (
            v_selected.astype("float32") if bf16_attention else v_selected
        )
        dprobs = xp.matmul(
            dcontext_compute[..., None, :], v_compute.swapaxes(-1, -2)
        )[..., 0, :]
        dv_selected = (
            probs_compute[..., None] * dcontext_compute[..., None, :]
        )

        scatter_batch = xp.broadcast_to(batch_ids, chunk_indices.shape)
        scatter_kv = xp.broadcast_to(kv_ids, chunk_indices.shape)
        xp.add.at(
            dv,
            (scatter_batch, chunk_indices, scatter_kv),
            dv_selected.astype(grad_dtype, copy=False),
        )
        del v_compute, v_selected, dv_selected

        # probs is zero at every invalid key, so the softmax Jacobian gives an
        # exactly zero score gradient there (including completely empty rows).
        dscores = _masked_softmax_backward(dprobs, probs_chunk, chunk_valid)
        if dlogit_bias is not None:
            dlogit_bias[:, :, q_start:q_end, :] = dscores

        dscores_compute = (
            dscores.astype("float32", copy=False)
            if bf16_attention
            else (
                dscores.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else dscores
            )
        )

        # K/Q branch.  The Q gradient is another batched [1,K] @ [K,D]
        # product; only dK requires an explicit outer-product tensor, and that
        # tensor is bounded by the query chunk.
        k_selected = k[batch_ids, chunk_indices, kv_ids, :]
        k_compute = (
            k_selected.astype("float32") if bf16_attention else k_selected
        )
        dq_heads[:, :, q_start:q_end, :] = (
            xp.matmul(dscores_compute[..., None, :], k_compute)[..., 0, :]
            * scale
        )
        dk_selected = (
            dscores_compute[..., None] * q_compute[..., None, :] * scale
        )
        xp.add.at(
            dk,
            (scatter_batch, chunk_indices, scatter_kv),
            dk_selected.astype(grad_dtype, copy=False),
        )

    dq = dq_heads.transpose(0, 2, 1, 3)
    return dq, dk, dv, dlogit_bias





_LOCAL_GROUPED_STORE_MODULE = None
_LOCAL_GROUPED_STORE_DISABLED = False


def _get_local_grouped_store_module():
    """Compile the tiny BF16 -> FP32 gradient scatter/reduction kernel."""
    global _LOCAL_GROUPED_STORE_MODULE, _LOCAL_GROUPED_STORE_DISABLED
    if BACKEND_NAME != "cupy" or _LOCAL_GROUPED_STORE_DISABLED:
        return None
    if _LOCAL_GROUPED_STORE_MODULE is not None:
        return _LOCAL_GROUPED_STORE_MODULE
    code = r"""
    __device__ __forceinline__ float local_bf16_to_float(unsigned short x) {
        union { unsigned int u; float f; } v;
        v.u = ((unsigned int)x) << 16;
        return v.f;
    }

    extern "C" __global__
    void local_grouped_store_grads(
        const unsigned short* dq_tmp,
        const unsigned short* dk_tmp,
        const unsigned short* dv_tmp,
        float* dq, float* dk, float* dv,
        const long long* kv_map,
        int batch, int q_heads, int kv_heads,
        int q_count, int k_count, int d_head, int seq_len,
        int q_start, int key_start, float scale) {
        long long n_dq = (long long)batch * q_heads * q_count * d_head;
        long long n_kv = (long long)batch * k_count * kv_heads * d_head;
        long long n = n_dq > n_kv ? n_dq : n_kv;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += (long long)blockDim.x * gridDim.x) {
            if (idx < n_dq) {
                long long t = idx;
                int d = (int)(t % d_head); t /= d_head;
                int q_local = (int)(t % q_count); t /= q_count;
                int h = (int)(t % q_heads); t /= q_heads;
                int b = (int)t;
                long long dst = ((((long long)b * seq_len + (q_start + q_local))
                                  * q_heads + h) * d_head + d);
                dq[dst] = local_bf16_to_float(dq_tmp[idx]) * scale;
            }
            if (idx < n_kv) {
                long long t = idx;
                int d = (int)(t % d_head); t /= d_head;
                int kvh = (int)(t % kv_heads); t /= kv_heads;
                int k_local = (int)(t % k_count); t /= k_count;
                int b = (int)t;
                float sk = 0.0f;
                float sv = 0.0f;
                for (int h = 0; h < q_heads; ++h) {
                    if ((int)kv_map[h] != kvh) continue;
                    long long src = ((((long long)b * q_heads + h) * k_count
                                      + k_local) * d_head + d);
                    sk += local_bf16_to_float(dk_tmp[src]);
                    sv += local_bf16_to_float(dv_tmp[src]);
                }
                long long dst = ((((long long)b * seq_len + (key_start + k_local))
                                  * kv_heads + kvh) * d_head + d);
                dk[dst] += sk * scale;
                dv[dst] += sv;
            }
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=("local_grouped_store_grads",),
        )
        module.get_function("local_grouped_store_grads")
        _LOCAL_GROUPED_STORE_MODULE = module
    except Exception:
        if _cublas_grouped_strict_enabled():
            raise
        _LOCAL_GROUPED_STORE_DISABLED = True
        return None
    return _LOCAL_GROUPED_STORE_MODULE


def _local_grouped_store_grads(
    dq_tmp, dk_tmp, dv_tmp, dq, dk, dv, kv_map,
    q_start, key_start, scale,
):
    module = _get_local_grouped_store_module()
    if module is None:
        return False
    batch, q_heads, q_count, d_head = map(int, dq_tmp.shape)
    b2, h2, k_count, d2 = map(int, dk_tmp.shape)
    if (b2, h2, d2) != (batch, q_heads, d_head) or dv_tmp.shape != dk_tmp.shape:
        return False
    if not all(is_bfloat16_dtype(a.dtype) for a in (dq_tmp, dk_tmp, dv_tmp)):
        return False
    if any(a.dtype != xp.float32 for a in (dq, dk, dv)):
        return False
    if not all(a.flags.c_contiguous for a in (dq_tmp, dk_tmp, dv_tmp, dq, dk, dv, kv_map)):
        return False
    n_kv_heads = int(dk.shape[2])
    n = max(int(dq_tmp.size), batch * k_count * n_kv_heads * d_head)
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    module.get_function("local_grouped_store_grads")(
        (blocks,), (threads,),
        (
            dq_tmp, dk_tmp, dv_tmp, dq, dk, dv, kv_map,
            np.int32(batch), np.int32(q_heads), np.int32(n_kv_heads),
            np.int32(q_count), np.int32(k_count), np.int32(d_head),
            np.int32(dq.shape[1]), np.int32(q_start), np.int32(key_start),
            np.float32(scale),
        ),
    )
    return True


def _row_stride_elems(array, row_axis):
    return int(array.strides[row_axis] // array.dtype.itemsize)


def _local_grouped_matrix_layout_ready(x):
    """Return whether one [B,T,H,D] tensor is safe for direct cuBLAS rows.

    The grouped path fixes ``b`` and ``h`` for each GEMM, so it only requires
    the innermost D elements of each token/head row to be contiguous.  The
    batch/token/head strides may be larger than a compact tensor.  This is
    important with packed QKV: after RoPE Q/K are compact, while V remains a
    strided column view into the packed projection output.
    """
    if x.ndim != 4:
        return False
    itemsize = int(x.dtype.itemsize)
    strides = tuple(int(s) for s in x.strides)
    return (
        itemsize > 0
        and strides[-1] == itemsize
        and all(s > 0 and (s % itemsize) == 0 for s in strides)
    )


def _local_grouped_fastpath_ready(q, k, v, bf16_pipeline):
    return (
        bf16_pipeline
        and _cublas_grouped_local_enabled()
        and _local_grouped_matrix_layout_ready(q)
        and _local_grouped_matrix_layout_ready(k)
        and _local_grouped_matrix_layout_ready(v)
        and _cublas_grouped_available()
    )


def _local_grouped_forward(
    q, k, v, window, scale, kv_map_host, query_chunk_size,
    context_dtype, return_cache,
):
    """0053/0053E cuBLAS local forward with zero Q/K/V packing."""
    cublas_mode = _cublas_local_mode()
    profile_prefix = f"local.{cublas_mode}"
    batch, query_length, n_q_heads, d_head = map(int, q.shape)
    q_row = _row_stride_elems(q, 1)
    k_row = _row_stride_elems(k, 1)
    v_row = _row_stride_elems(v, 1)

    chunks = []
    score_groups = []
    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        key_start = max(0, q_start - window + 1)
        key_end = q_end
        q_count = q_end - q_start
        k_count = key_end - key_start
        scores = xp.empty((batch, n_q_heads, q_count, k_count), dtype=q.dtype)
        probs = xp.empty_like(scores)
        problems = []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = int(q.data.ptr) + b * int(q.strides[0]) + q_start * int(q.strides[1]) + h * int(q.strides[2])
                b_ptr = int(k.data.ptr) + b * int(k.strides[0]) + key_start * int(k.strides[1]) + kvh * int(k.strides[2])
                c_ptr = int(scores.data.ptr) + b * int(scores.strides[0]) + h * int(scores.strides[1])
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, k_count, d_head,
                    q_row, k_row, k_count,
                    False, True,
                ))
        score_groups.append(RowMajorGemmGroup(tuple(problems)))
        chunks.append({
            "q_start": q_start, "q_end": q_end,
            "key_start": key_start, "key_end": key_end,
            "q_count": q_count, "k_count": k_count,
            "scores": scores, "probs": probs,
        })

    with local_detail_scope(f"{profile_prefix}.score_gemm"):
        if not _cublas_grouped_bf16_gemm(tuple(score_groups)):
            return None

    fused_softmax = False
    if _multi_chunk_local_softmax_enabled() and len(chunks) == 3:
        with local_detail_scope(f"{profile_prefix}.softmax"):
            fused_softmax = _local_causal_softmax_forward3_bf16_cuda(
                chunks, window, scale
            )
    if not fused_softmax:
        for chunk in chunks:
            with local_detail_scope(f"{profile_prefix}.softmax"):
                out = _local_causal_softmax_forward_bf16_cuda(
                    chunk["scores"], chunk["probs"],
                    chunk["q_start"], chunk["key_start"], window, scale,
                )
                if out is None:
                    if _cublas_grouped_strict_enabled():
                        raise RuntimeError("grouped local BF16 softmax fast path declined layout")
                    return None
    for chunk in chunks:
        del chunk["scores"]

    context_tmps = []
    context_groups = []
    for chunk in chunks:
        q_start = chunk["q_start"]
        key_start = chunk["key_start"]
        q_count = chunk["q_count"]
        k_count = chunk["k_count"]
        probs = chunk["probs"]
        tmp = xp.empty((batch, n_q_heads, q_count, d_head), dtype=q.dtype)
        context_tmps.append(tmp)
        problems = []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = int(probs.data.ptr) + b * int(probs.strides[0]) + h * int(probs.strides[1])
                b_ptr = int(v.data.ptr) + b * int(v.strides[0]) + key_start * int(v.strides[1]) + kvh * int(v.strides[2])
                c_ptr = int(tmp.data.ptr) + b * int(tmp.strides[0]) + h * int(tmp.strides[1])
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, d_head, k_count,
                    k_count, v_row, d_head,
                    False, False,
                ))
        context_groups.append(RowMajorGemmGroup(tuple(problems)))

    with local_detail_scope(f"{profile_prefix}.context_gemm"):
        if not _cublas_grouped_bf16_gemm(tuple(context_groups)):
            return None

    context_heads = xp.empty((batch, n_q_heads, query_length, d_head), dtype=context_dtype)
    for chunk, tmp in zip(chunks, context_tmps):
        with local_detail_scope(f"{profile_prefix}.context_store"):
            context_heads[:, :, chunk["q_start"]:chunk["q_end"], :] = tmp

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context
    return context, {
        "q": q, "k": k, "v": v,
        "window": window, "scale": scale,
        "bf16_attention": True,
        "bf16_tensorcore": True,
        "bf16_pipeline": True,
        "kv_head_indices_host": tuple(kv_map_host),
        "kv_head_indices": xp.asarray(kv_map_host, dtype=xp.int64),
        "chunks": tuple(chunks),
        "query_chunk_size": query_chunk_size,
        "grouped_cublas_local": True,
        "cublas_local_mode": cublas_mode,
        "fused_local_softmax3": fused_softmax,
    }


def _local_grouped_backward(dcontext, cache):
    """0053/0053E cuBLAS local backward."""
    cublas_mode = cache.get("cublas_local_mode", _cublas_local_mode())
    profile_prefix = f"local.{cublas_mode}"
    q, k, v = cache["q"], cache["k"], cache["v"]
    batch, query_length, n_q_heads, d_head = map(int, q.shape)
    n_kv_heads = int(k.shape[2])
    kv_map_host = cache["kv_head_indices_host"]
    kv_map = cache["kv_head_indices"]
    scale = float(cache["scale"])
    q_row = _row_stride_elems(q, 1)
    k_row = _row_stride_elems(k, 1)
    v_row = _row_stride_elems(v, 1)

    with local_detail_scope(f"{profile_prefix}.dcontext_cast"):
        dc = xp.ascontiguousarray(dcontext.astype(q.dtype, copy=False))
    dc_row = _row_stride_elems(dc, 1)
    dq = xp.zeros(q.shape, dtype=xp.float32)
    dk = xp.zeros(k.shape, dtype=xp.float32)
    dv = xp.zeros(v.shape, dtype=xp.float32)

    dprobs_groups = []
    for chunk in cache["chunks"]:
        q_start = chunk["q_start"]
        key_start = chunk["key_start"]
        q_count = chunk["q_count"]
        k_count = chunk["k_count"]
        dprobs = xp.empty_like(chunk["probs"])
        chunk["dprobs"] = dprobs
        problems = []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = int(dc.data.ptr) + b * int(dc.strides[0]) + q_start * int(dc.strides[1]) + h * int(dc.strides[2])
                b_ptr = int(v.data.ptr) + b * int(v.strides[0]) + key_start * int(v.strides[1]) + kvh * int(v.strides[2])
                c_ptr = int(dprobs.data.ptr) + b * int(dprobs.strides[0]) + h * int(dprobs.strides[1])
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, k_count, d_head,
                    dc_row, v_row, k_count,
                    False, True,
                ))
        dprobs_groups.append(RowMajorGemmGroup(tuple(problems)))

    with local_detail_scope(f"{profile_prefix}.dprobs_gemm"):
        if not _cublas_grouped_bf16_gemm(tuple(dprobs_groups)):
            return None

    fused_softmax_backward = False
    if _multi_chunk_local_softmax_enabled() and len(cache["chunks"]) == 3:
        with local_detail_scope(f"{profile_prefix}.softmax_backward"):
            fused_softmax_backward = _local_causal_softmax_backward3_bf16_cuda(
                cache["chunks"], cache["window"]
            )
    if not fused_softmax_backward:
        for chunk in cache["chunks"]:
            with local_detail_scope(f"{profile_prefix}.softmax_backward"):
                ds = _local_causal_softmax_backward_bf16_cuda(
                    chunk["dprobs"], chunk["probs"],
                    chunk["q_start"], chunk["key_start"], cache["window"],
                )
                if ds is None:
                    if _cublas_grouped_strict_enabled():
                        raise RuntimeError("grouped local BF16 softmax backward declined layout")
                    return None
                chunk["dscores"] = ds
    cache["fused_local_softmax3_backward"] = fused_softmax_backward
    for chunk in cache["chunks"]:
        del chunk["dprobs"]

    grad_groups = []
    work = []
    for chunk in cache["chunks"]:
        q_start = chunk["q_start"]
        key_start = chunk["key_start"]
        q_count = chunk["q_count"]
        k_count = chunk["k_count"]
        ds = chunk["dscores"]
        probs = chunk["probs"]
        dq_tmp = xp.empty((batch, n_q_heads, q_count, d_head), dtype=q.dtype)
        dk_tmp = xp.empty((batch, n_q_heads, k_count, d_head), dtype=q.dtype)
        dv_tmp = xp.empty_like(dk_tmp)
        work.append((chunk, dq_tmp, dk_tmp, dv_tmp))

        dq_problems = []
        dk_problems = []
        dv_problems = []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                ds_ptr = int(ds.data.ptr) + b * int(ds.strides[0]) + h * int(ds.strides[1])
                p_ptr = int(probs.data.ptr) + b * int(probs.strides[0]) + h * int(probs.strides[1])
                q_ptr = int(q.data.ptr) + b * int(q.strides[0]) + q_start * int(q.strides[1]) + h * int(q.strides[2])
                k_ptr = int(k.data.ptr) + b * int(k.strides[0]) + key_start * int(k.strides[1]) + kvh * int(k.strides[2])
                dc_ptr = int(dc.data.ptr) + b * int(dc.strides[0]) + q_start * int(dc.strides[1]) + h * int(dc.strides[2])
                dq_ptr = int(dq_tmp.data.ptr) + b * int(dq_tmp.strides[0]) + h * int(dq_tmp.strides[1])
                dk_ptr = int(dk_tmp.data.ptr) + b * int(dk_tmp.strides[0]) + h * int(dk_tmp.strides[1])
                dv_ptr = int(dv_tmp.data.ptr) + b * int(dv_tmp.strides[0]) + h * int(dv_tmp.strides[1])
                dq_problems.append(RowMajorGemmProblem(
                    ds_ptr, k_ptr, dq_ptr,
                    q_count, d_head, k_count,
                    k_count, k_row, d_head,
                    False, False,
                ))
                dk_problems.append(RowMajorGemmProblem(
                    ds_ptr, q_ptr, dk_ptr,
                    k_count, d_head, q_count,
                    k_count, q_row, d_head,
                    True, False,
                ))
                dv_problems.append(RowMajorGemmProblem(
                    p_ptr, dc_ptr, dv_ptr,
                    k_count, d_head, q_count,
                    k_count, dc_row, d_head,
                    True, False,
                ))
        grad_groups.extend((
            RowMajorGemmGroup(tuple(dq_problems)),
            RowMajorGemmGroup(tuple(dk_problems)),
            RowMajorGemmGroup(tuple(dv_problems)),
        ))

    with local_detail_scope(f"{profile_prefix}.grad_gemm"):
        if not _cublas_grouped_bf16_gemm(tuple(grad_groups)):
            return None

    for chunk, dq_tmp, dk_tmp, dv_tmp in work:
        with local_detail_scope(f"{profile_prefix}.grad_store"):
            if not _local_grouped_store_grads(
                dq_tmp, dk_tmp, dv_tmp, dq, dk, dv, kv_map,
                chunk["q_start"], chunk["key_start"], scale,
            ):
                return None
        del chunk["dscores"]

    return dq, dk, dv


# ---------------------------------------------------------------------------
# Specialized contiguous local-window attention
# ---------------------------------------------------------------------------

def _normalize_kv_head_mapping(kv_head_indices, n_q_heads, n_kv_heads):
    """Return both device and tiny host forms of a Q->KV head mapping.

    The host tuple is used only to reduce query-head K/V gradients back into
    native GQA heads.  Mixed attention passes a Python tuple, so the GPU hot
    path never synchronizes merely to inspect this tiny mapping.
    """
    if kv_head_indices is None:
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                "n_q_heads must be divisible by n_kv_heads when "
                "kv_head_indices is not supplied"
            )
        group_size = n_q_heads // n_kv_heads
        host = tuple(i // group_size for i in range(n_q_heads))
    elif isinstance(kv_head_indices, (tuple, list)):
        host = tuple(int(i) for i in kv_head_indices)
    else:
        # Direct public callers may still provide an ndarray.  This fallback is
        # intentionally outside the mixed-attention hot path.
        from ..backend import asnumpy
        host = tuple(int(i) for i in asnumpy(kv_head_indices).reshape(-1))

    if len(host) != n_q_heads:
        raise ValueError("kv_head_indices must have shape (n_q_heads,)")
    if any(i < 0 or i >= n_kv_heads for i in host):
        raise ValueError("kv_head_indices contains an invalid KV head")
    return xp.asarray(host, dtype=xp.int64), host


def local_window_attention_forward(
    q,
    k,
    v,
    window,
    kv_head_indices=None,
    scale=None,
    return_cache=True,
    query_chunk_size=None,
):
    """Specialized causal sliding-window attention with native GQA sharing.

    Query heads are grouped by their native KV head.  Each contiguous K/V
    span is therefore loaded and converted only once per KV group and CUDA
    matmul broadcasting shares it across all corresponding query heads.  This
    preserves the memory/computation advantage of GQA instead of physically
    repeating the same K/V tensor for every query head.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or k_d_head != d_head or key_length != query_length:
        raise ValueError("local self-attention requires matching q/k sequence shapes")

    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    window = min(window, key_length)
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)
    query_chunk_size = _resolve_local_query_chunk_size(
        query_length, query_chunk_size
    )

    _kv_map, kv_map_host = _normalize_kv_head_mapping(
        kv_head_indices, n_q_heads, n_kv_heads
    )
    q_heads = q.transpose(0, 2, 1, 3)  # [B,Hq,T,D]
    bf16_attention = is_bfloat16_dtype(q.dtype)
    bf16_tensorcore = _bf16_local_tensorcore_enabled(q.dtype)
    bf16_pipeline = (
        bf16_tensorcore
        and _fused_bf16_local_pipeline_enabled(q.dtype)
        and _get_local_bf16_pipeline_module() is not None
    )
    keep_bf16_context = (
        bf16_pipeline and _bf16_mixed_context_enabled(q.dtype)
    )
    context_dtype = (
        q.dtype
        if keep_bf16_context
        else (xp.float32 if bf16_attention else q.dtype)
    )

    # 0053: keep cuBLAS GEMM quality but dispatch all heterogeneous local
    # chunks/heads through cublasGemmGroupedBatchedEx.  The helper reads
    # Q/K/V directly from their native strided storage, so unlike the earlier
    # experimental WMMA path it does not trade GEMM quality for fewer calls.
    if _local_grouped_fastpath_ready(q, k, v, bf16_pipeline):
        with local_detail_scope("local.grouped.forward"):
            grouped = _local_grouped_forward(
                q, k, v, window, scale, kv_map_host, query_chunk_size,
                context_dtype, return_cache,
            )
        if grouped is not None:
            return grouped
        if _cublas_grouped_strict_enabled():
            raise RuntimeError("0053 grouped local forward unexpectedly declined")

    context_heads = xp.empty(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0

    # Local heads can be an arbitrary subset of the model's query heads, so
    # form the small static local-head -> native-KV groups once.
    q_heads_for_kv = [
        tuple(i for i, source in enumerate(kv_map_host) if source == kvh)
        for kvh in range(n_kv_heads)
    ]
    active_groups = tuple(
        (kvh, head_group)
        for kvh, head_group in enumerate(q_heads_for_kv)
        if head_group
    )
    chunk_caches = [] if return_cache else None

    for q_start, q_end in _query_chunks(query_length, query_chunk_size):
        key_start = max(0, q_start - window + 1)
        key_end = q_end

        # Always retain the exact reference mask in the backward cache.
        # The fused CUDA forward can reconstruct the same causal/window mask
        # internally, but backward must never fall back to an unmasked softmax
        # derivative if the CUDA fast path declines a particular array layout.
        q_positions = xp.arange(q_start, q_end, dtype=xp.int64)[:, None]
        k_positions = xp.arange(key_start, key_end, dtype=xp.int64)[None, :]
        valid = (
            (k_positions <= q_positions)
            & (k_positions >= (q_positions - window + 1))
        )[None, None, :, :]
        group_probs = [] if return_cache else None

        for kvh, head_group in active_groups:
            q_chunk = q_heads[:, head_group, q_start:q_end, :]
            # Keep the native K/V head exactly once.  The singleton head axis
            # broadcasts across all query heads in this GQA group.
            k_native = k[:, key_start:key_end, kvh, :][:, None, :, :]
            if bf16_tensorcore:
                # CuPy supports the target BF16 2-D GEMM even where generic
                # N-D BF16 matmul is unavailable. Fold query heads into the
                # row dimension; K is shared by the whole native GQA group.
                n_group_heads = len(head_group)
                q_count = q_end - q_start
                k_count = key_end - key_start
                if bf16_pipeline:
                    # Keep score/probability matrices in BF16 storage. The
                    # fused kernel promotes individual values internally for
                    # FP32 max/sum/exp and writes the final probabilities
                    # directly in the format consumed by the context GEMM and
                    # backward cache.
                    probs_chunk = xp.empty(
                        (batch, n_group_heads, q_count, k_count), dtype=q.dtype
                    )
                    scores = None
                else:
                    scores = xp.empty(
                        (batch, n_group_heads, q_count, k_count), dtype=xp.float32
                    )
                for batch_idx in range(batch):
                    with local_detail_scope("local.fwd.score_pack"):
                        q_2d = xp.ascontiguousarray(
                            q_chunk[batch_idx].reshape(
                                n_group_heads * q_count, d_head
                            )
                        )
                        k_2d = xp.ascontiguousarray(k_native[batch_idx, 0])
                    with local_detail_scope("local.fwd.score_gemm"):
                        score_bf16 = q_2d @ k_2d.T
                    if bf16_pipeline:
                        with local_detail_scope("local.fwd.bf16_softmax_pipeline"):
                            out = _local_causal_softmax_forward_bf16_cuda(
                                score_bf16.reshape(
                                    n_group_heads, q_count, k_count
                                ),
                                probs_chunk[batch_idx],
                                q_start,
                                key_start,
                                window,
                                scale,
                            )
                            if out is None:
                                raise RuntimeError(
                                    "BF16 local softmax pipeline declined an "
                                    "internally generated contiguous layout"
                                )
                    else:
                        with local_detail_scope("local.fwd.score_promote"):
                            scores[batch_idx] = (
                                score_bf16.reshape(
                                    n_group_heads, q_count, k_count
                                ).astype("float32")
                                * scale
                            )
                k_score = None
            else:
                if bf16_attention:
                    q_score = q_chunk.astype("float32") * scale
                    k_score = k_native.astype("float32")
                else:
                    q_score = q_chunk * (scale * score_prescale)
                    k_score = k_native
                scores = xp.matmul(q_score, k_score.swapaxes(-1, -2))

            if not bf16_pipeline:
                with local_detail_scope("local.fwd.softmax"):
                    probs_chunk = _local_causal_softmax_forward_cuda(
                        scores, q_start, key_start, window, (1.0 / score_prescale)
                    )
                    if probs_chunk is None:
                        probs_chunk = _masked_softmax_forward(
                            scores, valid, logit_multiplier=(1.0 / score_prescale)
                        )
                del scores
            if k_score is not None:
                del k_score

            v_native = v[:, key_start:key_end, kvh, :][:, None, :, :]
            if bf16_tensorcore:
                n_group_heads = len(head_group)
                q_count = q_end - q_start
                k_count = key_end - key_start
                for batch_idx in range(batch):
                    with local_detail_scope("local.fwd.context_pack"):
                        if bf16_pipeline:
                            probs_2d = xp.ascontiguousarray(
                                probs_chunk[batch_idx].reshape(
                                    n_group_heads * q_count, k_count
                                )
                            )
                        else:
                            probs_2d = xp.ascontiguousarray(
                                probs_chunk[batch_idx]
                                .reshape(n_group_heads * q_count, k_count)
                                .astype(q.dtype)
                            )
                        v_2d = xp.ascontiguousarray(v_native[batch_idx, 0])
                    with local_detail_scope("local.fwd.context_gemm"):
                        context_bf16 = probs_2d @ v_2d
                    with local_detail_scope("local.fwd.context_store"):
                        context_heads[
                            batch_idx, head_group, q_start:q_end, :
                        ] = context_bf16.reshape(
                            n_group_heads, q_count, d_head
                        )
            else:
                if bf16_attention:
                    v_compute = v_native.astype("float32")
                    probs_compute = probs_chunk
                else:
                    v_compute = v_native
                    probs_compute = (
                        probs_chunk.astype(q.dtype, copy=False)
                        if is_low_precision_dtype(q.dtype)
                        else probs_chunk
                    )
                context_heads[:, head_group, q_start:q_end, :] = xp.matmul(
                    probs_compute, v_compute
                )
            if return_cache:
                group_probs.append(
                    (kvh, head_group, _cache_probs_for_backward(probs_chunk, q.dtype))
                )

        if return_cache:
            chunk_caches.append(
                (q_start, q_end, key_start, key_end, valid, tuple(group_probs))
            )

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context

    return context, {
        "q": q,
        "k": k,
        "v": v,
        "window": window,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "bf16_tensorcore": bf16_tensorcore,
        "bf16_pipeline": bf16_pipeline,
        "kv_head_indices_host": kv_map_host,
        "active_groups": active_groups,
        "chunks": chunk_caches,
        "query_chunk_size": query_chunk_size,
    }


def local_window_attention_backward(dcontext, cache):
    """Backward for GQA-aware contiguous local attention.

    Native K/V tensors are never repeated per query head.  Each GQA query
    group produces one route-local K/V gradient via batched GEMMs, then the
    group dimension is reduced before updating the native K/V slice.
    """
    q, k, v = cache["q"], cache["k"], cache["v"]
    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    if cache.get("grouped_cublas_local", False):
        with local_detail_scope("local.grouped.backward"):
            grouped = _local_grouped_backward(dcontext, cache)
        if grouped is None:
            raise RuntimeError(
                "0053 grouped local backward failed after grouped forward succeeded"
            )
        return grouped

    batch, query_length, n_q_heads, d_head = q.shape
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    bf16_tensorcore = cache.get("bf16_tensorcore", False)
    bf16_pipeline = cache.get("bf16_pipeline", False)
    active_groups = cache["active_groups"]

    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)
    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)

    for q_start, q_end, key_start, key_end, valid, group_probs in cache["chunks"]:
        # ``group_probs`` mirrors ``active_groups`` and stores only the
        # probabilities required for backward.
        for (kvh, head_group), (cached_kvh, cached_heads, probs_chunk) in zip(
            active_groups, group_probs
        ):
            if kvh != cached_kvh or head_group != cached_heads:
                raise RuntimeError("local attention GQA cache is inconsistent")

            q_chunk = q_heads[:, head_group, q_start:q_end, :]
            dcontext_chunk = dcontext_heads[:, head_group, q_start:q_end, :]
            k_native = k[:, key_start:key_end, kvh, :][:, None, :, :]
            v_native = v[:, key_start:key_end, kvh, :][:, None, :, :]

            if bf16_tensorcore:
                if bf16_pipeline:
                    # Forward already cached normalized probabilities in BF16.
                    # Keep dP/dS in BF16 storage too; the fused backward kernel
                    # promotes individual values internally for the FP32
                    # softmax Jacobian and rounds only the final dS.
                    probs_compute = probs_chunk
                    dprobs = xp.empty(
                        (batch, len(head_group), q_end - q_start, key_end - key_start),
                        dtype=q.dtype,
                    )
                else:
                    with local_detail_scope("local.bwd.restore_probs"):
                        probs_compute = _restore_cached_probs(probs_chunk, True)
                    dprobs = xp.empty(
                        (batch, len(head_group), q_end - q_start, key_end - key_start),
                        dtype=xp.float32,
                    )
                n_group_heads = len(head_group)
                q_count = q_end - q_start
                k_count = key_end - key_start
                # dP = dO V^T. Fold heads into rows while keeping the shared
                # native V matrix exactly once per batch.
                dcontext_bf16 = []
                for batch_idx in range(batch):
                    with local_detail_scope("local.bwd.dprobs_pack"):
                        dc_2d = xp.ascontiguousarray(
                            dcontext_chunk[batch_idx]
                            .reshape(n_group_heads * q_count, d_head)
                            .astype(q.dtype)
                        )
                        dcontext_bf16.append(dc_2d)
                        v_2d = xp.ascontiguousarray(v_native[batch_idx, 0])
                    with local_detail_scope("local.bwd.dprobs_gemm"):
                        dp_bf16 = dc_2d @ v_2d.T
                    with local_detail_scope("local.bwd.dprobs_promote"):
                        if bf16_pipeline:
                            dprobs[batch_idx] = dp_bf16.reshape(
                                n_group_heads, q_count, k_count
                            )
                        else:
                            dprobs[batch_idx] = dp_bf16.reshape(
                                n_group_heads, q_count, k_count
                            ).astype("float32")

                with local_detail_scope("local.bwd.softmax"):
                    if bf16_pipeline:
                        dscores = _local_causal_softmax_backward_bf16_cuda(
                            dprobs,
                            probs_compute,
                            q_start,
                            key_start,
                            cache["window"],
                        )
                        if dscores is None:
                            raise RuntimeError(
                                "BF16 local backward pipeline declined an "
                                "internally generated contiguous layout"
                            )
                    else:
                        dscores = _local_causal_softmax_backward_cuda(
                            dprobs, probs_compute, q_start, key_start, cache["window"]
                        )
                        if dscores is None:
                            dscores = _softmax_backward(dprobs, probs_compute)
                            dscores = xp.where(valid, dscores, 0.0)

                dk_native = xp.empty(
                    (batch, k_count, d_head), dtype=grad_dtype
                )
                dv_native = xp.empty(
                    (batch, k_count, d_head), dtype=grad_dtype
                )
                for batch_idx in range(batch):
                    with local_detail_scope("local.bwd.qkpd_pack"):
                        q_2d = xp.ascontiguousarray(
                            q_chunk[batch_idx].reshape(
                                n_group_heads * q_count, d_head
                            )
                        )
                        k_2d = xp.ascontiguousarray(k_native[batch_idx, 0])
                        if bf16_pipeline:
                            p_2d = xp.ascontiguousarray(
                                probs_compute[batch_idx].reshape(
                                    n_group_heads * q_count, k_count
                                )
                            )
                            ds_2d = xp.ascontiguousarray(
                                dscores[batch_idx].reshape(
                                    n_group_heads * q_count, k_count
                                )
                            )
                        else:
                            p_2d = xp.ascontiguousarray(
                                probs_compute[batch_idx]
                                .reshape(n_group_heads * q_count, k_count)
                                .astype(q.dtype)
                            )
                            ds_2d = xp.ascontiguousarray(
                                dscores[batch_idx]
                                .reshape(n_group_heads * q_count, k_count)
                                .astype(q.dtype)
                            )
                        dc_2d = dcontext_bf16[batch_idx]

                    with local_detail_scope("local.bwd.dq_gemm"):
                        dq_bf16 = ds_2d @ k_2d
                    with local_detail_scope("local.bwd.dk_gemm"):
                        dk_bf16 = ds_2d.T @ q_2d
                    with local_detail_scope("local.bwd.dv_gemm"):
                        dv_bf16 = p_2d.T @ dc_2d
                    with local_detail_scope("local.bwd.grad_store"):
                        dq_heads[batch_idx, head_group, q_start:q_end, :] = (
                            dq_bf16.reshape(n_group_heads, q_count, d_head)
                            .astype(grad_dtype)
                            * scale
                        )
                        dk_native[batch_idx] = (
                            dk_bf16.astype(grad_dtype) * scale
                        )
                        dv_native[batch_idx] = dv_bf16.astype(grad_dtype)
            else:
                if bf16_attention:
                    dcontext_compute = dcontext_chunk.astype("float32")
                    q_compute = q_chunk.astype("float32")
                    k_compute = k_native.astype("float32")
                    v_compute = v_native.astype("float32")
                    probs_compute = _restore_cached_probs(probs_chunk, True)
                else:
                    dcontext_compute = dcontext_chunk
                    q_compute = q_chunk
                    k_compute = k_native
                    v_compute = v_native
                    probs_compute = (
                        probs_chunk.astype(q.dtype, copy=False)
                        if is_low_precision_dtype(q.dtype)
                        else probs_chunk
                    )

                dprobs = xp.matmul(
                    dcontext_compute, v_compute.swapaxes(-1, -2)
                )
                dv_group = xp.matmul(
                    probs_compute.swapaxes(-1, -2), dcontext_compute
                )
                dscores = _local_causal_softmax_backward_cuda(
                    dprobs, probs_compute, q_start, key_start, cache["window"]
                )
                if dscores is None:
                    dscores = _softmax_backward(dprobs, probs_compute)
                    dscores = xp.where(valid, dscores, 0.0)
                dscores_compute = (
                    dscores.astype("float32", copy=False)
                    if bf16_attention
                    else (
                        dscores.astype(q.dtype, copy=False)
                        if is_low_precision_dtype(q.dtype)
                        else dscores
                    )
                )

                dq_heads[:, head_group, q_start:q_end, :] = (
                    xp.matmul(dscores_compute, k_compute) * scale
                )
                dk_group = (
                    xp.matmul(dscores_compute.swapaxes(-1, -2), q_compute) * scale
                )

                # The K/V head is shared by all query heads in the group.
                dk_native = xp.sum(dk_group, axis=1)
                dv_native = xp.sum(dv_group, axis=1)
            with local_detail_scope("local.bwd.kv_accumulate"):
                dk[:, key_start:key_end, kvh, :] += dk_native.astype(
                    grad_dtype, copy=False
                )
                dv[:, key_start:key_end, kvh, :] += dv_native.astype(
                    grad_dtype, copy=False
                )

    return dq_heads.transpose(0, 2, 1, 3), dk, dv



def _grouped_dilated_enabled():
    """Enable 0054 direct-pointer grouped-cuBLAS dilated attention."""
    raw = os.environ.get(
        "MINI_LLM_CUBLAS_GROUPED_DILATED_GEMM", "0"
    ).strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no"}


def _dilated_grouped_fastpath_ready(q, k, v, bf16_pipeline):
    # The cuBLAS bridge is shared with the validated local-attention path, so
    # MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM remains the master backend switch.
    return (
        _grouped_dilated_enabled()
        and bf16_pipeline
        and _cublas_grouped_local_enabled()
        and _cublas_local_mode() == "grouped"
        and _local_grouped_matrix_layout_ready(q)
        and _local_grouped_matrix_layout_ready(k)
        and _local_grouped_matrix_layout_ready(v)
        and _cublas_grouped_available()
    )


def _dilated_direct_chunks(
    query_length, key_length, window, dilation, offset, query_chunk_size,
):
    """Build the exact specialized-dilated phase/chunk geometry on the host.

    0054 keeps this tiny metadata calculation on Python scalars while removing
    Q/K/V gathers from the GPU hot path.  Each returned chunk describes a
    regular strided matrix view into the original [B,T,H,D] tensors.
    """
    key_slots = ((window - 1 - offset) // dilation) + 1
    chunks = []
    for query_residue in range(dilation):
        phase_length = (query_length - 1 - query_residue) // dilation + 1
        if phase_length <= 0:
            continue
        key_residue = (query_residue - offset) % dilation
        key_phase_length = (key_length - 1 - key_residue) // dilation + 1
        if key_phase_length <= 0:
            continue
        alignment_shift = 0 if query_residue >= offset else -1
        phase_chunk = _resolve_dilated_query_chunk_size(
            phase_length, query_chunk_size
        )
        for phase_start, phase_end in _query_chunks(phase_length, phase_chunk):
            max_key_end = phase_end + alignment_shift
            key_end = min(key_phase_length, max(0, max_key_end))
            key_start = max(
                0, phase_start + alignment_shift - key_slots + 1
            )
            chunks.append({
                "query_residue": int(query_residue),
                "phase_start": int(phase_start),
                "phase_end": int(phase_end),
                "key_residue": int(key_residue),
                "key_start": int(key_start),
                "key_end": int(key_end),
                "alignment_shift": int(alignment_shift),
                "q_count": int(phase_end - phase_start),
                "k_count": int(max(0, key_end - key_start)),
            })
    return chunks


def _dilated_grouped_forward(
    q, k, v, window, dilation, offset, scale, kv_map_host,
    query_chunk_size, context_dtype, return_cache,
):
    """0054 grouped-cuBLAS dilated forward with zero Q/K/V gathers."""
    batch, query_length, n_q_heads, d_head = map(int, q.shape)
    key_length = int(k.shape[1])
    q_row = dilation * _row_stride_elems(q, 1)
    k_row = dilation * _row_stride_elems(k, 1)
    v_row = dilation * _row_stride_elems(v, 1)
    key_slots = ((window - 1 - offset) // dilation) + 1

    chunks = _dilated_direct_chunks(
        query_length, key_length, window, dilation, offset, query_chunk_size
    )
    score_groups = []
    live_chunks = []
    for chunk in chunks:
        q_count = chunk["q_count"]
        k_count = chunk["k_count"]
        if k_count <= 0:
            chunk["probs"] = None
            continue
        scores = xp.empty(
            (batch, n_q_heads, q_count, k_count), dtype=q.dtype
        )
        problems = []
        q_token0 = chunk["query_residue"] + chunk["phase_start"] * dilation
        k_token0 = chunk["key_residue"] + chunk["key_start"] * dilation
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = (
                    int(q.data.ptr) + b * int(q.strides[0])
                    + q_token0 * int(q.strides[1]) + h * int(q.strides[2])
                )
                b_ptr = (
                    int(k.data.ptr) + b * int(k.strides[0])
                    + k_token0 * int(k.strides[1]) + kvh * int(k.strides[2])
                )
                c_ptr = (
                    int(scores.data.ptr) + b * int(scores.strides[0])
                    + h * int(scores.strides[1])
                )
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, k_count, d_head,
                    q_row, k_row, k_count,
                    False, True,
                ))
        score_groups.append(RowMajorGemmGroup(tuple(problems)))
        chunk["scores"] = scores
        live_chunks.append(chunk)

    with local_detail_scope("dilated.grouped.score_gemm"):
        if score_groups and not _cublas_grouped_bf16_gemm(tuple(score_groups)):
            return None

    fused_softmax4 = False
    if len(live_chunks) == 4 and _multi_chunk_dilated_softmax_enabled():
        with local_detail_scope("dilated.grouped.softmax"):
            fused_softmax4 = _dilated_softmax_forward4_inplace_bf16_cuda(
                live_chunks, key_slots, scale
            )

    if not fused_softmax4:
        for chunk in live_chunks:
            q_indices = xp.arange(
                chunk["phase_start"], chunk["phase_end"], dtype=xp.int64
            )[:, None]
            k_indices = xp.arange(
                chunk["key_start"], chunk["key_end"], dtype=xp.int64
            )[None, :]
            max_keys = q_indices + chunk["alignment_shift"]
            valid = (
                (k_indices <= max_keys)
                & (k_indices >= (max_keys - key_slots + 1))
            )[None, None, :, :]
            chunk["valid"] = valid
            with local_detail_scope("dilated.grouped.softmax"):
                probs = _masked_softmax_forward_bf16_cuda(
                    chunk["scores"], valid, scale
                )
            if probs is None:
                return None
            chunk["probs"] = probs
            del chunk["scores"]

    context_groups = []
    context_tmps = []
    for chunk in live_chunks:
        q_count = chunk["q_count"]
        k_count = chunk["k_count"]
        probs = chunk["probs"]
        tmp = xp.empty(
            (batch, n_q_heads, q_count, d_head), dtype=q.dtype
        )
        context_tmps.append(tmp)
        problems = []
        k_token0 = chunk["key_residue"] + chunk["key_start"] * dilation
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = (
                    int(probs.data.ptr) + b * int(probs.strides[0])
                    + h * int(probs.strides[1])
                )
                b_ptr = (
                    int(v.data.ptr) + b * int(v.strides[0])
                    + k_token0 * int(v.strides[1]) + kvh * int(v.strides[2])
                )
                c_ptr = (
                    int(tmp.data.ptr) + b * int(tmp.strides[0])
                    + h * int(tmp.strides[1])
                )
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, d_head, k_count,
                    k_count, v_row, d_head,
                    False, False,
                ))
        context_groups.append(RowMajorGemmGroup(tuple(problems)))

    with local_detail_scope("dilated.grouped.context_gemm"):
        if context_groups and not _cublas_grouped_bf16_gemm(tuple(context_groups)):
            return None

    context_heads = xp.zeros(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )
    for chunk, tmp in zip(live_chunks, context_tmps):
        q_slice = slice(
            chunk["query_residue"] + chunk["phase_start"] * dilation,
            chunk["query_residue"] + chunk["phase_end"] * dilation,
            dilation,
        )
        with local_detail_scope("dilated.grouped.context_store"):
            context_heads[:, :, q_slice, :] = tmp

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context
    return context, {
        "q": q, "k": k, "v": v,
        "window": int(window), "dilation": int(dilation), "offset": int(offset),
        "key_slots": key_slots,
        "scale": float(scale),
        "bf16_attention": True, "bf16_pipeline": True,
        "kv_head_indices": xp.asarray(kv_map_host, dtype=xp.int64),
        "kv_head_indices_host": tuple(kv_map_host),
        "chunks": tuple(chunks),
        "query_chunk_size": query_chunk_size,
        "grouped_cublas_dilated": True,
        "fused_dilated_softmax4": fused_softmax4,
    }


def _dilated_grouped_backward(dcontext, cache):
    """0054 grouped-cuBLAS backward for direct strided dilated phases."""
    q, k, v = cache["q"], cache["k"], cache["v"]
    batch, query_length, n_q_heads, d_head = map(int, q.shape)
    n_kv_heads = int(k.shape[2])
    dilation = int(cache["dilation"])
    scale = float(cache["scale"])
    kv_map_host = cache["kv_head_indices_host"]

    q_row = dilation * _row_stride_elems(q, 1)
    k_row = dilation * _row_stride_elems(k, 1)
    v_row = dilation * _row_stride_elems(v, 1)
    with local_detail_scope("dilated.grouped.dcontext_cast"):
        dc = xp.ascontiguousarray(dcontext.astype(q.dtype, copy=False))
    dc_row = dilation * _row_stride_elems(dc, 1)

    dq = xp.zeros(q.shape, dtype=xp.float32)
    dk = xp.zeros(k.shape, dtype=xp.float32)
    dv = xp.zeros(v.shape, dtype=xp.float32)
    live_chunks = [c for c in cache["chunks"] if c.get("probs") is not None]

    dprobs_groups = []
    for chunk in live_chunks:
        q_count, k_count = chunk["q_count"], chunk["k_count"]
        dprobs = xp.empty_like(chunk["probs"])
        chunk["dprobs"] = dprobs
        q_token0 = chunk["query_residue"] + chunk["phase_start"] * dilation
        k_token0 = chunk["key_residue"] + chunk["key_start"] * dilation
        problems = []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                a_ptr = (
                    int(dc.data.ptr) + b * int(dc.strides[0])
                    + q_token0 * int(dc.strides[1]) + h * int(dc.strides[2])
                )
                b_ptr = (
                    int(v.data.ptr) + b * int(v.strides[0])
                    + k_token0 * int(v.strides[1]) + kvh * int(v.strides[2])
                )
                c_ptr = (
                    int(dprobs.data.ptr) + b * int(dprobs.strides[0])
                    + h * int(dprobs.strides[1])
                )
                problems.append(RowMajorGemmProblem(
                    a_ptr, b_ptr, c_ptr,
                    q_count, k_count, d_head,
                    dc_row, v_row, k_count,
                    False, True,
                ))
        dprobs_groups.append(RowMajorGemmGroup(tuple(problems)))

    with local_detail_scope("dilated.grouped.dprobs_gemm"):
        if dprobs_groups and not _cublas_grouped_bf16_gemm(tuple(dprobs_groups)):
            return None

    fused_softmax4_backward = False
    if cache.get("fused_dilated_softmax4", False):
        with local_detail_scope("dilated.grouped.softmax_backward"):
            fused_softmax4_backward = (
                _dilated_softmax_backward4_inplace_bf16_cuda(
                    live_chunks, int(cache["key_slots"])
                )
            )
        if not fused_softmax4_backward:
            raise RuntimeError(
                "0054A fused dilated backward softmax declined a cache "
                "created by the fused forward path"
            )
    else:
        for chunk in live_chunks:
            with local_detail_scope("dilated.grouped.softmax_backward"):
                ds = _masked_softmax_backward_bf16_cuda(
                    chunk["dprobs"], chunk["probs"], chunk["valid"]
                )
            if ds is None:
                return None
            chunk["dscores"] = ds
            del chunk["dprobs"]
    cache["fused_dilated_softmax4_backward"] = fused_softmax4_backward

    grad_groups = []
    work = []
    for chunk in live_chunks:
        q_count, k_count = chunk["q_count"], chunk["k_count"]
        ds = chunk["dscores"]
        probs = chunk["probs"]
        dq_tmp = xp.empty(
            (batch, n_q_heads, q_count, d_head), dtype=q.dtype
        )
        dk_tmp = xp.empty(
            (batch, n_q_heads, k_count, d_head), dtype=q.dtype
        )
        dv_tmp = xp.empty_like(dk_tmp)
        work.append((chunk, dq_tmp, dk_tmp, dv_tmp))

        q_token0 = chunk["query_residue"] + chunk["phase_start"] * dilation
        k_token0 = chunk["key_residue"] + chunk["key_start"] * dilation
        dq_problems, dk_problems, dv_problems = [], [], []
        for b in range(batch):
            for h in range(n_q_heads):
                kvh = int(kv_map_host[h])
                ds_ptr = (
                    int(ds.data.ptr) + b * int(ds.strides[0])
                    + h * int(ds.strides[1])
                )
                p_ptr = (
                    int(probs.data.ptr) + b * int(probs.strides[0])
                    + h * int(probs.strides[1])
                )
                q_ptr = (
                    int(q.data.ptr) + b * int(q.strides[0])
                    + q_token0 * int(q.strides[1]) + h * int(q.strides[2])
                )
                k_ptr = (
                    int(k.data.ptr) + b * int(k.strides[0])
                    + k_token0 * int(k.strides[1]) + kvh * int(k.strides[2])
                )
                dc_ptr = (
                    int(dc.data.ptr) + b * int(dc.strides[0])
                    + q_token0 * int(dc.strides[1]) + h * int(dc.strides[2])
                )
                dq_ptr = (
                    int(dq_tmp.data.ptr) + b * int(dq_tmp.strides[0])
                    + h * int(dq_tmp.strides[1])
                )
                dk_ptr = (
                    int(dk_tmp.data.ptr) + b * int(dk_tmp.strides[0])
                    + h * int(dk_tmp.strides[1])
                )
                dv_ptr = (
                    int(dv_tmp.data.ptr) + b * int(dv_tmp.strides[0])
                    + h * int(dv_tmp.strides[1])
                )
                dq_problems.append(RowMajorGemmProblem(
                    ds_ptr, k_ptr, dq_ptr,
                    q_count, d_head, k_count,
                    k_count, k_row, d_head,
                    False, False,
                ))
                dk_problems.append(RowMajorGemmProblem(
                    ds_ptr, q_ptr, dk_ptr,
                    k_count, d_head, q_count,
                    k_count, q_row, d_head,
                    True, False,
                ))
                dv_problems.append(RowMajorGemmProblem(
                    p_ptr, dc_ptr, dv_ptr,
                    k_count, d_head, q_count,
                    k_count, dc_row, d_head,
                    True, False,
                ))
        grad_groups.extend((
            RowMajorGemmGroup(tuple(dq_problems)),
            RowMajorGemmGroup(tuple(dk_problems)),
            RowMajorGemmGroup(tuple(dv_problems)),
        ))

    with local_detail_scope("dilated.grouped.grad_gemm"):
        if grad_groups and not _cublas_grouped_bf16_gemm(tuple(grad_groups)):
            return None

    q_heads_for_kv = [
        tuple(i for i, source in enumerate(kv_map_host) if source == kvh)
        for kvh in range(n_kv_heads)
    ]
    for chunk, dq_tmp, dk_tmp, dv_tmp in work:
        q_slice = slice(
            chunk["query_residue"] + chunk["phase_start"] * dilation,
            chunk["query_residue"] + chunk["phase_end"] * dilation,
            dilation,
        )
        k_slice = slice(
            chunk["key_residue"] + chunk["key_start"] * dilation,
            chunk["key_residue"] + chunk["key_end"] * dilation,
            dilation,
        )
        with local_detail_scope("dilated.grouped.grad_store"):
            dq[:, q_slice, :, :] = (
                dq_tmp.transpose(0, 2, 1, 3).astype(xp.float32) * scale
            )
            for kvh, head_group in enumerate(q_heads_for_kv):
                if not head_group:
                    continue
                if len(head_group) == 1:
                    dk_native = dk_tmp[:, head_group[0], :, :]
                    dv_native = dv_tmp[:, head_group[0], :, :]
                else:
                    dk_native = xp.sum(dk_tmp[:, head_group, :, :], axis=1)
                    dv_native = xp.sum(dv_tmp[:, head_group, :, :], axis=1)
                dk[:, k_slice, kvh, :] += dk_native.astype(xp.float32) * scale
                dv[:, k_slice, kvh, :] += dv_native.astype(xp.float32)
        del chunk["dscores"]

    return dq, dk, dv

def dilated_attention_forward(
    q,
    k,
    v,
    window,
    dilation,
    offset=0,
    kv_head_indices=None,
    scale=None,
    return_cache=True,
    query_chunk_size=None,
):
    """Specialized exact fixed-phase dilated causal attention.

    The BF16 indexed-pipeline fast path keeps QK scores, probabilities, dP and
    dS in BF16 storage around 2-D Tensor-Core GEMMs.  Max/sum/exp and the
    softmax Jacobian remain FP32 inside fused CUDA kernels.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or k_d_head != d_head or key_length != query_length:
        raise ValueError("dilated self-attention requires matching q/k sequence shapes")

    window = int(window)
    dilation = int(dilation)
    offset = int(offset)
    if window <= 0:
        raise ValueError("window must be positive")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if offset < 0 or offset >= dilation:
        raise ValueError("offset must satisfy 0 <= offset < dilation")
    if offset >= window:
        raise ValueError("offset must be smaller than window")
    key_slots = ((window - 1 - offset) // dilation) + 1

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)

    kv_map, kv_map_host = _normalize_kv_head_mapping(
        kv_head_indices, n_q_heads, n_kv_heads
    )
    bf16_attention = is_bfloat16_dtype(q.dtype)
    bf16_pipeline = (
        _bf16_indexed_pipeline_enabled(q.dtype)
        and _get_indexed_bf16_pipeline_module() is not None
    )
    context_dtype = xp.float32 if bf16_attention else q.dtype

    # 0054: the fixed-phase pattern is a set of regular strided matrices.
    # Dispatch every phase/chunk through one grouped-cuBLAS score call and one
    # context call, reading Q/K/V directly instead of materializing xp.take
    # gathers and issuing one CuPy matmul per batch/head.
    if _dilated_grouped_fastpath_ready(q, k, v, bf16_pipeline):
        with local_detail_scope("dilated.grouped.forward"):
            grouped = _dilated_grouped_forward(
                q, k, v, window, dilation, offset, scale, kv_map_host,
                query_chunk_size, context_dtype, return_cache,
            )
        if grouped is not None:
            return grouped
        if _cublas_grouped_strict_enabled():
            raise RuntimeError("0054 grouped dilated forward unexpectedly declined")

    q_heads = q.transpose(0, 2, 1, 3)
    context_heads = xp.zeros(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0
    chunk_caches = [] if return_cache else None

    for query_residue in range(dilation):
        phase_length = (query_length - 1 - query_residue) // dilation + 1
        if phase_length <= 0:
            continue
        key_residue = (query_residue - offset) % dilation
        key_phase_length = (key_length - 1 - key_residue) // dilation + 1
        if key_phase_length <= 0:
            continue

        alignment_shift = 0 if query_residue >= offset else -1
        phase_chunk = _resolve_dilated_query_chunk_size(
            phase_length, query_chunk_size
        )

        for phase_start, phase_end in _query_chunks(phase_length, phase_chunk):
            max_key_end = phase_end + alignment_shift
            key_end = min(key_phase_length, max(0, max_key_end))
            key_start = max(0, phase_start + alignment_shift - key_slots + 1)

            q_token_slice = slice(
                query_residue + phase_start * dilation,
                query_residue + phase_end * dilation,
                dilation,
            )
            q_chunk = q_heads[:, :, q_token_slice, :]

            if key_end <= key_start:
                if return_cache:
                    chunk_caches.append((
                        query_residue, phase_start, phase_end, key_residue,
                        key_start, key_end, None, None, alignment_shift,
                    ))
                continue

            key_token_slice = slice(
                key_residue + key_start * dilation,
                key_residue + key_end * dilation,
                dilation,
            )
            k_span = xp.take(k[:, key_token_slice, :, :], kv_map, axis=2)
            k_heads = k_span.transpose(0, 2, 1, 3)

            q_indices = xp.arange(phase_start, phase_end, dtype=xp.int64)[:, None]
            k_indices = xp.arange(key_start, key_end, dtype=xp.int64)[None, :]
            max_keys = q_indices + alignment_shift
            valid = (
                (k_indices <= max_keys)
                & (k_indices >= (max_keys - key_slots + 1))
            )[None, None, :, :]

            if bf16_pipeline:
                q_count = phase_end - phase_start
                k_count = key_end - key_start
                scores_bf16 = xp.empty(
                    (batch, n_q_heads, q_count, k_count), dtype=q.dtype
                )
                for bidx in range(batch):
                    for hidx in range(n_q_heads):
                        q2d = xp.ascontiguousarray(q_chunk[bidx, hidx])
                        k2d = xp.ascontiguousarray(k_heads[bidx, hidx])
                        scores_bf16[bidx, hidx] = q2d @ k2d.T
                probs_chunk = _masked_softmax_forward_bf16_cuda(
                    scores_bf16, valid, scale
                )
                if probs_chunk is None:
                    raise RuntimeError(
                        "BF16 indexed softmax declined an internally generated "
                        "dilated-attention layout"
                    )
                del scores_bf16
            else:
                if bf16_attention:
                    q_score = q_chunk.astype("float32") * scale
                    k_score = k_heads.astype("float32")
                else:
                    q_score = q_chunk * (scale * score_prescale)
                    k_score = k_heads
                scores = xp.matmul(q_score, k_score.swapaxes(-1, -2))
                probs_chunk = _masked_softmax_forward(
                    scores, valid, logit_multiplier=(1.0 / score_prescale)
                )
                del scores, k_score

            del k_heads, k_span
            v_span = xp.take(v[:, key_token_slice, :, :], kv_map, axis=2)
            v_heads = v_span.transpose(0, 2, 1, 3)
            if bf16_pipeline:
                q_count = phase_end - phase_start
                context_bf16 = xp.empty(
                    (batch, n_q_heads, q_count, d_head), dtype=q.dtype
                )
                for bidx in range(batch):
                    for hidx in range(n_q_heads):
                        p2d = xp.ascontiguousarray(probs_chunk[bidx, hidx])
                        v2d = xp.ascontiguousarray(v_heads[bidx, hidx])
                        context_bf16[bidx, hidx] = p2d @ v2d
                context_heads[:, :, q_token_slice, :] = context_bf16
            else:
                if bf16_attention:
                    probs_compute = probs_chunk
                    v_compute = v_heads.astype("float32")
                else:
                    probs_compute = (
                        probs_chunk.astype(q.dtype, copy=False)
                        if is_low_precision_dtype(q.dtype)
                        else probs_chunk
                    )
                    v_compute = v_heads
                context_heads[:, :, q_token_slice, :] = xp.matmul(
                    probs_compute, v_compute
                )

            if return_cache:
                chunk_caches.append((
                    query_residue, phase_start, phase_end, key_residue,
                    key_start, key_end,
                    _cache_probs_for_backward(probs_chunk, q.dtype),
                    valid, alignment_shift,
                ))

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context
    return context, {
        "q": q,
        "k": k,
        "v": v,
        "window": window,
        "dilation": dilation,
        "offset": offset,
        "key_slots": key_slots,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "bf16_pipeline": bf16_pipeline,
        "kv_head_indices": kv_map,
        "kv_head_indices_host": kv_map_host,
        "chunks": chunk_caches,
        "query_chunk_size": query_chunk_size,
    }


def dilated_attention_backward(dcontext, cache):
    """Backward for :func:`dilated_attention_forward`."""
    q, k, v = cache["q"], cache["k"], cache["v"]
    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    batch, query_length, n_q_heads, d_head = q.shape
    n_kv_heads = k.shape[2]
    dilation = cache["dilation"]
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    bf16_pipeline = cache.get("bf16_pipeline", False)
    kv_map = cache["kv_head_indices"]
    kv_map_host = cache["kv_head_indices_host"]

    if cache.get("grouped_cublas_dilated", False):
        with local_detail_scope("dilated.grouped.backward"):
            grouped = _dilated_grouped_backward(dcontext, cache)
        if grouped is not None:
            return grouped
        if _cublas_grouped_strict_enabled():
            raise RuntimeError("0054 grouped dilated backward unexpectedly declined")

    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)
    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)
    q_heads_for_kv = [
        tuple(i for i, source in enumerate(kv_map_host) if source == kvh)
        for kvh in range(n_kv_heads)
    ]

    for (
        query_residue, phase_start, phase_end, key_residue,
        key_start, key_end, probs_chunk, valid, alignment_shift,
    ) in cache["chunks"]:
        if probs_chunk is None:
            continue
        q_token_slice = slice(
            query_residue + phase_start * dilation,
            query_residue + phase_end * dilation,
            dilation,
        )
        key_token_slice = slice(
            key_residue + key_start * dilation,
            key_residue + key_end * dilation,
            dilation,
        )
        q_chunk = q_heads[:, :, q_token_slice, :]
        dcontext_chunk = dcontext_heads[:, :, q_token_slice, :]
        v_span = xp.take(v[:, key_token_slice, :, :], kv_map, axis=2)
        v_heads = v_span.transpose(0, 2, 1, 3)

        if bf16_pipeline:
            q_count = phase_end - phase_start
            k_count = key_end - key_start
            dprobs = xp.empty(
                (batch, n_q_heads, q_count, k_count), dtype=q.dtype
            )
            dv_heads = xp.empty(
                (batch, n_q_heads, k_count, d_head), dtype=q.dtype
            )
            dc_bf16 = []
            for bidx in range(batch):
                dc_heads = []
                for hidx in range(n_q_heads):
                    dc2d = xp.ascontiguousarray(
                        dcontext_chunk[bidx, hidx].astype(q.dtype, copy=False)
                    )
                    dc_heads.append(dc2d)
                    v2d = xp.ascontiguousarray(v_heads[bidx, hidx])
                    p2d = xp.ascontiguousarray(probs_chunk[bidx, hidx])
                    dprobs[bidx, hidx] = dc2d @ v2d.T
                    dv_heads[bidx, hidx] = p2d.T @ dc2d
                dc_bf16.append(dc_heads)
            dscores = _masked_softmax_backward_bf16_cuda(
                dprobs, probs_chunk, valid
            )
            if dscores is None:
                raise RuntimeError(
                    "BF16 indexed backward softmax declined an internally "
                    "generated dilated-attention layout"
                )

            k_span = xp.take(k[:, key_token_slice, :, :], kv_map, axis=2)
            k_heads = k_span.transpose(0, 2, 1, 3)
            dk_heads = xp.empty(
                (batch, n_q_heads, k_count, d_head), dtype=q.dtype
            )
            for bidx in range(batch):
                for hidx in range(n_q_heads):
                    ds2d = xp.ascontiguousarray(dscores[bidx, hidx])
                    q2d = xp.ascontiguousarray(q_chunk[bidx, hidx])
                    k2d = xp.ascontiguousarray(k_heads[bidx, hidx])
                    dq_heads[bidx, hidx, q_token_slice, :] = (
                        (ds2d @ k2d).astype(grad_dtype) * scale
                    )
                    dk_heads[bidx, hidx] = ds2d.T @ q2d
            dk_heads = dk_heads.astype(grad_dtype) * scale
            dv_heads = dv_heads.astype(grad_dtype)
        else:
            if bf16_attention:
                dcontext_compute = dcontext_chunk.astype("float32")
                q_compute = q_chunk.astype("float32")
                v_compute = v_heads.astype("float32")
                probs_compute = _restore_cached_probs(probs_chunk, True)
            else:
                dcontext_compute = dcontext_chunk
                q_compute = q_chunk
                v_compute = v_heads
                probs_compute = (
                    probs_chunk.astype(q.dtype, copy=False)
                    if is_low_precision_dtype(q.dtype)
                    else probs_chunk
                )

            dprobs = xp.matmul(dcontext_compute, v_compute.swapaxes(-1, -2))
            dv_heads = xp.matmul(probs_compute.swapaxes(-1, -2), dcontext_compute)
            dscores = _masked_softmax_backward(dprobs, probs_compute, valid)
            dscores_compute = (
                dscores.astype("float32", copy=False)
                if bf16_attention
                else (
                    dscores.astype(q.dtype, copy=False)
                    if is_low_precision_dtype(q.dtype)
                    else dscores
                )
            )
            k_span = xp.take(k[:, key_token_slice, :, :], kv_map, axis=2)
            k_heads = k_span.transpose(0, 2, 1, 3)
            k_compute = k_heads.astype("float32") if bf16_attention else k_heads
            dq_heads[:, :, q_token_slice, :] = (
                xp.matmul(dscores_compute, k_compute) * scale
            )
            dk_heads = (
                xp.matmul(dscores_compute.swapaxes(-1, -2), q_compute) * scale
            )

        for kvh, head_group in enumerate(q_heads_for_kv):
            if not head_group:
                continue
            if len(head_group) == 1:
                dk_native = dk_heads[:, head_group[0], :, :]
                dv_native = dv_heads[:, head_group[0], :, :]
            else:
                dk_native = xp.sum(dk_heads[:, head_group, :, :], axis=1)
                dv_native = xp.sum(dv_heads[:, head_group, :, :], axis=1)
            dk[:, key_token_slice, kvh, :] += dk_native.astype(
                grad_dtype, copy=False
            )
            dv[:, key_token_slice, kvh, :] += dv_native.astype(
                grad_dtype, copy=False
            )

    return dq_heads.transpose(0, 2, 1, 3), dk, dv


def global_sparse_attention_forward(
    q,
    k,
    v,
    stride,
    offset=0,
    include_current=True,
    kv_head_indices=None,
    scale=None,
    return_cache=True,
):
    """Specialized fixed-anchor whole-prefix sparse attention.

    With ``MINI_LLM_FUSED_BF16_INDEXED_PIPELINE=1`` the regular anchor branch
    uses 2-D BF16 Tensor-Core GEMMs and BF16 score/probability storage around a
    fused FP32 softmax kernel.  The single exact-current slot remains a tiny
    FP32 elementwise branch and is folded into the same BF16 softmax matrix.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or k_d_head != d_head or key_length != query_length:
        raise ValueError("global sparse self-attention requires matching q/k sequence shapes")

    stride = int(stride)
    offset = int(offset)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if offset < 0 or offset >= stride:
        raise ValueError("offset must satisfy 0 <= offset < stride")
    if not isinstance(include_current, (bool, xp.bool_)):
        raise TypeError("include_current must be a bool")

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)
    kv_map, kv_map_host = _normalize_kv_head_mapping(
        kv_head_indices, n_q_heads, n_kv_heads
    )

    if stride == 1:
        if offset != 0:
            raise ValueError("stride=1 requires offset=0")
        if return_cache:
            context, local_cache = local_window_attention_forward(
                q, k, v, key_length,
                kv_head_indices=kv_map_host, scale=scale, return_cache=True,
            )
            return context, {"fallback_local": local_cache}
        return local_window_attention_forward(
            q, k, v, key_length,
            kv_head_indices=kv_map_host, scale=scale, return_cache=False,
        )

    q_heads = q.transpose(0, 2, 1, 3)
    bf16_attention = is_bfloat16_dtype(q.dtype)
    bf16_pipeline = (
        _bf16_indexed_pipeline_enabled(q.dtype)
        and _get_indexed_bf16_pipeline_module() is not None
    )
    context_dtype = xp.float32 if bf16_attention else q.dtype
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0

    anchors = xp.arange(offset, key_length, stride, dtype=xp.int64)
    n_anchors = int(anchors.shape[0])
    add_current = bool(include_current)
    n_slots = n_anchors + int(add_current)

    if n_slots == 0:
        context = xp.zeros(q.shape, dtype=context_dtype)
        if not return_cache:
            return context
        return context, {
            "q": q, "k": k, "v": v, "stride": stride, "offset": offset,
            "include_current": include_current, "anchors": anchors,
            "probs": None, "valid": None, "scale": scale,
            "bf16_attention": bf16_attention, "bf16_pipeline": bf16_pipeline,
            "kv_head_indices": kv_map, "kv_head_indices_host": kv_map_host,
        }

    k_anchor_heads = None
    current_k_heads = None
    if bf16_pipeline:
        scores = xp.empty(
            (batch, n_q_heads, query_length, n_slots), dtype=q.dtype
        )
        slot = 0
        if n_anchors:
            k_anchor = xp.take(k[:, offset:key_length:stride, :, :], kv_map, axis=2)
            k_anchor_heads = k_anchor.transpose(0, 2, 1, 3)
            for bidx in range(batch):
                for hidx in range(n_q_heads):
                    q2d = xp.ascontiguousarray(q_heads[bidx, hidx])
                    k2d = xp.ascontiguousarray(k_anchor_heads[bidx, hidx])
                    scores[bidx, hidx, :, :n_anchors] = q2d @ k2d.T
            slot = n_anchors
        if add_current:
            current_k = xp.take(k, kv_map, axis=2)
            current_k_heads = current_k.transpose(0, 2, 1, 3)
            # Only one slot per query; keeping this tiny branch in FP32 avoids
            # introducing a custom diagonal-dot kernel.
            current_scores = xp.sum(
                q_heads.astype("float32") * current_k_heads.astype("float32"),
                axis=-1,
            )
            scores[..., slot] = current_scores.astype(q.dtype)
    else:
        if bf16_attention:
            q_score = q_heads.astype("float32") * scale
        else:
            q_score = q_heads * (scale * score_prescale)
        score_parts = []
        if n_anchors:
            k_anchor = xp.take(k[:, offset:key_length:stride, :, :], kv_map, axis=2)
            k_anchor_heads = k_anchor.transpose(0, 2, 1, 3)
            k_score = k_anchor_heads.astype("float32") if bf16_attention else k_anchor_heads
            score_parts.append(xp.matmul(q_score, k_score.swapaxes(-1, -2)))
        if add_current:
            current_k = xp.take(k, kv_map, axis=2)
            current_k_heads = current_k.transpose(0, 2, 1, 3)
            current_k_score = current_k_heads.astype("float32") if bf16_attention else current_k_heads
            score_parts.append(xp.sum(q_score * current_k_score, axis=-1, keepdims=True))
        scores = score_parts[0] if len(score_parts) == 1 else xp.concatenate(score_parts, axis=-1)

    queries = xp.arange(query_length, dtype=xp.int64)[:, None]
    valid_parts = []
    if n_anchors:
        valid_parts.append(anchors[None, :] <= queries)
    if add_current:
        positions = xp.arange(query_length, dtype=xp.int64)
        on_phase = (positions >= offset) & (((positions - offset) % stride) == 0)
        valid_parts.append((~on_phase)[:, None])
    valid_2d = valid_parts[0] if len(valid_parts) == 1 else xp.concatenate(valid_parts, axis=1)
    valid = valid_2d[None, None, :, :]

    if bf16_pipeline:
        probs = _masked_softmax_forward_bf16_cuda(scores, valid, scale)
        if probs is None:
            raise RuntimeError(
                "BF16 indexed softmax declined an internally generated "
                "global-sparse layout"
            )
    else:
        probs = _masked_softmax_forward(
            scores, valid, logit_multiplier=(1.0 / score_prescale)
        )
    del scores

    context_heads = xp.zeros(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )
    slot = 0
    if n_anchors:
        probs_anchor = probs[..., :n_anchors]
        v_anchor = xp.take(v[:, offset:key_length:stride, :, :], kv_map, axis=2)
        v_anchor_heads = v_anchor.transpose(0, 2, 1, 3)
        if bf16_pipeline:
            for bidx in range(batch):
                for hidx in range(n_q_heads):
                    p2d = xp.ascontiguousarray(probs_anchor[bidx, hidx])
                    v2d = xp.ascontiguousarray(v_anchor_heads[bidx, hidx])
                    context_heads[bidx, hidx] += (
                        p2d @ v2d
                    ).astype(context_dtype)
        else:
            v_compute = v_anchor_heads.astype("float32") if bf16_attention else v_anchor_heads
            probs_compute = probs_anchor if bf16_attention else (
                probs_anchor.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype) else probs_anchor
            )
            context_heads += xp.matmul(probs_compute, v_compute)
        slot = n_anchors

    if add_current:
        probs_current = probs[..., slot]
        current_v = xp.take(v, kv_map, axis=2)
        current_v_heads = current_v.transpose(0, 2, 1, 3)
        if bf16_pipeline:
            context_heads += (
                probs_current.astype("float32")[..., None]
                * current_v_heads.astype("float32")
            )
        else:
            current_v_compute = current_v_heads.astype("float32") if bf16_attention else current_v_heads
            probs_current_compute = probs_current if bf16_attention else (
                probs_current.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype) else probs_current
            )
            context_heads += probs_current_compute[..., None] * current_v_compute

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context
    return context, {
        "q": q, "k": k, "v": v, "stride": stride, "offset": offset,
        "include_current": include_current, "anchors": anchors,
        "probs": _cache_probs_for_backward(probs, q.dtype), "valid": valid,
        "scale": scale, "bf16_attention": bf16_attention,
        "bf16_pipeline": bf16_pipeline,
        "kv_head_indices": kv_map, "kv_head_indices_host": kv_map_host,
    }


def global_sparse_attention_backward(dcontext, cache):
    """Backward for :func:`global_sparse_attention_forward`."""
    if "fallback_local" in cache:
        return local_window_attention_backward(dcontext, cache["fallback_local"])

    q, k, v = cache["q"], cache["k"], cache["v"]
    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    batch, query_length, n_q_heads, d_head = q.shape
    n_kv_heads = k.shape[2]
    stride = cache["stride"]
    offset = cache["offset"]
    include_current = cache["include_current"]
    anchors = cache["anchors"]
    probs = cache["probs"]
    valid = cache["valid"]
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    bf16_pipeline = cache.get("bf16_pipeline", False)
    kv_map = cache["kv_head_indices"]
    kv_map_host = cache["kv_head_indices_host"]

    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq = xp.zeros(q.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)
    if probs is None:
        return dq, dk, dv

    q_heads = q.transpose(0, 2, 1, 3)
    dcontext_heads = dcontext.transpose(0, 2, 1, 3)
    n_anchors = int(anchors.shape[0])
    add_current = bool(include_current)

    if bf16_pipeline:
        dprobs = xp.empty(probs.shape, dtype=q.dtype)
        slot = 0
        dc_bf16 = dcontext_heads.astype(q.dtype, copy=False)
        v_anchor_heads = None
        current_v_heads = None
        if n_anchors:
            v_anchor = xp.take(v[:, offset:v.shape[1]:stride, :, :], kv_map, axis=2)
            v_anchor_heads = v_anchor.transpose(0, 2, 1, 3)
            for bidx in range(batch):
                for hidx in range(n_q_heads):
                    dc2d = xp.ascontiguousarray(dc_bf16[bidx, hidx])
                    v2d = xp.ascontiguousarray(v_anchor_heads[bidx, hidx])
                    dprobs[bidx, hidx, :, :n_anchors] = dc2d @ v2d.T
            slot = n_anchors
        if add_current:
            current_v = xp.take(v, kv_map, axis=2)
            current_v_heads = current_v.transpose(0, 2, 1, 3)
            dcurrent = xp.sum(
                dcontext_heads.astype("float32") * current_v_heads.astype("float32"),
                axis=-1,
            )
            dprobs[..., slot] = dcurrent.astype(q.dtype)
        dscores = _masked_softmax_backward_bf16_cuda(dprobs, probs, valid)
        if dscores is None:
            raise RuntimeError(
                "BF16 indexed backward softmax declined an internally "
                "generated global-sparse layout"
            )
    else:
        probs_f32 = _restore_cached_probs(probs, bf16_attention)
        q_compute = q_heads.astype("float32") if bf16_attention else q_heads
        dcontext_compute = dcontext_heads.astype("float32") if bf16_attention else dcontext_heads
        dprobs_parts = []
        v_anchor_heads = None
        if n_anchors:
            v_anchor = xp.take(v[:, offset:v.shape[1]:stride, :, :], kv_map, axis=2)
            v_anchor_heads = v_anchor.transpose(0, 2, 1, 3)
            v_compute = v_anchor_heads.astype("float32") if bf16_attention else v_anchor_heads
            dprobs_parts.append(xp.matmul(dcontext_compute, v_compute.swapaxes(-1, -2)))
        current_v_heads = None
        if add_current:
            current_v = xp.take(v, kv_map, axis=2)
            current_v_heads = current_v.transpose(0, 2, 1, 3)
            current_v_compute = current_v_heads.astype("float32") if bf16_attention else current_v_heads
            dprobs_parts.append(xp.sum(dcontext_compute * current_v_compute, axis=-1, keepdims=True))
        dprobs = dprobs_parts[0] if len(dprobs_parts) == 1 else xp.concatenate(dprobs_parts, axis=-1)
        dscores = _masked_softmax_backward(dprobs, probs_f32, valid)

    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk_anchor_heads = None
    dv_anchor_heads = None
    slot = 0
    if n_anchors:
        if bf16_pipeline:
            ds_anchor = dscores[..., :n_anchors]
            probs_anchor = probs[..., :n_anchors]
            k_anchor = xp.take(k[:, offset:k.shape[1]:stride, :, :], kv_map, axis=2)
            k_anchor_heads = k_anchor.transpose(0, 2, 1, 3)
            dk_anchor_heads = xp.empty(
                (batch, n_q_heads, n_anchors, d_head), dtype=grad_dtype
            )
            dv_anchor_heads = xp.empty_like(dk_anchor_heads)
            for bidx in range(batch):
                for hidx in range(n_q_heads):
                    ds2d = xp.ascontiguousarray(ds_anchor[bidx, hidx])
                    p2d = xp.ascontiguousarray(probs_anchor[bidx, hidx])
                    q2d = xp.ascontiguousarray(q_heads[bidx, hidx])
                    k2d = xp.ascontiguousarray(k_anchor_heads[bidx, hidx])
                    dc2d = xp.ascontiguousarray(
                        dcontext_heads[bidx, hidx].astype(q.dtype, copy=False)
                    )
                    dq_heads[bidx, hidx] += (ds2d @ k2d).astype(grad_dtype) * scale
                    dk_anchor_heads[bidx, hidx] = (ds2d.T @ q2d).astype(grad_dtype) * scale
                    dv_anchor_heads[bidx, hidx] = (p2d.T @ dc2d).astype(grad_dtype)
        else:
            probs_f32 = _restore_cached_probs(probs, bf16_attention)
            dscores_compute = dscores.astype("float32", copy=False) if bf16_attention else (
                dscores.astype(q.dtype, copy=False) if is_low_precision_dtype(q.dtype) else dscores
            )
            q_compute = q_heads.astype("float32") if bf16_attention else q_heads
            dcontext_compute = dcontext_heads.astype("float32") if bf16_attention else dcontext_heads
            ds_anchor = dscores_compute[..., :n_anchors]
            k_anchor = xp.take(k[:, offset:k.shape[1]:stride, :, :], kv_map, axis=2)
            k_anchor_heads = k_anchor.transpose(0, 2, 1, 3)
            k_compute = k_anchor_heads.astype("float32") if bf16_attention else k_anchor_heads
            dq_heads += xp.matmul(ds_anchor, k_compute) * scale
            dk_anchor_heads = xp.matmul(ds_anchor.swapaxes(-1, -2), q_compute) * scale
            probs_anchor = probs_f32[..., :n_anchors]
            probs_compute = probs_anchor if bf16_attention else (
                probs_anchor.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype) else probs_anchor
            )
            dv_anchor_heads = xp.matmul(probs_compute.swapaxes(-1, -2), dcontext_compute)
        slot = n_anchors

    dk_current_heads = None
    dv_current_heads = None
    if add_current:
        if bf16_pipeline:
            ds_current = dscores[..., slot].astype("float32")
            probs_current = probs[..., slot].astype("float32")
            q_compute_current = q_heads.astype("float32")
            dc_compute_current = dcontext_heads.astype("float32")
        else:
            probs_f32 = _restore_cached_probs(probs, bf16_attention)
            dscores_compute = dscores.astype("float32", copy=False) if bf16_attention else (
                dscores.astype(q.dtype, copy=False) if is_low_precision_dtype(q.dtype) else dscores
            )
            ds_current = dscores_compute[..., slot]
            probs_current = probs_f32[..., slot]
            q_compute_current = q_heads.astype("float32") if bf16_attention else q_heads
            dc_compute_current = dcontext_heads.astype("float32") if bf16_attention else dcontext_heads
        current_k = xp.take(k, kv_map, axis=2)
        current_k_heads = current_k.transpose(0, 2, 1, 3)
        current_k_compute = current_k_heads.astype("float32") if bf16_attention else current_k_heads
        dq_heads += ds_current[..., None] * current_k_compute * scale
        dk_current_heads = ds_current[..., None] * q_compute_current * scale
        dv_current_heads = probs_current[..., None] * dc_compute_current

    q_heads_for_kv = [
        tuple(i for i, source in enumerate(kv_map_host) if source == kvh)
        for kvh in range(n_kv_heads)
    ]
    for kvh, head_group in enumerate(q_heads_for_kv):
        if not head_group:
            continue
        if n_anchors:
            if len(head_group) == 1:
                dk_native = dk_anchor_heads[:, head_group[0], :, :]
                dv_native = dv_anchor_heads[:, head_group[0], :, :]
            else:
                dk_native = xp.sum(dk_anchor_heads[:, head_group, :, :], axis=1)
                dv_native = xp.sum(dv_anchor_heads[:, head_group, :, :], axis=1)
            dk[:, offset:k.shape[1]:stride, kvh, :] += dk_native.astype(grad_dtype, copy=False)
            dv[:, offset:v.shape[1]:stride, kvh, :] += dv_native.astype(grad_dtype, copy=False)
        if add_current:
            if len(head_group) == 1:
                dk_native = dk_current_heads[:, head_group[0], :, :]
                dv_native = dv_current_heads[:, head_group[0], :, :]
            else:
                dk_native = xp.sum(dk_current_heads[:, head_group, :, :], axis=1)
                dv_native = xp.sum(dv_current_heads[:, head_group, :, :], axis=1)
            dk[:, :, kvh, :] += dk_native.astype(grad_dtype, copy=False)
            dv[:, :, kvh, :] += dv_native.astype(grad_dtype, copy=False)

    return dq_heads.transpose(0, 2, 1, 3), dk, dv

# ---------------------------------------------------------------------------
# Specialized block-retrieval attention
# ---------------------------------------------------------------------------

def block_retrieval_attention_forward(
    q,
    k,
    v,
    selected_blocks,
    selected_weights,
    route_starts,
    block_size,
    routing_stride,
    kv_head_indices=None,
    scale=None,
    weight_mode="logit_bias",
    weight_scale=1.0,
    weight_eps=1e-8,
    return_cache=True,
):
    """Exact attention over router-selected contiguous history blocks.

    One router decision controls ``routing_stride`` consecutive query tokens.
    The generic indexed kernel expands the same selected K/V tokens once per
    query, producing a temporary ``[B,H,Q,K,D]`` gather.  Retrieval is block
    structured, so we instead gather each selected block once per route/head:

        selected K/V: [B,R,H,Kblocks*block_size,D]
        routed Q:     [B,R,H,routing_stride,D]

    and evaluate all routes as one batched matrix multiplication.  The result
    is mathematically identical to ``build_*_block_retrieval_plan`` followed by
    :func:`indexed_attention_forward` for router-produced valid selections.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B,T,H,Dh)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if selected_blocks.shape != selected_weights.shape:
        raise ValueError("selected_blocks and selected_weights must have the same shape")
    if selected_blocks.ndim != 4:
        raise ValueError("selected tensors must have shape (B,R,Hrouter,Kblocks)")
    if route_starts.ndim != 1:
        raise ValueError("route_starts must be one-dimensional")

    batch, query_length, n_q_heads, d_head = q.shape
    k_batch, key_length, n_kv_heads, k_d_head = k.shape
    if k_batch != batch or key_length != query_length or k_d_head != d_head:
        raise ValueError("block retrieval self-attention requires matching q/k sequence shapes")

    r_batch, n_routes, n_router_heads, n_selected = selected_blocks.shape
    if r_batch != batch or int(route_starts.shape[0]) != n_routes:
        raise ValueError("selected route dimensions must match batch/route_starts")
    if n_router_heads not in {1, n_q_heads}:
        raise ValueError("router head dimension must be 1 or equal n_q_heads")

    block_size = int(block_size)
    routing_stride = int(routing_stride)
    if block_size <= 0 or routing_stride <= 0:
        raise ValueError("block_size and routing_stride must be positive")
    if weight_mode not in {"logit_bias", "none"}:
        raise ValueError("weight_mode must be 'logit_bias' or 'none'")
    weight_scale = float(weight_scale)
    weight_eps = float(weight_eps)
    if weight_eps <= 0 or not math.isfinite(weight_eps):
        raise ValueError("weight_eps must be positive and finite")
    if not math.isfinite(weight_scale):
        raise ValueError("weight_scale must be finite")

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scale = float(scale)
    kv_map, kv_map_host = _normalize_kv_head_mapping(
        kv_head_indices, n_q_heads, n_kv_heads
    )
    bf16_attention = is_bfloat16_dtype(q.dtype)
    bf16_pipeline_requested = _bf16_retrieval_pipeline_enabled(q.dtype)
    context_dtype = xp.float32 if bf16_attention else q.dtype
    context = xp.zeros(q.shape, dtype=context_dtype)

    if n_routes == 0:
        if not return_cache:
            return context
        return context, {
            "q": q,
            "k": k,
            "v": v,
            "selected_blocks": selected_blocks,
            "selected_weights": selected_weights,
            "route_starts": route_starts,
            "block_size": block_size,
            "routing_stride": routing_stride,
            "kv_head_indices": kv_map,
            "kv_head_indices_host": kv_map_host,
            "scale": scale,
            "bf16_attention": bf16_attention,
            "bf16_pipeline": False,
            "weight_mode": weight_mode,
            "weight_scale": weight_scale,
            "weight_eps": weight_eps,
            "probs": None,
            "query_positions": xp.zeros((0, routing_stride), dtype=xp.int64),
            "query_valid": xp.zeros((0, routing_stride), dtype=bool),
            "key_indices": xp.zeros((batch, 0, n_q_heads, n_selected * block_size), dtype=xp.int64),
            "router_had_shared_head": n_router_heads == 1,
        }

    if n_router_heads == 1 and n_q_heads != 1:
        selected_for_heads = xp.broadcast_to(
            selected_blocks, (batch, n_routes, n_q_heads, n_selected)
        )
        weights_for_heads = xp.broadcast_to(
            selected_weights, (batch, n_routes, n_q_heads, n_selected)
        )
    else:
        selected_for_heads = selected_blocks
        weights_for_heads = selected_weights

    # Route query positions are disjoint by construction.  The final route may
    # be shorter than routing_stride for arbitrary sequence lengths, so retain
    # an explicit validity mask while keeping a regular batched shape.
    query_offsets = xp.arange(routing_stride, dtype=xp.int64)[None, :]
    query_positions_raw = route_starts[:, None] + query_offsets
    query_valid = query_positions_raw < query_length
    query_positions = xp.minimum(query_positions_raw, max(query_length - 1, 0))

    q_routes = q[:, query_positions, :, :].transpose(0, 1, 3, 2, 4)
    # q[:, positions] -> [B,R,S,H,D]; transpose -> [B,R,H,S,D]

    block_offsets = xp.arange(block_size, dtype=xp.int64)
    key_indices = (
        selected_for_heads[..., None] * block_size
        + block_offsets[None, None, None, None, :]
    ).reshape(batch, n_routes, n_q_heads, n_selected * block_size)

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_map[None, None, :, None]
    k_selected = k[batch_ids, key_indices, kv_ids, :]

    bf16_pipeline = (
        bf16_pipeline_requested
        and is_bfloat16_dtype(selected_weights.dtype)
        and _retrieval_bf16_shape_ok(
            routing_stride, n_selected * block_size, d_head
        )
        and _get_retrieval_bf16_pipeline_module() is not None
    )

    valid_scores = query_valid[None, :, None, :, None]
    if bf16_pipeline:
        # One strided Tensor-Core workload covers every [B,R,H] route matrix.
        # Scores stay BF16; scale and router logit bias are applied in FP32
        # inside the fused softmax kernel before probabilities are rounded back
        # to BF16 storage.
        scores = _retrieval_bf16_gemm_nt(q_routes, k_selected)
        if scores is None:
            raise RuntimeError(
                "BF16 retrieval score GEMM declined a validated route layout"
            )
        probs = _retrieval_softmax_forward_bf16(
            scores,
            selected_weights,
            query_valid,
            n_router_heads,
            block_size,
            scale,
            weight_mode,
            weight_scale,
            weight_eps,
        )
        if probs is None:
            raise RuntimeError(
                "BF16 retrieval softmax declined a validated route layout"
            )
        del scores, k_selected

        v_selected = v[batch_ids, key_indices, kv_ids, :]
        context_routes = _retrieval_bf16_gemm_nn(probs, v_selected)
        if context_routes is None:
            raise RuntimeError(
                "BF16 retrieval context GEMM declined a validated route layout"
            )
        if not _retrieval_route_scatter_bf16_to_f32(
            context_routes, context, query_positions, query_valid
        ):
            raise RuntimeError("BF16 retrieval route scatter fast path declined")
    else:
        if bf16_attention:
            q_score = q_routes.astype("float32") * scale
            k_score = k_selected.astype("float32")
        else:
            q_score = q_routes * scale
            k_score = k_selected
        scores = xp.matmul(q_score, k_score.swapaxes(-1, -2))

        if weight_mode == "logit_bias":
            weights_work = (
                weights_for_heads.astype("float32", copy=False)
                if is_low_precision_dtype(weights_for_heads.dtype)
                else weights_for_heads
            )
            block_bias = weight_scale * xp.log(weights_work + weight_eps)
            token_bias = xp.repeat(block_bias, block_size, axis=-1)
            scores = scores + token_bias[..., None, :]

        probs = _masked_softmax_forward(scores, valid_scores)
        del scores, q_score, k_score, k_selected

        v_selected = v[batch_ids, key_indices, kv_ids, :]
        if bf16_attention:
            v_compute = v_selected.astype("float32")
            probs_compute = probs
        else:
            v_compute = v_selected
            probs_compute = (
                probs.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs
            )
        context_routes = xp.matmul(probs_compute, v_compute)
        if not _retrieval_route_scatter_f32(
            context_routes, context, query_positions, query_valid
        ):
            # [B,R,H,S,D] -> [B,R,S,H,D]
            context_route_tokens = context_routes.transpose(0, 1, 3, 2, 4)
            flat_valid = query_valid.reshape(-1)
            flat_positions = query_positions.reshape(-1)[flat_valid]
            flat_context = context_route_tokens.reshape(
                batch, n_routes * routing_stride, n_q_heads, d_head
            )[:, flat_valid, :, :]
            context[:, flat_positions, :, :] = flat_context

    if not return_cache:
        return context

    return context, {
        "q": q,
        "k": k,
        "v": v,
        "selected_blocks": selected_blocks,
        "selected_weights": selected_weights,
        "route_starts": route_starts,
        "block_size": block_size,
        "routing_stride": routing_stride,
        "kv_head_indices": kv_map,
        "kv_head_indices_host": kv_map_host,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "bf16_pipeline": bf16_pipeline,
        "weight_mode": weight_mode,
        "weight_scale": weight_scale,
        "weight_eps": weight_eps,
        "probs": _cache_probs_for_backward(probs, q.dtype),
        "query_positions": query_positions,
        "query_valid": query_valid,
        "key_indices": key_indices,
        "router_had_shared_head": n_router_heads == 1,
    }


def block_retrieval_attention_backward(dcontext, cache):
    """Backward for :func:`block_retrieval_attention_forward`.

    K/V gradients are first reduced over all query tokens controlled by a route
    using GEMMs.  Only the much smaller route-level ``[B,R,H,K,D]`` results are
    scatter-added back to native historical K/V positions.  This removes the
    generic kernel's query-repeated K/V scatter volume.

    Returns ``dq, dk, dv, dweights``.  ``dweights`` has the same shape as the
    router's selected weight tensor and is zero for ``weight_mode='none'``.
    """
    q, k, v = cache["q"], cache["k"], cache["v"]
    if dcontext.shape != q.shape:
        raise ValueError("dcontext must have the same shape as q/context")

    selected_weights = cache["selected_weights"]
    batch, n_routes, n_router_heads, n_selected = selected_weights.shape
    _, query_length, n_q_heads, d_head = q.shape
    n_kv_heads = k.shape[2]
    block_size = int(cache["block_size"])
    routing_stride = int(cache["routing_stride"])
    bf16_attention = cache["bf16_attention"]
    scale = cache["scale"]
    kv_map = cache["kv_head_indices"]
    key_indices = cache["key_indices"]
    query_positions = cache["query_positions"]
    query_valid = cache["query_valid"]
    probs = cache["probs"]
    bf16_pipeline = cache.get("bf16_pipeline", False)

    grad_dtype = xp.float32 if bf16_attention else q.dtype
    dq = xp.zeros(q.shape, dtype=grad_dtype)
    dk = xp.zeros(k.shape, dtype=grad_dtype)
    dv = xp.zeros(v.shape, dtype=grad_dtype)
    dweights = xp.zeros(selected_weights.shape, dtype=grad_dtype)
    if n_routes == 0:
        return dq, dk, dv, dweights

    dcontext_routes = dcontext[:, query_positions, :, :].transpose(0, 1, 3, 2, 4)
    q_routes = q[:, query_positions, :, :].transpose(0, 1, 3, 2, 4)
    valid_q = query_valid[None, :, None, :, None]

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_map[None, None, :, None]
    v_selected = v[batch_ids, key_indices, kv_ids, :]

    if bf16_pipeline:
        # Invalid rows already have zero probabilities/dscores, so the clipped
        # final-route query position never contributes to dv/dk.  Avoiding the
        # reference xp.where() also avoids building two FP32 route tensors.
        dc_bf16 = dcontext_routes.astype(q.dtype, copy=False)
        dprobs = _retrieval_bf16_gemm_nt(dc_bf16, v_selected)
        dv_selected = _retrieval_bf16_gemm_tn(probs, dc_bf16)
        if dprobs is None or dv_selected is None:
            raise RuntimeError(
                "BF16 retrieval dP/dV GEMM declined a validated route layout"
            )
        dscores = _retrieval_softmax_backward_bf16(
            dprobs, probs, query_valid
        )
        if dscores is None:
            raise RuntimeError(
                "BF16 retrieval backward softmax declined a validated layout"
            )

        k_selected = k[batch_ids, key_indices, kv_ids, :]
        dq_routes = _retrieval_bf16_gemm_nn(dscores, k_selected, alpha=scale)
        dk_selected = _retrieval_bf16_gemm_tn(dscores, q_routes, alpha=scale)
        if dq_routes is None or dk_selected is None:
            raise RuntimeError(
                "BF16 retrieval dQ/dK GEMM declined a validated route layout"
            )

        if not _retrieval_route_scatter_bf16_to_f32(
            dq_routes, dq, query_positions, query_valid
        ):
            raise RuntimeError("BF16 retrieval dQ scatter fast path declined")
        if not _retrieval_kv_scatter_bf16_to_f32(
            dk_selected, dv_selected, dk, dv, key_indices, kv_map
        ):
            raise RuntimeError("BF16 retrieval dK/dV scatter fast path declined")

        if cache["weight_mode"] == "logit_bias":
            direct = _retrieval_bias_grad_bf16(
                dscores,
                selected_weights,
                n_router_heads,
                block_size,
                cache["weight_scale"],
                cache["weight_eps"],
            )
            if direct is None:
                raise RuntimeError(
                    "BF16 retrieval router-weight gradient fast path declined"
                )
            dweights = direct
    else:
        probs_f32 = _restore_cached_probs(probs, bf16_attention)
        dcontext_routes = xp.where(valid_q, dcontext_routes, 0.0)
        q_routes = xp.where(valid_q, q_routes, 0.0)

        if bf16_attention:
            dcontext_compute = dcontext_routes.astype("float32")
            q_compute = q_routes.astype("float32")
            v_compute = v_selected.astype("float32")
            probs_compute = probs_f32
        else:
            dcontext_compute = dcontext_routes
            q_compute = q_routes
            v_compute = v_selected
            probs_compute = (
                probs.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs
            )

        dprobs = xp.matmul(dcontext_compute, v_compute.swapaxes(-1, -2))
        dv_selected = xp.matmul(probs_compute.swapaxes(-1, -2), dcontext_compute)
        dscores = _masked_softmax_backward(dprobs, probs_f32, valid_q)
        dscores_compute = (
            dscores.astype("float32", copy=False)
            if bf16_attention
            else (
                dscores.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else dscores
            )
        )

        k_selected = k[batch_ids, key_indices, kv_ids, :]
        k_compute = k_selected.astype("float32") if bf16_attention else k_selected
        dq_routes = xp.matmul(dscores_compute, k_compute) * scale
        dk_selected = xp.matmul(dscores_compute.swapaxes(-1, -2), q_compute) * scale

        # Routes own disjoint query ranges, so Q gradients can be assigned without
        # a scatter-add.  The CUDA fast path indexes the regular route tensor
        # directly and avoids CuPy boolean-mask scans entirely.
        if not _retrieval_route_scatter_f32(
            dq_routes, dq, query_positions, query_valid
        ):
            dq_route_tokens = dq_routes.transpose(0, 1, 3, 2, 4).reshape(
                batch, n_routes * routing_stride, n_q_heads, d_head
            )
            flat_valid = query_valid.reshape(-1)
            flat_positions = query_positions.reshape(-1)[flat_valid]
            dq[:, flat_positions, :, :] = dq_route_tokens[:, flat_valid, :, :]

        # K/V may be selected by several routes or query heads, so accumulation is
        # genuinely irregular.  The CUDA path performs both FP32 atomic scatter
        # adds in one launch instead of two generic ``xp.add.at`` calls.
        dk_selected_grad = dk_selected.astype(grad_dtype, copy=False)
        dv_selected_grad = dv_selected.astype(grad_dtype, copy=False)
        if not _retrieval_kv_scatter_add_f32(
            dk_selected_grad, dv_selected_grad, dk, dv, key_indices, kv_map
        ):
            scatter_batch = xp.broadcast_to(batch_ids, key_indices.shape)
            scatter_kv = xp.broadcast_to(kv_ids, key_indices.shape)
            xp.add.at(dk, (scatter_batch, key_indices, scatter_kv), dk_selected_grad)
            xp.add.at(dv, (scatter_batch, key_indices, scatter_kv), dv_selected_grad)

        if cache["weight_mode"] == "logit_bias":
            # One block logit prior is shared by all exact tokens in that block and
            # every query token governed by the route.  Sum those score gradients
            # before applying d(alpha*log(w+eps))/dw.
            dbias_heads = dscores.reshape(
                batch,
                n_routes,
                n_q_heads,
                routing_stride,
                n_selected,
                block_size,
            ).sum(axis=(3, 5))
            if n_router_heads == 1 and n_q_heads != 1:
                dbias_router = xp.sum(dbias_heads, axis=2, keepdims=True)
            else:
                dbias_router = dbias_heads
            weights_work = selected_weights.astype(dbias_router.dtype, copy=False)
            dweights = dbias_router * (
                cache["weight_scale"] / (weights_work + cache["weight_eps"])
            )
            dweights = dweights.astype(grad_dtype, copy=False)

    return dq, dk, dv, dweights
