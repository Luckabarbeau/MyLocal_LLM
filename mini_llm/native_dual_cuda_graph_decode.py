"""CUDA-graph decode for batch-one BF16 matrix-pool dual-resolution models."""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype, resolve_dtype
from mini_llm.ops.cublas_fp16_gemm import bf16_matmul_into, prepare_fp16_gemm_context
from mini_llm.native_cuda_graph_decode import _module as _base_module


_MODULE = None


_SOURCE = r"""
#include <cuda_bf16.h>

#define REDUCE_RMS(INPUT_EXPR) \
    float sum_sq = 0.0f; \
    for (int col = tid; col < d_model; col += (int)blockDim.x) { \
        const float value = (INPUT_EXPR); sum_sq += value * value; \
    } \
    for (int offset = 16; offset > 0; offset >>= 1) \
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
    __shared__ float warp_sums[8]; \
    const int lane_r = tid & 31; const int warp_r = tid >> 5; \
    if (lane_r == 0) warp_sums[warp_r] = sum_sq; __syncthreads(); \
    if (warp_r == 0) { \
        sum_sq = lane_r < 8 ? warp_sums[lane_r] : 0.0f; \
        for (int offset = 16; offset > 0; offset >>= 1) \
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
        if (lane_r == 0) warp_sums[0] = rsqrtf(sum_sq / (float)d_model + eps); \
    } \
    __syncthreads(); const float inv_rms = warp_sums[0]

extern "C" __global__ void dual_reset_if_pending(
    int* local_start, int* local_count, int* sparse_count,
    int* summary_start, int* summary_count, int* incomplete_count,
    long long* document_position, bool* reset_pending, int layers)
{
    if (!reset_pending[0]) return;
    for (int layer = threadIdx.x; layer < layers; layer += blockDim.x) {
        local_start[layer] = 0; local_count[layer] = 0; sparse_count[layer] = 0;
        summary_start[layer] = 0; summary_count[layer] = 0;
        incomplete_count[layer] = 0;
    }
    if (threadIdx.x == 0) { document_position[0] = 0; reset_pending[0] = false; }
}

extern "C" __global__ void dual_qkv_rope_store(
    const __nv_bfloat16* qkv, const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table, __nv_bfloat16* q_out,
    __nv_bfloat16* local_k, __nv_bfloat16* local_v,
    __nv_bfloat16* sparse_k, __nv_bfloat16* sparse_v,
    int* local_start, int* local_count, int* sparse_count,
    const long long* document_position, const long long* absolute_position,
    int n_q_heads, int n_kv_heads, int local_capacity, int sparse_capacity,
    int sparse_phase)
{
    const int tid = threadIdx.x;
    __shared__ int local_slot;
    __shared__ int sparse_slot;
    __shared__ int sparse_selected;
    if (tid == 0) {
        int start = local_start[0], count = local_count[0];
        if (count < local_capacity) { local_slot = (start + count) % local_capacity; ++count; }
        else { local_slot = start; start = (start + 1) % local_capacity; }
        local_start[0] = start; local_count[0] = count;
        sparse_selected = sparse_phase >= 0 &&
            ((int)(absolute_position[0] % 16LL) == sparse_phase);
        sparse_slot = sparse_count[0];
        if (sparse_selected && sparse_slot < sparse_capacity) sparse_count[0] = sparse_slot + 1;
    }
    __syncthreads();
    const long long position = document_position[0];
    const long long trig_base = position * 32LL;
    const int q_width = n_q_heads * 64;
    const int kv_width = n_kv_heads * 64;
    for (int pair = tid; pair < (q_width >> 1); pair += blockDim.x) {
        int col = pair << 1, ph = pair & 31;
        float c = __bfloat162float(cos_table[trig_base + ph]);
        float s = __bfloat162float(sin_table[trig_base + ph]);
        float a = __bfloat162float(qkv[col]);
        float b = __bfloat162float(qkv[col + 1]);
        q_out[col] = __float2bfloat16_rn(a*c-b*s);
        q_out[col+1] = __float2bfloat16_rn(a*s+b*c);
    }
    for (int pair = tid; pair < (kv_width >> 1); pair += blockDim.x) {
        int col = pair << 1, kvh = col >> 6, lane = col & 63, ph = pair & 31;
        float c = __bfloat162float(cos_table[trig_base + ph]);
        float s = __bfloat162float(sin_table[trig_base + ph]);
        long long src = q_width + col;
        float a = __bfloat162float(qkv[src]);
        float b = __bfloat162float(qkv[src + 1]);
        __nv_bfloat16 ka = __float2bfloat16_rn(a*c-b*s);
        __nv_bfloat16 kb = __float2bfloat16_rn(a*s+b*c);
        long long dst = ((long long)kvh * local_capacity + local_slot) * 64LL + lane;
        local_k[dst] = ka; local_k[dst + 1] = kb;
        if (sparse_selected && sparse_slot < sparse_capacity && kvh == 0) {
            long long sd = (long long)sparse_slot * 64LL + lane;
            sparse_k[sd] = ka; sparse_k[sd + 1] = kb;
        }
    }
    for (int col = tid; col < kv_width; col += blockDim.x) {
        int kvh = col >> 6, lane = col & 63;
        __nv_bfloat16 value = qkv[q_width + kv_width + col];
        local_v[((long long)kvh * local_capacity + local_slot) * 64LL + lane] = value;
        if (sparse_selected && sparse_slot < sparse_capacity && kvh == 0)
            sparse_v[(long long)sparse_slot * 64LL + lane] = value;
    }
}

extern "C" __global__ void dual_local_attention(
    const __nv_bfloat16* q, const __nv_bfloat16* local_k,
    const __nv_bfloat16* local_v, const __nv_bfloat16* sparse_k,
    const __nv_bfloat16* sparse_v, __nv_bfloat16* output,
    const int* local_start, const int* local_count, const int* sparse_count,
    int local_capacity, int n_q_heads, int group_size, int sparse_phase,
    float scale)
{
    const int lane = threadIdx.x, head = blockIdx.x;
    if (lane >= 32 || head >= n_q_heads) return;
    const bool sparse = sparse_phase >= 0 && head == 1;
    int count = sparse ? sparse_count[0] : local_count[0];
    int use = count;
    if (!sparse) {
        int window = head == 0 ? 4096 : 512;
        if (use > window) use = window;
    }
    if (use <= 0) {
        output[(long long)head * 64 + lane] = __float2bfloat16_rn(0.0f);
        output[(long long)head * 64 + lane + 32] = __float2bfloat16_rn(0.0f);
        return;
    }
    int kvh = head / group_size;
    float q0 = __bfloat162float(q[(long long)head * 64 + lane]);
    float q1 = __bfloat162float(q[(long long)head * 64 + lane + 32]);
    float acc0 = 0.0f, acc1 = 0.0f, running_max = -3.402823466e+38F, running_sum = 0.0f;
    const unsigned mask = 0xffffffffu;
    int first = count - use;
    for (int index = 0; index < use; ++index) {
        long long base;
        if (sparse) base = (long long)(first + index) * 64LL;
        else {
            int slot = (local_start[0] + first + index) % local_capacity;
            base = ((long long)kvh * local_capacity + slot) * 64LL;
        }
        const __nv_bfloat16* keys = sparse ? sparse_k : local_k;
        const __nv_bfloat16* values = sparse ? sparse_v : local_v;
        float dot = q0 * __bfloat162float(keys[base + lane]) +
                    q1 * __bfloat162float(keys[base + lane + 32]);
        dot += __shfl_down_sync(mask, dot, 16); dot += __shfl_down_sync(mask, dot, 8);
        dot += __shfl_down_sync(mask, dot, 4); dot += __shfl_down_sync(mask, dot, 2);
        dot += __shfl_down_sync(mask, dot, 1);
        float alpha = 0.0f, beta = 0.0f;
        if (lane == 0) {
            float score = dot * scale;
            float next_max = fmaxf(running_max, score);
            alpha = running_sum == 0.0f ? 0.0f : __expf(running_max - next_max);
            beta = __expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
        }
        alpha = __shfl_sync(mask, alpha, 0); beta = __shfl_sync(mask, beta, 0);
        acc0 = acc0 * alpha + beta * __bfloat162float(values[base + lane]);
        acc1 = acc1 * alpha + beta * __bfloat162float(values[base + lane + 32]);
    }
    float inv = lane == 0 ? 1.0f / running_sum : 0.0f;
    inv = __shfl_sync(mask, inv, 0);
    output[(long long)head * 64 + lane] = __float2bfloat16_rn(acc0 * inv);
    output[(long long)head * 64 + lane + 32] = __float2bfloat16_rn(acc1 * inv);
}

extern "C" __global__ void dual_local_state_store(
    const float* residual, const __nv_bfloat16* local_delta,
    __nv_bfloat16* local_state, __nv_bfloat16* incomplete,
    int* incomplete_count, int d_model)
{
    const int tid = threadIdx.x;
    int offset = incomplete_count[0];
    for (int col = tid; col < d_model; col += blockDim.x) {
        __nv_bfloat16 value = __float2bfloat16_rn(
            residual[col] + __bfloat162float(local_delta[col]));
        local_state[col] = value;
        incomplete[(long long)offset * d_model + col] = value;
    }
    __syncthreads();
    if (tid == 0) incomplete_count[0] = offset + 1;
}

extern "C" __global__ void dual_global_attention(
    const __nv_bfloat16* q_pre, const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table, const __nv_bfloat16* summary_k,
    const __nv_bfloat16* summary_v, __nv_bfloat16* output,
    const int* summary_start, const int* summary_count,
    const long long* document_position, int summary_capacity, float scale)
{
    const int lane = threadIdx.x, head = blockIdx.x;
    if (lane >= 32 || head >= 2) return;
    int count = summary_count[0];
    if (count <= 0) {
        output[(long long)head * 64 + lane] = __float2bfloat16_rn(0.0f);
        output[(long long)head * 64 + lane + 32] = __float2bfloat16_rn(0.0f);
        return;
    }
    long long trig = document_position[0] * 32LL + lane;
    float c = __bfloat162float(cos_table[trig]);
    float s = __bfloat162float(sin_table[trig]);
    long long qb = (long long)head * 64;
    float a = __bfloat162float(q_pre[qb + lane]);
    float b = __bfloat162float(q_pre[qb + lane + 32]);
    float q0 = a*c-b*s, q1 = a*s+b*c;
    float acc0 = 0.0f, acc1 = 0.0f, running_max = -3.402823466e+38F, running_sum = 0.0f;
    const unsigned mask = 0xffffffffu;
    for (int index = 0; index < count; ++index) {
        int slot = (summary_start[0] + index) % summary_capacity;
        long long base = ((long long)head * summary_capacity + slot) * 64LL;
        float dot = q0 * __bfloat162float(summary_k[base + lane]) +
                    q1 * __bfloat162float(summary_k[base + lane + 32]);
        dot += __shfl_down_sync(mask, dot, 16); dot += __shfl_down_sync(mask, dot, 8);
        dot += __shfl_down_sync(mask, dot, 4); dot += __shfl_down_sync(mask, dot, 2);
        dot += __shfl_down_sync(mask, dot, 1);
        float alpha = 0.0f, beta = 0.0f;
        if (lane == 0) {
            float score = dot * scale;
            float next_max = fmaxf(running_max, score);
            alpha = running_sum == 0.0f ? 0.0f : __expf(running_max - next_max);
            beta = __expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
        }
        alpha = __shfl_sync(mask, alpha, 0); beta = __shfl_sync(mask, beta, 0);
        acc0 = acc0 * alpha + beta * __bfloat162float(summary_v[base + lane]);
        acc1 = acc1 * alpha + beta * __bfloat162float(summary_v[base + lane + 32]);
    }
    float inv = lane == 0 ? 1.0f / running_sum : 0.0f;
    inv = __shfl_sync(mask, inv, 0);
    output[qb + lane] = __float2bfloat16_rn(acc0 * inv);
    output[qb + lane + 32] = __float2bfloat16_rn(acc1 * inv);
}

extern "C" __global__ void dual_residual_rmsnorm(
    const float* residual, const __nv_bfloat16* local_delta,
    const __nv_bfloat16* global_delta, const __nv_bfloat16* gamma,
    float* residual_out, __nv_bfloat16* normalized,
    int d_model, float alpha, float eps, int use_global)
{
    const int tid = threadIdx.x;
    #define BRANCH_VALUE(col) __bfloat162float(__float2bfloat16_rn( \
        __bfloat162float(local_delta[col]) + (use_global ? __bfloat162float(global_delta[col]) : 0.0f)))
    REDUCE_RMS(residual[col] + alpha * BRANCH_VALUE(col));
    for (int col = tid; col < d_model; col += blockDim.x) {
        float value = residual[col] + alpha * BRANCH_VALUE(col);
        residual_out[col] = value;
        normalized[col] = __float2bfloat16_rn(
            value * inv_rms * __bfloat162float(gamma[col]));
    }
    #undef BRANCH_VALUE
}

extern "C" __global__ void dual_pool_reduce(
    const __nv_bfloat16* scores, const __nv_bfloat16* position_bias,
    const __nv_bfloat16* incomplete, __nv_bfloat16* pooled,
    const int* incomplete_count, int d_model)
{
    const int summary = blockIdx.x, tid = threadIdx.x;
    if (incomplete_count[0] != 128) {
        for (int col = tid; col < d_model; col += blockDim.x)
            pooled[(long long)summary * d_model + col] = __float2bfloat16_rn(0.0f);
        return;
    }
    __shared__ float reduction[256];
    __shared__ float weights[128];
    float value = -3.402823466e+38F;
    if (tid < 128) value = __bfloat162float(scores[tid * 4 + summary]) +
                           __bfloat162float(position_bias[tid * 4 + summary]);
    reduction[tid] = value; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    float maximum = reduction[0];
    float exponential = tid < 128 ? __expf(value - maximum) : 0.0f;
    reduction[tid] = exponential; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    if (tid < 128) weights[tid] = exponential / reduction[0];
    __syncthreads();
    for (int col = tid; col < d_model; col += blockDim.x) {
        float sum = 0.0f;
        for (int token = 0; token < 128; ++token)
            sum += weights[token] * __bfloat162float(incomplete[(long long)token * d_model + col]);
        pooled[(long long)summary * d_model + col] = __float2bfloat16_rn(sum);
    }
}

extern "C" __global__ void dual_summary_store(
    const __nv_bfloat16* projected_k, const __nv_bfloat16* projected_v,
    const __nv_bfloat16* cos_table, const __nv_bfloat16* sin_table,
    __nv_bfloat16* summary_k, __nv_bfloat16* summary_v,
    int* summary_start, int* summary_count, int* incomplete_count,
    const long long* document_position, int summary_capacity)
{
    const int tid = threadIdx.x;
    if (incomplete_count[0] != 128) return;
    int start = summary_start[0], count = summary_count[0];
    for (int index = tid; index < 4 * 128; index += blockDim.x) {
        int summary = index / 128, component = index % 128;
        int head = component >> 6, lane = component & 63;
        int temp_start = start, temp_count = count, slot = 0;
        for (int i = 0; i <= summary; ++i) {
            if (temp_count < summary_capacity) {
                slot = (temp_start + temp_count) % summary_capacity; ++temp_count;
            } else { slot = temp_start; temp_start = (temp_start + 1) % summary_capacity; }
        }
        int pair_lane = lane & 31;
        int even = lane < 32 ? lane : lane - 32;
        long long src = (long long)summary * 128 + head * 64;
        float a = __bfloat162float(projected_k[src + even]);
        float b = __bfloat162float(projected_k[src + even + 32]);
        long long trig = document_position[0] * 32LL + pair_lane;
        float c = __bfloat162float(cos_table[trig]);
        float s = __bfloat162float(sin_table[trig]);
        float rotated = lane < 32 ? a*c-b*s : a*s+b*c;
        long long dst = ((long long)head * summary_capacity + slot) * 64LL + lane;
        summary_k[dst] = __float2bfloat16_rn(rotated);
        summary_v[dst] = projected_v[index];
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < 4; ++i) {
            if (count < summary_capacity) ++count;
            else start = (start + 1) % summary_capacity;
        }
        summary_start[0] = start; summary_count[0] = count; incomplete_count[0] = 0;
    }
}

extern "C" __global__ void dual_advance_document(
    long long* document_position, bool* reset_pending,
    const int* token, int eos_token_id)
{
    if (threadIdx.x == 0) {
        document_position[0] += 1;
        if (eos_token_id >= 0 && token[0] == eos_token_id) reset_pending[0] = true;
    }
}
"""


def _module():
    global _MODULE
    if _MODULE is None:
        names = (
            "dual_reset_if_pending", "dual_qkv_rope_store",
            "dual_local_attention", "dual_local_state_store",
            "dual_global_attention", "dual_residual_rmsnorm",
            "dual_pool_reduce", "dual_summary_store",
            "dual_advance_document",
        )
        _MODULE = xp.RawModule(
            code=_SOURCE, options=("--std=c++14",), name_expressions=names,
        )
        for name in names:
            _MODULE.get_function(name)
    return _MODULE


class NativeDualBF16CudaGraphDecoder:
    """Captured exact-architecture decoder for Model B matrix-pool attention."""

    def __init__(self, engine, state):
        if BACKEND_NAME != "cupy" or not is_bfloat16_dtype(resolve_dtype(engine.dtype)):
            raise ValueError("dual CUDA Graph decode requires CuPy BF16")
        if state.batch_size != 1 or state.long_memory_enabled or not state.has_capacity(1):
            raise ValueError("dual CUDA Graph decode requires batch-one bounded state")
        if not engine.dual_resolution_enabled or not state.dual_resolution_enabled:
            raise ValueError("dual CUDA Graph decode requires dual-resolution state")
        cfg = engine.dual_resolution_config
        expected = (
            getattr(cfg, "compression_mode", None) == "matrix_pool"
            and int(cfg.block_size) == 128 and int(cfg.local_window) == 512
            and int(cfg.long_local_window) == 4096 and int(cfg.long_local_heads) == 1
            and int(cfg.summaries_per_block) == 4 and int(cfg.global_heads) == 2
            and int(cfg.global_head_dim) == 64 and bool(cfg.global_enabled)
            and int(getattr(cfg, "sparse_retrieval_stride", 0)) == 16
            and int(getattr(cfg, "sparse_retrieval_head", -1)) == 1
            and engine.d_model == 1024 and engine.n_kv_heads == 4
            and engine.head_dim == 64 and len(engine.blocks) == 20
        )
        if not expected:
            raise ValueError("dual CUDA Graph decode supports the Model B geometry only")
        if state.eos_token_id is not None and not 0 <= int(state.eos_token_id) < engine.vocab_size:
            raise ValueError("invalid EOS token for dual CUDA Graph decode")
        self.engine, self.state, self.module = engine, state, _module()
        self.base = _base_module()
        self.d_model = int(engine.d_model)
        self.d_ff = int(engine.blocks[0].d_ff)
        self.n_q = int(engine.blocks[0].n_q_heads)
        self.n_kv = int(engine.n_kv_heads)
        self.local_capacity = int(state.k_cache.shape[3])
        self.summary_capacity = int(state.dual_summary_k_cache.shape[3])
        self.sparse_capacity = int(state.dual_sparse_k_cache.shape[3])
        self.bf = resolve_dtype(engine.dtype)
        self.stream = xp.cuda.Stream(non_blocking=True)
        self.ctx = prepare_fp16_gemm_context(self.stream)
        self.token = xp.zeros((1,), dtype=xp.int32)
        self.absolute_position = xp.asarray([state.length], dtype=xp.int64)
        self.position_host = np.asarray([state.length], dtype=np.int64)
        self.ra = xp.empty((1, self.d_model), dtype=xp.float32)
        self.rb = xp.empty_like(self.ra)
        self.norm = xp.empty((1, self.d_model), dtype=self.bf)
        self.qkv = xp.empty((1, self.d_model + 2 * self.n_kv * 64), dtype=self.bf)
        self.q = xp.empty((1, self.d_model), dtype=self.bf)
        self.context = xp.empty_like(self.q)
        self.local_delta = xp.empty_like(self.q)
        self.local_state = xp.empty_like(self.q)
        self.global_q = xp.empty((1, 128), dtype=self.bf)
        self.global_context = xp.empty_like(self.global_q)
        self.global_delta = xp.empty((1, self.d_model), dtype=self.bf)
        self.gate = xp.empty((1, self.d_ff), dtype=self.bf)
        self.up = xp.empty_like(self.gate)
        self.hidden = xp.empty_like(self.gate)
        self.pool_scores = xp.empty((128, 4), dtype=self.bf)
        self.pooled = xp.empty((4, self.d_model), dtype=self.bf)
        self.pool_k = xp.empty((4, 128), dtype=self.bf)
        self.pool_v = xp.empty_like(self.pool_k)
        self.logits = xp.empty((1, engine.vocab_size), dtype=self.bf)
        self.embedding = engine.embedding.W.data
        self.lm = engine.lm_head
        sparse_layers = tuple(int(x) for x in cfg.sparse_retrieval_layers)
        phases = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
        phase_by_layer = dict(zip(sparse_layers, phases))
        self.layers = []
        for index, block in enumerate(engine.blocks):
            attention = block.attention
            expert = block.moe.expert
            if expert.W_gate_up is not None:
                raise ValueError("dual BF16 graph requires separate gate/up weights")
            required = (
                attention.Wqkv, attention.Wo, attention.W_pool_score,
                attention.pool_position, attention.W_global_q, attention.W_global_k,
                attention.W_global_v, attention.W_global_o,
            )
            if any(value is None for value in required):
                raise ValueError("dual CUDA Graph decode weights are incomplete")
            attention.ensure_rope_capacity(state.max_length)
            self.layers.append(dict(
                norm1=block.norm1.gamma, norm2=block.norm2.gamma,
                wqkv=attention.Wqkv, wo=attention.Wo,
                pool_score=attention.W_pool_score, pool_position=attention.pool_position,
                global_q=attention.W_global_q, global_k=attention.W_global_k,
                global_v=attention.W_global_v, global_o=attention.W_global_o,
                gate=expert.W_gate, up=expert.W_up, down=expert.W_down,
                cos=attention._rope_cos, sin=attention._rope_sin,
                local_k=state.k_cache[index, 0], local_v=state.v_cache[index, 0],
                sparse_k=state.dual_sparse_k_cache[index, 0, 0],
                sparse_v=state.dual_sparse_v_cache[index, 0, 0],
                summary_k=state.dual_summary_k_cache[index, 0],
                summary_v=state.dual_summary_v_cache[index, 0],
                incomplete=state.dual_incomplete_hidden[index, 0],
                local_start=state.dual_local_start[index],
                local_count=state.dual_local_count[index],
                sparse_count=state.dual_sparse_count[index],
                summary_start=state.dual_summary_start[index],
                summary_count=state.dual_summary_count[index],
                incomplete_count=state.dual_incomplete_count[index],
                phase=int(phase_by_layer.get(index, -1)),
                eps=float(block.norm1.eps), aa=float(block.alpha_attn),
                am=float(block.alpha_mlp),
            ))
        self.final_gamma = engine.final_norm.gamma
        self._warm_gemms()
        self.graphs = {}
        for use_global in (False, True):
            for boundary in (False, True):
                self.stream.begin_capture()
                with self.stream:
                    self._enqueue(use_global=use_global, boundary=boundary)
                graph = self.stream.end_capture()
                graph.upload(stream=self.stream)
                self.graphs[(use_global, boundary)] = graph
        self.stream.synchronize()
        self.document_position = int(state.dual_document_positions[0].item())
        self.reset_pending = bool(state.dual_reset_pending[0].item())
        self.last_length = int(state.length)
        state.dual_host_counters = None

    def _warm_gemms(self):
        layer = self.layers[0]
        with self.stream:
            bf16_matmul_into(self.norm, layer["wqkv"], self.qkv, context=self.ctx)
            bf16_matmul_into(self.context, layer["wo"], self.local_delta, context=self.ctx)
            bf16_matmul_into(self.local_state, layer["global_q"], self.global_q, context=self.ctx)
            bf16_matmul_into(self.global_context, layer["global_o"], self.global_delta, context=self.ctx)
            bf16_matmul_into(self.norm, layer["gate"], self.gate, context=self.ctx)
            bf16_matmul_into(self.norm, layer["up"], self.up, context=self.ctx)
            bf16_matmul_into(self.hidden, layer["down"], self.q, context=self.ctx)
            bf16_matmul_into(layer["incomplete"], layer["pool_score"], self.pool_scores, context=self.ctx)
            bf16_matmul_into(self.pooled, layer["global_k"], self.pool_k, context=self.ctx)
            bf16_matmul_into(self.pooled, layer["global_v"], self.pool_v, context=self.ctx)
            bf16_matmul_into(self.norm, self.lm, self.logits, context=self.ctx)
        self.stream.synchronize()

    def _enqueue(self, *, use_global, boundary):
        module, base, threads = self.module, self.base, 256
        module.get_function("dual_reset_if_pending")(
            (1,), (threads,), (
                self.state.dual_local_start, self.state.dual_local_count,
                self.state.dual_sparse_count, self.state.dual_summary_start,
                self.state.dual_summary_count, self.state.dual_incomplete_count,
                self.state.dual_document_positions, self.state.dual_reset_pending,
                np.int32(len(self.layers)),
            ),
        )
        base.get_function("embedding_rmsnorm_bf16")(
            (1,), (threads,), (
                self.token, self.embedding, self.layers[0]["norm1"],
                self.ra, self.norm, np.int32(self.engine.vocab_size),
                np.int32(self.d_model), np.float32(self.layers[0]["eps"]),
            ),
        )
        for index, layer in enumerate(self.layers):
            bf16_matmul_into(self.norm, layer["wqkv"], self.qkv, context=self.ctx)
            module.get_function("dual_qkv_rope_store")(
                (1,), (threads,), (
                    self.qkv, layer["cos"], layer["sin"], self.q,
                    layer["local_k"], layer["local_v"],
                    layer["sparse_k"], layer["sparse_v"],
                    layer["local_start"], layer["local_count"], layer["sparse_count"],
                    self.state.dual_document_positions, self.absolute_position,
                    np.int32(self.n_q), np.int32(self.n_kv),
                    np.int32(self.local_capacity), np.int32(self.sparse_capacity),
                    np.int32(layer["phase"]),
                ),
            )
            module.get_function("dual_local_attention")(
                (self.n_q,), (32,), (
                    self.q, layer["local_k"], layer["local_v"],
                    layer["sparse_k"], layer["sparse_v"], self.context,
                    layer["local_start"], layer["local_count"], layer["sparse_count"],
                    np.int32(self.local_capacity), np.int32(self.n_q), np.int32(4),
                    np.int32(layer["phase"]), np.float32(0.125),
                ),
            )
            bf16_matmul_into(self.context, layer["wo"], self.local_delta, context=self.ctx)
            module.get_function("dual_local_state_store")(
                (1,), (threads,), (
                    self.ra, self.local_delta, self.local_state, layer["incomplete"],
                    layer["incomplete_count"], np.int32(self.d_model),
                ),
            )
            if use_global:
                bf16_matmul_into(self.local_state, layer["global_q"], self.global_q, context=self.ctx)
                module.get_function("dual_global_attention")(
                    (2,), (32,), (
                        self.global_q, layer["cos"], layer["sin"],
                        layer["summary_k"], layer["summary_v"], self.global_context,
                        layer["summary_start"], layer["summary_count"],
                        self.state.dual_document_positions, np.int32(self.summary_capacity),
                        np.float32(0.125),
                    ),
                )
                bf16_matmul_into(
                    self.global_context, layer["global_o"], self.global_delta,
                    context=self.ctx,
                )
            module.get_function("dual_residual_rmsnorm")(
                (1,), (threads,), (
                    self.ra, self.local_delta, self.global_delta, layer["norm2"],
                    self.rb, self.norm, np.int32(self.d_model),
                    np.float32(layer["aa"]), np.float32(layer["eps"]),
                    np.int32(1 if use_global else 0),
                ),
            )
            bf16_matmul_into(self.norm, layer["gate"], self.gate, context=self.ctx)
            bf16_matmul_into(self.norm, layer["up"], self.up, context=self.ctx)
            base.get_function("swiglu_bf16")(
                ((self.d_ff + threads - 1) // threads,), (threads,), (
                    self.gate, self.up, self.hidden, np.int32(self.d_ff),
                ),
            )
            bf16_matmul_into(self.hidden, layer["down"], self.q, context=self.ctx)
            next_gamma = self.layers[index + 1]["norm1"] if index + 1 < len(self.layers) else self.final_gamma
            base.get_function("residual_rmsnorm_bf16")(
                (1,), (threads,), (
                    self.rb, self.q, next_gamma, self.ra, self.norm,
                    np.int32(self.d_model), np.float32(layer["am"]),
                    np.float32(layer["eps"]),
                ),
            )
            if boundary:
                bf16_matmul_into(
                    layer["incomplete"], layer["pool_score"], self.pool_scores,
                    context=self.ctx,
                )
                module.get_function("dual_pool_reduce")(
                    (4,), (threads,), (
                        self.pool_scores, layer["pool_position"], layer["incomplete"],
                        self.pooled, layer["incomplete_count"], np.int32(self.d_model),
                    ),
                )
                bf16_matmul_into(self.pooled, layer["global_k"], self.pool_k, context=self.ctx)
                bf16_matmul_into(self.pooled, layer["global_v"], self.pool_v, context=self.ctx)
                module.get_function("dual_summary_store")(
                    (1,), (threads,), (
                        self.pool_k, self.pool_v, layer["cos"], layer["sin"],
                        layer["summary_k"], layer["summary_v"],
                        layer["summary_start"], layer["summary_count"],
                        layer["incomplete_count"], self.state.dual_document_positions,
                        np.int32(self.summary_capacity),
                    ),
                )
        bf16_matmul_into(self.norm, self.lm, self.logits, context=self.ctx)
        module.get_function("dual_advance_document")(
            (1,), (1,), (
                self.state.dual_document_positions, self.state.dual_reset_pending,
                self.token, np.int32(-1 if self.state.eos_token_id is None else self.state.eos_token_id),
            ),
        )

    def decode_one(self, next_ids, state):
        if state is not self.state or not state.has_capacity(1):
            raise ValueError("dual CUDA Graph state mismatch or exhausted cache")
        ids = xp.asarray(next_ids, dtype=xp.int32).reshape(-1)
        if ids.size != 1:
            raise ValueError("dual CUDA Graph decode requires one token")
        current = xp.cuda.get_current_stream()
        if int(state.length) != self.last_length:
            current.synchronize()
            self.document_position = int(state.dual_document_positions[0].item())
            self.reset_pending = bool(state.dual_reset_pending[0].item())
        if self.reset_pending:
            self.document_position = 0
            self.reset_pending = False
        use_global = self.document_position >= 128
        boundary = (self.document_position + 1) % 128 == 0
        xp.cuda.runtime.memcpyAsync(
            int(self.token.data.ptr), int(ids.data.ptr), 4,
            xp.cuda.runtime.memcpyDeviceToDevice, int(current.ptr),
        )
        self.position_host[0] = state.length
        xp.cuda.runtime.memcpyAsync(
            int(self.absolute_position.data.ptr), int(self.position_host.ctypes.data), 8,
            xp.cuda.runtime.memcpyHostToDevice, int(current.ptr),
        )
        self.graphs[(use_global, boundary)].launch(stream=current)
        state.length += 1
        self.last_length = int(state.length)
        self.document_position += 1
        if state.eos_token_id is not None:
            self.reset_pending = int(ids[0].item()) == int(state.eos_token_id)
        self.engine.last_hidden = self.norm
        return self.logits
