"""Reusable CUDA Graph executor for batch-one external FP16 Llama decode.

This specialization intentionally leaves generic/native inference unchanged. It
owns every intermediate referenced by the captured graph and uses direct
cuBLAS dispatch because CuPy's high-level matmul wrapper rejects stream capture.
"""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.cublas_fp16_gemm import (
    fp16_matmul_into,
    prepare_fp16_gemm_context,
)


_MODULE = None


def _truthy(name, default):
    return os.environ.get(name, default).strip().lower() not in {
        "", "0", "false", "off", "no",
    }


def enabled():
    # Opt-in until the captured path matches the established fixed-token
    # numerical trajectory across a complete generation.
    return BACKEND_NAME == "cupy" and _truthy("MINI_LLM_CUDA_GRAPH_DECODE", "0")


def strict():
    return _truthy("MINI_LLM_CUDA_GRAPH_DECODE_STRICT", "0")


_SOURCE = r"""
#include <cuda_fp16.h>

#define REDUCE_RMS(INPUT_EXPR) \
    float sum_sq = 0.0f; \
    for (int col = tid; col < d_model; col += (int)blockDim.x) { \
        const float value = (INPUT_EXPR); \
        sum_sq += value * value; \
    } \
    for (int offset = 16; offset > 0; offset >>= 1) \
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
    __shared__ float warp_sums[8]; \
    const int lane_reduce = tid & 31; \
    const int warp_reduce = tid >> 5; \
    if (lane_reduce == 0) warp_sums[warp_reduce] = sum_sq; \
    __syncthreads(); \
    if (warp_reduce == 0) { \
        sum_sq = lane_reduce < 8 ? warp_sums[lane_reduce] : 0.0f; \
        for (int offset = 16; offset > 0; offset >>= 1) \
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
        if (lane_reduce == 0) warp_sums[0] = rsqrtf(sum_sq / (float)d_model + eps); \
    } \
    __syncthreads(); \
    const float inv_rms = warp_sums[0]

extern "C" __global__
void embedding_rmsnorm_fp16(
    const int* __restrict__ token,
    const half* __restrict__ embedding,
    const half* __restrict__ gamma,
    float* __restrict__ residual,
    half* __restrict__ normalized,
    int vocab_size,
    int d_model,
    float eps)
{
    const int tid = (int)threadIdx.x;
    int row = token[0];
    if (row < 0 || row >= vocab_size) row = 0;
    const long long base = (long long)row * d_model;
    REDUCE_RMS(__half2float(embedding[base + col]));
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const float value = __half2float(embedding[base + col]);
        residual[col] = value;
        normalized[col] = __float2half_rn(
            value * inv_rms * __half2float(gamma[col]));
    }
}

extern "C" __global__
void qkv_rope_store_fp16_d64_dynamic(
    const half* __restrict__ packed,
    const half* __restrict__ cos_table,
    const half* __restrict__ sin_table,
    half* __restrict__ q_out,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const int* __restrict__ position_ptr,
    int n_q_heads,
    int n_kv_heads,
    int cache_capacity)
{
    const int tid = (int)threadIdx.x;
    const int position = position_ptr[0];
    const int q_width = n_q_heads * 64;
    const int kv_width = n_kv_heads * 64;
    const long long trig_base = (long long)position * 32LL;

    for (int pair = tid; pair < n_q_heads * 32; pair += (int)blockDim.x) {
        const int head = pair >> 5;
        const int lane = pair & 31;
        const int src = head * 64 + lane;
        const float c = __half2float(cos_table[trig_base + lane]);
        const float s = __half2float(sin_table[trig_base + lane]);
        const float x0 = __half2float(packed[src]);
        const float x1 = __half2float(packed[src + 32]);
        q_out[src] = __float2half_rn(x0 * c - x1 * s);
        q_out[src + 32] = __float2half_rn(x0 * s + x1 * c);
    }
    for (int pair = tid; pair < n_kv_heads * 32; pair += (int)blockDim.x) {
        const int head = pair >> 5;
        const int lane = pair & 31;
        const int src = head * 64 + lane;
        const float c = __half2float(cos_table[trig_base + lane]);
        const float s = __half2float(sin_table[trig_base + lane]);
        const float x0 = __half2float(packed[q_width + src]);
        const float x1 = __half2float(packed[q_width + src + 32]);
        const long long dst =
            ((long long)head * cache_capacity + position) * 64LL + lane;
        k_cache[dst] = __float2half_rn(x0 * c - x1 * s);
        k_cache[dst + 32] = __float2half_rn(x0 * s + x1 * c);
    }
    for (int col = tid; col < kv_width; col += (int)blockDim.x) {
        const int head = col >> 6;
        const int lane = col & 63;
        const long long dst =
            ((long long)head * cache_capacity + position) * 64LL + lane;
        v_cache[dst] = packed[q_width + kv_width + col];
    }
}

extern "C" __global__
void dense_attention_fp16_d64_dynamic(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half* __restrict__ out,
    const int* __restrict__ position_ptr,
    int n_q_heads,
    int n_kv_heads,
    int cache_capacity,
    float scale)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int h = (int)blockIdx.x;
    if (h >= n_q_heads) return;
    const int position = position_ptr[0];
    const int kvh = h / (n_q_heads / n_kv_heads);
    const long long qbase = (long long)h * 64LL;
    const float q0 = __half2float(q[qbase + lane]);
    const float q1 = __half2float(q[qbase + lane + 32]);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    const unsigned mask = 0xffffffffu;

    for (int key = warp; key <= position; key += 4) {
        const long long base =
            ((long long)kvh * cache_capacity + key) * 64LL;
        float dot = q0 * __half2float(k_cache[base + lane])
                  + q1 * __half2float(k_cache[base + lane + 32]);
        dot += __shfl_down_sync(mask, dot, 16);
        dot += __shfl_down_sync(mask, dot, 8);
        dot += __shfl_down_sync(mask, dot, 4);
        dot += __shfl_down_sync(mask, dot, 2);
        dot += __shfl_down_sync(mask, dot, 1);
        const float score = __shfl_sync(mask, dot, 0) * scale;
        const float new_max = fmaxf(running_max, score);
        const float alpha = running_sum == 0.0f
            ? 0.0f : __expf(running_max - new_max);
        const float beta = __expf(score - new_max);
        running_sum = running_sum * alpha + beta;
        running_max = new_max;
        acc0 = acc0 * alpha + beta * __half2float(v_cache[base + lane]);
        acc1 = acc1 * alpha + beta * __half2float(v_cache[base + lane + 32]);
    }

    __shared__ float warp_max[4];
    __shared__ float warp_sum[4];
    __shared__ float warp_acc[4][64];
    __shared__ float block_max;
    __shared__ float block_sum;
    if (lane == 0) {
        warp_max[warp] = running_max;
        warp_sum[warp] = running_sum;
    }
    warp_acc[warp][lane] = acc0;
    warp_acc[warp][lane + 32] = acc1;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float merged_max = -3.402823466e+38F;
        for (int w = 0; w < 4; ++w) merged_max = fmaxf(merged_max, warp_max[w]);
        float merged_sum = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (warp_sum[w] > 0.0f)
                merged_sum += warp_sum[w] * __expf(warp_max[w] - merged_max);
        }
        block_max = merged_max;
        block_sum = merged_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float merged0 = 0.0f;
        float merged1 = 0.0f;
        for (int w = 0; w < 4; ++w) {
            const float factor = warp_sum[w] > 0.0f
                ? __expf(warp_max[w] - block_max) : 0.0f;
            merged0 += warp_acc[w][lane] * factor;
            merged1 += warp_acc[w][lane + 32] * factor;
        }
        const float inv = 1.0f / block_sum;
        out[qbase + lane] = __float2half_rn(merged0 * inv);
        out[qbase + lane + 32] = __float2half_rn(merged1 * inv);
    }
}

extern "C" __global__
void residual_add_rmsnorm_fp16(
    const float* __restrict__ residual,
    const half* __restrict__ branch,
    const half* __restrict__ gamma,
    float* __restrict__ residual_out,
    half* __restrict__ normalized,
    int d_model,
    float eps)
{
    const int tid = (int)threadIdx.x;
    REDUCE_RMS(residual[col] + __half2float(branch[col]));
    for (int col = tid; col < d_model; col += (int)blockDim.x) {
        const float value = residual[col] + __half2float(branch[col]);
        residual_out[col] = value;
        normalized[col] = __float2half_rn(
            value * inv_rms * __half2float(gamma[col]));
    }
}

extern "C" __global__
void swiglu_packed_fp16(
    const half* __restrict__ gate_up,
    half* __restrict__ hidden,
    int d_ff)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= d_ff) return;
    const float g = __half2float(gate_up[idx]);
    const float u = __half2float(gate_up[d_ff + idx]);
    const float silu = g / (1.0f + expf(-g));
    hidden[idx] = __float2half_rn(silu * u);
}
"""


def _module():
    global _MODULE
    if _MODULE is None:
        names = (
            "embedding_rmsnorm_fp16",
            "qkv_rope_store_fp16_d64_dynamic",
            "dense_attention_fp16_d64_dynamic",
            "residual_add_rmsnorm_fp16",
            "swiglu_packed_fp16",
        )
        module = xp.RawModule(
            code=_SOURCE,
            options=("--std=c++14",),
            name_expressions=names,
        )
        for name in names:
            module.get_function(name)
        _MODULE = module
    return _MODULE


def _c_fp16(array):
    if array.dtype != xp.float16:
        raise TypeError("CUDA Graph external decode requires FP16 weights")
    if array.flags.c_contiguous or array.flags.f_contiguous:
        return array
    return xp.ascontiguousarray(array)


class ExternalFP16CudaGraphDecoder:
    """One captured decode graph bound to one external generation state."""

    def __init__(self, engine, state):
        if BACKEND_NAME != "cupy":
            raise RuntimeError("CUDA Graph decode requires CuPy")
        if str(engine.dtype).lower() not in {"float16", "fp16"}:
            raise ValueError("CUDA Graph decode currently supports FP16 only")
        if state.batch_size != 1 or state.long_memory_enabled:
            raise ValueError("CUDA Graph decode requires batch-one linear KV state")
        if engine.sparse_attention or not engine.blocks:
            raise ValueError("CUDA Graph decode requires dense attention blocks")

        self.engine = engine
        self.state = state
        self.d_model = int(engine.d_model)
        self.vocab_size = int(engine.vocab_size)
        self.n_layers = len(engine.blocks)
        first = engine.blocks[0]
        self.n_q_heads = int(first.n_q_heads)
        self.n_kv_heads = int(first.n_kv_heads)
        self.d_head = int(first.d_head)
        self.d_ff = int(first.d_ff)
        self.cache_capacity = int(state.k_cache.shape[3])
        if self.d_head != 64:
            raise ValueError("CUDA Graph decode requires head_dim=64")
        if any(block.attention.rope_layout != "half" for block in engine.blocks):
            raise ValueError("CUDA Graph decode requires Llama half-layout RoPE")
        if any(not block.use_fp32_residual for block in engine.blocks):
            raise ValueError("CUDA Graph decode requires low-precision residual policy")

        self.capture_stream = xp.cuda.Stream(non_blocking=True)
        self.gemm_context = prepare_fp16_gemm_context(self.capture_stream)
        self.module = _module()
        self.token = xp.zeros((1,), dtype=xp.int32)
        self.position = xp.asarray([int(state.length)], dtype=xp.int32)
        self._position_host = np.asarray([int(state.length)], dtype=np.int32)
        self.residual_a = xp.empty((1, self.d_model), dtype=xp.float32)
        self.residual_b = xp.empty_like(self.residual_a)
        self.normalized = xp.empty((1, self.d_model), dtype=xp.float16)
        qkv_width = self.d_model + 2 * self.n_kv_heads * self.d_head
        self.qkv = xp.empty((1, qkv_width), dtype=xp.float16)
        self.q = xp.empty((1, self.n_q_heads * self.d_head), dtype=xp.float16)
        self.context = xp.empty_like(self.q)
        self.gate_up = xp.empty((1, 2 * self.d_ff), dtype=xp.float16)
        self.hidden = xp.empty((1, self.d_ff), dtype=xp.float16)
        self.logits = xp.empty((1, self.vocab_size), dtype=xp.float16)

        self.embedding = _c_fp16(engine.embedding.W.data)
        self.lm_head = _c_fp16(engine.lm_head)
        self.layers = []
        for index, block in enumerate(engine.blocks):
            attention = block.attention
            expert = block.moe.expert
            if attention.Wqkv is None or expert.W_gate_up is None:
                raise ValueError("CUDA Graph decode requires packed QKV and gate/up weights")
            attention.ensure_rope_capacity(self.cache_capacity)
            self.layers.append({
                "norm1": _c_fp16(block.norm1.gamma),
                "norm2": _c_fp16(block.norm2.gamma),
                "wqkv": _c_fp16(attention.Wqkv),
                "wo": _c_fp16(attention.Wo),
                "gate_up": _c_fp16(expert.W_gate_up),
                "down": _c_fp16(expert.W_down),
                "cos": _c_fp16(attention._rope_cos),
                "sin": _c_fp16(attention._rope_sin),
                "k_cache": state.k_cache[index],
                "v_cache": state.v_cache[index],
                "norm_eps": float(block.norm1.eps),
                "scale": float(attention.scale),
            })
        self.final_gamma = _c_fp16(engine.final_norm.gamma)

        # Compile and initialize every library/kernel outside capture. The warm
        # pass writes only the next uncommitted KV slot, which the first replay
        # overwrites with the real token.
        with self.capture_stream:
            self._enqueue()
        self.capture_stream.synchronize()
        self.capture_stream.begin_capture()
        with self.capture_stream:
            self._enqueue()
        self.graph = self.capture_stream.end_capture()
        self.graph.upload(stream=self.capture_stream)
        self.capture_stream.synchronize()

    def _enqueue(self):
        threads = 256
        self.module.get_function("embedding_rmsnorm_fp16")(
            (1,), (threads,), (
                self.token, self.embedding, self.layers[0]["norm1"],
                self.residual_a, self.normalized,
                np.int32(self.vocab_size), np.int32(self.d_model),
                np.float32(self.layers[0]["norm_eps"]),
            ),
        )
        for index, layer in enumerate(self.layers):
            fp16_matmul_into(
                self.normalized, layer["wqkv"], self.qkv,
                context=self.gemm_context,
            )
            self.module.get_function("qkv_rope_store_fp16_d64_dynamic")(
                (1,), (threads,), (
                    self.qkv, layer["cos"], layer["sin"], self.q,
                    layer["k_cache"], layer["v_cache"], self.position,
                    np.int32(self.n_q_heads), np.int32(self.n_kv_heads),
                    np.int32(self.cache_capacity),
                ),
            )
            self.module.get_function("dense_attention_fp16_d64_dynamic")(
                (self.n_q_heads,), (128,), (
                    self.q, layer["k_cache"], layer["v_cache"], self.context,
                    self.position, np.int32(self.n_q_heads),
                    np.int32(self.n_kv_heads), np.int32(self.cache_capacity),
                    np.float32(layer["scale"]),
                ),
            )
            fp16_matmul_into(
                self.context, layer["wo"], self.q,
                context=self.gemm_context,
            )
            self.module.get_function("residual_add_rmsnorm_fp16")(
                (1,), (threads,), (
                    self.residual_a, self.q, layer["norm2"],
                    self.residual_b, self.normalized,
                    np.int32(self.d_model), np.float32(layer["norm_eps"]),
                ),
            )
            fp16_matmul_into(
                self.normalized, layer["gate_up"], self.gate_up,
                context=self.gemm_context,
            )
            blocks = max(1, (self.d_ff + threads - 1) // threads)
            self.module.get_function("swiglu_packed_fp16")(
                (blocks,), (threads,), (
                    self.gate_up, self.hidden, np.int32(self.d_ff),
                ),
            )
            fp16_matmul_into(
                self.hidden, layer["down"], self.q,
                context=self.gemm_context,
            )
            next_gamma = (
                self.layers[index + 1]["norm1"]
                if index + 1 < self.n_layers else self.final_gamma
            )
            self.module.get_function("residual_add_rmsnorm_fp16")(
                (1,), (threads,), (
                    self.residual_b, self.q, next_gamma,
                    self.residual_a, self.normalized,
                    np.int32(self.d_model), np.float32(layer["norm_eps"]),
                ),
            )
        fp16_matmul_into(
            self.normalized, self.lm_head, self.logits,
            context=self.gemm_context,
        )

    def decode_one(self, next_ids, state):
        if state is not self.state:
            raise ValueError("CUDA Graph decoder is bound to a different generation state")
        if not state.has_capacity(1):
            raise ValueError("No capacity remaining in generation cache.")
        if not isinstance(next_ids, xp.ndarray):
            next_ids = xp.asarray(next_ids, dtype=xp.int32)
        else:
            next_ids = next_ids.astype(xp.int32, copy=False)
        if next_ids.size != 1:
            raise ValueError("CUDA Graph decode currently requires one token")
        current = xp.cuda.get_current_stream()
        xp.cuda.runtime.memcpyAsync(
            int(self.token.data.ptr), int(next_ids.data.ptr), 4,
            xp.cuda.runtime.memcpyDeviceToDevice, int(current.ptr),
        )
        self._position_host[0] = int(state.length)
        xp.cuda.runtime.memcpyAsync(
            int(self.position.data.ptr), int(self._position_host.ctypes.data), 4,
            xp.cuda.runtime.memcpyHostToDevice, int(current.ptr),
        )
        self.graph.launch(stream=current)
        state.length += 1
        self.engine.last_hidden = self.normalized
        return self.logits
