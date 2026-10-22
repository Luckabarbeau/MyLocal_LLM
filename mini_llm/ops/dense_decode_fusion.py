"""Fused FP16 kernels for Llama-style one-token dense GQA decode."""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


_MODULE = None
_DISABLED = False
_FAILURE = None


def _enabled() -> bool:
    raw = os.environ.get("MINI_LLM_FUSED_DENSE_DECODE", "1").strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no", ""}


def _strict() -> bool:
    raw = os.environ.get("MINI_LLM_FUSED_DENSE_DECODE_STRICT", "0").strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def failure_reason():
    return _FAILURE


def _fail(message):
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = str(message)


_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" __global__
void unpack_half_rope_store_fp16_d64(
    const half* __restrict__ packed,
    const half* __restrict__ cos_table,
    const half* __restrict__ sin_table,
    half* __restrict__ q_out,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int batch,
    int n_q_heads,
    int n_kv_heads,
    int cache_capacity,
    int cache_position,
    int rope_position)
{
    const int b = (int)blockIdx.x;
    if (b >= batch) return;
    const int tid = (int)threadIdx.x;
    const int q_width = n_q_heads * 64;
    const int kv_width = n_kv_heads * 64;
    const int packed_width = q_width + 2 * kv_width;
    const long long packed_base = (long long)b * packed_width;
    const long long q_base = (long long)b * q_width;
    const long long trig_base = (long long)rope_position * 32LL;

    for (int pair = tid; pair < n_q_heads * 32; pair += (int)blockDim.x) {
        const int head = pair >> 5;
        const int lane = pair & 31;
        const int src = head * 64 + lane;
        const float c = __half2float(cos_table[trig_base + lane]);
        const float s = __half2float(sin_table[trig_base + lane]);
        const float x0 = __half2float(packed[packed_base + src]);
        const float x1 = __half2float(packed[packed_base + src + 32]);
        q_out[q_base + src] = __float2half_rn(x0 * c - x1 * s);
        q_out[q_base + src + 32] = __float2half_rn(x0 * s + x1 * c);
    }

    for (int pair = tid; pair < n_kv_heads * 32; pair += (int)blockDim.x) {
        const int head = pair >> 5;
        const int lane = pair & 31;
        const int src_local = head * 64 + lane;
        const float c = __half2float(cos_table[trig_base + lane]);
        const float s = __half2float(sin_table[trig_base + lane]);
        const float x0 = __half2float(packed[packed_base + q_width + src_local]);
        const float x1 = __half2float(packed[packed_base + q_width + src_local + 32]);
        const long long dst =
            (((long long)b * n_kv_heads + head) * cache_capacity + cache_position) * 64LL + lane;
        k_cache[dst] = __float2half_rn(x0 * c - x1 * s);
        k_cache[dst + 32] = __float2half_rn(x0 * s + x1 * c);
    }

    for (int col = tid; col < kv_width; col += (int)blockDim.x) {
        const int head = col >> 6;
        const int lane = col & 63;
        const long long dst =
            (((long long)b * n_kv_heads + head) * cache_capacity + cache_position) * 64LL + lane;
        v_cache[dst] = packed[packed_base + q_width + kv_width + col];
    }
}

extern "C" __global__
void dense_decode_fp16_d64(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half* __restrict__ out,
    int batch,
    int n_q_heads,
    int n_kv_heads,
    int cache_capacity,
    int position,
    float scale)
{
    const int lane = (int)threadIdx.x;
    if (lane >= 32) return;
    const int linear = (int)blockIdx.x;
    const int b = linear / n_q_heads;
    const int h = linear - b * n_q_heads;
    if (b >= batch || h >= n_q_heads) return;
    const int group_size = n_q_heads / n_kv_heads;
    const int kvh = h / group_size;
    const long long qbase = ((long long)b * n_q_heads + h) * 64LL;
    const float q0 = __half2float(q[qbase + lane]);
    const float q1 = __half2float(q[qbase + lane + 32]);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    const unsigned mask = 0xffffffffu;

    for (int key = 0; key <= position; ++key) {
        const long long base =
            (((long long)b * n_kv_heads + kvh) * cache_capacity + key) * 64LL;
        float dot = q0 * __half2float(k_cache[base + lane])
                  + q1 * __half2float(k_cache[base + lane + 32]);
        dot += __shfl_down_sync(mask, dot, 16);
        dot += __shfl_down_sync(mask, dot, 8);
        dot += __shfl_down_sync(mask, dot, 4);
        dot += __shfl_down_sync(mask, dot, 2);
        dot += __shfl_down_sync(mask, dot, 1);
        const float score = __shfl_sync(mask, dot, 0) * scale;
        float alpha = 1.0f;
        float beta = 1.0f;
        if (lane == 0) {
            if (key == 0) {
                running_max = score;
                running_sum = 1.0f;
                alpha = 0.0f;
            } else {
                const float new_max = running_max > score ? running_max : score;
                alpha = __expf(running_max - new_max);
                beta = __expf(score - new_max);
                running_sum = running_sum * alpha + beta;
                running_max = new_max;
            }
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);
        acc0 = acc0 * alpha + beta * __half2float(v_cache[base + lane]);
        acc1 = acc1 * alpha + beta * __half2float(v_cache[base + lane + 32]);
    }

    float inv = lane == 0 ? 1.0f / running_sum : 0.0f;
    inv = __shfl_sync(mask, inv, 0);
    out[qbase + lane] = __float2half_rn(acc0 * inv);
    out[qbase + lane + 32] = __float2half_rn(acc1 * inv);
}

extern "C" __global__
void dense_decode_fp16_d64_multiwarp(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half* __restrict__ out,
    int batch,
    int n_q_heads,
    int n_kv_heads,
    int cache_capacity,
    int position,
    float scale)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int linear = (int)blockIdx.x;
    const int b = linear / n_q_heads;
    const int h = linear - b * n_q_heads;
    if (b >= batch || h >= n_q_heads) return;

    const int group_size = n_q_heads / n_kv_heads;
    const int kvh = h / group_size;
    const long long qbase = ((long long)b * n_q_heads + h) * 64LL;
    const float q0 = __half2float(q[qbase + lane]);
    const float q1 = __half2float(q[qbase + lane + 32]);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    const unsigned mask = 0xffffffffu;

    // Four warps independently process interleaved cache positions.
    for (int key = warp; key <= position; key += 4) {
        const long long base =
            (((long long)b * n_kv_heads + kvh) * cache_capacity + key) * 64LL;
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

    // Merge the four online-softmax states with the same max-rescaling
    // identity used within each warp.
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
        for (int w = 0; w < 4; ++w) {
            merged_max = fmaxf(merged_max, warp_max[w]);
        }
        float merged_sum = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (warp_sum[w] > 0.0f) {
                merged_sum += warp_sum[w] * __expf(warp_max[w] - merged_max);
            }
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
"""


def _module():
    global _MODULE, _DISABLED
    if not _enabled() or _DISABLED:
        return None
    if _MODULE is not None:
        return _MODULE
    try:
        module = xp.RawModule(
            code=_SOURCE,
            options=("--std=c++14",),
            name_expressions=(
                "unpack_half_rope_store_fp16_d64",
                "dense_decode_fp16_d64",
                "dense_decode_fp16_d64_multiwarp",
            ),
        )
        module.get_function("unpack_half_rope_store_fp16_d64")
        module.get_function("dense_decode_fp16_d64")
        module.get_function("dense_decode_fp16_d64_multiwarp")
        _MODULE = module
    except Exception as exc:
        _fail(f"CUDA compilation failed: {type(exc).__name__}: {exc}")
        _DISABLED = True
        if _strict():
            raise
        return None
    return _MODULE


def fused_qkv_half_rope_store_fp16(qkv, q, k_cache, v_cache, cos, sin, *, position, cache_position):
    global _DISABLED
    module = _module()
    arrays = (qkv, q, k_cache, v_cache, cos, sin)
    if module is None or any(a.dtype != xp.float16 or not a.flags.c_contiguous for a in arrays):
        return False
    batch = int(qkv.shape[0])
    n_q_heads = int(q.shape[2])
    n_kv_heads = int(k_cache.shape[1])
    capacity = int(k_cache.shape[2])
    try:
        module.get_function("unpack_half_rope_store_fp16_d64")(
            (batch,), (256,), (
                qkv, cos, sin, q, k_cache, v_cache,
                np.int32(batch), np.int32(n_q_heads), np.int32(n_kv_heads),
                np.int32(capacity), np.int32(cache_position), np.int32(position),
            ),
        )
        return True
    except Exception as exc:
        _fail(f"QKV launch failed: {type(exc).__name__}: {exc}")
        _DISABLED = True
        if _strict():
            raise
        return False


def fused_dense_attention_fp16(q, k_cache, v_cache, *, position, scale):
    global _DISABLED
    module = _module()
    arrays = (q, k_cache, v_cache)
    if module is None or any(a.dtype != xp.float16 or not a.flags.c_contiguous for a in arrays):
        return None
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 64:
        return None
    if k_cache.shape != v_cache.shape or k_cache.shape[-1] != 64:
        return None
    batch = int(q.shape[0])
    n_q_heads = int(q.shape[2])
    n_kv_heads = int(k_cache.shape[1])
    if n_q_heads % n_kv_heads:
        return None
    capacity = int(k_cache.shape[2])
    position = int(position)
    if not 0 <= position < capacity:
        return None
    out = xp.empty_like(q)
    try:
        multiwarp = position >= 8
        kernel_name = (
            "dense_decode_fp16_d64_multiwarp"
            if multiwarp else "dense_decode_fp16_d64"
        )
        module.get_function(kernel_name)(
            (batch * n_q_heads,), (128 if multiwarp else 32,), (
                q, k_cache, v_cache, out,
                np.int32(batch), np.int32(n_q_heads), np.int32(n_kv_heads),
                np.int32(capacity), np.int32(position), np.float32(scale),
            ),
        )
        return out
    except Exception as exc:
        _fail(f"attention launch failed: {type(exc).__name__}: {exc}")
        _DISABLED = True
        if _strict():
            raise
        return None
