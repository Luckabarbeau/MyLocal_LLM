"""Fused terminal-Landmark attention for 0058C long-memory inference.

Only the retrieval heads of one memory-aware Transformer layer use this path.
The kernel combines the bounded 4k working K/V ring with router-selected
external blocks whose K/V were preprojected once by the memory adapter.
"""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype

_MODULE = None
_DISABLED = False
_FAILURE = None


def _enabled():
    raw = os.environ.get("MINI_LLM_FUSED_TERMINAL_MEMORY_DECODE", "1").strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no", ""}


def terminal_memory_decode_failure_reason():
    return _FAILURE


def _strict():
    raw = os.environ.get("MINI_LLM_FUSED_TERMINAL_MEMORY_DECODE_STRICT", "0").strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def _record(reason):
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = str(reason)


def _get_module():
    global _MODULE, _DISABLED
    if not _enabled() or _DISABLED:
        return None
    if _MODULE is not None:
        return _MODULE
    code = r"""
    #include <cuda_bf16.h>

    extern "C" __global__
    void terminal_landmark_bf16_d64(
        const __nv_bfloat16* __restrict__ q,
        const __nv_bfloat16* __restrict__ local_k,
        const __nv_bfloat16* __restrict__ local_v,
        const __nv_bfloat16* __restrict__ mem_k,
        const __nv_bfloat16* __restrict__ mem_v,
        const int* __restrict__ query_heads,
        const int* __restrict__ local_kv_heads,
        const long long* __restrict__ selected_blocks,
        const float* __restrict__ gate_scores,
        const unsigned char* __restrict__ selected_valid,
        const unsigned char* __restrict__ route_valid,
        __nv_bfloat16* __restrict__ out,
        int batch,
        int n_q_heads,
        int n_local_kv_heads,
        int n_mem_heads,
        int n_mem_kv_heads,
        int local_capacity,
        int local_count,
        int cache_start,
        int mem_capacity,
        int mem_count,
        int mem_start,
        int first_token_offset,
        int selected_count,
        int block_size,
        float attn_scale,
        float gate_scale)
    {
        const int lane = (int)threadIdx.x;
        if (lane >= 32) return;
        const int linear = (int)blockIdx.x;
        const int b = linear / n_mem_heads;
        const int mh = linear - b * n_mem_heads;
        if (b >= batch || mh >= n_mem_heads) return;
        const long long out_base = ((long long)b * n_mem_heads + mh) * 64LL;
        if (!route_valid[b]) {
            out[out_base + lane] = __float2bfloat16_rn(0.0f);
            out[out_base + lane + 32] = __float2bfloat16_rn(0.0f);
            return;
        }

        const int qh = query_heads[mh];
        const int lkvh = local_kv_heads[mh];
        const int mkvh = mh * n_mem_kv_heads / n_mem_heads;
        const long long qbase = ((long long)b * n_q_heads + qh) * 64LL;
        const float q0 = __bfloat162float(q[qbase + lane]);
        const float q1 = __bfloat162float(q[qbase + lane + 32]);
        const unsigned mask = 0xffffffffu;

        float top_max = -3.402823466e+38F;
        float top_sum = 0.0f;
        int top_seen = 0;

        // Historical block gates participate in the top-level softmax as one
        // scalar item per selected block. Process them first; block values are
        // computed only after the final top-level normalization is known.
        if (lane == 0) {
            for (int r = 0; r < selected_count; ++r) {
                const long long ridx = (long long)b * selected_count + r;
                if (!selected_valid[ridx]) continue;
                const float s = gate_scores[ridx] * gate_scale;
                if (top_seen == 0) {
                    top_max = s;
                    top_sum = 1.0f;
                    top_seen = 1;
                } else {
                    const float nm = top_max > s ? top_max : s;
                    top_sum = top_sum * __expf(top_max - nm) + __expf(s - nm);
                    top_max = nm;
                    ++top_seen;
                }
            }
        }
        top_max = __shfl_sync(mask, top_max, 0);
        top_sum = __shfl_sync(mask, top_sum, 0);
        top_seen = __shfl_sync(mask, top_seen, 0);

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        // All working-window tokens compete directly with historical block gates.
        for (int key = 0; key < local_count; ++key) {
            int slot = cache_start + key;
            if (slot >= local_capacity) slot -= local_capacity;
            const long long base =
                (((long long)b * n_local_kv_heads + lkvh) * local_capacity + slot) * 64LL;
            float dot = q0 * __bfloat162float(local_k[base + lane])
                      + q1 * __bfloat162float(local_k[base + lane + 32]);
            dot += __shfl_down_sync(mask, dot, 16);
            dot += __shfl_down_sync(mask, dot, 8);
            dot += __shfl_down_sync(mask, dot, 4);
            dot += __shfl_down_sync(mask, dot, 2);
            dot += __shfl_down_sync(mask, dot, 1);
            const float score = __shfl_sync(mask, dot, 0) * attn_scale;
            float alpha = 1.0f;
            float beta = 1.0f;
            if (lane == 0) {
                if (top_seen == 0) {
                    top_max = score;
                    top_sum = 1.0f;
                    top_seen = 1;
                    alpha = 0.0f;
                    beta = 1.0f;
                } else {
                    const float nm = top_max > score ? top_max : score;
                    alpha = __expf(top_max - nm);
                    beta = __expf(score - nm);
                    top_sum = top_sum * alpha + beta;
                    top_max = nm;
                    ++top_seen;
                }
            }
            alpha = __shfl_sync(mask, alpha, 0);
            beta = __shfl_sync(mask, beta, 0);
            acc0 = acc0 * alpha + beta * __bfloat162float(local_v[base + lane]);
            acc1 = acc1 * alpha + beta * __bfloat162float(local_v[base + lane + 32]);
        }

        top_max = __shfl_sync(mask, top_max, 0);
        top_sum = __shfl_sync(mask, top_sum, 0);

        // Compute each selected block's conditional token softmax exactly once,
        // then multiply its block value by its already-normalized top-level gate.
        for (int r = 0; r < selected_count; ++r) {
            const long long ridx = (long long)b * selected_count + r;
            if (!selected_valid[ridx]) continue;
            const long long block = selected_blocks[ridx];
            if (block < 0) continue;
            const long long block_logical =
                (long long)first_token_offset + block * (long long)block_size;
            if (block_logical < 0 || block_logical + block_size > mem_count) continue;

            float inner_max = -3.402823466e+38F;
            float inner_sum = 0.0f;
            int inner_seen = 0;
            float bacc0 = 0.0f;
            float bacc1 = 0.0f;
            for (int j = 0; j < block_size; ++j) {
                const long long logical = block_logical + j;
                int slot = mem_start + (int)logical;
                if (slot >= mem_capacity) slot -= mem_capacity;
                const long long base =
                    (((long long)b * n_mem_kv_heads + mkvh) * mem_capacity + slot) * 64LL;
                float dot = q0 * __bfloat162float(mem_k[base + lane])
                          + q1 * __bfloat162float(mem_k[base + lane + 32]);
                dot += __shfl_down_sync(mask, dot, 16);
                dot += __shfl_down_sync(mask, dot, 8);
                dot += __shfl_down_sync(mask, dot, 4);
                dot += __shfl_down_sync(mask, dot, 2);
                dot += __shfl_down_sync(mask, dot, 1);
                const float score = __shfl_sync(mask, dot, 0) * attn_scale;
                float alpha = 1.0f;
                float beta = 1.0f;
                if (lane == 0) {
                    if (inner_seen == 0) {
                        inner_max = score;
                        inner_sum = 1.0f;
                        inner_seen = 1;
                        alpha = 0.0f;
                    } else {
                        const float nm = inner_max > score ? inner_max : score;
                        alpha = __expf(inner_max - nm);
                        beta = __expf(score - nm);
                        inner_sum = inner_sum * alpha + beta;
                        inner_max = nm;
                        ++inner_seen;
                    }
                }
                alpha = __shfl_sync(mask, alpha, 0);
                beta = __shfl_sync(mask, beta, 0);
                bacc0 = bacc0 * alpha + beta * __bfloat162float(mem_v[base + lane]);
                bacc1 = bacc1 * alpha + beta * __bfloat162float(mem_v[base + lane + 32]);
            }
            inner_sum = __shfl_sync(mask, inner_sum, 0);
            const float inv_inner = inner_sum > 0.0f ? 1.0f / inner_sum : 0.0f;
            const float gate = gate_scores[ridx] * gate_scale;
            // Keep the same unnormalised top-level numerator as the local
            // tokens.  The common 1/top_sum factor is applied once below.
            const float block_weight = __expf(gate - top_max);
            acc0 += block_weight * (bacc0 * inv_inner);
            acc1 += block_weight * (bacc1 * inv_inner);
        }

        const float inv_top = top_sum > 0.0f ? 1.0f / top_sum : 0.0f;
        out[out_base + lane] = __float2bfloat16_rn(acc0 * inv_top);
        out[out_base + lane + 32] = __float2bfloat16_rn(acc1 * inv_top);
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++14",),
            name_expressions=("terminal_landmark_bf16_d64",),
        )
        module.get_function("terminal_landmark_bf16_d64")
        _MODULE = module
        return module
    except Exception as exc:
        _record(f"CUDA kernel compilation failed: {type(exc).__name__}: {exc}")
        _DISABLED = True
        if _strict():
            raise
        return None


def _reference(
    q, local_k, local_v, mem_k, mem_v, query_heads, local_kv_heads,
    selected_blocks, gate_scores, selected_valid, route_valid,
    *, local_count, cache_start, mem_count, mem_start, first_token_offset,
    block_size, gate_scale, attn_scale,
):
    batch = int(q.shape[0])
    n_mem_heads = int(query_heads.shape[0])
    mem_capacity = int(mem_k.shape[2])
    out = xp.zeros((batch, 1, n_mem_heads, 64), dtype=q.dtype)
    for b in range(batch):
        if not bool(route_valid[b]):
            continue
        for mh in range(n_mem_heads):
            qh = int(query_heads[mh])
            kvh = int(local_kv_heads[mh])
            mkvh = mh * int(mem_k.shape[1]) // n_mem_heads
            qf = q[b, 0, qh].astype(xp.float32, copy=False)
            slots = (cache_start + xp.arange(local_count, dtype=xp.int64)) % int(local_k.shape[2])
            lk = local_k[b, kvh, slots].astype(xp.float32, copy=False)
            lv = local_v[b, kvh, slots].astype(xp.float32, copy=False)
            local_logits = xp.sum(lk * qf[None, :], axis=-1) * float(attn_scale)
            block_values = []
            block_gates = []
            for r in range(int(selected_blocks.shape[1])):
                if not bool(selected_valid[b, r]):
                    continue
                block = int(selected_blocks[b, r])
                logical_start = int(first_token_offset) + block * int(block_size)
                if logical_start < 0 or logical_start + int(block_size) > int(mem_count):
                    continue
                logical = logical_start + xp.arange(int(block_size), dtype=xp.int64)
                mem_slots = (int(mem_start) + logical) % mem_capacity
                mk = mem_k[b, mkvh, mem_slots].astype(xp.float32, copy=False)
                mv = mem_v[b, mkvh, mem_slots].astype(xp.float32, copy=False)
                logits = xp.sum(mk * qf[None, :], axis=-1) * float(attn_scale)
                probs = xp.exp(logits - xp.max(logits))
                probs /= xp.sum(probs)
                block_values.append(xp.sum(probs[:, None] * mv, axis=0))
                block_gates.append(gate_scores[b, r] * float(gate_scale))
            gates = xp.asarray(block_gates, dtype=xp.float32)
            combined = xp.concatenate((local_logits, gates), axis=0)
            probs = xp.exp(combined - xp.max(combined))
            probs /= xp.sum(probs)
            ctx = xp.sum(probs[:local_count, None] * lv, axis=0)
            for i, value in enumerate(block_values):
                ctx += probs[local_count + i] * value
            out[b, 0, mh] = ctx.astype(q.dtype, copy=False)
    return out


def terminal_landmark_decode(
    q, local_k, local_v, memory_store, route, *, query_heads,
    local_kv_heads, local_count, cache_start, gate_scale, attn_scale,
):
    """Return [B,1,Hmem,64] replacement contexts for terminal retrieval heads."""
    qheads = xp.ascontiguousarray(xp.asarray(query_heads, dtype=xp.int32))
    kvheads = xp.ascontiguousarray(xp.asarray(local_kv_heads, dtype=xp.int32))
    selected = route.selected_blocks
    gates = route.gate_scores
    selected_valid = route.selected_valid
    route_valid = route.route_valid

    if (
        _enabled()
        and is_bfloat16_dtype(q.dtype)
        and int(q.shape[-1]) == 64
        and q.flags.c_contiguous
        and local_k.flags.c_contiguous
        and local_v.flags.c_contiguous
        and memory_store.memory_k.flags.c_contiguous
        and memory_store.memory_v.flags.c_contiguous
    ):
        module = _get_module()
        if module is not None:
            out = xp.empty(
                (int(q.shape[0]), 1, int(qheads.size), 64), dtype=q.dtype
            )
            kernel = module.get_function("terminal_landmark_bf16_d64")
            try:
                kernel(
                    (int(q.shape[0]) * int(qheads.size),),
                    (32,),
                    (
                        q,
                        local_k,
                        local_v,
                        memory_store.memory_k,
                        memory_store.memory_v,
                        qheads,
                        kvheads,
                        selected,
                        gates,
                        selected_valid,
                        route_valid,
                        out,
                        np.int32(q.shape[0]),
                        np.int32(q.shape[2]),
                        np.int32(local_k.shape[1]),
                        np.int32(qheads.size),
                        np.int32(memory_store.memory_k.shape[1]),
                        np.int32(local_k.shape[2]),
                        np.int32(local_count),
                        np.int32(cache_start),
                        np.int32(memory_store.external_capacity_tokens),
                        np.int32(memory_store.token_count),
                        np.int32(memory_store.ring_start),
                        np.int32(route.first_token_offset),
                        np.int32(selected.shape[1]),
                        np.int32(memory_store.block_size),
                        np.float32(attn_scale),
                        np.float32(gate_scale),
                    ),
                )
                return out
            except Exception as exc:
                global _DISABLED
                _record(f"CUDA launch failed: {type(exc).__name__}: {exc}")
                _DISABLED = True
                if _strict():
                    raise

    return _reference(
        q, local_k, local_v, memory_store.memory_k, memory_store.memory_v,
        qheads, kvheads, selected, gates, selected_valid, route_valid,
        local_count=int(local_count), cache_start=int(cache_start),
        mem_count=int(memory_store.token_count), mem_start=int(memory_store.ring_start),
        first_token_offset=int(route.first_token_offset),
        block_size=int(memory_store.block_size),
        gate_scale=float(gate_scale), attn_scale=float(attn_scale),
    )
