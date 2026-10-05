"""Fused BF16 single-token QKV unpack, RoPE and KV-cache insertion.

Autoregressive decode produces one packed ``[Q|K|V]`` row from the cuBLAS QKV
projection.  The historical inference path then unpacked Q/K/V into views,
launched several elementwise RoPE operations, materialized rotated K, and
finally transposed/copied K/V into their cache slots.  For a single token those
operations are launch-latency dominated.

0059H keeps the packed QKV GEMM unchanged and fuses everything after it into a
single CUDA launch:

    packed BF16 [Q|K|V]
        -> rotated BF16 Q output
        -> rotated BF16 K written directly to K cache[position]
        -> BF16 V written directly to V cache[position]

The kernel deliberately targets the production BF16/d_head=64 decode geometry.
Unsupported layouts transparently use the proven vectorized fallback.
"""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype


_MODULE = None
_DISABLED = False
_FAILURE = None


def _env_enabled(name: str, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def fused_decode_qkv_enabled() -> bool:
    return BACKEND_NAME == "cupy" and _env_enabled(
        "MINI_LLM_FUSED_DECODE_QKV_ROPE_STORE", "1"
    )


def fused_decode_qkv_failure_reason():
    return _FAILURE


def _strict_mode() -> bool:
    return _env_enabled("MINI_LLM_FUSED_DECODE_QKV_ROPE_STORE_STRICT", "0")


def _record_failure(reason):
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = str(reason)


def _get_module():
    """Compile/cache the single-token BF16 post-QKV fusion kernel."""
    global _MODULE, _DISABLED
    if not fused_decode_qkv_enabled() or _DISABLED:
        return None
    if _MODULE is not None:
        return _MODULE

    code = r"""
    #include <cuda_bf16.h>

    // One block handles one batch item. The production decode batch is usually
    // one, but the grid naturally supports larger batches. d_head is fixed at
    // 64 so each RoPE pair can be addressed without divisions by arbitrary
    // runtime head widths.
    extern "C" __global__
    void unpack_rope_store_bf16_d64(
        const __nv_bfloat16* __restrict__ packed,
        const __nv_bfloat16* __restrict__ cos_table,
        const __nv_bfloat16* __restrict__ sin_table,
        __nv_bfloat16* __restrict__ q_out,
        __nv_bfloat16* __restrict__ k_cache,
        __nv_bfloat16* __restrict__ v_cache,
        int batch,
        int n_q_heads,
        int n_kv_heads,
        int cache_capacity,
        int position)
    {
        const int b = (int)blockIdx.x;
        if (b >= batch) return;

        const int q_width = n_q_heads * 64;
        const int kv_width = n_kv_heads * 64;
        const int packed_width = q_width + 2 * kv_width;
        const long long packed_base = (long long)b * packed_width;
        const long long q_base = (long long)b * q_width;
        const long long trig_base = (long long)position * 32LL;

        // Q: rotate packed pairs and emit compact [B,Hq,64].
        const int q_pairs = q_width >> 1;
        for (int pair = (int)threadIdx.x; pair < q_pairs; pair += (int)blockDim.x) {
            const int col = pair << 1;
            const int pair_in_head = pair & 31;
            const float c = __bfloat162float(cos_table[trig_base + pair_in_head]);
            const float s = __bfloat162float(sin_table[trig_base + pair_in_head]);
            const float a = __bfloat162float(packed[packed_base + col]);
            const float d = __bfloat162float(packed[packed_base + col + 1]);
            q_out[q_base + col] = __float2bfloat16_rn(a * c - d * s);
            q_out[q_base + col + 1] = __float2bfloat16_rn(a * s + d * c);
        }

        // K: rotate directly into native [B,Hkv,T,64] cache layout.
        const int kv_pairs = kv_width >> 1;
        for (int pair = (int)threadIdx.x; pair < kv_pairs; pair += (int)blockDim.x) {
            const int local_col = pair << 1;
            const int kvh = local_col >> 6;
            const int lane = local_col & 63;
            const int pair_in_head = pair & 31;
            const float c = __bfloat162float(cos_table[trig_base + pair_in_head]);
            const float s = __bfloat162float(sin_table[trig_base + pair_in_head]);
            const long long src = packed_base + q_width + local_col;
            const float a = __bfloat162float(packed[src]);
            const float d = __bfloat162float(packed[src + 1]);
            const long long dst =
                (((long long)b * n_kv_heads + kvh) * cache_capacity + position) * 64LL + lane;
            k_cache[dst] = __float2bfloat16_rn(a * c - d * s);
            k_cache[dst + 1] = __float2bfloat16_rn(a * s + d * c);
        }

        // V needs no positional transform: write packed values directly into
        // the native cache slot, avoiding a temporary V view/copy kernel.
        for (int local_col = (int)threadIdx.x; local_col < kv_width;
             local_col += (int)blockDim.x) {
            const int kvh = local_col >> 6;
            const int lane = local_col & 63;
            const long long src = packed_base + q_width + kv_width + local_col;
            const long long dst =
                (((long long)b * n_kv_heads + kvh) * cache_capacity + position) * 64LL + lane;
            v_cache[dst] = packed[src];
        }
    }
    """

    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++14",),
            name_expressions=("unpack_rope_store_bf16_d64",),
        )
        module.get_function("unpack_rope_store_bf16_d64")
        _MODULE = module
        return module
    except Exception as exc:
        _record_failure(
            f"CUDA kernel compilation failed: {type(exc).__name__}: {exc}"
        )
        _DISABLED = True
        if _strict_mode():
            raise
        return None


def fused_unpack_rope_store_bf16_d64(
    packed_qkv,
    q_out,
    k_cache,
    v_cache,
    cos_table,
    sin_table,
    *,
    position: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> bool:
    """Run the 0059H fusion; return ``False`` for a transparent fallback."""
    global _DISABLED
    if not fused_decode_qkv_enabled() or _DISABLED:
        return False
    if BACKEND_NAME != "cupy":
        return False
    arrays = (packed_qkv, q_out, k_cache, v_cache, cos_table, sin_table)
    if not all(is_bfloat16_dtype(a.dtype) for a in arrays):
        return False
    if not all(a.flags.c_contiguous for a in arrays):
        _record_failure("non-contiguous fused QKV/RoPE/cache input")
        return False
    if int(q_out.shape[-1]) != 64 or int(k_cache.shape[-1]) != 64:
        return False
    if int(k_cache.shape[-2]) != int(v_cache.shape[-2]):
        return False
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        return False

    batch = int(packed_qkv.shape[0])
    q_width = int(n_q_heads) * 64
    kv_width = int(n_kv_heads) * 64
    expected_width = q_width + 2 * kv_width
    if int(packed_qkv.size) != batch * expected_width:
        _record_failure(
            f"packed QKV width mismatch: got {int(packed_qkv.size)//max(batch,1)}, "
            f"expected {expected_width}"
        )
        return False
    if tuple(q_out.shape) != (batch, 1, int(n_q_heads), 64):
        return False
    if int(k_cache.shape[0]) != batch or int(k_cache.shape[1]) != int(n_kv_heads):
        return False
    position = int(position)
    capacity = int(k_cache.shape[2])
    if position < 0 or position >= capacity:
        return False

    # RoPE tables are [1,T,1,32]. They are shared across batch and heads.
    if int(cos_table.size) < (position + 1) * 32 or int(sin_table.size) < (position + 1) * 32:
        return False

    module = _get_module()
    if module is None:
        return False
    kernel = module.get_function("unpack_rope_store_bf16_d64")
    threads = 256
    try:
        kernel(
            (batch,),
            (threads,),
            (
                packed_qkv,
                cos_table,
                sin_table,
                q_out,
                k_cache,
                v_cache,
                np.int32(batch),
                np.int32(n_q_heads),
                np.int32(n_kv_heads),
                np.int32(capacity),
                np.int32(position),
            ),
        )
        return True
    except Exception as exc:
        _record_failure(f"CUDA launch failed: {type(exc).__name__}: {exc}")
        _DISABLED = True
        if _strict_mode():
            raise
        return False
