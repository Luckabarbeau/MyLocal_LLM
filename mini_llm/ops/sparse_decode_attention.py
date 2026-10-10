"""Fused one-token sparse attention for CuPy/BF16 autoregressive decode.

The training sparse-attention kernels are optimized for many query rows.  During
KV-cached generation there is exactly one query row, so materializing gathered
K/V tensors, score vectors and probability vectors is mostly launch/allocation
latency.  This module executes the complete sparse read for all query heads in
one CUDA launch:

    Q head -> direct cache addressing -> online softmax -> weighted V

Static local/dilated/global patterns are generated arithmetically inside the
kernel.  Learned retrieval heads consume the already-cached router-selected
block IDs and optional per-block logit biases.  The kernel deliberately owns no
routing logic and no Q/K/V projection so it can be validated independently and
fall back to the existing reference decode path.
"""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype


_FUSED_SPARSE_DECODE_MODULE = None
_FUSED_SPARSE_DECODE_DISABLED = False
_FUSED_SPARSE_DECODE_FAILURE = None

# Stable metadata codes shared with attention_inference.py.
HEAD_DENSE = 0
HEAD_LOCAL = 1
HEAD_DILATED = 2
HEAD_GLOBAL = 3
HEAD_RETRIEVAL = 4


def _env_enabled(name: str, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def fused_sparse_decode_enabled() -> bool:
    # 0059F: keep the experimental 0059E kernel opt-in until it has passed
    # the strict device validator.  A silent 0059E rejection previously added
    # wrapper overhead while users believed the fused kernel was being timed.
    return BACKEND_NAME == "cupy" and _env_enabled(
        "MINI_LLM_FUSED_SPARSE_DECODE", "0"
    )



def fused_sparse_decode_failure_reason():
    """Return the first compile/layout rejection seen by the fused path."""
    return _FUSED_SPARSE_DECODE_FAILURE


def _record_failure(reason):
    global _FUSED_SPARSE_DECODE_FAILURE
    if _FUSED_SPARSE_DECODE_FAILURE is None:
        _FUSED_SPARSE_DECODE_FAILURE = str(reason)


def _strict_mode() -> bool:
    return _env_enabled("MINI_LLM_FUSED_SPARSE_DECODE_STRICT", "0")


def _get_module():
    """Compile/cache the BF16 single-query sparse-attention kernel."""
    global _FUSED_SPARSE_DECODE_MODULE, _FUSED_SPARSE_DECODE_DISABLED
    if not fused_sparse_decode_enabled() or _FUSED_SPARSE_DECODE_DISABLED:
        return None
    if _FUSED_SPARSE_DECODE_MODULE is not None:
        return _FUSED_SPARSE_DECODE_MODULE

    code = r"""
    #include <cuda_bf16.h>

    // One CUDA block == one (batch, query-head) pair.  d_head=64 is the
    // production geometry, so one warp owns the head and each lane accumulates
    // two output dimensions.  Scores and online-softmax state stay in FP32.
    extern "C" __global__
    void sparse_decode_bf16_d64(
        const __nv_bfloat16* __restrict__ q,
        const __nv_bfloat16* __restrict__ k_cache,
        const __nv_bfloat16* __restrict__ v_cache,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ head_kind,
        const int* __restrict__ kv_head,
        const int* __restrict__ p0,
        const int* __restrict__ p1,
        const int* __restrict__ p2,
        const int* __restrict__ retrieval_qmap,
        const long long* __restrict__ selected_blocks,
        const float* __restrict__ retrieval_block_bias,
        int batch,
        int n_q_heads,
        int n_kv_heads,
        int cache_capacity,
        int position,
        int cache_start,
        int retrieval_key_shift,
        int retrieval_active,
        int retrieval_queries,
        int retrieval_blocks,
        int retrieval_block_size,
        float scale)
    {
        const int lane = (int)threadIdx.x;
        if (lane >= 32) return;
        const int linear = (int)blockIdx.x;
        const int b = linear / n_q_heads;
        const int h = linear - b * n_q_heads;
        if (b >= batch || h >= n_q_heads) return;

        const int kind = head_kind[h];
        const int kvh = kv_head[h];
        if (kvh < 0 || kvh >= n_kv_heads) return;

        // d_head is fixed to 64 for this specialized production kernel.
        const long long qbase = ((long long)b * n_q_heads + h) * 64LL;
        const float q0 = __bfloat162float(q[qbase + lane]);
        const float q1 = __bfloat162float(q[qbase + lane + 32]);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        // Avoid host C math headers: NVRTC environments (notably some CUDA 13
        // installations) do not expose <math.h> through RawModule include paths.
        // This finite sentinel is safely below any practical attention score.
        float running_max = -3.402823466e+38F; // lane 0 only
        float running_sum = 0.0f;      // meaningful in lane 0 only
        int keys_seen = 0;             // meaningful in lane 0 only
        const unsigned mask = 0xffffffffu;

        // Process one cache key with numerically stable online softmax.  No K/V
        // gather, score array or probability array is materialized.
        #define PROCESS_KEY(KEY_VALUE, BIAS_VALUE) do { \
            const int _key = (KEY_VALUE); \
            if (_key >= 0 && _key <= position && _key < cache_capacity) { \
                int _slot = cache_start + _key; \
                if (_slot >= cache_capacity) _slot -= cache_capacity; \
                const long long _base = \
                    (((long long)b * n_kv_heads + kvh) * cache_capacity + _slot) * 64LL; \
                float _dot = q0 * __bfloat162float(k_cache[_base + lane]) \
                           + q1 * __bfloat162float(k_cache[_base + lane + 32]); \
                _dot += __shfl_down_sync(mask, _dot, 16); \
                _dot += __shfl_down_sync(mask, _dot, 8); \
                _dot += __shfl_down_sync(mask, _dot, 4); \
                _dot += __shfl_down_sync(mask, _dot, 2); \
                _dot += __shfl_down_sync(mask, _dot, 1); \
                const float _score = __shfl_sync(mask, _dot, 0) * scale + (BIAS_VALUE); \
                float _alpha = 0.0f; \
                float _beta = 1.0f; \
                if (lane == 0) { \
                    if (keys_seen == 0) { \
                        running_max = _score; \
                        running_sum = 1.0f; \
                        keys_seen = 1; \
                    } else { \
                        const float _new_max = (running_max > _score) ? running_max : _score; \
                        _alpha = __expf(running_max - _new_max); \
                        _beta = __expf(_score - _new_max); \
                        running_sum = running_sum * _alpha + _beta; \
                        running_max = _new_max; \
                        keys_seen += 1; \
                    } \
                } \
                const int _seen = __shfl_sync(mask, keys_seen, 0); \
                if (_seen == 1) { \
                    _alpha = 0.0f; \
                    _beta = 1.0f; \
                } else { \
                    _alpha = __shfl_sync(mask, _alpha, 0); \
                    _beta = __shfl_sync(mask, _beta, 0); \
                } \
                acc0 = acc0 * _alpha + _beta * __bfloat162float(v_cache[_base + lane]); \
                acc1 = acc1 * _alpha + _beta * __bfloat162float(v_cache[_base + lane + 32]); \
            } \
        } while (0)

        if (kind == 0) { // dense
            for (int key = 0; key <= position; ++key) {
                PROCESS_KEY(key, 0.0f);
            }
        } else if (kind == 1) { // local: p0=window
            const int window = p0[h];
            int first = position - window + 1;
            if (first < 0) first = 0;
            for (int key = first; key <= position; ++key) {
                PROCESS_KEY(key, 0.0f);
            }
        } else if (kind == 2) { // dilated: p0=window,p1=dilation,p2=offset
            const int window = p0[h];
            const int dilation = p1[h];
            const int offset = p2[h];
            const int last = position - offset;
            if (last >= 0 && dilation > 0) {
                int first = position - window + 1;
                if (first < 0) first = 0;
                if (last >= first) {
                    const int n = (last - first) / dilation;
                    const int start = last - n * dilation;
                    for (int key = start; key <= last; key += dilation) {
                        PROCESS_KEY(key, 0.0f);
                    }
                }
            }
        } else if (kind == 3) { // global: p0=stride,p1=offset,p2=include_current
            const int stride = p0[h];
            const int offset = p1[h];
            if (stride > 0 && position >= offset) {
                for (int key = offset; key <= position; key += stride) {
                    PROCESS_KEY(key, 0.0f);
                }
            }
            if (p2[h]) {
                const bool current_is_anchor =
                    position >= offset && stride > 0 && ((position - offset) % stride == 0);
                if (!current_is_anchor) PROCESS_KEY(position, 0.0f);
            }
        } else if (kind == 4) { // retrieval
            if (retrieval_active) {
                const int rq = retrieval_qmap[h];
                if (rq >= 0 && rq < retrieval_queries) {
                    for (int r = 0; r < retrieval_blocks; ++r) {
                        const long long ridx =
                            ((long long)b * retrieval_queries + rq) * retrieval_blocks + r;
                        const long long block = selected_blocks[ridx];
                        const float bias = retrieval_block_bias[ridx];
                        const long long base = block * (long long)retrieval_block_size;
                        for (int j = 0; j < retrieval_block_size; ++j) {
                            PROCESS_KEY((int)(base + j + retrieval_key_shift), bias);
                        }
                    }
                }
            }
        }

        float inv = 0.0f;
        if (lane == 0 && keys_seen > 0) inv = 1.0f / running_sum;
        inv = __shfl_sync(mask, inv, 0);
        const long long obase = ((long long)b * n_q_heads + h) * 64LL;
        out[obase + lane] = __float2bfloat16_rn(acc0 * inv);
        out[obase + lane + 32] = __float2bfloat16_rn(acc1 * inv);

        #undef PROCESS_KEY
    }
    """

    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++14",),
            name_expressions=("sparse_decode_bf16_d64",),
        )
        # RawModule compilation is lazy. Force symbol resolution now so a
        # toolchain/architecture issue is caught once and can use the fallback.
        module.get_function("sparse_decode_bf16_d64")
        _FUSED_SPARSE_DECODE_MODULE = module
        return module
    except Exception as exc:
        _record_failure(f"CUDA kernel compilation failed: {type(exc).__name__}: {exc}")
        if _strict_mode():
            raise
        _FUSED_SPARSE_DECODE_DISABLED = True
        return None


def fused_sparse_decode_bf16_d64(
    q,
    k_cache,
    v_cache,
    out,
    *,
    head_kind,
    kv_head,
    p0,
    p1,
    p2,
    retrieval_qmap,
    selected_blocks,
    retrieval_block_bias,
    position: int,
    retrieval_active: bool,
    retrieval_queries: int,
    retrieval_blocks: int,
    retrieval_block_size: int,
    scale: float,
    cache_start: int = 0,
    retrieval_key_shift: int = 0,
) -> bool:
    """Write fused sparse attention into ``out`` and return whether it ran.

    The specialization is intentionally narrow: CuPy, BF16, d_head=64, and
    contiguous native-Hkv caches. Unsupported layouts transparently use the
    existing inference implementation unless strict mode is enabled.
    """
    module = _get_module()
    if module is None:
        if fused_sparse_decode_enabled() and _FUSED_SPARSE_DECODE_FAILURE is None:
            _record_failure("CUDA kernel module unavailable")
        return False

    arrays = (q, k_cache, v_cache, out)
    if any(not is_bfloat16_dtype(a.dtype) for a in arrays):
        _record_failure(
            "dtype rejection: expected BF16 q/k/v/out, got "
            + ", ".join(str(a.dtype) for a in arrays)
        )
        return False
    expected_q_shape = (1, int(head_kind.size), 64)
    if q.ndim != 4 or tuple(q.shape[1:]) != expected_q_shape:
        _record_failure(f"q shape rejection: got {tuple(q.shape)}, expected [B,{expected_q_shape[0]},{expected_q_shape[1]},{expected_q_shape[2]}]")
        return False
    if k_cache.ndim != 4 or v_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        _record_failure(f"K/V cache shape rejection: K={tuple(k_cache.shape)}, V={tuple(v_cache.shape)}")
        return False
    if int(k_cache.shape[-1]) != 64:
        _record_failure(f"d_head rejection: cache d_head={int(k_cache.shape[-1])}, expected 64")
        return False
    if tuple(out.shape) != tuple(q.shape):
        _record_failure(f"output shape rejection: out={tuple(out.shape)}, q={tuple(q.shape)}")
        return False
    if any(not a.flags.c_contiguous for a in arrays):
        labels = ("q", "k_cache", "v_cache", "out")
        bad = [name for name, arr in zip(labels, arrays) if not arr.flags.c_contiguous]
        _record_failure("non-contiguous fused input(s): " + ", ".join(bad))
        return False

    meta = (head_kind, kv_head, p0, p1, p2, retrieval_qmap)
    if any(a.dtype != xp.int32 or not a.flags.c_contiguous for a in meta):
        _record_failure("head metadata rejection: expected contiguous int32 arrays")
        return False
    if selected_blocks.dtype != xp.int64 or not selected_blocks.flags.c_contiguous:
        _record_failure(f"selected block rejection: dtype={selected_blocks.dtype}, contiguous={selected_blocks.flags.c_contiguous}")
        return False
    if retrieval_block_bias.dtype != xp.float32 or not retrieval_block_bias.flags.c_contiguous:
        _record_failure(f"retrieval bias rejection: dtype={retrieval_block_bias.dtype}, contiguous={retrieval_block_bias.flags.c_contiguous}")
        return False

    batch = int(q.shape[0])
    n_q_heads = int(q.shape[2])
    n_kv_heads = int(k_cache.shape[1])
    capacity = int(k_cache.shape[2])
    position = int(position)
    cache_start = int(cache_start)
    retrieval_key_shift = int(retrieval_key_shift)
    if not (0 <= cache_start < capacity):
        _record_failure(f"cache start rejection: cache_start={cache_start}, capacity={capacity}")
        return False
    if not (0 <= position < capacity):
        _record_failure(f"cache position rejection: position={position}, capacity={capacity}")
        return False

    if retrieval_active:
        expected = (batch, int(retrieval_queries), int(retrieval_blocks))
        if tuple(selected_blocks.shape) != expected:
            _record_failure(f"retrieval selection shape rejection: got {tuple(selected_blocks.shape)}, expected {expected}")
            return False
        if tuple(retrieval_block_bias.shape) != expected:
            _record_failure(f"retrieval bias shape rejection: got {tuple(retrieval_block_bias.shape)}, expected {expected}")
            return False

    kernel = module.get_function("sparse_decode_bf16_d64")
    kernel(
        (batch * n_q_heads,),
        (32,),
        (
            q,
            k_cache,
            v_cache,
            out,
            head_kind,
            kv_head,
            p0,
            p1,
            p2,
            retrieval_qmap,
            selected_blocks,
            retrieval_block_bias,
            np.int32(batch),
            np.int32(n_q_heads),
            np.int32(n_kv_heads),
            np.int32(capacity),
            np.int32(position),
            np.int32(cache_start),
            np.int32(retrieval_key_shift),
            np.int32(1 if retrieval_active else 0),
            np.int32(retrieval_queries),
            np.int32(retrieval_blocks),
            np.int32(retrieval_block_size),
            np.float32(scale),
        ),
    )
    return True
