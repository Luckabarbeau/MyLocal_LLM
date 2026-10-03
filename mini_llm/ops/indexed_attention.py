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
from .attention_selection import KeySelectionPlan


_DEFAULT_QUERY_CHUNK_SIZE = 128
_DEFAULT_LOCAL_QUERY_CHUNK_SIZE = 512
_DEFAULT_DILATED_QUERY_CHUNK_SIZE = 1024


_LOCAL_SOFTMAX_MODULE = None
_LOCAL_SOFTMAX_DISABLED = False


_RETRIEVAL_SCATTER_MODULE = None
_RETRIEVAL_SCATTER_DISABLED = False


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


def _masked_softmax_forward(scores, valid_mask, logit_multiplier=1.0):
    """Stable softmax that returns exactly zero for rows with no valid keys."""
    work = (
        scores.astype("float32", copy=False)
        if is_low_precision_dtype(scores.dtype)
        else scores
    )
    valid_mask = valid_mask.astype(bool, copy=False)
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
        return probs.astype(source_dtype)
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
        dscores = _softmax_backward(dprobs, probs_chunk)
        dscores = xp.where(chunk_valid, dscores, 0.0)
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
    context_dtype = xp.float32 if bf16_attention else q.dtype
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
                # N-D BF16 matmul is unavailable.  Fold query heads into the
                # row dimension; K is shared by the whole native GQA group.
                n_group_heads = len(head_group)
                q_count = q_end - q_start
                k_count = key_end - key_start
                scores = xp.empty(
                    (batch, n_group_heads, q_count, k_count), dtype=xp.float32
                )
                for batch_idx in range(batch):
                    q_2d = xp.ascontiguousarray(
                        q_chunk[batch_idx].reshape(n_group_heads * q_count, d_head)
                    )
                    k_2d = xp.ascontiguousarray(k_native[batch_idx, 0])
                    score_bf16 = q_2d @ k_2d.T
                    scores[batch_idx] = (
                        score_bf16.reshape(n_group_heads, q_count, k_count)
                        .astype("float32")
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
                    probs_2d = xp.ascontiguousarray(
                        probs_chunk[batch_idx]
                        .reshape(n_group_heads * q_count, k_count)
                        .astype(q.dtype)
                    )
                    v_2d = xp.ascontiguousarray(v_native[batch_idx, 0])
                    context_bf16 = probs_2d @ v_2d
                    context_heads[
                        batch_idx, head_group, q_start:q_end, :
                    ] = context_bf16.reshape(n_group_heads, q_count, d_head)
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

    batch, query_length, n_q_heads, d_head = q.shape
    scale = cache["scale"]
    bf16_attention = cache["bf16_attention"]
    bf16_tensorcore = cache.get("bf16_tensorcore", False)
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
                probs_compute = _restore_cached_probs(probs_chunk, True)
                n_group_heads = len(head_group)
                q_count = q_end - q_start
                k_count = key_end - key_start
                dprobs = xp.empty(
                    (batch, n_group_heads, q_count, k_count), dtype=xp.float32
                )
                # dP = dO V^T.  Fold heads into rows while keeping the shared
                # native V matrix exactly once per batch.
                dcontext_bf16 = []
                for batch_idx in range(batch):
                    dc_2d = xp.ascontiguousarray(
                        dcontext_chunk[batch_idx]
                        .reshape(n_group_heads * q_count, d_head)
                        .astype(q.dtype)
                    )
                    dcontext_bf16.append(dc_2d)
                    v_2d = xp.ascontiguousarray(v_native[batch_idx, 0])
                    dp_bf16 = dc_2d @ v_2d.T
                    dprobs[batch_idx] = dp_bf16.reshape(
                        n_group_heads, q_count, k_count
                    ).astype("float32")

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
                    q_2d = xp.ascontiguousarray(
                        q_chunk[batch_idx].reshape(n_group_heads * q_count, d_head)
                    )
                    k_2d = xp.ascontiguousarray(k_native[batch_idx, 0])
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

                    dq_bf16 = ds_2d @ k_2d
                    dk_bf16 = ds_2d.T @ q_2d
                    dv_bf16 = p_2d.T @ dc_2d
                    dq_heads[batch_idx, head_group, q_start:q_end, :] = (
                        dq_bf16.reshape(n_group_heads, q_count, d_head)
                        .astype(grad_dtype)
                        * scale
                    )
                    dk_native[batch_idx] = dk_bf16.astype(grad_dtype) * scale
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
            dk[:, key_start:key_end, kvh, :] += dk_native.astype(
                grad_dtype, copy=False
            )
            dv[:, key_start:key_end, kvh, :] += dv_native.astype(
                grad_dtype, copy=False
            )

    return dq_heads.transpose(0, 2, 1, 3), dk, dv


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

    The generic indexed implementation treats every selected token as an
    arbitrary gather.  Fixed dilation has much more structure: queries can be
    split by ``t % dilation`` and each phase attends to a contiguous causal
    window in a downsampled K/V sequence.  This implementation exploits that
    structure while preserving exactly the visibility of
    :func:`build_dilated_causal_plan`.

    For a query ``t = r + m*d`` the visible keys live in residue
    ``(r-offset) mod d``.  In reduced coordinates they form an ordinary
    trailing window ending at ``m`` or ``m-1`` depending on whether the phase
    crosses the sequence origin.
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
    q_heads = q.transpose(0, 2, 1, 3)
    bf16_attention = is_bfloat16_dtype(q.dtype)
    context_dtype = xp.float32 if bf16_attention else q.dtype
    context_heads = xp.zeros(
        (batch, n_q_heads, query_length, d_head), dtype=context_dtype
    )
    score_prescale = 1.0 / 32.0 if q.dtype == xp.float16 else 1.0
    chunk_caches = [] if return_cache else None

    for query_residue in range(dilation):
        # Empty residue classes occur when dilation exceeds sequence length.
        phase_length = (query_length - 1 - query_residue) // dilation + 1
        if phase_length <= 0:
            continue
        key_residue = (query_residue - offset) % dilation
        key_phase_length = (key_length - 1 - key_residue) // dilation + 1
        if key_phase_length <= 0:
            continue

        # If r < offset, t-offset lies in the previous reduced-sequence cell.
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

            q_token_slice = slice(
                query_residue + phase_start * dilation,
                query_residue + phase_end * dilation,
                dilation,
            )
            q_chunk = q_heads[:, :, q_token_slice, :]

            if key_end <= key_start:
                # No historical token exists yet for this phase (possible for
                # non-zero offsets at the very beginning of a sequence).
                if return_cache:
                    chunk_caches.append(
                        (
                            query_residue,
                            phase_start,
                            phase_end,
                            key_residue,
                            key_start,
                            key_end,
                            None,
                            None,
                            alignment_shift,
                        )
                    )
                continue

            key_token_slice = slice(
                key_residue + key_start * dilation,
                key_residue + key_end * dilation,
                dilation,
            )
            k_span = xp.take(k[:, key_token_slice, :, :], kv_map, axis=2)
            k_heads = k_span.transpose(0, 2, 1, 3)
            if bf16_attention:
                q_score = q_chunk.astype("float32") * scale
                k_score = k_heads.astype("float32")
            else:
                q_score = q_chunk * (scale * score_prescale)
                k_score = k_heads

            scores = xp.matmul(q_score, k_score.swapaxes(-1, -2))
            q_indices = xp.arange(phase_start, phase_end, dtype=xp.int64)[:, None]
            k_indices = xp.arange(key_start, key_end, dtype=xp.int64)[None, :]
            max_keys = q_indices + alignment_shift
            valid = (
                (k_indices <= max_keys)
                & (k_indices >= (max_keys - key_slots + 1))
            )[None, None, :, :]
            probs_chunk = _masked_softmax_forward(
                scores, valid, logit_multiplier=(1.0 / score_prescale)
            )
            del scores, k_score, k_heads, k_span

            v_span = xp.take(v[:, key_token_slice, :, :], kv_map, axis=2)
            v_heads = v_span.transpose(0, 2, 1, 3)
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
                chunk_caches.append(
                    (
                        query_residue,
                        phase_start,
                        phase_end,
                        key_residue,
                        key_start,
                        key_end,
                        _cache_probs_for_backward(probs_chunk, q.dtype),
                        valid,
                        alignment_shift,
                    )
                )

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
    kv_map = cache["kv_head_indices"]
    kv_map_host = cache["kv_head_indices_host"]

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
        query_residue,
        phase_start,
        phase_end,
        key_residue,
        key_start,
        key_end,
        probs_chunk,
        valid,
        alignment_shift,
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

    Global anchors are absolute positions ``offset + n * stride``.  Every query
    sees anchors at or before its own position and, when ``include_current`` is
    true, also sees its current exact token whenever that token is not already
    an anchor.  Unlike the generic indexed path, anchor K/V tensors are gathered
    only once per head group and reused by all queries.
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

    # stride=1, offset=0 is ordinary full causal attention.  Reuse the already
    # optimized contiguous local kernel instead of materializing a full anchor
    # implementation here.
    if stride == 1:
        if offset != 0:
            raise ValueError("stride=1 requires offset=0")
        if return_cache:
            context, local_cache = local_window_attention_forward(
                q,
                k,
                v,
                key_length,
                kv_head_indices=kv_map_host,
                scale=scale,
                return_cache=True,
            )
            return context, {"fallback_local": local_cache}
        return local_window_attention_forward(
            q,
            k,
            v,
            key_length,
            kv_head_indices=kv_map_host,
            scale=scale,
            return_cache=False,
        )

    q_heads = q.transpose(0, 2, 1, 3)  # [B,Hq,T,D]
    bf16_attention = is_bfloat16_dtype(q.dtype)
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
            "q": q,
            "k": k,
            "v": v,
            "stride": stride,
            "offset": offset,
            "include_current": include_current,
            "anchors": anchors,
            "probs": None,
            "valid": None,
            "scale": scale,
            "bf16_attention": bf16_attention,
            "kv_head_indices": kv_map,
            "kv_head_indices_host": kv_map_host,
        }

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
    else:
        k_anchor_heads = None

    current_k_heads = None
    if add_current:
        current_k = xp.take(k, kv_map, axis=2)
        current_k_heads = current_k.transpose(0, 2, 1, 3)
        current_k_score = (
            current_k_heads.astype("float32") if bf16_attention else current_k_heads
        )
        current_scores = xp.sum(q_score * current_k_score, axis=-1, keepdims=True)
        score_parts.append(current_scores)

    scores = score_parts[0] if len(score_parts) == 1 else xp.concatenate(score_parts, axis=-1)

    queries = xp.arange(query_length, dtype=xp.int64)[:, None]
    valid_parts = []
    if n_anchors:
        anchor_valid = anchors[None, :] <= queries
        valid_parts.append(anchor_valid)
    if add_current:
        positions = xp.arange(query_length, dtype=xp.int64)
        on_phase = (positions >= offset) & (((positions - offset) % stride) == 0)
        current_valid = (~on_phase)[:, None]
        valid_parts.append(current_valid)
    valid_2d = valid_parts[0] if len(valid_parts) == 1 else xp.concatenate(valid_parts, axis=1)
    valid = valid_2d[None, None, :, :]

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
        v_compute = v_anchor_heads.astype("float32") if bf16_attention else v_anchor_heads
        probs_compute = (
            probs_anchor
            if bf16_attention
            else (
                probs_anchor.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_anchor
            )
        )
        context_heads += xp.matmul(probs_compute, v_compute)
        slot = n_anchors

    if add_current:
        probs_current = probs[..., slot]
        current_v = xp.take(v, kv_map, axis=2)
        current_v_heads = current_v.transpose(0, 2, 1, 3)
        current_v_compute = (
            current_v_heads.astype("float32") if bf16_attention else current_v_heads
        )
        probs_current_compute = (
            probs_current
            if bf16_attention
            else (
                probs_current.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_current
            )
        )
        context_heads += probs_current_compute[..., None] * current_v_compute

    context = context_heads.transpose(0, 2, 1, 3)
    if not return_cache:
        return context

    return context, {
        "q": q,
        "k": k,
        "v": v,
        "stride": stride,
        "offset": offset,
        "include_current": include_current,
        "anchors": anchors,
        "probs": _cache_probs_for_backward(probs, q.dtype),
        "valid": valid,
        "scale": scale,
        "bf16_attention": bf16_attention,
        "kv_head_indices": kv_map,
        "kv_head_indices_host": kv_map_host,
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
    probs_f32 = _restore_cached_probs(probs, bf16_attention)
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
    q_compute = q_heads.astype("float32") if bf16_attention else q_heads
    dcontext_compute = (
        dcontext_heads.astype("float32") if bf16_attention else dcontext_heads
    )
    n_anchors = int(anchors.shape[0])
    add_current = bool(include_current)

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
        current_v_compute = (
            current_v_heads.astype("float32") if bf16_attention else current_v_heads
        )
        dprobs_parts.append(
            xp.sum(dcontext_compute * current_v_compute, axis=-1, keepdims=True)
        )
    dprobs = dprobs_parts[0] if len(dprobs_parts) == 1 else xp.concatenate(dprobs_parts, axis=-1)
    dscores = _softmax_backward(dprobs, probs_f32)
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

    dq_heads = xp.zeros(q_heads.shape, dtype=grad_dtype)
    dk_anchor_heads = None
    dv_anchor_heads = None
    slot = 0
    if n_anchors:
        ds_anchor = dscores_compute[..., :n_anchors]
        k_anchor = xp.take(k[:, offset:k.shape[1]:stride, :, :], kv_map, axis=2)
        k_anchor_heads = k_anchor.transpose(0, 2, 1, 3)
        k_compute = k_anchor_heads.astype("float32") if bf16_attention else k_anchor_heads
        dq_heads += xp.matmul(ds_anchor, k_compute) * scale
        dk_anchor_heads = xp.matmul(ds_anchor.swapaxes(-1, -2), q_compute) * scale

        probs_anchor = probs_f32[..., :n_anchors]
        probs_compute = (
            probs_anchor
            if bf16_attention
            else (
                probs_anchor.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_anchor
            )
        )
        dv_anchor_heads = xp.matmul(probs_compute.swapaxes(-1, -2), dcontext_compute)
        slot = n_anchors

    dk_current_heads = None
    dv_current_heads = None
    if add_current:
        ds_current = dscores_compute[..., slot]
        current_k = xp.take(k, kv_map, axis=2)
        current_k_heads = current_k.transpose(0, 2, 1, 3)
        current_k_compute = (
            current_k_heads.astype("float32") if bf16_attention else current_k_heads
        )
        dq_heads += ds_current[..., None] * current_k_compute * scale
        dk_current_heads = ds_current[..., None] * q_compute * scale

        probs_current = probs_f32[..., slot]
        probs_current_compute = (
            probs_current
            if bf16_attention
            else (
                probs_current.astype(q.dtype, copy=False)
                if is_low_precision_dtype(q.dtype)
                else probs_current
            )
        )
        dv_current_heads = probs_current_compute[..., None] * dcontext_compute

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
            dk[:, offset:k.shape[1]:stride, kvh, :] += dk_native.astype(
                grad_dtype, copy=False
            )
            dv[:, offset:v.shape[1]:stride, kvh, :] += dv_native.astype(
                grad_dtype, copy=False
            )

        if add_current:
            if len(head_group) == 1:
                dk_native = dk_current_heads[:, head_group[0], :, :]
                dv_native = dv_current_heads[:, head_group[0], :, :]
            else:
                dk_native = xp.sum(dk_current_heads[:, head_group, :, :], axis=1)
                dv_native = xp.sum(dv_current_heads[:, head_group, :, :], axis=1)
            dk[:, :, kvh, :] += dk_native.astype(grad_dtype, copy=False)
            dv[:, :, kvh, :] += dv_native.astype(grad_dtype, copy=False)

    dq = dq_heads.transpose(0, 2, 1, 3)
    return dq, dk, dv

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

    valid_scores = query_valid[None, :, None, :, None]
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
    probs_f32 = _restore_cached_probs(probs, bf16_attention)

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
    dcontext_routes = xp.where(valid_q, dcontext_routes, 0.0)
    q_routes = xp.where(valid_q, q_routes, 0.0)

    batch_ids = xp.arange(batch, dtype=xp.int64)[:, None, None, None]
    kv_ids = kv_map[None, None, :, None]
    v_selected = v[batch_ids, key_indices, kv_ids, :]

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
    dscores = _softmax_backward(dprobs, probs_f32)
    dscores = xp.where(valid_q, dscores, 0.0)
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
