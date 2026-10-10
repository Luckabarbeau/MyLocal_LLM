"""Learned causal block router for long-context retrieval.

The router answers a deliberately narrow question:

    Given the recent hidden-state context available *before* a routing
    boundary, which complete distant history blocks are most relevant?

Compressed block vectors are only used to search.  The selected block IDs are
later expanded back to full-resolution tokens/KV by ``context_blocks``.
"""

import math
import os

import numpy as np

from ..backend import (
    xp,
    BACKEND_NAME,
    is_bfloat16_dtype,
    is_low_precision_dtype,
)
from ..parameter import Parameter
from ..performance_profiler import retrieval_router_detail_scope
from .context_blocks import HistoryBlockPooler, complete_block_count
from .topk import selected_topk_softmax_forward, selected_topk_softmax_backward


_DIRECT_QUERY_POOL_MODULE = None
_DIRECT_QUERY_POOL_DISABLED = False


def _direct_query_pool_enabled(dtype):
    """Whether to use the BF16/CUDA direct learned-query pooling path."""
    raw = os.environ.get("MINI_LLM_DIRECT_QUERY_POOL", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no", ""}
    )


def _get_direct_query_pool_module():
    """Compile the direct BF16 learned-query pooling kernels lazily.

    The forward path deliberately keeps only the compact ``alpha`` tensor in
    FP32. Hidden states and pooled outputs remain BF16, and no
    ``[B,R,W,D]`` activation window is materialized.
    """
    global _DIRECT_QUERY_POOL_MODULE, _DIRECT_QUERY_POOL_DISABLED
    if BACKEND_NAME != "cupy" or _DIRECT_QUERY_POOL_DISABLED:
        return None
    if _DIRECT_QUERY_POOL_MODULE is not None:
        return _DIRECT_QUERY_POOL_MODULE

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
    void query_pool_softmax_bf16(
        const unsigned short* token_scores,
        const long long* route_starts,
        float* alpha,
        int batch, int seq_len, int n_routes, int query_window,
        float score_scale) {
        int row = (int)blockIdx.x;
        int total_rows = batch * n_routes;
        if (row >= total_rows) return;

        int b = row / n_routes;
        int r = row - b * n_routes;
        int start = (int)route_starts[r];
        int first = start - query_window;
        int valid_begin = 0;
        if (first < 0) {
            valid_begin = -first;
            first = 0;
        }

        extern __shared__ float sh[];
        float local_max = -3.402823466e+38F;
        for (int j = valid_begin + threadIdx.x; j < query_window; j += blockDim.x) {
            int token = first + (j - valid_begin);
            if (token >= 0 && token < start && token < seq_len) {
                float s = bf16_to_float(token_scores[(long long)b * seq_len + token]);
                local_max = fmaxf(local_max, s * score_scale);
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
        for (int j = valid_begin + threadIdx.x; j < query_window; j += blockDim.x) {
            int token = first + (j - valid_begin);
            if (token >= 0 && token < start && token < seq_len) {
                float s = bf16_to_float(token_scores[(long long)b * seq_len + token]);
                local_sum += expf(s * score_scale - row_max);
            }
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv_sum = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;

        long long alpha_base = (long long)row * query_window;
        for (int j = threadIdx.x; j < query_window; j += blockDim.x) {
            float a = 0.0f;
            if (j >= valid_begin) {
                int token = first + (j - valid_begin);
                if (token >= 0 && token < start && token < seq_len) {
                    float s = bf16_to_float(token_scores[(long long)b * seq_len + token]);
                    a = expf(s * score_scale - row_max) * inv_sum;
                }
            }
            alpha[alpha_base + j] = a;
        }
    }

    extern "C" __global__
    void query_pool_reduce_bf16(
        const unsigned short* x,
        const long long* route_starts,
        const float* alpha,
        unsigned short* pooled,
        float* pooled_work,
        int batch, int seq_len, int d_model, int n_routes, int query_window) {
        int row = (int)blockIdx.x;
        int d = (int)blockIdx.y * blockDim.x + threadIdx.x;
        int total_rows = batch * n_routes;
        if (row >= total_rows || d >= d_model) return;

        int b = row / n_routes;
        int r = row - b * n_routes;
        int start = (int)route_starts[r];
        int first = start - query_window;
        int valid_begin = 0;
        if (first < 0) {
            valid_begin = -first;
            first = 0;
        }

        long long alpha_base = (long long)row * query_window;
        float acc = 0.0f;
        for (int j = valid_begin; j < query_window; ++j) {
            int token = first + (j - valid_begin);
            if (token >= 0 && token < start && token < seq_len) {
                long long x_idx = ((long long)b * seq_len + token) * d_model + d;
                acc += alpha[alpha_base + j] * bf16_to_float(x[x_idx]);
            }
        }
        long long out_idx = ((long long)row * d_model) + d;
        pooled_work[out_idx] = acc;
        pooled[out_idx] = float_to_bf16(acc);
    }

    // Aggregate the scalar softmax-score gradient for each sequence token:
    //
    //   D_t = sum_r alpha_rt * g_r^T (x_t - y_r)
    //
    // Each token belongs to only a few overlapping causal windows. We scan
    // the compact route list instead of materializing [B,R,W,D] windows or
    // per-window D-dimensional gradients.
    extern "C" __global__
    void query_pool_token_ds_bf16(
        const unsigned short* x,
        const unsigned short* dpooled,
        const float* pooled_work,
        const long long* route_starts,
        const float* alpha,
        float* token_ds,
        int batch, int seq_len, int d_model, int n_routes, int query_window) {
        int bt = (int)blockIdx.x;
        int total_tokens = batch * seq_len;
        if (bt >= total_tokens) return;

        int b = bt / seq_len;
        int token = bt - b * seq_len;
        extern __shared__ float sh[];
        float total_ds = 0.0f;

        for (int r = 0; r < n_routes; ++r) {
            int start = (int)route_starts[r];
            int raw_first = start - query_window;
            int first = raw_first > 0 ? raw_first : 0;
            if (token < first || token >= start) continue;

            int j = token - raw_first;
            if (j < 0 || j >= query_window) continue;

            long long row = (long long)b * n_routes + r;
            long long x_base = ((long long)b * seq_len + token) * d_model;
            long long g_base = row * d_model;
            float local = 0.0f;
            for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
                float xv = bf16_to_float(x[x_base + d]);
                float yv = pooled_work[g_base + d];
                float gv = bf16_to_float(dpooled[g_base + d]);
                local += gv * (xv - yv);
            }
            sh[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    sh[threadIdx.x] += sh[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                total_ds += alpha[row * query_window + j] * sh[0];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) token_ds[bt] = total_ds;
    }

    // Build dx directly. The first term is the ordinary weighted-pooling
    // derivative; the second is the score path using the scalar D_t above.
    extern "C" __global__
    void query_pool_dx_bf16(
        const unsigned short* dpooled,
        const unsigned short* score_param,
        const long long* route_starts,
        const float* alpha,
        const float* token_ds,
        float* dx,
        int batch, int seq_len, int d_model, int n_routes, int query_window,
        float score_scale) {
        long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long total = (long long)batch * seq_len * d_model;
        if (linear >= total) return;

        int d = (int)(linear % d_model);
        long long bt = linear / d_model;
        int token = (int)(bt % seq_len);
        int b = (int)(bt / seq_len);

        float direct = 0.0f;
        for (int r = 0; r < n_routes; ++r) {
            int start = (int)route_starts[r];
            int raw_first = start - query_window;
            int first = raw_first > 0 ? raw_first : 0;
            if (token < first || token >= start) continue;
            int j = token - raw_first;
            if (j < 0 || j >= query_window) continue;

            long long row = (long long)b * n_routes + r;
            float a = alpha[row * query_window + j];
            direct += a * bf16_to_float(dpooled[row * d_model + d]);
        }

        float score_path = token_ds[bt] * bf16_to_float(score_param[d]) * score_scale;
        dx[linear] = direct + score_path;
    }

    // Accumulate dw = scale * sum_t D_t x_t directly in FP32. One block owns
    // each parameter element, so no atomics are needed even though gradients
    // are accumulated across microbatches.
    extern "C" __global__
    void query_pool_wgrad_bf16(
        const unsigned short* x,
        const float* token_ds,
        float* grad_score,
        int batch, int seq_len, int d_model, float score_scale) {
        int d = (int)blockIdx.x;
        if (d >= d_model) return;
        extern __shared__ float sh[];
        int total_tokens = batch * seq_len;
        float local = 0.0f;
        for (int bt = threadIdx.x; bt < total_tokens; bt += blockDim.x) {
            float xv = bf16_to_float(x[(long long)bt * d_model + d]);
            local += token_ds[bt] * xv;
        }
        sh[threadIdx.x] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) grad_score[d] += sh[0] * score_scale;
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=(
                "query_pool_softmax_bf16",
                "query_pool_reduce_bf16",
                "query_pool_token_ds_bf16",
                "query_pool_dx_bf16",
                "query_pool_wgrad_bf16",
            ),
        )
        module.get_function("query_pool_softmax_bf16")
        module.get_function("query_pool_reduce_bf16")
        module.get_function("query_pool_token_ds_bf16")
        module.get_function("query_pool_dx_bf16")
        module.get_function("query_pool_wgrad_bf16")
        _DIRECT_QUERY_POOL_MODULE = module
    except Exception:
        strict = os.environ.get(
            "MINI_LLM_DIRECT_QUERY_POOL_STRICT", "0"
        ).strip().lower()
        if strict not in {"0", "false", "off", "no", ""}:
            raise
        _DIRECT_QUERY_POOL_DISABLED = True
        return None
    return _DIRECT_QUERY_POOL_MODULE


def _direct_learned_query_pool_forward(x, route_starts, score_param, query_window):
    """Direct BF16 learned-query pooling without gathering activation windows.

    Returns ``(pooled, alpha_work, pooled_work)`` when the optimized CUDA path is usable,
    otherwise ``None`` so callers can fall back to the reference path.
    """
    if not _direct_query_pool_enabled(x.dtype):
        return None
    if not is_bfloat16_dtype(score_param.dtype):
        return None
    if x.ndim != 3 or route_starts.ndim != 1:
        return None
    if not x.flags.c_contiguous or not score_param.flags.c_contiguous:
        return None

    batch, seq_len, d_model = map(int, x.shape)
    n_routes = int(route_starts.shape[0])
    query_window = int(query_window)
    if n_routes == 0:
        return (
            xp.empty((batch, 0, d_model), dtype=x.dtype),
            xp.empty((batch, 0, query_window), dtype="float32"),
            xp.empty((batch, 0, d_model), dtype="float32"),
        )
    if n_routes * batch > 2147483647:
        return None

    module = _get_direct_query_pool_module()
    if module is None:
        return None

    # One global projection scores each sequence token exactly once.  This is
    # a regular 2-D BF16 GEMM and therefore stays on the Tensor-Core path.
    with retrieval_router_detail_scope("router.qpool.direct_score_gemm"):
        token_scores = x.reshape(batch * seq_len, d_model) @ score_param
        token_scores = token_scores.reshape(batch, seq_len)
    if not token_scores.flags.c_contiguous:
        token_scores = xp.ascontiguousarray(token_scores)
    if not route_starts.flags.c_contiguous:
        route_starts = xp.ascontiguousarray(route_starts)

    alpha_work = xp.empty((batch, n_routes, query_window), dtype="float32")
    pooled = xp.empty((batch, n_routes, d_model), dtype=x.dtype)
    # Keep the tiny [B,R,D] FP32 pooled value for the direct backward. This
    # preserves the exact FP32 softmax correction without retaining the huge
    # [B,R,W,D] activation windows.
    pooled_work = xp.empty((batch, n_routes, d_model), dtype="float32")

    threads = 256
    rows = batch * n_routes
    with retrieval_router_detail_scope("router.qpool.direct_softmax"):
        module.get_function("query_pool_softmax_bf16")(
            (rows,),
            (threads,),
            (
                token_scores,
                route_starts,
                alpha_work,
                np.int32(batch),
                np.int32(seq_len),
                np.int32(n_routes),
                np.int32(query_window),
                np.float32(1.0 / math.sqrt(d_model)),
            ),
            shared_mem=threads * 4,
        )

    reduce_threads = 128
    d_blocks = (d_model + reduce_threads - 1) // reduce_threads
    with retrieval_router_detail_scope("router.qpool.direct_reduce"):
        module.get_function("query_pool_reduce_bf16")(
            (rows, d_blocks),
            (reduce_threads,),
            (
                x,
                route_starts,
                alpha_work,
                pooled,
                pooled_work,
                np.int32(batch),
                np.int32(seq_len),
                np.int32(d_model),
                np.int32(n_routes),
                np.int32(query_window),
            ),
        )
    return pooled, alpha_work, pooled_work


def _direct_query_pool_backward_enabled():
    raw = os.environ.get(
        "MINI_LLM_DIRECT_QUERY_POOL_BACKWARD", "0"
    ).strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no", ""}


def _direct_learned_query_pool_backward(
    dpooled, cache, score_param, query_window
):
    """Direct BF16 learned-query backward without window tensors/scatters.

    The CUDA path computes the scalar per-token score gradient first, then
    gathers the at-most-few overlapping route contributions directly into dx
    and accumulates the score-vector gradient in FP32. Returns ``None`` when
    the optimized path is not applicable so callers can use the reference
    implementation.
    """
    if not _direct_query_pool_backward_enabled():
        return None
    if not cache.get("direct_forward", False):
        return None
    x = cache.get("x")
    pooled_work = cache.get("pooled_work")
    route_starts = cache.get("route_starts")
    alpha_work = cache.get("alpha_work")
    if x is None or pooled_work is None or route_starts is None or alpha_work is None:
        return None
    if not (is_bfloat16_dtype(x.dtype) and is_bfloat16_dtype(dpooled.dtype)):
        return None
    if not is_bfloat16_dtype(score_param.data.dtype):
        return None
    if score_param.grad.dtype != xp.dtype("float32"):
        return None
    arrays = (x, dpooled, pooled_work, route_starts, alpha_work, score_param.data)
    if any(not array.flags.c_contiguous for array in arrays):
        return None

    batch, seq_len, d_model = map(int, x.shape)
    n_routes = int(route_starts.shape[0])
    query_window = int(query_window)
    if dpooled.shape != (batch, n_routes, d_model):
        return None
    if pooled_work.shape != (batch, n_routes, d_model):
        return None
    if alpha_work.shape != (batch, n_routes, query_window):
        return None

    module = _get_direct_query_pool_module()
    if module is None:
        return None

    token_ds = xp.empty((batch, seq_len), dtype="float32")
    dx = xp.empty((batch, seq_len, d_model), dtype="float32")
    reduce_threads = 128
    with retrieval_router_detail_scope("router.qpool.bwd.direct_token_ds"):
        module.get_function("query_pool_token_ds_bf16")(
            (batch * seq_len,),
            (reduce_threads,),
            (
                x,
                dpooled,
                pooled_work,
                route_starts,
                alpha_work,
                token_ds,
                np.int32(batch),
                np.int32(seq_len),
                np.int32(d_model),
                np.int32(n_routes),
                np.int32(query_window),
            ),
            shared_mem=reduce_threads * 4,
        )

    threads = 256
    total = batch * seq_len * d_model
    blocks = (total + threads - 1) // threads
    with retrieval_router_detail_scope("router.qpool.bwd.direct_dx"):
        module.get_function("query_pool_dx_bf16")(
            (blocks,),
            (threads,),
            (
                dpooled,
                score_param.data,
                route_starts,
                alpha_work,
                token_ds,
                dx,
                np.int32(batch),
                np.int32(seq_len),
                np.int32(d_model),
                np.int32(n_routes),
                np.int32(query_window),
                np.float32(1.0 / math.sqrt(d_model)),
            ),
        )

    wgrad_threads = 256
    with retrieval_router_detail_scope("router.qpool.bwd.direct_wgrad"):
        module.get_function("query_pool_wgrad_bf16")(
            (d_model,),
            (wgrad_threads,),
            (
                x,
                token_ds,
                score_param.grad,
                np.int32(batch),
                np.int32(seq_len),
                np.int32(d_model),
                np.float32(1.0 / math.sqrt(d_model)),
            ),
            shared_mem=wgrad_threads * 4,
        )
    cache["direct_backward"] = True
    return dx


def eligible_route_starts(seq_len, config):
    """Return routing boundaries with enough causally eligible history.

    A route beginning at token ``s`` may only use query tokens ``< s`` and may
    only select history blocks whose end satisfies

        block_end <= s - exclude_recent_tokens.

    Boundaries with fewer than ``top_k_blocks`` candidates are omitted.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return xp.zeros((0,), dtype=xp.int64)

    starts = xp.arange(
        int(config.routing_stride), seq_len, int(config.routing_stride),
        dtype=xp.int64,
    )
    if starts.size == 0:
        return starts

    usable_history = starts - int(config.exclude_recent_tokens)
    candidate_counts = xp.maximum(usable_history, 0) // int(config.history_block_size)
    return starts[candidate_counts >= int(config.top_k_blocks)]


class CausalQueryPooler:
    """Pool a recent causal window immediately preceding each route start."""

    def __init__(
        self,
        d_model,
        query_window,
        strategy="mean",
        rng=None,
        input_std=0.02,
        name="query_pool",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.query_window = int(query_window)
        self.strategy = str(strategy)
        if self.d_model <= 0 or self.query_window <= 0:
            raise ValueError("d_model and query_window must be positive")
        if self.strategy not in {"mean", "last", "learned"}:
            raise ValueError("unsupported query pooling strategy")

        self.score_param = None
        if self.strategy == "learned":
            if rng is None:
                raise ValueError("learned query pooling requires rng")
            data = xp.asarray(
                rng.normal((self.d_model,), std=input_std, dtype=dtype)
            )
            self.score_param = Parameter(data, name=f"{name}.score")

    def parameters(self):
        return [] if self.score_param is None else [self.score_param]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def _window_indices(self, route_starts):
        offsets = xp.arange(self.query_window, dtype=route_starts.dtype)
        raw = route_starts[:, None] - self.query_window + offsets[None, :]
        valid = raw >= 0
        clipped = xp.maximum(raw, 0)
        return clipped, valid

    def forward(self, x, route_starts, window_metadata=None, valid_starts=None):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (batch, seq_len, d_model)")
        if route_starts.ndim != 1:
            raise ValueError("route_starts must be one-dimensional")
        if window_metadata is None:
            with retrieval_router_detail_scope("router.qpool.validate"):
                if route_starts.size:
                    if bool(xp.any(route_starts <= 0)) or bool(
                        xp.any(route_starts > x.shape[1])
                    ):
                        raise ValueError(
                            "route starts must satisfy 0 < start <= seq_len"
                        )

        batch = x.shape[0]
        n_routes = int(route_starts.shape[0])
        if valid_starts is None:
            valid_starts_b = xp.zeros((batch,), dtype=route_starts.dtype)
        else:
            valid_starts_b = xp.asarray(valid_starts, dtype=route_starts.dtype)
            if valid_starts_b.shape != (batch,):
                raise ValueError("valid_starts must have shape (batch,)")
            if bool(xp.any(valid_starts_b < 0)) or bool(
                xp.any(valid_starts_b > x.shape[1])
            ):
                raise ValueError("valid_starts must lie inside the query source")
            if route_starts.size and bool(
                xp.any(valid_starts_b[:, None] >= route_starts[None, :])
            ):
                raise ValueError(
                    "each causal query window must contain at least one valid token"
                )
        if n_routes == 0:
            pooled = xp.zeros((batch, 0, self.d_model), dtype=x.dtype)
            return pooled, {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "indices": xp.zeros((0, self.query_window), dtype=xp.int64),
                "valid": xp.zeros((0, self.query_window), dtype=bool),
            }

        # Sliding means can be evaluated with prefix sums in O(B*T*D + B*R*D)
        # rather than materializing [B,R,W,D] overlapping windows.  This is
        # essential for 0058A where R and W can both be 4096.
        if self.strategy == "mean":
            with retrieval_router_detail_scope("router.qpool.mean_prefix"):
                nominal_starts = xp.maximum(route_starts - self.query_window, 0)
                starts = xp.maximum(
                    nominal_starts[None, :], valid_starts_b[:, None]
                )
                counts_br = route_starts[None, :] - starts
                if bool(xp.any(counts_br <= 0)):
                    raise ValueError("mean query windows must contain at least one token")
                work = (
                    x.astype("float32", copy=False)
                    if is_low_precision_dtype(x.dtype)
                    else x
                )
                prefix = xp.cumsum(work, axis=1)
                batch_ids = xp.arange(batch, dtype=xp.int64)[:, None]
                end_ids = xp.broadcast_to(
                    (route_starts - 1)[None, :], (batch, n_routes)
                )
                end_sum = prefix[batch_ids, end_ids, :]
                before = starts - 1
                safe_before = xp.maximum(before, 0)
                start_sum = prefix[batch_ids, safe_before, :]
                start_sum = xp.where(
                    (before >= 0)[..., None], start_sum, 0.0
                )
                counts = counts_br.astype(work.dtype, copy=False)[..., None]
                pooled_work = (end_sum - start_sum) / counts
                pooled = (
                    pooled_work.astype(x.dtype, copy=False)
                    if is_low_precision_dtype(x.dtype)
                    else pooled_work
                )
            return pooled, {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "starts": starts,
                "counts": counts_br,
                "prefix_mean": True,
            }

        if window_metadata is None:
            with retrieval_router_detail_scope("router.qpool.indices"):
                indices, valid = self._window_indices(route_starts)
                if valid_starts is not None:
                    valid = valid[None, :, :] & (
                        indices[None, :, :] >= valid_starts_b[:, None, None]
                    )
        else:
            indices, valid = window_metadata
            expected_indices = (n_routes, self.query_window)
            if indices.shape != expected_indices or valid.shape not in {
                expected_indices, (batch, n_routes, self.query_window)
            }:
                raise ValueError("window_metadata has incompatible shape")

        if self.strategy == "last":
            pooled = x[:, route_starts - 1, :]
            cache = {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "indices": indices,
                "valid": valid,
            }
            return pooled, cache

        # The optimized learned path computes each token score once over
        # [B,T,D] and pools windows directly from x. The compact FP32 pooled
        # value is retained for the optional 0048C direct backward; the old
        # indices/valid metadata remains available for the reference fallback.
        if self.strategy == "learned" and valid_starts is None:
            direct = _direct_learned_query_pool_forward(
                x, route_starts, self.score_param.data, self.query_window
            )
            if direct is not None:
                pooled, alpha_work, pooled_work = direct
                cache = {
                    "x_shape": x.shape,
                    "x": x,
                    "route_starts": route_starts,
                    "indices": indices,
                    "valid": valid,
                    "alpha_work": alpha_work,
                    "pooled_work": pooled_work,
                    "scale": 1.0 / math.sqrt(self.d_model),
                    "direct_forward": True,
                }
                return pooled, cache

        with retrieval_router_detail_scope("router.qpool.gather"):
            windows = x[:, indices, :]  # (B, R, W, D)
            valid_f = valid.astype(windows.dtype, copy=False)

        scale = 1.0 / math.sqrt(self.d_model)
        with retrieval_router_detail_scope("router.qpool.score_gemm"):
            windows_2d = windows.reshape(-1, self.d_model)
            scores = (windows_2d @ self.score_param.data).reshape(
                batch, n_routes, self.query_window
            ) * scale
        with retrieval_router_detail_scope("router.qpool.softmax"):
            scores_work = (
                scores.astype("float32", copy=False)
                if is_low_precision_dtype(scores.dtype)
                else scores
            )
            neg_inf = xp.asarray(-xp.inf, dtype=scores_work.dtype)
            valid_b = valid if valid.ndim == 3 else valid[None, :, :]
            scores_work = xp.where(valid_b, scores_work, neg_inf)
            max_scores = xp.max(scores_work, axis=2, keepdims=True)
            exp_scores = xp.where(
                valid_b, xp.exp(scores_work - max_scores), 0.0
            )
            alpha_work = exp_scores / xp.sum(exp_scores, axis=2, keepdims=True)
            alpha = (
                alpha_work.astype(x.dtype, copy=False)
                if is_low_precision_dtype(x.dtype)
                else alpha_work
            )
        with retrieval_router_detail_scope("router.qpool.reduce"):
            pooled = xp.sum(windows * alpha[..., None], axis=2)
        cache = {
            "x_shape": x.shape,
            "x": x,
            "route_starts": route_starts,
            "indices": indices,
            "valid": valid,
            # Re-gather the query windows during backward instead of keeping
            # the large [B,R,W,D] tensor alive for the whole layer backward.
            "alpha_work": alpha_work,
            "scale": scale,
        }
        return pooled, cache

    def backward(self, dpooled, cache):
        x_shape = tuple(cache["x_shape"])
        batch, _, d_model = x_shape
        route_starts = cache["route_starts"]
        n_routes = int(route_starts.shape[0])
        if dpooled.shape != (batch, n_routes, d_model):
            raise ValueError("dpooled has incompatible shape")

        grad_dtype = (
            "float32" if is_low_precision_dtype(dpooled.dtype) else dpooled.dtype
        )
        if n_routes == 0:
            return xp.zeros(x_shape, dtype=grad_dtype)

        if self.strategy == "mean" and cache.get("prefix_mean", False):
            with retrieval_router_detail_scope("router.qpool.bwd.mean_prefix"):
                seq_len = int(x_shape[1])
                starts = cache["starts"].astype(xp.int64, copy=False)
                ends = route_starts.astype(xp.int64, copy=False)
                counts = cache["counts"].astype(grad_dtype, copy=False)
                contrib = dpooled.astype(grad_dtype, copy=False) / counts[..., None]
                diff = xp.zeros((batch, seq_len + 1, d_model), dtype=grad_dtype)
                # Route boundaries are unique, while clipped starts may coincide
                # for very short histories. add.at handles both cases without a
                # [B,R,W,D] temporary. Batch is normally one for long context.
                for b in range(batch):
                    xp.add.at(diff[b], starts[b], contrib[b])
                    xp.add.at(diff[b], ends, -contrib[b])
                dx = xp.cumsum(diff[:, :seq_len, :], axis=1)
            return dx

        if self.strategy == "learned":
            direct_dx = _direct_learned_query_pool_backward(
                dpooled, cache, self.score_param, self.query_window
            )
            if direct_dx is not None:
                return direct_dx

        with retrieval_router_detail_scope("router.qpool.bwd.alloc"):
            dx = xp.zeros(x_shape, dtype=grad_dtype)

        if self.strategy == "last":
            batch_ids = xp.broadcast_to(
                xp.arange(batch)[:, None], (batch, n_routes)
            )
            token_ids = xp.broadcast_to(
                (route_starts - 1)[None, :], (batch, n_routes)
            )
            xp.add.at(
                dx,
                (batch_ids, token_ids),
                dpooled.astype(grad_dtype, copy=False),
            )
            return dx

        indices = cache["indices"]
        valid = cache["valid"]
        with retrieval_router_detail_scope("router.qpool.bwd.gather"):
            windows = cache["x"][:, indices, :]
            alpha_work = cache["alpha_work"]
            scale = cache["scale"]
        with retrieval_router_detail_scope("router.qpool.bwd.direct"):
            alpha_compute = (
                alpha_work.astype(dpooled.dtype, copy=False)
                if is_low_precision_dtype(dpooled.dtype)
                else alpha_work
            )
            dwindow = alpha_compute[..., None] * dpooled[:, :, None, :]

        with retrieval_router_detail_scope("router.qpool.bwd.softmax"):
            dpooled_work = dpooled.astype("float32", copy=False)
            windows_work = windows.astype("float32", copy=False)
            dalpha = xp.sum(
                windows_work * dpooled_work[:, :, None, :], axis=-1
            )
            correction = xp.sum(alpha_work * dalpha, axis=2, keepdims=True)
            dscores = alpha_work * (dalpha - correction)
            valid_b = valid if valid.ndim == 3 else valid[None, :, :]
            dscores = xp.where(valid_b, dscores, 0.0)

        with retrieval_router_detail_scope("router.qpool.bwd.score_path"):
            score_vector_work = self.score_param.data.astype(
                "float32", copy=False
            )
            score_path = dscores[..., None] * score_vector_work * scale
            dwindow += score_path.astype(dwindow.dtype, copy=False)
        with retrieval_router_detail_scope("router.qpool.bwd.score_wgrad"):
            grad_score = xp.sum(
                dscores[..., None] * windows_work, axis=(0, 1, 2)
            ) * scale
            self.score_param.grad += grad_score

        with retrieval_router_detail_scope("router.qpool.bwd.scatter"):
            flat_indices = xp.broadcast_to(
                indices[None, :, :], (batch, n_routes, self.query_window)
            ).reshape(batch, -1)
            values = dwindow.reshape(batch, -1, d_model)
            batch_ids = xp.broadcast_to(
                xp.arange(batch)[:, None], flat_indices.shape
            )
            xp.add.at(
                dx,
                (batch_ids, flat_indices),
                values.astype(grad_dtype, copy=False),
            )
        return dx


class ContextRouter:
    """Select distant history blocks from the current causal working state."""

    def __init__(
        self,
        d_model,
        config,
        rng,
        input_std=0.02,
        name="context_router",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.config = config
        self.router_dim = int(config.router_dim)
        self.num_queries = int(config.num_queries)

        self.query_pooler = CausalQueryPooler(
            d_model=self.d_model,
            query_window=config.query_window,
            strategy=config.query_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.query_pool",
            dtype=dtype,
        )
        self.history_pooler = HistoryBlockPooler(
            d_model=self.d_model,
            block_size=config.history_block_size,
            strategy=config.history_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.history_pool",
            dtype=dtype,
        )

        query_data = xp.asarray(
            rng.normal(
                (self.d_model, self.num_queries * self.router_dim),
                std=input_std,
                dtype=dtype,
            )
        )
        history_data = xp.asarray(
            rng.normal(
                (self.d_model, self.router_dim), std=input_std, dtype=dtype
            )
        )
        self.W_query = Parameter(query_data, name=f"{name}.W_query")
        self.W_history = Parameter(history_data, name=f"{name}.W_history")
        # Routing geometry depends only on sequence length and immutable
        # ContextRouterConfig values. Reuse these small device tensors across
        # microbatches instead of rebuilding aranges, masks and query-window
        # indices 128 times per optimizer update.
        self._routing_metadata_cache = {}

    def _routing_metadata(self, seq_len):
        seq_len = int(seq_len)
        cached = self._routing_metadata_cache.get(seq_len)
        if cached is not None:
            return cached

        route_starts = eligible_route_starts(seq_len, self.config)
        query_indices, query_valid = self.query_pooler._window_indices(route_starts)
        n_blocks = complete_block_count(seq_len, self.config.history_block_size)
        block_ends = (
            xp.arange(n_blocks, dtype=route_starts.dtype) + 1
        ) * int(self.config.history_block_size)
        candidate_mask = block_ends[None, :] <= (
            route_starts[:, None] - int(self.config.exclude_recent_tokens)
        )
        cached = {
            "route_starts": route_starts,
            "query_indices": query_indices,
            "query_valid": query_valid,
            "candidate_mask": candidate_mask,
            "n_blocks": int(n_blocks),
        }
        self._routing_metadata_cache[seq_len] = cached
        return cached

    def parameters(self):
        return (
            [self.W_query, self.W_history]
            + self.query_pooler.parameters()
            + self.history_pooler.parameters()
        )

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (batch, seq_len, d_model)")

        batch, seq_len, _ = x.shape
        with retrieval_router_detail_scope("router.route_starts"):
            routing_metadata = self._routing_metadata(seq_len)
            route_starts = routing_metadata["route_starts"]
        with retrieval_router_detail_scope("router.qpool.forward"):
            query_pooled, query_cache = self.query_pooler.forward(
                x,
                route_starts,
                window_metadata=(
                    routing_metadata["query_indices"],
                    routing_metadata["query_valid"],
                ),
            )
        with retrieval_router_detail_scope("router.hpool.forward"):
            history_pooled, history_cache = self.history_pooler.forward(x)
        n_routes = int(route_starts.shape[0])
        n_blocks = int(history_pooled.shape[1])
        if n_blocks != routing_metadata["n_blocks"]:
            raise RuntimeError("cached routing metadata disagrees with history pooling")

        if n_routes == 0:
            shape = (batch, 0, self.num_queries, self.config.top_k_blocks)
            weights = xp.zeros(shape, dtype=x.dtype)
            selected = xp.zeros(shape, dtype=xp.int64)
            cache = {
                "x": x,
                "route_starts": route_starts,
                "query_pooled": query_pooled,
                "history_pooled": history_pooled,
                "query_cache": query_cache,
                "history_cache": history_cache,
                "scores": xp.zeros(
                    (batch, 0, self.num_queries, n_blocks), dtype=x.dtype
                ),
                "candidate_mask": xp.zeros((0, n_blocks), dtype=bool),
                "topk_cache": None,
            }
            return weights, selected, route_starts, cache

        with retrieval_router_detail_scope("router.qproj.forward"):
            query_proj = (
                query_pooled.reshape(-1, self.d_model) @ self.W_query.data
            ).reshape(batch, n_routes, self.num_queries, self.router_dim)
        with retrieval_router_detail_scope("router.hproj.forward"):
            history_proj = (
                history_pooled.reshape(-1, self.d_model) @ self.W_history.data
            ).reshape(batch, n_blocks, self.router_dim)

        query_3d = query_proj.reshape(
            batch, n_routes * self.num_queries, self.router_dim
        )
        history_t = xp.swapaxes(history_proj, 1, 2)

        # CuPy's generic N-D matmul path does not currently understand BF16
        # (ml_dtypes dtype code ``E``), even though its 2-D BF16 GEMM path is
        # supported.  Router score tensors are tiny compared with attention
        # activations, so perform this batched query/history product in FP32
        # for *all* low-precision model dtypes.  This also keeps Top-K logits
        # numerically stable and matches the existing MoE routing convention.
        with retrieval_router_detail_scope("router.score_gemm"):
            if is_low_precision_dtype(query_3d.dtype):
                query_score = query_3d.astype("float32", copy=False)
                history_score_t = history_t.astype("float32", copy=False)
            else:
                query_score = query_3d
                history_score_t = history_t
            scores = xp.matmul(query_score, history_score_t).reshape(
                batch, n_routes, self.num_queries, n_blocks
            ) / math.sqrt(self.router_dim)

        candidate_mask = routing_metadata["candidate_mask"]

        with retrieval_router_detail_scope("router.mask_topk"):
            scores_work = (
                scores.astype("float32", copy=False)
                if is_low_precision_dtype(scores.dtype)
                else scores
            )
            masked_scores = xp.where(
                candidate_mask[None, :, None, :], scores_work, -xp.inf
            )
            weights, selected, topk_cache = selected_topk_softmax_forward(
                masked_scores,
                int(self.config.top_k_blocks),
                output_dtype=x.dtype,
            )

        cache = {
            "x": x,
            "route_starts": route_starts,
            "query_pooled": query_pooled,
            "history_pooled": history_pooled,
            "query_proj": query_proj,
            "history_proj": history_proj,
            "query_cache": query_cache,
            "history_cache": history_cache,
            "scores": scores,
            "candidate_mask": candidate_mask,
            "topk_cache": topk_cache,
        }
        return weights, selected, route_starts, cache

    def backward(self, dweights, cache, dscores_extra=None):
        """Backward through selected routing weights and router representations.

        ``dscores_extra`` is an optional full-score gradient intended for
        auxiliary retrieval losses.  Invalid/non-causal candidate entries are
        always masked to zero.
        """
        x = cache["x"]
        batch = x.shape[0]
        route_starts = cache["route_starts"]
        n_routes = int(route_starts.shape[0])
        history_pooled = cache["history_pooled"]
        n_blocks = int(history_pooled.shape[1])

        if n_routes == 0:
            return xp.zeros_like(x)

        with retrieval_router_detail_scope("router.topk.backward"):
            dscores = selected_topk_softmax_backward(dweights, cache["topk_cache"])
            if dscores_extra is not None:
                if dscores_extra.shape != dscores.shape:
                    raise ValueError(
                        "dscores_extra must have the same shape as full scores"
                    )
                dscores = dscores + dscores_extra.astype(dscores.dtype, copy=False)
            dscores = xp.where(
                cache["candidate_mask"][None, :, None, :], dscores, 0.0
            )

        query_proj = cache["query_proj"]
        history_proj = cache["history_proj"]
        rq = n_routes * self.num_queries
        scale = 1.0 / math.sqrt(self.router_dim)

        dscores_3d = dscores.reshape(batch, rq, n_blocks)
        query_work = query_proj.reshape(batch, rq, self.router_dim).astype(
            dscores.dtype, copy=False
        )
        history_work = history_proj.astype(dscores.dtype, copy=False)
        with retrieval_router_detail_scope("router.score_bwd.query"):
            dquery_proj_work = (dscores_3d @ history_work) * scale
        with retrieval_router_detail_scope("router.score_bwd.history"):
            dhistory_proj_work = (
                xp.swapaxes(dscores_3d, 1, 2) @ query_work
            ) * scale

        compute_dtype = x.dtype
        dquery_proj = dquery_proj_work.astype(compute_dtype, copy=False).reshape(
            batch, n_routes, self.num_queries * self.router_dim
        )
        dhistory_proj = dhistory_proj_work.astype(compute_dtype, copy=False)

        with retrieval_router_detail_scope("router.qproj.backward"):
            query_pooled = cache["query_pooled"]
            query_2d = query_pooled.reshape(-1, self.d_model)
            dq_2d = dquery_proj.reshape(-1, self.num_queries * self.router_dim)
            self.W_query.grad += query_2d.T @ dq_2d
            dquery_pooled = (dq_2d @ self.W_query.data.T).reshape(
                query_pooled.shape
            )

        with retrieval_router_detail_scope("router.hproj.backward"):
            history_2d = history_pooled.reshape(-1, self.d_model)
            dh_2d = dhistory_proj.reshape(-1, self.router_dim)
            self.W_history.grad += history_2d.T @ dh_2d
            dhistory_pooled = (dh_2d @ self.W_history.data.T).reshape(
                history_pooled.shape
            )

        with retrieval_router_detail_scope("router.qpool.backward"):
            dx_query = self.query_pooler.backward(
                dquery_pooled, cache["query_cache"]
            )
        with retrieval_router_detail_scope("router.hpool.backward"):
            dx_history = self.history_pooler.backward(
                dhistory_pooled, cache["history_cache"]
            )
        return dx_query + dx_history
