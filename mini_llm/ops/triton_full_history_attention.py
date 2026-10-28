"""Native Triton rectangular causal attention for one full-history head.

This module accepts CuPy allocations directly. It never materializes scores,
probabilities, or a [Q,K] mask. Keys are laid out as [history, current], so the
lower-right causal boundary is ``key <= history_length + query``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - optional GPU dependency
    raise RuntimeError("full-history attention requires triton==3.8.0") from exc


_LOG2E = 1.4426950408889634
_HEAD_DIM = 64


class _Pointer:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _pointer(array):
    name = str(array.dtype)
    if name == "bfloat16":
        dtype = tl.bfloat16
    elif name == "float16":
        dtype = tl.float16
    elif name == "float32":
        dtype = tl.float32
    else:
        raise TypeError(f"unsupported full-history Triton dtype {array.dtype}")
    return _Pointer(array, dtype)


def _strides(array):
    return tuple(int(value // array.dtype.itemsize) for value in array.strides)


@dataclass(frozen=True)
class FullHistoryAttentionCache:
    query: object
    key: object
    value: object
    output: object
    lse_log2: object
    history_length: int


@triton.jit
def _forward_kernel(
    q, k, v, output, lse_log2,
    q_st, k_st, v_st, out_st, lse_st,
    query_length, key_length, history_length,
    scale_log2: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr,
    store_lse: tl.constexpr, fast_bf16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    query_mask = offs_m[:, None] < query_length
    q_block = tl.load(
        q + offs_m[:, None] * q_st + offs_d[None, :],
        mask=query_mask, other=0.0,
    )
    if not fast_bf16:
        q_block = q_block.to(tl.float32)

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, 64), tl.float32)
    query_start = pid_m * block_m
    key_end = tl.minimum(
        key_length, history_length + query_start + block_m
    )
    for key_start in tl.range(0, key_end, block_n, num_stages=2):
        key_offsets = key_start + offs_n
        key_mask = key_offsets < key_length
        k_block = tl.load(
            k + key_offsets[:, None] * k_st + offs_d[None, :],
            mask=key_mask[:, None], other=0.0,
        )
        if not fast_bf16:
            k_block = k_block.to(tl.float32)
        if fast_bf16:
            scores = tl.dot(q_block, tl.trans(k_block)) * scale_log2
        else:
            scores = tl.dot(
                q_block, tl.trans(k_block), input_precision="tf32x3"
            ) * scale_log2
        visible = (
            query_mask
            & key_mask[None, :]
            & (key_offsets[None, :] <= history_length + offs_m[:, None])
        )
        scores = tl.where(visible, scores, -float("inf"))
        tile_has_values = tl.sum(visible.to(tl.int32), axis=1) > 0
        tile_max = tl.max(scores, axis=1)
        new_max = tl.where(
            tile_has_values, tl.maximum(row_max, tile_max), row_max
        )
        probabilities = tl.where(
            visible, tl.exp2(scores - new_max[:, None]), 0.0
        )
        alpha = tl.where(tile_has_values, tl.exp2(row_max - new_max), 1.0)
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator *= alpha[:, None]
        v_block = tl.load(
            v + key_offsets[:, None] * v_st + offs_d[None, :],
            mask=key_mask[:, None], other=0.0,
        )
        if fast_bf16:
            accumulator = tl.dot(
                probabilities.to(tl.bfloat16), v_block, accumulator
            )
        else:
            accumulator = tl.dot(
                probabilities, v_block.to(tl.float32), accumulator,
                input_precision="tf32x3",
            )
        row_max = new_max

    nonempty = row_sum > 0.0
    result = tl.where(nonempty[:, None], accumulator / row_sum[:, None], 0.0)
    tl.store(
        output + offs_m[:, None] * out_st + offs_d[None, :],
        result, mask=query_mask,
    )
    if store_lse:
        safe_lse = tl.where(nonempty, row_max + tl.log2(row_sum), 0.0)
        tl.store(
            lse_log2 + offs_m * lse_st,
            safe_lse, mask=offs_m < query_length,
        )


@triton.jit
def _delta_kernel(output, doutput, delta, out_st, do_st, delta_st,
                  query_length, block_m: tl.constexpr):
    offs_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    mask = offs_m[:, None] < query_length
    out = tl.load(
        output + offs_m[:, None] * out_st + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        doutput + offs_m[:, None] * do_st + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    tl.store(
        delta + offs_m * delta_st,
        tl.sum(out * dout, axis=1), mask=offs_m < query_length,
    )


@triton.jit
def _backward_dkdv_kernel(
    q, k, v, doutput, lse_log2, delta, dkey, dvalue,
    q_st, k_st, v_st, do_st, lse_st, delta_st, dk_st, dv_st,
    query_length, key_length, history_length,
    scale: tl.constexpr, scale_log2: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr,
    fast_bf16: tl.constexpr,
):
    key_start = tl.program_id(0) * block_n
    key_offsets = key_start + tl.arange(0, block_n)
    offs_m_base = tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    key_mask = key_offsets < key_length
    k_block = tl.load(
        k + key_offsets[:, None] * k_st + offs_d[None, :],
        mask=key_mask[:, None], other=0.0,
    )
    v_block = tl.load(
        v + key_offsets[:, None] * v_st + offs_d[None, :],
        mask=key_mask[:, None], other=0.0,
    )
    if not fast_bf16:
        k_block = k_block.to(tl.float32)
        v_block = v_block.to(tl.float32)
    dk_acc = tl.zeros((block_n, 64), tl.float32)
    dv_acc = tl.zeros((block_n, 64), tl.float32)

    query_begin = tl.maximum(0, key_start - history_length)
    query_begin = (query_begin // block_m) * block_m
    for query_start in tl.range(query_begin, query_length, block_m, num_stages=2):
        query_offsets = query_start + offs_m_base
        query_mask = query_offsets < query_length
        q_block = tl.load(
            q + query_offsets[:, None] * q_st + offs_d[None, :],
            mask=query_mask[:, None], other=0.0,
        )
        upstream = tl.load(
            doutput + query_offsets[:, None] * do_st + offs_d[None, :],
            mask=query_mask[:, None], other=0.0,
        )
        if fast_bf16:
            upstream = upstream.to(tl.bfloat16)
        else:
            q_block = q_block.to(tl.float32)
            upstream = upstream.to(tl.float32)
        lse = tl.load(
            lse_log2 + query_offsets * lse_st,
            mask=query_mask, other=0.0,
        )
        row_delta = tl.load(
            delta + query_offsets * delta_st,
            mask=query_mask, other=0.0,
        )
        visible = (
            key_mask[:, None]
            & query_mask[None, :]
            & (key_offsets[:, None] <= history_length + query_offsets[None, :])
        )
        if fast_bf16:
            scores_t = tl.dot(k_block, tl.trans(q_block)) * scale_log2
        else:
            scores_t = tl.dot(
                k_block, tl.trans(q_block), input_precision="tf32x3"
            ) * scale_log2
        probabilities_t = tl.where(
            visible, tl.exp2(scores_t - lse[None, :]), 0.0
        )
        if fast_bf16:
            dv_acc = tl.dot(
                probabilities_t.to(tl.bfloat16), upstream, dv_acc
            )
            d_probabilities_t = tl.dot(v_block, tl.trans(upstream))
        else:
            dv_acc = tl.dot(
                probabilities_t, upstream, dv_acc, input_precision="tf32x3"
            )
            d_probabilities_t = tl.dot(
                v_block, tl.trans(upstream), input_precision="tf32x3"
            )
        d_scores_t = probabilities_t * (d_probabilities_t - row_delta[None, :])
        if fast_bf16:
            dk_acc = tl.dot(d_scores_t.to(tl.bfloat16), q_block, dk_acc)
        else:
            dk_acc = tl.dot(
                d_scores_t, q_block, dk_acc, input_precision="tf32x3"
            )

    tl.store(
        dkey + key_offsets[:, None] * dk_st + offs_d[None, :],
        dk_acc * scale, mask=key_mask[:, None],
    )
    tl.store(
        dvalue + key_offsets[:, None] * dv_st + offs_d[None, :],
        dv_acc, mask=key_mask[:, None],
    )


@triton.jit
def _backward_dq_kernel(
    q, k, v, doutput, lse_log2, delta, dquery,
    q_st, k_st, v_st, do_st, lse_st, delta_st, dq_st,
    query_length, key_length, history_length,
    scale: tl.constexpr, scale_log2: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr,
    fast_bf16: tl.constexpr,
):
    query_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    query_mask = query_offsets[:, None] < query_length
    q_block = tl.load(
        q + query_offsets[:, None] * q_st + offs_d[None, :],
        mask=query_mask, other=0.0,
    )
    upstream = tl.load(
        doutput + query_offsets[:, None] * do_st + offs_d[None, :],
        mask=query_mask, other=0.0,
    )
    if fast_bf16:
        upstream = upstream.to(tl.bfloat16)
    else:
        q_block = q_block.to(tl.float32)
        upstream = upstream.to(tl.float32)
    lse = tl.load(
        lse_log2 + query_offsets * lse_st,
        mask=query_offsets < query_length, other=0.0,
    )
    row_delta = tl.load(
        delta + query_offsets * delta_st,
        mask=query_offsets < query_length, other=0.0,
    )
    dq_acc = tl.zeros((block_m, 64), tl.float32)

    query_start = tl.program_id(0) * block_m
    key_end = tl.minimum(
        key_length, history_length + query_start + block_m
    )
    for key_start in tl.range(0, key_end, block_n, num_stages=2):
        key_offsets = key_start + offs_n
        key_mask = key_offsets < key_length
        k_block = tl.load(
            k + key_offsets[:, None] * k_st + offs_d[None, :],
            mask=key_mask[:, None], other=0.0,
        )
        v_block = tl.load(
            v + key_offsets[:, None] * v_st + offs_d[None, :],
            mask=key_mask[:, None], other=0.0,
        )
        if not fast_bf16:
            k_block = k_block.to(tl.float32)
            v_block = v_block.to(tl.float32)
        visible = (
            query_mask
            & key_mask[None, :]
            & (key_offsets[None, :] <= history_length + query_offsets[:, None])
        )
        if fast_bf16:
            scores = tl.dot(q_block, tl.trans(k_block)) * scale_log2
        else:
            scores = tl.dot(
                q_block, tl.trans(k_block), input_precision="tf32x3"
            ) * scale_log2
        probabilities = tl.where(
            visible, tl.exp2(scores - lse[:, None]), 0.0
        )
        if fast_bf16:
            d_probabilities = tl.dot(upstream, tl.trans(v_block))
        else:
            d_probabilities = tl.dot(
                upstream, tl.trans(v_block), input_precision="tf32x3"
            )
        d_scores = probabilities * (d_probabilities - row_delta[:, None])
        if fast_bf16:
            dq_acc = tl.dot(d_scores.to(tl.bfloat16), k_block, dq_acc)
        else:
            dq_acc = tl.dot(
                d_scores, k_block, dq_acc, input_precision="tf32x3"
            )

    tl.store(
        dquery + query_offsets[:, None] * dq_st + offs_d[None, :],
        dq_acc * scale, mask=query_mask,
    )


def _validate(query, key, value):
    if query.ndim != 2 or query.shape[1] != _HEAD_DIM:
        raise ValueError("full-history query must have shape [Q,64]")
    if key.ndim != 2 or key.shape != value.shape or key.shape[1] != _HEAD_DIM:
        raise ValueError("full-history K/V must have matching shape [K,64]")
    if int(key.shape[0]) < int(query.shape[0]):
        raise ValueError("full-history key length must be at least query length")
    if any(int(array.strides[-1]) != int(array.dtype.itemsize)
           for array in (query, key, value)):
        raise ValueError("full-history tensors require contiguous final dimensions")
    supported = {"bfloat16", "float16", "float32"}
    if any(str(array.dtype) not in supported for array in (query, key, value)):
        raise TypeError("full-history Triton tensors must be BF16, FP16, or FP32")


def full_history_attention_forward(query, key, value, *, return_cache=False):
    """Run one compact row of lower-right causal attention."""
    import cupy as cp

    _validate(query, key, value)
    query_length, key_length = int(query.shape[0]), int(key.shape[0])
    history_length = key_length - query_length
    output = cp.empty(query.shape, dtype=cp.float32)
    lse_log2 = cp.empty((query_length,), dtype=cp.float32) if return_cache else output
    if query_length:
        q_s, k_s, v_s, out_s, lse_s = map(
            _strides, (query, key, value, output, lse_log2)
        )
        block_m = block_n = 64
        fast_bf16 = all(
            str(array.dtype) == "bfloat16" for array in (query, key, value)
        )
        _forward_kernel[(triton.cdiv(query_length, block_m),)](
            _pointer(query), _pointer(key), _pointer(value),
            _pointer(output), _pointer(lse_log2),
            q_s[0], k_s[0], v_s[0], out_s[0], lse_s[0],
            query_length=query_length, key_length=key_length,
            history_length=history_length,
            scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
            block_m=block_m, block_n=block_n,
            store_lse=bool(return_cache), fast_bf16=fast_bf16,
            num_warps=4, num_stages=3,
        )
    if not return_cache:
        return output
    return output, FullHistoryAttentionCache(
        query=query, key=key, value=value, output=output,
        lse_log2=lse_log2, history_length=history_length,
    )


def full_history_attention_backward(doutput, cache):
    """Return FP32 dQ/dK/dV without quadratic intermediates."""
    import cupy as cp

    if not isinstance(cache, FullHistoryAttentionCache):
        raise TypeError("full-history backward requires its Triton forward cache")
    query, key, value = cache.query, cache.key, cache.value
    _validate(query, key, value)
    if doutput.shape != query.shape:
        raise ValueError("full-history upstream gradient must match query shape")
    if str(doutput.dtype) not in {"bfloat16", "float16", "float32"}:
        raise TypeError("full-history upstream gradient has unsupported dtype")
    query_length, key_length = int(query.shape[0]), int(key.shape[0])
    dquery = cp.empty(query.shape, dtype=cp.float32)
    dkey = cp.empty(key.shape, dtype=cp.float32)
    dvalue = cp.empty(value.shape, dtype=cp.float32)
    if not query_length:
        return dquery, dkey, dvalue
    delta = cp.empty((query_length,), dtype=cp.float32)
    q_s, k_s, v_s, do_s, out_s, lse_s, delta_s, dq_s, dk_s, dv_s = map(
        _strides,
        (query, key, value, doutput, cache.output, cache.lse_log2,
         delta, dquery, dkey, dvalue),
    )
    block_m = block_n = 64
    fast_bf16 = all(
        str(array.dtype) == "bfloat16" for array in (query, key, value)
    )
    _delta_kernel[(triton.cdiv(query_length, block_m),)](
        _pointer(cache.output), _pointer(doutput), _pointer(delta),
        out_s[0], do_s[0], delta_s[0], query_length=query_length,
        block_m=block_m, num_warps=4,
    )
    _backward_dkdv_kernel[(triton.cdiv(key_length, block_n),)](
        _pointer(query), _pointer(key), _pointer(value), _pointer(doutput),
        _pointer(cache.lse_log2), _pointer(delta),
        _pointer(dkey), _pointer(dvalue),
        q_s[0], k_s[0], v_s[0], do_s[0], lse_s[0], delta_s[0],
        dk_s[0], dv_s[0],
        query_length=query_length, key_length=key_length,
        history_length=int(cache.history_length), scale=_HEAD_DIM ** -0.5,
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m, block_n=block_n, fast_bf16=fast_bf16,
        num_warps=4, num_stages=3,
    )
    _backward_dq_kernel[(triton.cdiv(query_length, block_m),)](
        _pointer(query), _pointer(key), _pointer(value), _pointer(doutput),
        _pointer(cache.lse_log2), _pointer(delta), _pointer(dquery),
        q_s[0], k_s[0], v_s[0], do_s[0], lse_s[0], delta_s[0], dq_s[0],
        query_length=query_length, key_length=key_length,
        history_length=int(cache.history_length), scale=_HEAD_DIM ** -0.5,
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m, block_n=block_n, fast_bf16=fast_bf16,
        num_warps=4, num_stages=3,
    )
    return dquery, dkey, dvalue
