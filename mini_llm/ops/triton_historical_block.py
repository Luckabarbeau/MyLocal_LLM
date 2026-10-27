"""Opt-in Triton forward kernel for deterministic historical block reads.

The kernel preserves the compact 32-token deterministic sweep without
materializing gathered K/V, scores, or probabilities. It uses a two-pass
softmax so the second pass has the global row maximum before accumulating
P @ V. The established CuPy backward remains responsible for the VJP.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - capability failure
    raise RuntimeError(
        "the opt-in historical block kernel requires triton>=3.8"
    ) from exc


_LOG2E = 1.4426950408889634
_HEAD_DIM = 64
_BLOCK_SIZE = 32


@dataclass(frozen=True)
class DeterministicBlockReadCache:
    """Minimal forward state needed by the streaming block-read VJP."""

    output: object
    lse_log2: object
    first_local: int
    selected_tokens: int
    period: int


class _Pointer:
    """Expose a CuPy allocation through Triton's minimal tensor protocol."""

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _pointer(array):
    name = str(array.dtype)
    if name == "bfloat16":
        dtype = tl.bfloat16
    elif name == "bool":
        dtype = tl.int1
    elif name == "float32":
        dtype = tl.float32
    else:
        raise TypeError(f"unsupported historical block dtype {array.dtype}")
    return _Pointer(array, dtype)


def _strides(array):
    return tuple(int(value // array.dtype.itemsize) for value in array.strides)


def _selection_geometry(*, phase, period, source_offset, source_length):
    if period < _BLOCK_SIZE or period % _BLOCK_SIZE:
        raise ValueError("historical block period must be a multiple of 32")
    if phase < 0 or phase % _BLOCK_SIZE:
        raise ValueError("historical block phase must be a nonnegative multiple of 32")
    if source_offset < 0 or source_offset > source_length:
        raise ValueError("invalid historical source interval")
    if source_offset % _BLOCK_SIZE:
        raise ValueError("historical source offset must be 32-token aligned")

    first = phase
    if first < source_offset:
        first += ((source_offset - first + period - 1) // period) * period
    if first >= source_length:
        return None
    block_count = (source_length - 1 - first) // period + 1
    last = first + (block_count - 1) * period
    selected_tokens = (block_count - 1) * _BLOCK_SIZE + min(
        _BLOCK_SIZE, source_length - last
    )
    return first - source_offset, selected_tokens


@triton.jit
def _forward_kernel(
    q,
    k,
    v,
    valid,
    output,
    lse_log2,
    q_sb,
    q_st,
    k_sb,
    k_st,
    v_sb,
    v_st,
    valid_sb,
    valid_st,
    out_sb,
    out_st,
    lse_sb,
    lse_st,
    query_length,
    history_length,
    first_local,
    selected_tokens,
    period,
    scale_log2: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    store_lse: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    query_mask = offs_m[:, None] < query_length
    q_block = tl.load(
        q + batch * q_sb + offs_m[:, None] * q_st + offs_d[None, :],
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
        selected_offsets = selected_start + offs_n
        source_offsets = (
            first_local
            + (selected_offsets // 32) * period
            + selected_offsets % 32
        )
        key_mask = (
            (selected_offsets < selected_tokens)
            & (source_offsets < history_length)
        )
        key_valid = tl.load(
            valid + batch * valid_sb + source_offsets * valid_st,
            mask=key_mask,
            other=0,
        ).to(tl.int1)
        k_block = tl.load(
            k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(
            q_block, tl.trans(k_block), input_precision="tf32x3"
        ) * scale_log2
        attention_mask = query_mask & key_mask[None, :] & key_valid[None, :]
        scores = tl.where(attention_mask, scores, -float("inf"))
        tile_has_values = tl.sum(attention_mask.to(tl.int32), axis=1) > 0
        tile_max = tl.max(scores, axis=1)
        row_max = tl.where(
            tile_has_values, tl.maximum(row_max, tile_max), row_max
        )

    row_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, 64), tl.float32)
    for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
        selected_offsets = selected_start + offs_n
        source_offsets = (
            first_local
            + (selected_offsets // 32) * period
            + selected_offsets % 32
        )
        key_mask = (
            (selected_offsets < selected_tokens)
            & (source_offsets < history_length)
        )
        key_valid = tl.load(
            valid + batch * valid_sb + source_offsets * valid_st,
            mask=key_mask,
            other=0,
        ).to(tl.int1)
        k_block = tl.load(
            k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(
            q_block, tl.trans(k_block), input_precision="tf32x3"
        ) * scale_log2
        attention_mask = query_mask & key_mask[None, :] & key_valid[None, :]
        probabilities = tl.where(
            attention_mask, tl.exp2(scores - row_max[:, None]), 0.0
        )
        row_sum += tl.sum(probabilities, axis=1)
        v_block = tl.load(
            v + batch * v_sb + source_offsets[:, None] * v_st + offs_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(
            probabilities,
            v_block,
            accumulator,
            input_precision="tf32x3",
        )

    nonempty = row_sum > 0.0
    result = tl.where(nonempty[:, None], accumulator / row_sum[:, None], 0.0)
    tl.store(
        output + batch * out_sb + offs_m[:, None] * out_st + offs_d[None, :],
        result,
        mask=query_mask,
    )
    if store_lse:
        safe_lse = tl.where(
            nonempty, row_max + tl.log2(row_sum), 0.0
        )
        tl.store(
            lse_log2 + batch * lse_sb + offs_m * lse_st,
            safe_lse,
            mask=offs_m < query_length,
        )


@triton.jit
def _backward_delta_kernel(
    output,
    dcontext,
    active,
    gate,
    gate_grad,
    delta,
    out_sb,
    out_st,
    dc_sb,
    dc_st,
    active_sb,
    delta_sb,
    delta_st,
    query_length,
    block_m: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    row_mask = offs_m[:, None] < query_length
    enabled = tl.load(active + batch * active_sb).to(tl.int1)
    output_block = tl.load(
        output
        + batch * out_sb
        + offs_m[:, None] * out_st
        + offs_d[None, :],
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        dcontext
        + batch * dc_sb
        + offs_m[:, None] * dc_st
        + offs_d[None, :],
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.where(enabled, upstream, 0.0)
    gate_value = tl.load(gate).to(tl.float32)
    output_dot = tl.sum(output_block * upstream, axis=1)
    tl.store(
        delta + batch * delta_sb + offs_m * delta_st,
        output_dot * gate_value,
        mask=offs_m < query_length,
    )
    tl.atomic_add(gate_grad, tl.sum(output_dot, axis=0))


@triton.jit
def _backward_dkdv_kernel(
    q,
    k,
    v,
    valid,
    dcontext,
    active,
    gate,
    lse_log2,
    delta,
    d_k,
    d_v,
    q_sb,
    q_st,
    k_sb,
    k_st,
    v_sb,
    v_st,
    valid_sb,
    valid_st,
    dc_sb,
    dc_st,
    active_sb,
    lse_sb,
    lse_st,
    delta_sb,
    delta_st,
    dk_sb,
    dk_st,
    dv_sb,
    dv_st,
    query_length,
    history_length,
    first_local,
    selected_tokens,
    period,
    scale: tl.constexpr,
    scale_log2: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    batch = tl.program_id(1)
    selected_offsets = pid_n * block_n + tl.arange(0, block_n)
    source_offsets = (
        first_local
        + (selected_offsets // 32) * period
        + selected_offsets % 32
    )
    offs_m_base = tl.arange(0, block_m)
    offs_d = tl.arange(0, 64)
    key_mask = (
        (selected_offsets < selected_tokens)
        & (source_offsets < history_length)
    )
    key_valid = tl.load(
        valid + batch * valid_sb + source_offsets * valid_st,
        mask=key_mask,
        other=0,
    ).to(tl.int1)
    enabled = tl.load(active + batch * active_sb).to(tl.int1)
    k_block = tl.load(
        k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
        mask=key_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    v_block = tl.load(
        v + batch * v_sb + source_offsets[:, None] * v_st + offs_d[None, :],
        mask=key_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    gate_value = tl.load(gate).to(tl.float32)
    dk_acc = tl.zeros((block_n, 64), tl.float32)
    dv_acc = tl.zeros((block_n, 64), tl.float32)

    for query_start in tl.range(0, query_length, block_m, num_stages=2):
        offs_m = query_start + offs_m_base
        query_mask = offs_m[:, None] < query_length
        q_block = tl.load(
            q + batch * q_sb + offs_m[:, None] * q_st + offs_d[None, :],
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        upstream = tl.load(
            dcontext
            + batch * dc_sb
            + offs_m[:, None] * dc_st
            + offs_d[None, :],
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        upstream = tl.where(enabled, upstream * gate_value, 0.0)
        lse = tl.load(
            lse_log2 + batch * lse_sb + offs_m * lse_st,
            mask=offs_m < query_length,
            other=0.0,
        )
        row_delta = tl.load(
            delta + batch * delta_sb + offs_m * delta_st,
            mask=offs_m < query_length,
            other=0.0,
        )
        attention_mask = (
            key_mask[:, None]
            & key_valid[:, None]
            & (offs_m[None, :] < query_length)
            & enabled
        )
        scores_t = tl.dot(
            k_block, tl.trans(q_block), input_precision="tf32x3"
        ) * scale_log2
        scores_t = tl.where(attention_mask, scores_t, -float("inf"))
        probabilities_t = tl.where(
            attention_mask, tl.exp2(scores_t - lse[None, :]), 0.0
        )
        dv_acc = tl.dot(
            probabilities_t,
            upstream,
            dv_acc,
            input_precision="tf32x3",
        )
        d_probabilities_t = tl.dot(
            v_block, tl.trans(upstream), input_precision="tf32x3"
        )
        d_scores_t = probabilities_t * (
            d_probabilities_t - row_delta[None, :]
        )
        dk_acc = tl.dot(
            d_scores_t,
            q_block,
            dk_acc,
            input_precision="tf32x3",
        )

    dk_ptrs = (
        d_k + batch * dk_sb + source_offsets[:, None] * dk_st + offs_d[None, :]
    )
    dv_ptrs = (
        d_v + batch * dv_sb + source_offsets[:, None] * dv_st + offs_d[None, :]
    )
    store_mask = key_mask[:, None]
    old_dk = tl.load(dk_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    old_dv = tl.load(dv_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(dk_ptrs, old_dk + dk_acc * scale, mask=store_mask)
    tl.store(dv_ptrs, old_dv + dv_acc, mask=store_mask)


@triton.jit
def _backward_dq_clear_kernel(
    q,
    k,
    v,
    valid,
    dcontext,
    active,
    gate,
    lse_log2,
    delta,
    d_q,
    q_sb,
    q_st,
    k_sb,
    k_st,
    v_sb,
    v_st,
    valid_sb,
    valid_st,
    dc_sb,
    dc_st,
    active_sb,
    lse_sb,
    lse_st,
    delta_sb,
    delta_st,
    dq_sb,
    dq_st,
    query_length,
    history_length,
    first_local,
    selected_tokens,
    period,
    scale: tl.constexpr,
    scale_log2: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    query_mask = offs_m[:, None] < query_length
    enabled = tl.load(active + batch * active_sb).to(tl.int1)
    q_block = tl.load(
        q + batch * q_sb + offs_m[:, None] * q_st + offs_d[None, :],
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        dcontext
        + batch * dc_sb
        + offs_m[:, None] * dc_st
        + offs_d[None, :],
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)
    gate_value = tl.load(gate).to(tl.float32)
    upstream = tl.where(enabled, upstream * gate_value, 0.0)
    lse = tl.load(
        lse_log2 + batch * lse_sb + offs_m * lse_st,
        mask=offs_m < query_length,
        other=0.0,
    )
    row_delta = tl.load(
        delta + batch * delta_sb + offs_m * delta_st,
        mask=offs_m < query_length,
        other=0.0,
    )
    dq_acc = tl.zeros((block_m, 64), tl.float32)

    for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
        selected_offsets = selected_start + offs_n
        source_offsets = (
            first_local
            + (selected_offsets // 32) * period
            + selected_offsets % 32
        )
        key_mask = (
            (selected_offsets < selected_tokens)
            & (source_offsets < history_length)
        )
        key_valid = tl.load(
            valid + batch * valid_sb + source_offsets * valid_st,
            mask=key_mask,
            other=0,
        ).to(tl.int1)
        k_block = tl.load(
            k
            + batch * k_sb
            + source_offsets[:, None] * k_st
            + offs_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        v_block = tl.load(
            v
            + batch * v_sb
            + source_offsets[:, None] * v_st
            + offs_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        attention_mask = (
            query_mask
            & key_mask[None, :]
            & key_valid[None, :]
            & enabled
        )
        scores = tl.dot(
            q_block, tl.trans(k_block), input_precision="tf32x3"
        ) * scale_log2
        scores = tl.where(attention_mask, scores, -float("inf"))
        probabilities = tl.where(
            attention_mask, tl.exp2(scores - lse[:, None]), 0.0
        )
        d_probabilities = tl.dot(
            upstream, tl.trans(v_block), input_precision="tf32x3"
        )
        d_scores = probabilities * (
            d_probabilities - row_delta[:, None]
        )
        dq_acc = tl.dot(
            d_scores,
            k_block,
            dq_acc,
            input_precision="tf32x3",
        )

    dq_ptrs = (
        d_q + batch * dq_sb + offs_m[:, None] * dq_st + offs_d[None, :]
    )
    old_dq = tl.load(dq_ptrs, mask=query_mask, other=0.0).to(tl.float32)
    tl.store(dq_ptrs, old_dq + dq_acc * scale, mask=query_mask)
    tl.store(
        dcontext
        + batch * dc_sb
        + offs_m[:, None] * dc_st
        + offs_d[None, :],
        0.0,
        mask=query_mask & enabled,
    )


def deterministic_block_read_forward(
    query,
    key,
    value,
    valid,
    *,
    phase,
    period,
    source_offset,
    source_length,
    return_cache=False,
):
    """Compute one deterministic historical read without gathered intermediates."""
    import cupy as cp

    if query.ndim != 3 or query.shape[-1] != _HEAD_DIM:
        raise ValueError("historical block query must have shape [B,Q,64]")
    if key.shape != value.shape or key.ndim != 3 or key.shape[-1] != _HEAD_DIM:
        raise ValueError("historical block K/V must have matching shape [B,H,64]")
    if valid.shape != key.shape[:2] or query.shape[0] != key.shape[0]:
        raise ValueError("historical block batch or validity shape mismatch")
    if int(key.shape[1]) != int(source_length - source_offset):
        raise ValueError("historical K/V length does not match source coordinates")
    arrays = (query, key, value, valid)
    if any(int(array.strides[-1]) != int(array.dtype.itemsize) for array in arrays):
        raise ValueError("historical block tensors require contiguous final dimensions")

    geometry = _selection_geometry(
        phase=int(phase),
        period=int(period),
        source_offset=int(source_offset),
        source_length=int(source_length),
    )
    if geometry is None:
        output = cp.zeros(query.shape, dtype=cp.float32)
        if not return_cache:
            return output
        lse_log2 = cp.zeros(query.shape[:2], dtype=cp.float32)
        return output, DeterministicBlockReadCache(
            output=output,
            lse_log2=lse_log2,
            first_local=0,
            selected_tokens=0,
            period=int(period),
        )
    first_local, selected_tokens = geometry

    output = cp.empty(query.shape, dtype=cp.float32)
    lse_log2 = (
        cp.empty(query.shape[:2], dtype=cp.float32)
        if return_cache else output
    )
    q_s, k_s, v_s, valid_s, out_s = map(
        _strides, (query, key, value, valid, output)
    )
    lse_s = _strides(lse_log2)
    block_m = block_n = 64
    _forward_kernel[(triton.cdiv(int(query.shape[1]), block_m), int(query.shape[0]))](
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(output),
        _pointer(lse_log2),
        q_s[0],
        q_s[1],
        k_s[0],
        k_s[1],
        v_s[0],
        v_s[1],
        valid_s[0],
        valid_s[1],
        out_s[0],
        out_s[1],
        lse_s[0],
        lse_s[1],
        query_length=int(query.shape[1]),
        history_length=int(key.shape[1]),
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=block_n,
        store_lse=bool(return_cache),
        num_warps=4,
        num_stages=3,
    )
    if not return_cache:
        return output
    return output, DeterministicBlockReadCache(
        output=output,
        lse_log2=lse_log2,
        first_local=int(first_local),
        selected_tokens=int(selected_tokens),
        period=int(period),
    )


def deterministic_block_read_backward(
    dcontext,
    query,
    key,
    value,
    valid,
    active,
    gate,
    gate_grad,
    cache,
    *,
    dquery,
    dkey,
    dvalue,
):
    """Accumulate one deterministic historical-read VJP in place.

    ``dcontext`` is cleared only for active rows after all K/V gradients have
    consumed it. ``gate_grad``, ``dquery``, ``dkey`` and ``dvalue`` are additive
    destinations, matching the shared historical fan-in contract.
    """
    import cupy as cp

    if not isinstance(cache, DeterministicBlockReadCache):
        raise TypeError("historical block backward requires its forward cache")
    if query.ndim != 3 or query.shape[-1] != _HEAD_DIM:
        raise ValueError("historical block query must have shape [B,Q,64]")
    if key.shape != value.shape or key.ndim != 3 or key.shape[-1] != _HEAD_DIM:
        raise ValueError("historical block K/V must have matching shape [B,H,64]")
    if valid.shape != key.shape[:2] or query.shape[0] != key.shape[0]:
        raise ValueError("historical block batch or validity shape mismatch")
    if dcontext.shape != query.shape or dquery.shape != query.shape:
        raise ValueError("historical block dcontext/dquery must match query")
    if dkey.shape != key.shape or dvalue.shape != value.shape:
        raise ValueError("historical block dkey/dvalue must match K/V")
    if cache.output.shape != query.shape or cache.lse_log2.shape != query.shape[:2]:
        raise ValueError("historical block cache shape mismatch")
    if int(cache.period) < _BLOCK_SIZE or int(cache.period) % _BLOCK_SIZE:
        raise ValueError("invalid cached historical block period")
    if int(cache.first_local) < 0 or int(cache.selected_tokens) < 0:
        raise ValueError("invalid cached historical block geometry")
    if str(key.dtype) != "float32" or str(value.dtype) != "float32":
        raise TypeError("historical block backward requires FP32 K/V")
    if any(str(array.dtype) != "float32" for array in (dquery, dkey, dvalue)):
        raise TypeError("historical block gradient destinations must be FP32")
    if str(cache.output.dtype) != "float32" or str(cache.lse_log2.dtype) != "float32":
        raise TypeError("historical block output and LSE cache must be FP32")
    if str(valid.dtype) != "bool" or str(active.dtype) != "bool":
        raise TypeError("historical block validity and active masks must be bool")
    if str(gate.dtype) != "float32" or str(gate_grad.dtype) != "float32":
        raise TypeError("historical block gate and gate gradient must be FP32")
    if int(gate.size) != 1 or int(gate_grad.size) != 1:
        raise ValueError("historical block gate arguments must contain one scalar")
    if int(active.size) != int(query.shape[0]):
        raise ValueError("historical block active mask must contain one value per batch row")
    arrays = (
        query, key, value, dcontext, dquery, dkey, dvalue, cache.output
    )
    if any(int(array.strides[-1]) != int(array.dtype.itemsize) for array in arrays):
        raise ValueError("historical block tensors require contiguous final dimensions")

    active_flat = active.reshape(-1)
    delta = cp.empty(query.shape[:2], dtype=cp.float32)
    q_s, k_s, v_s, valid_s, dc_s, active_s, out_s, lse_s, delta_s = map(
        _strides,
        (
            query, key, value, valid, dcontext, active_flat,
            cache.output, cache.lse_log2, delta,
        ),
    )
    dq_s, dk_s, dv_s = map(_strides, (dquery, dkey, dvalue))
    query_length = int(query.shape[1])
    history_length = int(key.shape[1])
    block_m = block_n = 64
    dkdv_block_m = 32
    dq_block_n = 32
    grid_m = triton.cdiv(query_length, block_m)

    _backward_delta_kernel[(grid_m, int(query.shape[0]))](
        _pointer(cache.output),
        _pointer(dcontext),
        _pointer(active_flat),
        _pointer(gate),
        _pointer(gate_grad),
        _pointer(delta),
        out_s[0],
        out_s[1],
        dc_s[0],
        dc_s[1],
        active_s[0],
        delta_s[0],
        delta_s[1],
        query_length=query_length,
        block_m=block_m,
        num_warps=4,
    )
    if int(cache.selected_tokens):
        _backward_dkdv_kernel[(
            triton.cdiv(int(cache.selected_tokens), block_n),
            int(query.shape[0]),
        )](
            _pointer(query),
            _pointer(key),
            _pointer(value),
            _pointer(valid),
            _pointer(dcontext),
            _pointer(active_flat),
            _pointer(gate),
            _pointer(cache.lse_log2),
            _pointer(delta),
            _pointer(dkey),
            _pointer(dvalue),
            q_s[0],
            q_s[1],
            k_s[0],
            k_s[1],
            v_s[0],
            v_s[1],
            valid_s[0],
            valid_s[1],
            dc_s[0],
            dc_s[1],
            active_s[0],
            lse_s[0],
            lse_s[1],
            delta_s[0],
            delta_s[1],
            dk_s[0],
            dk_s[1],
            dv_s[0],
            dv_s[1],
            query_length=query_length,
            history_length=history_length,
            first_local=int(cache.first_local),
            selected_tokens=int(cache.selected_tokens),
            period=int(cache.period),
            scale=_HEAD_DIM ** -0.5,
            scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
            block_m=dkdv_block_m,
            block_n=block_n,
            num_warps=4,
            num_stages=2,
        )
    _backward_dq_clear_kernel[(grid_m, int(query.shape[0]))](
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(dcontext),
        _pointer(active_flat),
        _pointer(gate),
        _pointer(cache.lse_log2),
        _pointer(delta),
        _pointer(dquery),
        q_s[0],
        q_s[1],
        k_s[0],
        k_s[1],
        v_s[0],
        v_s[1],
        valid_s[0],
        valid_s[1],
        dc_s[0],
        dc_s[1],
        active_s[0],
        lse_s[0],
        lse_s[1],
        delta_s[0],
        delta_s[1],
        dq_s[0],
        dq_s[1],
        query_length=query_length,
        history_length=history_length,
        first_local=int(cache.first_local),
        selected_tokens=int(cache.selected_tokens),
        period=int(cache.period),
        scale=_HEAD_DIM ** -0.5,
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=dq_block_n,
        num_warps=4,
        num_stages=3,
    )
    return dcontext, dquery, dkey, dvalue


def prewarm_deterministic_block_read_forward(
    *, query_dtype, query_length, history_length, phase=0, period=384,
    backward=False,
):
    """Compile the generic kernel without executing a historical read.

    Geometry and strides are runtime arguments, so one production-like warmup
    covers aligned source crops, deterministic phases, and compact query lengths
    with the same pointer dtypes. Triton's persistent disk cache can then reuse
    the binary in later processes.
    """
    import cupy as cp

    query_length = int(query_length)
    history_length = int(history_length)
    geometry = _selection_geometry(
        phase=int(phase),
        period=int(period),
        source_offset=0,
        source_length=history_length,
    )
    if query_length <= 0 or geometry is None:
        return
    first_local, selected_tokens = geometry

    query = cp.empty((1,), dtype=cp.dtype(query_dtype))
    key = cp.empty((1,), dtype=cp.float32)
    value = cp.empty((1,), dtype=cp.float32)
    valid = cp.empty((1,), dtype=cp.bool_)
    output = cp.empty((1,), dtype=cp.float32)
    lse_log2 = cp.empty((1,), dtype=cp.float32)
    q_st = 16 * 64
    q_sb = query_length * q_st
    k_st = v_st = 64
    k_sb = v_sb = history_length * 64
    valid_st = 1
    valid_sb = history_length
    out_st = 64
    out_sb = query_length * 64
    block_m = block_n = 64
    dkdv_block_m = 32
    dq_block_n = 32
    _forward_kernel.warmup(
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(output),
        _pointer(lse_log2),
        q_sb,
        q_st,
        k_sb,
        k_st,
        v_sb,
        v_st,
        valid_sb,
        valid_st,
        out_sb,
        out_st,
        query_length,
        1,
        query_length=query_length,
        history_length=history_length,
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=block_n,
        store_lse=bool(backward),
        num_warps=4,
        num_stages=3,
        grid=(triton.cdiv(query_length, block_m), 1),
    )
    if not backward:
        return

    dcontext = cp.empty((1,), dtype=cp.float32)
    active = cp.empty((1,), dtype=cp.bool_)
    gate = cp.empty((1,), dtype=cp.float32)
    gate_grad = cp.empty((1,), dtype=cp.float32)
    delta = cp.empty((1,), dtype=cp.float32)
    dquery = cp.empty((1,), dtype=cp.float32)
    dkey = cp.empty((1,), dtype=cp.float32)
    dvalue = cp.empty((1,), dtype=cp.float32)
    dc_st = dq_st = q_st
    dc_sb = dq_sb = q_sb
    lse_sb = delta_sb = query_length
    lse_st = delta_st = 1

    _backward_delta_kernel.warmup(
        _pointer(output),
        _pointer(dcontext),
        _pointer(active),
        _pointer(gate),
        _pointer(gate_grad),
        _pointer(delta),
        out_sb,
        out_st,
        dc_sb,
        dc_st,
        1,
        delta_sb,
        delta_st,
        query_length=query_length,
        block_m=block_m,
        num_warps=4,
        grid=(triton.cdiv(query_length, block_m), 1),
    )
    _backward_dkdv_kernel.warmup(
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(dcontext),
        _pointer(active),
        _pointer(gate),
        _pointer(lse_log2),
        _pointer(delta),
        _pointer(dkey),
        _pointer(dvalue),
        q_sb,
        q_st,
        k_sb,
        k_st,
        v_sb,
        v_st,
        valid_sb,
        valid_st,
        dc_sb,
        dc_st,
        1,
        lse_sb,
        lse_st,
        delta_sb,
        delta_st,
        k_sb,
        k_st,
        v_sb,
        v_st,
        query_length=query_length,
        history_length=history_length,
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale=_HEAD_DIM ** -0.5,
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=dkdv_block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=2,
        grid=(triton.cdiv(selected_tokens, block_n), 1),
    )
    _backward_dq_clear_kernel.warmup(
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(dcontext),
        _pointer(active),
        _pointer(gate),
        _pointer(lse_log2),
        _pointer(delta),
        _pointer(dquery),
        q_sb,
        q_st,
        k_sb,
        k_st,
        v_sb,
        v_st,
        valid_sb,
        valid_st,
        dc_sb,
        dc_st,
        1,
        lse_sb,
        lse_st,
        delta_sb,
        delta_st,
        dq_sb,
        dq_st,
        query_length=query_length,
        history_length=history_length,
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale=_HEAD_DIM ** -0.5,
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=dq_block_n,
        num_warps=4,
        num_stages=3,
        grid=(triton.cdiv(query_length, block_m), 1),
    )
