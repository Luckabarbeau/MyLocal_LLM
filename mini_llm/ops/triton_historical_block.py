"""Opt-in Triton forward kernel for deterministic historical block reads.

The kernel preserves the compact 32-token deterministic sweep without
materializing gathered K/V, scores, or probabilities. It uses a two-pass
softmax so the second pass has the global row maximum before accumulating
P @ V. The established CuPy backward remains responsible for the VJP.
"""

from __future__ import annotations

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
    history_length,
    first_local,
    selected_tokens,
    period,
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
        return cp.zeros(query.shape, dtype=cp.float32)
    first_local, selected_tokens = geometry

    output = cp.empty(query.shape, dtype=cp.float32)
    q_s, k_s, v_s, valid_s, out_s = map(
        _strides, (query, key, value, valid, output)
    )
    block_m = block_n = 64
    _forward_kernel[(triton.cdiv(int(query.shape[1]), block_m), int(query.shape[0]))](
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(output),
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
        query_length=int(query.shape[1]),
        history_length=int(key.shape[1]),
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=3,
    )
    return output


def prewarm_deterministic_block_read_forward(
    *, query_dtype, query_length, history_length, phase=0, period=384
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
    q_st = 16 * 64
    q_sb = query_length * q_st
    k_st = v_st = 64
    k_sb = v_sb = history_length * 64
    valid_st = 1
    valid_sb = history_length
    out_st = 64
    out_sb = query_length * 64
    block_m = block_n = 64
    _forward_kernel.warmup(
        _pointer(query),
        _pointer(key),
        _pointer(value),
        _pointer(valid),
        _pointer(output),
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
        query_length=query_length,
        history_length=history_length,
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=int(period),
        scale_log2=(_HEAD_DIM ** -0.5) * _LOG2E,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=3,
        grid=(triton.cdiv(query_length, block_m), 1),
    )
