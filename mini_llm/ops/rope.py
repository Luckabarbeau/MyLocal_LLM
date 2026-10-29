from dataclasses import dataclass
import os

import numpy as np

from ..backend import BACKEND_NAME, dtype_name, is_bfloat16_dtype, xp


# Shared cache across all attention layers.  A fixed-length training run uses
# one table for every layer; inference can also reuse the same cached table.
_ROPE_TABLE_CACHE = {}
_FUSED_ROPE_BACKWARD_KERNEL = None
_FUSED_ROPE_BACKWARD_DISABLED = False


def clear_rope_cache():
    """Clear shared RoPE tables (mainly useful for tests/backend changes)."""
    _ROPE_TABLE_CACHE.clear()


def get_rope_cos_sin(seq_len, d_head, base, dtype):
    """Return cached RoPE cos/sin tables for positions ``[0, seq_len)``.

    Returned shapes are ``[1, seq_len, 1, d_head/2]`` so they broadcast with
    training tensors shaped ``[B,T,H,D]`` and can be sliced by inference.
    """
    if d_head % 2 != 0:
        raise ValueError("RoPE requires an even d_head.")

    # dtype can be a string, numpy dtype or cupy dtype.  str(dtype) gives a
    # stable cache key without moving any device data to the host.
    key = (int(seq_len), int(d_head), float(base), str(dtype))
    cached = _ROPE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    i = xp.arange(0, d_head, 2, dtype=xp.float32)
    inv_freq = 1.0 / (float(base) ** (i / float(d_head)))
    positions = xp.arange(seq_len, dtype=xp.float32)
    theta = positions[:, None] * inv_freq[None, :]
    cos = xp.cos(theta).astype(dtype, copy=False)[None, :, None, :]
    sin = xp.sin(theta).astype(dtype, copy=False)[None, :, None, :]

    _ROPE_TABLE_CACHE[key] = (cos, sin)
    return cos, sin


def _rope_cos_sin(seq_len, d_head, base, dtype):
    """Backward-compatible private alias."""
    return get_rope_cos_sin(seq_len, d_head, base, dtype)


@dataclass(frozen=True)
class PreparedRoPEPositions:
    """Per-forward RoPE lookup, shared by Q/K and every Transformer layer.

    The table is *not* a global cache of position indices: it belongs to the
    current forward and is retained only as long as its block/backward caches.
    """
    shape: tuple
    d_head: int
    base: float
    dtype: str
    cos: object
    sin: object


def prepare_rope_positions(position_ids, *, d_head, base=10_000.0, dtype="float32"):
    """Validate explicit positions and compute their RoPE factors once.

    A deep decoder sends the SAME source positions through many layers.  Keep
    those absolute positions (including noncontiguous routed blocks) rather
    than renumbering the compact context. One GPU-to-CPU validity check per
    forward replaces both checks on every Q/K call in every layer. Position
    indices must remain unchanged for the lifetime of this prepared object.
    """
    positions = xp.asarray(position_ids)
    if positions.ndim not in (1, 2):
        raise ValueError("position_ids must have shape [T] or [B,T]")
    if positions.dtype.kind not in {"i", "u"}:
        raise TypeError("position_ids must use an integer dtype")
    if positions.size and bool(xp.any(positions < 0)):
        raise ValueError("position_ids must be non-negative")
    if int(d_head) % 2:
        raise ValueError("RoPE requires an even d_head")
    # The reference table computes these same float32 angles from positions.
    # Evaluating only the active indices avoids both the GPU->CPU max() sync
    # and materializing/growing a [0, max_position] table on every compact
    # target length.  The factors are then shared by all Q/K/layers.
    i = xp.arange(0, int(d_head), 2, dtype=xp.float32)
    inv_freq = 1.0 / (float(base) ** (i / float(d_head)))
    theta = positions.astype(xp.float32)[..., None] * inv_freq
    cos_values = xp.cos(theta).astype(dtype, copy=False)
    sin_values = xp.sin(theta).astype(dtype, copy=False)
    if positions.ndim == 1:
        cos = cos_values[None, :, None, :]
        sin = sin_values[None, :, None, :]
    else:
        cos = cos_values[:, :, None, :]
        sin = sin_values[:, :, None, :]
    return PreparedRoPEPositions(
        shape=tuple(positions.shape), d_head=int(d_head),
        base=float(base), dtype=dtype_name(dtype), cos=cos, sin=sin,
    )


def rope_forward(x, base=10_000.0, position_ids=None):
    """Apply RoPE using implicit or explicit sample-local source positions.

    ``position_ids`` may be ``[T]`` (shared by the batch) or ``[B,T]``.  The
    default remains the historical contiguous ``0..T-1`` path.  Explicit IDs
    are required by hierarchical memory because compacted distant blocks must
    retain their original temporal distance from the recent/target region.
    """
    if x.ndim != 4:
        raise ValueError("rope_forward expects [B,T,H,D].")
    b, t, _, d = x.shape

    if isinstance(position_ids, PreparedRoPEPositions):
        # All checks are host-only metadata checks; no GPU reduction/copy.
        valid_shape = (int(t),) if len(position_ids.shape) == 1 else (int(b), int(t))
        if position_ids.shape != valid_shape:
            raise ValueError("prepared position_ids shape does not match RoPE input")
        if (position_ids.d_head != int(d) or position_ids.base != float(base)
                or position_ids.dtype != dtype_name(x.dtype)):
            raise ValueError("prepared RoPE table does not match head/base/dtype")
        cos, sin = position_ids.cos, position_ids.sin
    elif position_ids is None:
        cos, sin = get_rope_cos_sin(t, d, base, x.dtype)
    else:
        positions = xp.asarray(position_ids)
        if positions.ndim == 1:
            if int(positions.shape[0]) != int(t):
                raise ValueError("1D position_ids must have shape [T]")
        elif positions.ndim == 2:
            if tuple(positions.shape) != (int(b), int(t)):
                raise ValueError("2D position_ids must have shape [B,T]")
        else:
            raise ValueError("position_ids must have shape [T] or [B,T]")
        if positions.dtype.kind not in {"i", "u"}:
            raise TypeError("position_ids must use an integer dtype")
        if positions.size and bool(xp.any(positions < 0)):
            raise ValueError("position_ids must be non-negative")

        max_position = int(xp.max(positions).item()) if positions.size else -1
        table_cos, table_sin = get_rope_cos_sin(
            max_position + 1, d, base, x.dtype
        )
        base_cos = table_cos[0, :, 0, :]
        base_sin = table_sin[0, :, 0, :]
        if positions.ndim == 1:
            cos = base_cos[positions][None, :, None, :]
            sin = base_sin[positions][None, :, None, :]
        else:
            cos = base_cos[positions][:, :, None, :]
            sin = base_sin[positions][:, :, None, :]

    x0, x1 = x[..., 0::2], x[..., 1::2]
    y = xp.empty_like(x)
    y[..., 0::2] = x0 * cos - x1 * sin
    y[..., 1::2] = x0 * sin + x1 * cos
    return y, {"cos": cos, "sin": sin}


def _fused_rope_backward_enabled():
    return (
        BACKEND_NAME == "cupy"
        and os.environ.get("MINI_LLM_FUSED_ROPE_BACKWARD", "0")
        .strip().lower() in {"1", "true", "yes", "on"}
    )


def _get_fused_rope_backward_kernel():
    global _FUSED_ROPE_BACKWARD_KERNEL, _FUSED_ROPE_BACKWARD_DISABLED
    if _FUSED_ROPE_BACKWARD_DISABLED:
        return None
    if _FUSED_ROPE_BACKWARD_KERNEL is not None:
        return _FUSED_ROPE_BACKWARD_KERNEL
    code = r"""
    #include <cuda_bf16.h>

    extern "C" __global__
    void inverse_rope_bf16_factors(
        const float* __restrict__ dy,
        const __nv_bfloat16* __restrict__ cos_table,
        const __nv_bfloat16* __restrict__ sin_table,
        float* __restrict__ dx,
        long long total_pairs,
        int seq_len,
        int heads,
        int d_head,
        int factor_batches)
    {
        const int half = d_head >> 1;
        for (long long pair = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             pair < total_pairs;
             pair += (long long)blockDim.x * gridDim.x) {
            const long long token_head = pair / half;
            const int pair_lane = (int)(pair - token_head * half);
            const int token = (int)((token_head / heads) % seq_len);
            const int batch = (int)(token_head / ((long long)heads * seq_len));
            const long long factor = (
                ((long long)(factor_batches == 1 ? 0 : batch) * seq_len + token)
                * half + pair_lane
            );
            const long long base = pair << 1;
            const float a = dy[base];
            const float b = dy[base + 1];
            const float c = __bfloat162float(cos_table[factor]);
            const float s = __bfloat162float(sin_table[factor]);
            dx[base] = a * c + b * s;
            dx[base + 1] = -a * s + b * c;
        }
    }
    """
    try:
        _FUSED_ROPE_BACKWARD_KERNEL = xp.RawKernel(
            code, "inverse_rope_bf16_factors", options=("--std=c++11",)
        )
    except Exception:
        _FUSED_ROPE_BACKWARD_DISABLED = True
        return None
    return _FUSED_ROPE_BACKWARD_KERNEL


def _fused_rope_backward(dy, cos, sin):
    global _FUSED_ROPE_BACKWARD_DISABLED
    if (
        not _fused_rope_backward_enabled()
        or dy.ndim != 4
        or dy.dtype != xp.float32
        or not dy.flags.c_contiguous
        or not is_bfloat16_dtype(cos.dtype)
        or not is_bfloat16_dtype(sin.dtype)
        or not cos.flags.c_contiguous
        or not sin.flags.c_contiguous
    ):
        return None
    batch, seq_len, heads, d_head = map(int, dy.shape)
    if d_head % 2 or tuple(sin.shape) != tuple(cos.shape):
        return None
    factor_batches = int(cos.shape[0])
    if (
        factor_batches not in (1, batch)
        or tuple(cos.shape[1:]) != (seq_len, 1, d_head // 2)
    ):
        return None
    kernel = _get_fused_rope_backward_kernel()
    if kernel is None:
        return None
    dx = xp.empty_like(dy)
    total_pairs = int(dy.size // 2)
    threads = 256
    blocks = min((total_pairs + threads - 1) // threads, 65535)
    try:
        kernel(
            (blocks,), (threads,),
            (
                dy, cos, sin, dx, np.int64(total_pairs), np.int32(seq_len),
                np.int32(heads), np.int32(d_head), np.int32(factor_batches),
            ),
        )
    except Exception:
        _FUSED_ROPE_BACKWARD_DISABLED = True
        return None
    return dx


def rope_backward(dy, cache):
    cos, sin = cache["cos"], cache["sin"]
    fused = _fused_rope_backward(dy, cos, sin)
    if fused is not None:
        return fused
    dy0, dy1 = dy[..., 0::2], dy[..., 1::2]
    dx = xp.empty_like(dy)
    dx[..., 0::2] = dy0 * cos + dy1 * sin
    dx[..., 1::2] = -dy0 * sin + dy1 * cos
    return dx


def build_rope_matrix(position, d_head, base=10_000.0, dtype="float64"):
    if d_head % 2 != 0:
        raise ValueError("d_head must be even.")
    R = xp.zeros((d_head, d_head), dtype=dtype)
    for pair in range(d_head // 2):
        theta = float(position) / (float(base) ** (2.0 * pair / float(d_head)))
        c, s = xp.cos(theta), xp.sin(theta)
        i = 2 * pair
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    return R
