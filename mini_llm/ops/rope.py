from ..backend import xp


# Shared cache across all attention layers.  A fixed-length training run uses
# one table for every layer; inference can also reuse the same cached table.
_ROPE_TABLE_CACHE = {}


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

    if position_ids is None:
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


def rope_backward(dy, cache):
    cos, sin = cache["cos"], cache["sin"]
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
