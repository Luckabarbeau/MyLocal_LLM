"""Generic inference-only partial RoPE; no training or backward caches."""

import math
from numbers import Integral, Real

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.rope import (
    PreparedRoPEPositions,
    get_rope_cos_sin,
    prepare_rope_positions,
)


class RotaryEmbeddingInference:
    """Rotate the first ``rotary_dim`` lanes of [B,T,H,D], returning a copy.

    ``interleaved`` pairs adjacent prefix lanes; ``half`` pairs j with
    j + rotary_dim/2. Frequencies use rotary_dim, never the full head width.
    As in ops.rope, positions, frequencies, angles and trig evaluation are
    FP32 even for FP64 inputs. Factors are then cast to the input dtype;
    rotation arithmetic uses that dtype (including FP16 rounding).

    Explicit [T] / [B,T] positions override start_pos. Raw explicit positions
    incur the existing helper's validity reduction; prepared positions and
    implicit host offsets require only host metadata checks during forward.
    Implicit tables grow geometrically so the shared helper retains O(L*R),
    not O(L**2*R), factors over L single-token decoding calls, per dtype.
    """

    def __init__(self, rotary_dim, *, base=10000.0, layout="interleaved"):
        if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, Integral):
            raise TypeError("rotary_dim must be a positive even integer")
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("rotary_dim must be a positive even integer")
        if isinstance(base, bool) or not isinstance(base, Real):
            raise TypeError("base must be a finite positive host number")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("base must be finite and positive")
        if layout not in ("interleaved", "half"):
            raise ValueError("layout must be 'interleaved' or 'half'")
        self.rotary_dim = int(rotary_dim)
        self.base = float(base)
        self.layout = layout
        # Keep each dtype's capacity and factors together; switching dtypes
        # must not reuse another dtype's capacity with the wrong table.
        self._tables = {}

    def _validate_prepared(self, positions, batch, time, dtype):
        if positions.shape not in ((time,), (batch, time)):
            raise ValueError("prepared position_ids shape must match [T] or [B,T]")
        if (positions.d_head != self.rotary_dim or positions.base != self.base
                or positions.dtype != dtype_name(dtype)):
            raise ValueError("prepared RoPE rotary_dim/base/dtype does not match")
        factor_shape = ((1 if len(positions.shape) == 1 else batch),
                        time, 1, self.rotary_dim // 2)
        for factor in (positions.cos, positions.sin):
            if not isinstance(factor, xp.ndarray):
                raise TypeError("prepared RoPE factors must be backend arrays")
            if factor.shape != factor_shape:
                raise ValueError("prepared RoPE factor shape does not match")
            if factor.dtype != dtype:
                raise ValueError("prepared RoPE factor dtype does not match")
        return positions.cos, positions.sin

    def forward(self, x, start_pos=0, position_ids=None):
        """Return a new array of x's shape/dtype with an exact copied suffix.

        Empty time chunks are valid, but still validate all supplied metadata.
        No input or prepared factor is ever modified. start_pos on the implicit
        path must be a nonnegative host integer, not an array/device scalar.
        """
        if not isinstance(x, xp.ndarray):
            raise TypeError("x must be a backend xp.ndarray")
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,H,D]")
        if dtype_name(x.dtype) not in {"float16", "bfloat16", "float32", "float64"}:
            raise TypeError("x must have a floating-point dtype")
        batch, time, _, width = x.shape
        if self.rotary_dim > width:
            raise ValueError("rotary_dim must not exceed input head width D")

        if isinstance(position_ids, PreparedRoPEPositions):
            cos, sin = self._validate_prepared(position_ids, batch, time, x.dtype)
        elif position_ids is not None:
            positions = xp.asarray(position_ids)
            if positions.shape not in ((time,), (batch, time)):
                raise ValueError("position_ids must have shape [T] or [B,T]")
            prepared = prepare_rope_positions(
                positions, d_head=self.rotary_dim, base=self.base, dtype=x.dtype,
            )
            cos, sin = self._validate_prepared(prepared, batch, time, x.dtype)
        else:
            if isinstance(start_pos, bool) or not isinstance(start_pos, Integral):
                raise TypeError("start_pos must be a nonnegative host integer")
            if start_pos < 0:
                raise ValueError("start_pos must be nonnegative")
            if not time:
                return x.copy()
            start_pos = int(start_pos)
            end = start_pos + time
            key = dtype_name(x.dtype)
            table = self._tables.get(key)
            if table is None or table[0] < end:
                capacity = 1 << (end - 1).bit_length()
                table_cos, table_sin = get_rope_cos_sin(
                    capacity, self.rotary_dim, self.base, x.dtype,
                )
                table = (capacity, table_cos, table_sin)
                self._tables[key] = table
            cos = table[1][:, start_pos:end]
            sin = table[2][:, start_pos:end]

        y = x.copy()
        if self.layout == "half":
            half = self.rotary_dim // 2
            first, second = slice(0, half), slice(half, self.rotary_dim)
        else:
            first, second = slice(0, self.rotary_dim, 2), slice(1, self.rotary_dim, 2)
        a, b = x[..., first], x[..., second]
        y[..., first] = a * cos - b * sin
        y[..., second] = a * sin + b * cos
        return y
