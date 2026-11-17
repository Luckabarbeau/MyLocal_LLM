"""Inference-only FP32, four-stream Qwen4Exp hyper-connections.

Canonical matrices are output-first. Gamma is the converted multiplicative
scale (already includes any converter +1), not a zero-centred raw parameter.
No decoder, history, placement policy, or backward graph is owned here.
"""

import math

from ..backend import xp


def _fp32(name, value):
    if not isinstance(value, xp.ndarray):
        raise TypeError(f"{name} must be an array on the selected xp backend")
    if value.dtype != xp.float32:
        raise TypeError(f"{name} must have dtype float32, got {value.dtype}")
    return value


def _sigmoid(value):
    # Both exponentials have nonpositive arguments, including at saturation.
    z = xp.exp(-xp.abs(value))
    one = xp.float32(1)
    return xp.where(value >= 0, one / (one + z), z / (one + z))


class Qwen4ExpHyperConnection:
    """Dense FP32 reference connection, accepting streams shaped ``[..., 4, D]``.

    ``read`` returns ``(block_input, gains)``; gains are ``[..., 4]`` or None
    for a read-only connection. Pass each read's gains to its paired ``write``
    with the same original streams. All operations return fresh arrays and
    retain no activation state. Weights are copied at construction.
    """

    def __init__(self, gamma, down, up, inject=None, *, epsilon):
        gamma = _fp32("gamma", gamma)
        down = _fp32("down", down)
        up = _fp32("up", up)
        if gamma.ndim == 1 and gamma.size > 0 and gamma.size % 4 == 0:
            gamma = gamma.reshape(4, -1)
        if gamma.ndim != 2 or gamma.shape[0] != 4 or gamma.shape[1] <= 0:
            raise ValueError("gamma must have shape [4, D] or [4*D], D > 0")
        self.width = gamma.shape[1]
        wide = 4 * self.width
        if down.ndim != 2 or down.shape[0] <= 0 or down.shape[1] != wide:
            raise ValueError("down must have shape [R, 4*D], R > 0")
        if up.shape != (wide, down.shape[0]):
            raise ValueError("up must have shape [4*D, R]")
        if inject is not None:
            _fp32("inject", inject)
            if inject.shape != (4, wide):
                raise ValueError("inject must have shape [4, 4*D]")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        # Refuse values that become zero or infinity in the actual arithmetic.
        if epsilon > float(xp.finfo(xp.float32).max) or epsilon < float(xp.nextafter(xp.float32(0), xp.float32(1))):
            raise ValueError("epsilon must be representable as positive float32")
        self.epsilon = xp.float32(epsilon)
        self.gamma = gamma.copy()
        self.down = down.copy()
        self.up = up.copy()
        self.inject = None if inject is None else inject.copy()

    def _streams(self, streams):
        _fp32("streams", streams)
        if streams.ndim < 2 or streams.shape[-2:] != (4, self.width):
            raise ValueError(f"streams must have shape [..., 4, {self.width}]")

    def initialize(self, embedding):
        """Repeat each token embedding four times; never interpret tokens as history."""
        _fp32("embedding", embedding)
        if embedding.ndim < 1 or embedding.shape[-1] != self.width:
            raise ValueError(f"embedding must have shape [..., {self.width}]")
        return xp.repeat(embedding[..., None, :], 4, axis=-2)

    def _mix(self, streams):
        self._streams(streams)
        rms = xp.sqrt(xp.mean(streams * streams, axis=-1, keepdims=True,
                              dtype=xp.float32) + self.epsilon)
        normalized = (streams / rms) * self.gamma
        flat = normalized.reshape(streams.shape[:-2] + (4 * self.width,))
        low = (flat @ self.down.T) * xp.float32(0.25)
        low = low * _sigmoid(low)
        gates = _sigmoid(low @ self.up.T).reshape(streams.shape)
        mixed = xp.mean(normalized * gates, axis=-2, dtype=xp.float32)
        return mixed, flat

    def read(self, streams):
        """Read a feature-gated mix and the scalar write gain for each stream."""
        mixed, flat = self._mix(streams)
        gains = None
        if self.inject is not None:
            gains = xp.float32(2) * _sigmoid((flat @ self.inject.T) * xp.float32(0.25))
        return mixed, gains

    def write(self, streams, output, gains):
        """Add the block output to ORIGINAL streams, without mutating any input."""
        self._streams(streams)
        _fp32("output", output)
        _fp32("gains", gains)
        leading = streams.shape[:-2]
        if output.shape != leading + (self.width,):
            raise ValueError("output must match streams' leading dimensions and D")
        if gains.shape != leading + (4,):
            raise ValueError("gains must match streams' leading dimensions and four streams")
        return streams + gains[..., :, None] * output[..., None, :]

    def collapse(self, streams):
        """Final read-only mix; no injection matrix or conventional output norm needed."""
        return self._mix(streams)[0]
