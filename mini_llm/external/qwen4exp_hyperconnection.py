"""Inference-only FP32, four-stream Qwen4Exp hyper-connections.

Canonical matrices are output-first. Gamma is the converted multiplicative
scale (already includes any converter +1), not a zero-centred raw parameter.
No decoder, history, placement policy, or backward graph is owned here.
"""

import math

from ..backend import xp
from ..ops.hyperconnection_decode_gpu import eligible, gains_gpu, mix_gpu, write_gpu


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
    ``fast_decode=True`` opts into contiguous single-token CuPy FP32 fusion;
    other inputs retain the reference graph.
    """

    def __init__(self, gamma, down, up, inject=None, *, epsilon, fast_decode=False,
                 graph_decode=False, graph_pool_bytes=2 * 1024**2, native_read=False,
                 packed_bf16=False):
        if not isinstance(packed_bf16,bool):
            raise TypeError('packed_bf16 must be bool')
        if packed_bf16 and (xp.__name__!='cupy' or not native_read or not fast_decode or graph_decode):
            raise ValueError('Packed BF16 HC requires native single-token CuPy read')
        def weight(name,value):
            if not packed_bf16:
                return _fp32(name,value)
            if not isinstance(value,xp.ndarray) or value.dtype!=xp.uint16 or not value.flags.c_contiguous:
                raise TypeError(f'{name} must be contiguous CuPy uint16 BF16 storage')
            return value
        gamma = _fp32("gamma", gamma)
        down = weight("down", down)
        up = weight("up", up)
        self.packed_bf16=packed_bf16
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
            weight("inject", inject)
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
        self.fast_decode = fast_decode
        if not isinstance(graph_decode, bool):
            raise TypeError('graph_decode must be bool')
        if isinstance(graph_pool_bytes, bool) or not isinstance(graph_pool_bytes, int) or graph_pool_bytes < 1:
            raise ValueError('graph_pool_bytes must be a positive integer')
        if not isinstance(native_read, bool):
            raise TypeError('native_read must be bool')
        if native_read and graph_decode:
            raise ValueError('native_read and graph_decode are separate execution modes')
        self.native_read = native_read
        self.graph_decode = graph_decode
        self.graph_pool_bytes = graph_pool_bytes
        self._read_graph = None

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

    def _fast_eligible(self, streams):
        return self.fast_decode and eligible(streams, self.gamma, self.down, self.up)

    def _project(self, value, weight):
        if self.graph_decode:
            # CuPy's cuBLAS wrapper rejects stream capture; reuse exact FP32 GEMV.
            from ..ops.inference_matmul import matmul_inference
            result = matmul_inference(value.reshape(-1, value.shape[-1]), weight.T)
            return result.reshape(value.shape[:-1] + (weight.shape[0],))
        return value @ weight.T

    def _mix(self, streams):
        self._streams(streams)
        if self._fast_eligible(streams):
            return mix_gpu(streams, self.gamma, self.down, self.up, self.epsilon,
                           project=self._project if self.graph_decode else None)
        rms = xp.sqrt(xp.mean(streams * streams, axis=-1, keepdims=True,
                              dtype=xp.float32) + self.epsilon)
        normalized = (streams / rms) * self.gamma
        flat = normalized.reshape(streams.shape[:-2] + (4 * self.width,))
        low = self._project(flat, self.down) * xp.float32(0.25)
        low = low * _sigmoid(low)
        gates = _sigmoid(self._project(low, self.up)).reshape(streams.shape)
        mixed = xp.mean(normalized * gates, axis=-2, dtype=xp.float32)
        return mixed, flat

    def read(self, streams):
        """Read a feature-gated mix and the scalar write gain for each stream."""
        self._streams(streams)
        if self.packed_bf16:
            from ..ops.hyperconnection_read_gpu import read_gpu
            return read_gpu(streams,self.gamma,self.down,self.up,self.inject,self.epsilon)
        if self.native_read and self._fast_eligible(streams) and eligible(streams, self.inject):
            from ..ops.hyperconnection_read_gpu import read_gpu
            return read_gpu(streams, self.gamma, self.down, self.up, self.inject, self.epsilon)
        if self.graph_decode and eligible(streams, self.gamma, self.down, self.up):
            from ..ops.inference_graph import CapturedArrayInference
            graph = self._read_graph
            if graph is None:
                def enqueue(value, unused):
                    mixed, gains = self._read_uncaptured(value)
                    if gains is None:
                        return mixed.reshape(-1)
                    return xp.concatenate((mixed.reshape(-1), gains.reshape(-1)))
                graph = CapturedArrayInference(enqueue, streams, (),
                                               pool_limit_bytes=self.graph_pool_bytes)
                self._read_graph = graph
            if graph.matches(streams, ()) and graph.accepts_current_stream():
                packed = graph.forward(streams, ())
                leading = streams.shape[:-2]
                mixed = packed[:self.width].reshape(leading + (self.width,))
                gains = None if self.inject is None else packed[self.width:].reshape(leading + (4,))
                return mixed, gains
        return self._read_uncaptured(streams)

    def graph_statistics(self):
        """Owned graph-pool bytes only, not a global allocator estimate."""
        if self._read_graph is None:
            return dict(calls=0, pool_reserved_bytes=0, pool_used_bytes=0,
                        pool_limit_bytes=self.graph_pool_bytes)
        return self._read_graph.statistics()

    def _read_uncaptured(self, streams):
        mixed, flat = self._mix(streams)
        gains = None
        if self.inject is not None:
            if self._fast_eligible(streams) and eligible(streams, self.inject):
                gains = gains_gpu(flat, self.inject,
                                  project=self._project if self.graph_decode else None)
            else:
                gains = xp.float32(2) * _sigmoid(self._project(flat, self.inject) * xp.float32(0.25))
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
        if self.fast_decode and eligible(streams, output, gains):
            return write_gpu(streams, output, gains)
        return streams + gains[..., :, None] * output[..., None, :]

    def collapse(self, streams):
        """Final read-only mix; no injection matrix or conventional output norm needed."""
        if (self.graph_decode or self.native_read) and self.inject is None:
            return self.read(streams)[0]
        return self._mix(streams)[0]
