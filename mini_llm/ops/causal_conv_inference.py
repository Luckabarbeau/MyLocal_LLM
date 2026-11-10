"""Inference-only causal depthwise cross-correlation with caller-owned state."""

from dataclasses import dataclass
from numbers import Integral

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.silu import silu


_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_COMPUTE_DTYPES = {"float32", "float64"}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _float_array(value, name):
    if not isinstance(value, xp.ndarray):
        raise TypeError(f"{name} must be a backend xp.ndarray")
    if dtype_name(value.dtype) not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have float16, bfloat16, float32 or float64 dtype")


@dataclass
class CausalConvState:
    """Raw inputs, oldest first, in compute dtype; length counts consumed tokens.

    All history slots participate, including explicitly supplied initial history.
    Fresh states provide zero padding; length does not mask history slots.
    """

    history: xp.ndarray
    length: int = 0

    def reset(self):
        """Clear history in place without invalidating external buffer aliases."""
        self.history.fill(0)
        self.length = 0

    def copy(self):
        """Fork a session with an independent history buffer."""
        return CausalConvState(self.history.copy(), self.length)


class CausalDepthwiseConv1DInference:
    """Streaming cross-correlation with weights [channels, kernel_size].

    Tap j multiplies x[t - kernel_size + 1 + j]; the last tap is current.
    Arithmetic, bias and optional SiLU use compute_dtype (at least FP32).
    Only the final output is cast back to the input dtype. No session state
    or backward caches are retained on the operator.
    """

    def __init__(self, channels, kernel_size, *, activation=None,
                 compute_dtype="float32"):
        self.channels = _positive_int(channels, "channels")
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        if activation not in (None, "silu"):
            raise ValueError("activation must be None or 'silu'")
        try:
            compute_dtype = xp.dtype(compute_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("compute_dtype must be float32 or float64") from exc
        if dtype_name(compute_dtype) not in _COMPUTE_DTYPES:
            raise ValueError("compute_dtype must be float32 or float64")
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.weight = None
        self.bias = None

    def set_weights(self, weight, bias=None):
        """Validate and snapshot backend arrays in compute dtype atomically."""
        _float_array(weight, "weight")
        if weight.shape != (self.channels, self.kernel_size):
            raise ValueError("weight must have shape [channels, kernel_size]")
        if bias is not None:
            _float_array(bias, "bias")
            if bias.shape != (self.channels,):
                raise ValueError("bias must have shape [channels]")
        new_weight = weight.astype(self.compute_dtype, copy=True)
        new_bias = (None if bias is None else
                    bias.astype(self.compute_dtype, copy=True))
        self.weight, self.bias = new_weight, new_bias

    def create_state(self, batch_size):
        batch_size = _positive_int(batch_size, "batch_size")
        return CausalConvState(xp.zeros(
            (batch_size, self.kernel_size - 1, self.channels),
            dtype=self.compute_dtype,
        ))

    def _validate(self, x, state):
        _float_array(x, "x")
        if x.ndim != 3 or x.shape[2] != self.channels or x.shape[0] <= 0:
            raise ValueError("x must have shape [positive batch, time, channels]")
        if not isinstance(state, CausalConvState):
            raise TypeError("state must be a CausalConvState")
        _float_array(state.history, "state.history")
        expected = (x.shape[0], self.kernel_size - 1, self.channels)
        if state.history.shape != expected:
            raise ValueError(f"state.history must have shape {expected}")
        if state.history.dtype != self.compute_dtype:
            raise TypeError("state.history dtype must equal compute_dtype")
        if not getattr(state.history.flags, "writeable", True):
            raise ValueError("state.history must be writable")
        if isinstance(state.length, bool) or not isinstance(state.length, Integral):
            raise TypeError("state.length must be a nonnegative integer")
        if state.length < 0:
            raise ValueError("state.length must be nonnegative")
        if self.weight is None:
            raise RuntimeError("set_weights must be called before forward")

    def forward(self, x, state):
        """Return [B,T,C] and update the supplied state with raw inputs.

        Empty chunks return an empty output without changing state. Validation
        uses only array metadata, never host reads of backend array values.
        Working storage beyond the output is independent of chunk length.
        """
        self._validate(x, state)
        y = xp.empty(x.shape, dtype=x.dtype)
        history = state.history
        for t in range(x.shape[1]):
            current = x[:, t, :].astype(self.compute_dtype, copy=True)
            value = xp.zeros((x.shape[0], self.channels), dtype=self.compute_dtype)
            for j in range(self.kernel_size - 1):
                value += history[:, j, :] * self.weight[:, j]
            value += current * self.weight[:, -1]
            if self.bias is not None:
                value += self.bias
            if self.activation == "silu":
                value = silu(value)
            y[:, t, :] = value.astype(x.dtype, copy=False)
            if self.kernel_size > 1:
                # Explicit copy makes overlapping shifts backend-independent.
                history[:, :-1, :] = history[:, 1:, :].copy()
                history[:, -1, :] = current
        if x.shape[1]:
            state.length = int(state.length) + x.shape[1]
        return y

    def step(self, x, state):
        """Convenience [B,C] -> [B,C], identical to a one-token forward."""
        _float_array(x, "x")
        if x.ndim != 2:
            raise ValueError("step x must have shape [batch, channels]")
        return self.forward(x[:, None, :], state)[:, 0, :]
