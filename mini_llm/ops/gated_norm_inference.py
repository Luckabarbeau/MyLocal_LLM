"""Generic per-row RMS normalization followed by an elementwise gate."""

import math

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.causal_conv_inference import _float_array, _positive_int
from mini_llm.ops.rmsnorm_inference import RMSNormInference
from mini_llm.ops.silu import silu, sigmoid_stable


class GatedRMSNormInference:
    """Compute RMSNorm(x) * activation(gate) over the final feature axis.

    Normalization and gating use compute_dtype, then cast once to x.dtype.
    Weights are direct multiplicative gamma, never implicit (1 + gamma).
    The wrapped existing RMSNorm supplies the normalization implementation.
    """

    def __init__(self, width, eps=1e-6, *, gate_activation="silu", compute_dtype="float32"):
        self.width = _positive_int(width, "width")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if gate_activation not in {"silu", "sigmoid"}:
            raise ValueError("gate_activation must be silu or sigmoid")
        self.compute_dtype = xp.dtype(compute_dtype)
        if dtype_name(self.compute_dtype) not in {"float32", "float64"}:
            raise ValueError("compute_dtype must be float32 or float64")
        self.gate_activation = gate_activation
        self.norm = RMSNormInference(self.width, eps=eps, dtype=compute_dtype)

    def set_weights(self, weight):
        _float_array(weight, "weight")
        if weight.shape != (self.width,):
            raise ValueError("weight must have shape [width]")
        self.norm.set_weights(weight.astype(self.compute_dtype, copy=True))

    def forward(self, x, gate):
        _float_array(x, "x")
        _float_array(gate, "gate")
        if x.ndim < 1 or x.shape[-1] != self.width or gate.shape != x.shape:
            raise ValueError("x and gate must share shape [..., width]")
        if self.norm.gamma is None:
            raise RuntimeError("set_weights must be called before forward")
        z = gate.astype(self.compute_dtype, copy=False)
        activated = silu(z) if self.gate_activation == "silu" else sigmoid_stable(z)
        normalized = self.norm.forward(x.astype(self.compute_dtype, copy=False))
        return (normalized * activated).astype(x.dtype, copy=False)
