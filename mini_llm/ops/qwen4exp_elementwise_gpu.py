"""Opt-in FP32 Qwen4Exp elementwise graphs; no parent dispatch or fallback.

Each call launches one stride-aware ElementwiseKernel on the current stream,
allocating only its output(s). Only kernels/source are cached, per current device.
No input conversion, finite-value readback, synchronization, or scratch arrays.
"""

from functools import lru_cache

from ..backend import xp


_OPTIONS = ("--fmad=false",)
_SIGMOID = r"""
__device__ float qwen4exp_sigmoid(float v) {
    float z = exp(-fabs(v));
    float denominator = 1.0f + z;
    return v >= 0.0f ? 1.0f / denominator : z / denominator;
}
"""


def _validate(**arrays):
    if xp.__name__ != "cupy":
        raise TypeError("Qwen4Exp elementwise GPU operations require the CuPy backend")
    device = xp.cuda.runtime.getDevice()
    for name, a in arrays.items():
        if not isinstance(a, xp.ndarray) or a.dtype != xp.float32:
            raise TypeError(f"{name} must be a CuPy float32 array")
        if a.device.id != device:
            raise ValueError(f"{name} must be on the current device")
    return device


def _same_shape(a, b, names):
    if a.shape != b.shape:
        raise ValueError(f"{names} must have the same shape")


def _logaddexp_source():
    # Reuse the installed FP32 ufunc, including its double literal/equality
    # branch. A softplus rewrite would not preserve this arithmetic graph.
    try:
        op = next(op for op in xp.logaddexp._ops.ops
                  if op.in_types == (xp.dtype("float32"),) * 2
                  and op.out_types == (xp.dtype("float32"),))
        routine = op.routine
        preamble = xp.logaddexp._preamble
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("CuPy FP32 logaddexp ufunc source is unavailable") from exc
    if not isinstance(routine, str):
        raise RuntimeError("CuPy FP32 logaddexp routine must be source text")
    return (preamble + "\n__device__ float qwen4exp_logaddexp(float in0, float in1) {\n"
            "float out0;\n" + routine + "\nreturn out0;\n}\n")


@lru_cache(maxsize=None)
def _kernels(device):
    # The key is obtained by metadata validation before any kernel is selected.
    return (
        xp.ElementwiseKernel(
            "float32 gate, float32 up", "float32 hidden",
            "float activated = gate * qwen4exp_sigmoid(gate);\n"
            "hidden = activated * up;",
            "qwen4exp_shared_hidden", preamble=_SIGMOID, options=_OPTIONS),
        xp.ElementwiseKernel(
            "float32 routed, float32 shared, raw float32 gate_scalar", "float32 combined",
            "float admission = qwen4exp_sigmoid(gate_scalar[0]);\n"
            "float admitted = admission * shared;\n"
            "combined = routed + admitted;",
            "qwen4exp_combine_shared", preamble=_SIGMOID, options=_OPTIONS),
        xp.ElementwiseKernel(
            "float32 beta_logits, float32 alpha, float32 dt, float32 ssm_a",
            "float32 beta, float32 log_decay",
            "float neg_logits = -beta_logits;\n"
            "float softplus_beta = qwen4exp_logaddexp(0.0f, neg_logits);\n"
            "float neg_softplus = -softplus_beta;\n"
            "beta = exp(neg_softplus);\n"
            "float shifted_alpha = alpha + dt;\n"
            "float softplus_alpha = qwen4exp_logaddexp(0.0f, shifted_alpha);\n"
            "log_decay = ssm_a * softplus_alpha;",
            "qwen4exp_delta_coefficients", preamble=_logaddexp_source(), options=_OPTIONS),
    )


def shared_hidden(gate, up):
    """Return ``(gate * stable_sigmoid(gate)) * up`` for equal-shape FP32 arrays."""
    device = _validate(gate=gate, up=up)
    _same_shape(gate, up, "gate and up")
    return _kernels(device)[0](gate, up)


def combine_shared(routed, shared, gate_scalar):
    """Return ``routed + stable_sigmoid(gate_scalar) * shared``.

    routed/shared must have identical shapes; gate_scalar is any one-element
    current-device FP32 array (including a zero-dimensional device scalar).
    """
    device = _validate(routed=routed, shared=shared, gate_scalar=gate_scalar)
    _same_shape(routed, shared, "routed and shared")
    if gate_scalar.size != 1:
        raise ValueError("gate_scalar must contain exactly one element")
    return _kernels(device)[1](routed, shared, gate_scalar)


def delta_coefficients(beta_logits, alpha, dt, ssm_a):
    """Return fresh ``(beta, log_decay)`` in one launch, both logits-shaped.

    beta_logits and alpha have identical shapes. dt and ssm_a may broadcast to
    that shape, but must not expand it. All four inputs are device FP32 arrays.
    """
    device = _validate(beta_logits=beta_logits, alpha=alpha, dt=dt, ssm_a=ssm_a)
    _same_shape(beta_logits, alpha, "beta_logits and alpha")
    if xp.broadcast_shapes(beta_logits.shape, dt.shape, ssm_a.shape) != beta_logits.shape:
        raise ValueError("dt and ssm_a must broadcast without expanding the logits shape")
    return _kernels(device)[2](beta_logits, alpha, dt, ssm_a)
