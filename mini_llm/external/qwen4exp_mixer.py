"""Patch0207: Qwen4Exp Gated DeltaNet math, without loading/residency policy.

Inputs use the pinned llama.cpp converter's tiled value-head order. All dense
projections are output-first; callbacks may dispatch native packed weights.
"""

from dataclasses import dataclass
import math
from numbers import Real

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.causal_conv_inference import _positive_int
from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference


def _fp32(value, name, shape):
    if not isinstance(value, xp.ndarray) or value.dtype != xp.float32:
        raise TypeError(f"{name} must be a backend float32 array")
    if value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    return value


@dataclass(frozen=True)
class Qwen4ExpMixerSpec:
    d_model: int
    n_key_heads: int
    n_value_heads: int
    head_dim: int
    conv_kernel_size: int
    rms_eps: float

    def __post_init__(self):
        for name in ("d_model", "n_key_heads", "n_value_heads", "head_dim",
                     "conv_kernel_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.n_value_heads % self.n_key_heads:
            raise ValueError("value heads must be divisible by key heads")
        if (isinstance(self.rms_eps, bool) or not isinstance(self.rms_eps, Real)
                or not math.isfinite(self.rms_eps) or self.rms_eps <= 0):
            raise ValueError("rms_eps must be finite and positive")

    @property
    def inner_size(self):
        return self.n_value_heads * self.head_dim

    @property
    def qkv_channels(self):
        return 2 * self.n_key_heads * self.head_dim + self.inner_size

    @classmethod
    def from_metadata(cls, metadata):
        """Exact GGUF keys, no payload access or guessed geometry."""
        if metadata["general.architecture"] != "qwen4exp":
            raise ValueError("expected qwen4exp architecture")
        prefix = "qwen4exp."
        spec = cls(metadata[prefix + "embedding_length"],
                   metadata[prefix + "ssm.group_count"],
                   metadata[prefix + "ssm.time_step_rank"],
                   metadata[prefix + "ssm.state_size"],
                   metadata[prefix + "ssm.conv_kernel"],
                   metadata[prefix + "attention.layer_norm_rms_epsilon"])
        if _positive_int(metadata[prefix + "ssm.inner_size"], "ssm.inner_size") != spec.inner_size:
            raise ValueError("ssm.inner_size must equal value_heads * head_dim")
        return spec


class Qwen4ExpRecurrentMixerInference:
    """FP32 [B,T,D] forward / [B,D] step with explicit mutable session state.

    qkv, z, beta, alpha, out: output-first FP32 matrices or pure callbacks
    mapping [..., input_width] to FP32 [..., output_width]. Callbacks own any
    packed dispatch, storage and residency; this operator never decodes weights.
    Every V-indexed input (including conv V channels, A, dt, Z and output
    columns) must ALREADY use converted tiled order, not HF grouped order.

    State is the existing DeltaMixerState: raw oldest-first conv history
    [B,K-1,2*Hk*D+Hv*D] and key-by-value memory [B,Hv,Dkey,Dvalue]. No state
    lives on this adapter. A is already -exp(A_log), gamma already converted.
    """

    def __init__(self, spec, *, qkv, z, beta, alpha, out, conv_weight,
                 norm_weight, ssm_a, dt_bias, fast_coefficients=False, graph_decode=False):
        if not isinstance(spec, Qwen4ExpMixerSpec):
            raise TypeError("spec must be Qwen4ExpMixerSpec")
        if not isinstance(fast_coefficients, bool):
            raise TypeError('fast_coefficients must be bool')
        if not isinstance(graph_decode, bool):
            raise TypeError('graph_decode must be bool')
        if graph_decode and BACKEND_NAME != 'cupy':
            raise RuntimeError('GDN graph decode requires CuPy')
        self.fast_coefficients = fast_coefficients
        self.graph_decode = graph_decode
        self._decode_graph = None
        self.spec = spec
        self.projections = {}
        self._widths = {"qkv": (spec.qkv_channels, spec.d_model),
                        "z": (spec.inner_size, spec.d_model),
                        "beta": (spec.n_value_heads, spec.d_model),
                        "alpha": (spec.n_value_heads, spec.d_model),
                        "out": (spec.d_model, spec.inner_size)}
        for name, projection in (("qkv", qkv), ("z", z), ("beta", beta),
                                 ("alpha", alpha), ("out", out)):
            if not callable(projection):
                _fp32(projection, name, self._widths[name])
            self.projections[name] = projection
        self.ssm_a = _fp32(ssm_a, "ssm_a", (spec.n_value_heads,)).copy()
        self.dt_bias = _fp32(dt_bias, "dt_bias", (spec.n_value_heads,)).copy()
        if not bool(xp.all(xp.isfinite(self.ssm_a) & (self.ssm_a < 0))):
            raise ValueError("ssm_a must be finite, already-negative -exp(A_log)")
        _fp32(conv_weight, "conv_weight", (spec.qkv_channels, spec.conv_kernel_size))
        _fp32(norm_weight, "norm_weight", (spec.head_dim,))
        self.mixer = GatedDeltaMixerInference(
            spec.n_key_heads, spec.n_value_heads, spec.head_dim, spec.head_dim,
            spec.conv_kernel_size,
            head_map=tuple(h % spec.n_key_heads for h in range(spec.n_value_heads)),
            l2_eps=spec.rms_eps, rms_eps=spec.rms_eps,
            state_dtype="float32", gate_activation="sigmoid")
        self.mixer.set_weights(conv_weight, norm_weight)

    def _project(self, hidden, name):
        projection = self.projections[name]
        if callable(projection):
            result = projection(hidden)
        elif self.graph_decode:
            from mini_llm.ops.inference_matmul import matmul_inference
            result = matmul_inference(hidden, projection.T)
        else:
            result = hidden @ projection.T
        return _fp32(result, name + " output", (*hidden.shape[:-1], self._widths[name][0]))

    def create_state(self, batch_size):
        return self.mixer.create_state(batch_size)

    def forward(self, hidden, state):
        """Return FP32 [B,T,D]; advance conv/matrix state exactly T positions.

        Callbacks must not mutate hidden/state. A callback failure is propagated,
        not retried; an output-projection failure can leave state advanced.
        """
        if not isinstance(hidden, xp.ndarray) or hidden.dtype != xp.float32:
            raise TypeError("hidden must be a backend float32 array")
        if hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[-1] != self.spec.d_model:
            raise ValueError("hidden must be [positive batch,time,d_model]")
        if self.graph_decode and hidden.shape[:2] == (1, 1):
            from mini_llm.ops.causal_conv_inference import CausalConvState
            from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState
            from mini_llm.ops.inference_graph import CapturedArrayInference
            if not isinstance(state, DeltaMixerState):
                raise TypeError('state must be DeltaMixerState')
            arrays = (state.conv.history, state.delta.memory)
            expected = ((1, self.spec.conv_kernel_size-1, self.spec.qkv_channels),
                        (1, self.spec.n_value_heads, self.spec.head_dim, self.spec.head_dim))
            if (any(a.shape != shape or a.dtype != xp.float32 or not a.flags.c_contiguous
                    for a, shape in zip(arrays, expected)) or
                    state.conv.length != state.delta.length or state.length < 0):
                raise ValueError('GDN graph state shape/dtype/layout/length mismatch')
            if self._decode_graph is None:
                def enqueue(x, mutable):
                    private = DeltaMixerState(CausalConvState(mutable[0]), DeltaRuleState(mutable[1]))
                    return self._forward_uncaptured(x, private, capture_owned=True)
                self._decode_graph = CapturedArrayInference(enqueue, hidden, arrays,
                    pool_limit_bytes=16*1024**2)
            output = self._decode_graph.forward(hidden, arrays)
            state.conv.length = int(state.conv.length)+1
            state.delta.length = int(state.delta.length)+1
            return output
        return self._forward_uncaptured(hidden, state)

    def _forward_uncaptured(self, hidden, state, *, capture_owned=False):
        batch, time = hidden.shape[:2]
        spec = self.spec
        qkv = xp.ascontiguousarray(self._project(hidden, "qkv"))
        gate = self._project(hidden, "z").reshape(batch, time, spec.n_value_heads, spec.head_dim)
        beta_logits = self._project(hidden, "beta")
        alpha = self._project(hidden, "alpha")
        if self.fast_coefficients and BACKEND_NAME == 'cupy':
            from mini_llm.ops.qwen4exp_elementwise_gpu import delta_coefficients
            beta, log_decay = delta_coefficients(beta_logits, alpha, self.dt_bias, self.ssm_a)
        else:
            beta = xp.exp(-xp.logaddexp(xp.float32(0), -beta_logits))
            log_decay = self.ssm_a * xp.logaddexp(xp.float32(0), alpha + self.dt_bias)
        mixed = self.mixer.forward(qkv, gate, log_decay, beta, state,
                                   capture_owned=capture_owned)
        return self._project(mixed.reshape(batch, time, spec.inner_size), "out")

    def graph_statistics(self):
        return (self._decode_graph.statistics() if self._decode_graph is not None else
                dict(calls=0, pool_reserved_bytes=0, pool_used_bytes=0))

    def step(self, hidden, state):
        """[B,D] -> [B,D], using exactly the sequence-forward graph."""
        if not isinstance(hidden, xp.ndarray) or hidden.ndim != 2:
            raise ValueError("step hidden must be [batch,d_model]")
        return self.forward(hidden[:, None, :], state)[:, 0, :]
