"""Generic inference-only gated delta recurrence and projected sequence mixer.

No model names, GGUF parsing, projection weights, or implicit session caches.
The portable backend path prioritizes an explicit computational graph over
fused GPU performance. State storage is bounded independently of token count.
"""

from dataclasses import dataclass
import math
from numbers import Integral

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.causal_conv_inference import (
    CausalDepthwiseConv1DInference, _float_array, _positive_int,
)
from mini_llm.ops.gated_norm_inference import GatedRMSNormInference
from mini_llm.ops.gated_delta_decode import decode_enabled, try_decode
from mini_llm.ops.inference_matmul import matmul_inference


def l2_normalize(x, eps=1e-6):
    """Final-axis x / sqrt(sum(x*x) + eps), at least FP32 arithmetic.

    This is sum-based L2 normalization, not mean-based RMSNorm. In terms of
    RMSNorm its epsilon would be eps / width, followed by 1/sqrt(width).
    Low-precision inputs return FP32, avoiding an intermediate storage round.
    """
    _float_array(x, "x")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x needs a nonempty feature axis")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    dtype = xp.float64 if dtype_name(x.dtype) == "float64" else xp.float32
    value = x.astype(dtype, copy=False)
    return value / xp.sqrt(xp.sum(value * value, axis=-1, keepdims=True) + eps)


@dataclass
class DeltaRuleState:
    """Logical key-by-value memory [batch, value_heads, key_dim, value_dim]."""

    memory: xp.ndarray
    length: int = 0

    def reset(self):
        self.memory.fill(0)
        self.length = 0

    def copy(self):
        return DeltaRuleState(self.memory.copy(), self.length)


class GatedDeltaRuleInference:
    """Decay, predict from decayed memory, correct, then read UPDATED memory.

    Inputs q/k are already normalized if required by the caller. log_decay
    and beta are explicit per-token/value-head coefficients, not logits.
    A static head_map allows arbitrary sharing; default is repeat-interleave.
    Only query readout receives query_scale (default 1/sqrt(key_dim)).
    """

    def __init__(self, n_key_heads, n_value_heads, key_dim, value_dim, *,
                 head_map=None, query_scale=None, state_dtype="float32"):
        self.n_key_heads = _positive_int(n_key_heads, "n_key_heads")
        self.n_value_heads = _positive_int(n_value_heads, "n_value_heads")
        self.key_dim = _positive_int(key_dim, "key_dim")
        self.value_dim = _positive_int(value_dim, "value_dim")
        self.state_dtype = xp.dtype(state_dtype)
        if dtype_name(self.state_dtype) not in {"float32", "float64"}:
            raise ValueError("state_dtype must be float32 or float64")
        if head_map is None:
            if self.n_value_heads % self.n_key_heads:
                raise ValueError("Default head sharing requires divisible head counts")
            ratio = self.n_value_heads // self.n_key_heads
            head_map = tuple(h // ratio for h in range(self.n_value_heads))
        else:
            head_map = tuple(head_map)
        if len(head_map) != self.n_value_heads or any(
            isinstance(h, bool) or not isinstance(h, Integral) or
            h < 0 or h >= self.n_key_heads for h in head_map
        ):
            raise ValueError("head_map must contain one valid key head per value head")
        self.head_map = tuple(int(h) for h in head_map)
        self._head_indices = xp.asarray(self.head_map, dtype=xp.int32)
        self.query_scale = 1.0 / math.sqrt(self.key_dim) if query_scale is None else float(query_scale)
        if not math.isfinite(self.query_scale):
            raise ValueError("query_scale must be finite")

    def create_state(self, batch_size):
        batch_size = _positive_int(batch_size, "batch_size")
        return DeltaRuleState(xp.zeros(
            (batch_size, self.n_value_heads, self.key_dim, self.value_dim),
            dtype=self.state_dtype,
        ))

    def _validate(self, query, key, value, log_decay, beta, state):
        for name, array in (("query", query), ("key", key), ("value", value),
                            ("log_decay", log_decay), ("beta", beta)):
            _float_array(array, name)
        if query.ndim != 4 or query.shape[0] < 1:
            raise ValueError("query must have shape [positive batch,time,key_heads,key_dim]")
        batch, time = query.shape[:2]
        q_shape = (batch, time, self.n_key_heads, self.key_dim)
        v_shape = (batch, time, self.n_value_heads, self.value_dim)
        coefficients = (batch, time, self.n_value_heads)
        if query.shape != q_shape or key.shape != q_shape or value.shape != v_shape:
            raise ValueError("query/key/value dimensions do not match the operator")
        if log_decay.shape != coefficients or beta.shape != coefficients:
            raise ValueError("log_decay and beta must have shape [batch,time,value_heads]")
        if not isinstance(state, DeltaRuleState):
            raise TypeError("state must be DeltaRuleState")
        _float_array(state.memory, "state.memory")
        if state.memory.shape != (batch, self.n_value_heads, self.key_dim, self.value_dim):
            raise ValueError("state.memory shape does not match the operator/batch")
        if state.memory.dtype != self.state_dtype:
            raise TypeError("state.memory dtype must match state_dtype")
        if not getattr(state.memory.flags, "writeable", True):
            raise ValueError("state.memory must be writable")
        if isinstance(state.length, bool) or not isinstance(state.length, Integral) or state.length < 0:
            raise ValueError("state.length must be a nonnegative integer")
        return batch, time

    def forward(self, query, key, value, log_decay, beta, state):
        """Return [B,T,Hv,Dv], mutating only the caller-owned state.

        Validation reads metadata only; no backend scalars are copied to host.
        Coefficient value ranges are the caller's contract (no hidden sigmoid,
        clipping, or stability-changing approximation). Empty chunks are valid.
        """
        batch, time = self._validate(query, key, value, log_decay, beta, state)
        output = xp.empty((batch, time, self.n_value_heads, self.value_dim), dtype=value.dtype)
        memory = state.memory
        for t in range(time):
            q = query[:, t].astype(self.state_dtype, copy=False)[:, self._head_indices]
            k = key[:, t].astype(self.state_dtype, copy=False)[:, self._head_indices]
            v = value[:, t].astype(self.state_dtype, copy=False)
            decay = xp.exp(log_decay[:, t].astype(self.state_dtype, copy=False))
            memory *= decay[..., None, None]
            prediction = matmul_inference(k[..., None, :], memory)[..., 0, :]
            correction = (v - prediction) * beta[:, t].astype(self.state_dtype, copy=False)[..., None]
            memory += k[..., :, None] * correction[..., None, :]
            readout = matmul_inference((q * self.query_scale)[..., None, :], memory)[..., 0, :]
            output[:, t] = readout.astype(value.dtype, copy=False)
        state.length = int(state.length) + time
        return output

    def step(self, query, key, value, log_decay, beta, state):
        """Single-token inputs [B,H,D], coefficients [B,Hv]."""
        return self.forward(query[:, None], key[:, None], value[:, None],
                            log_decay[:, None], beta[:, None], state)[:, 0]


@dataclass
class DeltaMixerState:
    """Independent raw-convolution history and key-by-value recurrence."""

    conv: object
    delta: DeltaRuleState

    @property
    def length(self):
        if self.conv.length != self.delta.length:
            raise ValueError("Convolution and delta state positions differ")
        return self.delta.length

    def reset(self):
        self.conv.reset()
        self.delta.reset()

    def copy(self):
        return DeltaMixerState(self.conv.copy(), self.delta.copy())


class GatedDeltaMixerInference:
    """Compose generic ops on projected inputs; no model-specific weight names.

    qkv channels are global [all Q | all K | all V]. Gate shape is [B,T,Hv,Dv].
    Output stays [B,T,Hv,Dv]; a model adapter owns input/output projections,
    coefficient transforms and any checkpoint-specific channel rearrangement.
    """

    def __init__(self, n_key_heads, n_value_heads, key_dim, value_dim, conv_kernel_size, *,
                 head_map=None, l2_eps=1e-6, rms_eps=1e-6, state_dtype="float32",
                 gate_activation="silu"):
        self.rule = GatedDeltaRuleInference(n_key_heads, n_value_heads, key_dim, value_dim,
            head_map=head_map, state_dtype=state_dtype)
        self.q_width = self.rule.n_key_heads * self.rule.key_dim
        self.v_width = self.rule.n_value_heads * self.rule.value_dim
        self.channels = 2 * self.q_width + self.v_width
        if not math.isfinite(l2_eps) or l2_eps <= 0:
            raise ValueError("l2_eps must be finite and positive")
        self.l2_eps = float(l2_eps)
        self.conv = CausalDepthwiseConv1DInference(self.channels, conv_kernel_size,
            activation="silu", compute_dtype=state_dtype)
        self.norm = GatedRMSNormInference(self.rule.value_dim, eps=rms_eps,
            gate_activation=gate_activation, compute_dtype=state_dtype)
        self._decode_fastpath_enabled = decode_enabled()

    def set_weights(self, conv_weight, norm_weight, conv_bias=None):
        _float_array(norm_weight, "norm_weight")
        if norm_weight.shape != (self.rule.value_dim,):
            raise ValueError("norm_weight must have shape [value_dim]")
        self.conv.set_weights(conv_weight, conv_bias)
        self.norm.set_weights(norm_weight)

    def create_state(self, batch_size):
        return DeltaMixerState(self.conv.create_state(batch_size), self.rule.create_state(batch_size))

    def _split(self, qkv):
        batch, time = qkv.shape[:2]
        q = qkv[..., :self.q_width].reshape(batch, time, self.rule.n_key_heads, self.rule.key_dim)
        k = qkv[..., self.q_width:2*self.q_width].reshape(q.shape)
        v = qkv[..., 2*self.q_width:].reshape(batch, time, self.rule.n_value_heads, self.rule.value_dim)
        return q, k, v

    def forward(self, qkv, gate, log_decay, beta, state):
        _float_array(qkv, "qkv")
        _float_array(gate, "gate")
        if qkv.ndim != 3 or qkv.shape[-1] != self.channels:
            raise ValueError("qkv must have shape [batch,time,2*Hk*Dk+Hv*Dv]")
        if gate.shape != (*qkv.shape[:2], self.rule.n_value_heads, self.rule.value_dim):
            raise ValueError("gate must have shape [batch,time,value_heads,value_dim]")
        if not isinstance(state, DeltaMixerState):
            raise TypeError("state must be DeltaMixerState")
        if state.length < 0:
            raise ValueError("state.length must be nonnegative")
        # Validate both states and all coefficients before either state mutates.
        self.conv._validate(qkv, state.conv)
        self.rule._validate(*self._split(qkv), log_decay, beta, state.delta)
        if self.norm.norm.gamma is None:
            raise RuntimeError("set_weights must be called before forward")
        if self._decode_fastpath_enabled:
            decoded = try_decode(self, qkv, gate, log_decay, beta, state)
            if decoded is not None:
                return decoded
        convolved = self.conv.forward(qkv, state.conv)
        query, key, value = self._split(convolved)
        query = l2_normalize(query.astype(self.rule.state_dtype, copy=False), self.l2_eps)
        key = l2_normalize(key.astype(self.rule.state_dtype, copy=False), self.l2_eps)
        output = self.rule.forward(query, key, value, log_decay, beta, state.delta)
        return self.norm.forward(output, gate)
