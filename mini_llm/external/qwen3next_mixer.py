"""Qwen3-Next GGUF projections around the model-independent delta mixer.

Semantics follow transformers 86fdf1595ab819a7baa1d06f2902d00aa94feeb6
(modeling_qwen3_next.py) and llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f
(conversion/qwen.py:394-432, src/models/qwen3next.cpp:314-556).
This is only the recurrent token mixer, not a decoder or generation session.
"""

from dataclasses import dataclass
import math
import os
from numbers import Integral, Real
from typing import Any

from mini_llm.backend import BACKEND_NAME, dtype_name, xp
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

from .qwen3next_gguf import Qwen3NextGGUFIndex, _meta, _shape
from .gguf_projection import project_gguf_linear


_PROJECTIONS = ('attn_qkv.weight', 'attn_gate.weight', 'ssm_ba.weight',
                'ssm_out.weight')
_DENSE_TYPES = {'F32', 'F16', 'BF16'}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


@dataclass(frozen=True)
class Qwen3NextMixerSpec:
    """Dimensions derived from exact GGUF metadata, never inferred from payloads."""

    d_model: int
    layers: int
    n_key_heads: int
    n_value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel_size: int
    full_attention_interval: int
    rms_eps: float
    l2_eps: float = 1e-6

    @property
    def inner_size(self):
        return self.n_value_heads * self.value_dim

    @property
    def qkv_channels(self):
        return 2 * self.n_key_heads * self.key_dim + self.inner_size

    @classmethod
    def from_index(cls, index):
        prefix = index.architecture
        if prefix not in ('qwen3next', 'qwen3_next'):
            raise ValueError(f'Expected Qwen3-Next architecture, got {prefix!r}')

        def dimension(suffix):
            name = f'{prefix}.{suffix}'
            return _positive_integer(_meta(index.reader, name), name)

        d_model = _positive_integer(index.d_model, 'index.d_model')
        layers = _positive_integer(index.layers, 'index.layers')
        n_key_heads = dimension('ssm.group_count')
        n_value_heads = dimension('ssm.time_step_rank')
        key_dim = dimension('ssm.state_size')
        inner_size = dimension('ssm.inner_size')
        conv_kernel_size = dimension('ssm.conv_kernel')
        interval = dimension('full_attention_interval')
        if n_value_heads % n_key_heads:
            raise ValueError('ssm.time_step_rank must be divisible by ssm.group_count')
        if inner_size % n_value_heads:
            raise ValueError('ssm.inner_size must be divisible by ssm.time_step_rank')
        eps_name = f'{prefix}.attention.layer_norm_rms_epsilon'
        eps = _meta(index.reader, eps_name)
        if (isinstance(eps, bool) or not isinstance(eps, Real)
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError(f'{eps_name} must be finite and positive')
        return cls(d_model, layers, n_key_heads, n_value_heads, key_dim,
                   inner_size // n_value_heads, conv_kernel_size, interval,
                   float(eps))

    def tensor_shapes(self):
        """Canonical [output,input] matrix shapes, not GGML dimension order."""
        return {
            'attn_qkv.weight': (self.qkv_channels, self.d_model),
            'attn_gate.weight': (self.inner_size, self.d_model),
            'ssm_ba.weight': (2 * self.n_value_heads, self.d_model),
            'ssm_conv1d.weight': (self.qkv_channels, self.conv_kernel_size),
            'ssm_dt.bias': (self.n_value_heads,),
            'ssm_a': (self.n_value_heads,),
            'ssm_norm.weight': (self.value_dim,),
            'ssm_out.weight': (self.d_model, self.inner_size),
        }


class Qwen3NextRecurrentMixerInference:
    """FP32-only recurrent mixer with caller-owned generic state.

    ``from_index`` validates every header before accessing any tensor payload.
    Packed projection weights retain mmap views and the owning index; dense
    projections are decoded/cached only when first used. The reference CPU
    packed backend deliberately dequantizes per call; select ``native`` for
    the existing packed CPU kernel. Device packed copies are also lazy.
    """

    index: Qwen3NextGGUFIndex
    spec: Qwen3NextMixerSpec
    layer_id: int
    cpu_packed_backend: str
    projection_weights: dict
    _device_projections: dict
    ssm_a: xp.ndarray
    dt_bias: xp.ndarray
    mixer: Any

    @classmethod
    def from_index(cls, index, layer_id, cpu_packed_backend='reference'):
        if cpu_packed_backend not in ('reference', 'native'):
            raise ValueError("cpu_packed_backend must be 'reference' or 'native'")
        spec = Qwen3NextMixerSpec.from_index(index)
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral):
            raise TypeError('layer_id must be an integer')
        layer_id = int(layer_id)
        if not 0 <= layer_id < spec.layers:
            raise IndexError(f'Invalid Qwen3-Next layer {layer_id}')
        if (layer_id + 1) % spec.full_attention_interval == 0:
            raise ValueError(f'Layer {layer_id} is a full-attention layer, not recurrent')
        stem = f'blk.{layer_id}.'
        tensors = {}
        for suffix, expected in spec.tensor_shapes().items():
            name = stem + suffix
            tensor = index.tensors.get(name)
            if tensor is None:
                raise ValueError(f'Missing recurrent mixer tensor {name}')
            shape = _shape(tensor)
            if shape != expected:
                raise ValueError(f'{name}: expected shape {expected}, got {shape}')
            kind = tensor.tensor_type.name
            if kind not in _DENSE_TYPES and kind not in BLOCK_FORMATS:
                raise NotImplementedError(f'{name}: unsupported GGUF type {kind}')
            if kind in BLOCK_FORMATS and shape[-1] % BLOCK_FORMATS[kind][0]:
                raise ValueError(f'{name}: innermost dimension is not {kind} block-aligned')
            tensors[suffix] = tensor

        # Keep the dependency local: metadata inspection does not require the
        # generic inference implementation, nor does it load checkpoint bytes.
        from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference

        mixer = cls()
        mixer.index = index
        mixer.spec = spec
        mixer.layer_id = layer_id
        mixer.cpu_packed_backend = cpu_packed_backend
        mixer.projection_weights = {}
        mixer._device_projections = {}
        mixer._graph_decode_enabled = (BACKEND_NAME == 'cupy' and os.environ.get(
            'MINI_LLM_RECURRENT_GRAPH', '1').lower() in {'1', 'true', 'on', 'yes'})
        mixer._decode_graph = None
        for suffix in _PROJECTIONS:
            tensor = tensors[suffix]
            kind = tensor.tensor_type.name
            mixer.projection_weights[suffix] = (
                PackedGGUFWeight(tensor.data, _shape(tensor), kind)
                if kind in BLOCK_FORMATS else None
            )

        def small_weight(suffix):
            return xp.asarray(index.non_expert_weight(stem + suffix), dtype=xp.float32)

        mixer.ssm_a = small_weight('ssm_a')
        mixer.dt_bias = small_weight('ssm_dt.bias')
        mixer.mixer = GatedDeltaMixerInference(
            spec.n_key_heads, spec.n_value_heads, spec.key_dim, spec.value_dim,
            spec.conv_kernel_size, l2_eps=spec.l2_eps, rms_eps=spec.rms_eps,
            state_dtype='float32',
        )
        mixer.mixer.set_weights(small_weight('ssm_conv1d.weight'),
                                small_weight('ssm_norm.weight'))
        return mixer

    def _project(self, hidden, suffix):
        return project_gguf_linear(hidden, self.index, f'blk.{self.layer_id}.' + suffix,
            self.projection_weights, self._device_projections, suffix,
            cpu_packed_backend=self.cpu_packed_backend,
            linear=quantized_linear, upload=packed_to_device)

    def create_state(self, batch_size):
        """Return the generic DeltaMixerState; no state is stored on this adapter."""
        return self.mixer.create_state(batch_size)

    def _validate_hidden(self, hidden, ndim):
        if not isinstance(hidden, xp.ndarray):
            raise TypeError('hidden must be a backend xp.ndarray')
        if dtype_name(hidden.dtype) != 'float32':
            raise TypeError('Qwen3-Next mixer currently supports float32 hidden only')
        if hidden.ndim != ndim or hidden.shape[-1] != self.spec.d_model:
            layout = '[B,T,D]' if ndim == 3 else '[B,D]'
            raise ValueError(f'hidden must have shape {layout}, D={self.spec.d_model}')
        if hidden.shape[0] <= 0:
            raise ValueError('hidden batch size must be positive')

    def forward(self, hidden, state):
        """Project [B,T,D], mutate only explicit state, and return [B,T,D]."""
        self._validate_hidden(hidden, 3)
        if self._graph_decode_enabled and hidden.shape[1] == 1:
            from mini_llm.ops.causal_conv_inference import CausalConvState
            from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState
            from mini_llm.ops.inference_graph import CapturedArrayInference
            if isinstance(state, DeltaMixerState):
                mutable = (state.conv.history, state.delta.memory)
                expected = ((hidden.shape[0], self.spec.conv_kernel_size - 1, self.spec.qkv_channels),
                            (hidden.shape[0], self.spec.n_value_heads, self.spec.key_dim, self.spec.value_dim))
                eligible = (hidden.flags.c_contiguous and state.length >= 0 and
                    (self._decode_graph is None or self._decode_graph.accepts_current_stream()) and
                    all(isinstance(s.length, Integral) and not isinstance(s.length, bool) and s.length >= 0
                        for s in (state.conv, state.delta)) and
                    all(a.shape == shape and a.dtype == xp.float32 and a.flags.c_contiguous
                        for a, shape in zip(mutable, expected)))
                if eligible:
                    if self._decode_graph is None or not self._decode_graph.matches(hidden, mutable):
                        def enqueue(x, arrays):
                            scratch = DeltaMixerState(CausalConvState(arrays[0]), DeltaRuleState(arrays[1]))
                            return self._forward_uncaptured(x, scratch)
                        # Capture the portable graph: its state arithmetic is
                        # established, and it creates no global pending-event refs.
                        fused = self.mixer._decode_fastpath_enabled
                        self.mixer._decode_fastpath_enabled = False
                        try:
                            self._decode_graph = CapturedArrayInference(enqueue, hidden, mutable)
                        finally:
                            self.mixer._decode_fastpath_enabled = fused
                    output = self._decode_graph.forward(hidden, mutable)
                    state.conv.length = int(state.conv.length) + 1
                    state.delta.length = int(state.delta.length) + 1
                    return output
        return self._forward_uncaptured(hidden, state)

    def _forward_uncaptured(self, hidden, state):
        self._validate_hidden(hidden, 3)
        spec = self.spec
        batch, time = hidden.shape[:2]
        qkv = xp.ascontiguousarray(self._project(hidden, 'attn_qkv.weight'))
        gate = self._project(hidden, 'attn_gate.weight').reshape(
            batch, time, spec.n_value_heads, spec.value_dim)
        ratio = spec.n_value_heads // spec.n_key_heads
        ba = self._project(hidden, 'ssm_ba.weight').reshape(
            batch, time, spec.n_key_heads, 2 * ratio)
        # GGUF preserves [b_group, a_group] for each key head, not global
        # halves and not alternating b/a pairs within a group.
        b = ba[..., :ratio].reshape(batch, time, spec.n_value_heads)
        a = ba[..., ratio:].reshape(batch, time, spec.n_value_heads)
        beta = xp.exp(-xp.logaddexp(xp.float32(0), -b))
        # Conversion already wrote -exp(A_log) to ssm_a. Do not exponentiate it.
        log_decay = self.ssm_a * xp.logaddexp(xp.float32(0), a + self.dt_bias)
        mixed = self.mixer.forward(qkv, gate, log_decay, beta, state)
        return self._project(mixed.reshape(batch, time, spec.inner_size),
                             'ssm_out.weight')

    def step(self, hidden, state):
        """Convenience [B,D] -> [B,D], identical to a one-token forward."""
        self._validate_hidden(hidden, 2)
        return self.forward(hidden[:, None, :], state)[:, 0, :]
