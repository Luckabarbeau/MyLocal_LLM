"""Qwen3-Next GGUF full-attention branch (not a decoder block).

Input is already input-normalized FP32 hidden state. Residual connections and
input/post-attention normalization belong to the caller. GGUF Q/K rows retain
converter order: unlike Llama, they must not be unpermuted.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

from mini_llm.backend import BACKEND_NAME, dtype_name, xp
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.rmsnorm_inference import RMSNormInference
from mini_llm.ops.rope import PreparedRoPEPositions, prepare_rope_positions

from .qwen3next_gguf import _meta, _shape


_PROJECTIONS = ('attn_q.weight', 'attn_k.weight', 'attn_v.weight',
                'attn_output.weight')
_DENSE_TYPES = {'F32', 'F16', 'BF16'}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_real(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


@dataclass(frozen=True)
class Qwen3NextAttentionSpec:
    """Exact GGUF dimensions, independent of projection payloads."""

    d_model: int
    layers: int
    n_q_heads: int
    n_kv_heads: int
    key_dim: int
    value_dim: int
    rotary_dim: int
    rope_base: float
    rms_eps: float
    full_attention_interval: int

    @classmethod
    def from_index(cls, index):
        prefix = index.architecture
        if prefix not in ('qwen3next', 'qwen3_next'):
            raise ValueError(f'Expected Qwen3-Next architecture, got {prefix!r}')

        def dimension(suffix):
            name = f'{prefix}.{suffix}'
            return _positive_integer(_meta(index.reader, name), name)

        # No scaled/extended RoPE variant is silently interpreted as base RoPE.
        rope_prefix = f'{prefix}.rope.'
        allowed = {'dimension_count', 'freq_base', 'scaling.type', 'scaling.factor'}
        for name in index.reader.fields:
            if name.startswith(rope_prefix) and name[len(rope_prefix):] not in allowed:
                raise NotImplementedError(f'Unsupported RoPE metadata {name}')
        scaling = _meta(index.reader, rope_prefix + 'scaling.type', 'none')
        if scaling not in ('none', 'linear'):
            raise NotImplementedError(f'Unsupported RoPE scaling {scaling!r}')
        factor_name = rope_prefix + 'scaling.factor'
        if factor_name in index.reader.fields:
            factor = _positive_real(_meta(index.reader, factor_name), factor_name)
            if factor != 1.0:
                raise NotImplementedError('Only RoPE scaling factor 1 is supported')
        elif scaling == 'linear':
            raise ValueError('Explicit linear RoPE requires scaling.factor=1')

        spec = cls(
            _positive_integer(index.d_model, 'index.d_model'),
            _positive_integer(index.layers, 'index.layers'),
            dimension('attention.head_count'), dimension('attention.head_count_kv'),
            dimension('attention.key_length'), dimension('attention.value_length'),
            dimension('rope.dimension_count'),
            _positive_real(_meta(index.reader, rope_prefix + 'freq_base'),
                           rope_prefix + 'freq_base'),
            _positive_real(_meta(index.reader, f'{prefix}.attention.layer_norm_rms_epsilon'),
                           f'{prefix}.attention.layer_norm_rms_epsilon'),
            dimension('full_attention_interval'),
        )
        if spec.n_q_heads % spec.n_kv_heads:
            raise ValueError('attention.head_count must be divisible by head_count_kv')
        if spec.key_dim != spec.value_dim:
            raise ValueError('Qwen3-Next per-lane gate requires key_dim == value_dim')
        if spec.rotary_dim % 2 or spec.rotary_dim > spec.key_dim:
            raise ValueError('rope.dimension_count must be even and <= key_dim')
        return spec

    def tensor_shapes(self):
        """Canonical output/input matrix order, not GGML header order."""
        return {
            'attn_q.weight': (2 * self.n_q_heads * self.key_dim, self.d_model),
            'attn_k.weight': (self.n_kv_heads * self.key_dim, self.d_model),
            'attn_v.weight': (self.n_kv_heads * self.value_dim, self.d_model),
            'attn_output.weight': (self.d_model, self.n_q_heads * self.value_dim),
            'attn_q_norm.weight': (self.key_dim,),
            'attn_k_norm.weight': (self.key_dim,),
        }


class Qwen3NextAttentionInference:
    """Caller-owned native KV state with lazy GGUF projection residency."""

    @classmethod
    def from_index(cls, index, layer_id, cpu_packed_backend='reference'):
        if cpu_packed_backend not in ('reference', 'native'):
            raise ValueError("cpu_packed_backend must be 'reference' or 'native'")
        spec = Qwen3NextAttentionSpec.from_index(index)
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral):
            raise TypeError('layer_id must be an integer')
        layer_id = int(layer_id)
        if not 0 <= layer_id < spec.layers:
            raise IndexError(f'Invalid Qwen3-Next layer {layer_id}')
        if (layer_id + 1) % spec.full_attention_interval:
            raise ValueError(f'Layer {layer_id} is recurrent, not full-attention')
        stem = f'blk.{layer_id}.'
        for name in index.tensors:
            suffix = name[len(stem):] if name.startswith(stem) else ''
            if (suffix.startswith('attn_') and suffix.endswith('.bias')) or suffix in (
                    'attn_qkv.weight', 'attn_gate.weight', 'rope_freqs.weight'):
                raise NotImplementedError(f'Unsupported full-attention layout {name}')
        tensors = {}
        for suffix, expected in spec.tensor_shapes().items():
            name = stem + suffix
            tensor = index.tensors.get(name)
            if tensor is None:
                raise ValueError(f'Missing full-attention tensor {name}')
            shape = _shape(tensor)
            if shape != expected:
                raise ValueError(f'{name}: expected shape {expected}, got {shape}')
            kind = tensor.tensor_type.name
            supported = _DENSE_TYPES | set(BLOCK_FORMATS) if suffix in _PROJECTIONS else _DENSE_TYPES
            if kind not in supported:
                raise NotImplementedError(f'{name}: unsupported GGUF type {kind}')
            if kind in BLOCK_FORMATS and shape[-1] % BLOCK_FORMATS[kind][0]:
                raise ValueError(f'{name}: innermost dimension is not {kind} block-aligned')
            tensors[suffix] = tensor

        # Imports remain local so spec/header inspection needs no execution core.
        from mini_llm.ops.projected_attention_inference import ProjectedGQAAttentionInference
        from mini_llm.ops.rotary_inference import RotaryEmbeddingInference

        result = cls()
        result.index = index  # Retain the mmap owner for all packed views.
        result.spec = spec
        result.layer_id = layer_id
        result.cpu_packed_backend = cpu_packed_backend
        result.projection_weights = {}
        result._device_projections = {}
        for suffix in _PROJECTIONS:
            tensor = tensors[suffix]
            kind = tensor.tensor_type.name
            if kind in BLOCK_FORMATS:
                weight = PackedGGUFWeight(tensor.data, _shape(tensor), kind)
                # Only mark our view read-only; do not change the index owner's flags.
                weight.data.setflags(write=False)
                result.projection_weights[suffix] = weight
        result.q_norm = RMSNormInference(spec.key_dim, eps=spec.rms_eps)
        result.k_norm = RMSNormInference(spec.key_dim, eps=spec.rms_eps)
        for norm, suffix in ((result.q_norm, 'attn_q_norm.weight'),
                             (result.k_norm, 'attn_k_norm.weight')):
            norm.set_weights(xp.asarray(index.non_expert_weight(stem + suffix),
                                       dtype=xp.float32))
        result.rope = RotaryEmbeddingInference(spec.rotary_dim, base=spec.rope_base,
                                              layout='half')
        result.attention = ProjectedGQAAttentionInference(
            spec.n_q_heads, spec.n_kv_heads, spec.key_dim, spec.value_dim,
            dtype='float32', scale=1.0 / math.sqrt(spec.key_dim))
        return result

    def _project(self, hidden, suffix):
        from .gguf_projection import project_gguf_linear

        output = project_gguf_linear(
            hidden, self.index, f'blk.{self.layer_id}.' + suffix,
            self.projection_weights, self._device_projections, suffix,
            cpu_packed_backend=self.cpu_packed_backend)
        weight = self.projection_weights.get(suffix)
        if BACKEND_NAME == 'numpy' and isinstance(weight, xp.ndarray):
            weight.setflags(write=False)
        return output

    def create_state(self, batch_size, max_length):
        return self.attention.create_state(batch_size, max_length)

    def _validate_hidden(self, hidden, ndim):
        if not isinstance(hidden, xp.ndarray):
            raise TypeError('hidden must be a backend xp.ndarray')
        if dtype_name(hidden.dtype) != 'float32':
            raise TypeError('Qwen3-Next attention currently supports float32 hidden only')
        if hidden.ndim != ndim or hidden.shape[-1] != self.spec.d_model:
            layout = '[B,T,D]' if ndim == 3 else '[B,D]'
            raise ValueError(f'hidden must have shape {layout}, D={self.spec.d_model}')
        if hidden.shape[0] <= 0:
            raise ValueError('hidden batch size must be positive')

    def forward(self, hidden, state, position_ids=None):
        """Return [B,T,D]; cache chronology is independent of explicit RoPE IDs."""
        self._validate_hidden(hidden, 3)
        batch, time = hidden.shape[:2]
        if state.batch_size != batch:
            raise ValueError('hidden batch size does not match state')
        if state.length + time > state.max_length:
            raise ValueError('KV cache capacity exceeded')
        spec = self.spec
        positions = position_ids
        if positions is not None and not isinstance(positions, PreparedRoPEPositions):
            positions = prepare_rope_positions(
                positions, d_head=spec.rotary_dim, base=spec.rope_base, dtype='float32')
        qg = self._project(hidden, 'attn_q.weight').reshape(
            batch, time, spec.n_q_heads, 2 * spec.key_dim)
        # Converter order is [Q_head, gate_head] per head, not global Q/G halves.
        q, gate = qg[..., :spec.key_dim], qg[..., spec.key_dim:]
        k = self._project(hidden, 'attn_k.weight').reshape(
            batch, time, spec.n_kv_heads, spec.key_dim)
        v = self._project(hidden, 'attn_v.weight').reshape(
            batch, time, spec.n_kv_heads, spec.value_dim)
        q = self.rope.forward(self.q_norm.forward(q), start_pos=state.length,
                              position_ids=positions)
        k = self.rope.forward(self.k_norm.forward(k), start_pos=state.length,
                              position_ids=positions)
        context = self.attention.forward(q, k, v, state)
        # Exact stable sigmoid, including extreme negative gates without overflow.
        gated = context * xp.exp(-xp.logaddexp(xp.float32(0), -gate))
        return self._project(gated.reshape(batch, time, spec.n_q_heads * spec.value_dim),
                             'attn_output.weight')

    def step(self, hidden, state, position_ids=None):
        """One-token [B,D] convenience wrapper; positions use [1] or [B,1]."""
        self._validate_hidden(hidden, 2)
        return self.forward(hidden[:, None, :], state, position_ids=position_ids)[:, 0, :]
