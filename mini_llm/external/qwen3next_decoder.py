"""Qwen3-Next decoder-layer composition, not a separate model/generation engine.

Only checkpoint names and branch selection live here. Norms, mixers, routing,
packed projections, managed dispatch and activations reuse the existing backend.
"""

from dataclasses import dataclass
from numbers import Integral

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.rmsnorm_inference import RMSNormInference

from .qwen3next_attention import Qwen3NextAttentionInference
from .qwen3next_gguf import _meta, _shape
from .qwen3next_mixer import Qwen3NextRecurrentMixerInference, _positive_integer
from .qwen3next_moe import Qwen3NextMoEInference, Qwen3NextMoESpec


@dataclass
class Qwen3NextLayerState:
    """Bounded layer handle around existing native KV or delta/convolution state.

    Exclusive caller ownership is required. Runtime failures after a mixer has
    executed are not transactional; reset or discard that state before retrying.
    """

    layer_id: int
    batch_size: int
    max_length: int
    mixer_state: object

    @property
    def length(self):
        return self.mixer_state.length

    def reset(self):
        self.mixer_state.reset()


class Qwen3NextDecoderLayerInference:
    """x -> norm -> mixer -> +x -> norm -> (routed + gated shared) -> +residual.

    Inputs/outputs are FP32 [B,T,D]. No token embeddings, final norm, LM head,
    automatic GPU budgets, global session mutation or CUDA graph capture here.
    A caller may share one existing executor across all layers.
    """

    @classmethod
    def from_index(cls, index, layer_id, *, expert_executor, cpu_packed_backend='reference'):
        if isinstance(layer_id, bool) or not isinstance(layer_id, Integral):
            raise TypeError('layer_id must be an integer')
        layer_id = int(layer_id)
        # Validate layer FFN/norm headers before constructing either branch.
        moe_spec = Qwen3NextMoESpec.from_index(index, layer_id)
        stem = f'blk.{layer_id}.'
        for suffix in ('attn_norm.weight', 'post_attention_norm.weight'):
            tensor = index.tensors.get(stem + suffix)
            if tensor is None or _shape(tensor) != (moe_spec.d_model,):
                raise ValueError(f'{stem + suffix}: expected normalization shape {(moe_spec.d_model,)}')
            if tensor.tensor_type.name not in {'F32', 'F16', 'BF16'}:
                raise NotImplementedError(f'{stem + suffix}: unsupported normalization type')
        interval = _positive_integer(_meta(index.reader,
            f'{index.architecture}.full_attention_interval'), 'full_attention_interval')
        attention = (layer_id + 1) % interval == 0
        mixer_class = Qwen3NextAttentionInference if attention else Qwen3NextRecurrentMixerInference
        result = cls()
        result.index, result.layer_id, result.d_model = index, layer_id, moe_spec.d_model
        result.is_attention = attention
        # Reject incompatible executor configuration before reading mixer payloads.
        result.moe = Qwen3NextMoEInference.from_index(index, layer_id,
            expert_executor=expert_executor, cpu_packed_backend=cpu_packed_backend)
        result.mixer = mixer_class.from_index(index, layer_id, cpu_packed_backend)
        result.norm1 = RMSNormInference(moe_spec.d_model, eps=result.mixer.spec.rms_eps)
        result.norm2 = RMSNormInference(moe_spec.d_model, eps=result.mixer.spec.rms_eps)
        for norm, suffix in ((result.norm1, 'attn_norm.weight'),
                             (result.norm2, 'post_attention_norm.weight')):
            norm.set_weights(xp.asarray(index.non_expert_weight(stem + suffix), dtype=xp.float32))
        return result

    def create_state(self, batch_size, max_length):
        batch = _positive_integer(batch_size, 'batch_size')
        capacity = _positive_integer(max_length, 'max_length')
        native = (self.mixer.create_state(batch, capacity) if self.is_attention else
                  self.mixer.create_state(batch))
        return Qwen3NextLayerState(self.layer_id, batch, capacity, native)

    def _validate_state(self, state):
        if not isinstance(state, Qwen3NextLayerState) or state.layer_id != self.layer_id:
            raise ValueError('State belongs to a different decoder layer')
        _positive_integer(state.batch_size, 'state.batch_size')
        _positive_integer(state.max_length, 'state.max_length')
        if (isinstance(state.length, bool) or not isinstance(state.length, Integral)
                or not 0 <= state.length <= state.max_length):
            raise ValueError('Invalid layer state position/capacity')

    def copy_state(self, state):
        self._validate_state(state)
        native = (self.mixer.attention.copy_state(state.mixer_state) if self.is_attention
                  else state.mixer_state.copy())
        return Qwen3NextLayerState(self.layer_id, state.batch_size, state.max_length, native)

    def forward(self, hidden, state, position_ids=None):
        self._validate_state(state)
        if not isinstance(hidden, xp.ndarray) or dtype_name(hidden.dtype) != 'float32':
            raise TypeError('Qwen3-Next layer requires backend float32 hidden')
        if hidden.ndim != 3 or hidden.shape != (state.batch_size, hidden.shape[1], self.d_model):
            raise ValueError('hidden must have shape [state.batch_size,time,d_model]')
        if state.length + hidden.shape[1] > state.max_length:
            raise ValueError('Layer state capacity exceeded')
        if position_ids is not None and not self.is_attention:
            raise ValueError('Explicit RoPE positions apply only to attention layers')
        if not hidden.shape[1]:
            return hidden.copy()
        normalized = self.norm1.forward(hidden)
        if self.is_attention:
            mixed = self.mixer.forward(normalized, state.mixer_state, position_ids=position_ids)
        else:
            mixed = self.mixer.forward(normalized, state.mixer_state)
        residual = hidden + mixed
        return residual + self.moe.forward(self.norm2.forward(residual))

    def step(self, hidden, state, position_ids=None):
        if not isinstance(hidden, xp.ndarray) or hidden.ndim != 2:
            raise ValueError('step hidden must have shape [B,D]')
        return self.forward(hidden[:, None, :], state, position_ids=position_ids)[:, 0]
