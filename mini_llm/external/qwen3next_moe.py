"""Thin Qwen3-Next GGUF MoE mapping over existing inference primitives.

Routed banks remain with the caller-owned HeterogeneousExpertExecutor. Only
router/admission weights are eager; shared FFN projections load on first use.
"""

from dataclasses import dataclass

from mini_llm.backend import dtype_name, xp
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS
from mini_llm.ops.router_inference import RouterInference
from mini_llm.ops.silu import silu

from .gguf_projection import project_gguf_linear
from .qwen3next_gguf import _meta, _shape
from .qwen3next_mixer import _positive_integer

_DENSE_TYPES = {'F32', 'F16', 'BF16'}
_SHARED = ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight')


@dataclass(frozen=True)
class Qwen3NextMoESpec:
    d_model: int
    d_ff: int
    shared_d_ff: int
    n_experts: int
    top_k: int

    @classmethod
    def from_index(cls, index, layer_id):
        """Validate this branch's metadata/headers without touching payloads."""
        if index.architecture not in ('qwen3next', 'qwen3_next'):
            raise ValueError('Expected Qwen3-Next architecture')
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError('layer_id must be an integer')
        if not 0 <= layer_id < index.layers:
            raise IndexError(f'Invalid Qwen3-Next layer {layer_id}')
        prefix = index.architecture
        # Do not guess the semantics of pruned/grouped/bias-corrected variants.
        for suffix in ('expert_weights_scale', 'expert_weights_norm', 'expert_gating_func',
                       'expert_group_count', 'expert_group_used_count'):
            if f'{prefix}.{suffix}' in index.reader.fields:
                raise NotImplementedError(f'Unsupported explicit routing metadata {suffix}')
        spec = cls(
            _positive_integer(index.d_model, 'd_model'),
            _positive_integer(index.d_ff, 'd_ff'),
            _positive_integer(_meta(index.reader, f'{prefix}.expert_shared_feed_forward_length'),
                              'expert_shared_feed_forward_length'),
            _positive_integer(index.expert_count, 'expert_count'),
            _positive_integer(index.experts_per_token, 'expert_used_count'),
        )
        explicit_ff = _meta(index.reader, f'{prefix}.expert_feed_forward_length', spec.d_ff)
        if _positive_integer(explicit_ff, 'expert_feed_forward_length') != spec.d_ff:
            raise ValueError('Expert width metadata disagrees with routed tensors')
        if spec.top_k > spec.n_experts:
            raise ValueError('expert_used_count exceeds expert_count')
        stem = f'blk.{layer_id}.'
        shapes = {
            'ffn_gate_inp.weight': (spec.n_experts, spec.d_model),
            'ffn_gate_inp_shexp.weight': (spec.d_model,),
            'ffn_gate_shexp.weight': (spec.shared_d_ff, spec.d_model),
            'ffn_up_shexp.weight': (spec.shared_d_ff, spec.d_model),
            'ffn_down_shexp.weight': (spec.d_model, spec.shared_d_ff),
            'ffn_down_exps.weight': (spec.n_experts, spec.d_model, spec.d_ff),
        }
        if stem + 'ffn_gate_up_exps.weight' in index.tensors:
            shapes['ffn_gate_up_exps.weight'] = (spec.n_experts, 2 * spec.d_ff, spec.d_model)
        else:
            shapes['ffn_gate_exps.weight'] = (spec.n_experts, spec.d_ff, spec.d_model)
            shapes['ffn_up_exps.weight'] = (spec.n_experts, spec.d_ff, spec.d_model)
        for name in index.tensors:
            if name.startswith(stem + 'ffn_') and name[len(stem):] not in shapes:
                raise NotImplementedError(f'Unsupported FFN tensor/layout {name}')
        for suffix, shape in shapes.items():
            tensor = index.tensors.get(stem + suffix)
            if tensor is None:
                raise ValueError(f'Missing MoE tensor {stem + suffix}')
            if _shape(tensor) != shape:
                raise ValueError(f'{stem + suffix}: expected shape {shape}, got {_shape(tensor)}')
            kind = tensor.tensor_type.name
            allowed = (set(BLOCK_FORMATS) if '_exps.weight' in suffix else
                       _DENSE_TYPES | set(BLOCK_FORMATS) if suffix in _SHARED else _DENSE_TYPES)
            if kind not in allowed:
                raise NotImplementedError(f'{stem + suffix}: unsupported GGUF type {kind}')
            if kind in BLOCK_FORMATS and shape[-1] % BLOCK_FORMATS[kind][0]:
                raise ValueError(f'{stem + suffix}: innermost dimension must be block-aligned')
        return spec


class Qwen3NextMoEInference:
    """Renormalized routed SwiGLU plus independently sigmoid-gated shared FFN.

    Both branches consume the same already-normalized FP32 hidden state. This
    adapter never owns/closes the supplied executor or substitutes expert IDs.
    """

    @classmethod
    def from_index(cls, index, layer_id, *, expert_executor, cpu_packed_backend='reference'):
        spec = Qwen3NextMoESpec.from_index(index, layer_id)
        if cpu_packed_backend not in ('reference', 'native'):
            raise ValueError("cpu_packed_backend must be 'reference' or 'native'")
        if (expert_executor.d_model, expert_executor.d_ff) != (spec.d_model, spec.d_ff):
            raise ValueError('Executor dimensions do not match MoE tensors')
        if expert_executor.compute_dtype != 'float32':
            raise ValueError('Qwen3-Next adapter requires float32 expert compute')
        result = cls()
        result.index, result.layer_id, result.spec = index, layer_id, spec
        result.cpu_packed_backend = cpu_packed_backend
        result.projection_weights, result._device_projections = {}, {}
        stem = f'blk.{layer_id}.'
        result.router = RouterInference(spec.d_model, spec.n_experts, spec.top_k,
                                       renormalize_selected=True,
                                       normalization_floor=2.0 ** -14)
        result.router.W_router = xp.asarray(index.non_expert_weight(stem + 'ffn_gate_inp.weight'),
                                            dtype=xp.float32).T
        result.router.b_router = xp.zeros(spec.n_experts, dtype=xp.float32)
        result.shared_gate = xp.asarray(index.non_expert_weight(stem + 'ffn_gate_inp_shexp.weight'),
                                        dtype=xp.float32)
        result.experts = ExpertsInference(spec.d_model, spec.d_ff, spec.n_experts,
                                          expert_executor=expert_executor, layer_id=layer_id)
        return result

    def _project(self, hidden, suffix):
        return project_gguf_linear(
            hidden, self.index, f'blk.{self.layer_id}.' + suffix,
            self.projection_weights, self._device_projections, suffix,
            cpu_packed_backend=self.cpu_packed_backend)

    def forward(self, hidden):
        if not isinstance(hidden, xp.ndarray) or dtype_name(hidden.dtype) != 'float32':
            raise TypeError('Qwen3-Next MoE requires backend float32 hidden')
        if hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[-1] != self.spec.d_model:
            raise ValueError('hidden must have shape [positive batch,time,d_model]')
        if not hidden.shape[1]:
            return xp.empty_like(hidden)
        weights, ids = self.router.forward(hidden)
        routed = self.experts.forward(hidden, weights, ids)
        gate = self._project(hidden, 'ffn_gate_shexp.weight')
        up = self._project(hidden, 'ffn_up_shexp.weight')
        shared = self._project(silu(gate) * up, 'ffn_down_shexp.weight')
        admission = hidden @ self.shared_gate
        sigmoid = xp.exp(-xp.logaddexp(xp.float32(0), -admission))
        return routed + shared * sigmoid[..., None]
