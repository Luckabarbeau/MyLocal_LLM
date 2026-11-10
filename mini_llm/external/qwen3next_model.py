"""Qwen3-Next model orchestration over the shared NumPy/CuPy inference backend.

This adapter owns checkpoint mapping and session composition, not new attention,
recurrence, routing, expert kernels, sampling or training implementations.
"""

from dataclasses import dataclass, field
from numbers import Integral

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.embedding_inference import RowEmbeddingInference
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.quantized_weight import BLOCK_FORMATS
from mini_llm.ops.rmsnorm_inference import RMSNormInference

from .gguf_projection import project_gguf_linear
from .gguf_tensor import decode_gguf_row
from .qwen3next_attention import Qwen3NextAttentionSpec
from .qwen3next_decoder import Qwen3NextDecoderLayerInference
from .qwen3next_gguf import _meta, _shape
from .qwen3next_mixer import Qwen3NextMixerSpec, _positive_integer
from .qwen3next_moe import Qwen3NextMoESpec


@dataclass(frozen=True)
class Qwen3NextModelConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    tie_word_embeddings: bool


def inspect_qwen3next_model(index):
    """Header-only compatibility and retained-backbone estimate; no payload reads."""
    if index.architecture not in ('qwen3next', 'qwen3_next'):
        raise ValueError('Expected Qwen3-Next architecture')
    embedding = index.tensors.get('token_embd.weight')
    if embedding is None or len(_shape(embedding)) != 2 or _shape(embedding)[1] != index.d_model:
        raise ValueError('Missing/invalid token_embd.weight [vocab,hidden]')
    vocab = _positive_integer(_shape(embedding)[0], 'vocab_size')
    declared = _meta(index.reader, f'{index.architecture}.vocab_size', vocab)
    if _positive_integer(declared, 'vocab_size') != vocab:
        raise ValueError('Vocabulary metadata disagrees with embedding shape')
    config = Qwen3NextModelConfig(index.d_model, vocab, index.layers,
        _positive_integer(_meta(index.reader, f'{index.architecture}.context_length'), 'context_length'),
        'output.weight' not in index.tensors)
    attn, recurrent = Qwen3NextAttentionSpec.from_index(index), Qwen3NextMixerSpec.from_index(index)
    expected = {'token_embd.weight': (vocab, index.d_model), 'output_norm.weight': (index.d_model,)}
    if not config.tie_word_embeddings:
        expected['output.weight'] = (vocab, index.d_model)
    for layer_id in range(index.layers):
        Qwen3NextMoESpec.from_index(index, layer_id)
        stem = f'blk.{layer_id}.'
        mixer = attn if (layer_id + 1) % attn.full_attention_interval == 0 else recurrent
        expected.update({stem + key: shape for key, shape in mixer.tensor_shapes().items()})
        expected.update({stem + name: (index.d_model,)
                         for name in ('attn_norm.weight', 'post_attention_norm.weight')})
        # MoESpec already validates every FFN header, format and supported name.
        expected.update({name: _shape(tensor) for name, tensor in index.tensors.items()
                         if name.startswith(stem + 'ffn_')})
    extra = set(index.tensors) - set(expected)
    if extra:
        raise NotImplementedError(f'Unsupported Qwen3-Next tensors: {sorted(extra)[:4]}')
    resident = 0
    output_name = 'token_embd.weight' if config.tie_word_embeddings else 'output.weight'
    for name, shape in expected.items():
        tensor = index.tensors.get(name)
        if tensor is None or _shape(tensor) != shape:
            raise ValueError(f'{name}: expected shape {shape}')
        kind = tensor.tensor_type.name
        if kind not in {'F32', 'F16', 'BF16', *BLOCK_FORMATS}:
            raise NotImplementedError(f'{name}: unsupported GGUF type {kind}')
        if kind in BLOCK_FORMATS and shape[-1] % BLOCK_FORMATS[kind][0]:
            raise ValueError(f'{name}: packed innermost dimension must be block-aligned')
        dense_norm = name == 'output_norm.weight' or name.endswith((
            'attn_norm.weight', 'post_attention_norm.weight',
            'attn_q_norm.weight', 'attn_k_norm.weight'))
        if dense_norm and kind not in {'F32', 'F16', 'BF16'}:
            raise NotImplementedError(f'{name}: normalization requires a dense gamma')
        if '_exps.weight' in name or (name == 'token_embd.weight' and name != output_name):
            continue
        elements = 1
        for dim in shape:
            elements *= dim
        expanded_control = name.endswith(('ssm_conv1d.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_norm.weight'))
        resident += (elements // BLOCK_FORMATS[kind][0] * BLOCK_FORMATS[kind][1]
                     if kind in BLOCK_FORMATS and not expanded_control else elements * 4)
    return config, attn, recurrent, resident


@dataclass
class Qwen3NextGenerationState:
    layer_states: tuple
    batch_size: int
    max_length: int
    owner: object = field(repr=False)
    length: int = 0
    failed: bool = False

    def reset(self):
        for state in self.layer_states:
            state.reset()
        self.length, self.failed = 0, False

    def remaining_capacity(self):
        return self.max_length - self.length

    def has_capacity(self, n_tokens=1):
        return not self.failed and self.length + n_tokens <= self.max_length


class Qwen3NextGGUFModel:
    """FP32 causal model with mmap embedding rows and packed backbone/experts.

    create_generation_state/prefill/decode_one/forward match external engine
    contracts. Sessions must execute serially when sharing the managed executor.
    A failed partial forward poisons its state until reset; no silent retry.
    """

    @classmethod
    def from_index(cls, index, *, dtype='float32', cpu_packed_backend='native',
                   expert_executor=None, expert_cache_budget_bytes=0, cpu_workers=0,
                   promote_after=3, embedding_cache_bytes=4 * 1024**2,
                   device_memory_budget_bytes=4 * 1024**3, device_reserve_bytes=2 * 1024**3,
                   workspace_reserve_bytes=256 * 1024**2, routing_profiler=None):
        if dtype != 'float32':
            raise ValueError('Qwen3-Next model currently requires dtype=float32')
        if cpu_packed_backend not in ('native', 'reference'):
            raise ValueError('cpu_packed_backend must be native or reference')
        config, attn, recurrent, resident = inspect_qwen3next_model(index)
        for name, value in (('device_memory_budget_bytes', device_memory_budget_bytes),
                            ('device_reserve_bytes', device_reserve_bytes),
                            ('workspace_reserve_bytes', workspace_reserve_bytes),
                            ('expert_cache_budget_bytes', expert_cache_budget_bytes),
                            ('embedding_cache_bytes', embedding_cache_bytes)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if expert_executor is not None:
            if (expert_executor.d_model, expert_executor.d_ff, expert_executor.compute_dtype) != (
                    index.d_model, index.d_ff, 'float32'):
                raise ValueError('Executor dimensions/precision do not match model')
            expert_cache_budget_bytes = expert_executor.cache.device_budget_bytes
        result = cls()
        result.index, result.config, result.dtype = index, config, 'float32'
        result.attention_spec, result.recurrent_spec = attn, recurrent
        result.resident_weight_bytes = resident
        result.expert_cache_budget_bytes = int(expert_cache_budget_bytes)
        result.device_memory_budget_bytes = int(device_memory_budget_bytes)
        result.device_reserve_bytes = int(device_reserve_bytes)
        result.workspace_reserve_bytes = int(workspace_reserve_bytes)
        result.cpu_packed_backend = cpu_packed_backend
        result._owner, result._closed = object(), False
        result._check_memory(result.memory_plan(1, 1), 0)
        result._owns_executor = expert_executor is None
        result.expert_executor = expert_executor or HeterogeneousExpertExecutor(
            index.provider, model_id=index.source_path or f'qwen3next:{id(index)}',
            d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
            device_budget_bytes=expert_cache_budget_bytes, cpu_workers=cpu_workers,
            promote_after=promote_after, cpu_packed_backend=cpu_packed_backend,
            routing_profiler=routing_profiler)
        try:
            result.blocks = tuple(Qwen3NextDecoderLayerInference.from_index(index, layer_id,
                expert_executor=result.expert_executor, cpu_packed_backend=cpu_packed_backend)
                for layer_id in range(config.num_hidden_layers))
            result.embedding = RowEmbeddingInference(config.vocab_size, config.hidden_size,
                lambda row: decode_gguf_row(index.tensors['token_embd.weight'], row),
                cache_bytes=embedding_cache_bytes)
            result.final_norm = RMSNormInference(config.hidden_size, eps=attn.rms_eps)
            result.final_norm.set_weights(xp.asarray(index.non_expert_weight('output_norm.weight'), dtype=xp.float32))
            result._head_weights, result._device_head = {}, {}
            result.output_name = 'token_embd.weight' if config.tie_word_embeddings else 'output.weight'
            result._construction_complete = True
        except Exception:
            if result._owns_executor:
                result.expert_executor.close()
            raise
        return result

    def memory_plan(self, batch_size, max_length, *, tokens=1, all_logits=False):
        b = _positive_integer(batch_size, 'batch_size')
        length = _positive_integer(max_length, 'max_length')
        if length > self.config.max_position_embeddings:
            raise ValueError('Requested state exceeds model context_length')
        a, r = self.attention_spec, self.recurrent_spec
        n_attn = self.config.num_hidden_layers // a.full_attention_interval
        n_rec = self.config.num_hidden_layers - n_attn
        state = b * 4 * (n_attn * a.n_kv_heads * length * (a.key_dim + a.value_dim)
            + n_rec * ((r.conv_kernel_size - 1) * r.qkv_channels
                       + r.n_value_heads * r.key_dim * r.value_dim))
        width = max(r.qkv_channels, a.n_q_heads * a.key_dim * 2, self.config.hidden_size)
        scratch = (16 * b * tokens * width * 4 + 6 * b * a.n_q_heads * tokens * length * 4
                   + b * (tokens if all_logits else 1) * self.config.vocab_size * 4)
        workspace = self.workspace_reserve_bytes + scratch
        total = self.resident_weight_bytes + self.expert_cache_budget_bytes + state + workspace
        return dict(resident_weight_bytes=self.resident_weight_bytes,
                    expert_cache_budget_bytes=self.expert_cache_budget_bytes,
                    state_bytes=state, workspace_bytes=workspace,
                    estimated_total_bytes=total, device_budget_bytes=self.device_memory_budget_bytes)

    def _pending_device_bytes(self):
        """Reserve future lazy weights/cache, excluding uploads already resident."""
        if not getattr(self, '_construction_complete', False):
            return self.resident_weight_bytes + self.expert_cache_budget_bytes
        from .qwen3next_attention import _PROJECTIONS as attention_projections
        from .qwen3next_mixer import _PROJECTIONS as recurrent_projections
        from .qwen3next_moe import _SHARED
        sources = [(self.output_name, self._head_weights, self._device_head, self.output_name)]
        for block in self.blocks:
            stem = f'blk.{block.layer_id}.'
            for op, names in ((block.mixer, attention_projections if block.is_attention else recurrent_projections),
                              (block.moe, _SHARED)):
                sources.extend((stem + suffix, op.projection_weights, op._device_projections, suffix)
                               for suffix in names)
        pending = max(0, self.expert_cache_budget_bytes - self.expert_executor.cache.used_bytes)
        for name, host_cache, device_cache, key in sources:
            tensor = self.index.tensors[name]
            kind = tensor.tensor_type.name
            if (kind in BLOCK_FORMATS and key in device_cache) or (
                    kind not in BLOCK_FORMATS and isinstance(host_cache.get(key), xp.ndarray)):
                continue
            elements = 1
            for dim in _shape(tensor):
                elements *= dim
            pending += (elements // BLOCK_FORMATS[kind][0] * BLOCK_FORMATS[kind][1]
                        if kind in BLOCK_FORMATS else elements * 4)
        return pending

    def _check_memory(self, plan, extra_bytes):
        if BACKEND_NAME == 'cupy':
            if plan['estimated_total_bytes'] > self.device_memory_budget_bytes:
                raise MemoryError('Model/backbone/state/workspace exceeds explicit device budget')
            free, _ = xp.cuda.runtime.memGetInfo()
            if (extra_bytes + self._pending_device_bytes() + plan['workspace_bytes']
                    + self.device_reserve_bytes > free):
                raise MemoryError('Insufficient free VRAM after configured safety reserve')

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Model is closed')

    def create_generation_state(self, batch_size=1, max_length=None):
        self._require_open()
        if max_length is None:
            raise ValueError('Explicit max_length is required for bounded hybrid state')
        plan = self.memory_plan(batch_size, max_length)
        self._check_memory(plan, plan['state_bytes'])
        return Qwen3NextGenerationState(tuple(block.create_state(batch_size, max_length)
            for block in self.blocks), int(batch_size), int(max_length), self._owner)

    def _validate_state(self, state):
        self._require_open()
        if not isinstance(state, Qwen3NextGenerationState) or state.owner is not self._owner:
            raise ValueError('Generation state belongs to another model')
        if state.failed:
            raise RuntimeError('Partial forward failed; reset or discard generation state')
        if (len(state.layer_states) != len(self.blocks) or isinstance(state.length, bool)
                or not isinstance(state.length, Integral) or not 0 <= state.length <= state.max_length):
            raise ValueError('Invalid generation state layout/length')
        for block, layer in zip(self.blocks, state.layer_states):
            block._validate_state(layer)
            if (layer.length, layer.batch_size, layer.max_length) != (
                    state.length, state.batch_size, state.max_length):
                raise ValueError('Hybrid layer state positions/geometry disagree')

    def copy_generation_state(self, state):
        self._validate_state(state)
        plan = self.memory_plan(state.batch_size, state.max_length)
        self._check_memory(plan, plan['state_bytes'])
        return Qwen3NextGenerationState(tuple(block.copy_state(layer)
            for block, layer in zip(self.blocks, state.layer_states)),
            state.batch_size, state.max_length, self._owner, state.length)

    def _forward(self, token_ids, state, *, all_logits):
        self._validate_state(state)
        tokens = xp.asarray(token_ids)
        if tokens.ndim != 2 or tokens.shape[0] != state.batch_size or tokens.shape[1] < 1:
            raise ValueError('token_ids must have shape [state.batch_size,positive time]')
        if tokens.dtype.kind not in 'iu':
            raise TypeError('token_ids must have integer dtype')
        if not state.has_capacity(tokens.shape[1]):
            raise ValueError('Generation state capacity exceeded')
        plan = self.memory_plan(state.batch_size, state.max_length,
                                tokens=tokens.shape[1], all_logits=all_logits)
        self._check_memory(plan, 0)
        hidden = self.embedding.forward(tokens)  # Validate every ID before any state mutation.
        try:
            for block, layer in zip(self.blocks, state.layer_states):
                hidden = block.forward(hidden, layer)
            hidden = self.final_norm.forward(hidden if all_logits else hidden[:, -1:])
            logits = project_gguf_linear(hidden, self.index, self.output_name,
                self._head_weights, self._device_head, self.output_name,
                cpu_packed_backend=self.cpu_packed_backend)
        except Exception:
            state.failed = True
            raise
        state.length += tokens.shape[1]
        return logits if all_logits else logits[:, 0]

    def prefill(self, token_ids, state):
        """Append one prompt chunk; return last-position [B,vocab] logits."""
        return self._forward(token_ids, state, all_logits=False)

    def decode_one(self, token_ids, state):
        tokens = xp.asarray(token_ids)
        if tokens.ndim == 1:
            tokens = tokens[:, None]
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError('decode_one expects [B] or [B,1] token IDs')
        return self._forward(tokens, state, all_logits=False)

    def forward(self, token_ids):
        """All causal logits [B,T,vocab], intended for bounded parity fixtures."""
        tokens = xp.asarray(token_ids)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError('token_ids must have shape [positive batch,positive time]')
        state = self.create_generation_state(tokens.shape[0], tokens.shape[1])
        return self._forward(tokens, state, all_logits=True)

    def close(self):
        if not self._closed and self._owns_executor:
            self.expert_executor.close()
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
