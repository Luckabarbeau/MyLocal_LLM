"""Whole-token replay over borrowed, already-frozen native expert plans.

No dispatch is installed here. The parent serializes calls, validates the device
status/native service after forward, increments generation-state.length, and
closes this adapter before releasing any plan, service, or canonical weight.
Injected component fixtures must supply capture-ready pure GPU projections;
GGUF models additionally require already-resident projection payloads.
"""
import os
from typing import Any, cast

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp as _xp
from mini_llm.ops.gated_delta_inference import DeltaMixerState
from mini_llm.ops.inference_graph import CapturedArrayInference
from mini_llm.ops.qsa_graph_gpu import _array, set_position_gpu
from .qwen4exp_attention import QSAConfig
from .qwen4exp_ple import PLEState, _integer
from .qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
from .qwen4exp_qsa_block_graph import Qwen4ExpQSABlockGraph
from .qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph


xp: Any = _xp


class _NativeBlock:
    """Only the validated lifecycle math; never owns a graph or native lease."""
    _validate_plan = Qwen4ExpRecurrentBlockGraph._validate_plan
    _account_warmup = Qwen4ExpRecurrentBlockGraph._account_warmup

    def __init__(self, block, plan):
        self.block, self.plan = block, plan
        self.handoff, self.fingerprint = plan.native_handoff, plan.fingerprint
        self._warm_profile_before = None


class Qwen4ExpTokenGraph:
    pool_limit_bytes = 256 * 1024**2
    qsa_cache_limit_bytes = 6 * 1024**2

    def __init__(self, model):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('Whole-token graph requires CuPy')
        self.model = model
        self.graph = None
        self.closed = False
        self.calls = self.captures = 0
        self._native, self._qsa, self._ple_layer = self._configuration(model)
        self._source = self._source_identity()
        self._pre = {
            i: Qwen4ExpPreMoEGraph(b.attention_hc, b.mixer, b.ffn_hc, b.moe.router)
            for i, b in enumerate(model.blocks) if b.recurrent
        }
        # Fixed indices are independent of request/token values. Only caches and
        # scalar status are bound; history and runtime RoPE inputs are read-only.
        self._layout = []
        bound, offset = [], 0
        for b in model.blocks:
            size = 2 if b.recurrent else 5
            self._layout.append((offset, size))
            bound.extend(range(offset, offset + size))
            offset += size
        self._history_index = offset if self._ple_layer is not None else None
        offset += self._ple_layer is not None
        self._runtime_index = offset
        self._bound = tuple(bound)

    @staticmethod
    def _configuration(model):
        """Read metadata only: no projection calls, state writes, pools or I/O."""
        if BACKEND_NAME != 'cupy':
            raise ValueError('Whole-token graph requires CuPy')
        if model._closed or model.diagnostic_callback is not None or not model.blocks:
            raise ValueError('Whole-token graph requires a live model without diagnostics')
        width = model.config.hidden_size
        if (not model.final_hc.native_read or not model.final_hc.fast_decode or
                model.final_hc.graph_decode or model.final_hc.inject is not None or
                model.final_hc.width != width):
            raise ValueError('Whole-token graph requires a native read-only final HC')
        ple_layers = tuple(i for i, b in enumerate(model.blocks) if b.ple is not None)
        if ple_layers not in ((), (1,)):
            raise ValueError('Whole-token graph supports at most one PLE, at layer 1')
        native, qsa = [], {}
        first_config = None
        executor = None
        for i, b in enumerate(model.blocks):
            if (not b.attention_hc.native_read or not b.ffn_hc.native_read or
                    not b.attention_hc.fast_decode or not b.ffn_hc.fast_decode or
                    b.attention_hc.graph_decode or b.ffn_hc.graph_decode or
                    b.attention_hc.width != width or b.ffn_hc.width != width or
                    not b.moe.router._decode_fastpath_enabled or
                    not getattr(b.moe.router, '_capturable_logits', False) or
                    not getattr(b.moe, '_shared_graph_enabled', False)):
                raise ValueError('Whole-token graph requires native HC/router and capture-ready shared projections')
            e = b.moe.experts.expert_executor
            if executor is not None and e is not executor:
                raise ValueError('Whole-token graph requires one frozen executor')
            executor = e
            p = e._device_expert_plans.get(i)
            if p is None or p.layer != i or p.native_handoff is None:
                raise ValueError('Whole-token graph requires every native plan already built')
            context = _NativeBlock(b, p)
            context._validate_plan()
            native.append(context)
            if i and (p.device != native[0].plan.device or p.stream_ptr != native[0].plan.stream_ptr or
                      p.fingerprint != native[0].fingerprint):
                raise ValueError('Whole-token plans must share fingerprint/device/stream')
            if b.recurrent:
                if (b.mixer.graph_decode or b.mixer.spec.d_model != width or
                        not b.mixer.mixer._decode_fastpath_enabled):
                    raise ValueError('Whole-token graph requires unnested native recurrent math')
            else:
                a = getattr(model, '_qsa_block_segments', {}).get(i)
                if (not isinstance(a, Qwen4ExpQSABlockGraph) or a.closed or
                        not a.complete_block or a.block is not b or a.plan is not p or
                        a.handoff is not p.native_handoff or a.fingerprint != p.fingerprint or
                        a.attention is not b.mixer or not b.mixer.fast_decode or
                        not b.mixer._frontend_capturable):
                    raise ValueError('Whole-token graph requires every complete QSA adapter already built')
                c = b.mixer.config
                if not isinstance(c, QSAConfig) or c.d_model != width:
                    raise ValueError('Unsupported whole-token QSA architecture')
                if first_config is not None and c != first_config:
                    raise ValueError('All QSA configurations and runtime factors must match exactly')
                first_config = c
                a._validate_plan()
                _array('QSA validation status', a.validation_status, xp.uint32, ())
                qsa[i] = a
            if b.ple is not None and (not b.recurrent or b.ple.width != width or
                    getattr(b.ple, '_closed', False) or not callable(getattr(b.ple, 'apply_embedding', None))):
                raise ValueError('Whole-token PLE requires capture-ready apply_embedding')
        weights = getattr(model, 'weights', None)
        if weights is not None:
            if (weights.closed or weights.store.closed or weights.plan is None or
                    len(model.blocks) != 48 or len(qsa) != 12 or
                    tuple(b.recurrent for b in model.blocks) != weights.headers.recurrent or
                    first_config != weights.headers.attention):
                raise ValueError('Whole-token graph requires the validated 48-layer canonical architecture')
            # Metadata/resident handles only; never load or copy a weight here.
            required = ['output.weight']
            for i, b in enumerate(model.blocks):
                names = (('attn_qkv', 'attn_gate', 'ssm_beta', 'ssm_alpha', 'ssm_out') if b.recurrent else
                         ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'indexer.k_proj'))
                names += ('ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp')
                if b.ple is not None:
                    names += ('ple_key', 'ple_value')
                required.extend(f'blk.{i}.{name}.weight' for name in names)
            for name in required:
                desc = weights.store.tensors[name]
                if desc.quant_type not in ('F32', 'F16', 'BF16') and name not in weights.device:
                    raise ValueError(f'Whole-token projection must already be GPU resident: {name}')
            from mini_llm.ops.mixed_q8_gpu import FORMATS
            output = weights.device.get('output.weight')
            if output is None or output.quant_type not in FORMATS:
                raise ValueError('Whole-token output requires a resident supported packed GPU projection')
            if os.environ.get('MINI_LLM_LINEAR_GPU_Q8') != '1':
                raise ValueError('Whole-token output requires MINI_LLM_LINEAR_GPU_Q8=1 activation compute')
        return tuple(native), qsa, ple_layers[0] if ple_layers else None

    def _source_identity(self):
        m = self.model
        return (id(m.embedding), id(m.output), id(m.final_hc),
                tuple((id(b), id(b.attention_hc), id(b.mixer), id(b.ffn_hc), id(b.moe), id(b.ple))
                      for b in m.blocks),
                tuple((name, id(value)) for name, value in sorted(
                    getattr(getattr(m, 'weights', None), 'device', {}).items())))

    def _validate_live(self):
        if self.closed:
            raise ValueError('Whole-token graph is closed')
        for context in self._native:
            context._validate_plan()
        if self._source_identity() != self._source:
            raise ValueError('Whole-token graph canonical source identity changed')
        # Operator flags/configuration cannot silently switch an existing graph
        # to a different arithmetic path, even before its first capture.
        native, qsa, ple = self._configuration(self.model)
        if (ple != self._ple_layer or any(a.plan is not b.plan for a, b in zip(native, self._native)) or
                any(qsa[i] is not a for i, a in self._qsa.items())):
            raise ValueError('Whole-token graph borrowed adapters changed')

    @classmethod
    def is_eligible(cls, model, state):
        """Metadata-only preflight. A live parent's _token_graph fails closed."""
        live = getattr(model, '_token_graph', None)
        if isinstance(live, cls) and not live.closed:
            live._validate_live()  # Never turn invalid live native leases into fallback.
        try:
            native, qsa, ple = cls._configuration(model)
            cls._validate_states(model, state, native, qsa, ple)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return False
        return True

    @classmethod
    def _validate_states(cls, model, state, native, qsa, ple):
        model._validate(state)
        t = _integer(state.length, 'state.length')
        if t >= state.max_length:
            raise ValueError('Whole-token generation state capacity exceeded')
        rec_ids = {i for i, b in enumerate(model.blocks) if b.recurrent}
        if (set(state.recurrent) != rec_ids or set(state.qsa) != set(qsa) or
                set(state.ple) != (set() if ple is None else {ple})):
            raise ValueError('Whole-token state layer layout mismatch')
        arrays = []
        for i, b in enumerate(model.blocks):
            if b.recurrent:
                s, spec = state.recurrent[i], b.mixer.spec
                if (not isinstance(s, DeltaMixerState) or cast(Any, s.conv).length != t or s.delta.length != t):
                    raise ValueError('Whole-token recurrent state positions mismatch')
                _array('conv history', cast(Any, s.conv).history, xp.float32,
                       (1, spec.conv_kernel_size - 1, spec.qkv_channels))
                _array('delta memory', s.delta.memory, xp.float32,
                       (1, spec.n_value_heads, spec.head_dim, spec.head_dim))
                arrays.extend((cast(Any, s.conv).history, s.delta.memory))
            else:
                s, a = state.qsa[i], qsa[i]
                a.attention._state(s)
                c = a.attention.config
                if (s.length != t or s.max_length != state.max_length or
                        t + 1 > c.index_topk + c.block_size - 1 or t >= np.iinfo(np.int32).max):
                    raise ValueError('Whole-token QSA requires contiguous bounded all-cells context')
                caches = s.keys, s.values, s.raw_index, s.pooled
                if sum(x.nbytes for x in caches) > cls.qsa_cache_limit_bytes:
                    raise ValueError('Whole-token QSA cache payload exceeds 6 MiB per layer')
                for x in caches:
                    _array('QSA cache', x, xp.float32)
                _array('QSA status', a.validation_status, xp.uint32, ())
                arrays.extend((*caches, a.validation_status))
        if ple is not None:
            s = state.ple[ple]
            if not isinstance(s, PLEState) or len(s.predecessors) != 2:
                raise ValueError('Whole-token PLE requires explicit immutable history')
            for token in s.predecessors:
                _integer(token, 'PLE predecessor', -1, 2**31 - 1)
            _array('PLE history', s.normalized_history, xp.float32, (9, 4, model.config.hidden_size))
            arrays.append(s.normalized_history)
        # Check aliases on *every* call, not just the first pool construction.
        # This includes read-only PLE history, which may not alias bound caches.
        ranges = []
        for a in arrays:
            if a.nbytes:
                start = a.data.ptr
                ranges.append((start, start + a.nbytes))
        ranges.sort()
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            if start < end:
                raise ValueError('Whole-token state/status operands must not alias')
        if any(a.device.id != native[0].plan.device for a in arrays):
            raise ValueError('Whole-token state must be on the native plan device')
        return tuple(arrays)

    def _retire_old_pools(self):
        retired, visited = set(), set()

        def retire(owner, attribute):
            if owner is None:
                return
            graph = getattr(owner, attribute, None)
            if graph is not None:
                if id(graph) not in retired:
                    graph.close()
                    retired.add(id(graph))
                setattr(owner, attribute, None)

        def segment(owner):
            if id(owner) in visited:
                return
            visited.add(id(owner))
            retire(owner, 'graph')
            for member in getattr(owner, 'members', ()):
                segment(member)

        for owner in (*getattr(self.model, '_pre_moe_segments', {}).values(), *self._qsa.values()):
            segment(owner)
        for context in self._native:
            retire(context.handoff, 'graph')
            context.handoff.graph_branch = None
            retire(context.block.moe, '_shared_graph')
            retire(getattr(context.block.mixer, '_frontend_graph', None), 'graph')
        for a in self._qsa.values():
            retire(a.frontend, 'graph')

    def _prepare_warmup(self):
        # All pending production accounting must be flushed before snapshots.
        for context in self._native:
            context.handoff.flush_statistics()
        for i, context in enumerate(self._native):
            h = context.handoff
            before = h.read_profile().copy() if h.route_heat is not None else None
            context._warm_profile_before = before
            if i in self._qsa:
                self._qsa[i]._warm_profile_before = before

    def _enqueue(self, x, arrays):
        stream = xp.cuda.get_current_stream()
        warming = not stream.is_capturing()
        streams = self.model.final_hc.initialize(x[0])
        new_history = None
        for i, (b, context, (start, size)) in enumerate(zip(self.model.blocks, self._native, self._layout)):
            if i == self._ple_layer:
                streams, new_history = b.ple.apply_embedding(streams, x[1], arrays[self._history_index])
            operands = arrays[start:start + size]
            if not b.recurrent:
                operands[4].fill(0)
                streams = self._qsa[i]._enqueue(streams, (*operands, *arrays[self._runtime_index:]))
                continue
            try:
                residual, hidden, gains, weights, ids = self._pre[i]._enqueue(streams, operands)
                mixed = context.handoff.enqueue(hidden.reshape(1, -1),
                    (weights * b.moe.scale).reshape(-1), ids.reshape(-1), None, gpu_branch=b.moe)
                streams = b.ffn_hc.write(residual, mixed.reshape(-1), gains)
            finally:
                if warming:
                    stream.synchronize()
                    context._account_warmup()
        logits = self.model.output(self.model.final_hc.collapse(streams)).reshape(1, -1)
        if logits.shape != (1, self.model.config.vocab_size) or logits.dtype != xp.float32:
            raise ValueError('Whole-token output must be FP32 [1,vocab]')
        return logits if new_history is None else (logits, new_history)

    def forward(self, token, state):
        self._validate_live()
        arrays = self._validate_states(self.model, state, self._native, self._qsa, self._ple_layer)
        token = _integer(token, 'token', 0, self.model.config.vocab_size - 1)
        t = int(state.length)
        # Row I/O and token hashing are deliberately outside the captured graph.
        embeddings = [self.model.embedding(token)]
        old_ple = None
        if self._ple_layer is not None:
            old_ple = state.ple[self._ple_layer]
            embeddings.append(self.model.blocks[self._ple_layer].ple.embedding(token, old_ple.predecessors))
        for embedding in embeddings:
            _array('token embedding', embedding, xp.float32, (self.model.config.hidden_size,))
        x = xp.stack(embeddings)
        try:
            if self._qsa:
                first = next(iter(self._qsa.values()))
                # _operands validates this geometry but never reads its values.
                # Reuse contiguous scratch; broadcasting produces a strided view.
                if not hasattr(self, '_operand_streams'):
                    self._operand_streams = xp.empty((4, x.shape[1]), dtype=xp.float32)
                _, operands = first._operands(self._operand_streams,
                                               state.qsa[first.plan.layer], t)
                runtime = operands[5:]
                c = first.attention.config
                set_position_gpu(runtime[0], t, min(state.max_length,
                    c.index_topk + c.block_size - 1, np.iinfo(np.int32).max) - 1)
                arrays += runtime
            if self.graph is not None and not self.graph.matches(x, arrays):
                self.graph.close()
                self.graph = None
            if self.graph is None:
                self._prepare_warmup()
                self._retire_old_pools()
                try:
                    self.graph = CapturedArrayInference(self._enqueue, x, arrays,
                        pool_limit_bytes=self.pool_limit_bytes, copy_back_mutable=False,
                        bound_mutable=self._bound)
                except xp.cuda.memory.OutOfMemoryError as exc:
                    raise MemoryError('Whole-token graph exceeds its 256 MiB private pool or available device memory') from exc
                self.captures += 1
            # Fresh copies leave the private pool and survive replay/retirement.
            result: Any = self.graph.forward(x, arrays, borrow_output=False)
            replacement = None
            if old_ple is not None:
                logits, history = result
                replacement = PLEState((int(old_ple.predecessors[1]), token), history)
            else:
                logits = result
            for s in state.recurrent.values():
                s.conv.length = t + 1
                s.delta.length = t + 1
            for s in state.qsa.values():
                s.length = t + 1
            if old_ple is not None:
                state.ple[self._ple_layer] = replacement
            self.calls += 1
            return logits
        except BaseException:
            if self.graph is not None and not self.graph.calls:
                self.graph.close()
                self.graph = None
            raise
        finally:
            stream = xp.cuda.get_current_stream()
            if not stream.is_capturing():
                for context in self._native:
                    context.plan.retired.record(stream)
                    context.plan.pending = True

    def graph_statistics(self) -> dict[str, Any]:
        statistics = (self.graph.statistics() if self.graph is not None else
                      dict(pool_reserved_bytes=0, pool_used_bytes=0))
        return dict(statistics, calls=self.calls, captures=self.captures,
                    pool_limit_bytes=self.pool_limit_bytes, scope='whole_token',
                    layers=len(self._native), qsa_layers=len(self._qsa),
                    ple_layers=int(self._ple_layer is not None), bound_state=True,
                    runtime_position=bool(self._qsa), native_handoff=True, closed=self.closed)

    def close(self):
        """Fence the whole token before flushing (never closing) borrowed services."""
        if self.closed:
            return
        if xp.cuda.get_current_stream().is_capturing():
            raise ValueError('Whole-token graph retirement must be outside capture')
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self.closed = True
        for context in self._native:
            context.handoff.flush_statistics()
