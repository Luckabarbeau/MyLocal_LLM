"""Opt-in complete short-context QSA decoder block over a frozen native plan.

No model dispatch is installed here. The caller excludes PLE/diagnostics, makes
all projection callbacks capture-ready, keeps the plan/service and its protected
resident leases alive, and performs the final global validation/service check.
Outputs are borrowed until the next serialized replay. A nonzero device status
invalidates the request: length advances speculatively, as in attention.step.
"""
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.finite_checks_gpu import finite_status_gpu
from mini_llm.ops.inference_graph import CapturedArrayInference
from mini_llm.ops.qsa_graph_gpu import (
    _array, commit_qsa_graph_gpu, pool_qsa_graph_gpu, qsa_decode_graph_gpu,
    set_position_gpu,
)
from mini_llm.ops.rope import get_rope_cos_sin
from .qwen4exp_attention import Qwen4ExpAttention, _integer
from .qwen4exp_qsa_frontend_graph import QSAFrontendGraph
from .qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph


class Qwen4ExpQSABlockGraph(Qwen4ExpRecurrentBlockGraph):
    """Reuse native lifecycle only; the entire captured computational graph is QSA."""
    pool_limit_bytes = 8*1024**2
    complete_block = True

    def __init__(self, block, plan, validation_status):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('Complete QSA block graph requires CuPy')
        if block.recurrent or block.ple is not None or not isinstance(block.mixer, Qwen4ExpAttention):
            raise ValueError('Complete QSA graph requires a QSA block without PLE')
        a, router = block.mixer, block.moe.router
        if (not a.fast_decode or not a._frontend_capturable or
                not block.attention_hc.native_read or not block.ffn_hc.native_read or
                not router._decode_fastpath_enabled or not getattr(router, '_capturable_logits', False)):
            raise ValueError('QSA graph requires native HC and capture-ready attention/router')
        _array('validation_status', validation_status, xp.uint32, ())
        self.block, self.plan = block, plan
        self.handoff, self.fingerprint = plan.native_handoff, plan.fingerprint
        self.attention = a
        self.attention_hc, self.ffn_hc, self.router = block.attention_hc, block.ffn_hc, router
        self.validation_status = validation_status
        self.graph = None
        self.calls = self.captures = 0
        self.closed = False
        self.bind_state = True
        self._warm_profile_before = None
        self.table = None
        self._validate_plan()  # Inherited metadata-only check; never service.check().
        # Retirement must precede any new pool allocation, not occur in capture.
        self.handoff.flush_statistics()
        retired = set()
        frontend = a._frontend_graph
        for owner, attribute in ((self.handoff, 'graph'), (block.moe, '_shared_graph'),
                                 (frontend, 'graph')):
            if owner is None:
                continue
            graph = getattr(owner, attribute)
            if graph is not None:
                if id(graph) not in retired:
                    graph.close()
                    retired.add(id(graph))
                setattr(owner, attribute, None)
        self.handoff.graph_branch = None
        # Only tiny runtime operands are owned here, never full projection tables.
        self.frontend = QSAFrontendGraph(a)
        self.position = xp.empty((), dtype=xp.int64)

    def _enqueue(self, streams, arrays):
        stream = xp.cuda.get_current_stream()
        warming = not stream.is_capturing()
        try:
            keys, values, history, pooled, status, position, cos, sin, pool_cos, pool_sin = arrays
            a, c = self.attention, self.attention.config
            hidden, gains = self.attention_hc.read(streams)
            qg, raw_k, v, raw, q, k = self.frontend._enqueue(hidden, (cos, sin))
            # Always enqueue the same graph; incomplete blocks produce zeros and
            # never read uncommitted raw history. Only complete pools are written.
            pool = pool_qsa_graph_gpu(history, raw, position, c.block_size)
            pool = self.frontend._rotate(a.norms[3].forward(pool[None]), pool_cos, pool_sin)[0]
            scale = c.attention_scale if c.attention_scale is not None else 1 / np.sqrt(c.head_dim)
            gated = qsa_decode_graph_gpu(q, qg, k, v, keys, values, position, scale)
            mixed = a._project('output', gated, check_finite=False)
            finite_status_gpu(hidden, qg, raw_k, v, raw, mixed, out=status)
            commit_qsa_graph_gpu(status, keys, values, history, pooled, k, v, raw, pool,
                                 position, c.block_size)
            residual = self.attention_hc.write(streams, mixed, gains)
            hidden, gains = self.ffn_hc.read(residual)
            weights, ids = self.router.forward(hidden.reshape(1, 1, -1))
            routed_shared = self.handoff.enqueue(hidden.reshape(1, -1),
                (weights*self.block.moe.scale).reshape(-1), ids.reshape(-1), None,
                gpu_branch=self.block.moe)
            output = self.ffn_hc.write(residual, routed_shared.reshape(-1), gains)
        finally:
            # The inherited accounting excludes private native batches but keeps
            # errors visible. No readback or Python accounting on graph replay.
            if warming:
                stream.synchronize()
                self._account_warmup()
        return output

    def _operands(self, streams, state, position):
        a, c = self.attention, self.attention.config
        a._state(state)  # Preserve original owner/length/capacity/shape contract.
        t = _integer(position, 'scalar text position', 0)
        if t != state.length:
            raise ValueError('position must equal state.length (zero-origin contiguous text)')
        if t >= state.max_length:
            raise ValueError('QSA cache capacity exceeded')
        width = c.index_topk + c.block_size - 1
        if t + 1 > width or t >= np.iinfo(np.int32).max:
            raise ValueError('Complete QSA graph requires the bounded all-cells context')
        _array('streams', streams, xp.float32, (4, c.d_model))
        caches = state.keys, state.values, state.raw_index, state.pooled
        for name, array in zip(('keys', 'values', 'raw_index', 'pooled'), caches):
            _array(name, array, xp.float32)
        _array('validation_status', self.validation_status, xp.uint32, ())
        if self.table is None or self.table[0] != state.max_length:
            self.table = (state.max_length, *get_rope_cos_sin(
                state.max_length, c.rotary_dim, c.rope_base, xp.float32))
        start = t + 1 - c.block_size if (t + 1) % c.block_size == 0 else 0
        factors = tuple(table[0, p, 0].reshape(1, -1)
                        for p in (t, start) for table in self.table[1:])
        for factor in factors:
            _array('RoPE factor', factor, xp.float32, (1, c.rotary_dim // 2))
        return t, (*caches, self.validation_status, self.position, *factors)

    def forward(self, streams, state, position):
        if self.closed:
            raise ValueError('QSA block graph is closed')
        self._validate_plan()
        try:
            t, arrays = self._operands(streams, state, position)
            if self.graph is not None and not self.graph.matches(streams, arrays):
                # Bound cache identity, not merely shape, defines a request.
                self.graph.close()
                self.graph = None
            max_position = min(state.max_length,
                self.attention.config.index_topk + self.attention.config.block_size - 1,
                np.iinfo(np.int32).max) - 1
            set_position_gpu(self.position, t, max_position)
            if self.graph is None:
                self.handoff.flush_statistics()
                if self.handoff.route_heat is not None:
                    self._warm_profile_before = self.handoff.read_profile().copy()
                self.graph = CapturedArrayInference(self._enqueue, streams, arrays,
                    pool_limit_bytes=self.pool_limit_bytes, copy_back_mutable=False,
                    bound_mutable=(0, 1, 2, 3, 4))
                self.captures += 1
            # Position/factors copy into fixed graph operands; cache/status bind
            # only after private warmup. Read-only inputs are never copied back.
            output = self.graph.forward(streams, arrays, borrow_output=True)
            self.calls += 1
            state.length = t + 1
            return output
        except BaseException:
            if self.graph is not None and not self.graph.calls:
                self.graph.close()
                self.graph = None
            raise
        finally:
            # The native join records an event inside capture. Supersede it with
            # a full-block fence OUTSIDE capture, before request-bound retirement.
            stream = xp.cuda.get_current_stream()
            if not stream.is_capturing():
                self.plan.retired.record(stream)
                self.plan.pending = True

    def graph_statistics(self):
        statistics = (self.graph.statistics() if self.graph is not None else
                      dict(pool_reserved_bytes=0, pool_used_bytes=0))
        return dict(statistics, calls=self.calls, captures=self.captures,
            scope='qsa_decoder_block', complete_block=True, native_handoff=True,
            cpu_handoff='native_mapped_service', bound_state=True,
            runtime_position=True, serial_fp32_attention=True,
            gpu_arithmetic='q8_1' if self.plan.q8 is not None else 'fp32', closed=self.closed)

    # close(), _validate_plan() and _account_warmup() deliberately use the same
    # borrowed-plan lifecycle as the complete recurrent block implementation.
