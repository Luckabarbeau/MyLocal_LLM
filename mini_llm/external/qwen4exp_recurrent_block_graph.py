"""Whole recurrent decoder replay over an already-frozen native expert plan.

The caller owns the plan/service and must close this graph before retiring them.
Outputs are borrowed until the next replay, as for Qwen4ExpPreMoEGraph.
"""
from typing import Any

from mini_llm.backend import xp as _xp
from .qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph

xp: Any = _xp


class Qwen4ExpRecurrentBlockGraph(Qwen4ExpPreMoEGraph):
    complete_block = True

    def __init__(self, segment, block, plan):
        if not block.recurrent or block.ple is not None:
            raise ValueError('Complete recurrent graph requires a recurrent block without PLE')
        if (segment.attention_hc is not block.attention_hc or
                segment.mixer is not block.mixer or segment.ffn_hc is not block.ffn_hc or
                segment.router is not block.moe.router):
            raise ValueError('Recurrent segment and block must share their operators')
        super().__init__(segment.attention_hc, segment.mixer, segment.ffn_hc, segment.router)
        self.bind_state = segment.bind_state
        self.block, self.plan = block, plan
        self.handoff = plan.native_handoff
        self.fingerprint = plan.fingerprint
        self.closed = False
        self._warm_profile_before = None
        self._validate_plan()
        # Retire all older pools outside capture. In particular, the handoff's
        # retirement event precedes the final HC write, so it is not sufficient
        # to retire the old segment/shared graphs on its own.
        self.handoff.flush_statistics()
        retired = set()
        for owner, attribute in ((segment, 'graph'), (self.handoff, 'graph'),
                                 (block.moe, '_shared_graph')):
            graph = getattr(owner, attribute)
            if graph is not None:
                if id(graph) not in retired:
                    graph.close()
                    retired.add(id(graph))
                setattr(owner, attribute, None)
        self.handoff.graph_branch = None

    def _validate_plan(self):
        """Metadata only; blocking service readback belongs at the token boundary."""
        p, h = self.plan, self.handoff
        e = p.executor
        if (e.cache._admission_limit != 0 or e.cache.idle_eviction_requests or
                p.fingerprint != self.fingerprint or
                p.cache_fingerprint(e) != self.fingerprint):
            raise ValueError('Recurrent graph requires the current frozen cache fingerprint')
        if (e._device_expert_plans.get(p.layer) is not p or
                p.native_handoff is not h or h is None or h.plan is not p or
                self.block.moe.experts.expert_executor is not e or
                self.block.moe.experts.layer_id != p.layer):
            raise ValueError('Recurrent graph requires the live native plan identity')
        if (not h.capture_enabled or h.service is None or h.stage is not None or
                not h.handle or not callable(h.enqueue) or
                not callable(self.block.moe.enqueue_gpu_branch) or
                not callable(self.block.moe.combine_gpu_branch)):
            raise ValueError('Recurrent graph requires an unstaged captured native service')
        stream = xp.cuda.get_current_stream()
        if stream.is_capturing():
            raise ValueError('Recurrent graph setup/replay must be outside capture')
        if stream.ptr != p.stream_ptr or xp.cuda.runtime.getDevice() != p.device:
            raise ValueError('Recurrent graph must reuse the original plan replay stream/device')

    def _enqueue(self, streams, arrays):
        stream = xp.cuda.get_current_stream()
        warming = not stream.is_capturing()
        try:
            residual, hidden, gains, weights, ids = super()._enqueue(streams, arrays)
            routed_shared = self.handoff.enqueue(hidden.reshape(1, -1),
                (weights*self.block.moe.scale).reshape(-1), ids.reshape(-1), None,
                gpu_branch=self.block.moe)
            output = self.ffn_hc.write(residual, routed_shared.reshape(-1), gains)
        finally:
            # Only private warmup executes a native batch here. During capture
            # this method merely enqueues GPU work; no host statistics/callbacks.
            if warming:
                stream.synchronize()
                self._account_warmup()
        return output

    def _account_warmup(self):
        h = self.handoff
        h.bridge.mylocal_ggml_handoff_stats(h.handle, h.stats.ctypes.data)
        if h.stats[3] != h.accounted[3]:
            # Leave errors unaccounted so flush_statistics cannot lose them.
            raise RuntimeError('Native recurrent block graph warmup failed')
        h.service.check()
        if h.stats[0] != h.accounted[0]:
            h.accounted[:] = h.stats
            if h.timeline is not None:
                h.timeline_warmup_ordinals.append(int(h.stats[0]))
            if h.route_heat is not None:
                # Preserve production heat across request-bound recaptures.
                h.profile_warmup[:] += h.read_profile()-self._warm_profile_before

    def forward(self, streams, state) -> Any:
        if self.closed:
            raise ValueError('Recurrent block graph is closed')
        self._validate_plan()
        arrays = (state.conv.history, state.delta.memory) if hasattr(state, 'conv') and hasattr(state, 'delta') else ()
        recapture = self.graph is None or (self.bind_state and
                                          not self.graph.matches(streams, arrays))
        if recapture:
            self.handoff.flush_statistics()
            if self.handoff.route_heat is not None:
                self._warm_profile_before = self.handoff.read_profile().copy()
        try:
            return super().forward(streams, state)
        except BaseException:
            # If construction succeeded but the first replay was rejected,
            # retire its executable before any caller releases native handles.
            if self.graph is not None and not self.graph.calls:
                self.graph.close()
                self.graph = None
            raise
        finally:
            # Supersede captured event records before a request's graph retires.
            # Match DeviceExpertPlan.forward without recording into failed capture.
            stream = xp.cuda.get_current_stream()
            if not stream.is_capturing():
                self.plan.retired.record(stream)
                self.plan.pending = True

    def graph_statistics(self) -> dict[str, Any]:
        return dict(super().graph_statistics(), scope='recurrent_decoder_block',
                    complete_block=True, native_handoff=True,
                    cpu_handoff='native_mapped_service',
                    gpu_arithmetic='q8_1' if self.plan.q8 is not None else 'fp32',
                    closed=self.closed)

    def close(self):
        """Fence the whole layer before the borrowed plan/service can be freed."""
        if self.closed:
            return
        if xp.cuda.get_current_stream().is_capturing():
            raise ValueError('Recurrent graph retirement must be outside capture')
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self.closed = True
        self.handoff.flush_statistics()
