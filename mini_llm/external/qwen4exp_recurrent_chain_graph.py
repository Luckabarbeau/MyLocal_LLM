"""Replay two or three adjacent complete recurrent blocks in one CUDA graph.

Members and their frozen plans/native service are borrowed, not owned. The caller
must close the chain BEFORE closing any member, plan or executor. Do not replay a
member independently while the chain is live. Outputs are borrowed until the next
replay, with the same serialized-stream ownership as Qwen4ExpPreMoEGraph.
"""
from collections.abc import Sequence
from typing import Any

from mini_llm.backend import xp as _xp
from mini_llm.ops.gated_delta_inference import DeltaMixerState
from mini_llm.ops.inference_graph import CapturedArrayInference
from .qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph

xp: Any = _xp


class Qwen4ExpRecurrentChainGraph:
    complete_block = True

    def __init__(self, members: Sequence[Qwen4ExpRecurrentBlockGraph]):
        members = tuple(members)
        if not 2 <= len(members) <= 3:
            raise ValueError('Recurrent chain requires two or three complete blocks')
        if any(not isinstance(m, Qwen4ExpRecurrentBlockGraph) for m in members):
            raise ValueError('Recurrent chain members must be complete recurrent block graphs')
        self.members = members
        self.plan = members[0].plan
        self.bind_state = members[0].bind_state
        self._validate_members()
        # Retire whole-block pools, not the member wrappers or their native plans.
        # Validate the entire chain first, so a rejected upgrade is nondestructive.
        retired = set()
        for member in members:
            if member.graph is not None:
                if id(member.graph) not in retired:
                    member.graph.close()
                    retired.add(id(member.graph))
                member.graph = None
        self.pool_limit_bytes = 16 * 1024**2 * len(members)
        self.graph = None
        self.calls = 0
        self.captures = 0
        self.closed = False

    def _validate_members(self):
        head = self.plan
        for offset, member in enumerate(self.members):
            p = member.plan
            if (member.closed or not member.complete_block or not member.block.recurrent or
                    member.block.ple is not None):
                raise ValueError('Recurrent chain requires live complete recurrent blocks without PLE')
            if (p.executor is not head.executor or p.layer != head.layer + offset or
                    p.device != head.device or p.stream_ptr != head.stream_ptr):
                raise ValueError('Recurrent chain requires consecutive plans on the same executor/device/stream')
            if member.bind_state != self.bind_state:
                raise ValueError('Recurrent chain members must share the bound state policy')
            member._validate_plan()  # Metadata only; no service readback per token.

    def _validate_states(self, streams, states):
        if len(states) != len(self.members):
            raise ValueError('Recurrent chain state count must match its members')
        arrays = []
        length = None
        for member, state in zip(self.members, states):
            spec = member.mixer.spec
            if (not isinstance(streams, xp.ndarray) or
                    streams.shape != (4, spec.d_model) or streams.dtype != xp.float32 or
                    not isinstance(state, DeltaMixerState) or
                    state.conv.length != state.delta.length or state.length < 0 or
                    state.conv.history.shape != (1, spec.conv_kernel_size-1, spec.qkv_channels) or
                    state.delta.memory.shape != (1, spec.n_value_heads, spec.head_dim, spec.head_dim)):
                raise ValueError('Recurrent chain state/input geometry or positions mismatch')
            if length is not None and state.length != length:
                raise ValueError('Recurrent chain state positions must agree across all members')
            length = state.length
            arrays.extend((state.conv.history, state.delta.memory))
        if any(not isinstance(a, xp.ndarray) or a.dtype != xp.float32 or
               not a.flags.c_contiguous for a in (streams, *arrays)):
            raise ValueError('Recurrent chain operands must be contiguous FP32')
        if any(a.device.id != self.plan.device for a in (streams, *arrays)):
            raise ValueError('Recurrent chain operands must be on the plan device')
        return tuple(arrays)

    def _enqueue(self, streams, arrays):
        for j, member in enumerate(self.members):
            streams = member._enqueue(streams, arrays[2*j:2*j+2])
        return streams

    def forward(self, streams, states: Sequence[DeltaMixerState]) -> Any:
        if self.closed:
            raise ValueError('Recurrent chain graph is closed')
        self._validate_members()
        states = tuple(states)
        # Validate every block before warmup, pool retirement or any GPU enqueue.
        arrays = self._validate_states(streams, states)
        recapture = self.graph is None or (self.bind_state and
                                          not self.graph.matches(streams, arrays))
        try:
            if recapture:
                if self.graph is not None:
                    self.graph.close()
                    self.graph = None
                for member in self.members:
                    member.handoff.flush_statistics()
                    if member.handoff.route_heat is not None:
                        member._warm_profile_before = member.handoff.read_profile().copy()
                # Each member drains/accounts native work only during private
                # warmup. Capture calls the same enqueue with no host callbacks.
                self.graph = CapturedArrayInference(self._enqueue, streams, arrays,
                    pool_limit_bytes=self.pool_limit_bytes,
                    bound_mutable=tuple(range(len(arrays))) if self.bind_state else ())
                self.captures += 1
            result = self.graph.forward(streams, arrays, borrow_output=True)
            self.calls += 1
            for state in states:
                state.conv.length = int(state.conv.length) + 1
                state.delta.length = int(state.delta.length) + 1
            return result
        except BaseException:
            # Construction failure leaves no assigned executable; the capture
            # wrapper/member warmup drains private native users before unwinding.
            if self.graph is not None and not self.graph.calls:
                self.graph.close()
                self.graph = None
            raise
        finally:
            # Captured event records are not valid external retirement fences.
            # Cover every plan, including failed construction/first replay, and
            # never record into a failed/active capture (0313 lifetime rule).
            stream = xp.cuda.get_current_stream()
            if not stream.is_capturing():
                for member in self.members:
                    member.plan.retired.record(stream)
                    member.plan.pending = True

    def graph_statistics(self) -> dict[str, Any]:
        statistics = (self.graph.statistics() if self.graph is not None else
                      dict(pool_reserved_bytes=0, pool_used_bytes=0))
        return dict(statistics, calls=self.calls, captures=self.captures,
                    bound_state=self.bind_state, pool_limit_bytes=self.pool_limit_bytes,
                    scope='recurrent_decoder_chain', complete_block=True,
                    chain_length=len(self.members), native_handoff=True,
                    cpu_handoff='native_mapped_service', closed=self.closed)

    def close(self):
        """Fence the whole chain before flushing borrowed native handoffs."""
        if self.closed:
            return
        if xp.cuda.get_current_stream().is_capturing():
            raise ValueError('Recurrent chain retirement must be outside capture')
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self.closed = True
        for member in self.members:
            member.handoff.flush_statistics()
