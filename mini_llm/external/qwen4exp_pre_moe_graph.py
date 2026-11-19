"""Captured recurrent HC/mixer/FFN-read/router segment, not expert placement.

Outputs are internal borrowed buffers, valid until this segment's next replay.
The model's serialized main-stream dependency chain orders all consumers before
reuse. PLE and QSA stay outside; state copies preserve request isolation.
"""
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.causal_conv_inference import CausalConvState
from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState
from mini_llm.ops.inference_graph import CapturedArrayInference


class Qwen4ExpPreMoEGraph:
    pool_limit_bytes = 16*1024**2

    def __init__(self, attention_hc, mixer, ffn_hc, router):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('Pre-MoE graph requires CuPy')
        if (not attention_hc.native_read or not ffn_hc.native_read or
                mixer.graph_decode or not router._decode_fastpath_enabled or
                not getattr(router, '_capturable_logits', False)):
            raise ValueError('Pre-MoE graph requires native HC, capture-ready router and unnested mixer')
        self.attention_hc, self.mixer, self.ffn_hc, self.router = attention_hc, mixer, ffn_hc, router
        self.graph = None

    def _enqueue(self, streams, arrays):
        state = DeltaMixerState(CausalConvState(arrays[0]), DeltaRuleState(arrays[1]))
        hidden, gains = self.attention_hc.read(streams)
        mixed = self.mixer._forward_uncaptured(hidden[None,None,:], state, capture_owned=True)[0,0]
        residual = self.attention_hc.write(streams, mixed, gains)
        hidden, gains = self.ffn_hc.read(residual)
        weights, ids = self.router.forward(hidden.reshape(1,1,-1))
        return residual, hidden, gains, weights, ids

    def forward(self, streams, state):
        spec = self.mixer.spec
        if (streams.shape != (4,spec.d_model) or streams.dtype != xp.float32 or
                not isinstance(state, DeltaMixerState) or state.conv.length != state.delta.length or
                state.length < 0 or state.conv.history.shape != (1,spec.conv_kernel_size-1,spec.qkv_channels) or
                state.delta.memory.shape != (1,spec.n_value_heads,spec.head_dim,spec.head_dim)):
            raise ValueError('Pre-MoE state/input geometry or positions mismatch')
        arrays = (state.conv.history, state.delta.memory)
        if any(a.dtype != xp.float32 or not a.flags.c_contiguous for a in arrays):
            raise ValueError('Pre-MoE state must be contiguous FP32')
        if self.graph is None:
            self.graph = CapturedArrayInference(self._enqueue, streams, arrays,
                                               pool_limit_bytes=self.pool_limit_bytes)
        result = self.graph.forward(streams, arrays, borrow_output=True)
        state.conv.length = int(state.conv.length)+1
        state.delta.length = int(state.delta.length)+1
        return result

    def graph_statistics(self):
        return (self.graph.statistics() if self.graph is not None else
                dict(calls=0,pool_reserved_bytes=0,pool_used_bytes=0))
