"""Short-context QSA projection/norm/RoPE replay with runtime factor inputs.

The KV/indexer algorithm, validation-before-commit and position handling remain
in Qwen4ExpAttention.step. No position or selection width is frozen in a graph.
"""
import os
from mini_llm.backend import xp
from mini_llm.ops.inference_graph import CapturedArrayInference
from mini_llm.ops.rope import get_rope_cos_sin


class QSAFrontendGraph:
    pool_limit_bytes = 2*1024**2

    def __init__(self, attention):
        self.attention = attention
        self.graph = None
        self.table = None

    def _rotate(self, value, cos, sin):
        half = self.attention.config.rotary_dim//2
        out = value.copy()
        a, b = value[..., :half], value[..., half:2*half]
        # Same separate FP32 products/additions as RotaryEmbeddingInference.
        out[..., :half] = a*cos-b*sin
        out[..., half:2*half] = a*sin+b*cos
        return out

    def _enqueue(self, hidden, arrays):
        a = self.attention
        c = a.config
        cos, sin = arrays
        qg = a._project('query_gate', hidden, check_finite=False).reshape(c.n_q_heads, 2*c.head_dim)
        raw_k = a._project('key', hidden, check_finite=False).reshape(c.n_kv_heads, c.head_dim)
        v = a._project('value', hidden, check_finite=False).reshape(c.n_kv_heads, c.head_dim)
        raw = a._project('index_key', hidden, check_finite=False)
        q = self._rotate(a.norms[0].forward(qg[:, :c.head_dim]), cos, sin)
        k = self._rotate(a.norms[1].forward(raw_k), cos, sin)
        return qg, raw_k, v, raw, q, k

    def forward(self, hidden, position):
        c = self.attention.config
        end = position+1
        if self.table is None or self.table[0]<end:
            capacity = 1 << (end-1).bit_length()
            cos, sin = get_rope_cos_sin(capacity,c.rotary_dim,c.rope_base,xp.float32)
            self.table = capacity, cos, sin
        factors = tuple(a[0,position,0].reshape(1,-1) for a in self.table[1:])
        if self.graph is None:
            self.graph = CapturedArrayInference(self._enqueue,hidden,factors,
                                               pool_limit_bytes=self.pool_limit_bytes,
                copy_back_mutable=os.environ.get('MINI_LLM_GRAPH_READONLY_INPUTS','0')!='1')
        return self.graph.forward(hidden,factors,borrow_output=True)

    def statistics(self):
        return self.graph.statistics() if self.graph is not None else dict(calls=0,pool_reserved_bytes=0)
