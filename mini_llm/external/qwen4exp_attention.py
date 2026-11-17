"""Patch0208: standalone FP32, single-sequence TEXT QSA; no weight loader.

Positions are contiguous scalar text positions starting at zero. IMRoPE selects
axes interleaved, but rotates HALF-prefix lane pairs (not adjacent lanes).
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from ..backend import BACKEND_NAME, asnumpy, xp
from ..ops.qsa_decode_gpu import qsa_decode_gpu
from ..ops.rmsnorm_inference import RMSNormInference
from ..ops.rotary_inference import RotaryEmbeddingInference
from .qwen4exp_hyperconnection import _fp32, _sigmoid


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class QSAConfig:
    d_model: int
    n_q_heads: int = 24
    n_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    index_heads: int = 4
    index_dim: int = 128
    index_topk: int = 2048
    block_size: int = 4
    rope_base: float = 1e7
    sections: tuple = (11, 11, 10, 0)
    epsilon: float = 1e-6
    attention_scale: float | None = None
    tie_policy: str = "ggml_cpu"

    def __post_init__(self):
        for name in ("d_model", "n_q_heads", "n_kv_heads", "head_dim", "rotary_dim",
                     "index_heads", "index_dim", "index_topk", "block_size"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("query heads must divide into KV head groups")
        if self.rotary_dim % 2 or self.rotary_dim > min(self.head_dim, self.index_dim):
            raise ValueError("rotary_dim must be even and fit both head widths")
        sections = tuple(_integer(v, "section", 0) for v in self.sections)
        # With these bounds every frequency selects one of t/h/w, never extra e.
        if (len(sections) != 4 or sum(sections) != self.rotary_dim // 2 or sections[3]
                or any(sections[a] != len(range(a, self.rotary_dim // 2, 3)) for a in range(3))):
            raise ValueError("sections must cover the interleaved text axes exactly")
        object.__setattr__(self, "sections", sections)
        for name in ("epsilon", "rope_base"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not np.nextafter(np.float32(0), np.float32(1)) <= value <= np.finfo(np.float32).max:
                raise ValueError(f"{name} must be positive finite FP32")
        if self.attention_scale is not None and (not math.isfinite(self.attention_scale)
                                                or self.attention_scale <= 0
                                                or self.attention_scale > np.finfo(np.float32).max):
            raise ValueError("attention_scale must be positive finite FP32")
        if self.tie_policy not in ("ggml_cpu", "strata"):
            raise ValueError("tie_policy must be ggml_cpu or strata")


def select_cells(scores, width, policy="ggml_cpu"):
    """Return ascending cell IDs. CPU policy emulates libstdc++ partial_sort.

    GGML's value-only comparator does NOT specify lower-index ties. This
    reference uses a host score copy, including on CuPy; not a fast GPU kernel.
    """
    _fp32("scores", scores)
    if scores.ndim != 1 or not bool(xp.all(~xp.isnan(scores))):
        raise ValueError("scores must be a vector without NaNs")
    width = _integer(width, "width")
    if width > scores.size:
        raise ValueError("width exceeds score count")
    values = asnumpy(scores)
    if policy == "strata":
        ids = np.lexsort((np.arange(values.size), -values))[:width]
    elif policy == "ggml_cpu":
        # Transcription of libstdc++ __heap_select/__adjust_heap. Sorting the
        # resulting heap cannot change membership, so omit __sort_heap here.
        ids = list(range(width))

        def adjust(hole, value):
            top = hole
            child = hole
            while child < (width - 1) // 2:
                child = 2 * (child + 1)
                if values[ids[child]] > values[ids[child - 1]]:
                    child -= 1
                ids[hole] = ids[child]
                hole = child
            if width % 2 == 0 and child == (width - 2) // 2:
                child = 2 * (child + 1)
                ids[hole] = ids[child - 1]
                hole = child - 1
            parent = (hole - 1) // 2
            while hole > top and values[ids[parent]] > values[value]:
                ids[hole] = ids[parent]
                hole = parent
                parent = (hole - 1) // 2
            ids[hole] = value

        for parent in range((width - 2) // 2, -1, -1):
            adjust(parent, ids[parent])
        for candidate in range(width, values.size):
            if values[candidate] > values[ids[0]]:
                adjust(0, candidate)
    else:
        raise ValueError("unknown tie policy")
    return xp.asarray(np.sort(ids), dtype=xp.int64)


@dataclass
class QSAState:
    """Mutable caller-owned state; do not share across concurrent calls/streams."""
    owner: object
    max_length: int
    keys: object
    values: object
    raw_index: object
    pooled: object
    length: int = 0


class Qwen4ExpAttention:
    """Already-normalized hidden [D] -> attention output [D], no residual.

    Six projections are output-first FP32 matrices or pure vector callbacks.
    Callback outputs must be FP32 backend arrays. Converted norm gamma is used
    directly. State commits only after projection/shape validation and output.
    fast_decode opts into CuPy all-causal-cell attention without index scoring;
    diagnostics and contexts beyond the selection width retain the reference.
    """

    def __init__(self, config, *, query_gate, key, value, output, index_query,
                 index_key, gamma_q, gamma_k, gamma_index_q, gamma_index_k,
                                  fast_decode=False):
        if not isinstance(config, QSAConfig):
            raise TypeError("config must be QSAConfig")
        self.config = c = config
        self.fast_decode = bool(fast_decode)
        self._owner = object()
        self.projections = {}
        sizes = {"query_gate": (2*c.n_q_heads*c.head_dim, c.d_model),
                 "key": (c.n_kv_heads*c.head_dim, c.d_model),
                 "value": (c.n_kv_heads*c.head_dim, c.d_model),
                 "output": (c.d_model, c.n_q_heads*c.head_dim),
                 "index_query": (c.index_heads*c.index_dim, c.d_model),
                 "index_key": (c.index_dim, c.d_model)}
        for name, weight in (("query_gate", query_gate), ("key", key), ("value", value),
                             ("output", output), ("index_query", index_query), ("index_key", index_key)):
            if not callable(weight):
                self._array(name, weight, sizes[name])
                weight = weight.copy()
            self.projections[name] = (weight, sizes[name][0])
        self.norms = []
        for gamma, dim in ((gamma_q, c.head_dim), (gamma_k, c.head_dim),
                           (gamma_index_q, c.index_dim), (gamma_index_k, c.index_dim)):
            self._array("gamma", gamma, (dim,))
            norm = RMSNormInference(dim, eps=float(c.epsilon))
            norm.set_weights(gamma.copy())
            self.norms.append(norm)
        self.rope = RotaryEmbeddingInference(c.rotary_dim, base=c.rope_base, layout="half")

    @staticmethod
    def _array(name, array, shape):
        _fp32(name, array)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not bool(xp.all(xp.isfinite(array))):
            raise ValueError(f"{name} must be finite")
        return array

    def _project(self, name, hidden):
        weight, size = self.projections[name]
        result = weight(hidden) if callable(weight) else weight @ hidden
        return self._array(name, result, (size,))

    def _rotate(self, array, position):
        return self.rope.forward(array[None, None], start_pos=position)[0, 0]

    def create_state(self, max_length):
        c = self.config
        capacity = _integer(max_length, "max_length")
        shape = (capacity, c.n_kv_heads, c.head_dim)
        return QSAState(self._owner, capacity, xp.zeros(shape, dtype=xp.float32),
                        xp.zeros(shape, dtype=xp.float32),
                        xp.zeros((capacity, c.index_dim), dtype=xp.float32),
                        xp.zeros((capacity // c.block_size + 1, c.index_dim), dtype=xp.float32))

    def _state(self, state):
        if not isinstance(state, QSAState) or state.owner is not self._owner:
            raise ValueError("state owner mismatch")
        c = self.config
        cap = _integer(state.max_length, "max_length")
        length = _integer(state.length, "state length", 0)
        if length > cap:
            raise ValueError("invalid state length")
        for array, shape in ((state.keys, (cap, c.n_kv_heads, c.head_dim)),
                             (state.values, (cap, c.n_kv_heads, c.head_dim)),
                             (state.raw_index, (cap, c.index_dim)),
                             (state.pooled, (cap // c.block_size + 1, c.index_dim))):
            _fp32("cache", array)
            if array.shape != shape:
                raise ValueError("invalid cache shape")

    def reset(self, state):
        self._state(state)
        for array in (state.keys, state.values, state.raw_index, state.pooled):
            array.fill(0)
        state.length = 0

    def step(self, hidden, state, *, position, diagnostics=False):
        """Return output, or (output, diagnostic dict) with copied scores/IDs.

        No arbitrary/multimodal position vectors, offsets, holes, sliding window,
        batching, cache quantization or custom masks. Causality is cell <= t.
        """
        c = self.config
        self._state(state)
        t = _integer(position, "scalar text position", 0)
        if t != state.length:
            raise ValueError("position must equal state.length (zero-origin contiguous text)")
        if t >= state.max_length:
            raise ValueError("QSA cache capacity exceeded")
        self._array("hidden", hidden, (c.d_model,))
        qg = self._project("query_gate", hidden).reshape(c.n_q_heads, 2*c.head_dim)
        k = self._project("key", hidden).reshape(c.n_kv_heads, c.head_dim)
        v = self._project("value", hidden).reshape(c.n_kv_heads, c.head_dim)
        iq = self._project("index_query", hidden).reshape(c.index_heads, c.index_dim)
        raw = self._project("index_key", hidden)
        q = self._rotate(self.norms[0].forward(qg[:, :c.head_dim]), t)
        k = self._rotate(self.norms[1].forward(k), t)
        iq = self._rotate(self.norms[2].forward(iq), t)
        n = t + 1
        completed, tail = divmod(n, c.block_size)
        new_pool = None
        if not tail:
            start = n - c.block_size
            pooled = raw.copy() if c.block_size == 1 else state.raw_index[start].copy()
            # Preserve the source's left-to-right block summation.
            for j in range(start + 1, t):
                pooled += state.raw_index[j]
            if c.block_size != 1:
                pooled += raw
            pooled *= xp.float32(1 / c.block_size)
            new_pool = self._rotate(self.norms[3].forward(pooled[None]), start)[0]
        scale = xp.float32(c.attention_scale if c.attention_scale is not None else 1/math.sqrt(c.head_dim))
        # Sequential text admits exactly [0,t]. At this width selection keeps
        # every causal cell, independently of scores/ties. Pooling above must
        # still run so later sparse steps see the identical index history.
        if (self.fast_decode and BACKEND_NAME == "cupy" and not diagnostics
                and n <= c.index_topk + c.block_size - 1):
            gated = qsa_decode_gpu(q, qg, k, v, state.keys, state.values, n, scale)
        else:
            dead = self.norms[3].forward(raw if t == 0 else state.raw_index[0])
            block_scores = xp.empty((completed + bool(tail),), dtype=xp.float32)
            for b in range(block_scores.size):
                bk = (new_pool if new_pool is not None and b == completed - 1
                      else state.pooled[b]) if b < completed else dead
                dots = xp.maximum(iq @ bk, xp.float32(0))
                score = dots[0].copy()
                for h in range(1, c.index_heads):
                    score += dots[h]
                block_scores[b] = score
            scores = xp.repeat(block_scores, c.block_size)[:n]
            scores[completed*c.block_size:] += xp.float32(1e9)
            # Only [0,t] exists in sequential prefill: the original causal mask is
            # enforced by construction, independently of the selection mask.
            ids = select_cells(scores, min(n, c.index_topk + c.block_size - 1), c.tie_policy)
            contexts = xp.empty((c.n_q_heads, c.head_dim), dtype=xp.float32)
            old_ids = ids[ids < t]
            has_current = bool(xp.any(ids == t))
            for h in range(c.n_q_heads):
                kh = h // (c.n_q_heads // c.n_kv_heads)
                # Bounded selected scratch, never concatenate/grow the persistent KV.
                keys = xp.empty((ids.size, c.head_dim), dtype=xp.float32)
                vals = xp.empty_like(keys)
                keys[:old_ids.size] = state.keys[old_ids, kh]
                vals[:old_ids.size] = state.values[old_ids, kh]
                if has_current:
                    keys[-1], vals[-1] = k[kh], v[kh]
                logits = (keys @ q[h]) * scale
                probability = xp.exp(logits - xp.max(logits))
                probability /= xp.sum(probability, dtype=xp.float32)
                contexts[h] = probability @ vals
            gated = contexts * _sigmoid(qg[:, c.head_dim:])
        result = self._project("output", gated.reshape(-1))
        # All callbacks and output shapes validated before any state write.
        state.keys[t], state.values[t], state.raw_index[t] = k, v, raw
        if new_pool is not None:
            state.pooled[completed - 1] = new_pool
        state.length = n
        if diagnostics:
            return result, {"selected_ids": ids.copy(), "cell_scores": scores.copy(),
                            "block_scores": block_scores.copy(), "tail_start": completed*c.block_size}
        return result

    def forward(self, hidden, state, *, start_position):
        """Sequential prefill [T,D]; not transactional across callback failures."""
        self._state(state)
        _fp32("hidden", hidden)
        if hidden.ndim != 2 or hidden.shape[1] != self.config.d_model:
            raise ValueError("prefill hidden must be [T,D]")
        if _integer(start_position, "start_position", 0) != state.length:
            raise ValueError("start_position must equal state.length")
        if state.length + hidden.shape[0] > state.max_length:
            raise ValueError("QSA cache capacity exceeded")
        if not bool(xp.all(xp.isfinite(hidden))):
            raise ValueError("hidden must be finite")
        output = xp.empty_like(hidden)
        for i in range(hidden.shape[0]):
            output[i] = self.step(hidden[i], state, position=start_position+i)
        return output
