"""Model-independent projected GQA using native-head, caller-owned KV caches.

The numerical grouped attention routine is shared with the existing native
GQAAttentionInference path. Projection width, model width, Q/K normalization,
RoPE and output gating are deliberately outside this operator.
"""

import math
from numbers import Integral

from mini_llm.backend import dtype_name, is_bfloat16_dtype, xp
from mini_llm.inference_state import GenerationState
from mini_llm.ops.causal_conv_inference import _float_array, _positive_int


def grouped_attention_forward(query, key, value, *, scale, causal_mask=None,
                              probability_dtype=None, softmax_dtype="float32"):
    """[B,Tq,Hq,Dk] attends to native [B,Hkv,Tk,Dk/Dv] caches.

    Query heads share contiguous groups of K/V heads without repeating caches.
    Each query must have at least one visible key. The default FP32 softmax
    and optional probability dtype preserve the native inference math; FP64
    softmax can be requested explicitly for a high-precision reference path.
    """
    for name, array in (("query", query), ("key", key), ("value", value)):
        _float_array(array, name)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query and native key/value must be four-dimensional")
    batch, t_query, n_q_heads, key_dim = query.shape
    n_kv_heads, t_key = key.shape[1:3]
    value_dim = value.shape[-1]
    if (batch < 1 or n_q_heads < 1 or n_kv_heads < 1 or key_dim < 1 or value_dim < 1
            or n_q_heads % n_kv_heads or key.shape != (batch, n_kv_heads, t_key, key_dim)
            or value.shape != (batch, n_kv_heads, t_key, value_dim)):
        raise ValueError("Incompatible grouped-query/key/value shapes")
    if causal_mask is not None:
        if not isinstance(causal_mask, xp.ndarray) or causal_mask.shape != (t_query, t_key):
            raise ValueError("causal_mask must have shape [Tq,Tk]")
        if causal_mask.dtype != xp.dtype(bool):
            raise TypeError("causal_mask must have boolean dtype")
    if not t_query:
        return xp.empty((batch, 0, n_q_heads, value_dim), dtype=value.dtype)
    if not t_key:
        raise ValueError("Nonempty queries require at least one key")
    group_size = n_q_heads // n_kv_heads
    probability_dtype = query.dtype if probability_dtype is None else probability_dtype
    bf16_attention = is_bfloat16_dtype(query.dtype)
    if t_query == 1:
        q = query[:, 0].reshape(batch, n_kv_heads, group_size, key_dim)
        q_scores = q.astype(xp.float32, copy=False) if bf16_attention else q
        k_scores = key.astype(xp.float32, copy=False) if bf16_attention else key
        scores = xp.matmul(q_scores * scale, k_scores.swapaxes(-2, -1)).reshape(
            batch, n_q_heads, 1, t_key)
    else:
        q = query.reshape(batch, t_query, n_kv_heads, group_size, key_dim).transpose(0, 2, 3, 1, 4)
        q_scores = q.astype(xp.float32, copy=False) if bf16_attention else q
        k_scores = key.astype(xp.float32, copy=False) if bf16_attention else key
        scores = xp.matmul(q_scores * scale,
                           k_scores[:, :, xp.newaxis].swapaxes(-2, -1)).reshape(
                               batch, n_q_heads, t_query, t_key)
    if causal_mask is not None:
        scores = xp.where(causal_mask[None, None], scores, -xp.inf)
    scores_compute = scores.astype(softmax_dtype)
    scores_compute -= xp.max(scores_compute, axis=-1, keepdims=True)
    probs = xp.exp(scores_compute)
    probs /= xp.sum(probs, axis=-1, keepdims=True)
    v_compute = value.astype(xp.float32, copy=False) if bf16_attention else value
    if t_query == 1:
        p = probs.reshape(batch, n_kv_heads, group_size, t_key)
        p = p if bf16_attention else p.astype(probability_dtype, copy=False)
        result = xp.matmul(p, v_compute).reshape(batch, 1, n_q_heads, value_dim)
    else:
        p = probs.reshape(batch, n_kv_heads, group_size, t_query, t_key)
        p = p if bf16_attention else p.astype(probability_dtype, copy=False)
        result = xp.matmul(p, v_compute[:, :, xp.newaxis]).transpose(0, 3, 1, 2, 4).reshape(
            batch, t_query, n_q_heads, value_dim)
    return result.astype(query.dtype, copy=False) if bf16_attention else result


class ProjectedGQAAttentionInference:
    """Append projected K/V then compute exact chronological causal attention.

    Uses existing GenerationState as a per-operator native-head cache holder.
    Cache position is the host length counter, not any model's RoPE position.
    Unequal key and value dimensions are supported. No weights, projections,
    model names, RoPE convention, or gate layout are owned by this component.
    """

    def __init__(self, n_q_heads, n_kv_heads, key_dim, value_dim=None, *,
                 dtype="float32", scale=None):
        self.n_q_heads = _positive_int(n_q_heads, "n_q_heads")
        self.n_kv_heads = _positive_int(n_kv_heads, "n_kv_heads")
        self.key_dim = _positive_int(key_dim, "key_dim")
        self.value_dim = self.key_dim if value_dim is None else _positive_int(value_dim, "value_dim")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.dtype = xp.dtype(dtype)
        if dtype_name(self.dtype) not in {"float16", "bfloat16", "float32", "float64"}:
            raise TypeError("Cache dtype must be floating-point")
        self.scale = 1.0 / math.sqrt(self.key_dim) if scale is None else float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")

    def create_state(self, batch_size, max_length):
        batch = _positive_int(batch_size, "batch_size")
        capacity = _positive_int(max_length, "max_length")
        return GenerationState(
            k_cache=xp.zeros((batch, self.n_kv_heads, capacity, self.key_dim), dtype=self.dtype),
            v_cache=xp.zeros((batch, self.n_kv_heads, capacity, self.value_dim), dtype=self.dtype),
            length=0, max_length=capacity, batch_size=batch)

    def _validate_state(self, state):
        if not isinstance(state, GenerationState):
            raise TypeError("state must be GenerationState created for this operator")
        batch = _positive_int(state.batch_size, "state.batch_size")
        capacity = _positive_int(state.max_length, "state.max_length")
        if (isinstance(state.length, bool) or not isinstance(state.length, Integral)
                or not 0 <= state.length <= capacity):
            raise ValueError("state.length must be within the cache capacity")
        for name, array, width in (("k_cache", state.k_cache, self.key_dim),
                                   ("v_cache", state.v_cache, self.value_dim)):
            _float_array(array, name)
            if array.shape != (batch, self.n_kv_heads, capacity, width):
                raise ValueError(f"Invalid {name} geometry")
            if array.dtype != self.dtype:
                raise TypeError(f"{name} dtype does not match operator")
            if not getattr(array.flags, "writeable", True):
                raise ValueError(f"{name} must be writable")
        return batch, capacity

    def copy_state(self, state):
        """Copy this operator's KV state, not unrelated model session metadata."""
        self._validate_state(state)
        return GenerationState(state.k_cache.copy(), state.v_cache.copy(),
                               int(state.length), int(state.max_length), int(state.batch_size))

    def forward(self, query, key, value, state):
        batch, capacity = self._validate_state(state)
        for name, array in (("query", query), ("key", key), ("value", value)):
            _float_array(array, name)
            if array.dtype != self.dtype:
                raise TypeError(f"{name} dtype does not match cache/operator")
        if query.ndim != 4:
            raise ValueError("query must have shape [B,T,Hq,Dk]")
        time = query.shape[1]
        if (query.shape != (batch, time, self.n_q_heads, self.key_dim)
                or key.shape != (batch, time, self.n_kv_heads, self.key_dim)
                or value.shape != (batch, time, self.n_kv_heads, self.value_dim)):
            raise ValueError("Projected query/key/value geometry mismatch")
        start = int(state.length)
        end = start + time
        if end > capacity:
            raise ValueError("KV cache capacity exceeded")
        if not time:
            return xp.empty((batch, 0, self.n_q_heads, self.value_dim), dtype=self.dtype)
        state.k_cache[:, :, start:end] = key.transpose(0, 2, 1, 3)
        state.v_cache[:, :, start:end] = value.transpose(0, 2, 1, 3)
        mask = None if time == 1 else xp.arange(end)[None, :] <= xp.arange(start, end)[:, None]
        softmax_dtype = xp.float64 if dtype_name(self.dtype) == "float64" else xp.float32
        result = grouped_attention_forward(query, state.k_cache[:, :, :end],
            state.v_cache[:, :, :end], scale=self.scale, causal_mask=mask,
            probability_dtype=self.dtype, softmax_dtype=softmax_dtype)
        state.length = end
        return result
