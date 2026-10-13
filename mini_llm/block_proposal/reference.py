"""0080A tiny, exact, single-head CPU/GPU attention reference.

Not a production decoder. It tests full PLAN bidirectionality and the
strict DEC causal cache/teacher-forcing equality before implementing 0080C.
"""

from mini_llm.backend import xp
from .contracts import prefix_lm_allow_mask


def _softmax(x):
    z = x - xp.max(x, axis=-1, keepdims=True)
    a = xp.exp(z)
    return a / xp.sum(a, axis=-1, keepdims=True)


def reference_attention(inputs, q_proj, k_proj, v_proj, *, block_size):
    """Single-head masked attention on 2K rows of one independent block."""
    k = int(block_size)
    if inputs.ndim != 2 or inputs.shape[0] != 2*k:
        raise ValueError('expected one block of 2K feature rows')
    q, keys, values = inputs @ q_proj, inputs @ k_proj, inputs @ v_proj
    d = int(q.shape[-1])
    if d <= 0 or keys.shape[-1] != d:
        raise ValueError('query/key dimension mismatch')
    logits = (q @ keys.T) / (d ** 0.5)
    logits = xp.where(prefix_lm_allow_mask(k), logits, -xp.inf)
    return _softmax(logits) @ values


def reference_incremental_decoding(plan_inputs, dec_inputs, q_proj, k_proj, v_proj):
    """One PLAN prefill; then append DEC positions one by one to K/V cache."""
    k = int(plan_inputs.shape[0])
    if plan_inputs.ndim != 2 or dec_inputs.shape != plan_inputs.shape:
        raise ValueError('plan and decoder input rows must be [K,d]')
    q = plan_inputs @ q_proj
    cached_keys, cached_values = plan_inputs @ k_proj, plan_inputs @ v_proj
    scale = q.shape[-1] ** 0.5
    plan_state = _softmax((q @ cached_keys.T) / scale) @ cached_values
    decisions = []
    for j in range(k):
        x = dec_inputs[j:j+1]
        cached_keys = xp.concatenate((cached_keys, x @ k_proj), axis=0)
        cached_values = xp.concatenate((cached_values, x @ v_proj), axis=0)
        decision = _softmax(((x @ q_proj) @ cached_keys.T) / scale) @ cached_values
        decisions.append(decision)
    return plan_state, xp.concatenate(decisions, axis=0)
