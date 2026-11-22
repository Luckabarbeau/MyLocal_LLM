"""Exact FP32 host history helpers; no history truncation or quantization.

Transfers fence the current stream before releasing pageable host sources or
publishing a commit. A state is caller-owned and must not be used concurrently.
"""
import numpy as np

from ..backend import BACKEND_NAME, asnumpy, xp


def state_storage(value):
    if value not in ('device', 'host'):
        raise ValueError("qsa_state_storage must be 'device' or 'host'")
    return value


def upload(value):
    result = xp.asarray(value, dtype=xp.float32)
    if BACKEND_NAME == 'cupy':
        xp.cuda.get_current_stream().synchronize()
    return result


def _copy_rows(source, target):
    if BACKEND_NAME == 'cupy':
        target.set(source, stream=xp.cuda.get_current_stream())
        xp.cuda.get_current_stream().synchronize()
    else:
        target[...] = source


def stage_prefix(state, t):
    """Backfill only missing committed rows; normal decode needs no upload."""
    start = state.staged_prefix_length
    if start < t:
        for source, target in ((state.keys, state.staged_keys),
                               (state.values, state.staged_values)):
            _copy_rows(source[start:t], target[start:t])
        state.staged_prefix_length = t


def gather(state, ids):
    """Stage only selected old tokens, in ascending order, for all KV heads."""
    host_ids = asnumpy(ids)
    count = host_ids.size
    # Selected rows overwrite the contiguous-prefix cache, even for diagnostics.
    state.staged_prefix_length = 0
    for source, target in ((state.keys, state.staged_keys),
                           (state.values, state.staged_values)):
        rows = source[host_ids]
        _copy_rows(rows, target[:count])
    return state.staged_keys[:count], state.staged_values[:count]


def block_scores(iq, pooled, completed, tail, dead, new_pool):
    """Batched index dots, then the reference's left-to-right head reduction.

    NumPy retains the original matvec arithmetic. CuPy uses a single batched
    matmul (not a Python launch loop over up to 65536 pooled blocks).
    """
    count = completed + bool(tail)
    blocks = xp.empty((count, iq.shape[1]), dtype=xp.float32)
    blocks[:completed] = pooled[:completed]
    if new_pool is not None:
        blocks[completed - 1] = new_pool
    if tail:
        blocks[-1] = dead
    if BACKEND_NAME == 'cupy':
        dots = xp.maximum(xp.matmul(iq[None], blocks[:, :, None])[:, :, 0], xp.float32(0))
        scores = dots[:, 0].copy()
        for h in range(1, iq.shape[0]):
            scores += dots[:, h]
        return scores
    scores = xp.empty(count, dtype=xp.float32)
    for b in range(count):
        dots = xp.maximum(iq @ blocks[b], xp.float32(0))
        score = dots[0].copy()
        for h in range(1, iq.shape[0]):
            score += dots[h]
        scores[b] = score
    return scores


def commit(state, k, v, raw, new_pool, t, completed, *, cache_prefix=False):
    # Finish all reads/copies before the first host history write.
    kh, vh, rh = (asnumpy(a).copy() for a in (k, v, raw))
    if new_pool is not None:
        state.pooled[completed - 1] = new_pool
    if cache_prefix:
        state.staged_keys[t], state.staged_values[t] = k, v
    if BACKEND_NAME == 'cupy':
        xp.cuda.get_current_stream().synchronize()
    state.keys[t], state.values[t], state.raw_index[t] = kh, vh, rh
    if cache_prefix:
        state.staged_prefix_length = t + 1
