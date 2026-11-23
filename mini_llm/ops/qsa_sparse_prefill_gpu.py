"""Bounded sparse-QSA prefill with canonical per-query causal selection.

Projections run per chunk; selected KV uploads and FP32 attention run in small
query batches. Historical storage stays on the host. No history is truncated,
no future pooled block is visible, and the reference tie selector is retained.
"""

import math
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external import qwen4exp_hoststate as hoststate
from mini_llm.external.qwen4exp_attention import select_cells
from mini_llm.external.qwen4exp_hyperconnection import _sigmoid
from .qsa_prefill_gpu import _rotate


def _reference_attention(q, qg, keys, values, scale, out):
    rows, heads, dim = q.shape
    groups = heads // keys.shape[1]
    for t in range(rows):
        for h in range(heads):
            kh = h // groups
            logits = (keys[t, kh] @ q[t, h]) * xp.float32(scale)
            probability = xp.exp(logits - xp.max(logits))
            probability /= xp.sum(probability, dtype=xp.float32)
            out[t, h] = probability @ values[t, kh]
    out *= _sigmoid(qg[:, :, dim:])


def selected_attention(q, qg, keys, values, scale):
    """Reference FP32 attention over a bounded batch of selected history.

    Batched cuBLAS GEMM is intentionally not used: its reduction differs from
    the reference GEMV and can cross downstream activation-quantizer boundaries.
    CuPy's cuBLAS wrapper also disallows capture, so no graph gain is claimed.
    """
    out = xp.empty_like(q)
    _reference_attention(q, qg, keys, values, scale, out)
    return out


def try_sparse_qsa_chunk(model, layer, attention, hidden, state):
    c = attention.config
    rows = hidden.shape[0]
    start, end = state.length, state.length + rows
    width = c.index_topk + c.block_size - 1
    if (BACKEND_NAME != 'cupy' or state.qsa_state_storage != 'host'
            or not attention.fast_decode or start < width or rows < 1
            or os.environ.get('MINI_LLM_QSA_SPARSE_PREFILL', '0') != '1'):
        return None
    attention._state(state)
    if end > state.max_length:
        raise ValueError('QSA cache capacity exceeded')
    # Bound selected K/V plus worst-case union staging to roughly 64 MiB.
    per_query_bytes = 4 * width * c.n_kv_heads * c.head_dim * 4
    batch_rows = max(1, min(16, (64 << 20) // per_query_bytes))
    from mini_llm.external.qwen4exp_prefill import _project
    from .finite_checks_gpu import all_finite_gpu
    names = dict(query_gate='attn_q', key='attn_k', value='attn_v',
                 index_key='indexer.k_proj', index_query='indexer.q_proj',
                 output='attn_output')
    def project(name, value):
        return _project(model, attention.projections[name][0],
                        f'blk.{layer}.{names[name]}.weight', value)
    qg = project('query_gate', hidden).reshape(rows, c.n_q_heads, 2*c.head_dim)
    raw_k = project('key', hidden).reshape(rows, c.n_kv_heads, c.head_dim)
    v = project('value', hidden).reshape(rows, c.n_kv_heads, c.head_dim)
    raw = project('index_key', hidden)
    # Dense index-query GEMM changes FP32 reduction order relative to vector
    # projection. Tiny score changes can alter exact top-k membership and later
    # MoE routes; retain that vector graph even when other projections batch.
    raw_iq = xp.stack([attention._project('index_query', row, check_finite=False)
                       for row in hidden]).reshape(rows, c.index_heads, c.index_dim)
    positions = xp.arange(start, end, dtype=xp.int64)
    q = _rotate(attention, attention.norms[0].forward(qg[..., :c.head_dim]), positions, end)
    k = _rotate(attention, attention.norms[1].forward(raw_k), positions, end)
    iq = _rotate(attention, attention.norms[2].forward(raw_iq), positions, end)
    first_block, last_block = start // c.block_size, end // c.block_size
    pooled = None
    blocks = state.pooled[:last_block].copy()
    if last_block > first_block:
        old = hoststate.upload(state.raw_index[first_block*c.block_size:start])
        pool_rows = xp.concatenate((old, raw), axis=0)
        pool_rows = pool_rows[:(last_block-first_block)*c.block_size].reshape(
            -1, c.block_size, c.index_dim)
        pooled = pool_rows[:, 0].copy()
        for j in range(1, c.block_size):
            pooled += pool_rows[:, j]
        pooled *= xp.float32(1/c.block_size)
        pooled = _rotate(attention, attention.norms[3].forward(pooled),
                         xp.arange(first_block, last_block, dtype=xp.int64)*c.block_size, end)
        blocks[first_block:] = pooled
    dead = (state.dead_index if state.dead_index is not None else
            attention.norms[3].forward(hoststate.upload(state.raw_index[0])))
    scale = c.attention_scale if c.attention_scale is not None else 1/math.sqrt(c.head_dim)
    output = xp.empty((rows, c.n_q_heads, c.head_dim), xp.float32)
    for begin in range(0, rows, batch_rows):
        stop = min(rows, begin + batch_rows)
        selections = []
        for i in range(begin, stop):
            n = start + i + 1
            completed, tail = divmod(n, c.block_size)
            scores = hoststate.block_scores(iq[i], blocks, completed, tail, dead, None)
            scores = xp.repeat(scores, c.block_size)[:n]
            scores[completed*c.block_size:] += xp.float32(1e9)
            selections.append(asnumpy(select_cells(scores, width, c.tie_policy)))
        ids = np.stack(selections)
        # Upload each historical cell only once per batch, even if all queries
        # select it. Current-chunk cells are read directly from projected K/V.
        old_ids = np.unique(ids[ids < start])
        old_count = len(old_ids)
        old_k = hoststate.upload(state.keys[old_ids])
        old_v = hoststate.upload(state.values[old_ids])
        source_k = xp.concatenate((old_k, k[:stop]), axis=0)
        source_v = xp.concatenate((old_v, v[:stop]), axis=0)
        mapped = np.where(ids < start, np.searchsorted(old_ids, ids), old_count + ids-start)
        device_ids = xp.asarray(mapped)
        selected_k = xp.ascontiguousarray(source_k[device_ids].transpose(0, 2, 1, 3))
        selected_v = xp.ascontiguousarray(source_v[device_ids].transpose(0, 2, 1, 3))
        output[begin:stop] = selected_attention(q[begin:stop], qg[begin:stop],
                                              selected_k, selected_v, scale)
    result = project('output', output.reshape(rows, -1))
    if (not all_finite_gpu(hidden, qg, raw_k, v, raw, raw_iq, q, k)
            or not all_finite_gpu(iq, result)
            or (pooled is not None and not all_finite_gpu(pooled))):
        raise ValueError('QSA hidden and projections must be finite')
    kh, vh, rh = (asnumpy(a).copy() for a in (k, v, raw))
    if pooled is not None:
        state.pooled[first_block:last_block] = pooled
    xp.cuda.get_current_stream().synchronize()
    state.keys[start:end], state.values[start:end], state.raw_index[start:end] = kh, vh, rh
    state.dead_index = dead
    state.staged_prefix_length = 0
    state.length = end
    return result
