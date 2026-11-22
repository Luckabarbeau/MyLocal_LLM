"""Batched causal QSA for the all-cell prefix, with exact singleton attention.

Packed projections, norms, RoPE, pooling, attention and host commits operate on
whole chunks. Attention masks each query at its own position. Sparse histories
beyond the selection width explicitly retain the reference path.
"""
from functools import lru_cache
import math
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.rope import get_rope_cos_sin
from mini_llm.ops.qsa_decode_gpu import _SOURCE as SERIAL
from mini_llm.ops.qsa_warp_dots_gpu import _DOTS_SOURCE, finalizer_source
from mini_llm.external import qwen4exp_hoststate as hoststate


@lru_cache(maxsize=8)
def _kernels(device):
    dot = _DOTS_SOURCE.replace('    const int h = blockIdx.x, kh = h / groups;',
        '    int token = blockIdx.z; n += token+1;\n'
        '    q += (long long)token*gridDim.x*256; k += (long long)token*(gridDim.x/groups)*256;\n'
        '    logits += (long long)token*gridDim.x*logit_stride;\n'
        '    const int h = blockIdx.x, kh = h / groups;')
    final = finalizer_source(SERIAL, value_tile=32).replace(
        '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;',
        '    int token = blockIdx.z; n += token+1;\n'
        '    q += (long long)token*gridDim.x*dim; qg += (long long)token*gridDim.x*2*dim;\n'
        '    k += (long long)token*(gridDim.x/groups)*dim; v += (long long)token*(gridDim.x/groups)*dim;\n'
        '    out += (long long)token*gridDim.x*dim; logits += (long long)token*gridDim.x*logit_stride;\n'
        '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;')
    with xp.cuda.Device(device):
        return (xp.RawKernel(dot, 'qsa_warp_dots', options=('--std=c++14','--fmad=false')),
                xp.RawKernel(final, 'qsa_decode', options=('--std=c++14','--fmad=false')))


def attention_chunk(q, qg, k, v, keys, values, start, scale):
    """Validated contiguous [T,H,D] operands, historical cache plus chunk K/V."""
    rows, heads, dim = q.shape
    end = start+rows
    dots, final = _kernels(q.device.id)
    logits = xp.empty((rows, heads, end), xp.float32)
    output = xp.empty_like(q)
    strides = tuple(np.int64(s//a.itemsize) for a in
                    (q[0], qg[0], k[0], v[0], keys, values) for s in a.strides)
    qs0,qs1,_,_,k0,k1,_,_,ks0,ks1,ks2,_,_,_ = strides
    groups = np.int32(heads//k.shape[1])
    dots((heads,(end+3)//4,rows), (128,),
        (q,k,keys,logits,np.int32(start),groups,np.float32(scale),np.int32(end),
         qs0,qs1,k0,k1,ks0,ks1,ks2))
    final((heads,(dim+31)//32,rows), (128,),
        (q,qg,k,v,keys,values,output,np.int32(start),np.int32(dim),groups,
         np.float32(scale),logits,np.int32(end),*strides))
    return output


def _rotate(attention, value, positions, end):
    c = attention.config
    cos, sin = get_rope_cos_sin(1 << (end-1).bit_length(), c.rotary_dim, c.rope_base, xp.float32)
    half = c.rotary_dim//2
    cos, sin = cos[0,positions,0], sin[0,positions,0]
    if value.ndim == 3:
        cos, sin = cos[:,None,:], sin[:,None,:]
    out = value.copy()
    a, b = value[..., :half], value[...,half:2*half]
    out[..., :half] = a*cos-b*sin
    out[...,half:2*half] = a*sin+b*cos
    return out


def try_qsa_chunk(model, layer, attention, hidden, state):
    c = attention.config
    rows = hidden.shape[0]
    start, end = state.length, state.length+rows
    if (BACKEND_NAME != 'cupy' or state.qsa_state_storage != 'host'
            or not attention.fast_decode or c.head_dim != 256 or c.n_q_heads > 24
            or end > c.index_topk+c.block_size-1 or rows < 1
            or os.environ.get('MINI_LLM_QSA_WARP_DOTS','0') != '1'):
        return None
    attention._state(state)
    if end > state.max_length:
        raise ValueError('QSA cache capacity exceeded')
    from mini_llm.external.qwen4exp_prefill import _project
    from .finite_checks_gpu import all_finite_gpu
    stem = f'blk.{layer}.'
    names = dict(query_gate='attn_q',key='attn_k',value='attn_v',
                 index_key='indexer.k_proj',output='attn_output')
    def project(name, x):
        return _project(model, attention.projections[name][0], stem+names[name]+'.weight', x)
    qg = project('query_gate', hidden).reshape(rows,c.n_q_heads,2*c.head_dim)
    raw_k = project('key', hidden).reshape(rows,c.n_kv_heads,c.head_dim)
    v = project('value', hidden).reshape(rows,c.n_kv_heads,c.head_dim)
    raw = project('index_key', hidden)
    positions = xp.arange(start,end,dtype=xp.int64)
    q = _rotate(attention, attention.norms[0].forward(qg[...,:c.head_dim]), positions, end)
    k = _rotate(attention, attention.norms[1].forward(raw_k), positions, end)
    completed_start, completed_end = start//c.block_size, end//c.block_size
    pooled = None
    if completed_end > completed_start:
        first = completed_start*c.block_size
        old = xp.asarray(state.raw_index[first:start])
        pool_rows = xp.concatenate((old,raw),axis=0)[:(completed_end-completed_start)*c.block_size]
        pool_rows = pool_rows.reshape(-1,c.block_size,c.index_dim)
        pooled = pool_rows[:,0].copy()
        for j in range(1,c.block_size):
            pooled += pool_rows[:,j]
        pooled *= xp.float32(1/c.block_size)
        pooled = attention.norms[3].forward(pooled)
        pooled = _rotate(attention, pooled,
            xp.arange(completed_start,completed_end,dtype=xp.int64)*c.block_size,end)
    hoststate.stage_prefix(state,start)
    state.staged_keys[start:end] = k
    state.staged_values[start:end] = v
    scale = c.attention_scale if c.attention_scale is not None else 1/math.sqrt(c.head_dim)
    gated = attention_chunk(q,qg,k,v,state.staged_keys,state.staged_values,start,scale)
    result = project('output',gated.reshape(rows,-1))
    if not all_finite_gpu(hidden,qg,raw_k,v,raw,q,k,result) or (
            pooled is not None and not all_finite_gpu(pooled)):
        raise ValueError('QSA hidden and projections must be finite')
    # Three bulk transfers replace three per token. No canonical historical
    # state changes before the complete chunk's finite validation succeeds.
    kh,vh,rh = (asnumpy(a).copy() for a in (k,v,raw))
    if pooled is not None:
        state.pooled[completed_start:completed_end] = pooled
    xp.cuda.get_current_stream().synchronize()
    state.keys[start:end],state.values[start:end],state.raw_index[start:end] = kh,vh,rh
    state.length = end
    state.staged_prefix_length = end
    return result
