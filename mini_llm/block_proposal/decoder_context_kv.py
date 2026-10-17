"""0080BK: existing trained decoder attention reads causal trunk context as KV only.

No new parameters. Context positions are never queries and never run through
small-decoder FFNs. Both decoder layers use their own existing Wq/Wk/Wv/Wo.
All large projections are rank-2 GEMMs (CuPy BF16 compatible). Batched
attention scores are accumulated in FP32 for BF16 compatibility.
"""
from dataclasses import dataclass

from mini_llm.backend import xp, is_bfloat16_dtype, is_low_precision_dtype
from mini_llm.ops.rope import get_rope_cos_sin, rope_backward


@dataclass(frozen=True)
class DecoderContextKVCache:
    local: object
    memory: object
    q_rope: dict
    k_rope: dict
    q: object
    k: object
    v: object
    probs: object
    merged: object
    memory_len: int
    attention: object


def _rotate(x, cos, sin):
    y = xp.empty_like(x)
    a, b = x[..., 0::2], x[..., 1::2]
    y[..., 0::2] = a * cos - b * sin
    y[..., 1::2] = a * sin + b * cos
    return y


def decoder_context_forward(attn, local, memory, valid, allow):
    """Joint softmax: 8 local queries over [M read-only contextual, 8 local] KV.

    local [G,2K,D], memory [G,M,D], valid [G,M], allow [2K,2K].
    The caller ensures every memory index <= this K-block's causal boundary.
    """
    if local.ndim != 3 or memory.ndim != 3 or valid.ndim != 2:
        raise ValueError('0080BK expected local[G,T,D], memory[G,M,D], valid[G,M]')
    g, t, d = local.shape
    m = int(memory.shape[1]); h = attn.n_q_heads; dh = attn.d_head
    if g == 0 or t == 0 or m == 0 or memory.shape != (g,m,d) or d != attn.d_model or valid.shape != (g,m):
        raise ValueError('0080BK incompatible context and decoder shapes')
    if tuple(allow.shape) != (t,t):
        raise ValueError('0080BK requires existing local prefix-LM allow mask')
    if attn.n_q_heads != attn.n_kv_heads:
        raise NotImplementedError('0080BK initially requires full multihead decoder (Q heads = KV heads)')
    # Zero padded copies before projection, avoiding false context values/NaNs.
    memory = xp.where(valid[:,:,None], memory, 0)
    a = local.reshape(-1,d)
    b = memory.reshape(-1,d)
    q0 = (a @ attn.Wq.data).reshape(g,t,h,dh)
    k0_local = (a @ attn.Wk.data).reshape(g,t,h,dh)
    v_local = (a @ attn.Wv.data).reshape(g,t,h,dh)
    k0_mem = (b @ attn.Wk.data).reshape(g,m,h,dh)
    v_mem = (b @ attn.Wv.data).reshape(g,m,h,dh)
    k0 = xp.concatenate((k0_mem,k0_local),axis=1)
    v = xp.concatenate((v_mem,v_local),axis=1)
    # RoPE local offset M, memory 0..M-1. Unlike explicit GPU position IDs,
    # this uses a cached table without device-to-host synchronization.
    cos,sin = get_rope_cos_sin(m+t,dh,attn.rope_base,q0.dtype)
    q_cos,q_sin=cos[:,m:,:,:],sin[:,m:,:,:]
    q=_rotate(q0,q_cos,q_sin)
    k=_rotate(k0,cos,sin)
    qcache={'cos':q_cos,'sin':q_sin}
    kcache={'cos':cos,'sin':sin}
    # FP32 scores/softmax avoid CuPy BF16 N-D GEMM dtype 'E'.
    qh=xp.transpose(q.astype(xp.float32,copy=False),(0,2,1,3))
    kh=xp.transpose(k.astype(xp.float32,copy=False),(0,2,1,3))
    vh=xp.transpose(v.astype(xp.float32,copy=False),(0,2,1,3))
    scores=xp.matmul(qh,kh.swapaxes(-1,-2))*attn.scale
    allowed=xp.concatenate((xp.broadcast_to(valid[:,None,:],(g,t,m)),
                            xp.broadcast_to(allow[None,:,:],(g,t,t))),axis=2)
    scores=xp.where(allowed[:,None,:,:],scores,xp.asarray(-1.e30,dtype=xp.float32))
    probs=xp.exp(scores-xp.max(scores,axis=-1,keepdims=True))
    probs=xp.where(allowed[:,None,:,:],probs,0.)
    probs=probs / xp.sum(probs,axis=-1,keepdims=True)
    out=xp.matmul(probs,vh)
    merged=xp.transpose(out,(0,2,1,3)).reshape(g,t,d)
    mc=merged.astype(local.dtype,copy=False) if is_bfloat16_dtype(local.dtype) else merged
    result=(mc.reshape(-1,d) @ attn.Wo.data).reshape(g,t,d)
    return result,DecoderContextKVCache(local,memory,qcache,kcache,q,k,v,probs,merged,m,attn)


def decoder_context_backward(d_output,cache):
    """Manual exact VJP; updates existing decoder attention parameters.

    Returns d_local and d_memory. Memory receives only K/V gradients.
    """
    attn=cache.attention
    local, memory=cache.local,cache.memory
    g,t,d=local.shape
    m=cache.memory_len; h=attn.n_q_heads; dh=attn.d_head
    y2=d_output.reshape(-1,d)
    merged2=cache.merged.astype(y2.dtype,copy=False).reshape(-1,d)
    # Low-precision gradients always use 2D GEMMs, just like GQAAttention.
    attn.Wo.grad += merged2.T @ y2
    dmerged=(y2 @ attn.Wo.data.T).reshape(g,t,d).astype(xp.float32)
    dc=xp.transpose(dmerged.reshape(g,t,h,dh),(0,2,1,3))
    qh=xp.transpose(cache.q.astype(xp.float32),(0,2,1,3))
    kh=xp.transpose(cache.k.astype(xp.float32),(0,2,1,3))
    vh=xp.transpose(cache.v.astype(xp.float32),(0,2,1,3))
    dp=xp.matmul(dc,vh.swapaxes(-1,-2))
    dv=xp.matmul(cache.probs.swapaxes(-1,-2),dc)
    ds=cache.probs*(dp-xp.sum(dp*cache.probs,axis=-1,keepdims=True))
    dq=xp.matmul(ds,kh)*attn.scale
    dk=xp.matmul(ds.swapaxes(-1,-2),qh)*attn.scale
    dq=xp.transpose(dq,(0,2,1,3))
    dk=xp.transpose(dk,(0,2,1,3))
    dv=xp.transpose(dv,(0,2,1,3))
    dq=rope_backward(dq,cache.q_rope).reshape(-1,d)
    dk=rope_backward(dk,cache.k_rope)
    dk_mem=dk[:,:m].reshape(-1,d)
    dk_loc=dk[:,m:].reshape(-1,d)
    dv_mem=dv[:,:m].reshape(-1,d)
    dv_loc=dv[:,m:].reshape(-1,d)
    if is_low_precision_dtype(local.dtype):
        dq=dq.astype(local.dtype,copy=False)
        dk_mem=dk_mem.astype(local.dtype,copy=False)
        dk_loc=dk_loc.astype(local.dtype,copy=False)
        dv_mem=dv_mem.astype(local.dtype,copy=False)
        dv_loc=dv_loc.astype(local.dtype,copy=False)
    a=local.reshape(-1,d); b=memory.reshape(-1,d)
    attn.Wq.grad += a.T @ dq
    attn.Wk.grad += a.T @ dk_loc + b.T @ dk_mem
    attn.Wv.grad += a.T @ dv_loc + b.T @ dv_mem
    dx=(dq @ attn.Wq.data.T + dk_loc @ attn.Wk.data.T +
        dv_loc @ attn.Wv.data.T).reshape(g,t,d)
    dm=(dk_mem @ attn.Wk.data.T + dv_mem @ attn.Wv.data.T).reshape(g,m,d)
    return dx,dm
