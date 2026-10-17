"""0080BJ: compact, parameter-free contextual cross-attention for K-block decoding.

Exactly 2*K query positions read up to M causal, main-trunk hidden states.
The M source states are keys and values ONLY: no feed-forward decoder pass on
history, no 64K activation expansion, and no vocabulary materialization.

This is a zero-new-parameter diagnostic variant designed to compare against an
existing direct-latent checkpoint. A learned Q/K/V/O version is a separate
experiment. BF16 matmuls are deliberately accumulated in FP32; CuPy's general
N-D BF16 matmul path has failed on this codebase ('dtype E').
"""
import math
from dataclasses import dataclass
from mini_llm.backend import xp, BACKEND_NAME


@dataclass(frozen=True)
class ContextAttentionCache:
    query: object
    memory: object
    probs: object
    valid: object
    scale: float
    gain: float


def context_attention_forward(query, memory, valid, *, gain=0.1):
    """Return query+gain*Attn(query, memory, memory), and manual-VJP cache.

    query [G,2K,D], memory [G,M,D], valid [G,M]. At least one memory
    location must be valid in every row. No future input is permitted: the
    gatherer is responsible for source positions at/before each block boundary.
    """
    if query.ndim != 3 or memory.ndim != 3 or valid.ndim != 2:
        raise ValueError('context attention requires query[G,Q,D], memory[G,M,D], valid[G,M]')
    g, q, d = map(int, query.shape)
    if memory.shape[0] != g or memory.shape[2] != d or valid.shape != (g, memory.shape[1]):
        raise ValueError('incompatible context attention dimensions')
    if g == 0 or q == 0 or memory.shape[1] == 0:
        raise ValueError('context attention expects nonempty dimensions')
    if not math.isfinite(float(gain)) or gain < 0:
        raise ValueError('context attention gain must be finite and nonnegative')
    # Fixed-width input dtypes may be BF16; FP32 avoids CuPy's N-D BF16 path.
    Q = query.astype(xp.float32, copy=False)
    M = memory.astype(xp.float32, copy=False)
    mask = xp.asarray(valid, dtype=xp.bool_)
    # Avoid synchronizing the GPU for each decoder group. The causal gatherer
    # always includes the boundary state; CPU unit tests validate arbitrary masks.
    if BACKEND_NAME == "numpy" and not bool(xp.all(xp.any(mask,axis=1))):
        raise ValueError('each block requires at least one past contextual state')
    scale = 1.0 / math.sqrt(d)
    score = xp.matmul(Q, xp.swapaxes(M,1,2)) * scale
    score = xp.where(mask[:,None,:], score, xp.asarray(-1e30,dtype=xp.float32))
    score = score - xp.max(score,axis=-1,keepdims=True)
    prob = xp.exp(score)
    prob = xp.where(mask[:,None,:], prob, 0.0)
    prob /= xp.sum(prob,axis=-1,keepdims=True)
    out = Q + float(gain) * xp.matmul(prob,M)
    return out, ContextAttentionCache(Q,M,prob,mask,scale,float(gain))


def context_attention_backward(d_output, cache):
    """Return exact analytical (d_query,d_memory), both FP32."""
    dy = d_output.astype(xp.float32,copy=False)
    p, q, m = cache.probs, cache.query, cache.memory
    # Residual identity plus attention's V, softmax, K and Q derivatives.
    dout = dy * cache.gain
    d_p = xp.matmul(dout,xp.swapaxes(m,1,2))
    d_memory = xp.matmul(xp.swapaxes(p,1,2),dout)
    d_scores = p * (d_p - xp.sum(p*d_p,axis=-1,keepdims=True))
    d_scores = xp.where(cache.valid[:,None,:],d_scores,0.0)
    d_query = dy + xp.matmul(d_scores,m) * cache.scale
    d_memory += xp.matmul(xp.swapaxes(d_scores,1,2),q) * cache.scale
    return d_query,d_memory


def gather_causal_working_context(hidden, rows, boundaries, prefix, length, *, offsets=None):
    """Gather the last `length` working states ending at each K-block boundary.

    This indexing contract is the central anti-leakage invariant: source
    positions are NEVER greater than `boundaries`, even at the final block.
    Raw routed history is deliberately outside this compact local read.
    Returns (memory, valid, clipped_indices). Invalid earlier slots may point
    at the first working state, but are masked in forward and backward.
    """
    if length not in (32,64,128):
        raise ValueError('context length must be 32, 64 or 128')
    if hidden.ndim != 3 or rows.ndim != 1 or boundaries.ndim != 1 or rows.shape != boundaries.shape:
        raise ValueError('invalid causal context gather shapes')
    prefix=int(prefix)
    if prefix<0 or prefix >= hidden.shape[1]:
        raise ValueError('invalid working prefix length')
    if offsets is None:
        offsets=xp.arange(length-1,-1,-1,dtype=xp.int64)
    raw=boundaries[:,None]-offsets[None,:]
    valid=raw>=prefix
    indices=xp.maximum(raw,prefix)
    return hidden[rows[:,None],indices,:],valid,indices
