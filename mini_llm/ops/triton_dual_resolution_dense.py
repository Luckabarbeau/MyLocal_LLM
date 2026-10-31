"""Optional tiled Triton attention for 0102C global summaries and compression.

The same mathematically exact, online-softmax attention kernel covers:
* global: [B,T,16,64] queries against [B,M,4,64] summary K/V, with
  EOS-safe strictly completed-block masks;
* compressor: [number_of_blocks,128,H,64] bidirectional self-attention.

A separate query-tile backward and key-tile backward avoids floating-point
atomics (in particular, GQA K/V gradients include *all* shared query heads).
Neither attention probability matrices nor T x T masks are materialized.

Experimental GPU code. The CuPy path remains the reference/default.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _skip_empty_tiles_enabled():
    """Experimental exact short circuit for fully masked global tiles."""
    return os.environ.get("MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES", "0").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_global_enabled():
    """Bound summary/query loops for validated contiguous Model B documents."""
    return os.environ.get(
        "MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise RuntimeError("0102C Triton kernels require Triton") from exc

_LOG2E = 1.4426950408889634
_D = 64
_BM = 32
_BN = 64
_SPARSE_BM = 64
_SPARSE_BN = 32


class _Pointer:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _ptr(array):
    name = str(array.dtype)
    dtype = {
        "float32": tl.float32, "float16": tl.float16,
        "bfloat16": tl.bfloat16, "int32": tl.int32,
        "uint8": tl.uint8,
    }.get(name)
    if dtype is None:
        raise TypeError(f"unsupported Triton attention dtype: {name}")
    return _Pointer(array, dtype)


def _strides(array):
    return tuple(int(stride // array.dtype.itemsize) for stride in array.strides)


@triton.jit
def _visible(q_index, k_index, qdoc, qpos, kdoc, kend, kval,
             n_q, n_k, GLOBAL: tl.constexpr):
    valid = (q_index[:, None] < n_q) & (k_index[None, :] < n_k)
    if GLOBAL:
        valid = valid & (qdoc[:, None] == kdoc[None, :])
        valid = valid & (kend[None, :] < qpos[:, None])
        valid = valid & (kval[None, :] != 0)
    return valid


@triton.jit
def _forward(
    Q, K, V, QDOC, QPOS, KDOC, KEND, KVALID, O, LSE, AVAIL,
    qsb, qst: tl.constexpr, qsh: tl.constexpr,
    ksb, kst: tl.constexpr, ksh: tl.constexpr,
    vsb, vst: tl.constexpr, vsh: tl.constexpr,
    osb, ost: tl.constexpr, osh: tl.constexpr,
    lsb, lst: tl.constexpr, lsh: tl.constexpr,
    qdsb, qdst: tl.constexpr,
    qpsb, qpst: tl.constexpr,
    kdsb, kdst: tl.constexpr,
    kesb, kest: tl.constexpr,
    kvsb, kvst: tl.constexpr,
    asb, ast: tl.constexpr,
    NQ, NK, HQ: tl.constexpr,
    GROUP: tl.constexpr, SCALE_LOG2: tl.constexpr,
    GLOBAL: tl.constexpr, BF16: tl.constexpr, SKIP_EMPTY: tl.constexpr,
    BOUNDED: tl.constexpr, HISTORY_BLOCK: tl.constexpr,
    SUMMARIES_PER_BLOCK: tl.constexpr, Q_TILE_OFFSET: tl.constexpr,
    SPARSE_STRIDE: tl.constexpr, SPARSE_PHASE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    tile = tl.program_id(0) + Q_TILE_OFFSET
    bh = tl.program_id(1)
    batch = bh // HQ
    head = bh % HQ
    kvhead = head // GROUP
    m = tile * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, 64)
    q = tl.load(Q + batch*qsb + m[:,None]*qst + head*qsh + d[None,:],
                mask=m[:,None] < NQ, other=0.)
    if GLOBAL:
        qdoc = tl.load(QDOC + batch*qdsb + m*qdst, mask=m < NQ, other=-1)
        if SPARSE_STRIDE > 0:
            qpos = m
        else:
            qpos = tl.load(QPOS + batch*qpsb + m*qpst, mask=m < NQ, other=-1)
    else:
        qdoc = tl.zeros((BM,), tl.int32)
        qpos = tl.zeros((BM,), tl.int32)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM,64), tl.float32)
    ever_visible = tl.zeros((BM,), tl.int32)
    key_stop = NK
    if BOUNDED:
        max_query = tl.minimum((tile + 1) * BM, NQ) - 1
        if SPARSE_STRIDE > 0:
            key_stop = tl.minimum(NK, tl.maximum(0, (max_query - SPARSE_PHASE) // SPARSE_STRIDE + 1))
        else:
            key_stop = tl.minimum(
                NK, (max_query // HISTORY_BLOCK) * SUMMARIES_PER_BLOCK
            )
    for key_start in tl.range(0, key_stop, BN):
        kn = key_start + n
        if GLOBAL:
            kdoc = tl.load(KDOC + batch*kdsb + kn*kdst, mask=kn < NK, other=-2)
            if SPARSE_STRIDE > 0:
                kend = SPARSE_PHASE + kn * SPARSE_STRIDE - 1
                kval = tl.full((BN,), 1, tl.int32)
            else:
                kend = tl.load(KEND + batch*kesb + kn*kest, mask=kn < NK, other=-2)
                kval = tl.load(KVALID + batch*kvsb + kn*kvst, mask=kn < NK, other=0)
        else:
            kdoc = tl.zeros((BN,), tl.int32)
            kend = tl.zeros((BN,), tl.int32)
            kval = tl.full((BN,), 1, tl.int32)
        allowed = _visible(m, kn, qdoc, qpos, kdoc, kend, kval,
                           NQ, NK, GLOBAL)
        # A whole tile with no preceding completed summary contributes exactly
        # zero to the online softmax. Do not load K/V or launch dot products.
        if SKIP_EMPTY:
            has_pairs = tl.sum(allowed.to(tl.int32)) > 0
        else:
            has_pairs = True
        if has_pairs:
            if GLOBAL:
                ever_visible = ever_visible | (tl.sum(allowed.to(tl.int32), axis=1) > 0).to(tl.int32)
            k = tl.load(K + batch*ksb + kn[:,None]*kst + kvhead*ksh + d[None,:],
                        mask=kn[:,None] < NK, other=0.)
            v = tl.load(V + batch*vsb + kn[:,None]*vst + kvhead*vsh + d[None,:],
                        mask=kn[:,None] < NK, other=0.)
            if BF16:
                scores = tl.dot(q, tl.trans(k)) * SCALE_LOG2
            else:
                scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)),
                                input_precision='tf32x3') * SCALE_LOG2
            scores = tl.where(allowed, scores, -float('inf'))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            safe_maximum = tl.where(new_maximum == -float('inf'), 0., new_maximum)
            correction = tl.where(maximum == -float('inf'), 0.,
                                  tl.exp2(maximum - safe_maximum))
            p = tl.exp2(scores - safe_maximum[:,None])
            denominator = denominator * correction + tl.sum(p, axis=1)
            accumulator = accumulator * correction[:,None]
            if BF16:
                accumulator = tl.dot(p.to(tl.bfloat16), v, accumulator)
            else:
                accumulator = tl.dot(p, v.to(tl.float32), accumulator,
                                     input_precision='tf32x3')
            maximum = new_maximum
    nonempty = denominator > 0.
    result = accumulator / tl.where(nonempty[:,None], denominator[:,None], 1.)
    lse = tl.where(nonempty, maximum + tl.log2(tl.maximum(denominator,1e-30)),0.)
    tl.store(O + batch*osb + m[:,None]*ost + head*osh + d[None,:],
             result, mask=m[:,None] < NQ)
    tl.store(LSE + batch*lsb + m*lst + head*lsh, lse, mask=m < NQ)
    if GLOBAL and SPARSE_STRIDE == 0:
        tl.store(AVAIL + batch*asb + m*ast, ever_visible,
                 mask=(m < NQ) & (head == 0))


@triton.jit
def _dq(
    Q,K,V,QDOC,QPOS,KDOC,KEND,KVALID,O,DO,LSE,DQ,
    qsb:tl.constexpr,qst:tl.constexpr,qsh:tl.constexpr,
    ksb:tl.constexpr,kst:tl.constexpr,ksh:tl.constexpr,
    vsb:tl.constexpr,vst:tl.constexpr,vsh:tl.constexpr,
    osb:tl.constexpr,ost:tl.constexpr,osh:tl.constexpr,
    dosb:tl.constexpr,dost:tl.constexpr,dosh:tl.constexpr,
    lsb:tl.constexpr,lst:tl.constexpr,lsh:tl.constexpr,
    dqsb:tl.constexpr,dqst:tl.constexpr,dqsh:tl.constexpr,
    qdsb:tl.constexpr,qdst:tl.constexpr,qpsb:tl.constexpr,qpst:tl.constexpr,
    kdsb:tl.constexpr,kdst:tl.constexpr,kesb:tl.constexpr,kest:tl.constexpr,
    kvsb:tl.constexpr,kvst:tl.constexpr,
    NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,GROUP:tl.constexpr,
    SCALE:tl.constexpr,SCALE_LOG2:tl.constexpr,GLOBAL:tl.constexpr,
    BF16:tl.constexpr,SKIP_EMPTY:tl.constexpr,BOUNDED:tl.constexpr,
    HISTORY_BLOCK:tl.constexpr,SUMMARIES_PER_BLOCK:tl.constexpr,
    Q_TILE_OFFSET:tl.constexpr,SPARSE_STRIDE:tl.constexpr,SPARSE_PHASE:tl.constexpr,
    BM:tl.constexpr,BN:tl.constexpr,
):
    tile=tl.program_id(0)+Q_TILE_OFFSET
    bh=tl.program_id(1)
    batch=bh//HQ
    head=bh%HQ
    kvhead=head//GROUP
    m=tile*BM+tl.arange(0,BM)
    n=tl.arange(0,BN)
    d=tl.arange(0,64)
    q=tl.load(Q+batch*qsb+m[:,None]*qst+head*qsh+d[None,:],mask=m[:,None]<NQ,other=0.)
    o=tl.load(O+batch*osb+m[:,None]*ost+head*osh+d[None,:],mask=m[:,None]<NQ,other=0.)
    dout=tl.load(DO+batch*dosb+m[:,None]*dost+head*dosh+d[None,:],mask=m[:,None]<NQ,other=0.)
    lse=tl.load(LSE+batch*lsb+m*lst+head*lsh,mask=m<NQ,other=0.)
    delta=tl.sum(o.to(tl.float32)*dout.to(tl.float32),axis=1)
    if GLOBAL:
        qdoc=tl.load(QDOC+batch*qdsb+m*qdst,mask=m<NQ,other=-1)
        if SPARSE_STRIDE > 0:
            qpos=m
        else:
            qpos=tl.load(QPOS+batch*qpsb+m*qpst,mask=m<NQ,other=-1)
    else:
        qdoc=tl.zeros((BM,),tl.int32)
        qpos=tl.zeros((BM,),tl.int32)
    acc=tl.zeros((BM,64),tl.float32)
    key_stop=NK
    if BOUNDED:
        max_query=tl.minimum((tile+1)*BM,NQ)-1
        if SPARSE_STRIDE > 0:
            key_stop=tl.minimum(NK,tl.maximum(0,(max_query-SPARSE_PHASE)//SPARSE_STRIDE+1))
        else:
            key_stop=tl.minimum(
                NK,(max_query//HISTORY_BLOCK)*SUMMARIES_PER_BLOCK
            )
    for key_start in tl.range(0,key_stop,BN):
        kn=key_start+n
        if GLOBAL:
            kdoc=tl.load(KDOC+batch*kdsb+kn*kdst,mask=kn<NK,other=-2)
            if SPARSE_STRIDE > 0:
                kend=SPARSE_PHASE+kn*SPARSE_STRIDE-1
                kval=tl.full((BN,),1,tl.int32)
            else:
                kend=tl.load(KEND+batch*kesb+kn*kest,mask=kn<NK,other=-2)
                kval=tl.load(KVALID+batch*kvsb+kn*kvst,mask=kn<NK,other=0)
        else:
            kdoc=tl.zeros((BN,),tl.int32)
            kend=tl.zeros((BN,),tl.int32)
            kval=tl.full((BN,),1,tl.int32)
        valid=_visible(m,kn,qdoc,qpos,kdoc,kend,kval,NQ,NK,GLOBAL)
        if SKIP_EMPTY:
            has_pairs = tl.sum(valid.to(tl.int32)) > 0
        else:
            has_pairs = True
        if has_pairs:
            k=tl.load(K+batch*ksb+kn[:,None]*kst+kvhead*ksh+d[None,:],mask=kn[:,None]<NK,other=0.)
            v=tl.load(V+batch*vsb+kn[:,None]*vst+kvhead*vsh+d[None,:],mask=kn[:,None]<NK,other=0.)
            if BF16:
                s=tl.dot(q,tl.trans(k))*SCALE_LOG2
                dp=tl.dot(dout.to(tl.bfloat16),tl.trans(v))
            else:
                s=tl.dot(q.to(tl.float32),tl.trans(k.to(tl.float32)),input_precision='tf32x3')*SCALE_LOG2
                dp=tl.dot(dout.to(tl.float32),tl.trans(v.to(tl.float32)),input_precision='tf32x3')
            p=tl.where(valid,tl.exp2(s-lse[:,None]),0.)
            ds=p*(dp-delta[:,None])
            if BF16:
                acc=tl.dot(ds.to(tl.bfloat16),k,acc)
            else:
                acc=tl.dot(ds,k.to(tl.float32),acc,input_precision='tf32x3')
    tl.store(DQ+batch*dqsb+m[:,None]*dqst+head*dqsh+d[None,:],
             acc*SCALE,mask=m[:,None]<NQ)


@triton.jit
def _dkdv(
    Q,K,V,QDOC,QPOS,KDOC,KEND,KVALID,O,DO,LSE,DK,DV,
    qsb:tl.constexpr,qst:tl.constexpr,qsh:tl.constexpr,
    ksb:tl.constexpr,kst:tl.constexpr,ksh:tl.constexpr,
    vsb:tl.constexpr,vst:tl.constexpr,vsh:tl.constexpr,
    osb:tl.constexpr,ost:tl.constexpr,osh:tl.constexpr,
    dosb:tl.constexpr,dost:tl.constexpr,dosh:tl.constexpr,
    lsb:tl.constexpr,lst:tl.constexpr,lsh:tl.constexpr,
    dksb:tl.constexpr,dkst:tl.constexpr,dksh:tl.constexpr,
    dvsb:tl.constexpr,dvst:tl.constexpr,dvsh:tl.constexpr,
    qdsb:tl.constexpr,qdst:tl.constexpr,qpsb:tl.constexpr,qpst:tl.constexpr,
    kdsb:tl.constexpr,kdst:tl.constexpr,kesb:tl.constexpr,kest:tl.constexpr,
    kvsb:tl.constexpr,kvst:tl.constexpr,
    NQ:tl.constexpr,NK:tl.constexpr,HK:tl.constexpr,GROUP:tl.constexpr,
    SCALE:tl.constexpr,SCALE_LOG2:tl.constexpr,GLOBAL:tl.constexpr,
    BF16:tl.constexpr,SKIP_EMPTY:tl.constexpr,BOUNDED:tl.constexpr,
    HISTORY_BLOCK:tl.constexpr,SUMMARIES_PER_BLOCK:tl.constexpr,
    Q_TILE_OFFSET:tl.constexpr,SPARSE_STRIDE:tl.constexpr,SPARSE_PHASE:tl.constexpr,
    ADD_DKDV:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
):
    ktile=tl.program_id(0)
    bh=tl.program_id(1)
    batch=bh//HK
    kvhead=bh%HK
    kn=ktile*BN+tl.arange(0,BN)
    mbase=tl.arange(0,BM)
    d=tl.arange(0,64)
    k=tl.load(K+batch*ksb+kn[:,None]*kst+kvhead*ksh+d[None,:],mask=kn[:,None]<NK,other=0.)
    v=tl.load(V+batch*vsb+kn[:,None]*vst+kvhead*vsh+d[None,:],mask=kn[:,None]<NK,other=0.)
    if GLOBAL:
        kdoc=tl.load(KDOC+batch*kdsb+kn*kdst,mask=kn<NK,other=-2)
        if SPARSE_STRIDE > 0:
            kend=SPARSE_PHASE+kn*SPARSE_STRIDE-1
            kval=tl.full((BN,),1,tl.int32)
        else:
            kend=tl.load(KEND+batch*kesb+kn*kest,mask=kn<NK,other=-2)
            kval=tl.load(KVALID+batch*kvsb+kn*kvst,mask=kn<NK,other=0)
    else:
        kdoc=tl.zeros((BN,),tl.int32)
        kend=tl.zeros((BN,),tl.int32)
        kval=tl.full((BN,),1,tl.int32)
    dkacc=tl.zeros((BN,64),tl.float32)
    dvacc=tl.zeros((BN,64),tl.float32)
    query_begin=0
    if BOUNDED:
        first_summary=ktile*BN
        if SPARSE_STRIDE > 0:
            query_begin=((SPARSE_PHASE + first_summary*SPARSE_STRIDE)//BM)*BM
        else:
            query_begin=(first_summary//SUMMARIES_PER_BLOCK+1)*HISTORY_BLOCK
    for group in range(GROUP):
        head=kvhead*GROUP+group
        for query_start in tl.range(query_begin,NQ,BM):
            m=query_start+mbase
            if GLOBAL:
                qdoc=tl.load(QDOC+batch*qdsb+m*qdst,mask=m<NQ,other=-1)
                if SPARSE_STRIDE > 0:
                    qpos=m
                else:
                    qpos=tl.load(QPOS+batch*qpsb+m*qpst,mask=m<NQ,other=-1)
            else:
                qdoc=tl.zeros((BM,),tl.int32)
                qpos=tl.zeros((BM,),tl.int32)
            valid=_visible(m,kn,qdoc,qpos,kdoc,kend,kval,NQ,NK,GLOBAL)
            if SKIP_EMPTY:
                has_pairs = tl.sum(valid.to(tl.int32)) > 0
            else:
                has_pairs = True
            if has_pairs:
                q=tl.load(Q+batch*qsb+m[:,None]*qst+head*qsh+d[None,:],mask=m[:,None]<NQ,other=0.)
                o=tl.load(O+batch*osb+m[:,None]*ost+head*osh+d[None,:],mask=m[:,None]<NQ,other=0.)
                dout=tl.load(DO+batch*dosb+m[:,None]*dost+head*dosh+d[None,:],mask=m[:,None]<NQ,other=0.)
                lse=tl.load(LSE+batch*lsb+m*lst+head*lsh,mask=m<NQ,other=0.)
                delta=tl.sum(o.to(tl.float32)*dout.to(tl.float32),axis=1)
                if BF16:
                    st=tl.dot(k,tl.trans(q))*SCALE_LOG2
                    dp=tl.dot(v,tl.trans(dout.to(tl.bfloat16)))
                else:
                    st=tl.dot(k.to(tl.float32),tl.trans(q.to(tl.float32)),input_precision='tf32x3')*SCALE_LOG2
                    dp=tl.dot(v.to(tl.float32),tl.trans(dout.to(tl.float32)),input_precision='tf32x3')
                p=tl.where(tl.trans(valid),tl.exp2(st-lse[None,:]),0.)
                ds=p*(dp-delta[None,:])
                if BF16:
                    dvacc=tl.dot(p.to(tl.bfloat16),dout.to(tl.bfloat16),dvacc)
                    dkacc=tl.dot(ds.to(tl.bfloat16),q,dkacc)
                else:
                    dvacc=tl.dot(p,dout.to(tl.float32),dvacc,input_precision='tf32x3')
                    dkacc=tl.dot(ds,q.to(tl.float32),dkacc,input_precision='tf32x3')
    if ADD_DKDV:
        dkacc=dkacc*SCALE+tl.load(
            DK+batch*dksb+kn[:,None]*dkst+kvhead*dksh+d[None,:],
            mask=kn[:,None]<NK,other=0.)
        dvacc=dvacc+tl.load(
            DV+batch*dvsb+kn[:,None]*dvst+kvhead*dvsh+d[None,:],
            mask=kn[:,None]<NK,other=0.)
    else:
        dkacc=dkacc*SCALE
    tl.store(DK+batch*dksb+kn[:,None]*dkst+kvhead*dksh+d[None,:],dkacc,mask=kn[:,None]<NK)
    tl.store(DV+batch*dvsb+kn[:,None]*dvst+kvhead*dvsh+d[None,:],dvacc,mask=kn[:,None]<NK)


@dataclass(frozen=True)
class TiledAttentionCache:
    q: object
    k: object
    v: object
    output: object
    lse: object
    available: object
    metadata: tuple
    scale: float
    global_mask: bool
    skip_empty: bool
    bounded: bool
    history_block_size: int
    summaries_per_block: int
    sparse_stride: int = 0
    sparse_phase: int = 0


def _validate(q, k, v, scale):
    import cupy as cp
    if not all(isinstance(t, cp.ndarray) for t in (q,k,v)):
        raise TypeError('Triton attention needs CuPy tensors')
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:
        raise ValueError('Q,K,V must have [batch,sequence,heads,64] layout')
    if q.shape[0]!=k.shape[0] or q.shape[3]!=64 or k.shape[3]!=64:
        raise ValueError('Triton attention requires head dim 64 and matching batch')
    if q.shape[2]%k.shape[2]:
        raise ValueError('number of query heads must be divisible by KV heads')
    if any(t.strides[-1]!=t.dtype.itemsize for t in (q,k,v)):
        raise ValueError('last dimension must be contiguous')
    if not (0 < float(scale) < 10):
        raise ValueError('invalid attention scale')
    if any(str(t.dtype) not in ('float32','bfloat16') for t in (q,k,v)):
        raise ValueError('Triton attention needs FP32 or BF16 tensors')
    if not (str(q.dtype)==str(k.dtype)==str(v.dtype)):
        raise ValueError('Triton Q,K,V must have matching dtypes')


def _metadata(q, k, *, documents=None, positions=None, block_documents=None,
              block_ends=None, block_valid=None):
    import cupy as cp
    if documents is None:
        zq=cp.zeros((1,1),dtype=cp.int32)
        zk=cp.zeros((1,1),dtype=cp.int32)
        return (zq,zq,zk,zk,zk),False
    provided=(documents,positions,block_documents,block_ends,block_valid)
    if any(t is None for t in provided):
        raise ValueError('global attention needs all five compact metadata fields')
    if documents.shape!=q.shape[:2] or positions.shape!=q.shape[:2]:
        raise ValueError('global attention query metadata shape mismatch')
    if block_documents.shape!=k.shape[:2] or block_ends.shape!=k.shape[:2] or block_valid.shape!=k.shape[:2]:
        raise ValueError('global attention summary metadata shape mismatch')
    return tuple(cp.ascontiguousarray(t,dtype=cp.int32) for t in provided),True


def forward(q,k,v,*,scale,documents=None,positions=None,
            block_documents=None,block_ends=None,block_valid=None,
            bounded_causal=False,history_block_size=128,summaries_per_block=1,
            sparse_stride=0,sparse_phase=0,out=None,lse=None):
    """FP32 output; full bidirectional attention if metadata is omitted."""
    import cupy as cp
    _validate(q,k,v,scale)
    sparse_stride = int(sparse_stride)
    sparse_phase = int(sparse_phase)
    if sparse_stride:
        if documents is None or documents.shape != q.shape[:2]:
            raise ValueError("sparse stride requires query document IDs")
        expected_keys = max(0, (q.shape[1] - sparse_phase + sparse_stride - 1) // sparse_stride)
        if not 0 <= sparse_phase < sparse_stride or k.shape[1] != expected_keys:
            raise ValueError("sparse stride requires a valid phase and matching K/V view")
        documents = cp.asarray(documents, dtype=cp.int32)
        key_documents = documents[:, sparse_phase::sparse_stride]
        meta = (documents, documents, key_documents, key_documents, key_documents)
        global_mask = True
    else:
        meta,global_mask=_metadata(q,k,documents=documents,positions=positions,
            block_documents=block_documents,block_ends=block_ends,block_valid=block_valid)
    batch,nq,hq,_=q.shape
    nk=k.shape[1]
    if out is None:
        out=cp.zeros(q.shape,dtype=cp.float32)
    elif out.shape != q.shape or str(out.dtype) != 'float32':
        raise ValueError('provided attention output must be matching FP32 storage')
    if lse is None:
        lse=cp.zeros(q.shape[:3],dtype=cp.float32)
    elif lse.shape != q.shape[:3] or str(lse.dtype) != 'float32':
        raise ValueError('provided attention LSE must be matching FP32 storage')
    availability=(cp.zeros((1,1),dtype=cp.uint8) if sparse_stride else
                  cp.zeros(q.shape[:2],dtype=cp.uint8))
    skip_empty = global_mask and _skip_empty_tiles_enabled()
    bounded = bool(global_mask and bounded_causal and _bounded_global_enabled())
    if sparse_stride and not global_mask:
        raise ValueError("sparse stride requires causal metadata")
    history_block_size = int(history_block_size)
    summaries_per_block = int(summaries_per_block)
    if bounded and not sparse_stride and (history_block_size <= 0 or summaries_per_block <= 0
                    or history_block_size % _BM):
        raise ValueError('bounded global attention requires a positive BM-aligned block and summary count')
    cache=TiledAttentionCache(
        q,k,v,out,lse,availability.view(cp.bool_),meta,float(scale),global_mask,
        skip_empty,bounded,history_block_size,summaries_per_block,
        sparse_stride,sparse_phase,
    )
    if nk==0 or (bounded and not sparse_stride and nq<=history_block_size):
        return out,cache
    qstr,kstr,vstr,ostr,lstr=map(_strides,(q,k,v,out,lse))
    bm = _SPARSE_BM if sparse_stride else _BM
    bn = _SPARSE_BN if sparse_stride else _BN
    q_tile_offset = history_block_size // bm if bounded and not sparse_stride else 0
    q_tiles = triton.cdiv(max(0, nq - q_tile_offset * bm), bm)
    qds,qps,kds,kes,kvs=map(_strides,meta)
    astr=_strides(availability)
    _forward[(q_tiles,batch*hq)](
        *map(_ptr,(q,k,v,*meta,out,lse,availability)),
        *qstr[:3],*kstr[:3],*vstr[:3],*ostr[:3],*lstr[:3],
        *qds[:2],*qps[:2],*kds[:2],*kes[:2],*kvs[:2],*astr[:2],
        NQ=nq,NK=nk,HQ=hq,GROUP=hq//k.shape[2],
        SCALE_LOG2=float(scale)*_LOG2E,GLOBAL=global_mask,SKIP_EMPTY=skip_empty,
        BOUNDED=bounded,HISTORY_BLOCK=history_block_size,
        SUMMARIES_PER_BLOCK=summaries_per_block,Q_TILE_OFFSET=q_tile_offset,
        SPARSE_STRIDE=sparse_stride,SPARSE_PHASE=sparse_phase,
        BF16=str(q.dtype)=='bfloat16',BM=bm,BN=bn,
        num_warps=4,num_stages=2,
    )
    return out,cache


def backward(dout,cache,*,dq=None,dk=None,dv=None,add_dkdv=False):
    """Return deterministic FP32 dQ,dK,dV (GQA reductions included)."""
    import cupy as cp
    q,k,v=cache.q,cache.k,cache.v
    if dout.shape!=q.shape:
        raise ValueError('upstream gradient shape mismatch')
    nq=q.shape[1]
    nk=k.shape[1]
    if dq is None:
        dq=cp.zeros(q.shape,dtype=cp.float32)
    if dk is None:
        dk=cp.zeros(k.shape,dtype=cp.float32)
    if dv is None:
        dv=cp.zeros(v.shape,dtype=cp.float32)
    if (dq.shape != q.shape or dk.shape != k.shape or dv.shape != v.shape or
            any(str(x.dtype) != 'float32' for x in (dq,dk,dv))):
        raise ValueError('provided attention gradients must be matching FP32 storage')
    if nk==0 or (cache.bounded and not cache.sparse_stride and nq<=cache.history_block_size):
        return dq,dk,dv
    qstr,kstr,vstr,ostr,dostr,lstr,dqstr,dkstr,dvstr=map(
        _strides,(q,k,v,cache.output,dout,cache.lse,dq,dk,dv))
    qds,qps,kds,kes,kvs=map(_strides,cache.metadata)
    common=(*map(_ptr,(q,k,v,*cache.metadata,cache.output,dout,cache.lse)),)
    bm = _SPARSE_BM if cache.sparse_stride else _BM
    bn = _SPARSE_BN if cache.sparse_stride else _BN
    q_tile_offset = cache.history_block_size // bm if cache.bounded and not cache.sparse_stride else 0
    q_tiles = triton.cdiv(max(0, nq - q_tile_offset * bm), bm)
    config=dict(NQ=nq,NK=nk,GROUP=q.shape[2]//k.shape[2],
                SCALE=cache.scale,SCALE_LOG2=cache.scale*_LOG2E,
                GLOBAL=cache.global_mask,SKIP_EMPTY=cache.skip_empty,
                BOUNDED=cache.bounded,HISTORY_BLOCK=cache.history_block_size,
                SUMMARIES_PER_BLOCK=cache.summaries_per_block,
                Q_TILE_OFFSET=q_tile_offset,
                SPARSE_STRIDE=cache.sparse_stride,SPARSE_PHASE=cache.sparse_phase,
                BF16=str(q.dtype)=='bfloat16',
                BM=bm,BN=bn,num_warps=4,num_stages=2)
    _dq[(q_tiles,q.shape[0]*q.shape[2])](
        *common,_ptr(dq),
        *qstr[:3],*kstr[:3],*vstr[:3],*ostr[:3],*dostr[:3],*lstr[:3],
        *dqstr[:3],*qds[:2],*qps[:2],*kds[:2],*kes[:2],*kvs[:2],
        HQ=q.shape[2],**config)
    _dkdv[(triton.cdiv(nk,bn),q.shape[0]*k.shape[2])](
        *common,_ptr(dk),_ptr(dv),
        *qstr[:3],*kstr[:3],*vstr[:3],*ostr[:3],*dostr[:3],*lstr[:3],
        *dkstr[:3],*dvstr[:3],*qds[:2],*qps[:2],*kds[:2],*kes[:2],*kvs[:2],
        HK=k.shape[2],ADD_DKDV=bool(add_dkdv),**config)
    return dq,dk,dv
