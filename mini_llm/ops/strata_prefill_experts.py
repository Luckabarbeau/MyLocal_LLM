"""Explicit reuse of Strata's fused int8 tensor-core expert kernels only.

Requires scripts/build_strata_prefill.py output, not a Strata server/engine.
This prefill arithmetic policy differs from decode's Q8 row-dot policy. Routing
still evaluates the full softmax/top-k, executes every selected expert, and uses
ascending-expert FP32 accumulation. No speculative decode or tail skipping.
"""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from .prefill_experts import _PackedStageRing


@lru_cache(maxsize=4)
def library(path):
    path = str(Path(path).resolve(strict=True))
    os.environ.setdefault('STRATA_PF_FUSED', '1')
    lib = C.CDLL(path)
    ptr, integer, size, count = C.c_void_p, C.c_int, C.c_size_t, C.c_int64
    signatures = {
        'width': ([], integer), 'ff': ([], integer),
        'type': ([C.c_char_p], integer), 'supported': ([integer,integer], integer),
        'act_bytes': ([count,count], size), 'group_bytes': ([count,integer], size),
        'quantize': ([ptr,count,count,ptr,ptr], None),
        'group': ([ptr,count,integer,integer,ptr,ptr,ptr,ptr], None),
        'experts': ([integer,integer,ptr,integer,integer,size,size,size,size,integer,count,
                     ptr,ptr,ptr,ptr,ptr,ptr], integer),
    }
    for name,(arguments,result) in signatures.items():
        fn = getattr(lib,'mylocal_strata_prefill_'+name)
        fn.argtypes,fn.restype = arguments,result
    return lib


_COMBINE = r'''
extern "C" __global__ void combine_strata_rows(const float* values, const int* slots,
        const long long* ids, const float* weights, float* output, int width, int k, int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=tokens*width) return;
    int token=i/width,col=i%width,order[16];
    for(int j=0;j<k;++j) order[j]=j;
    for(int j=1;j<k;++j) {
        int slot=order[j],p=j;
        while(p>0 && ids[token*k+order[p-1]]>ids[token*k+slot]) { order[p]=order[p-1]; --p; }
        order[p]=slot;
    }
    float sum=0.f;
    for(int j=0;j<k;++j) {
        int pair=token*k+order[j];
        sum=__fadd_rn(sum,__fmul_rn(values[(long long)slots[pair]*width+col],weights[pair]));
    }
    output[i]=sum;
}
'''


@lru_cache(maxsize=8)
def _combine(device):
    with xp.cuda.Device(device):
        return xp.RawKernel(_COMBINE,'combine_strata_rows',options=('--std=c++14','--fmad=false'))


def try_strata_prefill(experts,x,routes,ids):
    from .expert_residency import HeterogeneousExpertExecutor,ExpertKey
    executor=getattr(experts,'expert_executor',None)
    if (BACKEND_NAME!='cupy' or not isinstance(executor,HeterogeneousExpertExecutor)
            or executor.cache._admission_limit!=0):
        return None
    path=os.environ.get('MINI_LLM_STRATA_PREFILL_LIBRARY')
    if not path:
        raise ValueError('Strata fused prefill requires MINI_LLM_STRATA_PREFILL_LIBRARY')
    lib=library(path)
    width,ff=lib.mylocal_strata_prefill_width(),lib.mylocal_strata_prefill_ff()
    if (executor.d_model!=width or executor.d_ff!=ff or x.shape[0]!=1
            or x.dtype!=xp.float32 or routes.dtype!=xp.float32 or ids.dtype!=xp.int64
            or routes.shape!=ids.shape or routes.shape[:2]!=x.shape[:2]
            or not 1<=routes.shape[-1]<=16):
        raise ValueError('Strata fused kernel geometry/routing contract mismatch')
    tokens=x.shape[1]; k=routes.shape[-1]; n=tokens*k; ne=experts.n_experts
    if not 1<=ne<=512 or tokens<1 or not all(a.flags.c_contiguous for a in (x,routes,ids)):
        raise ValueError('Strata fused inputs must be contiguous, nonempty and use at most 512 experts')
    host_ids=asnumpy(ids).reshape(tokens,k)
    if (np.any(host_ids<0) or np.any(host_ids>=ne)
            or np.any(np.diff(np.sort(host_ids,axis=1),axis=1)==0)):
        raise ValueError('Invalid/duplicate expert IDs')
    active=tuple(int(i) for i in np.unique(host_ids))
    sources={e:executor.cpu_source_weights(ExpertKey(executor.model_id,experts.layer_id,e)) for e in active}
    gate,up,down=sources[active[0]]
    gu_type=lib.mylocal_strata_prefill_type(gate.quant_type.encode())
    down_type=lib.mylocal_strata_prefill_type(down.quant_type.encode())
    if (gu_type<0 or down_type<0 or gate.quant_type!=up.quant_type
            or not lib.mylocal_strata_prefill_supported(gu_type,down_type)):
        raise NotImplementedError(f'Strata fused prefill does not cover {gate.quant_type}/{down.quant_type}')
    layout=tuple((w.shape,w.quant_type,w.nbytes) for w in (gate,up,down))
    if any(tuple((w.shape,w.quant_type,w.nbytes) for w in ws)!=layout for ws in sources.values()):
        raise ValueError('Strata fused experts require uniform layer formats')
    stream=xp.cuda.get_current_stream()
    stream.synchronize()  # Complete any preceding native CPU service use.
    packed=xp.empty(lib.mylocal_strata_prefill_act_bytes(tokens,width),xp.uint8)
    scratch=xp.empty(lib.mylocal_strata_prefill_group_bytes(n,ne),xp.uint8)
    slots,src=xp.empty(n,xp.int32),xp.empty(n,xp.int32)
    hidden=xp.empty(lib.mylocal_strata_prefill_act_bytes(n,ff),xp.uint8)
    values=xp.empty((n,width),xp.float32)
    output=xp.empty((tokens,width),xp.float32)
    ids32=xp.ascontiguousarray(ids,dtype=xp.int32)
    lib.mylocal_strata_prefill_quantize(x.data.ptr,tokens,width,packed.data.ptr,stream.ptr)
    lib.mylocal_strata_prefill_group(ids32.data.ptr,n,k,ne,scratch.data.ptr,slots.data.ptr,src.data.ptr,stream.ptr)
    ring=_PackedStageRing(sum(w.nbytes for w in (gate,up,down)),slots=8)
    try:
        with executor.dispatch_scope(experts.layer_id,active):
            for first in range(0,ne,8):
                last=min(ne,first+8)
                pointers=np.zeros(last-first,np.uint64)
                staged=[]
                for e in range(first,last):
                    if e not in sources:
                        continue
                    key=ExpertKey(executor.model_id,experts.layer_id,e)
                    resident=executor.cache.resident.get(key)
                    if resident is not None:
                        expert=resident[0]
                        base=expert.gate.data.data.ptr
                        contiguous=(expert.up.data.data.ptr==base+gate.nbytes and
                                    expert.down.data.data.ptr==base+gate.nbytes+up.nbytes)
                    else:
                        contiguous=False
                    if contiguous:
                        pointers[e-first]=base
                    else:
                        slot,ws=ring.stage(sources[e],stream)
                        pointers[e-first]=ws[0].data.data.ptr
                        staged.append(slot)
                        executor.counters['prefill_streamed_bytes']+=sum(w.nbytes for w in sources[e])
                if not pointers.any():
                    continue
                rc=lib.mylocal_strata_prefill_experts(first,last,pointers.ctypes.data,
                    gu_type,down_type,gate.nbytes//ff,down.nbytes//width,
                    gate.nbytes,gate.nbytes+up.nbytes,ne,n,scratch.data.ptr,packed.data.ptr,
                    src.data.ptr,hidden.data.ptr,values.data.ptr,stream.ptr)
                if rc:
                    raise RuntimeError(f'Strata fused batch rejected: {rc}')
                for slot in staged:
                    ring.consumed(slot,stream)
            _combine(xp.cuda.runtime.getDevice())(((tokens*width+255)//256,),(256,),
                (values,slots,ids,routes,output,np.int32(width),np.int32(k),np.int32(tokens)))
            executor.counters['prefill_strata_layers']+=1
            executor.counters['prefill_strata_routed_rows']+=n
    finally:
        stream.synchronize()
        ring.fence()
    return output[None]
