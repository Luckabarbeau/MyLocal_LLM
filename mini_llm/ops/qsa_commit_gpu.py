"""Conditional QSA cache commit after device finite validation, no host fence."""
from functools import lru_cache
import numpy as np
from mini_llm.backend import xp


@lru_cache(maxsize=16)
def _kernel(device):
    source=r'''
    extern "C" __global__ void commit_qsa(const unsigned int* bad,
        const float* k, const float* v, const float* raw, const float* pool,
        float* keys, float* values, float* history, float* pooled,
        long long t, long long block, int kd, int rd, int write_pool) {
        if (*bad) return;
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i<kd) { keys[t*kd+i]=k[i]; values[t*kd+i]=v[i]; }
        if (i<rd) {
            history[t*rd+i]=raw[i];
            if (write_pool) pooled[block*rd+i]=pool[i];
        }
    }
    '''
    with xp.cuda.Device(device):
        return xp.RawKernel(source,'commit_qsa',options=('--std=c++11',))


def commit_qsa(status,state,k,v,raw,new_pool,t,completed):
    # Only explicitly opted-in model execution supplies this status; standalone
    # QSA retains synchronous validation and its existing general array path.
    arrays=(state.keys,state.values,state.raw_index,state.pooled,k,v,raw)
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError('Deferred QSA commit requires contiguous caches/projections')
    kd,rd=k.size,raw.size
    _kernel(xp.cuda.runtime.getDevice())(((max(kd,rd)+255)//256,),(256,),
        (status,k,v,raw,raw if new_pool is None else new_pool,
         state.keys,state.values,state.raw_index,state.pooled,
         np.int64(t),np.int64(completed-1),np.int32(kd),np.int32(rd),np.int32(new_pool is not None)))
