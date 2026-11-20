"""Opt-in native mapped-doorbell CPU service; no per-layer CUDA host node.

A single native thread per row pool consumes stream-published requests. CUDA
waits are bounded, not an unsupported cuStreamWaitValue32 assumption. One model
completion check rejects failed tokens before successful state advancement.
"""
import ctypes as C
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from mini_llm.backend import xp
from .mapped_packed_experts import _runtime

_SOURCE=r'''
__device__ unsigned int acquire(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ void release(unsigned int* p,unsigned int v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p),"r"(v) : "memory");
}
__device__ unsigned long long timer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
extern "C" __global__ void publish(unsigned int* flags,unsigned int* failure) {
    if(threadIdx.x || blockIdx.x || acquire(failure)) return;
    unsigned int request=acquire(flags);
    if(request==0xffffffffu) { release(failure,3); return; }
    __threadfence_system();
    release(flags,request+1);
}
extern "C" __global__ void publish_inputs(const unsigned int* x,const int* metadata,
    unsigned int* host_x,int* host_metadata,unsigned int* flags,unsigned int* failure,int d,int k) {
    __shared__ unsigned int bad;
    if(!threadIdx.x) bad=acquire(failure);
    __syncthreads();
    if(bad) return;
    for(int i=threadIdx.x;i<d;i+=blockDim.x) host_x[i]=x[i];
    for(int i=threadIdx.x;i<=k;i+=blockDim.x) host_metadata[i]=metadata[i];
    // Every writer must finish its system-scope stores before lane zero
    // publishes readiness. A block barrier alone does not order host visibility.
    __threadfence_system();
    __syncthreads();
    if(!threadIdx.x && !acquire(failure)) {
        unsigned int request=acquire(flags);
        if(request==0xffffffffu) release(failure,3);
        else release(flags,request+1);
    }
}
extern "C" __global__ void await_cpu(const unsigned int* flags,unsigned int* failure,
                                     unsigned long long timeout_ns,unsigned int* device_status) {
    if(threadIdx.x || blockIdx.x) return;
    unsigned int request=acquire(flags),bad=acquire(failure);
    unsigned long long start=timer();
    while(!bad && acquire(flags+16)!=request) {
        if(timer()-start>timeout_ns) { release(failure,1); bad=1; break; }
        __nanosleep(64);
        bad=acquire(failure);
    }
    // Completion acquires the worker's preceding failure publication too.
    if(!bad) bad=acquire(failure);
    // Downstream copies/poison read VRAM, not one PCIe status per payload warp.
    device_status[0]=bad;
}
extern "C" __global__ void copy_complete_rows(const float* src,float* dst,const unsigned int* failure,int n) {
    int i=blockIdx.x*256+threadIdx.x;
    // A timed-out CPU job may still write src: never read its incomplete slab.
    if(i<n && !failure[0]) dst[i]=src[i];
}
extern "C" __global__ void poison_failed(float* out,const unsigned int* failure,int d) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<d && failure[0]) out[i]=__int_as_float(0x7fc00000);
}
'''


@lru_cache(maxsize=8)
def _kernels(device):
    with xp.cuda.Device(device):
        module=xp.RawModule(code=_SOURCE,options=('--std=c++17',))
        return tuple(module.get_function(n) for n in ('publish','await_cpu','poison_failed','copy_complete_rows','publish_inputs'))


def _mapped(size):
    owner=xp.cuda.alloc_pinned_memory(size)
    host=np.frombuffer(owner,np.uint32,count=size//4)
    host.fill(0)
    if host.ctypes.data%64: raise RuntimeError('Native service flags must be cache-line aligned')
    pointer=C.c_void_p()
    error=_runtime().cudaHostGetDevicePointer(C.byref(pointer),host.ctypes.data,0)
    if error: raise RuntimeError(f'Native service mapped flags unavailable: {error}')
    memory=xp.cuda.UnownedMemory(pointer.value,size,owner,device_id=xp.cuda.runtime.getDevice())
    device=xp.ndarray((size//4,),xp.uint32,memptr=xp.cuda.MemoryPointer(memory,0))
    return SimpleNamespace(owner=owner,host=host,device=device)


def mapped_output(host,owner):
    pointer=C.c_void_p()
    error=_runtime().cudaHostGetDevicePointer(C.byref(pointer),host.ctypes.data,0)
    if error: raise RuntimeError(f'Native service output mapping unavailable: {error}')
    memory=xp.cuda.UnownedMemory(pointer.value,host.nbytes,owner,device_id=xp.cuda.runtime.getDevice())
    return xp.ndarray(host.shape,host.dtype,memptr=xp.cuda.MemoryPointer(memory,0))


class NativeExpertService:
    def __init__(self,pool):
        self.pool=pool
        self.bridge=pool.executor._bridge
        b=self.bridge
        b.mylocal_ggml_service_create.argtypes=[C.c_void_p,C.c_void_p]
        b.mylocal_ggml_service_create.restype=C.c_void_p
        b.mylocal_ggml_service_add.argtypes=[C.c_void_p]*3
        b.mylocal_ggml_service_add.restype=C.c_int
        b.mylocal_ggml_service_remove.argtypes=[C.c_void_p]*2
        b.mylocal_ggml_service_remove.restype=C.c_int
        b.mylocal_ggml_service_destroy.argtypes=[C.c_void_p]
        b.mylocal_ggml_service_destroy.restype=None
        self.publish,self.wait,self.poison,self.copy_rows,self.publish_inputs=_kernels(xp.cuda.runtime.getDevice())
        self.clients={}
        self.stage_arena=None
        self.failure=_mapped(64)
        self.device_failure=xp.zeros(1,xp.uint32)
        self.handle=b.mylocal_ggml_service_create(pool._handle,self.failure.host.ctypes.data)
        if not self.handle: raise RuntimeError('Cannot create bounded native expert service')
        self.timeout_ns=np.uint64(2_000_000_000)

    def register(self,handoff):
        flags=_mapped(128)
        self.pool.executor._check(self.bridge.mylocal_ggml_service_add(
            self.handle,handoff,flags.host.ctypes.data))
        self.clients[handoff]=flags
        return flags

    def unregister(self,handoff):
        self.pool.executor._check(self.bridge.mylocal_ggml_service_remove(self.handle,handoff))
        del self.clients[handoff]

    def check(self):
        # Readback is ordered after the model's GPU consumers, once per token.
        # Accessing only a host scalar without this dependency would race.
        error=int(self.device_failure.get()[0])
        if error:
            reason={1:'bounded GPU wait timed out',2:'native CPU batch failed',3:'epoch exhausted'}.get(error,str(error))
            raise RuntimeError(f'Native expert service failed: {reason}; unload failed executor')

    def close(self):
        if self.handle:
            if self.clients: raise RuntimeError('Retire native handoffs before closing their service')
            self.bridge.mylocal_ggml_service_destroy(self.handle)
            self.handle=None
            # Every client has fenced its GPU readers before unregistering.
            self.stage_arena=None
