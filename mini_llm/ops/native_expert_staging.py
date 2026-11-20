"""Opt-in bounded service-thread CPU/PCIe split; no CUDA from host callbacks.

A single 16MiB packed arena is shared by the serial model residual chain. Each
bundle is <=8MiB; the GPU pulls complete published host bundles before compute. No weight cache
promotion or full expert-bank copy is performed. CPU and GPU Q8 remain distinct.
"""
import ctypes as C
from functools import lru_cache
import os
from types import SimpleNamespace
import numpy as np
from mini_llm.backend import xp

ARENA_BYTES = 16*1024**2
SLOT_BYTES = ARENA_BYTES//2


def stage_limit():
    value = os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_LIMIT', '0')
    if value not in ('0', '1', '2'):
        raise ValueError('MINI_LLM_NATIVE_MOE_STAGE_LIMIT must be 0, 1 or 2')
    return int(value)


_EXTRA = r'''
extern "C" __global__ void await_stage(const unsigned int* flags,const unsigned int* stage,
    unsigned int* failure,unsigned long long timeout_ns,unsigned int* status) {
    if(threadIdx.x || blockIdx.x) return;
    unsigned int request=acquire(flags),bad=acquire(failure);
    unsigned long long start=timer();
    while(!bad && acquire(stage)!=request) {
        if(timer()-start>timeout_ns) { release(failure,1); bad=1; break; }
        __nanosleep(64); bad=acquire(failure);
    }
    if(!bad) bad=acquire(failure);
    status[0]=bad;
}
extern "C" __global__ void staged_pointers(const int* map,unsigned long long base,
    unsigned long long stride,unsigned long long gate,unsigned long long up,
    unsigned long long* selected,const unsigned int* bad,unsigned int* count,
    const unsigned long long* host_sources,unsigned long long* sources,int k) {
    int e=threadIdx.x;
    if(!e) count[0]=0;
    if(e<2) sources[e]=bad[0] ? 0 : host_sources[e];
    __syncthreads();
    if(e>=k) return;
    // Do not race an incomplete host map after timeout or producer failure.
    int slot=bad[0] ? -1 : map[e];
    if(slot>=0) atomicMax(count,unsigned(slot+1));
    unsigned long long ptr=slot<0 ? 0 : base+stride*slot;
    selected[e*3]=ptr; selected[e*3+1]=ptr ? ptr+gate : 0;
    selected[e*3+2]=ptr ? ptr+gate+up : 0;
}
extern "C" __global__ void copy_staged_packed(const unsigned char* host,unsigned char* device,
    unsigned long long stride,unsigned long long bytes,const unsigned int* count,
    const unsigned long long* sources) {
    unsigned int slot=blockIdx.y;
    if(slot>=count[0]) return;
    unsigned long long i=(blockIdx.x*256ull+threadIdx.x)*16ull;
    host=sources[slot] ? reinterpret_cast<const unsigned char*>(sources[slot]) : host+stride*slot;
    device+=stride*slot;
    if(i+16<=bytes) *reinterpret_cast<uint4*>(device+i)=*reinterpret_cast<const uint4*>(host+i);
    else for(unsigned long long j=i;j<bytes;++j) device[j]=host[j];
}
extern "C" __global__ void combine_staged(const float* gpu,const float* staged,const float* cpu,
    const unsigned long long* selected,const int* map,const float* routes,
    const unsigned int* bad,float* out,int d,int k) {
    int col=blockIdx.x*256+threadIdx.x;
    if(col>=d) return;
    if(bad[0]) { out[col]=__int_as_float(0x7fc00000); return; }
    float result=0.0f; int miss=0;
    for(int e=0;e<k;++e) {
        float term;
        if(selected[e*3]) term=gpu[e*d+col];
        else if(map[e]>=0) term=staged[e*d+col];
        else { term=cpu[miss*d+col]*routes[e]; ++miss; }
        result+=term;
    }
    out[col]=result;
}
'''


@lru_cache(maxsize=8)
def _kernels(device):
    from .native_expert_service import _SOURCE
    with xp.cuda.Device(device):
        module=xp.RawModule(code=_SOURCE+_EXTRA,options=('--std=c++17','--fmad=false'))
        return tuple(module.get_function(n) for n in ('await_stage','staged_pointers','combine_staged','copy_staged_packed'))


def _arena(service):
    if getattr(service,'stage_arena',None) is None:
        owner=xp.cuda.alloc_pinned_memory(ARENA_BYTES)
        host=np.frombuffer(owner,np.uint8)
        device=xp.empty(ARENA_BYTES,xp.uint8)
        from .native_expert_service import mapped_output
        mapped=mapped_output(host.view(np.float32),owner).view(xp.uint8)
        service.stage_arena=SimpleNamespace(owner=owner,host=host,device=device,mapped=mapped)
    return service.stage_arena


class NativeExpertStaging:
    def __init__(self,handoff,limit):
        from .native_expert_service import _mapped
        from .mixed_q8_gpu import GroupedQ8Plan
        p=handoff.plan
        if p.q8 is None or handoff.service is None or limit not in (1,2):
            raise ValueError('Native staging requires Q8 GPU experts and native service')
        parts=np.asarray([p.resident_row_bytes[0]*p.f,p.resident_row_bytes[1]*p.f,
                          p.resident_row_bytes[2]*p.d],dtype=np.uintp)
        if int(parts.sum())>SLOT_BYTES:
            raise ValueError('Complete expert exceeds bounded native stage slot')
        self.handoff=handoff
        self.service=handoff.service
        self.arena=_arena(self.service)
        self.source_mode=os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_SOURCE','bounce')
        if self.source_mode not in ('bounce','pinned'):
            raise ValueError('Native stage source must be bounce or pinned')
        self.flags=_mapped(192)
        self.mapping=self.flags.host.view(np.int32)[16:16+p.k]
        self.mapping.fill(-1)
        self.device_mapping=self.flags.device.view(xp.int32)[16:16+p.k]
        self.host_sources=self.flags.host.view(np.uint64)[16:18]
        self.mapped_sources=self.flags.device.view(xp.uint64)[16:18]
        self.device_sources=xp.zeros(2,xp.uint64)
        self.aliases=np.zeros(p.n_experts,np.uint64)
        # Each alias addresses one complete bundle inside one registered region.
        # Query actual CUDA aliases; never assume host/device pointer equality.
        source=p.executor._resident_source
        if source is not None and source.pinned_bytes:
            from .expert_residency import ExpertKey
            from .native_expert_service import _runtime
            for expert in range(p.n_experts):
                key=ExpertKey(p.executor.model_id,p.layer,expert)
                host=source.dma_pointer(key)
                if not host or expert in p.resident: continue
                pointer=C.c_void_p()
                error=_runtime().cudaHostGetDevicePointer(C.byref(pointer),host,0)
                if error: raise RuntimeError(f'Pinned expert device alias unavailable: {error}')
                self.aliases[expert]=pointer.value
        self.wait,self.select,self.combine,self.copy=_kernels(p.device)
        self.count=xp.zeros(1,xp.uint32)
        self.plan=SimpleNamespace(d=p.d,f=p.f,k=p.k,device=p.device,formats=p.formats,
            resident_row_bytes=p.resident_row_bytes,routes=p.routes,
            selected=xp.empty_like(p.selected),hidden=xp.empty_like(p.hidden),terms=xp.empty_like(p.terms))
        self.q8=GroupedQ8Plan(self.plan)
        self.parts=parts
        self.limit=limit
        # Exploratory0292 supports splitting larger miss sets, not blanket staging.
        self.minimum=5
        self.stats=np.zeros(3,np.uint64)
        self.accounted=np.zeros(3,np.uint64)
        self.production=np.zeros(3,np.uint64)
        b=self.service.bridge
        b.mylocal_ggml_stage_create.argtypes=[C.c_void_p]*2+[C.c_int]*2+[C.c_size_t]+[C.c_void_p]*5+[C.c_int]
        b.mylocal_ggml_stage_create.restype=C.c_void_p
        b.mylocal_ggml_service_set_stage.argtypes=[C.c_void_p]*3
        b.mylocal_ggml_service_set_stage.restype=C.c_int
        b.mylocal_ggml_stage_stats.argtypes=[C.c_void_p,C.c_void_p]
        b.mylocal_ggml_stage_stats.restype=None
        b.mylocal_ggml_stage_destroy.argtypes=[C.c_void_p]
        b.mylocal_ggml_stage_destroy.restype=None
        self.handle=b.mylocal_ggml_stage_create(handoff.handle,self.arena.host.ctypes.data,
            limit,self.minimum,SLOT_BYTES,parts.ctypes.data,self.mapping.ctypes.data,self.flags.host.ctypes.data,
            self.aliases.ctypes.data,self.host_sources.ctypes.data,int(self.source_mode=='pinned'))
        if not self.handle:
            raise RuntimeError('Cannot configure bounded native expert staging')
        try:
            self.service.pool.executor._check(b.mylocal_ggml_service_set_stage(
                self.service.handle,handoff.handle,self.handle))
        except Exception:
            b.mylocal_ggml_stage_destroy(self.handle); self.handle=None
            raise

    def enqueue(self,x):
        h=self.handoff; p=h.plan; s=self.service
        self.wait((1,),(32,),(h.service_flags.device,self.flags.device,s.failure.device,
                              s.timeout_ns,s.device_failure))
        self.select((1,),(32,),(self.device_mapping,np.uint64(self.arena.device.data.ptr),
            np.uint64(SLOT_BYTES),np.uint64(self.parts[0]),np.uint64(self.parts[1]),
            self.plan.selected,s.device_failure,self.count,self.mapped_sources,self.device_sources,np.int32(p.k)))
        self.copy(((int(self.parts.sum())+4095)//4096,self.limit),(256,),
            (self.arena.mapped,self.arena.device,np.uint64(SLOT_BYTES),
             np.uint64(self.parts.sum()),self.count,self.device_sources))
        self.q8.forward(x)

    def combine_output(self):
        p=self.handoff.plan
        self.combine(((p.d+255)//256,),(256,),
            (p.terms,self.plan.terms,p.cpu_rows,p.selected,self.device_mapping,p.routes,
             self.service.device_failure,p.output,np.int32(p.d),np.int32(p.k)))

    def collect(self,*,warmup=False):
        self.service.bridge.mylocal_ggml_stage_stats(self.handle,self.stats.ctypes.data)
        delta=self.stats-self.accounted
        self.accounted[:]=self.stats
        if not warmup: self.production+=delta
        return tuple(int(v) for v in delta)

    def close(self):
        # Caller unregisters from the service and fences GPU readers first.
        if self.handle:
            self.service.bridge.mylocal_ggml_stage_destroy(self.handle)
            self.handle=None
