"""Bounded packed host tier readable by CUDA; not a VRAM-resident cache.

Canonical mmap sources remain immutable. One pinned arena holds explicitly
selected complete expert copies. GPU aliases retain the arena even after the
manager is dropped. No full-bank decode, per-token allocation or CUDA callback.
"""
import ctypes as C
import ctypes.util
from functools import lru_cache
from pathlib import Path
import numpy as np
from mini_llm.backend import BACKEND_NAME,xp
from .quantized_weight import PackedGGUFWeight
from .quantized_linear import DevicePackedGGUFWeight,PackedExpertFFNInference


@lru_cache(maxsize=1)
def _runtime():
    name=C.util.find_library('cudart')
    if not name: raise RuntimeError('Mapped packed experts require CUDA runtime')
    lib=C.CDLL(name)
    lib.cudaHostGetDevicePointer.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint]
    lib.cudaHostGetDevicePointer.restype=C.c_int
    return lib


def available_bytes():
    values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(values['MemAvailable'].split()[0])*1024


class MappedPackedExpertTier:
    """Single-device setup-only arena; caller retires GPU consumers before close.

    Weight consumers must not write the aliases. The only precision conversion
    is in the selected packed arithmetic kernel, never in storage or transfer.
    """
    def __init__(self,budget_bytes,*,reserve_bytes=4*1024**3):
        if BACKEND_NAME!='cupy' or xp.cuda.runtime.is_hip:
            raise RuntimeError('Mapped packed experts require CUDA CuPy')
        for name,value in (('budget_bytes',budget_bytes),('reserve_bytes',reserve_bytes)):
            if isinstance(value,bool) or not isinstance(value,int) or value<(1 if name=='budget_bytes' else 0):
                raise ValueError(f'{name} must be a nonnegative integer (budget positive)')
        if available_bytes()<budget_bytes+reserve_bytes:
            raise MemoryError('Insufficient available RAM for bounded mapped expert arena')
        self.device=xp.cuda.runtime.getDevice()
        self.owner=xp.cuda.alloc_pinned_memory(budget_bytes)
        self.nbytes=memoryview(self.owner).nbytes
        if self.nbytes>budget_bytes:
            self.owner=None
            raise MemoryError('Pinned allocator rounding exceeds mapped expert budget')
        self.host=np.frombuffer(self.owner,np.uint8)
        pointer=C.c_void_p()
        error=_runtime().cudaHostGetDevicePointer(C.byref(pointer),self.host.ctypes.data,0)
        if error:
            self.host=None; self.owner=None
            raise RuntimeError(f'cudaHostGetDevicePointer failed: {error}')
        self.device_pointer=pointer.value
        self.used_bytes=0
        self.experts={}
        self.sources={}
        self.closed=False

    def add(self,key,weights):
        if self.closed: raise RuntimeError('Mapped packed arena is closed')
        if key in self.experts: raise ValueError('Mapped expert already installed')
        weights=tuple(weights)
        if len(weights)!=3 or not all(isinstance(w,PackedGGUFWeight) for w in weights):
            raise TypeError('Mapped expert requires packed gate/up/down')
        g,u,d=weights
        if len(g.shape)!=2 or g.shape!=u.shape or d.shape!=(g.shape[1],g.shape[0]):
            raise ValueError('Invalid mapped expert projection shapes')
        total=sum(w.nbytes for w in weights)
        start=(self.used_bytes+255)//256*256
        if start+total>self.nbytes: raise MemoryError('Mapped expert budget exhausted')
        arrays=[]; offset=start
        # Array owner is the pinned allocation, not this manager (no self-cycle).
        memory=xp.cuda.UnownedMemory(self.device_pointer+start,total,self.owner,device_id=self.device)
        with xp.cuda.Device(self.device):
            for w in weights:
                np.copyto(self.host[offset:offset+w.nbytes],w.data.reshape(-1))
                data=xp.ndarray((w.nbytes,),dtype=xp.uint8,
                    memptr=xp.cuda.MemoryPointer(memory,offset-start))
                arrays.append(DevicePackedGGUFWeight(data,w.shape,w.quant_type))
                offset+=w.nbytes
        expert=PackedExpertFFNInference(*arrays)
        self.experts[key]=expert; self.sources[key]=weights
        self.used_bytes=offset
        return expert

    def statistics(self):
        return dict(storage='bounded_pinned_host',execution='cuda_pcie_reads',
            reserved_host_bytes=self.nbytes,used_host_bytes=self.used_bytes,
            gpu_weight_bytes=0,experts=len(self.experts))

    def close(self):
        """Caller must first retire every stream consuming this tier's weights."""
        self.closed=True
        self.experts.clear(); self.sources.clear()
        self.host=None; self.owner=None
