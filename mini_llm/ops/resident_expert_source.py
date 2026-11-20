"""Compact, bounded anonymous host copies, independent of immutable GGUF sources.

Only selected complete packed bundles are copied. Registration is paced, not a
single huge cudaHostAlloc; an unregistered suffix remains valid native CPU RAM.
The caller retires all CPU/DMA consumers before close. PLE is never copied.
"""
import mmap
import os
from pathlib import Path
import time
import numpy as np
from mini_llm.backend import BACKEND_NAME,xp
from .expert_source import ExpertBundle
from .quantized_weight import PackedGGUFWeight
from .expert_host_prefetch import memory_info

_QUARANTINE=[]


def host_available_bytes(*,cgroup_root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup')):
    available=memory_info()['MemAvailable']
    lines=membership.read_text().splitlines()
    unified=[line.split('::',1)[1] for line in lines if line.startswith('0::')]
    if not unified:
        if any('memory' in line.split(':')[1].split(',') for line in lines):
            raise RuntimeError('Resident source requires verified cgroup-v2 or host RAM bounds')
        return available
    parts=Path(unified[0]).parts[1:]
    if '..' in parts: raise RuntimeError('Unsupported cgroup namespace path')
    directory=cgroup_root.joinpath(*parts)
    while True:
        limit=directory/'memory.max'
        if limit.is_file():
            value=limit.read_text().strip()
            if value!='max':
                used=int((directory/'memory.current').read_text())
                available=min(available,max(0,int(value)-used))
        if directory==cgroup_root: break
        directory=directory.parent
    return available


class ResidentPackedExpertSource:
    """Setup-only compact host complement; sources/fallback stay in the GGUF store."""
    def __init__(self,provider,keys,*,budget_bytes,reserve_bytes=4*1024**3,
                 pin=False,pin_limit_bytes=None,chunk_bytes=32*1024**2,progress=None):
        page=os.sysconf('SC_PAGE_SIZE')
        for name,value in (('budget_bytes',budget_bytes),('reserve_bytes',reserve_bytes),('chunk_bytes',chunk_bytes)):
            if isinstance(value,bool) or not isinstance(value,int) or value<(1 if name!='reserve_bytes' else 0):
                raise ValueError(f'{name} must be a positive integer (reserve may be zero)')
        if not isinstance(pin,bool): raise TypeError('pin must be bool')
        if chunk_bytes%page or chunk_bytes>64*1024**2:
            raise ValueError('Registration chunk must be page aligned and <=64MiB')
        if pin and (BACKEND_NAME!='cupy' or xp.cuda.runtime.is_hip):
            raise RuntimeError('Pinned resident source requires CUDA CuPy')
        if pin_limit_bytes is not None and (isinstance(pin_limit_bytes,bool) or
                not isinstance(pin_limit_bytes,int) or pin_limit_bytes<0):
            raise ValueError('pin_limit_bytes must be nonnegative')
        keys=tuple(keys)
        if len(keys)>32768 or len(set(keys))!=len(keys):
            raise ValueError('Resident source keys must be unique and bounded')
        planned=[]; used=0
        for key in keys:
            weights=tuple(provider(key))
            # Existing source contract validates complete shapes/read-only bytes.
            bundle=ExpertBundle(key,weights,weights,'mmap')
            if not all(isinstance(w,PackedGGUFWeight) for w in weights):
                raise TypeError('Resident source requires packed projections')
            if bundle.nbytes>64*1024**2:
                raise ValueError('Resident expert bundle exceeds bounded64MiB transfer unit')
            start=(used+page-1)//page*page
            end=start+bundle.nbytes
            if (end+page-1)//page*page>budget_bytes: continue
            planned.append((key,weights,start,end)); used=end
        if not planned: raise MemoryError('Resident expert budget cannot fit one complete bundle')
        self.nbytes=(used+page-1)//page*page
        if host_available_bytes()<self.nbytes+reserve_bytes:
            raise MemoryError('Insufficient available RAM/cgroup headroom for resident expert source')
        self.mapping=mmap.mmap(-1,self.nbytes,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
        self.raw=np.frombuffer(self.mapping,np.uint8)
        self.bundles={}; self.offsets={}; self.registered=[]
        self.pinned_bytes=0; self.pin_error=None; self.closed=False
        self.requested_keys=len(keys); self.reserve_bytes=reserve_bytes
        self.used_bytes=used; self.seconds=0.0
        pin_end=min(self.nbytes,self.nbytes if pin_limit_bytes is None else pin_limit_bytes//page*page)
        pin_enabled=pin and pin_end>0
        next_progress=chunk_bytes
        started=time.perf_counter()
        try:
            for key,weights,start,end in planned:
                offset=start; copied=[]
                for weight in weights:
                    view=self.raw[offset:offset+weight.nbytes]
                    np.copyto(view,weight.data.reshape(-1))
                    view.flags.writeable=False
                    copied.append(PackedGGUFWeight(view,weight.shape,weight.quant_type))
                    offset+=weight.nbytes
                self.bundles[key]=ExpertBundle(key,tuple(copied),self.mapping,'host')
                self.offsets[key]=(start,end)
                # CUDA memcpy cannot span separately registered host ranges.
                # End paced regions only at page-aligned expert boundaries; an
                # optional pin cap may leave the final expert CPU-only.
                ready=min((end+page-1)//page*page,pin_end)
                count=ready-self.pinned_bytes
                if pin_enabled and count>0 and (count>=chunk_bytes or end==used or ready==pin_end):
                    if host_available_bytes()<reserve_bytes+count:
                        raise MemoryError('RAM reserve reached during paced source registration')
                    pointer=self.raw.ctypes.data+self.pinned_bytes
                    try:
                        xp.cuda.runtime.hostRegister(pointer,count,2) # mapped, owned RW anonymous storage
                    except xp.cuda.runtime.CUDARuntimeError as exc:
                        if exc.status not in (2,801): raise
                        self.pin_error=str(exc); pin_enabled=False
                    else:
                        self.registered.append((pointer,count)); self.pinned_bytes+=count
                if end>=next_progress or end==used:
                    if host_available_bytes()<reserve_bytes:
                        raise MemoryError('RAM reserve reached while filling resident expert source')
                    if progress is not None: progress(end,used)
                    next_progress=end+chunk_bytes
            self.seconds=time.perf_counter()-started
        except Exception:
            # Clear local exports too; close drops the mmap owner only after unpin.
            copied=[]; view=None
            self.close()
            raise

    def weights(self,key):
        if self.closed: raise RuntimeError('Resident expert source is closed')
        return self.bundles[key].weights

    def dma_pointer(self,key):
        if self.closed: raise RuntimeError('Resident expert source is closed')
        bounds=self.offsets.get(key)
        if bounds is None or bounds[1]>self.pinned_bytes: return 0
        return self.raw.ctypes.data+bounds[0]

    def statistics(self):
        return dict(mode='compact_anonymous_host',requested_experts=self.requested_keys,
            resident_experts=len(self.bundles),complete_complement=len(self.bundles)==self.requested_keys,
            private_weight_copy_bytes=self.nbytes,used_bytes=self.used_bytes,
            pinned_expert_source_bytes=self.pinned_bytes,pin_error=self.pin_error,
            ram_reserve_bytes=self.reserve_bytes,load_seconds=self.seconds,
            gpu_weight_bytes=0,ple_copy_bytes=0)

    def close(self):
        if self.closed: return
        try:
            for pointer,_ in reversed(self.registered): xp.cuda.runtime.hostUnregister(pointer)
        except Exception:
            # Unknown driver registration lifetime must not free its backing.
            _QUARANTINE.append((self.mapping,self.raw,list(self.registered)))
            raise
        finally:
            self.closed=True
        self.registered.clear(); self.bundles.clear(); self.offsets.clear()
        self.raw=None
        # Exported CPU views retain the mmap owner; do not forcibly invalidate them.
        self.mapping=None
