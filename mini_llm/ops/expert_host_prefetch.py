"""Budgeted startup warming of canonical packed expert mmap pages, without copies.

Only routed gate/up/down tensors are touched. PLE/embeddings/backbone are never
scanned here. Pages remain reclaimable by the OS: this is not mlock/pinned RAM,
and no second complete packed or floating-point expert copy is allocated.
"""
import ctypes
import os
from pathlib import Path
import time

import numpy as np

GiB = 1024**3
_SUFFIXES = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')


def memory_info():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:', 'MemAvailable:')):
            values[line.split(':', 1)[0]] = int(line.split()[1])*1024
    if len(values) != 2:
        raise RuntimeError('Linux MemAvailable is required for safe expert warmup')
    return values


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def resident_bytes(raw):
    """Linux mincore estimate, rounded/clamped per region; no payload reads."""
    if not raw.size:
        return 0
    page = os.sysconf('SC_PAGE_SIZE')
    pointer = int(raw.ctypes.data)
    aligned = pointer // page * page
    pages = (pointer-aligned+int(raw.nbytes)+page-1)//page
    flags = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.mincore
    fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    fn.restype = ctypes.c_int
    if fn(ctypes.c_void_p(aligned), pages*page, flags):
        raise OSError(ctypes.get_errno(), 'mincore failed for expert mapping')
    count = int(np.count_nonzero(np.ctypeslib.as_array(flags) & 1))
    return min(int(raw.nbytes), count*page)


def warm_expert_pages(store, *, budget_bytes='auto', reserve_bytes=8*GiB,
                      chunk_bytes=32*1024**2, progress=None):
    """Fault at most budget bytes of expert payload into reclaimable host RAM.

    'auto' uses MemAvailable minus reserve; explicit budgets are capped by the
    same current availability. Recheck the reserve before every bounded chunk.
    Return touched and independently queried resident payload estimates. mincore
    can include partial boundary pages; OS pressure can reclaim warmed pages.
    """
    _nonnegative(reserve_bytes, 'reserve_bytes')
    _nonnegative(chunk_bytes, 'chunk_bytes')
    if chunk_bytes < 1 or chunk_bytes > 64*1024**2:
        raise ValueError('chunk_bytes must be in [1, 64 MiB]')
    if budget_bytes != 'auto':
        _nonnegative(budget_bytes, 'budget_bytes')
    descriptors = sorted((d for name, d in store.tensors.items() if name.endswith(_SUFFIXES)),
                         key=lambda d: (d.shard_index, d.data_offset))
    total = sum(d.nbytes for d in descriptors)
    before = memory_info()
    safe = max(0, before['MemAvailable']-reserve_bytes)
    target = min(total, safe, total if budget_bytes == 'auto' else budget_bytes)
    touched, checksum, regions = 0, 0, []
    start = time.perf_counter()
    reason = None
    page = os.sysconf('SC_PAGE_SIZE')
    for descriptor in descriptors:
        if touched >= target or reason:
            break
        raw = store.tensor_bytes(descriptor.name)
        if raw.dtype != np.uint8 or not raw.flags.c_contiguous or raw.flags.writeable:
            raise ValueError('Expert prefetch requires read-only contiguous canonical bytes')
        offset = 0
        while offset < descriptor.nbytes and touched < target:
            if memory_info()['MemAvailable'] < reserve_bytes:
                reason = 'system RAM reserve reached'
                break
            count = min(chunk_bytes, descriptor.nbytes-offset, target-touched)
            view = raw[offset:offset+count]
            # One byte per OS page faults the original mapping, not a copy.
            checksum += int(np.sum(view[::page], dtype=np.uint64)) + int(view[-1])
            regions.append((descriptor.name, offset, count))
            touched += count
            offset += count
            if progress is not None:
                progress(touched, target)
    resident = 0
    for name, offset, count in regions:
        resident += resident_bytes(store.tensor_bytes(name)[offset:offset+count])
    return dict(mode='reclaimable_mmap_pages', total_expert_bytes=total,
        requested_budget_bytes=budget_bytes, planned_bytes=target, touched_bytes=touched,
        resident_bytes_estimate=resident, ram_reserve_bytes=reserve_bytes,
        available_before_bytes=before['MemAvailable'], available_after_bytes=memory_info()['MemAvailable'],
        complete_expert_source=touched == total, stop_reason=reason,
        seconds=time.perf_counter()-start, checksum=checksum,
        private_weight_copy_bytes=0, pinned_expert_source_bytes=0,
        note='Resident estimate is page-rounded; pages remain OS-reclaimable, not permanently pinned.')
