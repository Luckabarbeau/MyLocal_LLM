"""Bounded fixed-stride packed expert slots with complete synchronous publication.

Canonical host weights remain caller-owned. One pinned staging slab and one
copy stream are reused; no dense expert expansion or per-miss GPU weight malloc.
Projection arrays use UnownedMemory leases, so even a retained projection view
prevents its old slot being recycled. Cache policy must separately protect
in-flight consumers before dropping its handles. Async admission is not claimed.
"""
import threading

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from .quantized_linear import DevicePackedGGUFWeight, PackedExpertFFNInference
from .quantized_weight import PackedGGUFWeight


class _Lease:
    def __init__(self, arena, slot):
        self.arena, self.slot = arena, slot
        self.reusable = True

    def __del__(self):
        if self.reusable:
            self.arena._release(self.slot)


class PackedExpertSlotArena:
    def __init__(self, budget_bytes, stride_bytes, *, staging_budget_bytes=64*1024**2,
                 slot_sizes=None):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('Packed expert slots require CuPy')
        for name, value in (('budget_bytes', budget_bytes), ('stride_bytes', stride_bytes),
                            ('staging_budget_bytes', staging_budget_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if stride_bytes % 256 or stride_bytes > budget_bytes:
            raise ValueError('Slot stride must be 256-byte aligned and fit the budget')
        if stride_bytes > staging_budget_bytes:
            raise MemoryError('Complete expert bundle exceeds staging budget')
        self.stride_bytes = stride_bytes
        self._sizes=None if slot_sizes is None else tuple(slot_sizes)
        if self._sizes is not None and (not self._sizes or any(
                isinstance(s,bool) or not isinstance(s,int) or s<=0 or s%256 or s>stride_bytes
                for s in self._sizes) or sum(self._sizes)>budget_bytes):
            raise ValueError('Invalid bounded size-class slot layout')
        self.capacity = budget_bytes // stride_bytes if self._sizes is None else len(self._sizes)
        self.nbytes = self.capacity * stride_bytes if self._sizes is None else sum(self._sizes)
        self._offsets=None if self._sizes is None else tuple(np.cumsum((0,)+self._sizes[:-1],dtype=np.int64))
        self.device_id = xp.cuda.runtime.getDevice()
        self._lock = threading.RLock()
        self._free = list(range(self.capacity-1, -1, -1)) if self._sizes is None else []
        self._free_by_size=None if self._sizes is None else {s:[] for s in sorted(set(self._sizes))}
        if self._sizes is not None:
            for slot in range(self.capacity-1,-1,-1):
                self._free_by_size[self._sizes[slot]].append(slot)
        self._quarantined = []
        self._pinned = xp.cuda.alloc_pinned_memory(stride_bytes)
        self._staging_bytes = memoryview(self._pinned).nbytes
        if self._staging_bytes > staging_budget_bytes:
            raise MemoryError('Pinned allocator slab exceeds staging budget')
        self._host = np.frombuffer(self._pinned, np.uint8, count=stride_bytes)
        self._data = xp.empty(self.nbytes, dtype=xp.uint8)
        self._stream = xp.cuda.Stream(non_blocking=True)
        self.promotions = 0
        self.transfer_bytes = 0

    def reservation_size(self,total):
        if self._free_by_size is None:
            if total>self.stride_bytes: raise MemoryError('Complete expert bundle exceeds slot stride')
            return self.stride_bytes
        for size in self._free_by_size:
            if total<=size: return size
        raise MemoryError('No size class fits complete expert bundle')

    def has_free_slot(self,size):
        with self._lock:
            return bool(self._free if self._free_by_size is None else self._free_by_size.get(size,()))

    def _release(self, slot):
        with self._lock:
            (self._free if self._free_by_size is None else self._free_by_size[self._sizes[slot]]).append(slot)

    def materialize(self, weights):
        if len(weights) != 3 or not all(isinstance(w, PackedGGUFWeight) for w in weights):
            raise TypeError('A slot requires complete packed gate/up/down weights')
        gate, up, down = weights
        if (len(gate.shape) != 2 or gate.shape != up.shape or
                down.shape != (gate.shape[1], gate.shape[0])):
            raise ValueError('Invalid packed expert projection shapes')
        total = sum(w.nbytes for w in weights)
        if total > self.stride_bytes:
            raise MemoryError('Complete expert bundle exceeds slot stride')
        # Device format/shape validation happens before reserving a slot or DMA.
        from .quantized_linear import supports_packed_gpu_type
        if any(not supports_packed_gpu_type(w.quant_type) for w in weights):
            raise NotImplementedError('Unsupported packed GPU projection in slot')
        with self._lock, xp.cuda.Device(self.device_id):
            size=self.reservation_size(total)
            free=self._free if self._free_by_size is None else self._free_by_size[size]
            if not free:
                raise MemoryError('All compatible packed expert slots are leased or quarantined')
            lease = _Lease(self, free.pop())
            started = False
            try:
                offset = 0
                for weight in weights:
                    np.copyto(self._host[offset:offset+weight.nbytes], weight.data.reshape(-1))
                    offset += weight.nbytes
                pointer = int(self._data.data.ptr) + (lease.slot*self.stride_bytes
                    if self._offsets is None else int(self._offsets[lease.slot]))
                started = True
                xp.cuda.runtime.memcpyAsync(pointer, self._host.ctypes.data, total,
                    xp.cuda.runtime.memcpyHostToDevice, self._stream.ptr)
                self._stream.synchronize()  # Do not publish an incomplete bundle.
                started = False
                memory = xp.cuda.UnownedMemory(pointer, size, lease,
                                              device_id=self.device_id)
                offset, projections = 0, []
                for weight in weights:
                    array = xp.ndarray((weight.nbytes,), dtype=xp.uint8,
                        memptr=xp.cuda.MemoryPointer(memory, offset))
                    projections.append(DevicePackedGGUFWeight(array, weight.shape, weight.quant_type))
                    offset += weight.nbytes
                expert = PackedExpertFFNInference(*projections)
                self.promotions += 1
                self.transfer_bytes += total
                return expert
            except Exception:
                if started:
                    try:
                        self._stream.synchronize()
                    except Exception:
                        # Unknown DMA completion must never permit slot reuse.
                        lease.reusable = False
                        self._quarantined.append(lease)
                raise

    def statistics(self):
        with self._lock:
            free=len(self._free) if self._free_by_size is None else sum(len(v) for v in self._free_by_size.values())
            return dict(capacity=self.capacity, free_slots=free,
                        leased_slots=self.capacity-free,
                        size_classes=None if self._free_by_size is None else {
                            str(s):dict(capacity=self._sizes.count(s),free=len(v)) for s,v in self._free_by_size.items()},
                        quarantined_slots=len(self._quarantined), stride_bytes=self.stride_bytes,
                        device_reserved_bytes=self.nbytes, staging_bytes=self._staging_bytes,
                        promotions=self.promotions, transfer_bytes=self.transfer_bytes,
                        asynchronous=False)
