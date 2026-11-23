"""Long-chunk native expert caller, pinned to native_prefill_cuda.REVISION.

Caller CPU -> independent RAM stagers -> independent CUDA copy issuer -> GPU.
The chunk-wide walk includes unrouted experts, in layer/expert order. This is
only the source's long streamed native path: CPU/MMQ/short chunks are NOT
implemented. The default 128 MiB / 128-slot ring is an explicit bounded-memory
policy, NOT the source auto memory planner (which may choose 512 slots).

One instance belongs to one serialized model caller on CUDA device 0. Call
begin_chunk(model, T, topk=None) before attention, run(block.moe.experts, x,
route_weights, ids) in block order, then end_chunk() in a finally clause.
run returns new FP32 [T,D] routed output; shared-expert addition belongs to the
parent. IDs must be valid [0,E), inputs/scales finite (native kernel contract).
No IDs, route weights, counts, or intermediate activations are read on CPU.
end_chunk fences once, retires threads/scopes, and preserves reusable buffers.
close also releases buffers; neither method leaves idle background threads.

Cache admission must already be frozen with set_admission_budget(0). All layer
IDs are dispatch-protected before pointers are taken and until chunk retirement.
The parent must retain its model call lock throughout; changing placements or
using another stream during a chunk is unsupported and checked at run seams.
"""
from collections import Counter
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import os
import threading

import numpy as np

from . import native_prefill_cuda as native


_COMBINE = r'''
extern "C" __global__ void rank_combine(const float* dm, const int* slot,
    const float* weights, float* out, int T, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    int t = i / D, d = i % D;
    float v = 0.0f;
    for (int r = 0; r < K; ++r) {
        int j = t * K + r;
        v = fmaf(weights[j], dm[slot[j] * D + d], v);
    }
    out[i] = v;
}
'''


@dataclass
class _Layer:
    experts: object
    weights: list
    sequences: list
    gu_kind: str
    down_kind: str
    ptrs: object = None
    host_ptrs: object = None


class NativePrefillPipeline:
    """Persistent per-model owner; configuration is explicit, never auto-sized.

    ring_slots is an upper bound, further capped by device_ring_bytes / largest
    expert blob. It may be 2 for wraparound tests; production defaults exceed
    eight slots, but a byte budget too small for two experts is rejected.
    Counters are cumulative actual weight bytes/reads, not routing estimates.
    """

    def __init__(self, *, device_ring_bytes=128 * 1024**2, ring_slots=128,
                 host_depth=16, workers=None, stream_min=3072):
        workers = max(2, min(4, (os.cpu_count() or 1) // 4)) if workers is None else workers
        for name, value in (('device_ring_bytes', device_ring_bytes),
                            ('ring_slots', ring_slots), ('host_depth', host_depth),
                            ('workers', workers), ('stream_min', stream_min)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if ring_slots < 2:
            raise ValueError('ring_slots must be at least two')
        self.device_ring_bytes, self.ring_slots = device_ring_bytes, ring_slots
        self.host_depth, self.workers, self.stream_min = host_depth, workers, stream_min
        self.counters = Counter()
        self._model = None
        self._active = self._closed = False
        self._buffers_key = self._workspace_key = None
        self._device = self._host_owner = self._host = None
        self._workspace = None
        self._copy = self._combine = None
        self._threads = []
        self._cv = threading.Condition()

    def _check_parent(self):
        if threading.get_ident() != self._parent:
            raise RuntimeError('pipeline must be called by its begin_chunk thread')
        if self._cp.cuda.runtime.getDevice() != 0:
            raise RuntimeError('native pipeline requires CUDA device 0')
        if self._cp.cuda.get_current_stream().ptr != self._compute.ptr:
            raise RuntimeError('chunk must stay on its original compute stream')

    @staticmethod
    def _validate_weights(weights, experts, cp, resident):
        if len(weights) != 3:
            raise ValueError('three packed projections required')
        d, f = experts.d_model, experts.d_ff
        for w, shape in zip(weights, ((f, d), (f, d), (d, f))):
            if tuple(w.shape) != shape or w.quant_type not in native.FORMATS:
                raise ValueError('unsupported packed weight shape/format')
            _, qk, qb = native.FORMATS[w.quant_type]
            a = w.data
            cls = cp.ndarray if resident else np.ndarray
            if not isinstance(a, cls) or a.dtype != np.uint8:
                raise TypeError('packed data must be contiguous uint8 host/device bytes')
            if (not a.flags.c_contiguous or shape[1] % qk or
                    a.nbytes != shape[0] * (shape[1] // qk) * qb):
                raise ValueError('invalid packed byte length/alignment')
            if resident and a.device.id != 0:
                raise ValueError('resident packed bytes must be on CUDA device 0')
        kinds = tuple(w.quant_type for w in weights)
        if kinds[0] != kinds[1] or kinds[0] not in native.GU_FORMATS or kinds[2] not in native.DOWN_FORMATS:
            raise ValueError('unsupported gate/up or down formats')
        return kinds[0], kinds[2]

    def begin_chunk(self, model, token_count, topk=None):
        """Snapshot/protect all layers and start staging before the layer loop.

        model.blocks[*].moe.experts supplies d_model/d_ff/n_experts/layer_id and
        expert_executor; topk may instead be derived from every moe.spec.top_k.
        Host sources are resolved ONLY here, on the calling thread, and retained
        through retirement. Returns self. A failed begin retires partial work.
        """
        if self._closed or self._active:
            raise RuntimeError('pipeline is closed or already has an active chunk')
        if self._model is not None and self._model is not model:
            raise RuntimeError('pipeline belongs to a different model')
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < self.stream_min:
            raise ValueError(f'long native path requires token_count >= {self.stream_min}; CPU/MMQ is not implemented')
        moes = [b.moe for b in model.blocks]
        if not moes:
            raise ValueError('model has no MoE layers')
        if topk is None:
            ks = {m.spec.top_k for m in moes}
            if len(ks) != 1:
                raise ValueError('top_k must agree across layers')
            topk = ks.pop()
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ValueError('topk must be a positive integer')
        import cupy as cp
        from .expert_residency import ExpertKey
        if cp.cuda.runtime.getDevice() != 0:
            raise RuntimeError('native pipeline requires CUDA device 0')
        self._cp, self._parent = cp, threading.get_ident()
        self._compute = cp.cuda.get_current_stream()
        self._scopes = ExitStack()
        self._layers, self._seq, self._snapshots = [], [], []
        self._threads = []
        self._stop, self._error = False, None
        self._issued = self._consumed = self._next_job = self._next_layer = 0
        self._active = True
        self._T, self._K = token_count, topk
        try:
            executors = {}
            for m in moes:
                ex = m.experts
                if ex.d_model <= 0 or ex.d_model % 256 or ex.d_ff <= 0 or ex.d_ff % 128:
                    raise ValueError('source geometry requires D%256=0 and F%128=0')
                if not 1 <= topk <= ex.n_experts <= 1024:
                    raise ValueError('require 1 <= topk <= experts <= 1024')
                executor = ex.expert_executor
                if executor.cache._admission_limit != 0:
                    raise ValueError('native pipeline requires frozen admission')
                # Enter every scope before reading ANY resident pointer.
                self._scopes.enter_context(executor.dispatch_scope(ex.layer_id, range(ex.n_experts)))
                executors[id(executor)] = executor
            for executor in executors.values():
                self._snapshots.append((executor, dict(executor.cache.resident)))
            largest = 0
            for m in moes:
                ex, bundles, sequences, formats = m.experts, [], [], set()
                executor = ex.expert_executor
                snapshot = next(s for e, s in self._snapshots if e is executor)
                for e in range(ex.n_experts):
                    key = ExpertKey(executor.model_id, ex.layer_id, e)
                    entry = snapshot.get(key)
                    if entry is not None:
                        handle = entry[0]
                        weights = (handle.gate, handle.up, handle.down)
                        seq = None
                    else:
                        weights = tuple(executor.cpu_source_weights(key))
                        seq = len(self._seq)
                        self._seq.append(weights)
                    formats.add(self._validate_weights(weights, ex, cp, entry is not None))
                    bundles.append(weights)
                    sequences.append(seq)
                    largest = max(largest, sum(w.data.nbytes for w in weights))
                if len(formats) != 1:
                    raise ValueError('each layer requires homogeneous packed formats')
                gu, down = formats.pop()
                self._layers.append(_Layer(ex, bundles, sequences, gu, down))
            # Alignment is ring-slot alignment only, never a change to GGUF blobs.
            stride = (largest + 255) // 256 * 256
            capacity = min(self.ring_slots, self.device_ring_bytes // stride)
            if capacity < 2:
                raise ValueError('device_ring_bytes cannot fit two expert slots')
            self.capacity, self.slot_bytes = capacity, stride
            key = (capacity, stride, self.host_depth)
            if key != self._buffers_key:
                self._device = cp.empty((capacity, stride), cp.uint8)
                self._host_owner = cp.cuda.alloc_pinned_memory(self.host_depth * stride)
                self._host = np.frombuffer(self._host_owner, np.uint8,
                                    count=self.host_depth * stride).reshape(self.host_depth, stride)
                self._copied = [cp.cuda.Event(disable_timing=True) for _ in range(capacity)]
                self._used = [cp.cuda.Event(disable_timing=True) for _ in range(capacity)]
                self._dma = [cp.cuda.Event(disable_timing=True) for _ in range(self.host_depth)]
                self._buffers_key = key
            if self._copy is None:
                self._copy = cp.cuda.Stream(non_blocking=True)
                self._combine = cp.RawKernel(_COMBINE, 'rank_combine', options=('--std=c++17',))
            self._ready = [-1] * self.host_depth
            for layer in self._layers:
                table = np.empty((layer.experts.n_experts, 3), np.int64)
                for e, (weights, seq) in enumerate(zip(layer.weights, layer.sequences)):
                    offset = 0
                    for j, w in enumerate(weights):
                        table[e, j] = (w.data.data.ptr if seq is None else
                            self._device.data.ptr + (seq % capacity) * stride + offset)
                        offset += w.data.nbytes
                layer.host_ptrs = table  # Retain async upload owner until the fence.
                layer.ptrs = cp.asarray(table)
            self._copy.wait_event(self._compute.record())
            self._model = model
            self._threads = [threading.Thread(target=self._stage, name=f'native-stage-{i}')
                             for i in range(self.workers)]
            self._threads.append(threading.Thread(target=self._issue, name='native-copy-issuer'))
            for thread in self._threads:
                thread.start()
            return self
        except BaseException:
            self.end_chunk()
            raise

    def _fail(self, error):
        with self._cv:
            if self._error is None:
                self._error = error
            self._stop = True
            self._cv.notify_all()

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError('native prefill background worker failed') from self._error

    def _stage_one(self, idx):
        """RAM-only seam for synthetic delayed/failing worker tests."""
        offset = 0
        dst = self._host[idx % self.host_depth].ctypes.data
        for w in self._seq[idx]:
            ctypes.memmove(dst + offset, w.data.ctypes.data, w.data.nbytes)
            offset += w.data.nbytes

    def _stage(self):
        try:
            with self._cp.cuda.Device(0):
                while True:
                    with self._cv:
                        if self._stop or self._next_job >= len(self._seq):
                            return
                        idx = self._next_job
                        self._next_job += 1
                        self._cv.wait_for(lambda: self._stop or self._issued > idx - self.host_depth)
                        if self._stop:
                            return
                    # Host memory cannot be overwritten until its prior DMA ends.
                    if idx >= self.host_depth:
                        self._dma[idx % self.host_depth].synchronize()
                    self._stage_one(idx)
                    with self._cv:
                        self._ready[idx % self.host_depth] = idx
                        self._cv.notify_all()
        except BaseException as exc:
            self._fail(exc)

    def _issue(self):
        try:
            cp = self._cp
            with cp.cuda.Device(0), self._copy:
                for idx, weights in enumerate(self._seq):
                    sl, hs = idx % self.capacity, idx % self.host_depth
                    with self._cv:
                        self._cv.wait_for(lambda: self._stop or (
                            idx < self._consumed + self.capacity and self._ready[hs] == idx))
                        if self._stop:
                            return
                    if idx >= self.capacity:
                        self._copy.wait_event(self._used[sl])
                    size = sum(w.data.nbytes for w in weights)
                    cp.cuda.runtime.memcpyAsync(self._device.data.ptr + sl * self.slot_bytes,
                        self._host[hs].ctypes.data, size,
                        cp.cuda.runtime.memcpyHostToDevice, self._copy.ptr)
                    self._dma[hs].record(self._copy)
                    self._copied[sl].record(self._copy)
                    with self._cv:
                        self.counters['weight_bytes_streamed'] += size
                        self.counters['experts_streamed'] += 1
                        self._issued = idx + 1
                        self._cv.notify_all()
        except BaseException as exc:
            self._fail(exc)

    def _check_snapshot(self):
        for executor, snapshot in self._snapshots:
            current = executor.cache.resident
            if (executor.cache._admission_limit != 0 or current.keys() != snapshot.keys() or
                    any(current[k][0] is not v[0] for k, v in snapshot.items())):
                raise RuntimeError('frozen placement snapshot changed during chunk')

    def _get_workspace(self, ex):
        key = (self._T, self._K, ex.n_experts, ex.d_model, ex.d_ff)
        if self._workspace_key != key:
            cp, r = self._cp, self._T * self._K
            self._workspace = (native.allocate_group(r, ex.n_experts, self._K),
                cp.empty((self._T, ex.d_model // 64, 80), cp.uint8),
                cp.empty((r, ex.d_ff // 64, 80), cp.uint8),
                cp.empty((r, ex.d_model), cp.float32))
            self._workspace_key = key
        return self._workspace

    def run(self, experts, x, routes, ids):
        """Quantize once, group on GPU, execute ordered waves, rank-order fmaf.

        x: contiguous CUDA0 FP32 [T,D]; routes: FP32 [T,K]; ids: INT32 [T,K].
        All metadata uses ALL T*K routes and ALL E experts, never wave counts.
        Output owns its allocation (not overwritten by subsequent run calls).
        """
        if not self._active:
            raise RuntimeError('begin_chunk is required')
        self._check_parent()
        self._raise_error()
        self._check_snapshot()
        if self._next_layer >= len(self._layers) or self._layers[self._next_layer].experts is not experts:
            raise ValueError('run must follow the chunk layer order exactly')
        native._array(x, np.float32, 'x', (self._T, experts.d_model))
        native._array(routes, np.float32, 'routes', (self._T, self._K))
        native._array(ids, np.int32, 'ids', (self._T, self._K))
        layer = self._layers[self._next_layer]
        cp, stream = self._cp, self._compute
        tb, xa, H, Dm = self._get_workspace(experts)
        native.quantize(x, out=xa)
        native.group(ids, experts.n_experts, out=tb)
        e0 = 0
        while e0 < experts.n_experts:
            e1, streamed = e0, []
            while e1 < experts.n_experts and e1 - e0 < 128:
                seq = layer.sequences[e1]
                if seq is not None:
                    if len(streamed) == max(1, self.capacity // 3):
                        break
                    streamed.append(seq)
                e1 += 1
            if streamed:
                with self._cv:
                    self._cv.wait_for(lambda: self._error is not None or self._issued > streamed[-1])
                    self._raise_error()
                stream.wait_event(self._copied[streamed[-1] % self.capacity])
            native.execute_batch(e0, e1, self._T * self._K, experts.n_experts,
                layer.ptrs, xa, tb, embd=experts.d_model, ff=experts.d_ff,
                gu_kind=layer.gu_kind, down_kind=layer.down_kind, H=H, Dm=Dm)
            for seq in streamed:
                self._used[seq % self.capacity].record(stream)
            if streamed:
                with self._cv:
                    self._consumed = streamed[-1] + 1
                    self._cv.notify_all()
            for e in range(e0, e1):
                if layer.sequences[e] is None:
                    self.counters['resident_reads'] += 1
                    self.counters['resident_weight_bytes'] += sum(w.data.nbytes for w in layer.weights[e])
            self.counters['batches'] += 1
            e0 = e1
        out = cp.empty((self._T, experts.d_model), cp.float32)
        self._combine(((out.size + 255) // 256,), (256,),
            (Dm, tb.slot, routes, out, np.int32(self._T), np.int32(experts.d_model), np.int32(self._K)))
        self._next_layer += 1
        self.counters['layers'] += 1
        return out

    def end_chunk(self):
        """Stop/join staging and issuance, fence once, release snapshot scopes.

        Legal after a partial chunk or failed run. Background failures propagate
        after cleanup; the owner remains reusable. No per-layer host GPU fences.
        """
        if not self._active:
            return
        self._check_parent()
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        try:
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join()
            if self._copy is not None:
                self._compute.wait_event(self._copy.record())
            self._compute.synchronize()
        finally:
            self._scopes.close()
            self._threads = []
            self._layers, self._seq, self._snapshots = [], [], []
            self._active = False
        self._raise_error()

    def close(self):
        """Retire any chunk and release persistent allocations, even on failure."""
        try:
            self.end_chunk()
        finally:
            self._device = self._host = self._host_owner = self._workspace = None
            self._copy = self._combine = None
            self._copied = self._used = self._dma = None
            self._closed = True
