"""Opt-in stream-ordered native CPU handoff for a frozen device expert plan.

No Python CUDA callback and no CUDA calls from the host function. The main
stream publishes pinned activations/routes, the CPU stream runs the existing
native pool, and an event joins complete CPU rows before ordered combination.
Canonical sources and resident leases survive until the retirement fence.
Routing heat is paused in this fixed-placement experiment; aggregate counters
are collected only at an explicit synchronization/statistics boundary.
"""
import ctypes as C
import ctypes.util
import os
from functools import lru_cache

import numpy as np
from mini_llm.backend import xp


@lru_cache(maxsize=1)
def _runtime():
    name = C.util.find_library('cudart')
    if not name:
        raise RuntimeError('Native expert handoff requires the CUDA runtime library')
    library = C.CDLL(name)
    library.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    library.cudaLaunchHostFunc.restype = C.c_int
    return library


def _bind(bridge):
    bridge.mylocal_ggml_handoff_create.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_void_p, C.c_int]
    bridge.mylocal_ggml_handoff_create.restype = C.c_void_p
    bridge.mylocal_ggml_handoff_run.argtypes = [C.c_void_p]
    bridge.mylocal_ggml_handoff_run.restype = None
    bridge.mylocal_ggml_handoff_stats.argtypes = [C.c_void_p, C.c_void_p]
    bridge.mylocal_ggml_handoff_stats.restype = None
    bridge.mylocal_ggml_handoff_destroy.argtypes = [C.c_void_p]
    bridge.mylocal_ggml_handoff_destroy.restype = None


class NativeExpertHandoff:
    def __init__(self, plan):
        from .expert_residency import ExpertKey
        self.plan = plan
        self.pool = plan.executor._ggml_row_pool
        self.bridge = self.pool.executor._bridge
        _bind(self.bridge)
        self.runtime = _runtime()
        # Registration is a setup seam, never concurrent with previous native
        # callbacks sharing the pool. No packed payload is copied here.
        xp.cuda.get_current_stream().synchronize()
        registry = np.full(plan.n_experts, -1, np.int32)
        for e in range(plan.n_experts):
            if e not in plan.resident:
                key = ExpertKey(plan.executor.model_id, plan.layer, e)
                registry[e] = self.pool.register(key, plan.executor._host_weights(key))
        self.handle = self.bridge.mylocal_ggml_handoff_create(self.pool._handle,
            plan.host_input.ctypes.data, plan.host_metadata.ctypes.data,
            plan.host_output.ctypes.data, plan.k, registry.ctypes.data, plan.n_experts)
        if not self.handle:
            raise RuntimeError('Could not construct bounded native expert handoff')
        self.function = C.cast(self.bridge.mylocal_ggml_handoff_run, C.c_void_p).value
        self.stream = xp.cuda.Stream(non_blocking=True)
        self.input_ready = xp.cuda.Event(disable_timing=True)
        self.cpu_ready = xp.cuda.Event(disable_timing=True)
        self.stats = np.zeros(6, np.uint64)
        self.accounted = np.zeros(6, np.uint64)
        self.production = np.zeros(6, np.uint64)
        self.keys = tuple(ExpertKey(plan.executor.model_id, plan.layer, e) for e in plan.resident)
        plan.executor.cache.protect(self.keys)
        self.graph = None
        self.capture_enabled = os.environ.get('MINI_LLM_NATIVE_MOE_GRAPH', '0') == '1'

    def forward(self, x, routes, ids, on_cpu_submitted):
        if not self.capture_enabled:
            return self.enqueue(x, routes, ids, on_cpu_submitted)
        if self.graph is None:
            from .inference_graph import CapturedArrayInference
            self.flush_statistics()
            self.graph = CapturedArrayInference(
                lambda value, arrays: self.enqueue(value, arrays[0], arrays[1], None),
                x, (routes, ids), pool_limit_bytes=1024**2,
                copy_back_mutable=os.environ.get('MINI_LLM_GRAPH_READONLY_INPUTS','0')!='1')
            # Private warmup performs one real native batch; do not count it as
            # a production route. Capture itself does not execute the callback.
            self.bridge.mylocal_ggml_handoff_stats(self.handle, self.stats.ctypes.data)
            if self.stats[3] != self.accounted[3]:
                raise RuntimeError('Native expert graph warmup failed')
            self.accounted[:] = self.stats
        result = self.graph.forward(x, (routes, ids))
        # The model's shared stream waits on hidden's producer, not this graph's
        # CPU-result join. It can overlap the host node even after graph launch.
        if on_cpu_submitted is not None:
            on_cpu_submitted()
        return result

    def enqueue(self, x, routes, ids, on_cpu_submitted):
        p = self.plan
        stream = xp.cuda.get_current_stream()
        p.plan_kernel((1,), (32,), (ids, routes, p.residency, p.selected,
            p.routes, p.metadata, np.int32(p.n_experts), np.int32(p.k)))
        # CuPy ndarray.get forbids D2H capture even for owned pinned slabs.
        # CUDA itself supports these copies; use its explicit pointer API.
        runtime = xp.cuda.runtime
        runtime.memcpyAsync(p.host_metadata.ctypes.data, p.metadata.data.ptr,
                            p.metadata.nbytes, runtime.memcpyDeviceToHost, stream.ptr)
        runtime.memcpyAsync(p.host_input.ctypes.data, x.data.ptr,
                            x.nbytes, runtime.memcpyDeviceToHost, stream.ptr)
        self.input_ready.record(stream)
        self.stream.wait_event(self.input_ready)
        # Native function is noexcept, enters neither Python nor any CUDA API,
        # and finishes all row-pool consumers before returning to the runtime.
        error = self.runtime.cudaLaunchHostFunc(self.stream.ptr, self.function, self.handle)
        if error:
            raise RuntimeError(f'cudaLaunchHostFunc failed: {error}')
        # Even an all-resident dispatch copies a small bounded slab; selections
        # are device-resident and zero miss rows are never consumed.
        try:
            runtime.memcpyAsync(p.cpu_rows.data.ptr, p.host_output.ctypes.data,
                                p.cpu_rows.nbytes, runtime.memcpyHostToDevice, self.stream.ptr)
            if on_cpu_submitted is not None:
                on_cpu_submitted()
            p.gate_up(((p.k*p.f+3)//4,), (128,), (x,p.selected,p.hidden)+p.gate_args)
            p.down(((p.k*p.d+3)//4,), (128,),
                (p.hidden,p.selected,p.routes,p.terms)+p.down_args)
        finally:
            # Includes callback/kernel failure: don't recycle pinned storage or
            # destroy the shared CPU pool before the native consumers finish.
            self.cpu_ready.record(self.stream)
            stream.wait_event(self.cpu_ready)
            p.retired.record(stream)
            p.pending = True
        p.combine_kernel(((p.d+255)//256,), (256,), (p.terms,p.cpu_rows,
            p.selected,p.routes,p.output)+p.combine_args)
        out = p.output.copy()
        p.retired.record(stream)
        return out

    def flush_statistics(self):
        if not self.handle:
            return
        p = self.plan
        if p.pending:
            p.retired.synchronize()
        self.bridge.mylocal_ggml_handoff_stats(self.handle, self.stats.ctypes.data)
        delta = self.stats-self.accounted
        runs, misses, hits, errors, elapsed_ns, batches = (int(v) for v in delta)
        self.accounted[:] = self.stats
        self.production += delta
        if errors:
            raise RuntimeError(f'Native expert handoff encountered {errors} invalid/failed dispatches')
        e = p.executor
        e.counters['device_plan_dispatches'] += runs
        e.counters['native_handoff_dispatches'] += runs
        e.counters['cpu_submitted'] += misses
        e.counters['ggml_cpu_experts'] += misses
        e.counters['ggml_row_batches'] += batches
        e.counters['native_handoff_cpu_ns'] += elapsed_ns
        e.counters['gpu_experts'] += hits
        e.counters['decode_activation_d2h'] += runs
        e.counters['decode_results_h2d'] += runs
        cache = e.cache
        cache.counters['requests'] += misses+hits
        cache.counters['misses'] += misses
        cache.counters['hits'] += hits
        cache.counters['cpu_fallbacks'] += misses
        cache.counters['admission_deferred'] += misses
        cache._step += misses+hits

    def statistics(self):
        return dict(layer=self.plan.layer, calls=int(self.production[0]),
            cpu_experts=int(self.production[1]), gpu_experts=int(self.production[2]),
            cpu_seconds=int(self.production[4])*1e-9,
            graph=None if self.graph is None else self.graph.statistics(),
            routing_heat='paused while placement is frozen')

    def close(self):
        if self.handle:
            try:
                self.flush_statistics()
            finally:
                self.bridge.mylocal_ggml_handoff_destroy(self.handle)
                self.handle = None
                self.plan.executor.cache.unprotect(self.keys)
