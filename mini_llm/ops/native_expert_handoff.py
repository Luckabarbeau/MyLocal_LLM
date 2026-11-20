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
import time
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
    bridge.mylocal_ggml_clock_ns.argtypes=[]
    bridge.mylocal_ggml_clock_ns.restype=C.c_uint64
    bridge.mylocal_ggml_handoff_timeline_enable.argtypes=[C.c_void_p]
    bridge.mylocal_ggml_handoff_timeline_enable.restype=C.c_int
    bridge.mylocal_ggml_handoff_timeline.argtypes=[C.c_void_p,C.c_void_p,C.c_int]
    bridge.mylocal_ggml_handoff_timeline.restype=C.c_int
    bridge.mylocal_ggml_handoff_profile_enable.argtypes=[C.c_void_p]
    bridge.mylocal_ggml_handoff_profile_enable.restype=C.c_int
    bridge.mylocal_ggml_handoff_profile.argtypes=[C.c_void_p,C.c_void_p,C.c_int]
    bridge.mylocal_ggml_handoff_profile.restype=C.c_int
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
        self.resident_host_registered=0
        for e in range(plan.n_experts):
            if e not in plan.resident:
                key = ExpertKey(plan.executor.model_id, plan.layer, e)
                registry[e] = self.pool.register(key, plan.executor.cpu_source_weights(key))
                source=plan.executor._resident_source
                self.resident_host_registered+=int(source is not None and key in source.bundles)
        self.handle = self.bridge.mylocal_ggml_handoff_create(self.pool._handle,
            plan.host_input.ctypes.data, plan.host_metadata.ctypes.data,
            plan.host_output.ctypes.data, plan.k, registry.ctypes.data, plan.n_experts)
        if not self.handle:
            raise RuntimeError('Could not construct bounded native expert handoff')
        self.route_heat=None
        self.profile_warmup=None
        if os.environ.get('MINI_LLM_NATIVE_MOE_ROUTE_PROFILE','0')=='1':
            try:
                self.pool.executor._check(self.bridge.mylocal_ggml_handoff_profile_enable(self.handle))
                self.route_heat=np.zeros(plan.n_experts,np.uint64)
                self.profile_warmup=np.zeros_like(self.route_heat)
            except Exception:
                self.bridge.mylocal_ggml_handoff_destroy(self.handle); self.handle=None
                raise
        self.timeline=None
        self.timeline_warmup_ordinals=[]
        self.clock_sample=None
        if os.environ.get('MINI_LLM_NATIVE_MOE_TIMELINE','0')=='1':
            try:
                if os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_LIMIT','0')!='0' or self.pool.scheduler!='tasks':
                    raise ValueError('Native timeline requires unstaged task-scheduled handoffs')
                self.pool.executor._check(self.bridge.mylocal_ggml_handoff_timeline_enable(self.handle))
                before=time.perf_counter_ns()
                native=int(self.bridge.mylocal_ggml_clock_ns())
                after=time.perf_counter_ns()
                if not before<=native<=after:
                    raise RuntimeError('Native/Python monotonic clocks do not share an epoch')
                self.clock_sample=(before,native,after)
                self.timeline=np.empty((2048,9),np.uint64)
            except Exception:
                self.bridge.mylocal_ggml_handoff_destroy(self.handle); self.handle=None
                raise
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
        self.graph_branch = None
        self.capture_enabled = os.environ.get('MINI_LLM_NATIVE_MOE_GRAPH', '0') == '1'
        self.service=None
        self.service_flags=None
        self.stage=None
        self.kernel_publish=os.environ.get('MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH','0')=='1'
        if self.kernel_publish and os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE','0')!='1':
            plan.executor.cache.unprotect(self.keys)
            self.bridge.mylocal_ggml_handoff_destroy(self.handle)
            self.handle=None
            raise ValueError('Kernel input publication requires the native service')
        if os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE','0')=='1':
            try:
                if self.pool._native_service is None:
                    from .native_expert_service import NativeExpertService
                    self.pool._native_service=NativeExpertService(self.pool)
                self.service=self.pool._native_service
                from .native_expert_service import mapped_output
                self.service_output=mapped_output(plan.host_output,plan.output_owner)
                if self.kernel_publish:
                    self.service_input=mapped_output(plan.host_input,plan.input_owner)
                    self.service_metadata=mapped_output(plan.host_metadata,plan.input_owner)
                self.service_flags=self.service.register(self.handle)
                from .native_expert_staging import stage_limit,NativeExpertStaging
                limit=stage_limit()
                if limit and plan.formats[0] in ('IQ2_S','IQ3_S','IQ3_XXS'):
                    self.stage=NativeExpertStaging(self,limit)
            except Exception:
                if self.service is not None and self.handle in self.service.clients:
                    self.service.unregister(self.handle)
                plan.executor.cache.unprotect(self.keys)
                self.bridge.mylocal_ggml_handoff_destroy(self.handle)
                self.handle=None
                raise

    def forward(self, x, routes, ids, on_cpu_submitted, *, gpu_branch=None):
        if gpu_branch is not None and (not self.capture_enabled or self.service is None or
                on_cpu_submitted is not None):
            raise ValueError('GPU branch requires captured native service and no host callback')
        if self.graph is not None and self.graph_branch is not gpu_branch:
            raise ValueError('A live native graph cannot change its GPU branch owner')
        if not self.capture_enabled:
            return self.enqueue(x, routes, ids, on_cpu_submitted)
        if self.graph is None:
            from .inference_graph import CapturedArrayInference
            self.flush_statistics()
            self.graph_branch = gpu_branch
            self.graph = CapturedArrayInference(
                lambda value, arrays: self.enqueue(value, arrays[0], arrays[1], None, gpu_branch=gpu_branch),
                x, (routes, ids), pool_limit_bytes=1024**2,
                copy_back_mutable=os.environ.get('MINI_LLM_GRAPH_READONLY_INPUTS','0')!='1')
            # Private warmup performs one real native batch; do not count it as
            # a production route. Capture itself does not execute the callback.
            self.bridge.mylocal_ggml_handoff_stats(self.handle, self.stats.ctypes.data)
            if self.stats[3] != self.accounted[3]:
                raise RuntimeError('Native expert graph warmup failed')
            self.accounted[:] = self.stats
            if self.timeline is not None:
                self.timeline_warmup_ordinals.append(int(self.stats[0]))
            if self.stage is not None:
                self.stage.collect(warmup=True)
            if self.route_heat is not None:
                self.profile_warmup[:]=self.read_profile()
            if self.service is not None:
                self.service.check()
        result = self.graph.forward(x, (routes, ids))
        # The model's shared stream waits on hidden's producer, not this graph's
        # CPU-result join. It can overlap the host node even after graph launch.
        if on_cpu_submitted is not None:
            on_cpu_submitted()
        return result

    def enqueue(self, x, routes, ids, on_cpu_submitted, *, gpu_branch=None):
        p = self.plan
        stream = xp.cuda.get_current_stream()
        p.plan_kernel((1,), (32,), (ids, routes, p.residency, p.selected,
            p.routes, p.metadata, np.int32(p.n_experts), np.int32(p.k)))
        # CuPy ndarray.get forbids D2H capture even for owned pinned slabs.
        # CUDA itself supports these copies; use its explicit pointer API.
        runtime = xp.cuda.runtime
        if not self.kernel_publish:
            runtime.memcpyAsync(p.host_metadata.ctypes.data, p.metadata.data.ptr,
                                p.metadata.nbytes, runtime.memcpyDeviceToHost, stream.ptr)
            runtime.memcpyAsync(p.host_input.ctypes.data, x.data.ptr,
                                x.nbytes, runtime.memcpyDeviceToHost, stream.ptr)
        branch_output = None
        if self.service is not None:
            if self.kernel_publish:
                self.service.publish_inputs((1,),(256,),
                    (x,p.metadata,self.service_input,self.service_metadata,self.service_flags.device,
                     self.service.failure.device,np.int32(p.d),np.int32(p.k)))
            else:
                self.service.publish((1,),(32,),(self.service_flags.device,self.service.failure.device))
            try:
                if gpu_branch is not None:
                    branch_output = gpu_branch.enqueue_gpu_branch(x)
                if on_cpu_submitted is not None:
                    on_cpu_submitted()
                p.launch_resident(x)
                if self.stage is not None:
                    self.stage.enqueue(x)
            finally:
                self.service.wait((1,),(32,),
                    (self.service_flags.device,self.service.failure.device,self.service.timeout_ns,self.service.device_failure))
                self.service.copy_rows(((p.cpu_rows.size+255)//256,),(256,),
                    (self.service_output,p.cpu_rows,self.service.device_failure,np.int32(p.cpu_rows.size)))
                p.retired.record(stream)
                p.pending=True
        else:
            self._enqueue_callback(x,on_cpu_submitted,stream,runtime)
        if self.stage is not None:
            self.stage.combine_output()
        else:
            p.combine_kernel(((p.d+255)//256,), (256,), (p.terms,p.cpu_rows,
                p.selected,p.routes,p.output)+p.combine_args)
        if self.service is not None:
            self.service.poison(((p.d+255)//256,),(256,),
                (p.output,self.service.device_failure,np.int32(p.d)))
        out = (p.output.copy() if gpu_branch is None else
               gpu_branch.combine_gpu_branch(p.output,branch_output))
        p.retired.record(stream)
        return out

    def _enqueue_callback(self,x,on_cpu_submitted,stream,runtime):
        p=self.plan
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
            p.launch_resident(x)
        finally:
            # Includes callback/kernel failure: don't recycle pinned storage or
            # destroy the shared CPU pool before the native consumers finish.
            self.cpu_ready.record(self.stream)
            stream.wait_event(self.cpu_ready)
            p.retired.record(stream)
            p.pending = True

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
        staged,transfer_bytes,transfer_ns=self.stage.collect() if self.stage is not None else (0,0,0)
        if self.stage is not None:
            e.counters['native_staged_experts']=e.counters.get('native_staged_experts',0)+staged
            e.counters['native_stage_transfer_bytes']=e.counters.get('native_stage_transfer_bytes',0)+transfer_bytes
            e.counters['native_stage_host_fill_ns']=e.counters.get('native_stage_host_fill_ns',0)+transfer_ns
        e.counters['device_plan_dispatches'] += runs
        e.counters['native_handoff_dispatches'] += runs
        e.counters['cpu_submitted'] += misses-staged
        e.counters['ggml_cpu_experts'] += misses-staged
        e.counters['ggml_row_batches'] += batches
        e.counters['native_handoff_cpu_ns'] += elapsed_ns
        e.counters['gpu_experts'] += hits+staged
        if p.q8 is not None:
            e.counters['q8_gpu_experts'] += hits+staged
        e.counters['decode_activation_d2h'] += runs
        e.counters['decode_results_h2d'] += runs
        cache = e.cache
        cache.counters['requests'] += misses+hits
        cache.counters['misses'] += misses
        cache.counters['hits'] += hits
        cache.counters['cpu_fallbacks'] += misses
        cache.counters['admission_deferred'] += misses
        cache._step += misses+hits

    def read_profile(self):
        # Caller first fences this handoff's retirement event. Do not snapshot a
        # timed-out producer while its CPU job may still be active.
        if self.service is not None and int(self.service.failure.host[0]):
            raise RuntimeError('Cannot profile a failed native service')
        self.pool.executor._check(self.bridge.mylocal_ggml_handoff_profile(
            self.handle,self.route_heat.ctypes.data,len(self.route_heat)))
        return self.route_heat

    def statistics(self):
        profile=None if self.route_heat is None else (self.read_profile()-self.profile_warmup).tolist()
        return dict(layer=self.plan.layer, calls=int(self.production[0]),
                    kernel_input_publication=self.kernel_publish,
                    captured_gpu_branch=self.graph_branch is not None,
                    route_heat=profile,
                    profile_resident_ids=None if profile is None else sorted(self.plan.resident),
                    expert_packed_bytes=self.plan.expert_bytes,
                    staged_experts=0 if self.stage is None else int(self.stage.production[0]),
                    staging_copy_bytes=0 if self.stage is None else int(self.stage.production[1]),
                    native_elapsed_scope='CPU + host staging join' if self.stage is not None else 'CPU',
                    staging_source_mode=None if self.stage is None else self.stage.source_mode,
                            resident_host_registered_experts=self.resident_host_registered,
                    cpu_handoff='native_mapped_service' if self.service is not None else 'cuda_host_function',
                    gpu_arithmetic='q8_1' if self.plan.q8 is not None else 'fp32',
            cpu_experts=int(self.production[1])-(0 if self.stage is None else int(self.stage.production[0])),
            gpu_experts=int(self.production[2])+(0 if self.stage is None else int(self.stage.production[0])),
            cpu_seconds=int(self.production[4])*1e-9,
            graph=None if self.graph is None else self.graph.statistics(),
            routing_heat='paused while placement is frozen')

    def timeline_snapshot(self):
        if self.timeline is None: return None
        if self.plan.pending: self.plan.retired.synchronize()
        if self.service is not None and int(self.service.failure.host[0]):
            raise RuntimeError('Cannot snapshot a failed native service')
        count=self.bridge.mylocal_ggml_handoff_timeline(self.handle,self.timeline.ctypes.data,len(self.timeline))
        if count<0: raise RuntimeError('Native timeline snapshot failed')
        return dict(layer=self.plan.layer,clock_sample=self.clock_sample,
            private_warmup_ordinals=self.timeline_warmup_ordinals,
            columns=('ordinal','start_monotonic_ns','end_monotonic_ns','cpu_experts','error',
                     'gate_up_start_ns','hidden_quant_start_ns','down_start_ns','down_end_ns'),
            rows=self.timeline[:count].tolist(),capacity=len(self.timeline),
            caveat='Includes private graph warmup; align timestamps, not occurrences')

    def close(self):
        if self.handle:
            try:
                self.flush_statistics()
            finally:
                if self.graph is not None:
                    self.graph.close()
                    self.graph=None
                self.graph_branch=None
                if self.service is not None:
                    self.service.unregister(self.handle)
                if self.stage is not None:
                    self.stage.close()
                self.bridge.mylocal_ggml_handoff_destroy(self.handle)
                self.handle = None
                self.plan.executor.cache.unprotect(self.keys)
