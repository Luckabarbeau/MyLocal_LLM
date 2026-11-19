"""Frozen-residency decode plan: persistent metadata/scratch, no packed copies.

Canonical ownership stays in the executor. GPU kernels select resident pointers;
complete CPU misses use the existing native row pool. Not a placement mover or a
whole-token CUDA graph. Serial single-stream dispatch only, at most64 plans.
"""
from functools import lru_cache

import numpy as np
from mini_llm.backend import xp
from .quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from .quantized_linear import PackedExpertFFNInference
from . import iq_packed_gpu as iq

_SOURCE = r'''
extern "C" __global__ void plan_routes(const long long* ids, const float* routes,
    const unsigned long long* resident, unsigned long long* selected,
    float* sorted_routes, int* metadata, int n, int k) {
    int slot = threadIdx.x;
    if (!slot) metadata[k] = 0;
    __syncthreads();
    if (slot >= k) return;
    long long raw_id = ids[slot];
    int id = int(raw_id), rank = 0;
    bool valid = raw_id >= 0 && raw_id < n;
    for (int j = 0; j < k; ++j) {
        rank += ids[j] < raw_id || (ids[j] == raw_id && j < slot);
        if (j != slot && ids[j] == raw_id) valid = false;
    }
    if (!valid) atomicExch(metadata+k, -1);
    metadata[rank] = id;
    sorted_routes[rank] = routes[slot];
    for (int p = 0; p < 3; ++p)
        selected[rank*3+p] = valid ? resident[id*3+p] : 0;
}
extern "C" __global__ void combine_planned(const float* gpu, const float* cpu,
    const unsigned long long* selected, const float* routes, float* out, int d, int k) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= d) return;
    float result = 0.0f;
    int miss = 0;
    for (int e = 0; e < k; ++e) {
        float term;
        if (selected[e*3]) term = gpu[e*d+col];
        else { term = cpu[miss*d+col]*routes[e]; ++miss; }
        result += term;
    }
    out[col] = result;
}
'''


@lru_cache(maxsize=8)
def _module(device):
    with xp.cuda.Device(device):
        return xp.RawModule(code=_SOURCE, options=('--std=c++11', '--fmad=false'))


def _unexpected_promotion():
    raise RuntimeError('Device plan requires frozen admission')


class DeviceExpertPlan:
    def __init__(self, executor, layer, n_experts, k, source):
        self.executor, self.layer, self.n_experts, self.k = executor, layer, n_experts, k
        self.d, self.f = executor.d_model, executor.d_ff
        self.device = xp.cuda.runtime.getDevice()
        self.stream_ptr = xp.cuda.get_current_stream().ptr
        self.formats = tuple(w.quant_type for w in source)
        self.expert_bytes = sum(w.nbytes for w in source)
        expected = ((self.f, self.d), (self.f, self.d), (self.d, self.f))
        if (not all(isinstance(w, PackedGGUFWeight) and iq.supports_type(w.quant_type)
                    for w in source) or tuple(w.shape for w in source) != expected):
            raise NotImplementedError('Device plan requires supported packed IQ projections')
        self.resident = {}
        pointers = np.zeros((n_experts, 3), np.uint64)
        for key, (expert, _) in executor.cache.resident.items():
            if key.model_id != executor.model_id or key.layer_id != layer:
                continue
            if (not isinstance(expert, PackedExpertFFNInference) or not expert.fuse_gate_up or
                    tuple(w.quant_type for w in (expert.gate, expert.up, expert.down)) != self.formats or
                    any(w.data.device.id != self.device or w.shape != shape for w, shape in
                        zip((expert.gate, expert.up, expert.down), expected))):
                raise NotImplementedError('Device plan requires uniform projection formats per layer')
            self.resident[key.expert_id] = expert
            pointers[key.expert_id] = [w.data.data.ptr for w in (expert.gate, expert.up, expert.down)]
        self.residency = xp.asarray(pointers)
        self.selected = xp.empty((k, 3), xp.uint64)
        self.routes = xp.empty(k, xp.float32)
        self.metadata = xp.empty(k+1, xp.int32)
        self.hidden = xp.empty((k, self.f), xp.float32)
        self.terms = xp.empty((k, self.d), xp.float32)
        self.cpu_rows = xp.empty((k, self.d), xp.float32)
        self.output = xp.empty((1, self.d), xp.float32)
        self.input_owner = xp.cuda.alloc_pinned_memory((self.d+k+1)*4)
        self.output_owner = xp.cuda.alloc_pinned_memory(k*self.d*4)
        self.host_input = np.frombuffer(self.input_owner, np.float32, self.d).reshape(1, self.d)
        self.host_metadata = np.frombuffer(self.input_owner, np.int32, k+1, offset=self.d*4)
        self.host_output = np.frombuffer(self.output_owner, np.float32, k*self.d).reshape(k, self.d)
        self.ready = xp.cuda.Event(disable_timing=True)
        self.retired = xp.cuda.Event(disable_timing=True)
        self.pending = False
        self.fingerprint = self.cache_fingerprint(executor)
        self.sources = {}
        self.registered_ids = np.empty(k,np.int32)
        module = _module(self.device)
        self.plan_kernel = module.get_function('plan_routes')
        self.combine_kernel = module.get_function('combine_planned')
        from .iq_grouped_warp_gpu import module as warp_module
        module = warp_module(self.device, iq._codebook_storage())
        self.gate_up = module.get_function('iq_warp_gate_up')
        self.down = module.get_function('iq_warp_down_weighted')
        self.layout = tuple((iq._FORMAT_ID[fmt], *BLOCK_FORMATS[fmt]) for fmt in self.formats)
        self.gate_args = tuple(np.int32(v) for v in
            (self.d,self.f,3,*self.layout[0],*self.layout[1],256,k))
        self.down_args = tuple(np.int32(v) for v in (self.d,self.f,3)) + (np.int64(1),) + tuple(
            np.int32(v) for v in (*self.layout[2],256,k))
        self.combine_args = np.int32(self.d),np.int32(k)

    @staticmethod
    def cache_fingerprint(executor):
        cache = executor.cache
        return (cache.counters['promotions'], cache.counters['evictions'], cache.device_budget_bytes)

    def close(self):
        if self.pending:
            self.retired.synchronize()
            self.pending = False

    def forward(self, x, routes, ids, *, on_cpu_submitted=None):
        from .expert_residency import ExpertKey
        stream = xp.cuda.get_current_stream()
        if stream.ptr != self.stream_ptr or x.device.id != self.device:
            return None
        if stream.is_capturing():
            raise RuntimeError('Device expert CPU handoff is an uncaptured dispatch seam')
        if self.pending and not self.retired.done:
            self.retired.synchronize()  # Protect pinned DMA slabs on unusually early reuse.
        self.pending = False
        d, f, k = self.d, self.f, self.k
        self.plan_kernel((1,), (32,), (ids, routes, self.residency, self.selected,
            self.routes, self.metadata, np.int32(self.n_experts), np.int32(k)))
        self.metadata.get(out=self.host_metadata, stream=stream, blocking=False)
        x.get(out=self.host_input, stream=stream, blocking=False)
        self.ready.record(stream)
        self.ready.synchronize()  # Exactly one producer/metadata fence for CPU dispatch.
        if self.host_metadata[k] != 0:
            self.retired.record(stream)
            self.pending = True
            return None  # Repeated/invalid IDs retain the established general dispatcher.
        selected_ids = tuple(int(e) for e in self.host_metadata[:k])
        misses = [e for e in selected_ids if e not in self.resident]
        executor = self.executor
        job = None
        try:
            with executor.dispatch_scope(self.layer, selected_ids):
                for expert in selected_ids:
                    key = ExpertKey(executor.model_id, self.layer, expert)
                    executor.cache.request(key, self.expert_bytes, _unexpected_promotion)
                if misses:
                    for row,expert in enumerate(misses):
                        if expert not in self.sources:
                            key = ExpertKey(executor.model_id, self.layer, expert)
                            self.sources[expert] = executor._ggml_row_pool.register(key,executor._host_weights(key))
                        self.registered_ids[row] = self.sources[expert]
                    job = executor._ggml_row_pool._submit_registered(self.host_input,
                        self.registered_ids[:len(misses)],self.host_output[:len(misses)])
                    executor.counters['cpu_submitted'] += len(misses)
                    executor.counters['ggml_cpu_experts'] += len(misses)
                    executor.counters['ggml_row_batches'] += 1
                try:
                    if on_cpu_submitted is not None:
                        on_cpu_submitted()
                    if len(misses) < k:
                        self.gate_up(((k*f+3)//4,), (128,), (x,self.selected,self.hidden)+self.gate_args)
                        self.down(((k*d+3)//4,), (128,),
                            (self.hidden,self.selected,self.routes,self.terms)+self.down_args)
                        executor.counters['gpu_experts'] += k-len(misses)
                finally:
                    if job is not None:
                        job.result()
                if misses:
                    self.cpu_rows[:len(misses)].set(self.host_output[:len(misses)], stream=stream)
                    executor.counters['decode_results_h2d'] += 1
                self.combine_kernel(((d+255)//256,), (256,), (self.terms, self.cpu_rows,
                    self.selected, self.routes, self.output)+self.combine_args)
                out = self.output.copy()  # Fresh ownership across later replays.
                executor.counters['device_plan_dispatches'] += 1
                executor.counters['decode_activation_d2h'] += 1
                return out
        finally:
            # Includes output/metadata consumers and protects pinned buffers on failure.
            self.retired.record(stream)
            self.pending = True


def try_forward(executor, layer, x, routes, ids, n_experts, *, on_cpu_submitted=None):
    """Return None without changing routing for unsupported/flexible placement."""
    if (executor.cache._admission_limit != 0 or executor.cache.idle_eviction_requests):
        for plan in executor._device_expert_plans.values():
            plan.close()
        executor._device_expert_plans.clear()
        return None
    k = routes.size
    if (executor.execution_mode != 'mixed' or executor.routing_profiler is not None or
            executor._ggml_row_pool is None or not 1 <= k <= min(16, executor._ggml_row_pool.capacity) or
            not 1 <= n_experts <= 512 or not 1 <= executor.d_model <= 4096 or
            not 1 <= executor.d_ff <= 4096 or x.shape != (1, executor.d_model) or
            x.dtype != xp.float32 or routes.dtype != xp.float32 or ids.dtype != xp.int64 or
            ids.size != k or not all(a.flags.c_contiguous and
                a.device.id == xp.cuda.runtime.getDevice() for a in (x, routes, ids))):
        return None
    plans = executor._device_expert_plans
    fingerprint = DeviceExpertPlan.cache_fingerprint(executor)
    plan = plans.get(layer)
    if plan is not None and (plan.fingerprint != fingerprint or plan.k != k):
        plan.close()
        del plans[layer]
        plan = None
    if plan is None:
        if len(plans) >= 64:
            return None
        from .expert_residency import ExpertKey
        source = executor._host_weights(ExpertKey(executor.model_id, layer, 0))
        if not all(isinstance(w, PackedGGUFWeight) and iq.supports_type(w.quant_type) for w in source):
            return None
        try:
            plan = DeviceExpertPlan(executor, layer, n_experts, k, source)
        except NotImplementedError:
            return None
        plans[layer] = plan
    return plan.forward(x, routes, ids, on_cpu_submitted=on_cpu_submitted)
