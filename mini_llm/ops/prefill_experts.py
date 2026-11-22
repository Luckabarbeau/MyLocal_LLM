"""Layer/chunk expert dispatch: group all token rows before executing weights.

Resident experts retain the decode plan's Q8_1 row-dot and SwiGLU arithmetic.
CPU groups retain GGML's CPU activation quantizers. Optional ``stream`` runs
misses on the same GPU Q8 path using transient packed weights, like Strata's
prompt walk; that explicit policy is not bitwise CPU/GPU arithmetic equivalence.
No experts are skipped and no expert is expanded to dense weights.
"""
from functools import lru_cache
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp


_SOURCE = r'''
extern "C" __global__ void add_expert_rows(float* output, const float* values,
        const int* tokens, const float* weights, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    int row = i / width, col = i % width;
    long long destination = (long long)tokens[row] * width + col;
    output[destination] = __fadd_rn(output[destination], __fmul_rn(values[i], weights[row]));
}
'''


@lru_cache(maxsize=8)
def _kernels(device):
    with xp.cuda.Device(device):
        add = xp.RawModule(code=_SOURCE, options=('--std=c++11', '--fmad=false')).get_function('add_expert_rows')
        swiglu = xp.ElementwiseKernel('float32 gate, float32 up', 'float32 hidden',
            'hidden = (gate / (1.0f + expf(-gate))) * up;',
            'prefill_q8_plan_swiglu', options=('--fmad=false',))
    return add, swiglu


def _gpu_rows(packed_input, token_rows, weights):
    from .mixed_q8_gpu import linear_packed
    from .q8_1_gpu import quantize_q8_1
    gate, up, down = weights
    selected = xp.ascontiguousarray(packed_input[token_rows])
    g = linear_packed(selected, gate)
    u = linear_packed(selected, up)
    _, swiglu = _kernels(xp.cuda.runtime.getDevice())
    hidden = swiglu(g, u)
    return linear_packed(quantize_q8_1(hidden), down)


class _PackedStageRing:
    """Eight bounded packed slots, with independent copy/compute completion."""
    def __init__(self, max_bytes, slots=8):
        self.device = xp.empty((slots, max_bytes), xp.uint8)
        self.owner = xp.cuda.alloc_pinned_memory(slots*max_bytes)
        self.host = np.frombuffer(self.owner, np.uint8, count=slots*max_bytes).reshape(slots, max_bytes)
        self.copy_stream = xp.cuda.Stream(non_blocking=True)
        self.ready = [xp.cuda.Event(disable_timing=True) for _ in range(slots)]
        self.done = [xp.cuda.Event(disable_timing=True) for _ in range(slots)]
        self.used = [False]*slots
        self.next = 0

    def stage(self, weights, compute_stream):
        from .quantized_linear import DevicePackedGGUFWeight
        slot = self.next
        self.next = (slot+1) % len(self.used)
        if self.used[slot]:
            self.done[slot].synchronize()
        offset, handles = 0, []
        for weight in weights:
            stop = offset+weight.nbytes
            self.host[slot, offset:stop] = weight.data.reshape(-1)
            handles.append(DevicePackedGGUFWeight(self.device[slot, offset:stop], weight.shape, weight.quant_type))
            offset = stop
        self.device[slot, :offset].set(self.host[slot, :offset], stream=self.copy_stream)
        self.ready[slot].record(self.copy_stream)
        compute_stream.wait_event(self.ready[slot])
        return slot, tuple(handles)

    def consumed(self, slot, stream):
        self.done[slot].record(stream)
        self.used[slot] = True

    def fence(self):
        self.copy_stream.synchronize()


def try_prefill_experts(experts, x, routes, ids, *, mode='grouped'):
    """Return [1,T,D], or None before dispatch when the exact path is ineligible.

    One D2H activation per layer, grouped CPU jobs, and multirow packed GPU
    products replace T independent decode dispatches. Accumulation visits expert
    IDs in ascending order, preserving each token's weighted-add boundaries.
    The caller serializes the model/executor and poisons state on execution errors.
    """
    from .expert_residency import HeterogeneousExpertExecutor
    from .quantized_linear import PackedExpertFFNInference
    from .quantized_weight import PackedGGUFWeight
    from .mixed_q8_gpu import FORMATS
    from .q8_1_gpu import quantize_q8_1

    executor = getattr(experts, 'expert_executor', None)
    if mode not in ('grouped', 'stream'):
        raise ValueError('Prefill expert mode must be grouped or stream')
    if (BACKEND_NAME != 'cupy' or not isinstance(executor, HeterogeneousExpertExecutor)
            or not executor.device_expert_plan or executor.cache._admission_limit != 0
            or executor.cache.idle_eviction_requests or executor.routing_profiler is not None
            or os.environ.get('MINI_LLM_EXPERT_GPU_Q8', '0') != '1'
            or executor.execution_mode != 'mixed' or executor._ggml_cpu is None
            or x.ndim != 3 or x.shape[0] != 1 or x.dtype != xp.float32
            or routes.dtype != xp.float32 or ids.dtype != xp.int64):
        return None
    tokens, width = x.shape[1:]
    if routes.shape != ids.shape or routes.shape[:2] != (1, tokens) or not 1 <= routes.shape[-1] <= 16:
        raise ValueError('Invalid chunk routing shapes')
    host_ids = asnumpy(ids[0])
    if (np.any(host_ids < 0) or np.any(host_ids >= experts.n_experts)
            or np.any(np.diff(np.sort(host_ids, axis=1), axis=1) == 0)):
        raise ValueError('Chunk routes must be unique valid experts per token')
    active = tuple(int(i) for i in np.unique(host_ids))
    groups = []
    # Resolve capabilities before submitting work. Do not mutate placement.
    from .expert_residency import ExpertKey
    for expert_id in active:
        key = ExpertKey(executor.model_id, experts.layer_id, expert_id)
        resident = executor.cache.resident.get(key)
        weights = None
        if resident is not None:
            expert = resident[0]
            if not isinstance(expert, PackedExpertFFNInference) or not expert.fuse_gate_up:
                return None
            device_weights = (expert.gate, expert.up, expert.down)
            if any(w.quant_type not in FORMATS for w in device_weights):
                return None
        else:
            weights = executor.cpu_source_weights(key)
            if not all(isinstance(w, PackedGGUFWeight) for w in weights):
                return None
            if mode == 'stream' and any(w.quant_type not in FORMATS for w in weights):
                return None
            device_weights = None
        rows, slots = np.nonzero(host_ids == expert_id)
        groups.append((key, rows.astype(np.int32), slots, device_weights, weights))
    stream = xp.cuda.get_current_stream()
    # Also completes the preceding native-service dispatch before CPU resources
    # and transient host slabs can be used by this layer's grouped jobs.
    stream.synchronize()
    host_x = asnumpy(x[0]) if mode == 'grouped' and any(g[3] is None for g in groups) else None
    host_routes = asnumpy(routes[0])
    packed = quantize_q8_1(x[0])
    output = xp.zeros((tokens, width), dtype=xp.float32)
    add, _ = _kernels(xp.cuda.runtime.getDevice())
    pending = []
    jobs = {}
    staging = None
    if mode == 'stream':
        missing_sizes = [sum(w.nbytes for w in g[4]) for g in groups if g[3] is None]
        if missing_sizes:
            staging = _PackedStageRing(max(missing_sizes))
    with executor.dispatch_scope(experts.layer_id, active):
        try:
            # All CPU expert groups start before resident GPU work; no per-token
            # synchronization, route transfer or CPU task submission is required.
            for key, rows, slots, device_weights, weights in groups:
                if device_weights is None and mode == 'grouped':
                    selected = np.ascontiguousarray(host_x[rows])
                    jobs[key.expert_id] = executor._submit_cpu(executor._run_cpu, key, weights, selected)
                    pending.append(jobs[key.expert_id])
                    executor.counters['cpu_submitted'] += len(rows)
            gpu_values = {}
            for key, rows, slots, device_weights, weights in groups:
                if device_weights is not None or mode == 'stream':
                    staged_slot = None
                    if device_weights is None:
                        staged_slot, device_weights = staging.stage(weights, stream)
                        executor.counters['prefill_streamed_bytes'] += sum(w.nbytes for w in weights)
                    row_ids = xp.asarray(rows)
                    gpu_values[key.expert_id] = _gpu_rows(packed, row_ids, device_weights)
                    executor.counters['prefill_gpu_rows'] += len(rows)
                    if staged_slot is not None:
                        staging.consumed(staged_slot, stream)
            for key, rows, slots, device_weights, weights in groups:
                if key.expert_id in jobs:
                    host_value = jobs[key.expert_id].result()
                    value = xp.asarray(host_value)
                    stream.synchronize()  # Fence pageable host lifetime once per group.
                    executor.counters['prefill_cpu_rows'] += len(rows)
                else:
                    value = gpu_values.pop(key.expert_id)
                row_ids = xp.asarray(rows)
                route = xp.asarray(np.ascontiguousarray(host_routes[rows, slots]))
                add(((len(rows) * width + 255)//256,), (256,),
                    (output, value, row_ids, route, np.int32(len(rows)), np.int32(width)))
            executor.counters['prefill_grouped_layers'] += 1
            executor.counters['prefill_grouped_experts'] += len(groups)
            executor.counters['prefill_routed_rows'] += tokens * routes.shape[-1]
        finally:
            # Never leave a CPU task holding checkpoint mappings after failure.
            import sys
            error = None
            for job in pending:
                try:
                    job.result()
                except Exception as exc:
                    if error is None:
                        error = exc
            stream.synchronize()
            if staging is not None:
                staging.fence()
            if error is not None and sys.exc_info()[0] is None:
                raise error
    return output[None]
