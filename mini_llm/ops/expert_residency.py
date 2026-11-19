"""Optional, inference-only expert residency layer shared across model families.

A host provider owns the canonical weights (e.g. RAM, memmap or a GGUF view).
Only sufficiently reused experts are promoted to a bounded device cache.
On a miss the expert is *computed on the CPU*, not unconditionally copied.

The opt-in asynchronous dispatch overlaps host execution of nonresident
experts with device execution of resident experts. On CuPy it uses per-worker
nonblocking streams plus pinned staging for transfers. Dynamic residency must
run *outside* CUDA graph capture. Existing fully resident training/inference
paths never instantiate this layer.
"""

from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import threading
from typing import Callable

from mini_llm.ops.native_packed_cpu import packed_cpu_expert
from mini_llm.ops.expert_victim_index import ExpertVictimIndex

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.experts_inference import ExpertFFNInference
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from mini_llm.ops.quantized_linear import (
    PackedExpertFFNInference, packed_to_device, quantized_linear,
    supports_packed_gpu_type,
)


@dataclass(frozen=True)
class ExpertKey:
    """Uniquely identify an expert across models and layers."""

    model_id: str
    layer_id: int
    expert_id: int


@dataclass
class _Usage:
    score: float = 0.0
    last_step: int = 0
    hits: int = 0


class AdaptiveExpertCache:
    """Device-residency policy with explicit byte budget and host fallback.

    `request` returns a resident device handle or None (use CPU). The caller
    provides an exact retained-device-byte estimate and a materializer. The
    policy is independent of CuPy so it can be regression-tested on CPU.

    Scores decay with successive expert *requests* (rather than tokens);
    the policy is deliberately conservative about replacing popular experts.
    Any temporary allocations during materialization are outside this budget.
    Reserve sufficient physical VRAM for kernels, KV state and capture pools.
    victim_policy='heap' opts into bounded log-density ranking instead of the
    default sort. Floating near ties need not match sort bit-for-bit; exact
    current-score admission thresholds are unchanged.
    """

    def __init__(
        self,
        device_budget_bytes: int,
        *,
        promote_after: int = 3,
        score_half_life: int = 8192,
        replacement_margin: float = 1.05,
        idle_eviction_requests: int = 0,
        victim_policy: str = "sort",
    ):
        if int(device_budget_bytes) != device_budget_bytes or device_budget_bytes < 0:
            raise ValueError("device_budget_bytes must be a nonnegative integer")
        if int(promote_after) != promote_after or promote_after < 1:
            raise ValueError("promote_after must be a positive integer")
        if int(score_half_life) != score_half_life or score_half_life < 1:
            raise ValueError("score_half_life must be a positive integer")
        if not math.isfinite(replacement_margin) or replacement_margin < 1.0:
            raise ValueError("replacement_margin must be finite and >= 1")
        if int(idle_eviction_requests) != idle_eviction_requests or idle_eviction_requests < 0:
            raise ValueError("idle_eviction_requests must be a nonnegative integer")
        if victim_policy not in {"sort", "heap"}:
            raise ValueError("victim_policy must be 'sort' or 'heap'")
        self.victim_policy = victim_policy
        self._victim_index = (ExpertVictimIndex(int(score_half_life))
                              if victim_policy == "heap" else None)
        self.idle_eviction_requests = int(idle_eviction_requests)
        self.device_budget_bytes = int(device_budget_bytes)
        self.promote_after = int(promote_after)
        self.score_half_life = int(score_half_life)
        self.replacement_margin = float(replacement_margin)
        self.resident = {}  # ExpertKey -> (device handle, retained bytes)
        self.used_bytes = 0
        self._usage = {}
        self._step = 0
        self.counters = Counter()
        # Protected keys are selected within the current MoE dispatch.
        # In-flight groups retain cache handles until a caller-stream CUDA
        # event completes. This prevents reusing their storage too early.
        self._protected = Counter()
        self._inflight_groups = []  # [(event, frozenset(ExpertKey))]
        self._admission_limit = None
        self._admission_window = 1
        self._admission_start = 0
        self._admission_attempts = 0

    def set_admission_budget(self, max_promotions=None, *, window_requests=480):
        """Bound synchronous placement attempts, not expert execution or routing.

        None preserves unrestricted adaptive admission; zero freezes placement.
        A positive limit admits at most that many complete-bundle attempts in
        each request-count window, including failed materialization attempts.
        Set only on the executor's serialized dispatch thread. Configuration
        starts a new window without evicting weights or resetting routing heat.
        """
        if (max_promotions is not None and (isinstance(max_promotions, bool) or
                not isinstance(max_promotions, int) or max_promotions < 0)):
            raise ValueError('max_promotions must be None or a nonnegative integer')
        if (isinstance(window_requests, bool) or not isinstance(window_requests, int)
                or window_requests < 1):
            raise ValueError('window_requests must be a positive integer')
        self._admission_limit = max_promotions
        self._admission_window = window_requests
        self._admission_start = self._step
        self._admission_attempts = 0

    def protect(self, keys):
        for key in keys:
            self._protected[key] += 1

    def unprotect(self, keys):
        for key in keys:
            if self._protected[key] <= 1:
                self._protected.pop(key, None)
            else:
                self._protected[key] -= 1

    def record_inflight(self, keys, event):
        """Retain expert handles while GPU kernels reading them are pending."""
        # Below-budget caches may never evict, so reap completed fences here too.
        self._busy_keys()
        relevant = frozenset(k for k in keys if k in self.resident)
        if relevant:
            self._inflight_groups.append((event, relevant))

    def _busy_keys(self):
        waiting = []
        busy = set(self._protected)
        for event, keys in self._inflight_groups:
            try:
                # CuPy exposes nonblocking completion as .done; CPU test
                # fences and other event adapters may provide query().
                completed = bool(event.done if hasattr(event, "done") else event.query())
            except Exception:
                # An unfamiliar/unqueryable event must not lead to use-after-free.
                completed = False
            if not completed:
                waiting.append((event, keys))
                busy.update(keys)
        self._inflight_groups = waiting
        return busy

    def evict_idle(self, min_idle_requests: int, *, target_bytes: int = 0):
        """Discard unused GPU copies without moving canonical host weights.

        Never evicts protected experts or experts with pending GPU reads.
        target_bytes=0 removes all eligible idle entries; a positive target
        stops once resident_bytes <= target_bytes. No device-wide sync.
        """
        if int(min_idle_requests) != min_idle_requests or min_idle_requests < 0:
            raise ValueError("min_idle_requests must be nonnegative")
        if int(target_bytes) != target_bytes or target_bytes < 0:
            raise ValueError("target_bytes must be nonnegative")
        busy = self._busy_keys()
        evicted = 0
        for key in sorted(self.resident, key=lambda k: self._usage[k].last_step):
            if target_bytes and self.used_bytes <= target_bytes:
                break
            if key in busy or self._step - self._usage[key].last_step < min_idle_requests:
                continue
            self.used_bytes -= self.resident.pop(key)[1]
            if self._victim_index is not None:
                self._victim_index.remove(key)
            self.counters["evictions"] += 1
            self.counters["idle_evictions"] += 1
            evicted += 1
        return evicted

    def _effective_score(self, key):
        usage = self._usage[key]
        return usage.score * 2.0 ** (
            -(self._step - usage.last_step) / self.score_half_life
        )

    def request(self, key: ExpertKey, size_bytes: int, materialize: Callable):
        """Return cached/device handle, or None when CPU execution is cheaper.

        `materialize()` should return a device handle retaining `size_bytes`.
        Failed materialization leaves the cache usable; an OOM is counted and
        the caller can take the CPU path. On failures unrelated to OOM the
        exception propagates rather than masking an incorrect model.
        """
        if not isinstance(key, ExpertKey):
            raise TypeError("key must be ExpertKey")
        if int(size_bytes) != size_bytes or size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")
        size_bytes = int(size_bytes)
        self._step += 1
        usage = self._usage.setdefault(key, _Usage())
        usage.score = self._effective_score(key) + 1.0
        usage.last_step = self._step
        usage.hits += 1
        self.counters["requests"] += 1
        if self.idle_eviction_requests:
            self.evict_idle(self.idle_eviction_requests)

        if key in self.resident:
            if self._victim_index is not None:
                self._victim_index.update(key, usage.score, self.resident[key][1], usage.last_step)
            self.counters["hits"] += 1
            return self.resident[key][0]
        self.counters["misses"] += 1
        if usage.hits < self.promote_after or size_bytes > self.device_budget_bytes:
            self.counters["cpu_fallbacks"] += 1
            return None

        if self._admission_limit is not None:
            if self._step - self._admission_start > self._admission_window:
                self._admission_start = self._step - 1
                self._admission_attempts = 0
            if self._admission_attempts >= self._admission_limit:
                self.counters['admission_deferred'] += 1
                self.counters['cpu_fallbacks'] += 1
                return None

        candidates = []
        freed = 0
        required = self.used_bytes + size_bytes - self.device_budget_bytes
        if required > 0:
            busy = self._busy_keys()
            # Prefer experts with the lowest decayed request density per byte.
            # Sort keys only: retaining a victim handle here keeps its slot
            # leased even after deletion, causing a false allocation failure.
            victims = (self._victim_index.victims() if self._victim_index is not None
                       else nullcontext(sorted(
                           self.resident,
                           key=lambda key: (self._effective_score(key) / self.resident[key][1],
                                            self._usage[key].last_step),
                       )))
            with victims as ordered_victims:
                for victim in ordered_victims:
                    victim_size = self.resident[victim][1]
                    if victim in busy:
                        self.counters["busy_victim_skips"] += 1
                        continue
                    # Avoid replacing a more valuable resident with a cold expert.
                    victim_density = self._effective_score(victim) / victim_size
                    if usage.score / size_bytes <= self.replacement_margin * victim_density:
                        break
                    candidates.append(victim)
                    freed += victim_size
                    if freed >= required:
                        break
            if freed < required:
                self.counters["cpu_fallbacks"] += 1
                return None

        self._admission_attempts += 1
        # Evict only after deciding promotion is possible. Keeping the removed
        # objects alive until after materialization can temporarily exceed the
        # budget, so release them here and rely on the caller's VRAM reserve.
        for victim in candidates:
            self.used_bytes -= self.resident[victim][1]
            del self.resident[victim]
            if self._victim_index is not None:
                self._victim_index.remove(victim)
            self.counters["evictions"] += 1
            self.counters["swap_evictions"] += 1
        try:
            handle = materialize()
        except Exception as exc:
            # Only memory allocation failures are recoverable; syntax, shape,
            # or kernel failures must not silently turn into a CPU fallback.
            is_memory_error = isinstance(exc, MemoryError)
            if BACKEND_NAME == "cupy":
                is_memory_error |= isinstance(exc, xp.cuda.memory.OutOfMemoryError)
            if not is_memory_error:
                raise
            self.counters["allocation_failures"] += 1
            self.counters["cpu_fallbacks"] += 1
            return None
        self.resident[key] = (handle, size_bytes)
        if self._victim_index is not None:
            self._victim_index.update(key, usage.score, size_bytes, usage.last_step)
        self.used_bytes += size_bytes
        self.counters["promotions"] += 1
        return handle

    def clear(self):
        """Release GPU weights safely at model unload (may block for CUDA)."""
        for event, _ in self._inflight_groups:
            event.synchronize()
        self._inflight_groups.clear()
        self._protected.clear()
        self.resident.clear()
        if self._victim_index is not None:
            self._victim_index.clear()
        self._usage.clear()
        self._step = 0
        self._admission_start = 0
        self._admission_attempts = 0
        self.used_bytes = 0
        self.counters.clear()

    def statistics(self):
        return {
            **dict(self.counters),
            "admission_budget": self._admission_limit,
            "admission_window_requests": self._admission_window,
            "resident_experts": len(self.resident),
            "resident_bytes": self.used_bytes,
            "device_budget_bytes": self.device_budget_bytes,
            "protected_experts": len(self._protected),
            "inflight_dispatches": len(self._inflight_groups),
            "hit_rate": (self.counters["hits"] / self.counters["requests"]
                         if self.counters["requests"] else 0.0),
        }


class CPUDequantizedExpertCache:
    """Optional bounded host LRU for repeatedly selected GGUF experts.

    Unlike the device expert cache, this caches the *decoded FP32* matrices so
    an expert running repeatedly on CPU is not dequantized on every token.
    The canonical provider still owns the packed GGUF bytes. The cache only
    retains immutable reference matrices, so it is safe to share between CPU
    worker threads. A zero budget disables retention entirely.
    """

    def __init__(self, budget_bytes: int = 0):
        if int(budget_bytes) != budget_bytes or budget_bytes < 0:
            raise ValueError("host_dequant_budget_bytes must be a nonnegative integer")
        self.budget_bytes = int(budget_bytes)
        self.used_bytes = 0
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.counters = Counter()

    def get(self, key: ExpertKey, weights):
        """Return host FP32 GGUF projections, decoding on a cache miss."""
        if not all(isinstance(w, PackedGGUFWeight) for w in weights):
            raise TypeError("Host decoded cache accepts complete packed experts only")
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.counters["hits"] += 1
                return self._entries[key][0]
            self.counters["misses"] += 1
            # Hold the lock across decoding to avoid duplicate allocations
            # when concurrent workers request the same expert. The NumPy
            # decoder is a correctness implementation, not a CPU fast kernel.
            dense = tuple(np.ascontiguousarray(w.dequantize_reference(), dtype=np.float32)
                          for w in weights)
            for matrix in dense:
                matrix.flags.writeable = False
            size = sum(w.nbytes for w in dense)
            if size <= self.budget_bytes:
                while self.used_bytes + size > self.budget_bytes:
                    _, (_, evicted_size) = self._entries.popitem(last=False)
                    self.used_bytes -= evicted_size
                    self.counters["evictions"] += 1
                self._entries[key] = (dense, size)
                self.used_bytes += size
            return dense

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.used_bytes = 0
            self.counters.clear()

    def statistics(self):
        with self._lock:
            return {
                **dict(self.counters),
                "resident_experts": len(self._entries),
                "used_bytes": self.used_bytes,
                "budget_bytes": self.budget_bytes,
            }


class HeterogeneousExpertExecutor:
    """Optional MoE expert executor plugged into `ExpertsInference`.

    `host_provider(ExpertKey)` returns (gate, up, down) either as existing
    dense matrices [(D,F), (D,F), (F,D)] or GGUF PackedGGUFWeight matrices
    [(F,D), (F,D), (D,F)] with row-major output-first orientation.
    Packed matrices stay packed on GPU; no intermediate dense GPU weights.
    Providers may return zero-copy mmap views into larger expert tensors.

    A device budget of zero means CPU-only. On NumPy, CPU execution remains
    active even if a positive device budget is supplied. Setting cpu_workers
    to a positive integer enables the asynchronous submit/resolve API. On
    CuPy, CPU workers stage activations through pinned host memory on their
    own nonblocking CUDA streams; queued resident-GPU kernels execute in
    parallel. Only the managed inference path uses this API. The CPU packed
    decode remains a reference kernel, not yet a high-throughput CPU GEMV.
    """

    def __init__(
        self, host_provider: Callable,
        *,
        model_id: str,
        d_model: int,
        d_ff: int,
        device_budget_bytes: int = 0,
        compute_dtype: str = "float16",
        promote_after: int = 3,
        score_half_life: int = 8192,
        cpu_workers: int = 0,
        host_dequant_budget_bytes: int = 0,
        cpu_packed_backend: str = "reference",
        idle_eviction_requests: int = 0,
        routing_profiler=None,
        execution_mode: str = "mixed",
        batch_cpu_decode: bool = True,
        group_after_admission: bool = False,
        expert_slot_stride_bytes: int = 0,
        slot_staging_budget_bytes: int = 64 * 1024**2,
        expert_victim_policy: str = "sort",
        group_mixed_decode: bool = False,
        device_expert_plan: bool = False,
    ):
        if execution_mode not in {"mixed", "cpu", "gpu"}:
            raise ValueError("execution_mode must be 'mixed', 'cpu' or 'gpu'")
        if execution_mode == "gpu" and BACKEND_NAME != "cupy":
            raise RuntimeError("GPU execution requires MINI_LLM_BACKEND=cupy")
        self.execution_mode = execution_mode
        self.batch_cpu_decode = bool(batch_cpu_decode)
        if not isinstance(group_after_admission, bool):
            raise TypeError('group_after_admission must be bool')
        self.group_after_admission = group_after_admission
        if not isinstance(group_mixed_decode, bool):
            raise TypeError('group_mixed_decode must be bool')
        self.group_mixed_decode = group_mixed_decode
        if not isinstance(device_expert_plan, bool):
            raise TypeError('device_expert_plan must be bool')
        if device_expert_plan and BACKEND_NAME != 'cupy':
            raise RuntimeError('Device expert plans require CuPy')
        self.device_expert_plan = device_expert_plan
        self._device_expert_plans = {}
        if (isinstance(expert_slot_stride_bytes, bool) or
                not isinstance(expert_slot_stride_bytes, int) or expert_slot_stride_bytes < 0):
            raise ValueError('expert_slot_stride_bytes must be a nonnegative integer')
        self.expert_slot_stride_bytes = expert_slot_stride_bytes
        self.slot_staging_budget_bytes = slot_staging_budget_bytes
        self._slot_arena = None
        if routing_profiler is not None and routing_profiler.identity["model_id"] != model_id:
            raise ValueError("Routing trace model identity does not match executor")
        self.routing_profiler = routing_profiler
        if int(cpu_workers) != cpu_workers or cpu_workers < 0:
            raise ValueError("cpu_workers must be a nonnegative integer")
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if int(d_model) < 1 or int(d_ff) < 1:
            raise ValueError("d_model and d_ff must be positive")
        if compute_dtype not in {"float16", "float32"}:
            raise ValueError("compute_dtype must be 'float16' or 'float32'")
        if cpu_packed_backend not in {"reference", "auto", "native", "ggml", "ggml_pool"}:
            raise ValueError("cpu_packed_backend must be reference, auto, native, ggml or ggml_pool")
        self.cpu_packed_backend = cpu_packed_backend
        self._ggml_cpu = None
        if cpu_packed_backend in ('ggml', 'ggml_pool'):
            if compute_dtype != 'float32':
                raise ValueError('GGML packed CPU execution requires float32 activations')
            from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor
            self._ggml_cpu = GGMLPackedCPUExecutor()
        self._dispatch_depth = 0
        self.host_provider = host_provider
        self.model_id = model_id
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.compute_dtype = compute_dtype
        self.cache = AdaptiveExpertCache(
            device_budget_bytes, promote_after=1 if execution_mode == "gpu" else promote_after,
            score_half_life=score_half_life,
            idle_eviction_requests=idle_eviction_requests,
            victim_policy=expert_victim_policy,
        )
        self.host_cache = CPUDequantizedExpertCache(host_dequant_budget_bytes)
        self.cpu_workers = int(cpu_workers)
        self._pool = (ThreadPoolExecutor(max_workers=self.cpu_workers,
                                         thread_name_prefix="mylocal-moe-cpu")
                      if self.cpu_workers else None)
        self._ggml_row_pool = None
        if cpu_packed_backend == 'ggml_pool':
            if not self.cpu_workers:
                raise ValueError('ggml_pool requires positive cpu_workers')
            from .ggml_packed_cpu import GGMLPackedCPURowPool
            self._ggml_row_pool = GGMLPackedCPURowPool(
                self._ggml_cpu, self.d_model, self.d_ff, self.cpu_workers)
        self._worker_local = threading.local()
        self.counters = Counter()

    @contextmanager
    def dispatch_scope(self, layer_id: int, expert_ids):
        """Protect this router's working set until GPU kernels are enqueued.

        One CUDA event per MoE layer (rather than per expert) protects the
        selected resident weights while all earlier main-stream kernels run.
        Cache misses and CPU worker synchronization stay outside graphs.
        """
        keys = tuple(ExpertKey(self.model_id, int(layer_id), int(e)) for e in expert_ids)
        self.cache.protect(keys)
        self._dispatch_depth += 1
        try:
            yield
        finally:
            try:
                if BACKEND_NAME == "cupy" and any(k in self.cache.resident for k in keys):
                    event = xp.cuda.Event(disable_timing=True)
                    event.record(xp.cuda.get_current_stream())
                    self.cache.record_inflight(keys, event)
            finally:
                self._dispatch_depth -= 1
                self.cache.unprotect(keys)

    def _record_direct_gpu_use(self, key):
        # Direct executor.forward()/submit() calls don't have a layer scope.
        if BACKEND_NAME == "cupy" and not self._dispatch_depth:
            event = xp.cuda.Event(disable_timing=True)
            event.record(xp.cuda.get_current_stream())
            self.cache.record_inflight((key,), event)

    @property
    def overlaps_cpu_gpu(self):
        return self._pool is not None

    def close(self):
        """Wait for outstanding CPU tasks; leave synchronous execution available."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._ggml_row_pool is not None:
            self._ggml_row_pool.close()
            self._ggml_row_pool = None
        for plan in self._device_expert_plans.values():
            plan.close()
        self._device_expert_plans.clear()
        if self._slot_arena is not None:
            # Arena backing cannot retire before consumers on other streams.
            for event, _ in self.cache._inflight_groups:
                event.synchronize()

    @property
    def resident_device_bytes(self):
        """Physical expert storage; a slot arena owns even its unused slots."""
        return self.cache.used_bytes if self._slot_arena is None else self._slot_arena.nbytes

    def statistics(self):
        return {
            "device": self.cache.statistics(),
            "host_dequant": self.host_cache.statistics(),
            "execution": dict(self.counters),
            "cpu_workers": self.cpu_workers,
            "cpu_packed_backend": self.cpu_packed_backend,
            "slots": None if self._slot_arena is None else self._slot_arena.statistics(),
        }

    def _host_weights(self, key):
        gate, up, down = self.host_provider(key)
        provided = (gate, up, down)
        packed = [isinstance(w, PackedGGUFWeight) for w in provided]
        if any(packed) and not all(packed):
            raise ValueError(f"{key}: mixed dense/packed expert projections are unsupported")
        if all(packed):
            expected = ((self.d_ff, self.d_model), (self.d_ff, self.d_model),
                        (self.d_model, self.d_ff))
        else:
            provided = tuple(np.asarray(w) for w in provided)
            expected = ((self.d_model, self.d_ff), (self.d_model, self.d_ff),
                        (self.d_ff, self.d_model))
        for matrix, shape in zip(provided, expected):
            if matrix.shape != shape:
                raise ValueError(f"{key}: expert weight {matrix.shape} != {shape}")
        return provided

    def reserve_device_slots(self):
        """Commit the approved fixed-slot arena before the first routing call."""
        if (BACKEND_NAME != 'cupy' or not self.expert_slot_stride_bytes or
                self.cache.device_budget_bytes < self.expert_slot_stride_bytes):
            return None
        if self._slot_arena is None:
            from .packed_expert_slots import PackedExpertSlotArena
            self._slot_arena = PackedExpertSlotArena(self.cache.device_budget_bytes,
                self.expert_slot_stride_bytes, staging_budget_bytes=self.slot_staging_budget_bytes)
        return self._slot_arena.statistics()

    def _device_expert(self, weights):
        gate, up, down = weights
        if isinstance(gate, PackedGGUFWeight):
            if self.expert_slot_stride_bytes:
                self.reserve_device_slots()
                if self._slot_arena is None:
                    raise MemoryError('Expert cache budget cannot fit one slot')
                return self._slot_arena.materialize(weights)
            # Allocate only the retained compressed byte count; this is the
            # same expert forward interface as the existing resident path.
            return PackedExpertFFNInference(
                packed_to_device(gate), packed_to_device(up), packed_to_device(down),
            )
        expert = ExpertFFNInference(self.d_model, self.d_ff, self.compute_dtype)
        # Pack on host *before* uploading to avoid transiently allocating two
        # copies of the gate/up weights in VRAM. This uses the existing fused
        # expert forward interface without duplicating the math.
        gate_up = np.concatenate((gate, up), axis=1)
        expert.W_gate_up = xp.asarray(gate_up, dtype=self.compute_dtype)
        expert.W_gate = expert.W_gate_up[:, :self.d_ff]
        expert.W_up = expert.W_gate_up[:, self.d_ff:]
        expert.W_down = xp.asarray(down, dtype=self.compute_dtype)
        return expert

    @staticmethod
    def _bounded_projection(x, weight):
        rows = x.reshape(-1, x.shape[-1])
        result = np.empty((len(rows), weight.shape[0]), dtype=x.dtype)
        for i, row in enumerate(rows):
            if x.dtype != np.float64:
                result[i] = weight.matvec_reference(row, tile_rows=1)
            else:
                for j in range(weight.shape[0]):
                    decoded = weight.select_outer(j).dequantize_reference()
                    result[i, j] = row @ decoded.astype(np.float64)
        return result.reshape(x.shape[:-1] + (weight.shape[0],))

    def _cpu_forward_numpy(self, key, weights, x_host):
        """Execute a selected expert entirely on host; never call CuPy here."""
        dtype = np.float64 if x_host.dtype == np.float64 else np.float32
        x_cpu = np.asarray(x_host, dtype=dtype)
        gate, up, down = weights
        if isinstance(gate, PackedGGUFWeight):
            if self._ggml_cpu is not None:
                result = self._ggml_cpu.expert(x_cpu, weights)
                self.counters['ggml_cpu_experts'] += 1
                return result.astype(x_host.dtype, copy=False)
            if self.cpu_packed_backend != "reference":
                native = packed_cpu_expert(x_cpu, weights, backend=self.cpu_packed_backend)
                if native is not None:
                    self.counters["native_cpu_experts"] += 1
                    return native.astype(x_host.dtype, copy=False)
                self.counters["native_cpu_fallbacks"] += 1
            if self.host_cache.budget_bytes:
                gmat, umat, dmat = self.host_cache.get(key, weights)
                g = x_cpu @ gmat.T
                u = x_cpu @ umat.T
            else:
                g = self._bounded_projection(x_cpu, gate)
                u = self._bounded_projection(x_cpu, up)
        else:
            g = x_cpu @ np.asarray(gate, dtype=dtype)
            u = x_cpu @ np.asarray(up, dtype=dtype)
        h = (g / (1.0 + np.exp(-g))) * u
        if isinstance(down, PackedGGUFWeight):
            result = (h @ dmat.T if self.host_cache.budget_bytes else
                      self._bounded_projection(h, down))
        else:
            result = h @ np.asarray(down, dtype=dtype)
        return np.asarray(result, dtype=x_host.dtype)

    def _run_cpu(self, key, weights, x, trace=None):
        with trace.span("cpu_compute") if trace is not None else nullcontext():
            return self._cpu_forward_numpy(key, weights, x)

    def _cpu_forward(self, key, weights, x, trace=None):
        """Synchronous reference route, compatible with the original API."""
        with trace.span("activation_d2h", gpu=True) if trace is not None else nullcontext():
            host_x = asnumpy(x)
        host_y = self._run_cpu(key, weights, host_x, trace)
        with trace.span("activation_h2d", gpu=True) if trace is not None else nullcontext():
            output = xp.asarray(host_y, dtype=x.dtype)
        if trace is not None and BACKEND_NAME == "cupy":
            trace.activation_bytes("d2h", x.nbytes)
            trace.activation_bytes("h2d", output.nbytes)
        return output

    def _copy_compute_copy(self, key, weights, x, ready_event, device_id, gpu_y, trace=None):
        """One CuPy worker: staged D2H, host GEMV, staged H2D.

        The input-ready event handles producer-stream ordering (including
        the routing gather). A dedicated nonblocking stream per worker keeps
        copies independent of main-stream GPU expert kernels. Both copies
        complete before the Future resolves: output lifetime and main-stream
        consumption therefore require no cross-stream synchronization and
        pinned buffers cannot be recycled while DMA is outstanding. The caller
        allocates gpu_y before recording ready_event so the worker cannot write
        a recycled caller-stream buffer before its previous readers complete.
        """
        with xp.cuda.Device(device_id):
            stream = getattr(self._worker_local, "stream", None)
            if stream is None:
                stream = xp.cuda.Stream(non_blocking=True)
                self._worker_local.stream = stream
            in_mem = xp.cuda.alloc_pinned_memory(int(x.nbytes))
            host_x = np.frombuffer(in_mem, dtype=x.dtype,
                                   count=int(x.size)).reshape(x.shape)
            stream.wait_event(ready_event)
            with stream:
                with trace.span("activation_d2h", gpu=True) if trace is not None else nullcontext():
                    x.get(out=host_x, stream=stream, blocking=False)
            with trace.span("d2h_wait") if trace is not None else nullcontext():
                stream.synchronize()  # CPU must not read before D2H completes.
            host_y = self._run_cpu(key, weights, host_x, trace)
            out_mem = xp.cuda.alloc_pinned_memory(int(host_y.nbytes))
            host_out = np.frombuffer(out_mem, dtype=host_y.dtype,
                                     count=int(host_y.size)).reshape(host_y.shape)
            np.copyto(host_out, host_y)
            with stream:
                with trace.span("activation_h2d", gpu=True) if trace is not None else nullcontext():
                    gpu_y.set(host_out, stream=stream)
            with trace.span("h2d_wait") if trace is not None else nullcontext():
                stream.synchronize()  # Guarantees host buffer reuse and visibility.
            if trace is not None:
                trace.activation_bytes("d2h", x.nbytes)
                trace.activation_bytes("h2d", gpu_y.nbytes)
            return gpu_y

    def _prepare(self, layer_id, expert_id, x):
        if self.routing_profiler is None:
            return (*self._select(layer_id, expert_id, x), None)
        trace = self.routing_profiler.new_expert(layer_id, expert_id,
                                                math.prod(x.shape[:-1]))
        before = dict(self.cache.counters)
        key = ExpertKey(self.model_id, int(layer_id), int(expert_id))
        # A plain hit cannot evict anything. Avoid a full cache snapshot/scan
        # for every observed hit; idle eviction and misses still need it.
        resident_before = ({key} if key in self.cache.resident and
                           not self.cache.idle_eviction_requests else set(self.cache.resident))
        with trace.span("selection"):
            key, expert, weights = self._select(layer_id, expert_id, x, trace)
        trace.selected(key, expert, self.cache, before, resident_before,
                       cache_enabled=bool(BACKEND_NAME == "cupy" and
                                          self.cache.device_budget_bytes))
        return key, expert, weights, trace

    def _materialize(self, weights, trace):
        with trace.span("promotion", gpu=True) if trace is not None else nullcontext():
            return self._device_expert(weights)

    def _select(self, layer_id, expert_id, x, trace=None):
        if x.shape[-1] != self.d_model:
            raise ValueError("Expert input has incorrect hidden width")
        key = ExpertKey(self.model_id, int(layer_id), int(expert_id))
        weights = None
        if (self.execution_mode != "cpu" and BACKEND_NAME == "cupy"
                and self.cache.device_budget_bytes):
            if key in self.cache.resident:
                size = self.cache.resident[key][1]
                expert = self.cache.request(key, size, lambda: None)
                return key, expert, None
            weights = self._host_weights(key)
            if isinstance(weights[0], PackedGGUFWeight):
                unsupported = [w.quant_type for w in weights
                               if not supports_packed_gpu_type(w.quant_type)]
                if unsupported:
                    if self.execution_mode == "gpu":
                        raise NotImplementedError(f"No packed GPU kernel for {unsupported}")
                    self.counters["unsupported_gpu_fallbacks"] += 1
                    return key, None, weights
                size = self.expert_slot_stride_bytes or sum(w.nbytes for w in weights)
            else:
                size = sum(w.size for w in weights) * np.dtype(self.compute_dtype).itemsize
            expert = self.cache.request(key, size, lambda: self._materialize(weights, trace))
            if expert is not None:
                return key, expert, None
        if self.execution_mode == "gpu":
            raise RuntimeError("GPU execution could not obtain residency within the device budget")
        if weights is None:
            weights = self._host_weights(key)
        return key, None, weights

    def submit(self, layer_id: int, expert_id: int, x):
        """Start expert work without waiting for nonresident CPU execution.

        Returns either a ready/queued backend array (GPU-resident expert) or
        a Future (CPU expert). The caller *must* resolve every submitted task,
        even if a different expert throws, before reusing input workspaces.
        """
        if self._pool is None:
            return self.forward(layer_id, expert_id, x)
        key, expert, weights, trace = self._prepare(layer_id, expert_id, x)
        if expert is not None:
            self.counters["gpu_experts"] += 1
            with trace.span("gpu_compute", gpu=True) if trace is not None else nullcontext():
                output = expert.forward(x)
            self._record_direct_gpu_use(key)
            return output
        self.counters["cpu_submitted"] += 1
        if BACKEND_NAME == "cupy":
            # Record after the gather that produced this expert's activations;
            # a worker stream must wait for this event before reading x.
            device_id = int(xp.cuda.runtime.getDevice())
            # Pool reuse is stream-ordered. Allocate on the consuming stream
            # before its ready fence, not later on the worker's default stream.
            gpu_y = xp.empty(x.shape, dtype=x.dtype)
            ready = xp.cuda.Event(disable_timing=True)
            ready.record(xp.cuda.get_current_stream())
            return self._pool.submit(self._copy_compute_copy, key, weights,
                                     x, ready, device_id, gpu_y, trace)
        return self._pool.submit(self._run_cpu, key, weights, x, trace)

    def try_forward_device_plan(self, layer_id, x, weights, indices, n_experts, *, on_cpu_submitted=None):
        if not self.device_expert_plan:
            return None
        from .device_expert_plan import try_forward
        return try_forward(self, layer_id, x, weights, indices, n_experts,
                                   on_cpu_submitted=on_cpu_submitted)

    def try_forward_decode_batch(self, layer_id, expert_ids, x, routing_weights, slots, *,
                                     on_cpu_submitted=None):
        """Share one immutable activation across CPU misses in singleton decode.

        Called only inside dispatch_scope, with unique ascending IDs. FP32-only
        eligibility preserves the existing multiply then sequential-add boundaries.
        All scratch is dispatch-local, O(selected experts * D); CPU math and GPU
        expert.forward are unchanged. Profiling retains the per-expert copy path.
        """
        if (not self.batch_cpu_decode or self._pool is None or
                self.routing_profiler is not None or not self._dispatch_depth or
                x.shape != (1, self.d_model) or x.dtype != xp.float32 or
                routing_weights.dtype != xp.float32):
            return None
        expert_ids = tuple(expert_ids)
        if not expert_ids or expert_ids != tuple(sorted(set(expert_ids))):
            return None
        prepared = [self._prepare(layer_id, e, x) for e in expert_ids]
        cpu = [i for i, (_, expert, _, _) in enumerate(prepared) if expert is None]
        if self.group_after_admission and not cpu and BACKEND_NAME == 'cupy':
            from .grouped_quantized_experts import (
                supports_grouped_quantized_experts, grouped_quantized_experts_forward)
            experts = [item[1] for item in prepared]
            if (all(isinstance(e, PackedExpertFFNInference) and e.fuse_gate_up for e in experts)
                    and supports_grouped_quantized_experts(
                        x, experts, routing_weights, routing_slots=slots)):
                # _prepare already counted requests and published complete bundles.
                # The surrounding dispatch scope protects new and old residents alike.
                if on_cpu_submitted is not None:
                    on_cpu_submitted()
                output = grouped_quantized_experts_forward(
                    x, experts, routing_weights, routing_slots=slots)
                self.counters['gpu_experts'] += len(experts)
                self.counters['grouped_gpu_dispatches'] += 1
                self.counters['admission_grouped_dispatches'] += 1
                return output
        outputs = [None] * len(prepared)
        jobs = []
        gpu = [i for i, (_, expert, _, _) in enumerate(prepared) if expert is not None]
        mixed_terms = None
        mixed_experts = None
        if self.group_mixed_decode and cpu and gpu and BACKEND_NAME == 'cupy':
            from .grouped_quantized_experts import supports_grouped_quantized_experts
            from .iq_packed_gpu import supports_type
            candidates = [prepared[i][1] for i in gpu]
            if (all(isinstance(e, PackedExpertFFNInference) and e.fuse_gate_up and
                    all(supports_type(w.quant_type) for w in (e.gate, e.up, e.down))
                    for e in candidates) and supports_grouped_quantized_experts(
                        x, candidates, routing_weights[:len(gpu)])):
                mixed_experts = candidates
        if cpu:
            if BACKEND_NAME == "cupy":
                stream = xp.cuda.get_current_stream()
                in_mem = xp.cuda.alloc_pinned_memory(int(x.nbytes))
                host_x = np.frombuffer(in_mem, dtype=x.dtype, count=int(x.size)).reshape(x.shape)
                x.get(out=host_x, stream=stream, blocking=False)
                stream.synchronize()  # Includes the activation's current-stream producer.
                self.counters["decode_activation_d2h"] += 1
                out_mem = xp.cuda.alloc_pinned_memory(len(cpu) * self.d_model * x.dtype.itemsize)
                host_out = np.frombuffer(out_mem, dtype=x.dtype,
                                         count=len(cpu) * self.d_model).reshape(len(cpu), self.d_model)
            else:
                host_x = np.array(x, copy=True)
                host_out = np.empty((len(cpu), self.d_model), dtype=x.dtype)
            host_x.flags.writeable = False
        try:
            row_pool = self._ggml_row_pool
            if cpu and row_pool is not None and len(cpu) <= row_pool.capacity and all(
                    isinstance(w, PackedGGUFWeight) for i in cpu for w in prepared[i][2]):
                jobs.append(row_pool.submit(host_x, [prepared[i][2] for i in cpu], out=host_out))
                self.counters['cpu_submitted'] += len(cpu)
                self.counters['ggml_cpu_experts'] += len(cpu)
                self.counters['ggml_row_batches'] += 1
            else:
                for row, i in enumerate(cpu):
                    key, _, weights, _ = prepared[i]
                    jobs.append(self._pool.submit(
                        self._run_cpu_decode_row, key, weights, host_x, host_out[row:row + 1]))
                    self.counters["cpu_submitted"] += 1
            # Launch independent shared/resident work only after CPU inputs are published.
            if on_cpu_submitted is not None:
                on_cpu_submitted()
            if mixed_experts is not None:
                from .iq_packed_gpu import grouped_terms
                gpu_slots = xp.asarray([slots[i] for i in gpu], dtype=xp.int32)
                mixed_terms = grouped_terms(x, mixed_experts, routing_weights[gpu_slots])
                self.counters['gpu_experts'] += len(gpu)
                self.counters['mixed_grouped_gpu_dispatches'] += 1
            else:
                for i, (_, expert, _, _) in enumerate(prepared):
                    if expert is not None:
                        self.counters["gpu_experts"] += 1
                        outputs[i] = expert.forward(x)
        finally:
            # Drain even after submission/GPU failure, and retain the original
            # exception if one is already propagating through this finally.
            error = None
            for job in jobs:
                try:
                    job.result()
                except Exception as exc:
                    if error is None:
                        error = exc
            import sys
            if error is not None and sys.exc_info()[0] is None:
                raise error
        if cpu:
            if BACKEND_NAME == "cupy":
                device_out = xp.empty(host_out.shape, dtype=x.dtype)
                device_out.set(host_out, stream=stream)
                stream.synchronize()  # Pinned slab must outlive DMA, including on return.
                self.counters["decode_results_h2d"] += 1
            else:
                device_out = host_out
            for row, i in enumerate(cpu):
                outputs[i] = device_out[row:row + 1]
        if mixed_terms is not None:
            from .mixed_expert_combine_gpu import combine
            positions = {i: row for row, i in enumerate(gpu)}
            positions.update({i: -row-1 for row, i in enumerate(cpu)})
            output = combine(mixed_terms, device_out, routing_weights,
                             tuple(positions[i] for i in range(len(prepared))), tuple(slots))
            self.counters['decode_batch_dispatches'] += 1
            return output
        output = xp.zeros_like(x)
        for slot, expert_out in zip(slots, outputs):
            output += routing_weights[slot:slot + 1] * expert_out
        self.counters["decode_batch_dispatches"] += 1
        return output

    def _run_cpu_decode_row(self, key, weights, host_x, destination):
        # Futures retain no per-expert output arrays after copying their row.
        np.copyto(destination, self._run_cpu(key, weights, host_x))

    def try_forward_grouped(self, layer_id, expert_ids, x, routing_weights, slots):
        """Batch compatible resident decode experts inside the caller's scope.

        Eligibility is observational: a miss/unsupported layout changes no
        request counters and falls through to ordinary hybrid dispatch.
        """
        if (self.execution_mode == "cpu" or BACKEND_NAME != "cupy" or not self._dispatch_depth or
                self.cache.idle_eviction_requests or x.shape[0] != 1):
            return None
        from .grouped_quantized_experts import (
            supports_grouped_quantized_experts, grouped_quantized_experts_forward)
        keys = tuple(ExpertKey(self.model_id, int(layer_id), int(e)) for e in expert_ids)
        if not keys or any(key not in self.cache.resident for key in keys):
            return None
        experts = [self.cache.resident[key][0] for key in keys]
        if (any(not isinstance(expert, PackedExpertFFNInference) or not expert.fuse_gate_up
                for expert in experts) or
                not supports_grouped_quantized_experts(x, experts, routing_weights)):
            return None
        tickets = [self._prepare(layer_id, expert_id, x)[3] for expert_id in expert_ids]
        trace = tickets[0]
        # One group interval, not E copies of the same kernel time.
        with trace.span("grouped_gpu_compute", gpu=True) if trace is not None else nullcontext():
            output = grouped_quantized_experts_forward(x, experts, routing_weights,
                                                       routing_slots=slots)
        self.counters["gpu_experts"] += len(experts)
        self.counters["grouped_gpu_dispatches"] += 1
        return output

    @staticmethod
    def resolve(work):
        """Collect one submitted expert result, propagating any worker error."""
        return work.result() if isinstance(work, Future) else work

    def forward(self, layer_id: int, expert_id: int, x):
        """Synchronous entry point retained for existing callers."""
        key, expert, weights, trace = self._prepare(layer_id, expert_id, x)
        if expert is not None:
            with trace.span("gpu_compute", gpu=True) if trace is not None else nullcontext():
                output = expert.forward(x)
            self._record_direct_gpu_use(key)
            return output
        return self._cpu_forward(key, weights, x, trace)
