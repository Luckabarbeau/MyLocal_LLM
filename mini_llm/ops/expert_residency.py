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
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
from typing import Callable

from mini_llm.ops.native_packed_cpu import packed_cpu_expert

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.experts_inference import ExpertFFNInference
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from mini_llm.ops.quantized_linear import (
    PackedExpertFFNInference, packed_to_device, quantized_linear,
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
    """

    def __init__(
        self,
        device_budget_bytes: int,
        *,
        promote_after: int = 3,
        score_half_life: int = 8192,
        replacement_margin: float = 1.05,
        idle_eviction_requests: int = 0,
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
        relevant = frozenset(k for k in keys if k in self.resident)
        if relevant:
            self._inflight_groups.append((event, relevant))

    def _busy_keys(self):
        waiting = []
        busy = set(self._protected)
        for event, keys in self._inflight_groups:
            try:
                completed = bool(event.query())
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
            self.counters["hits"] += 1
            return self.resident[key][0]
        self.counters["misses"] += 1
        if usage.hits < self.promote_after or size_bytes > self.device_budget_bytes:
            self.counters["cpu_fallbacks"] += 1
            return None

        candidates = []
        freed = 0
        required = self.used_bytes + size_bytes - self.device_budget_bytes
        if required > 0:
            busy = self._busy_keys()
            # Prefer experts with the lowest decayed request density per byte.
            for victim, (_, victim_size) in sorted(
                self.resident.items(),
                key=lambda pair: (self._effective_score(pair[0]) / pair[1][1],
                                  self._usage[pair[0]].last_step),
            ):
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

        # Evict only after deciding promotion is possible. Keeping the removed
        # objects alive until after materialization can temporarily exceed the
        # budget, so release them here and rely on the caller's VRAM reserve.
        for victim in candidates:
            self.used_bytes -= self.resident[victim][1]
            del self.resident[victim]
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
        self._usage.clear()
        self._step = 0
        self.used_bytes = 0
        self.counters.clear()

    def statistics(self):
        return {
            **dict(self.counters),
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
    ):
        if int(cpu_workers) != cpu_workers or cpu_workers < 0:
            raise ValueError("cpu_workers must be a nonnegative integer")
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if int(d_model) < 1 or int(d_ff) < 1:
            raise ValueError("d_model and d_ff must be positive")
        if compute_dtype not in {"float16", "float32"}:
            raise ValueError("compute_dtype must be 'float16' or 'float32'")
        if cpu_packed_backend not in {"reference", "auto", "native"}:
            raise ValueError("cpu_packed_backend must be 'reference', 'auto' or 'native'")
        self.cpu_packed_backend = cpu_packed_backend
        self._dispatch_depth = 0
        self.host_provider = host_provider
        self.model_id = model_id
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.compute_dtype = compute_dtype
        self.cache = AdaptiveExpertCache(
            device_budget_bytes, promote_after=promote_after,
            score_half_life=score_half_life,
            idle_eviction_requests=idle_eviction_requests,
        )
        self.host_cache = CPUDequantizedExpertCache(host_dequant_budget_bytes)
        self.cpu_workers = int(cpu_workers)
        self._pool = (ThreadPoolExecutor(max_workers=self.cpu_workers,
                                         thread_name_prefix="mylocal-moe-cpu")
                      if self.cpu_workers else None)
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

    def statistics(self):
        return {
            "device": self.cache.statistics(),
            "host_dequant": self.host_cache.statistics(),
            "execution": dict(self.counters),
            "cpu_workers": self.cpu_workers,
            "cpu_packed_backend": self.cpu_packed_backend,
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

    def _device_expert(self, weights):
        gate, up, down = weights
        if isinstance(gate, PackedGGUFWeight):
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

    def _cpu_forward_numpy(self, key, weights, x_host):
        """Execute a selected expert entirely on host; never call CuPy here."""
        dtype = np.float64 if x_host.dtype == np.float64 else np.float32
        x_cpu = np.asarray(x_host, dtype=dtype)
        gate, up, down = weights
        if isinstance(gate, PackedGGUFWeight):
            if self.cpu_packed_backend != "reference":
                native = packed_cpu_expert(x_cpu, weights, backend=self.cpu_packed_backend)
                if native is not None:
                    self.counters["native_cpu_experts"] += 1
                    return native.astype(x_host.dtype, copy=False)
                self.counters["native_cpu_fallbacks"] += 1
            if self.host_cache.budget_bytes:
                gmat, umat, dmat = self.host_cache.get(key, weights)
            else:
                gmat, umat, dmat = (w.dequantize_reference() for w in weights)
            g = x_cpu @ gmat.T
            u = x_cpu @ umat.T
        else:
            g = x_cpu @ np.asarray(gate, dtype=dtype)
            u = x_cpu @ np.asarray(up, dtype=dtype)
        h = (g / (1.0 + np.exp(-g))) * u
        if isinstance(down, PackedGGUFWeight):
            result = h @ dmat.T
        else:
            result = h @ np.asarray(down, dtype=dtype)
        return np.asarray(result, dtype=x_host.dtype)

    def _cpu_forward(self, key, weights, x):
        """Synchronous reference route, compatible with the original API."""
        return xp.asarray(self._cpu_forward_numpy(key, weights, asnumpy(x)),
                          dtype=x.dtype)

    def _copy_compute_copy(self, key, weights, x, ready_event, device_id):
        """One CuPy worker: staged D2H, host GEMV, staged H2D.

        The input-ready event handles producer-stream ordering (including
        the routing gather). A dedicated nonblocking stream per worker keeps
        copies independent of main-stream GPU expert kernels. Both copies
        complete before the Future resolves: output lifetime and main-stream
        consumption therefore require no cross-stream synchronization and
        pinned buffers cannot be recycled while DMA is outstanding.
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
            x.get(out=host_x, stream=stream, blocking=False)
            stream.synchronize()  # CPU must not read before D2H completes.
            host_y = self._cpu_forward_numpy(key, weights, host_x)
            out_mem = xp.cuda.alloc_pinned_memory(int(host_y.nbytes))
            host_out = np.frombuffer(out_mem, dtype=host_y.dtype,
                                     count=int(host_y.size)).reshape(host_y.shape)
            np.copyto(host_out, host_y)
            gpu_y = xp.empty(host_y.shape, dtype=x.dtype)
            gpu_y.set(host_out, stream=stream)
            stream.synchronize()  # Guarantees host buffer reuse and visibility.
            return gpu_y

    def _select(self, layer_id, expert_id, x):
        if x.shape[-1] != self.d_model:
            raise ValueError("Expert input has incorrect hidden width")
        key = ExpertKey(self.model_id, int(layer_id), int(expert_id))
        weights = None
        if BACKEND_NAME == "cupy" and self.cache.device_budget_bytes:
            if key in self.cache.resident:
                size = self.cache.resident[key][1]
                expert = self.cache.request(key, size, lambda: None)
                return key, expert, None
            weights = self._host_weights(key)
            if isinstance(weights[0], PackedGGUFWeight):
                size = sum(w.nbytes for w in weights)
            else:
                size = sum(w.size for w in weights) * np.dtype(self.compute_dtype).itemsize
            expert = self.cache.request(key, size, lambda: self._device_expert(weights))
            if expert is not None:
                return key, expert, None
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
        key, expert, weights = self._select(layer_id, expert_id, x)
        if expert is not None:
            self.counters["gpu_experts"] += 1
            output = expert.forward(x)
            self._record_direct_gpu_use(key)
            return output
        self.counters["cpu_submitted"] += 1
        if BACKEND_NAME == "cupy":
            # Record after the gather that produced this expert's activations;
            # a worker stream must wait for this event before reading x.
            device_id = int(xp.cuda.runtime.getDevice())
            ready = xp.cuda.Event(disable_timing=True)
            ready.record(xp.cuda.get_current_stream())
            return self._pool.submit(self._copy_compute_copy, key, weights,
                                     x, ready, device_id)
        return self._pool.submit(self._cpu_forward_numpy, key, weights, x)

    @staticmethod
    def resolve(work):
        """Collect one submitted expert result, propagating any worker error."""
        return work.result() if isinstance(work, Future) else work

    def forward(self, layer_id: int, expert_id: int, x):
        """Synchronous entry point retained for existing callers."""
        key, expert, weights = self._select(layer_id, expert_id, x)
        if expert is not None:
            output = expert.forward(x)
            self._record_direct_gpu_use(key)
            return output
        return self._cpu_forward(key, weights, x)
