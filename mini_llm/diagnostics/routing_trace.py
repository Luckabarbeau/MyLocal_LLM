"""0106I opt-in observations for managed inference, never routing decisions.

Summary mode reuses RoutingPlan's existing host counts. Detailed mode copies
routing IDs/weights (and can synchronize), retains bounded records, and records
CUDA event intervals without adding device synchronizations. Event intervals
are stream elapsed times, not isolated kernel or PCIe bandwidth measurements.
"""

from collections import Counter, deque
from contextlib import contextmanager
import json
import threading
import time

from mini_llm.backend import BACKEND_NAME, asnumpy, xp


class RoutingTrace:
    def __init__(self, *, model_id, quantization, expert_mapping, workload,
                 mode="summary", max_records=1024, max_pending_events=4096):
        identity = dict(model_id=model_id, quantization=quantization,
                        expert_mapping=expert_mapping, workload=workload)
        if any(not isinstance(v, str) or not v.strip() for v in identity.values()):
            raise ValueError("Trace identity requires nonempty model/quantization/mapping/workload")
        if mode not in {"summary", "detailed"}:
            raise ValueError("mode must be summary or detailed")
        for value in (max_records, max_pending_events):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("Trace limits must be positive integers")
        self.identity = identity
        self.mode = mode
        self.max_pending_events = int(max_pending_events)
        self._routes = deque(maxlen=int(max_records))
        self._executions = deque(maxlen=int(max_records))
        self._pending = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._dispatch = 0
        self._tokens = Counter()
        self._frequencies = Counter()
        self._counts = Counter()
        self._timings = Counter()

    @contextmanager
    def layer_scope(self, layer_id, plan, expert_ids, weights):
        """Observe the existing plan; never rebuild it or change its ordering."""
        layer_id = int(layer_id)
        start = time.perf_counter()
        ids = mixes = None
        if self.mode == "detailed":
            ids = asnumpy(expert_ids).reshape(plan.N, plan.k).tolist()
            mixes = asnumpy(weights).reshape(plan.N, plan.k).tolist()
        with self._lock:
            dispatch = self._dispatch
            self._dispatch += 1
            token_start = self._tokens[layer_id]
            self._tokens[layer_id] += plan.N
            self._counts["layer_dispatches"] += 1
            self._counts["routed_assignments"] += plan.N * plan.k
            for expert in plan.active_expert_ids():
                self._frequencies[(layer_id, int(expert))] += plan.get_assignment_count(expert)
            if ids is not None:
                if len(self._routes) == self._routes.maxlen:
                    self._counts["dropped_route_records"] += 1
                self._routes.append(dict(dispatch_id=dispatch, layer_id=layer_id,
                                         token_start=token_start, expert_ids=ids,
                                         routing_weights=mixes))
            self._timings["routing_observation_host_ms"] += (time.perf_counter() - start) * 1000
        previous = getattr(self._local, "dispatch", None)
        self._local.dispatch = dispatch
        try:
            yield
        finally:
            self._local.dispatch = previous

    def new_expert(self, layer_id, expert_id, rows):
        record = dict(dispatch_id=getattr(self._local, "dispatch", None),
                      layer_id=int(layer_id), expert_id=int(expert_id), rows=int(rows),
                      path=None, cache_hit=None, cache_miss=None, promoted=False,
                      evicted_experts=[], transferred_bytes=0,
                      activation_d2h_bytes=0, activation_h2d_bytes=0, timings_ms={})
        with self._lock:
            if self.mode == "detailed":
                if len(self._executions) == self._executions.maxlen:
                    self._counts["dropped_execution_records"] += 1
                self._executions.append(record)
        return _ExpertTrace(self, record)

    @contextmanager
    def wait_scope(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._time({"timings_ms": {}}, name + "_host_ms",
                       (time.perf_counter() - start) * 1000)

    def _time(self, record, name, milliseconds):
        with self._lock:
            record["timings_ms"][name] = record["timings_ms"].get(name, 0.0) + milliseconds
            self._timings[name] += milliseconds

    def _enqueue(self, record, name, start, end):
        with self._lock:
            if len(self._pending) >= self.max_pending_events:
                self._counts["dropped_gpu_timers"] += 1
            else:
                self._pending.append((record, name, start, end))

    def snapshot(self):
        """Poll completed events only; pending measurements are not zero times.

        Drain CPU work and, outside the measured region, synchronize the caller
        stream before this call when a complete benchmark report is required.
        """
        with self._lock:
            waiting = []
            for record, name, start, end in self._pending:
                if not end.done:
                    waiting.append((record, name, start, end))
                    continue
                elapsed = float(xp.cuda.get_elapsed_time(start, end))
                record["timings_ms"][name] = record["timings_ms"].get(name, 0.0) + elapsed
                self._timings[name] += elapsed
            self._pending = waiting
            return dict(
                schema_version=1, identity=dict(self.identity), mode=self.mode,
                counters=dict(self._counts), timings_ms=dict(self._timings),
                pending_gpu_timers=len(self._pending),
                tokens_by_layer={str(k): v for k, v in sorted(self._tokens.items())},
                frequencies=[dict(layer_id=k[0], expert_id=k[1], selections=v)
                             for k, v in sorted(self._frequencies.items())],
                routes=[dict(r, expert_ids=[list(row) for row in r["expert_ids"]],
                             routing_weights=[list(row) for row in r["routing_weights"]])
                        for r in self._routes],
                executions=[dict(r, timings_ms=dict(r["timings_ms"]),
                                 evicted_experts=[dict(v) for v in r["evicted_experts"]])
                            for r in self._executions],
                timing_semantics={
                    "host": "Host wall intervals; concurrent workers overlap, do not sum as layer latency",
                    "gpu": "Detailed-only CUDA stream event intervals, polled without synchronization",
                    "promotion": "Includes upload/conversion enqueue interval, not isolated DMA bandwidth",
                    "wait": "Host dependency waits, not all device synchronization overhead",
                },
            )

    def write_json(self, path):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.snapshot(), stream, indent=2, allow_nan=False)
            stream.write("\n")


class _ExpertTrace:
    def __init__(self, owner, record):
        self.owner = owner
        self.record = record

    @contextmanager
    def span(self, name, *, gpu=False):
        start_event = None
        if gpu and self.owner.mode == "detailed" and BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            start_event.record(xp.cuda.get_current_stream())
        start = time.perf_counter()
        try:
            yield
        finally:
            self.owner._time(self.record, name + "_host_ms", (time.perf_counter() - start) * 1000)
            if start_event is not None:
                end = xp.cuda.Event()
                end.record(xp.cuda.get_current_stream())
                self.owner._enqueue(self.record, name + "_gpu_ms", start_event, end)

    def selected(self, key, expert, cache, before, resident_before, *, cache_enabled):
        """Called on the dispatcher thread, never from CPU workers."""
        hit = key in resident_before if cache_enabled else None
        promoted = cache.counters["promotions"] > before.get("promotions", 0)
        evicted = [dict(layer_id=k.layer_id, expert_id=k.expert_id)
                   for k in resident_before if k not in cache.resident]
        path = "cpu" if expert is None else ("gpu_hit" if hit else "gpu_promote")
        transferred = cache.resident[key][1] if promoted else 0
        with self.owner._lock:
            self.record.update(path=path, cache_hit=hit,
                               cache_miss=(not hit) if cache_enabled else None,
                               promoted=promoted, evicted_experts=evicted,
                               transferred_bytes=transferred)
            counts = self.owner._counts
            counts[path + "_experts"] += 1
            counts[path + "_assignments"] += self.record["rows"]
            counts["cache_requests"] += int(cache_enabled)
            counts["cache_hits"] += int(hit is True)
            counts["cache_misses"] += int(cache_enabled and not hit)
            counts["promoted_experts"] += int(promoted)
            counts["evicted_experts"] += len(evicted)
            counts["transferred_bytes"] += transferred

    def activation_bytes(self, direction, count):
        name = "activation_" + direction + "_bytes"
        with self.owner._lock:
            self.record[name] += int(count)
            self.owner._counts[name] += int(count)
