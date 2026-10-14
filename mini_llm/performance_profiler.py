"""Opt-in coarse performance profiling for training hot paths.

The profiler is intentionally disabled by default.  When enabled it brackets
coarse operations with backend synchronization so the reported times are useful
for locating bottlenecks on asynchronous GPU backends.  Because those
synchronizations perturb throughput, profiling should be limited to a few
optimizer steps.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

from .backend import BACKEND_NAME, synchronize, xp

# The ordinary per-kernel profiler synchronizes at every boundary.  Suppress
# these nested scopes *only inside* the detailed decoder-group interval so
# hundreds of groups remain asynchronous. The enclosing main Transformer
# still uses the original synchronized instrumentation.
_SUPPRESS_COARSE_SCOPE = ContextVar('block_proposal_suppress_coarse_scope', default=False)


class _PerformanceProfiler:
    def __init__(self):
        self.enabled = False
        self._records = OrderedDict()
        self._pending_gpu_events = []
        raw_detail = os.environ.get("MINI_LLM_MOE_DETAIL_PROFILE", "0").strip().lower()
        self.moe_detail_enabled = raw_detail not in {"0", "false", "off", "no", ""}
        raw_local_detail = os.environ.get(
            "MINI_LLM_LOCAL_DETAIL_PROFILE", "0"
        ).strip().lower()
        self.local_detail_enabled = raw_local_detail not in {
            "0", "false", "off", "no", ""
        }
        raw_retrieval_detail = os.environ.get(
            "MINI_LLM_RETRIEVAL_ROUTER_DETAIL_PROFILE", "0"
        ).strip().lower()
        self.retrieval_router_detail_enabled = raw_retrieval_detail not in {
            "0", "false", "off", "no", ""
        }
        raw_block_detail = os.environ.get(
            "MINI_LLM_BLOCK_PROPOSAL_DETAIL_PROFILE", "0"
        ).strip().lower()
        self.block_proposal_detail_enabled = raw_block_detail not in {
            "0", "false", "off", "no", ""
        }
        raw_rmsnorm_detail = os.environ.get(
            "MINI_LLM_RMSNORM_DETAIL_PROFILE", "0"
        ).strip().lower()
        self.rmsnorm_detail_enabled = raw_rmsnorm_detail not in {
            "0", "false", "off", "no", ""
        }

    def configure(self, enabled: bool, reset: bool = False):
        self.enabled = bool(enabled)
        if reset:
            self.reset()

    def reset(self):
        self._records = OrderedDict()
        self._pending_gpu_events = []

    def _record(self, name: str, elapsed_s: float):
        total, count = self._records.get(name, (0.0, 0))
        self._records[name] = (total + float(elapsed_s), count + 1)

    @contextmanager
    def scope(self, name: str):
        if not self.enabled or _SUPPRESS_COARSE_SCOPE.get():
            yield
            return

        # GPU work is asynchronous.  Profiling only a few explicitly requested
        # steps lets us synchronize at scope boundaries in exchange for simple,
        # trustworthy timings without threading profiler state through the model.
        synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize()
            elapsed = time.perf_counter() - start
            self._record(name, elapsed)

    @contextmanager
    def moe_detail_scope(self, name: str):
        """Profile fine-grained MoE work without synchronizing each region.

        Two records are produced on CuPy:

        ``<name>.host`` measures Python/CuPy submission time.  It therefore
        highlights indexing, allocation, Python loops, and any implicit host
        synchronization.

        ``<name>.gpu`` uses CUDA events on the current stream and measures the
        device work queued by the region.  Events are resolved only when the
        report is requested, so hundreds of expert-level regions do not force
        a synchronization apiece.

        The scope is active only when both the normal performance profiler and
        ``MINI_LLM_MOE_DETAIL_PROFILE=1`` are enabled.
        """
        if not self.enabled or not self.moe_detail_enabled:
            yield
            return

        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            end_event = xp.cuda.Event()
            start_event.record()
        else:
            start_event = end_event = None

        try:
            yield
        finally:
            host_elapsed = time.perf_counter() - host_start
            self._record(f"{name}.host", host_elapsed)
            if BACKEND_NAME == "cupy":
                end_event.record()
                self._pending_gpu_events.append(
                    (f"{name}.gpu", start_event, end_event)
                )

    @contextmanager
    def block_proposal_scope(self, name: str):
        """Use low-overhead deferred CUDA timing when block detail is enabled.

        Normal --profile-steps retains historical synchronized timing.  The
        detailed mode deliberately avoids synchronizing at each of hundreds
        of small decoder groups; the already-existing main-trunk boundaries
        still establish synchronized, separate forward/backward totals.
        """
        if self.enabled and self.block_proposal_detail_enabled:
            token = _SUPPRESS_COARSE_SCOPE.set(True)
            try:
                with self.block_proposal_detail_scope(name):
                    yield
            finally:
                _SUPPRESS_COARSE_SCOPE.reset(token)
        else:
            with self.scope(name):
                yield

    @contextmanager
    def block_proposal_detail_scope(self, name: str):
        """Report .host and .gpu times without a per-group CUDA sync.

        CUDA events are resolved only at the end-of-step profiler report.
        The .gpu time is stream elapsed time, not isolated kernel occupancy;
        nested scopes overlap and should never be summed blindly.
        """
        if not self.enabled or not self.block_proposal_detail_enabled:
            yield
            return
        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            end_event = xp.cuda.Event()
            start_event.record()
        else:
            start_event = end_event = None
        try:
            yield
        finally:
            self._record(f"{name}.host", time.perf_counter() - host_start)
            if BACKEND_NAME == "cupy":
                end_event.record()
                self._pending_gpu_events.append((f"{name}.gpu", start_event, end_event))

    @contextmanager
    def local_detail_scope(self, name: str):
        """Profile fine-grained local-attention work with deferred CUDA events.

        Like :meth:`moe_detail_scope`, this records ``.host`` submission time
        and ``.gpu`` CUDA-event time without synchronizing every sub-operation.
        It is active only while the normal profiler is enabled and
        ``MINI_LLM_LOCAL_DETAIL_PROFILE=1``.
        """
        if not self.enabled or not self.local_detail_enabled:
            yield
            return

        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            end_event = xp.cuda.Event()
            start_event.record()
        else:
            start_event = end_event = None

        try:
            yield
        finally:
            host_elapsed = time.perf_counter() - host_start
            self._record(f"{name}.host", host_elapsed)
            if BACKEND_NAME == "cupy":
                end_event.record()
                self._pending_gpu_events.append(
                    (f"{name}.gpu", start_event, end_event)
                )

    @contextmanager
    def retrieval_router_detail_scope(self, name: str):
        """Profile retrieval-router sub-operations with deferred CUDA events.

        This mirrors the MoE/local detail profilers: ``.host`` measures Python
        and CuPy submission time while ``.gpu`` measures queued CUDA work.  CUDA
        events are resolved only when the profile report is requested, avoiding
        a synchronization at every small router operation.

        The scope is active only while the normal performance profiler is
        enabled and ``MINI_LLM_RETRIEVAL_ROUTER_DETAIL_PROFILE=1``.
        """
        if not self.enabled or not self.retrieval_router_detail_enabled:
            yield
            return

        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            end_event = xp.cuda.Event()
            start_event.record()
        else:
            start_event = end_event = None

        try:
            yield
        finally:
            host_elapsed = time.perf_counter() - host_start
            self._record(f"{name}.host", host_elapsed)
            if BACKEND_NAME == "cupy":
                end_event.record()
                self._pending_gpu_events.append(
                    (f"{name}.gpu", start_event, end_event)
                )

    @contextmanager
    def rmsnorm_detail_scope(self, name: str):
        """Profile fused RMSNorm sub-kernels with deferred CUDA events."""
        if not self.enabled or not self.rmsnorm_detail_enabled:
            yield
            return

        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start_event = xp.cuda.Event()
            end_event = xp.cuda.Event()
            start_event.record()
        else:
            start_event = end_event = None

        try:
            yield
        finally:
            host_elapsed = time.perf_counter() - host_start
            self._record(f"{name}.host", host_elapsed)
            if BACKEND_NAME == "cupy":
                end_event.record()
                self._pending_gpu_events.append(
                    (f"{name}.gpu", start_event, end_event)
                )

    def _resolve_pending_gpu_events(self):
        if not self._pending_gpu_events:
            return
        # One synchronization resolves every fine-grained event pair.  This is
        # deliberately deferred to reporting so detail profiling does not
        # serialize each expert operation.
        synchronize()
        pending = self._pending_gpu_events
        self._pending_gpu_events = []
        for name, start_event, end_event in pending:
            elapsed_ms = xp.cuda.get_elapsed_time(start_event, end_event)
            self._record(name, float(elapsed_ms) / 1000.0)

    def rows(self):
        self._resolve_pending_gpu_events()
        rows = []
        for name, (total, count) in self._records.items():
            rows.append(
                {
                    "name": name,
                    "total_s": float(total),
                    "count": int(count),
                    "avg_ms": 1000.0 * float(total) / max(1, int(count)),
                }
            )
        return sorted(rows, key=lambda row: row["total_s"], reverse=True)

    def gpu_memory(self):
        if BACKEND_NAME != "cupy":
            return None
        pool = xp.get_default_memory_pool()
        free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
        return {
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
        }

    def format_report(self, title: str = "Performance profile") -> str:
        rows = self.rows()
        lines = [title, "-" * len(title)]
        if not rows:
            lines.append("(no timed regions recorded)")
        else:
            lines.append(f"{'region':42s} {'total ms':>12s} {'calls':>8s} {'avg ms':>12s}")
            for row in rows:
                lines.append(
                    f"{row['name'][:42]:42s} "
                    f"{1000.0 * row['total_s']:12.3f} "
                    f"{row['count']:8d} "
                    f"{row['avg_ms']:12.3f}"
                )

        memory = self.gpu_memory()
        if memory is not None:
            gib = 1024.0 ** 3
            lines.append("")
            lines.append(
                "GPU memory after profiled step: "
                f"used={memory['pool_used_bytes'] / gib:.2f} GiB, "
                f"reserved={memory['pool_reserved_bytes'] / gib:.2f} GiB, "
                f"device_free={memory['device_free_bytes'] / gib:.2f} GiB"
            )
        return "\n".join(lines)


PERFORMANCE_PROFILER = _PerformanceProfiler()


def configure_performance_profiler(enabled: bool, reset: bool = False):
    PERFORMANCE_PROFILER.configure(enabled, reset=reset)


def reset_performance_profiler():
    PERFORMANCE_PROFILER.reset()


def performance_scope(name: str):
    return PERFORMANCE_PROFILER.scope(name)


def moe_detail_scope(name: str):
    return PERFORMANCE_PROFILER.moe_detail_scope(name)


def performance_report(title: str = "Performance profile") -> str:
    return PERFORMANCE_PROFILER.format_report(title)


def local_detail_scope(name: str):
    return PERFORMANCE_PROFILER.local_detail_scope(name)


def retrieval_router_detail_scope(name: str):
    return PERFORMANCE_PROFILER.retrieval_router_detail_scope(name)


def block_proposal_scope(name: str):
    """Coarse legacy timing, or deferred timing for selected decoder groups."""
    return PERFORMANCE_PROFILER.block_proposal_scope(name)


def block_proposal_detail_scope(name: str):
    """Opt-in deferred event/host timing for 0080 grouped computations."""
    return PERFORMANCE_PROFILER.block_proposal_detail_scope(name)
