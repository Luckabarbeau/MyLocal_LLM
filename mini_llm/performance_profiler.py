"""Opt-in coarse performance profiling for training hot paths.

The profiler is intentionally disabled by default.  When enabled it brackets
coarse operations with backend synchronization so the reported times are useful
for locating bottlenecks on asynchronous GPU backends.  Because those
synchronizations perturb throughput, profiling should be limited to a few
optimizer steps.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager

from .backend import BACKEND_NAME, synchronize, xp


class _PerformanceProfiler:
    def __init__(self):
        self.enabled = False
        self._records = OrderedDict()

    def configure(self, enabled: bool, reset: bool = False):
        self.enabled = bool(enabled)
        if reset:
            self.reset()

    def reset(self):
        self._records = OrderedDict()

    @contextmanager
    def scope(self, name: str):
        if not self.enabled:
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
            total, count = self._records.get(name, (0.0, 0))
            self._records[name] = (total + elapsed, count + 1)

    def rows(self):
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


def performance_report(title: str = "Performance profile") -> str:
    return PERFORMANCE_PROFILER.format_report(title)
