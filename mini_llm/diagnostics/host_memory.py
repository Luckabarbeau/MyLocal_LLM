"""Read-only Linux host-memory observations and a SOFT budget guard.

This is not an OS limit: samples are non-atomic and cannot prevent another
application's sudden allocation, or an allocation between checks. Callers must
check before loading and at bounded inference boundaries. RSS includes resident
file-backed mmap pages, not just anonymous allocations. No RLIMIT_AS is set:
full GGUF mappings can legitimately have much larger virtual address spaces.
Nothing here loads a backend/model, clears caches, or changes swap settings.

This file can be loaded with importlib.util.spec_from_file_location to avoid the
legacy mini_llm package initializers (which import numerical backends).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
import re


GiB = 1024**3


def _unsigned(text: str) -> int | None:
    text = text.strip()
    return int(text) if text and text.isascii() and text.isdecimal() else None


def _parse_proc_kib(text: str) -> dict[str, int | None]:
    """Parse proc's explicitly labelled kB (1024-byte) fields, not guesses."""
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        parts = value.split()
        count = _unsigned(parts[0]) if len(parts) == 2 and parts[1] == "kB" else None
        result[key.strip()] = None if count is None else count * 1024
    return result


def _absolute_path(text: str) -> PurePosixPath | None:
    if not text.startswith("/") or any(p in (".", "..") for p in text.split("/")):
        return None
    return PurePosixPath(text)


def _mount_unescape(text: str) -> str:
    return re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), text)


def _resolve_cgroup_v2(cgroup_text: str, mountinfo_text: str) -> tuple[Path | None, str]:
    """Map an absolute unified membership through an actual cgroup2 mount.

    Reject relative/traversing membership, including namespace paths containing
    '..'. Never substitute the mount root for an unresolvable process leaf.
    Only the leaf is observed; ancestor limits are not traversed or inferred.
    """
    memberships = [line[3:] for line in cgroup_text.splitlines() if line.startswith("0::")]
    if len(memberships) != 1:
        return None, "host-only: no unique cgroup-v2 membership"
    member = _absolute_path(memberships[0])
    if member is None:
        return None, "host-only: invalid cgroup-v2 membership path"
    candidates = []
    for line in mountinfo_text.splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        if not separator or not after.split() or after.split()[0] != "cgroup2" or len(fields) < 6:
            continue
        root = _absolute_path(_mount_unescape(fields[3]))
        mount = _absolute_path(_mount_unescape(fields[4]))
        if root is None or mount is None:
            continue
        try:
            relative = member.relative_to(root)
        except ValueError:
            continue
        candidates.append((len(root.parts), Path(str(mount / relative))))
    if not candidates:
        return None, "host-only: cgroup-v2 leaf not resolvable from mountinfo"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = {path for depth, path in candidates if depth == candidates[0][0]}
    if len(best) != 1:
        return None, "host-only: ambiguous cgroup-v2 mounts"
    return best.pop(), "cgroup-v2 leaf detected (ancestor limits not sampled)"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return ""


def read_host_memory() -> dict[str, int | str | None]:
    """Return JSON-safe byte counts; unavailable/unsupported counts are None.

    available_bytes is min(MemAvailable, max(0, memory.max-memory.current))
    for a detected finite cgroup-v2 leaf, otherwise MemAvailable. Swap is
    reported separately and never counted as available RAM. Missing MemAvailable
    is not approximated using MemFree. An unresolved leaf falls back to host
    observations with an explicit note. A detected but unreadable/malformed
    cgroup budget makes effective availability unknown (fail closed).
    """
    mem = _parse_proc_kib(_read_text(Path("/proc/meminfo")))
    status = _parse_proc_kib(_read_text(Path("/proc/self/status")))
    host_available = mem.get("MemAvailable")
    snapshot = {
        "available_bytes": host_available,
        "host_available_bytes": host_available,
        "rss_bytes": status.get("VmRSS"),
        "peak_rss_bytes": status.get("VmHWM"),
        "process_swap_bytes": status.get("VmSwap"),
        "swap_total_bytes": mem.get("SwapTotal"),
        "swap_free_bytes": mem.get("SwapFree"),
        "cgroup_limit_bytes": None,
        "cgroup_current_bytes": None,
        "available_source": "host MemAvailable" if host_available is not None else "unknown",
    }
    leaf, note = _resolve_cgroup_v2(
        _read_text(Path("/proc/self/cgroup")),
        _read_text(Path("/proc/self/mountinfo")),
    )
    snapshot["cgroup_note"] = note
    if leaf is not None:
        maximum = _read_text(leaf / "memory.max").strip()
        limit = _unsigned(maximum)
        current = _unsigned(_read_text(leaf / "memory.current"))
        snapshot["cgroup_limit_bytes"] = limit
        snapshot["cgroup_current_bytes"] = current
        if maximum == "max":
            snapshot["cgroup_note"] += "; unlimited memory.max"
        elif limit is not None and current is not None and host_available is not None:
            snapshot["available_bytes"] = min(host_available, max(0, limit - current))
            snapshot["available_source"] = "min(host MemAvailable, cgroup-v2 leaf remaining)"
        else:
            snapshot["available_bytes"] = None
            snapshot["available_source"] = "unknown"
            snapshot["cgroup_note"] += "; incomplete or invalid memory budget"
    return snapshot


class MemoryBudgetError(MemoryError):
    """A soft memory observation is missing or violates the requested budget."""


class HostMemoryGuard:
    """Observe budgets without enforcing OS limits or retaining sample history.

    Both modes require known nonnegative integer available_bytes and rss_bytes,
    and reject current RSS above max_rss_bytes. If peak_rss_bytes is reported,
    it must also be valid and <= max_rss_bytes. Thus an earlier process-lifetime
    RSS peak remains a violation even after memory is released; a missing peak
    is permitted because current RSS is the mandatory observation.

    Normal checks require available >= reserve. Loading preflight additionally
    reserves max(0, max_rss-current_rss) for planned RSS growth. Equality passes.
    last_snapshot holds the latest returned sample, including rejected samples;
    a sampler exception leaves the previous snapshot intact. No history grows.
    """

    def __init__(
        self,
        reserve_bytes: int = 16 * GiB,
        max_rss_bytes: int = 12 * GiB,
        sample_fn: Callable = read_host_memory,
    ):
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a nonnegative integer byte count")
        if type(max_rss_bytes) is not int or max_rss_bytes <= 0:
            raise ValueError("max_rss_bytes must be a positive integer byte count")
        if not callable(sample_fn):
            raise ValueError("sample_fn must be callable")
        self.reserve_bytes = reserve_bytes
        self.max_rss_bytes = max_rss_bytes
        self.sample_fn = sample_fn
        self.last_snapshot = None

    def check(self, loading: bool = False) -> dict:
        """Return the observation or raise MemoryBudgetError with its reason."""
        if type(loading) is not bool:
            raise ValueError("loading must be a bool")
        try:
            snapshot = self.sample_fn()
        except Exception as exc:
            raise MemoryBudgetError(f"host memory sampling failed: {exc}") from exc
        self.last_snapshot = snapshot
        if not isinstance(snapshot, dict):
            raise MemoryBudgetError("host memory sample must be a dict")
        for key in ("available_bytes", "rss_bytes"):
            value = snapshot.get(key)
            if type(value) is not int or value < 0:
                raise MemoryBudgetError(f"mandatory {key} is missing or invalid: {value!r}")
        rss = snapshot["rss_bytes"]
        peak = snapshot.get("peak_rss_bytes")
        if peak is not None and (type(peak) is not int or peak < 0):
            raise MemoryBudgetError(f"peak_rss_bytes is invalid: {peak!r}")
        if rss > self.max_rss_bytes:
            raise MemoryBudgetError(f"RSS {rss} bytes exceeds max_rss_bytes={self.max_rss_bytes}")
        if peak is not None and peak > self.max_rss_bytes:
            raise MemoryBudgetError(
                f"process-lifetime peak RSS {peak} bytes exceeds max_rss_bytes={self.max_rss_bytes}"
            )
        headroom = max(0, self.max_rss_bytes - rss) if loading else 0
        required = self.reserve_bytes + headroom
        if snapshot["available_bytes"] < required:
            raise MemoryBudgetError(
                f"available memory {snapshot['available_bytes']} bytes below required {required} "
                f"bytes (reserve={self.reserve_bytes}, planned RSS headroom={headroom}, "
                f"loading={loading})"
            )
        return snapshot
