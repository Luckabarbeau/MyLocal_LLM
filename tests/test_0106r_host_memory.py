"""CPU-only, text-injected observations; no package/backend or model imports."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "mini_llm/diagnostics/host_memory.py"
SPEC = importlib.util.spec_from_file_location("host_memory_0106r", MODULE_PATH)
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)

GiB = 1024**3
BYTE_KEYS = {
    "available_bytes", "host_available_bytes", "rss_bytes", "peak_rss_bytes",
    "process_swap_bytes", "swap_total_bytes", "swap_free_bytes",
    "cgroup_limit_bytes", "cgroup_current_bytes",
}
MOUNTINFO = "36 25 0:32 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n"


def sample(available=32 * GiB, rss=2 * GiB, peak=None):
    return {"available_bytes": available, "rss_bytes": rss, "peak_rss_bytes": peak}


def fake_files(monkeypatch, *, maximum="max", current="1024", membership="0::/jobs/worker\n",
               mountinfo=MOUNTINFO, meminfo=None, status=None):
    files = {
        "/proc/meminfo": meminfo if meminfo is not None else (
            "MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n"
            "SwapTotal: 8388608 kB\nSwapFree: 7340032 kB\n"
        ),
        "/proc/self/status": status if status is not None else (
            "Name:\tpython\nVmRSS:\t2097152 kB\nVmHWM: 3145728 kB\nVmSwap: 64 kB\n"
        ),
        "/proc/self/cgroup": membership,
        "/proc/self/mountinfo": mountinfo,
        "/sys/fs/cgroup/jobs/worker/memory.max": maximum,
        "/sys/fs/cgroup/jobs/worker/memory.current": current,
    }
    reads = []

    def read(path):
        reads.append(str(path))
        return files.get(str(path), "")

    monkeypatch.setattr(memory, "_read_text", read)
    return files, reads


def test_standalone_import_does_not_load_backend():
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1]); "
        "assert not any(k.split('.')[0] in "
        "{'mini_llm', 'numpy', 'cupy', 'torch', 'resource'} for k in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(MODULE_PATH)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_proc_parser_units_whitespace_and_missing_fields():
    parsed = memory._parse_proc_kib(
        "MemAvailable:\t123 kB\nVmRSS: 0 kB\nVmHWM: 999 kB\n"
        "Name: python\nno colon here\nSwapFree: -1 kB\n"
        "SwapTotal: 123 MB\nVmSwap: 2\nBad: nan kB\nFloat: 1.5 kB\n"
    )
    assert parsed["MemAvailable"] == 123 * 1024
    assert parsed["VmRSS"] == 0
    assert parsed["VmHWM"] == 999 * 1024
    assert "MemFree" not in parsed
    for key in ("Name", "SwapFree", "SwapTotal", "VmSwap", "Bad", "Float"):
        assert parsed[key] is None


@pytest.mark.parametrize("text, expected", [
    ("123\n", 123), ("0", 0), ("max\n", None), ("", None),
    ("-1", None), ("1.0", None), ("inf", None), ("١", None),
])
def test_cgroup_value_parser(text, expected):
    assert memory._unsigned(text) == expected


def test_finite_cgroup_effective_availability_and_json(monkeypatch):
    fake_files(monkeypatch, maximum=str(10 * GiB), current=str(3 * GiB))
    observed = memory.read_host_memory()
    assert BYTE_KEYS <= observed.keys()
    assert observed["available_bytes"] == 7 * GiB
    assert observed["host_available_bytes"] == 32 * GiB
    assert observed["rss_bytes"] == 2 * GiB
    assert observed["peak_rss_bytes"] == 3 * GiB
    assert observed["process_swap_bytes"] == 64 * 1024
    assert observed["swap_total_bytes"] == 8 * GiB
    assert observed["swap_free_bytes"] == 7 * GiB
    assert observed["cgroup_limit_bytes"] == 10 * GiB
    assert observed["cgroup_current_bytes"] == 3 * GiB
    assert "min(" in observed["available_source"]
    assert json.loads(json.dumps(observed, allow_nan=False)) == observed
    assert all(observed[key] is None or type(observed[key]) is int for key in BYTE_KEYS)


@pytest.mark.parametrize("maximum,current,expected", [
    (64 * GiB, GiB, 32 * GiB), (GiB, 2 * GiB, 0), (0, 0, 0),
])
def test_finite_cgroup_min_and_saturated_remaining(monkeypatch, maximum, current, expected):
    fake_files(monkeypatch, maximum=str(maximum), current=str(current))
    assert memory.read_host_memory()["available_bytes"] == expected


def test_unlimited_cgroup_does_not_count_swap_as_ram(monkeypatch):
    fake_files(monkeypatch)
    observed = memory.read_host_memory()
    assert observed["available_bytes"] == 32 * GiB
    assert observed["cgroup_limit_bytes"] is None
    assert observed["cgroup_current_bytes"] == 1024
    assert "unlimited" in observed["cgroup_note"]


@pytest.mark.parametrize("maximum,current", [("", "1"), ("bad", "1"), ("123", ""), ("123", "-1")])
def test_detected_incomplete_cgroup_budget_is_unknown(monkeypatch, maximum, current):
    fake_files(monkeypatch, maximum=maximum, current=current)
    observed = memory.read_host_memory()
    assert observed["available_bytes"] is None
    assert "incomplete" in observed["cgroup_note"]
    with pytest.raises(memory.MemoryBudgetError, match="available_bytes"):
        memory.HostMemoryGuard(sample_fn=lambda: observed).check()


@pytest.mark.parametrize("membership", [
    "", "5:memory:/legacy\n", "0::relative\n", "0::/../worker\n",
    "0::/../../docker/worker\n", "0::/jobs/./worker\n", "0::/a\n0::/b\n",
])
def test_invalid_or_missing_v2_leaf_never_guesses_mount_root(monkeypatch, membership):
    files, reads = fake_files(monkeypatch, membership=membership)
    files["/sys/fs/cgroup/memory.max"] = "1"
    files["/sys/fs/cgroup/memory.current"] = "1"
    observed = memory.read_host_memory()
    assert observed["available_bytes"] == 32 * GiB
    assert observed["cgroup_limit_bytes"] is None
    assert "host-only" in observed["cgroup_note"]
    assert not any(path.startswith("/sys/") for path in reads)


def test_mount_root_mapping_and_escaped_mountpoint():
    mounts = "36 25 0:32 /jobs /custom\\040mount rw - cgroup2 cgroup rw\n"
    leaf, note = memory._resolve_cgroup_v2("0::/jobs/worker\n", mounts)
    assert leaf == Path("/custom mount/worker")
    assert "detected" in note
    leaf, note = memory._resolve_cgroup_v2("0::/other/worker\n", mounts)
    assert leaf is None
    assert "host-only" in note
    # A namespace-relative root must not be assumed to be this mount's root.
    assert memory._resolve_cgroup_v2("0::/\n", mounts)[0] is None
    assert memory._resolve_cgroup_v2("0::/\n", MOUNTINFO)[0] == Path("/sys/fs/cgroup")


@pytest.mark.parametrize("mountinfo", [
    "", MOUNTINFO.replace(" - cgroup2 ", " - cgroup "), "malformed mountinfo",
    MOUNTINFO + MOUNTINFO.replace("/sys/fs/cgroup", "/other/cgroup"),
])
def test_missing_wrong_or_ambiguous_mounts_fall_back(monkeypatch, mountinfo):
    _, reads = fake_files(monkeypatch, mountinfo=mountinfo)
    observed = memory.read_host_memory()
    assert observed["available_bytes"] == 32 * GiB
    assert "host-only" in observed["cgroup_note"]
    assert not any(path.startswith("/sys/") for path in reads)


def test_missing_proc_values_are_none_not_free_memory_guesses(monkeypatch):
    fake_files(monkeypatch, meminfo="MemFree: 999999999 kB\n", status="Name: python\n")
    observed = memory.read_host_memory()
    assert all(observed[key] is None for key in BYTE_KEYS - {"cgroup_current_bytes"})
    assert observed["cgroup_current_bytes"] == 1024


@pytest.mark.parametrize("exception", [FileNotFoundError(), PermissionError(), UnicodeError()])
def test_unreadable_proc_text_is_supported(monkeypatch, exception):
    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(Path, "read_text", fail)
    observed = memory.read_host_memory()
    assert all(observed[key] is None for key in BYTE_KEYS)
    assert "host-only" in observed["cgroup_note"]


def test_defaults_and_boundary_equality():
    observed = sample(available=16 * GiB, rss=12 * GiB, peak=12 * GiB)
    guard = memory.HostMemoryGuard(sample_fn=lambda: observed)
    assert guard.reserve_bytes == 16 * GiB
    assert guard.max_rss_bytes == 12 * GiB
    assert guard.last_snapshot is None
    assert guard.check() is observed
    assert guard.check(loading=True) is observed
    assert guard.last_snapshot is observed


def test_loading_preflight_requires_planned_headroom():
    observed = sample(available=25 * GiB, rss=2 * GiB)
    guard = memory.HostMemoryGuard(sample_fn=lambda: observed)
    assert guard.check() is observed
    with pytest.raises(memory.MemoryBudgetError, match="planned RSS headroom=10737418240"):
        guard.check(loading=True)
    assert guard.last_snapshot is observed
    observed["available_bytes"] = 26 * GiB
    assert guard.check(loading=True) is observed


@pytest.mark.parametrize("loading", [False, True])
@pytest.mark.parametrize("observed,reason", [
    (sample(available=16 * GiB - 1), "available memory"),
    (sample(rss=12 * GiB + 1), "RSS"),
    (sample(rss=GiB, peak=12 * GiB + 1), "process-lifetime peak RSS"),
])
def test_over_budget_samples_fail_and_are_retained(loading, observed, reason):
    guard = memory.HostMemoryGuard(sample_fn=lambda: observed)
    with pytest.raises(memory.MemoryBudgetError, match=reason) as failure:
        guard.check(loading=loading)
    assert isinstance(failure.value, MemoryError)
    assert guard.last_snapshot is observed


@pytest.mark.parametrize("key", ["available_bytes", "rss_bytes"])
@pytest.mark.parametrize("value", [None, -1, True, 1.5, float("inf"), float("nan"), "999999999999"])
def test_mandatory_unknown_or_invalid_values_fail_closed(key, value):
    observed = sample()
    observed[key] = value
    guard = memory.HostMemoryGuard(sample_fn=lambda: observed)
    with pytest.raises(memory.MemoryBudgetError, match=key):
        guard.check()
    assert guard.last_snapshot is observed


@pytest.mark.parametrize("key", ["available_bytes", "rss_bytes"])
def test_absent_mandatory_field_fails_closed(key):
    observed = sample()
    del observed[key]
    with pytest.raises(memory.MemoryBudgetError, match=key):
        memory.HostMemoryGuard(sample_fn=lambda: observed).check()


@pytest.mark.parametrize("peak", [-1, False, "123", float("nan")])
def test_reported_invalid_peak_fails_closed(peak):
    with pytest.raises(memory.MemoryBudgetError, match="peak_rss_bytes"):
        memory.HostMemoryGuard(sample_fn=lambda: sample(peak=peak)).check()


@pytest.mark.parametrize("kwargs", [
    {"reserve_bytes": -1}, {"reserve_bytes": True}, {"reserve_bytes": 1.0},
    {"reserve_bytes": None}, {"max_rss_bytes": 0}, {"max_rss_bytes": -1},
    {"max_rss_bytes": False}, {"max_rss_bytes": float("inf")}, {"sample_fn": None},
])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        memory.HostMemoryGuard(**kwargs)


def test_zero_reserve_and_missing_optional_peak_are_supported():
    observed = {"available_bytes": 0, "rss_bytes": 1}
    assert memory.HostMemoryGuard(0, 1, lambda: observed).check(loading=True) is observed


def test_only_latest_snapshot_is_retained():
    observations = iter([sample(), sample(available=0)])
    guard = memory.HostMemoryGuard(sample_fn=lambda: next(observations))
    first = guard.check()
    with pytest.raises(memory.MemoryBudgetError):
        guard.check()
    assert guard.last_snapshot is not first
    assert guard.last_snapshot["available_bytes"] == 0
    assert set(vars(guard)) == {"reserve_bytes", "max_rss_bytes", "sample_fn", "last_snapshot"}


def test_sampler_failure_is_descriptive_and_preserves_last_observation():
    observed = sample()
    guard = memory.HostMemoryGuard(sample_fn=lambda: observed)
    guard.check()

    def fail():
        raise OSError("proc read denied")

    guard.sample_fn = fail
    with pytest.raises(memory.MemoryBudgetError, match="sampling failed: proc read denied"):
        guard.check()
    assert guard.last_snapshot is observed


def test_invalid_sample_and_loading_flag():
    guard = memory.HostMemoryGuard(sample_fn=lambda: None)
    with pytest.raises(memory.MemoryBudgetError, match="must be a dict"):
        guard.check()
    with pytest.raises(ValueError, match="loading must be a bool"):
        guard.check(loading="yes")
