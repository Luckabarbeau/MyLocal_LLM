import importlib.util
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


profile = _load("profile_0087a_training", "benchmarks/profile_0087a_training.py")
analyzer = _load("analyze_0087a_nsys", "benchmarks/analyze_0087a_nsys.py")


def test_disabled_range_controller_is_noop():
    events = []
    controller = profile.RangeController(
        False,
        push=lambda label: events.append(("push", label)),
        pop=lambda: events.append(("pop", None)),
    )
    with controller.range("forward"):
        with controller.scope("attention.local.kernel.forward"):
            events.append(("body", None))
    assert events == [("body", None)]


def test_scope_translation_is_balanced_and_coalesces_duplicate_nested_labels():
    events = []
    controller = profile.RangeController(
        True,
        push=lambda label: events.append(("push", label)),
        pop=lambda: events.append(("pop", None)),
    )
    with controller.scope("attention.local.forward"):
        with controller.scope("attention.local.kernel.forward"):
            pass
    with controller.scope("block.norm1.forward"):
        pass
    with controller.scope("unmapped.detail"):
        pass
    assert events == [
        ("push", "attention_local"),
        ("pop", None),
        ("push", "rmsnorm_residual"),
        ("pop", None),
    ]
    assert set(profile.REQUIRED_RANGES).issuperset(
        {"attention_local", "rmsnorm_residual"}
    )


def test_interval_union_overlap_and_gap_math():
    intervals = [(10, 20), (15, 30), (40, 50), (50, 55), (70, 70)]
    assert analyzer.union_intervals(intervals) == [(10, 30), (40, 55)]
    assert analyzer.interval_duration(intervals) == 35
    assert analyzer.interval_intersections(intervals, [(0, 12), (25, 45)]) == [
        (10, 12),
        (25, 30),
        (40, 45),
    ]


def _synthetic_trace(path: Path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL, end INTEGER, text TEXT, textId INTEGER,
            domainId INTEGER, globalTid INTEGER, rangeId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER NOT NULL, end INTEGER NOT NULL, streamId INTEGER,
            correlationId INTEGER, shortName INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER NOT NULL, end INTEGER NOT NULL, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (1, profile.CAPTURE_NAME),
            (2, "kernel_a"),
            (3, "kernel_b"),
            (4, "cudaLaunchKernel"),
        ],
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, 1000, None, 1, 7, 11, 1),
            (100, 900, "microbatch_total", None, 0, 11, 2),
            (100, 450, "forward", None, 0, 11, 3),
            (450, 900, "backward", None, 0, 11, 4),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
        [
            (110, 200, 1, 10, 2),
            (220, 400, 1, 20, 3),
            (500, 850, 1, 30, 2),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (90, 100, 10, 11, 4),
            (205, 210, 20, 11, 4),
            (300, 310, 30, 11, 4),
        ],
    )
    connection.commit()
    connection.close()


def test_schema_discovery_and_synthetic_timeline_analysis(tmp_path):
    trace = tmp_path / "trace.sqlite"
    _synthetic_trace(trace)
    result, tables = analyzer.analyze_sqlite(
        trace,
        starvation_threshold_us=0.0,
    )
    assert result["clock_and_boundary"]["analysis_duration_ms"] == pytest.approx(0.0008)
    assert result["gpu_execution"]["union_execution_ms"] == pytest.approx(0.00053)
    assert result["gpu_execution"]["gpu_execution_gap_proxy_fraction"] == pytest.approx(0.3375)
    assert result["gaps"]["candidate_host_starved_ms"] == pytest.approx(0.00011)
    assert result["gaps"]["fixed_gpu_work_speedup_upper_bound"] == pytest.approx(
        1.0 / (1.0 - 0.1375)
    )
    assert result["kernel_ranking"][0]["name"] == "kernel_a"
    assert {row["phase"] for row in result["phase_ranking"]} == {"forward", "backward"}
    assert any(row["kind"] == "gap" for row in tables["intervals"])


def test_registered_nvtx_helper_gpu_smoke():
    if not bool(int(__import__("os").environ.get("MINI_LLM_RUN_0087A_GPU_SMOKE", "0"))):
        pytest.skip("set MINI_LLM_RUN_0087A_GPU_SMOKE=1 for optional NVTX/CuPy smoke")
    import cupy as cp

    capture = profile.build_registered_capture()
    cp.cuda.runtime.deviceSynchronize()
    capture.start()
    value = cp.arange(128, dtype=cp.float32).sum()
    cp.cuda.runtime.deviceSynchronize()
    capture.end()
    assert float(value) == pytest.approx(sum(range(128)))
