from pathlib import Path
import sqlite3

import pytest

from benchmarks import analyze_0089o_historical_nsys as analyzer


def _synthetic_trace(path: Path) -> None:
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
            (1, analyzer.CAPTURE_NAME),
            (2, "route_kernel"),
            (3, "read_kernel"),
            (4, "encoder_kernel"),
            (5, "coarse_kernel"),
            (6, "other_kernel"),
            (7, "cudaLaunchKernel"),
        ],
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, 1000, None, 1, 7, 11, 1),
            (100, 300, analyzer.COARSE_HISTORICAL_RANGE, None, 0, 11, 2),
            (110, 150, "historical.layer.route", None, 0, 11, 3),
            (160, 200, "historical.layer.block_read", None, 0, 11, 4),
            (210, 280, "historical.encoder.recompute", None, 0, 11, 5),
            (215, 275, "historical.encoder.forward", None, 0, 11, 6),
            (220, 260, "historical.encoder.block_attention", None, 0, 11, 7),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (120, 125, 10, 11, 7),
            (170, 175, 20, 11, 7),
            (225, 230, 30, 11, 7),
            (285, 290, 40, 11, 7),
            (350, 355, 50, 11, 7),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
        [
            (500, 550, 1, 10, 2),
            (550, 650, 1, 20, 3),
            (650, 680, 1, 30, 4),
            (680, 700, 1, 40, 5),
            (700, 800, 1, 50, 6),
        ],
    )
    connection.commit()
    connection.close()


def test_correlation_attribution_uses_submission_range_not_gpu_timestamp(tmp_path):
    trace = tmp_path / "trace.sqlite"
    _synthetic_trace(trace)

    result, tables = analyzer.analyze_sqlite(trace)

    execution = result["historical_gpu_execution"]
    assert execution["event_count"] == 4
    assert execution["kernel_count"] == 4
    assert execution["gpu_execution_union_ms"] == pytest.approx(0.0002)
    assert execution["summed_kernel_duration_ms"] == pytest.approx(0.0002)
    assert result["accounting"]["timestamp_overlap_used_for_source_attribution"] is False
    assert result["accounting"]["capture_gpu_events_not_attributed_to_historical"] == 1

    by_source = {row["source_range"]: row for row in tables["ranges"]}
    assert by_source["historical.layer.route"]["summed_kernel_duration_ms"] == pytest.approx(
        0.00005
    )
    assert by_source["historical.layer.block_read"]["summed_kernel_duration_ms"] == pytest.approx(
        0.0001
    )
    assert by_source["historical.encoder.block_attention"]["phase"] == "recompute"
    assert by_source[analyzer.UNCLASSIFIED_RANGE]["summed_kernel_duration_ms"] == pytest.approx(
        0.00002
    )


def test_deepest_range_and_recompute_ancestry_are_preserved(tmp_path):
    trace = tmp_path / "trace.sqlite"
    _synthetic_trace(trace)

    result, tables = analyzer.analyze_sqlite(trace)

    encoder_event = next(
        row for row in tables["events"] if row["name"] == "encoder_kernel"
    )
    assert encoder_event["source_range"] == "historical.encoder.block_attention"
    assert encoder_event["phase"] == "recompute"
    assert "historical.encoder.recompute" in encoder_event["source_ancestry"]
    assert result["accounting"]["unclassified_historical_event_count"] == 1
