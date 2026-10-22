#!/usr/bin/env python3
"""Analyze an 0087A Nsight Systems SQLite export without fixed schema names."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Iterable

CAPTURE_NAME = "PROFILE_0087A_MICROBATCH"
ANALYSIS_RANGE = "microbatch_total"
PROFILE_PHASES = {
    "forward", "attention_full", "attention_local", "historical_retrieval",
    "ffn", "rmsnorm_residual", "lm_head_loss", "backward",
    "attention_backward_recompute", "ffn_backward", "gradient_accumulation",
}


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ns(values: Iterable[int]) -> dict[str, float | int | None]:
    samples = [int(value) for value in values]
    return {
        "count": len(samples),
        "median_us": None if not samples else percentile(samples, 0.5) / 1_000.0,
        "p90_us": None if not samples else percentile(samples, 0.9) / 1_000.0,
        "max_us": None if not samples else max(samples) / 1_000.0,
        "total_ms": sum(samples) / 1_000_000.0,
    }


def union_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    valid = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[list[int]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in union_intervals(intervals))


def clip_interval(start: int, end: int, window: tuple[int, int]) -> tuple[int, int] | None:
    clipped = (max(int(start), window[0]), min(int(end), window[1]))
    return clipped if clipped[1] > clipped[0] else None


def interval_intersections(
    intervals: Iterable[tuple[int, int]], windows: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    left = union_intervals(intervals)
    right = union_intervals(windows)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def discover_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    result = {}
    for (table,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        escaped = table.replace('"', '""')
        result[table] = [
            row[1] for row in connection.execute(f'PRAGMA table_info("{escaped}")')
        ]
    return result


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _rows(connection: sqlite3.Connection, table: str, columns: list[str]):
    query = "SELECT " + ", ".join(_quoted(column) for column in columns)
    query += " FROM " + _quoted(table)
    for values in connection.execute(query):
        yield dict(zip(columns, values))


def _string_map(connection: sqlite3.Connection, schema: dict[str, list[str]]) -> dict[int, str]:
    candidates = [
        table for table, columns in schema.items()
        if {"id", "value"}.issubset(columns) and "STRING" in table.upper()
    ]
    strings: dict[int, str] = {}
    for table in candidates:
        for row in _rows(connection, table, ["id", "value"]):
            strings[int(row["id"])] = str(row["value"])
    return strings


def _resolve_text(row: dict, strings: dict[int, str]) -> str:
    for column in ("text", "name", "value"):
        if row.get(column) not in (None, ""):
            return str(row[column])
    for column in ("textId", "nameId", "demangledName", "shortName", "mangledName"):
        value = row.get(column)
        if value is not None and int(value) in strings:
            return strings[int(value)]
    return "<unresolved>"


def _nvtx_ranges(connection, schema, strings) -> list[dict]:
    result = []
    for table, columns in schema.items():
        if "NVTX" not in table.upper() or not {"start", "end"}.issubset(columns):
            continue
        selected = [
            column for column in (
                "start", "end", "text", "textId", "domainId", "globalTid", "rangeId"
            ) if column in columns
        ]
        for row in _rows(connection, table, selected):
            if row.get("end") is None:
                continue
            result.append(
                {
                    "table": table,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "name": _resolve_text(row, strings),
                    "domain_id": row.get("domainId"),
                    "thread_id": row.get("globalTid"),
                    "range_id": row.get("rangeId"),
                }
            )
    return result


def _capture_window(ranges: list[dict], capture_name: str) -> tuple[int, int]:
    matches = [entry for entry in ranges if entry["name"] == capture_name]
    if not matches:
        names = sorted({entry["name"] for entry in ranges})
        raise ValueError(f"NVTX capture range {capture_name!r} not found; ranges include {names[:30]}")
    return min(entry["start"] for entry in matches), max(entry["end"] for entry in matches)


def _analysis_window(
    ranges: list[dict], capture: tuple[int, int], analysis_name: str,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    windows = []
    for entry in ranges:
        if entry["name"] != analysis_name:
            continue
        clipped = clip_interval(entry["start"], entry["end"], capture)
        if clipped is not None:
            windows.append(clipped)
    if not windows:
        return capture, [capture]
    return (min(start for start, _ in windows), max(end for _, end in windows)), windows


def _gpu_events(connection, schema, strings, window) -> list[dict]:
    result = []
    for table, columns in schema.items():
        upper = table.upper()
        if "CUPTI_ACTIVITY_KIND_" not in upper or not {"start", "end"}.issubset(columns):
            continue
        if "KERNEL" in upper:
            kind = "kernel"
        elif "MEMCPY" in upper:
            kind = "memcpy"
        elif "MEMSET" in upper:
            kind = "memset"
        else:
            continue
        selected = [
            column for column in (
                "start", "end", "correlationId", "deviceId", "contextId", "streamId",
                "demangledName", "shortName", "mangledName", "nameId", "copyKind",
                "srcKind", "dstKind", "bytes", "value",
            ) if column in columns
        ]
        for row in _rows(connection, table, selected):
            clipped = clip_interval(row["start"], row["end"], window)
            if clipped is None:
                continue
            result.append(
                {
                    "table": table,
                    "kind": kind,
                    "start": clipped[0],
                    "end": clipped[1],
                    "raw_start": int(row["start"]),
                    "raw_end": int(row["end"]),
                    "duration_ns": clipped[1] - clipped[0],
                    "name": _resolve_text(row, strings) if kind == "kernel" else kind,
                    "correlation_id": row.get("correlationId"),
                    "stream_id": row.get("streamId"),
                    "bytes": row.get("bytes"),
                    "copy_kind": row.get("copyKind"),
                    "src_kind": row.get("srcKind"),
                    "dst_kind": row.get("dstKind"),
                }
            )
    return sorted(result, key=lambda entry: (entry["start"], entry["end"]))


def _runtime_apis(connection, schema, strings, capture) -> list[dict]:
    result = []
    for table, columns in schema.items():
        upper = table.upper()
        if "CUPTI_ACTIVITY_KIND_RUNTIME" not in upper and "CUPTI_ACTIVITY_KIND_DRIVER" not in upper:
            continue
        if not {"start", "end"}.issubset(columns):
            continue
        selected = [
            column for column in (
                "start", "end", "correlationId", "globalTid", "nameId", "name"
            ) if column in columns
        ]
        for row in _rows(connection, table, selected):
            clipped = clip_interval(row["start"], row["end"], capture)
            if clipped is None:
                continue
            result.append(
                {
                    "table": table,
                    "start": clipped[0],
                    "end": clipped[1],
                    "raw_start": int(row["start"]),
                    "raw_end": int(row["end"]),
                    "duration_ns": clipped[1] - clipped[0],
                    "name": _resolve_text(row, strings),
                    "correlation_id": row.get("correlationId"),
                    "thread_id": row.get("globalTid"),
                }
            )
    return sorted(result, key=lambda entry: (entry["start"], entry["end"]))


def _synchronizations(connection, schema, strings, capture, apis) -> list[dict]:
    result = [entry for entry in apis if "synchron" in entry["name"].lower()]
    for table, columns in schema.items():
        if "SYNCHRONIZATION" not in table.upper() or not {"start", "end"}.issubset(columns):
            continue
        selected = [column for column in ("start", "end", "correlationId", "syncType") if column in columns]
        for row in _rows(connection, table, selected):
            clipped = clip_interval(row["start"], row["end"], capture)
            if clipped is None:
                continue
            result.append(
                {
                    "table": table,
                    "start": clipped[0],
                    "end": clipped[1],
                    "duration_ns": clipped[1] - clipped[0],
                    "name": f"CUPTI synchronization type {row.get('syncType')}",
                    "correlation_id": row.get("correlationId"),
                    "thread_id": None,
                }
            )
    return result


def _gap_rows(
    merged_gpu: list[tuple[int, int]], gpu_events: list[dict], apis: list[dict],
    analysis_windows: list[tuple[int, int]], starvation_threshold_ns: int,
) -> list[dict]:
    api_by_correlation = {
        int(entry["correlation_id"]): entry
        for entry in apis if entry["correlation_id"] is not None
    }
    event_at_start: dict[int, list[dict]] = {}
    for event in gpu_events:
        event_at_start.setdefault(event["start"], []).append(event)
    rows = []
    for window_start, window_end in union_intervals(analysis_windows):
        inside = [
            (max(start, window_start), min(end, window_end))
            for start, end in merged_gpu if end > window_start and start < window_end
        ]
        cursor = window_start
        for start, end in inside:
            if start > cursor:
                next_events = event_at_start.get(start, [])
                correlated = [
                    api_by_correlation[int(event["correlation_id"])]
                    for event in next_events
                    if event["correlation_id"] is not None
                    and int(event["correlation_id"]) in api_by_correlation
                ]
                submit_end = max((entry["raw_end"] for entry in correlated), default=None)
                starved_ns = 0
                classification = "uncorrelated_or_device_side_gap"
                if submit_end is not None:
                    if submit_end <= cursor:
                        classification = "work_already_queued_or_device_side_gap"
                    else:
                        starved_ns = max(0, min(start, submit_end) - cursor)
                        classification = "host_submission_starvation_candidate"
                duration = start - cursor
                rows.append(
                    {
                        "start_ns": cursor,
                        "end_ns": start,
                        "duration_ns": duration,
                        "duration_us": duration / 1_000.0,
                        "next_submission_end_ns": submit_end,
                        "host_starved_ns": starved_ns if duration >= starvation_threshold_ns else 0,
                        "classification": classification,
                        "above_threshold": duration >= starvation_threshold_ns,
                    }
                )
            cursor = max(cursor, end)
        if cursor < window_end:
            duration = window_end - cursor
            rows.append(
                {
                    "start_ns": cursor,
                    "end_ns": window_end,
                    "duration_ns": duration,
                    "duration_us": duration / 1_000.0,
                    "next_submission_end_ns": None,
                    "host_starved_ns": 0,
                    "classification": "analysis_boundary_tail",
                    "above_threshold": duration >= starvation_threshold_ns,
                }
            )
    return rows


def _group_totals(rows: list[dict], key: str) -> list[dict]:
    totals: dict[str, dict] = {}
    for row in rows:
        name = str(row[key])
        target = totals.setdefault(name, {key: name, "count": 0, "total_ns": 0, "max_ns": 0})
        target["count"] += 1
        target["total_ns"] += int(row["duration_ns"])
        target["max_ns"] = max(target["max_ns"], int(row["duration_ns"]))
    result = []
    for target in totals.values():
        target["total_ms"] = target["total_ns"] / 1_000_000.0
        target["max_us"] = target["max_ns"] / 1_000.0
        result.append(target)
    return sorted(result, key=lambda entry: entry["total_ns"], reverse=True)


def _phase_rows(ranges, gpu_intervals, analysis_windows) -> list[dict]:
    labels = sorted({entry["name"] for entry in ranges if entry["name"] in PROFILE_PHASES})
    result = []
    for label in labels:
        phase_windows = []
        for entry in ranges:
            if entry["name"] != label:
                continue
            for analysis in analysis_windows:
                clipped = clip_interval(entry["start"], entry["end"], analysis)
                if clipped is not None:
                    phase_windows.append(clipped)
        if not phase_windows:
            continue
        merged_phase = union_intervals(phase_windows)
        gpu_overlap = interval_intersections(gpu_intervals, merged_phase)
        result.append(
            {
                "phase": label,
                "range_count": len(phase_windows),
                "elapsed_union_ns": interval_duration(merged_phase),
                "elapsed_union_ms": interval_duration(merged_phase) / 1_000_000.0,
                "gpu_execution_union_ns": interval_duration(gpu_overlap),
                "gpu_execution_union_ms": interval_duration(gpu_overlap) / 1_000_000.0,
            }
        )
    return sorted(result, key=lambda entry: entry["elapsed_union_ns"], reverse=True)


def analyze_sqlite(
    path: Path,
    *,
    capture_name: str = CAPTURE_NAME,
    analysis_range: str = ANALYSIS_RANGE,
    starvation_threshold_us: float = 10.0,
) -> tuple[dict, dict[str, list[dict]]]:
    connection = sqlite3.connect(str(path))
    try:
        schema = discover_schema(connection)
        strings = _string_map(connection, schema)
        ranges = _nvtx_ranges(connection, schema, strings)
        capture = _capture_window(ranges, capture_name)
        cpu_analysis_envelope, cpu_analysis_windows = _analysis_window(
            ranges, capture, analysis_range,
        )
        all_apis = _runtime_apis(connection, schema, strings, capture)
        apis = [
            entry for entry in all_apis
            if any(
                entry["raw_end"] > start and entry["raw_start"] < end
                for start, end in cpu_analysis_windows
            )
        ]
        submitted_correlations = {
            int(entry["correlation_id"])
            for entry in apis if entry["correlation_id"] is not None
        }
        all_gpu_events = _gpu_events(connection, schema, strings, capture)
        gpu_events = [
            entry for entry in all_gpu_events
            if entry["correlation_id"] is not None
            and int(entry["correlation_id"]) in submitted_correlations
        ]
        analysis_windows = []
        for start, end in cpu_analysis_windows:
            window_correlations = {
                int(entry["correlation_id"])
                for entry in apis
                if entry["correlation_id"] is not None
                and entry["raw_end"] > start and entry["raw_start"] < end
            }
            projected_end = max(
                [end] + [
                    entry["raw_end"] for entry in gpu_events
                    if int(entry["correlation_id"]) in window_correlations
                ]
            )
            analysis_windows.append((start, min(projected_end, capture[1])))
        analysis_envelope = (
            min(start for start, _ in analysis_windows),
            max(end for _, end in analysis_windows),
        )
        gpu_intervals = [(entry["start"], entry["end"]) for entry in gpu_events]
        merged_gpu = union_intervals(gpu_intervals)
        analysis_duration = interval_duration(analysis_windows)
        gpu_union = interval_duration(gpu_intervals)
        synchronizations = _synchronizations(connection, schema, strings, analysis_envelope, apis)
        gaps = _gap_rows(
            merged_gpu,
            gpu_events,
            apis,
            analysis_windows,
            int(starvation_threshold_us * 1_000.0),
        )

        api_by_corr = {
            int(entry["correlation_id"]): entry
            for entry in apis if entry["correlation_id"] is not None
        }
        enqueue_delays = []
        queued = 0
        correlated_gpu = 0
        for event in gpu_events:
            correlation = event.get("correlation_id")
            if correlation is None or int(correlation) not in api_by_corr:
                continue
            correlated_gpu += 1
            delay = event["raw_start"] - api_by_corr[int(correlation)]["raw_end"]
            enqueue_delays.append(delay)
            if api_by_corr[int(correlation)]["raw_end"] <= event["raw_start"]:
                queued += 1

        internal_gap_values = [
            row["duration_ns"] for row in gaps
            if row["classification"] != "analysis_boundary_tail"
        ]
        candidate_starved_ns = sum(row["host_starved_ns"] for row in gaps)
        gap_fraction = 0.0 if analysis_duration == 0 else 1.0 - gpu_union / analysis_duration
        avoidable_fraction = 0.0 if analysis_duration == 0 else candidate_starved_ns / analysis_duration
        result = {
            "schema_version": "0087A-analysis-v1",
            "input_sqlite": str(path.resolve()),
            "clock_and_boundary": {
                "timestamp_unit": "nanoseconds",
                "clock_alignment": "NVTX, CUDA API and CUPTI GPU activity timestamps are from the same Nsight Systems export timebase",
                "capture_range": capture_name,
                "capture_start_ns": capture[0],
                "capture_end_ns": capture[1],
                "capture_duration_ms": (capture[1] - capture[0]) / 1_000_000.0,
                "analysis_range": analysis_range if analysis_windows != [capture] else capture_name,
                "analysis_start_ns": analysis_envelope[0],
                "analysis_end_ns": analysis_envelope[1],
                "analysis_window_count": len(analysis_windows),
                "analysis_duration_ms": analysis_duration / 1_000_000.0,
                "post_measurement_capture_tail_excluded": analysis_windows != [capture],
            },
            "gpu_execution": {
                "event_count": len(gpu_events),
                "kernel_count": sum(entry["kind"] == "kernel" for entry in gpu_events),
                "memcpy_count": sum(entry["kind"] == "memcpy" for entry in gpu_events),
                "memset_count": sum(entry["kind"] == "memset" for entry in gpu_events),
                "union_execution_ms": gpu_union / 1_000_000.0,
                "gpu_execution_gap_proxy_fraction": gap_fraction,
                "gpu_execution_gap_proxy_percent": 100.0 * gap_fraction,
                "warning": "This union-of-kernel/memory-operation proxy is not SM occupancy, Tensor Core utilization, or nvidia-smi GPU-Util.",
            },
            "gaps": {
                "internal_execution_gap": summarize_ns(internal_gap_values),
                "threshold_us": starvation_threshold_us,
                "candidate_host_starved_ms": candidate_starved_ns / 1_000_000.0,
                "candidate_avoidable_host_starvation_fraction": avoidable_fraction,
                "fixed_gpu_work_speedup_upper_bound": (
                    math.inf if avoidable_fraction >= 1.0 else 1.0 / (1.0 - avoidable_fraction)
                ),
                "qualification": "Only the correlated portion where the next CUDA submission completed after prior GPU work drained is labeled a host-starvation candidate. Uncorrelated and already-queued gaps are not attributed to Python.",
            },
            "submission": {
                "cuda_api_count": len(apis),
                "correlated_gpu_event_count": correlated_gpu,
                "next_work_already_queued_count": queued,
                "enqueue_delay": summarize_ns(enqueue_delays),
                "synchronization_count": len(synchronizations),
                "synchronization_cpu_api_ms": sum(entry["duration_ns"] for entry in synchronizations) / 1_000_000.0,
            },
            "kernel_ranking": _group_totals(
                [entry for entry in gpu_events if entry["kind"] == "kernel"], "name"
            ),
            "cuda_api_ranking": _group_totals(apis, "name"),
            "phase_ranking": _phase_rows(ranges, gpu_intervals, analysis_windows),
            "schema_discovery": {table: columns for table, columns in schema.items() if (
                "CUPTI_ACTIVITY_KIND" in table.upper() or "NVTX" in table.upper() or "STRING" in table.upper()
            )},
        }
        tables = {
            "intervals": [
                {
                    "kind": entry["kind"],
                    "name": entry["name"],
                    "start_ns": entry["start"],
                    "end_ns": entry["end"],
                    "duration_ns": entry["duration_ns"],
                    "stream_id": entry["stream_id"],
                    "correlation_id": entry["correlation_id"],
                }
                for entry in gpu_events
            ] + [{"kind": "gap", "name": entry["classification"], **entry} for entry in gaps],
            "kernels": result["kernel_ranking"],
            "phases": result["phase_ranking"],
            "apis": result["cuda_api_ranking"],
        }
        return result, tables
    finally:
        connection.close()


def _export_sqlite(report: Path, destination: Path) -> None:
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys is required to export an .nsys-rep file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            nsys, "export", "--type", "sqlite", "--force-overwrite=true",
            f"--output={destination}", str(report),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("nsys export failed: " + (completed.stderr or completed.stdout).strip())


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight .sqlite or .nsys-rep")
    parser.add_argument("--sqlite", type=Path, help="SQLite export destination for .nsys-rep input")
    parser.add_argument("--capture-name", default=CAPTURE_NAME)
    parser.add_argument("--analysis-range", default=ANALYSIS_RANGE)
    parser.add_argument("--starvation-threshold-us", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trace = args.trace.expanduser().resolve()
    if trace.suffix == ".nsys-rep":
        sqlite_path = (
            args.sqlite.expanduser().resolve() if args.sqlite
            else args.output_json.with_suffix(".sqlite").resolve()
        )
        _export_sqlite(trace, sqlite_path)
    else:
        sqlite_path = trace
    result, tables = analyze_sqlite(
        sqlite_path,
        capture_name=args.capture_name,
        analysis_range=args.analysis_range,
        starvation_threshold_us=args.starvation_threshold_us,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prefix = args.output_prefix or args.output_json.with_suffix("")
    for name, rows in tables.items():
        _write_csv(Path(str(prefix) + f"_{name}.csv"), rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
