#!/usr/bin/env python3
"""Correlation-correct analysis of PATCH 0089O historical Nsight captures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import analyze_0087a_nsys as timeline

CAPTURE_NAME = "PROFILE_0087A_MICROBATCH"
COARSE_HISTORICAL_RANGE = "historical_retrieval"
HISTORICAL_PREFIX = "historical."
UNCLASSIFIED_RANGE = "historical.unclassified"


def _is_historical_range(name: str) -> bool:
    return name == COARSE_HISTORICAL_RANGE or name.startswith(HISTORICAL_PREFIX)


def _active_historical_ranges(ranges: list[dict], timestamp: int) -> list[dict]:
    return [
        entry for entry in ranges
        if _is_historical_range(str(entry["name"]))
        and int(entry["start"]) <= timestamp < int(entry["end"])
    ]


def _phase(active: list[dict], source: str) -> str:
    names = {str(entry["name"]) for entry in active}
    if "historical.encoder.recompute" in names:
        return "recompute"
    if (
        source.startswith("historical.backward.")
        or source.startswith("historical.close.")
        or ".backward" in source
    ):
        return "backward"
    return "forward"


def _source_assignment(ranges: list[dict], timestamp: int) -> dict | None:
    active = _active_historical_ranges(ranges, timestamp)
    if not active:
        return None
    detailed = [
        entry for entry in active
        if str(entry["name"]).startswith(HISTORICAL_PREFIX)
    ]
    if detailed:
        leaf = min(
            detailed,
            key=lambda entry: (
                int(entry["end"]) - int(entry["start"]),
                -int(entry["start"]),
            ),
        )
        source = str(leaf["name"])
        source_duration = int(leaf["end"]) - int(leaf["start"])
        classified = True
    else:
        source = UNCLASSIFIED_RANGE
        source_duration = min(
            int(entry["end"]) - int(entry["start"]) for entry in active
        )
        classified = False
    ancestry = [
        str(entry["name"])
        for entry in sorted(
            active,
            key=lambda entry: (
                -(int(entry["end"]) - int(entry["start"])),
                int(entry["start"]),
            ),
        )
    ]
    return {
        "source_range": source,
        "phase": _phase(active, source),
        "ancestry": ancestry,
        "source_duration_ns": source_duration,
        "classified": classified,
    }


def _correlation_assignments(apis: list[dict], ranges: list[dict]) -> dict[int, dict]:
    assignments: dict[int, dict] = {}
    for api in apis:
        correlation = api.get("correlation_id")
        if correlation is None:
            continue
        assignment = _source_assignment(ranges, int(api["raw_start"]))
        if assignment is None:
            continue
        candidate = {
            **assignment,
            "api_name": str(api["name"]),
            "api_start_ns": int(api["raw_start"]),
            "api_end_ns": int(api["raw_end"]),
        }
        key = int(correlation)
        previous = assignments.get(key)
        if previous is None or (
            not previous["classified"] and candidate["classified"]
        ) or (
            previous["classified"] == candidate["classified"]
            and candidate["source_duration_ns"] < previous["source_duration_ns"]
        ):
            assignments[key] = candidate
    return assignments


def _attributed_events(
    gpu_events: list[dict], assignments: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    attributed = []
    uncorrelated = []
    for event in gpu_events:
        correlation = event.get("correlation_id")
        assignment = (
            assignments.get(int(correlation)) if correlation is not None else None
        )
        if assignment is None:
            uncorrelated.append(event)
            continue
        attributed.append(
            {
                **event,
                "start": int(event["raw_start"]),
                "end": int(event["raw_end"]),
                "duration_ns": int(event["raw_end"]) - int(event["raw_start"]),
                "source_range": assignment["source_range"],
                "phase": assignment["phase"],
                "source_ancestry": assignment["ancestry"],
                "api_name": assignment["api_name"],
            }
        )
    return attributed, uncorrelated


def _range_rows(events: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        grouped[(event["phase"], event["source_range"])].append(event)
    result = []
    for (phase, source), rows in grouped.items():
        intervals = [(row["start"], row["end"]) for row in rows]
        kernel_rows = [row for row in rows if row["kind"] == "kernel"]
        memcpy_rows = [row for row in rows if row["kind"] == "memcpy"]
        memset_rows = [row for row in rows if row["kind"] == "memset"]
        result.append(
            {
                "phase": phase,
                "source_range": source,
                "event_count": len(rows),
                "kernel_count": len(kernel_rows),
                "memcpy_count": len(memcpy_rows),
                "memset_count": len(memset_rows),
                "gpu_execution_union_ms": timeline.interval_duration(intervals) / 1_000_000.0,
                "summed_gpu_duration_ms": sum(row["duration_ns"] for row in rows) / 1_000_000.0,
                "summed_kernel_duration_ms": sum(row["duration_ns"] for row in kernel_rows) / 1_000_000.0,
                "summed_memcpy_duration_ms": sum(row["duration_ns"] for row in memcpy_rows) / 1_000_000.0,
                "summed_memset_duration_ms": sum(row["duration_ns"] for row in memset_rows) / 1_000_000.0,
                "memcpy_bytes": sum(int(row.get("bytes") or 0) for row in memcpy_rows),
            }
        )
    return sorted(result, key=lambda row: row["summed_gpu_duration_ms"], reverse=True)


def _phase_rows(events: list[dict]) -> list[dict]:
    result = []
    for phase in sorted({event["phase"] for event in events}):
        rows = [event for event in events if event["phase"] == phase]
        intervals = [(row["start"], row["end"]) for row in rows]
        result.append(
            {
                "phase": phase,
                "event_count": len(rows),
                "kernel_count": sum(row["kind"] == "kernel" for row in rows),
                "gpu_execution_union_ms": timeline.interval_duration(intervals) / 1_000_000.0,
                "summed_gpu_duration_ms": sum(row["duration_ns"] for row in rows) / 1_000_000.0,
            }
        )
    return sorted(result, key=lambda row: row["summed_gpu_duration_ms"], reverse=True)


def _kernel_rows(events: list[dict], total_kernel_ns: int) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for event in events:
        if event["kind"] != "kernel":
            continue
        grouped[(event["phase"], event["source_range"], event["name"])].append(
            int(event["duration_ns"])
        )
    result = []
    for (phase, source, name), durations in grouped.items():
        total_ns = sum(durations)
        result.append(
            {
                "phase": phase,
                "source_range": source,
                "kernel_name": name,
                "launches": len(durations),
                "total_ms": total_ns / 1_000_000.0,
                "median_us": statistics.median(durations) / 1_000.0,
                "max_us": max(durations) / 1_000.0,
                "percent_of_historical_kernel_duration": (
                    0.0 if total_kernel_ns == 0
                    else 100.0 * total_ns / total_kernel_ns
                ),
            }
        )
    return sorted(result, key=lambda row: row["total_ms"], reverse=True)


def analyze_sqlite(
    path: Path,
    *,
    capture_name: str = CAPTURE_NAME,
) -> tuple[dict, dict[str, list[dict]]]:
    connection = sqlite3.connect(str(path))
    try:
        schema = timeline.discover_schema(connection)
        strings = timeline._string_map(connection, schema)
        ranges = timeline._nvtx_ranges(connection, schema, strings)
        capture = timeline._capture_window(ranges, capture_name)
        apis = timeline._runtime_apis(connection, schema, strings, capture)
        gpu_events = timeline._gpu_events(connection, schema, strings, capture)
        assignments = _correlation_assignments(apis, ranges)
        events, uncorrelated = _attributed_events(gpu_events, assignments)

        historical_api_correlations = set(assignments)
        event_correlations = {
            int(event["correlation_id"])
            for event in events if event.get("correlation_id") is not None
        }
        intervals = [(event["start"], event["end"]) for event in events]
        kernel_events = [event for event in events if event["kind"] == "kernel"]
        total_kernel_ns = sum(event["duration_ns"] for event in kernel_events)
        ranges_table = _range_rows(events)
        kernels_table = _kernel_rows(events, total_kernel_ns)
        phases_table = _phase_rows(events)
        source_names = sorted({row["source_range"] for row in ranges_table})
        detailed_ranges = sorted(
            {
                str(entry["name"]) for entry in ranges
                if str(entry["name"]).startswith(HISTORICAL_PREFIX)
            }
        )
        unclassified_events = [
            event for event in events
            if event["source_range"] == UNCLASSIFIED_RANGE
        ]

        result: dict[str, Any] = {
            "schema_version": "0089O-analysis-v1",
            "input_sqlite": str(path.resolve()),
            "capture": {
                "name": capture_name,
                "start_ns": capture[0],
                "end_ns": capture[1],
                "duration_ms": (capture[1] - capture[0]) / 1_000_000.0,
            },
            "accounting": {
                "method": (
                    "CUDA API submissions are assigned to the deepest containing "
                    "historical NVTX range; complete GPU activities are mapped by "
                    "CUPTI correlation ID. GPU execution union and summed durations "
                    "are reported separately."
                ),
                "timestamp_overlap_used_for_source_attribution": False,
                "summed_duration_warning": (
                    "Summed GPU duration can exceed elapsed time when streams overlap; "
                    "it is not wall-clock latency."
                ),
                "historical_api_correlations": len(historical_api_correlations),
                "historical_correlations_with_gpu_activity": len(event_correlations),
                "historical_correlations_without_gpu_activity": len(
                    historical_api_correlations - event_correlations
                ),
                "capture_gpu_events_not_attributed_to_historical": len(uncorrelated),
                "unclassified_historical_event_count": len(unclassified_events),
                "unclassified_historical_summed_ms": sum(
                    event["duration_ns"] for event in unclassified_events
                ) / 1_000_000.0,
                "unclassified_historical_percent": (
                    0.0 if not events else 100.0 * sum(
                        event["duration_ns"] for event in unclassified_events
                    ) / sum(event["duration_ns"] for event in events)
                ),
            },
            "historical_gpu_execution": {
                "event_count": len(events),
                "kernel_count": len(kernel_events),
                "memcpy_count": sum(event["kind"] == "memcpy" for event in events),
                "memset_count": sum(event["kind"] == "memset" for event in events),
                "gpu_execution_union_ms": timeline.interval_duration(intervals) / 1_000_000.0,
                "summed_gpu_duration_ms": sum(event["duration_ns"] for event in events) / 1_000_000.0,
                "summed_kernel_duration_ms": total_kernel_ns / 1_000_000.0,
            },
            "ranges_present": detailed_ranges,
            "attributed_source_ranges": source_names,
            "range_ranking": ranges_table,
            "phase_ranking": phases_table,
            "kernel_ranking": kernels_table,
        }
        tables = {
            "ranges": ranges_table,
            "phases": phases_table,
            "kernels": kernels_table,
            "events": [
                {
                    "phase": event["phase"],
                    "source_range": event["source_range"],
                    "kind": event["kind"],
                    "name": event["name"],
                    "start_ns": event["start"],
                    "end_ns": event["end"],
                    "duration_ns": event["duration_ns"],
                    "stream_id": event.get("stream_id"),
                    "correlation_id": event.get("correlation_id"),
                    "api_name": event["api_name"],
                    "source_ancestry": " > ".join(event["source_ancestry"]),
                }
                for event in events
            ],
        }
        return result, tables
    finally:
        connection.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight .sqlite or .nsys-rep")
    parser.add_argument("--sqlite", type=Path, help="SQLite export for .nsys-rep input")
    parser.add_argument("--capture-name", default=CAPTURE_NAME)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trace = args.trace.expanduser().resolve()
    if trace.suffix == ".nsys-rep":
        sqlite_path = (
            args.sqlite.expanduser().resolve()
            if args.sqlite else args.output_json.with_suffix(".sqlite").resolve()
        )
        timeline._export_sqlite(trace, sqlite_path)
    else:
        sqlite_path = trace
    result, tables = analyze_sqlite(
        sqlite_path,
        capture_name=args.capture_name,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prefix = args.output_prefix or args.output_json.with_suffix("")
    for name, rows in tables.items():
        timeline._write_csv(Path(str(prefix) + f"_{name}.csv"), rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
