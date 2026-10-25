#!/usr/bin/env python3
"""Classify matched PATCH 0089K depth-stack Nsight Systems captures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sqlite3
import statistics
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import analyze_0087a_nsys as timeline

CAPTURE_RANGE = "0089K_STACK_FWD_BWD"
CATEGORIES = (
    "GEMM",
    "Attention",
    "SwiGLU forward",
    "SwiGLU backward",
    "RMSNorm",
    "Residual",
    "BF16/FP32 conversion",
    "QKV/RoPE processing",
    "Parameter-gradient accumulation",
    "Memory/data movement",
    "Other elementwise",
    "Unknown",
)


def classify_operation(kind: str, name: str, source_range: str | None) -> str:
    """Classify one GPU activity, preferring explicit symbols over attribution."""
    lower = name.lower()
    source = (source_range or "").lower()
    if kind in {"memcpy", "memset"}:
        return "Memory/data movement"
    if "swiglu_fwd" in lower or ("silu" in lower and "backward" not in lower):
        return "SwiGLU forward"
    if "swiglu_bwd" in lower or "silu_backward" in lower:
        return "SwiGLU backward"
    if any(token in lower for token in ("_dkdv_kernel", "_dq_kernel", "_delta_kernel")):
        return "Attention"
    if lower == "_fwd_kernel" or "flex_attention" in lower:
        return "Attention"
    if "rmsnorm" in lower or "rsqrt" in lower or "mean_mul" in lower or "pow_sum" in lower:
        return "RMSNorm"
    if "rope" in lower:
        return "QKV/RoPE processing"
    if any(token in lower for token in ("cutlass::kernel", "nvjet_", "sgemm", "gemm")):
        return "GEMM"
    if "residual_add" in lower:
        return "Residual"
    if "wgrad" in source and any(token in lower for token in ("add", "copy")):
        return "Parameter-gradient accumulation"
    if "copy" in lower:
        if "bfloat16_float32" in lower or "float32_bfloat16" in lower or "_to_copy" in lower:
            return "BF16/FP32 conversion"
        return "Memory/data movement"
    if "take" in lower or "transpose" in lower or "contiguous" in lower:
        return "Memory/data movement"
    if "qkv_projection" in source or "rope" in source:
        return "QKV/RoPE processing"
    if "norm" in source:
        return "RMSNorm"
    if "dx_add" in source:
        return "Residual"
    if any(token in lower for token in ("add", "multiply", "subtract", "mul", "sub")):
        if source in {"0089k_stack_forward", "0089k_stack_backward"}:
            return "Residual"
        return "Other elementwise"
    return "Unknown"


def infer_dtype(name: str) -> str:
    lower = name.lower()
    has_bf16 = "bf16" in lower or "bfloat16" in lower
    has_fp32 = "f32" in lower or "float32" in lower or "sgemm" in lower
    if has_bf16 and has_fp32:
        return "BF16/FP32"
    if has_bf16:
        return "BF16"
    if has_fp32:
        return "FP32"
    return "unknown"


def infer_dimensions(category: str, source_range: str | None) -> str:
    source = (source_range or "").lower()
    if "gate_gemm" in source or "up_gemm" in source:
        return "[6912,1024] x [1024,4096]"
    if "gate_wgrad" in source or "up_wgrad" in source:
        return "[1024,6912] x [6912,4096]"
    if "gate_dx" in source or "up_dx" in source:
        return "[6912,4096] x [4096,1024]"
    if "down_gemm" in source:
        return "[6912,4096] x [4096,1024]"
    if "down_wgrad" in source:
        return "[4096,6912] x [6912,1024]"
    if "down_dx" in source:
        return "[6912,1024] x [1024,4096]"
    if "qkv_projection.forward" in source:
        return "[6912,1024] x packed [1024,1536]"
    if "output_projection" in source:
        return "[6912,1024] x [1024,1024]"
    if category == "Attention":
        return "B=3,T=2304,Hq=16,Hkv=4,D=64"
    if category.startswith("SwiGLU"):
        return "[6912,4096]"
    if category in {"RMSNorm", "Residual", "QKV/RoPE processing"}:
        return "[3,2304,1024] (unless kernel is head-shaped)"
    return "unknown from Nsight Systems symbol"


def opportunity(category: str) -> str:
    return {
        "GEMM": "Low unless matched GEMM totals diverge; preserve cuBLAS/CUTLASS.",
        "Attention": "Low if totals match; current custom Triton path is already specialized.",
        "SwiGLU forward": "Fuse/re-tune only if its complete-stack excess is material.",
        "SwiGLU backward": "Fuse/re-tune only if its complete-stack excess is material.",
        "RMSNorm": "Inspect generated-kernel count and residual boundaries.",
        "Residual": "Potential fusion target, bounded by this category's total time.",
        "BF16/FP32 conversion": "Eliminate redundant conversions or fold them into adjacent kernels.",
        "QKV/RoPE processing": "Inspect packing/materialization before replacing GEMMs.",
        "Parameter-gradient accumulation": "Consider direct accumulation only with complete-stack evidence.",
        "Memory/data movement": "Reuse buffers or eliminate materialization where cache ownership permits.",
        "Other elementwise": "Map to source before attempting fusion.",
        "Unknown": "No optimization claim until source mapping is established.",
    }[category]


def _source_range(ranges: list[dict], timestamp: int) -> str | None:
    active = [
        entry for entry in ranges
        if entry["start"] <= timestamp <= entry["end"]
        and entry["name"] != CAPTURE_RANGE
    ]
    if not active:
        return None
    return min(active, key=lambda entry: entry["end"] - entry["start"])["name"]


def _family_rows(events: list[dict], total_kernel_ns: int) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event["kind"] != "kernel":
            continue
        key = (event["name"], event["category"])
        row = grouped.setdefault(
            key,
            {
                "kernel_name": event["name"],
                "classification": event["category"],
                "durations_ns": [],
                "source_ranges": set(),
            },
        )
        row["durations_ns"].append(event["duration_ns"])
        if event["source_range"]:
            row["source_ranges"].add(event["source_range"])
    result = []
    for row in grouped.values():
        durations = row.pop("durations_ns")
        total_ns = sum(durations)
        sources = sorted(row.pop("source_ranges"))
        result.append(
            {
                **row,
                "launches": len(durations),
                "total_ms": total_ns / 1_000_000.0,
                "median_us": statistics.median(durations) / 1_000.0,
                "percent_of_summed_kernel_duration": (
                    0.0 if total_kernel_ns == 0 else 100.0 * total_ns / total_kernel_ns
                ),
                "dtype_if_known": infer_dtype(row["kernel_name"]),
                "dimensions_if_known": infer_dimensions(
                    row["classification"], sources[0] if len(sources) == 1 else None
                ),
                "source_ranges": sources,
                "potential_optimization": opportunity(row["classification"]),
            }
        )
    return sorted(result, key=lambda row: row["total_ms"], reverse=True)


def analyze_capture(path: Path, backend: str) -> dict[str, Any]:
    base, _ = timeline.analyze_sqlite(
        path, capture_name=CAPTURE_RANGE, analysis_range=CAPTURE_RANGE
    )
    connection = sqlite3.connect(str(path))
    try:
        schema = timeline.discover_schema(connection)
        strings = timeline._string_map(connection, schema)
        ranges = timeline._nvtx_ranges(connection, schema, strings)
        capture = timeline._capture_window(ranges, CAPTURE_RANGE)
        apis = timeline._runtime_apis(connection, schema, strings, capture)
        api_by_correlation = {
            int(row["correlation_id"]): row
            for row in apis if row["correlation_id"] is not None
        }
        events = []
        for event in timeline._gpu_events(connection, schema, strings, capture):
            correlation = event.get("correlation_id")
            if correlation is None or int(correlation) not in api_by_correlation:
                continue
            api = api_by_correlation[int(correlation)]
            source = _source_range(ranges, int(api["raw_start"]))
            events.append(
                {
                    **event,
                    "source_range": source,
                    "category": classify_operation(event["kind"], event["name"], source),
                }
            )
    finally:
        connection.close()

    category_events: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        category_events[event["category"]].append(event)
    category_rows = []
    for category in CATEGORIES:
        selected = category_events.get(category, [])
        intervals = [(row["start"], row["end"]) for row in selected]
        summed_ns = sum(row["duration_ns"] for row in selected)
        category_rows.append(
            {
                "category": category,
                "event_count": len(selected),
                "kernel_count": sum(row["kind"] == "kernel" for row in selected),
                "memcpy_count": sum(row["kind"] == "memcpy" for row in selected),
                "memset_count": sum(row["kind"] == "memset" for row in selected),
                "summed_duration_ms": summed_ns / 1_000_000.0,
                "execution_union_ms": timeline.interval_duration(
                    timeline.union_intervals(intervals)
                ) / 1_000_000.0,
                "opportunity": opportunity(category),
            }
        )
    total_kernel_ns = sum(
        row["duration_ns"] for row in events if row["kind"] == "kernel"
    )
    return {
        "backend": backend,
        "input_sqlite": str(path.resolve()),
        "clock_and_boundary": base["clock_and_boundary"],
        "gpu_execution": base["gpu_execution"],
        "gaps": base["gaps"],
        "submission": base["submission"],
        "category_breakdown": category_rows,
        "top_kernel_families": _family_rows(events, total_kernel_ns)[:20],
        "accounting": {
            "summed_kernel_duration_ms": total_kernel_ns / 1_000_000.0,
            "summed_all_event_duration_ms": sum(
                row["duration_ns"] for row in events
            ) / 1_000_000.0,
            "global_execution_union_ms": base["gpu_execution"]["union_execution_ms"],
            "category_durations_are_sums": True,
            "category_unions_may_overlap_across_streams": True,
            "nested_nvtx_ranges_not_summed": True,
        },
    }


def compare(cupy: dict[str, Any], torch: dict[str, Any]) -> list[dict]:
    cupy_rows = {row["category"]: row for row in cupy["category_breakdown"]}
    torch_rows = {row["category"]: row for row in torch["category_breakdown"]}
    rows = []
    for category in CATEGORIES:
        left = cupy_rows[category]
        right = torch_rows[category]
        rows.append(
            {
                "category": category,
                "cupy_summed_ms": left["summed_duration_ms"],
                "torch_summed_ms": right["summed_duration_ms"],
                "difference_ms": left["summed_duration_ms"] - right["summed_duration_ms"],
                "cupy_kernel_count": left["kernel_count"],
                "torch_kernel_count": right["kernel_count"],
                "cupy_event_count": left["event_count"],
                "torch_event_count": right["event_count"],
            }
        )
    return sorted(rows, key=lambda row: row["difference_ms"], reverse=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cupy", type=Path, required=True)
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cupy = analyze_capture(args.cupy, "cupy")
    torch = analyze_capture(args.torch, "torch-compile")
    comparison = compare(cupy, torch)
    for family in cupy["top_kernel_families"]:
        family["pytorch_equivalent"] = (
            "same classified operation exists"
            if any(row["classification"] == family["classification"] for row in torch["top_kernel_families"])
            else "no confidently classified equivalent in top 20"
        )
    for family in torch["top_kernel_families"]:
        family["cupy_equivalent"] = (
            "same classified operation exists"
            if any(row["classification"] == family["classification"] for row in cupy["top_kernel_families"])
            else "no confidently classified equivalent in top 20"
        )
    report = {
        "schema_version": "0089K-comparative-nsys-v1",
        "capture_range": CAPTURE_RANGE,
        "methodology": {
            "authoritative_latency": "unprofiled CUDA-event benchmark; not this capture",
            "gpu_execution": "union of correlated kernel/memcpy/memset intervals",
            "category_time": "summed exclusive GPU event durations by classification",
            "nvtx": "used for source attribution only; nested ranges are not added",
            "unknown_policy": "ambiguous kernels remain Unknown",
        },
        "cupy": cupy,
        "torch": torch,
        "comparison": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prefix = args.output_prefix
    _write_csv(Path(f"{prefix}_categories.csv"), comparison)
    _write_csv(Path(f"{prefix}_cupy_top20.csv"), cupy["top_kernel_families"])
    _write_csv(Path(f"{prefix}_torch_top20.csv"), torch["top_kernel_families"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
