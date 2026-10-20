#!/usr/bin/env python3
"""Read-only summary of existing MyLocal_LLM --profile-steps output.

The in-training profiler synchronizes GPU scopes and reports nested timings.
Do NOT sum parent and child regions or use profiled throughput for optimization.
"""

import argparse
import json
import re
from pathlib import Path

REPORT_RE = re.compile(r"^Performance profile: optimizer step\s+(\d+)\s*$")
ROW_RE = re.compile(r"^\s*(\S+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$")
WALL_RE = re.compile(r"^Profiled optimizer-step wall time:\s*([\d.]+)s")

PHASES = (
    "train.data_batch", "train.model_forward", "train.loss_forward",
    "train.loss_backward", "train.model_backward", "train.grad_clip",
    "train.optimizer_step", "train.refresh_compute_buffers", "train.zero_grad",
)
BLOCKS = (
    "block.norm1.forward", "block.attention.forward", "block.norm2.forward",
    "block.moe.forward", "block.moe.backward", "block.norm2.backward",
    "block.attention.backward", "block.norm1.backward",
)


def parse_profile(text):
    profiles = []
    active = None
    in_table = False
    for line in text.splitlines():
        match = REPORT_RE.match(line)
        if match:
            active = {"step": int(match.group(1)), "regions": {}, "wall_ms": None}
            profiles.append(active)
            in_table = False
            continue
        if active is None:
            continue
        if line.strip().startswith("region") and "total ms" in line:
            in_table = True
            continue
        if in_table:
            matched = ROW_RE.match(line)
            if matched:
                name, total, count, average = matched.groups()
                active["regions"][name] = {
                    "total_ms": float(total), "calls": int(count),
                    "avg_ms": float(average),
                }
                continue
            if active["regions"]:
                in_table = False
        wall = WALL_RE.match(line)
        if wall:
            active["wall_ms"] = float(wall.group(1)) * 1000.0
    return profiles


def summarize(profile):
    regions = profile["regions"]
    wall = profile["wall_ms"]
    phase = {name: regions[name]["total_ms"] for name in PHASES if name in regions}
    blocks = {name: regions[name]["total_ms"] for name in BLOCKS if name in regions}
    attention = {name: record["total_ms"] for name, record in regions.items()
                 if name.startswith("attention.") and ".plan." not in name}
    head = {name: record["total_ms"] for name, record in regions.items()
            if name.startswith("model.output_projection.") or name.startswith("model.loss.")}
    detail = {name: record["total_ms"] for name, record in regions.items()
              if (name.startswith("local.") or name.startswith("moe.expert."))
              and (name.endswith(".gpu") or name.endswith(".host"))}
    return {
        "step": profile["step"], "wall_ms": wall,
        "phases_ms": phase, "block_components_ms": blocks,
        "attention_subregions_ms": attention, "lm_head_subregions_ms": head,
        "detail_ms": detail,
        "unscoped_ms_estimate": max(0.0, wall - sum(phase.values())) if wall is not None else None,
    }


def table(title, entries, denominator=None, max_rows=None):
    print(f"\n{title}")
    print("-" * len(title))
    ranked = sorted(entries.items(), key=lambda item: -item[1])
    if max_rows is not None:
        ranked = ranked[:max_rows]
    for name, ms in ranked:
        pct = f" {100 * ms / denominator:5.1f}%" if denominator else ""
        print(f"  {name:<43} {ms:>10.1f} ms{pct}")
    if not ranked:
        print("  (no matching scopes in this log)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="console output captured with tee")
    parser.add_argument("--json", type=Path, default=None, help="optional machine-readable summary")
    args = parser.parse_args(argv)
    reports = parse_profile(args.log.read_text(errors="replace"))
    if not reports or not any(r["regions"] for r in reports):
        parser.error("No populated 'Performance profile: optimizer step' table found")
    summaries = [summarize(report) for report in reports if report["regions"]]
    for summary in summaries:
        print(f"\n=== Profiled optimizer step {summary['step']} ===")
        print("GPU scopes use synchronization; this is NOT normal step throughput.")
        print("Parent/child rows are nested: do NOT sum across different tables.")
        wall = summary["wall_ms"]
        if wall is not None:
            print(f"Instrumented wall time: {wall / 1000:.3f} s")
        table("Top-level training phases (approximately disjoint)",
              summary["phases_ms"], denominator=wall)
        table("Transformer-block subcomponents (disjoint within block forward/backward)",
              summary["block_components_ms"], max_rows=12)
        table("Attention hot subregions (nested; do not sum)",
              summary["attention_subregions_ms"], max_rows=15)
        table("LM head / cross-entropy subregions (nested)",
              summary["lm_head_subregions_ms"], max_rows=12)
        gpu_detail = {key: value for key, value in summary["detail_ms"].items()
                      if key.endswith(".gpu")}
        host_detail = {key: value for key, value in summary["detail_ms"].items()
                       if key.endswith(".host")}
        table("Optional dense FFN / local detail CUDA-event timings", gpu_detail, max_rows=16)
        table("Optional dense FFN / local detail host-submission timings", host_detail, max_rows=16)
        print("Notes: 'block.moe' refers to DenseFFN on ffn_mode=dense. "
              "The 'moe.expert.*' detail names also apply to that path.")
        print("A nested attention or output-projection timer belongs to its parent "
              "and must not be added again to model time.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"\nSaved structured report: {args.json}")


if __name__ == "__main__":
    main()
