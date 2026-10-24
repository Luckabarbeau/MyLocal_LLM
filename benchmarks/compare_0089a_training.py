#!/usr/bin/env python3
"""Validate and compare matched grouped/Triton 0089A full-model reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


_MODEL_KEYS = (
    "checkpoint",
    "parameter_count",
    "layers",
    "d_model",
    "d_ff",
    "q_heads",
    "kv_heads",
    "dtype",
    "activation_checkpoint",
    "lm_head_chunk_tokens",
)
_HASH_KEYS = ("inputs", "targets", "loss_mask", "history_valid_starts")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss-abs-tolerance", type=float, default=5e-3)
    parser.add_argument("--loss-rel-tolerance", type=float, default=2e-3)
    args = parser.parse_args(argv)
    if args.loss_abs_tolerance < 0 or args.loss_rel_tolerance < 0:
        parser.error("loss tolerances must be non-negative")
    return args


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must contain one JSON object: {path}")
    return value


def _batch_identity(report: dict) -> list[dict]:
    identities = []
    for batch in report["workload"]["batches"]:
        hashes = batch["fixture_hashes"]
        identities.append(
            {
                "compact_working_length": int(batch["compact_working_length"]),
                "computed_rows": int(batch["computed_rows"]),
                "supervised_rows": int(batch["supervised_rows"]),
                "router_gumbel_seed": int(batch["router_gumbel_seed"]),
                "fixture_hashes": {name: str(hashes[name]) for name in _HASH_KEYS},
            }
        )
    return identities


def _throughput(report: dict, median_ms: float) -> dict[str, float]:
    workload = report["workload"]
    scale = 1000.0 / median_ms
    return {
        "computed_rows_per_second": (
            float(workload["total_computed_rows_per_sample"]) * scale
        ),
        "supervised_rows_per_second": (
            float(workload["total_supervised_rows_per_sample"]) * scale
        ),
    }


def compare_reports(
    grouped: dict,
    triton: dict,
    *,
    loss_abs_tolerance: float = 5e-3,
    loss_rel_tolerance: float = 2e-3,
) -> dict:
    errors: list[str] = []
    for label, report, expected_backend in (
        ("grouped", grouped, "grouped"),
        ("triton", triton, "triton"),
    ):
        if report.get("schema_version") != "0089A-v1":
            errors.append(f"{label} report has unsupported schema")
        if report.get("status") != "measured":
            errors.append(f"{label} report is not an unprofiled measurement")
        if report.get("runtime", {}).get("attention_backend") != expected_backend:
            errors.append(f"{label} report does not use {expected_backend} attention")
        safety = report.get("safety", {})
        if any(
            bool(safety.get(name, True))
            for name in (
                "optimizer_constructed",
                "optimizer_update",
                "checkpoint_written",
                "weights_modified",
            )
        ):
            errors.append(f"{label} report failed the no-update safety contract")
    if not bool(triton.get("runtime", {}).get("attention_strict")):
        errors.append("Triton report was not run in strict mode")

    grouped_model = {name: grouped.get("model", {}).get(name) for name in _MODEL_KEYS}
    triton_model = {name: triton.get("model", {}).get(name) for name in _MODEL_KEYS}
    if grouped_model != triton_model:
        errors.append("model/checkpoint metadata differs")

    grouped_batches = _batch_identity(grouped)
    triton_batches = _batch_identity(triton)
    if grouped_batches != triton_batches:
        errors.append("sampled batch identities or fixture hashes differ")

    grouped_flags = grouped.get("runtime", {}).get("effective_flags", {})
    triton_flags = triton.get("runtime", {}).get("effective_flags", {})
    for name in grouped_flags:
        if name in {"MINI_LLM_TRAIN_ATTN_BACKEND", "MINI_LLM_TRAIN_ATTN_STRICT"}:
            continue
        if grouped_flags.get(name) != triton_flags.get(name):
            errors.append(f"effective runtime flag differs: {name}")

    grouped_loss = float(grouped["last_loss_device_resident_until_boundary"])
    triton_loss = float(triton["last_loss_device_resident_until_boundary"])
    loss_abs_error = abs(grouped_loss - triton_loss)
    loss_rel_error = loss_abs_error / max(abs(grouped_loss), abs(triton_loss), 1e-12)
    loss_ok = (
        math.isfinite(grouped_loss)
        and math.isfinite(triton_loss)
        and loss_abs_error <= loss_abs_tolerance
        and loss_rel_error <= loss_rel_tolerance
    )
    if not loss_ok:
        errors.append("final scalar loss exceeds the declared parity tolerances")

    grouped_ms = float(grouped["timing"]["gpu_event"]["median_ms"])
    triton_ms = float(triton["timing"]["gpu_event"]["median_ms"])
    if grouped_ms <= 0 or triton_ms <= 0:
        errors.append("GPU-event medians must be positive")

    valid = not errors
    speedup = grouped_ms / triton_ms if valid else None
    latency_reduction = (grouped_ms - triton_ms) / grouped_ms if valid else None
    return {
        "schema_version": "0089A-comparison-v1",
        "status": "valid" if valid else "invalid",
        "errors": errors,
        "parity": {
            "grouped_loss": grouped_loss,
            "triton_loss": triton_loss,
            "absolute_error": loss_abs_error,
            "relative_error": loss_rel_error,
            "absolute_tolerance": float(loss_abs_tolerance),
            "relative_tolerance": float(loss_rel_tolerance),
            "accepted": bool(loss_ok),
        },
        "matched_batch_identity": grouped_batches if grouped_batches == triton_batches else None,
        "timing": {
            "grouped_gpu_event": grouped["timing"]["gpu_event"],
            "triton_gpu_event": triton["timing"]["gpu_event"],
            "grouped_host_wall": grouped["timing"]["host_wall_outer_synchronized"],
            "triton_host_wall": triton["timing"]["host_wall_outer_synchronized"],
            "grouped_throughput": _throughput(grouped, grouped_ms),
            "triton_throughput": _throughput(triton, triton_ms),
            "speedup_grouped_over_triton": speedup,
            "latency_reduction_fraction": latency_reduction,
            "note": (
                "Complete no-optimizer model-body + chunked LM-head/loss + numerical "
                "backward; compilation, setup, zero_grad, and optimizer are excluded."
            ),
        },
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = compare_reports(
        _load(args.grouped),
        _load(args.triton),
        loss_abs_tolerance=args.loss_abs_tolerance,
        loss_rel_tolerance=args.loss_rel_tolerance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
