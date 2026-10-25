#!/usr/bin/env python3
"""Alternating A/B benchmark for the opt-in BF16 SwiGLU vec2 kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


def summary(samples: list[float]) -> dict:
    median = float(statistics.median(samples))
    return {
        "count": len(samples),
        "samples_ms": samples,
        "median_ms": median,
        "p10_ms": float(np.percentile(samples, 10)),
        "p90_ms": float(np.percentile(samples, 90)),
        "mad_ms": float(statistics.median(abs(value - median) for value in samples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3 * 2304)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.repeats) <= 0 or args.warmup < 0:
        parser.error("rows, cols, and repeats must be positive; warmup must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    os.environ["MINI_LLM_BACKEND"] = "cupy"
    os.environ["MINI_LLM_FUSED_SWIGLU"] = "1"

    import cupy as cp
    from mini_llm.ops.experts import _fused_swiglu_forward

    random = cp.random.RandomState(89)
    g = random.standard_normal((args.rows, args.cols)).astype(cp.float32).astype("bfloat16")
    u = random.standard_normal((args.rows, args.cols)).astype(cp.float32).astype("bfloat16")
    stream = cp.cuda.get_current_stream()

    def invoke(enabled: bool, measure: bool):
        os.environ["MINI_LLM_SWIGLU_VEC2"] = "1" if enabled else "0"
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        output = _fused_swiglu_forward(g, u)
        end.record(stream)
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
        if output is None:
            raise RuntimeError("fused BF16 SwiGLU path did not dispatch")
        if not measure:
            return output, None, None
        return output, float(cp.cuda.get_elapsed_time(start, end)), wall_ms

    baseline_output, _, _ = invoke(False, False)
    candidate_output, _, _ = invoke(True, False)
    parity = bool(cp.array_equal(baseline_output, candidate_output).item())
    if not parity:
        raise RuntimeError("vec2 SwiGLU output is not bitwise equal to the scalar kernel")
    del baseline_output, candidate_output

    for _ in range(args.warmup):
        output, _, _ = invoke(False, False)
        del output
        output, _, _ = invoke(True, False)
        del output

    gpu_samples = {"baseline": [], "candidate": []}
    wall_samples = {"baseline": [], "candidate": []}
    for sample_index in range(args.repeats):
        order = (False, True) if sample_index % 2 == 0 else (True, False)
        for enabled in order:
            output, gpu_ms, wall_ms = invoke(enabled, True)
            label = "candidate" if enabled else "baseline"
            gpu_samples[label].append(gpu_ms)
            wall_samples[label].append(wall_ms)
            del output

    baseline = summary(gpu_samples["baseline"])
    candidate = summary(gpu_samples["candidate"])
    report = {
        "schema_version": "0089L-swiglu-vec2-v1",
        "shape": [args.rows, args.cols],
        "dtype": "bfloat16",
        "parity": {"bitwise_equal": parity},
        "warmup": args.warmup,
        "alternating_pairs": args.repeats,
        "gpu_event": {"baseline": baseline, "candidate": candidate},
        "host_wall": {
            "baseline": summary(wall_samples["baseline"]),
            "candidate": summary(wall_samples["candidate"]),
        },
        "change": {
            "saved_ms": baseline["median_ms"] - candidate["median_ms"],
            "reduction_fraction": 1.0 - candidate["median_ms"] / baseline["median_ms"],
            "speedup": baseline["median_ms"] / candidate["median_ms"],
        },
        "memory": {
            "pool_used_bytes": int(cp.get_default_memory_pool().used_bytes()),
            "pool_reserved_bytes": int(cp.get_default_memory_pool().total_bytes()),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
