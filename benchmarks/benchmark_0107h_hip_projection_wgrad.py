#!/usr/bin/env python3
"""HIP projection wgrad: full-8K timings including operand casts/decomposition.

Run via scripts/run_r9700_preflight.sh. No model/checkpoint is loaded or changed.
Default shapes cover square and narrower input projections, not model dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["8192,1024,1024", "8192,256,1024"],
                        help="T,Din,Dout triples")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shapes = []
    for spec in args.shapes:
        shape = tuple(int(value) for value in spec.split(","))
        if len(shape) != 3 or min(shape) <= 0:
            parser.error("shapes must be positive T,Din,Dout triples")
        shapes.append(shape)
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")

    import cupy as cp
    from mini_llm.ops.hip_projection_wgrad import _split_gradient, projection_wgrad

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("requires HIP, not CUDA")
    rng = np.random.default_rng(107)
    stream = cp.cuda.get_current_stream()

    def measure(fn):
        for _ in range(args.warmup):
            result = fn()
        stream.synchronize()
        device_ms, wall_ms = [], []
        start, end = cp.cuda.Event(), cp.cuda.Event()
        for _ in range(args.repeats):
            before = perf_counter()
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            wall_ms.append((perf_counter() - before) * 1000)
            device_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
        return {"device_median_ms": median(device_ms), "device_samples_ms": device_ms,
                "wall_median_ms": median(wall_ms), "wall_samples_ms": wall_ms}, result

    records = []
    for t, din, dout in shapes:
        a = cp.asarray(rng.normal(0, 0.1, (t, din)), dtype=cp.float32).astype("bfloat16")
        original_g = cp.asarray(rng.normal(0, 0.1, (t, dout)), dtype=cp.float32)
        for dtype in ("bfloat16", "float32"):
            g = original_g.astype(dtype, copy=False)

            def baseline():
                return a.astype(cp.float32).T @ g.astype(cp.float32, copy=False)

            reference_timing, reference = measure(baseline)
            candidate_timing, actual = measure(lambda: projection_wgrad(a, g))
            got = cp.asnumpy(actual).astype(np.float64)
            want = cp.asnumpy(reference).astype(np.float64)
            delta = np.abs(got - want)
            worst = np.unravel_index(int(np.argmax(delta)), delta.shape)
            error = {
                "max_absolute": float(delta.max()),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(want), 1e-300)),
                "max_absolute_over_reference_max": float(delta.max() / max(np.abs(want).max(), 1e-300)),
                "max_pointwise_relative": float((delta / np.maximum(np.abs(want), 1e-300)).max()),
                "worst_actual": float(got[worst]), "worst_reference": float(want[worst]),
                "allclose_atol_rtol_1e_5": bool(np.allclose(got, want, atol=1e-5, rtol=1e-5)),
                "dtype": "float32", "eps": float(np.finfo(np.float32).eps),
                "finite_difference_step": None,
            }
            reconstruction = None
            if dtype == "float32":
                parts = _split_gradient(g)
                reconstructed = sum(cp.asnumpy(part.astype(cp.float32)).astype(np.float64)
                                    for part in parts).astype(np.float32)
                source = cp.asnumpy(g)
                difference = np.abs(reconstructed.astype(np.float64) - source.astype(np.float64))
                mismatch = reconstructed != source
                reconstruction = {
                    "count": source.size, "mismatches": int(mismatch.sum()),
                    "max_absolute": float(difference.max()),
                    "max_relative": float((difference / np.maximum(np.abs(source), 1e-300)).max()),
                    "rounding": "sum BF16 components in FP64 then round once to FP32",
                }
                if np.any(mismatch):
                    index = np.unravel_index(int(np.argmax(difference)), source.shape)
                    reconstruction.update(original=float(source[index]),
                                          reconstructed=float(reconstructed[index]))
                del parts, reconstructed, source, difference, mismatch
            record = {
                "shape_T_Din_Dout": [t, din, dout], "gradient_dtype": dtype,
                "baseline": reference_timing, "candidate": candidate_timing,
                "device_speedup": reference_timing["device_median_ms"] / candidate_timing["device_median_ms"],
                "wall_speedup": reference_timing["wall_median_ms"] / candidate_timing["wall_median_ms"],
                "error_vs_cupy_fp32": error, "gradient_reconstruction": reconstruction,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        del a, original_g, g, actual, reference

    result = {
        "hip": True, "cupy_version": cp.__version__, "seed": 107, "normal_std": 0.1,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "warmup": args.warmup, "repeats": args.repeats, "records": records,
        "scope": "whole helper including casts, decomposition, allocation and all GEMMs; device events and synchronized wall latency",
        "baseline": "activation.astype(FP32).T @ gradient.astype(FP32, copy=False)",
        "correctness": all(record["error_vs_cupy_fp32"]["allclose_atol_rtol_1e_5"]
                           and (record["gradient_reconstruction"] is None
                                or record["gradient_reconstruction"]["mismatches"] == 0)
                           for record in records),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["correctness"]:
        raise SystemExit("projection wgrad accuracy/reconstruction failed; see exact errors above")


if __name__ == "__main__":
    main()
