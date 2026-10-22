#!/usr/bin/env python3
"""Forward-only A/B for the opt-in 0086B fused local-attention kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seq", type=int, default=2304)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--fused-tile-n", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("local-all", "layer0-history-valid", "layer4-history-valid"),
        default=("local-all", "layer0-history-valid", "layer4-history-valid"),
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _selection(heads):
    if heads == tuple(range(heads[0], heads[0] + len(heads))):
        return slice(heads[0], heads[0] + len(heads))
    return heads


def _event_call(call, stream):
    import cupy as cp

    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    output = call()
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)), output


def _time_pair(grouped, fused, warmup, repeats):
    import cupy as cp

    for _ in range(warmup):
        grouped()
        fused()
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    grouped_ms, fused_ms = [], []
    for repeat in range(repeats):
        order = ((grouped, grouped_ms), (fused, fused_ms))
        if repeat & 1:
            order = tuple(reversed(order))
        for call, samples in order:
            elapsed, _ = _event_call(call, stream)
            samples.append(elapsed)
    return {
        "grouped": {
            "median_ms": float(np.median(grouped_ms)),
            "p10_ms": float(np.percentile(grouped_ms, 10)),
            "p90_ms": float(np.percentile(grouped_ms, 90)),
        },
        "fused": {
            "median_ms": float(np.median(fused_ms)),
            "p10_ms": float(np.percentile(fused_ms, 10)),
            "p90_ms": float(np.percentile(fused_ms, 90)),
        },
    }


def _metrics(actual, expected):
    import cupy as cp

    difference = actual - expected
    expected_norm = cp.linalg.norm(expected.reshape(-1))
    return {
        "max_abs": float(cp.max(cp.abs(difference)).item()),
        "rmse": float(cp.sqrt(cp.mean(difference * difference)).item()),
        "relative_l2": float((cp.linalg.norm(difference.reshape(-1)) / expected_norm).item()),
        "finite": bool(cp.all(cp.isfinite(actual)).item()),
    }


def main():
    args = _args()
    os.environ.setdefault("MINI_LLM_BACKEND", "cupy")
    os.environ.setdefault("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_CUBLAS_MODE", "grouped")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS", "1")
    os.environ.setdefault("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "auto")
    os.environ.setdefault("MINI_LLM_BF16_MIXED_CONTEXT", "0")
    os.environ["MINI_LLM_FUSED_LOCAL_TILE_N"] = str(args.fused_tile_n)

    import cupy as cp

    from mini_llm.config import ModelConfig
    from mini_llm.ops.fused_local_attention import fused_local_kernel_attributes
    from mini_llm.ops.indexed_attention import local_window_attention_forward
    from mini_llm.runtime_defaults import apply_runtime_defaults

    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    rng = np.random.default_rng(args.seed)
    q_all = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 16, 64)).astype(np.float32),
        dtype="bfloat16",
    )
    k = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, 64)).astype(np.float32),
        dtype="bfloat16",
    )
    v = cp.asarray(
        rng.normal(0.0, 0.2, (args.batch, args.seq, 4, 64)).astype(np.float32),
        dtype="bfloat16",
    )
    definitions = {
        "local-all": tuple(range(1, 16)),
        "layer0-history-valid": tuple(h for h in range(1, 16) if h not in (12, 13)),
        "layer4-history-valid": tuple(range(1, 10)),
    }
    results = []
    for name in args.cases:
        heads = definitions[name]
        q = q_all[:, :, _selection(heads), :]
        mapping = tuple(head // 4 for head in heads)

        def grouped_call():
            os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = "grouped"
            return local_window_attention_forward(
                q, k, v, 512, kv_head_indices=mapping, return_cache=False
            )

        def fused_call():
            os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = "fused"
            os.environ["MINI_LLM_TRAIN_ATTN_STRICT"] = "1"
            return local_window_attention_forward(
                q, k, v, 512, kv_head_indices=mapping, return_cache=False
            )

        grouped_output = grouped_call()
        fused_output = fused_call()
        cp.cuda.get_current_stream().synchronize()
        timing = _time_pair(
            grouped_call, fused_call, args.warmup, args.repeats
        )
        speedup = timing["grouped"]["median_ms"] / timing["fused"]["median_ms"]
        results.append({
            "case": name,
            "original_heads": list(heads),
            "kv_map": list(mapping),
            "q_shape": list(map(int, q.shape)),
            "q_strides_bytes": list(map(int, q.strides)),
            "q_c_contiguous": bool(q.flags.c_contiguous),
            "timing": timing,
            "grouped_over_fused": float(speedup),
            "numerical_fused_vs_grouped": _metrics(fused_output, grouped_output),
        })

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties.get("name")
    report = {
        "patch": "0086B",
        "backend": "cupy",
        "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
        "batch": args.batch,
        "seq": args.seq,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "fused_tile_n": args.fused_tile_n,
        "kernel_attributes": fused_local_kernel_attributes(),
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
