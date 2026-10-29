#!/usr/bin/env python3
"""Focused CUDA timing for bounded Model B global attention."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--repeat-cycles", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def measure(context, warmup_cycles, repeat_cycles):
    import cupy as cp
    from mini_llm.ops.triton_dual_resolution_dense import backward, forward

    blocks = context // 128
    summaries = blocks * 4
    rng = cp.random.RandomState(10210 + context)
    q = (rng.standard_normal((1, context, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    k = (rng.standard_normal((1, summaries, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    v = (rng.standard_normal((1, summaries, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    dout = rng.standard_normal(q.shape, dtype=cp.float32) * 0.1
    documents = cp.zeros((1, context), dtype=cp.int32)
    positions = cp.arange(context, dtype=cp.int32)[None]
    block_documents = cp.zeros((1, summaries), dtype=cp.int32)
    ends = cp.repeat(cp.arange(blocks, dtype=cp.int32) * 128 + 127, 4)[None]
    valid = cp.ones((1, summaries), dtype=cp.int32)

    def execute(mode, record):
        os.environ["MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED"] = str(mode)
        start_forward = cp.cuda.Event()
        end_forward = cp.cuda.Event()
        end_backward = cp.cuda.Event()
        start_forward.record()
        output, cache = forward(
            q, k, v, scale=0.125,
            documents=documents, positions=positions,
            block_documents=block_documents, block_ends=ends, block_valid=valid,
            bounded_causal=True, history_block_size=128, summaries_per_block=4,
        )
        end_forward.record()
        gradients = backward(dout, cache)
        end_backward.record()
        end_backward.synchronize()
        result = None
        if record:
            result = {
                "bounded": bool(mode),
                "forward_ms": float(cp.cuda.get_elapsed_time(start_forward, end_forward)),
                "backward_ms": float(cp.cuda.get_elapsed_time(end_forward, end_backward)),
                "total_ms": float(cp.cuda.get_elapsed_time(start_forward, end_backward)),
            }
        del output, cache, gradients
        return result

    order = (0, 1, 1, 0)
    for _ in range(warmup_cycles):
        for mode in order:
            execute(mode, False)
    samples = []
    for _ in range(repeat_cycles):
        for mode in order:
            samples.append(execute(mode, True))

    summary = {}
    for mode in (False, True):
        selected = [row for row in samples if row["bounded"] is mode]
        summary[str(int(mode))] = {
            "samples": len(selected),
            "forward_ms": median(row["forward_ms"] for row in selected),
            "backward_ms": median(row["backward_ms"] for row in selected),
            "total_ms": median(row["total_ms"] for row in selected),
        }
    return {"context": context, "summaries": summaries, "summary": summary, "samples": samples}


def main():
    args = parse_args()
    if os.environ.get("MINI_LLM_BACKEND") != "cupy":
        raise RuntimeError("set MINI_LLM_BACKEND=cupy")
    result = {
        "benchmark": "bounded-global-0102x",
        "warmup_cycles": args.warmup_cycles,
        "repeat_cycles": args.repeat_cycles,
        "contexts": [measure(context, args.warmup_cycles, args.repeat_cycles)
                     for context in args.contexts],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
