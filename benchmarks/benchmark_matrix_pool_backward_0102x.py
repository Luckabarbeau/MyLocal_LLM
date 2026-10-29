#!/usr/bin/env python3
"""Focused timing for the Model B fused matrix-pooling backward."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--repeat-cycles", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def measure(context, warmup_cycles, repeat_cycles):
    import cupy as cp
    from mini_llm.backend import RandomStream
    from mini_llm.config import ModelConfig
    from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention

    config = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    operator = MatrixPoolDualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01, rng=RandomStream(10220 + context),
        dtype="bfloat16", config=config,
    )
    rng = np.random.default_rng(10230 + context)
    hidden = cp.asarray(rng.normal(scale=0.1, size=(1, context, 1024)).astype("float32"))
    upstream = cp.asarray(
        rng.normal(scale=0.1, size=(1, context // 128, 4, 1024)).astype("float32")
    )
    metadata = operator._prepare_metadata(None, 1, context, None)
    _, cache = operator._pool_forward(hidden, metadata, True)
    cache["block_indices"] = metadata["block_indices"][cache["rows"], cache["slots"]]

    def execute(mode, record):
        os.environ["MINI_LLM_MATRIX_POOL_R4_LOGITS_BACKWARD"] = str(mode)
        operator.zero_grad()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        dx = operator._pool_backward(upstream, cache, hidden.shape, True)
        end.record()
        end.synchronize()
        result = None
        if record:
            result = {
                "r4_logits": bool(mode),
                "backward_ms": float(cp.cuda.get_elapsed_time(start, end)),
            }
        del dx
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
        selected = [row["backward_ms"] for row in samples if row["r4_logits"] is mode]
        summary[str(int(mode))] = {
            "samples": len(selected),
            "backward_ms": median(selected),
        }
    return {"context": context, "blocks": context // 128,
            "summary": summary, "samples": samples}


def main():
    args = parse_args()
    if os.environ.get("MINI_LLM_BACKEND") != "cupy":
        raise RuntimeError("set MINI_LLM_BACKEND=cupy")
    os.environ["MINI_LLM_MATRIX_POOL_FUSED_BACKWARD"] = "1"
    result = {
        "benchmark": "matrix-pool-backward-0102x",
        "contexts": [measure(context, args.warmup_cycles, args.repeat_cycles)
                     for context in args.contexts],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
