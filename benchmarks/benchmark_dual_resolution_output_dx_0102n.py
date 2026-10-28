#!/usr/bin/env python3
"""Paired full-model A/B benchmark for the 0102N BF16 output-projection dX."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compression-dim", type=int, default=512)
    parser.add_argument("--score-last-k", type=int, default=128)
    parser.add_argument("--head-chunk-tokens", type=int, default=64)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--repeat-pairs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.context < 129 or args.context > 65536:
        parser.error("context must be in [129, 65536]")
    if min(args.layers, args.batch_size, args.compression_dim,
           args.score_last_k, args.head_chunk_tokens, args.repeat_pairs) < 1:
        parser.error("layers, sizes, and repeat-pairs must be positive")
    if args.warmup_pairs < 0:
        parser.error("warmup-pairs must be nonnegative")
    return args


def run(args):
    os.environ["MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL"] = "triton"
    os.environ["MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL"] = "triton"
    os.environ["MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL"] = "triton"

    from mini_llm.backend import BACKEND_NAME, synchronize, xp
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata

    if BACKEND_NAME != "cupy":
        raise RuntimeError("paired production benchmark requires MINI_LLM_BACKEND=cupy")

    base = ModelConfig.dual_resolution_1024_20l()
    dual = replace(
        base.dual_resolution,
        compression_dim=args.compression_dim,
        global_query_chunk=512,
        compression_block_chunk=64,
    )
    config = replace(
        base,
        context_length=args.context,
        n_layers=args.layers,
        dual_resolution=dual,
        dtype="bfloat16",
    )
    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    model.refresh_compute_buffers()

    rng = np.random.default_rng(args.seed + 1)
    eos_id = int(config.vocab_size - 1)
    raw = rng.integers(
        0, eos_id, size=(args.batch_size, args.context + 1), dtype=np.int64
    )
    inputs = xp.asarray(raw[:, :-1], dtype=xp.int64)
    scored = min(args.score_last_k, args.context)
    targets = xp.asarray(raw[:, 1:][:, -scored:], dtype=xp.int64)
    metadata = build_dual_resolution_metadata(
        raw[:, :-1], eos_token_id=eos_id,
        block_size=config.dual_resolution.block_size,
    )

    def iteration(enabled):
        os.environ["MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX"] = str(int(enabled))
        model.zero_grad()
        synchronize()
        start = time.perf_counter()
        head, cache = model.forward_body(
            inputs, activation_checkpoint=True,
            dual_resolution_metadata=metadata,
        )
        synchronize()
        forward_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        loss, loss_cache = model.chunked_lm_head_loss_forward(
            head, targets, chunk_tokens=args.head_chunk_tokens,
            target_slice=(args.context - scored, args.context),
            return_device_loss=True,
        )
        loss_value = float(loss.item())
        synchronize()
        loss_ms = (time.perf_counter() - start) * 1000.0
        del head

        start = time.perf_counter()
        dx = model.chunked_lm_head_backward(loss_cache)
        model.backward_body(dx, cache)
        synchronize()
        backward_ms = (time.perf_counter() - start) * 1000.0
        del dx, cache, loss_cache
        return {
            "bf16_output_dx": bool(enabled),
            "loss": loss_value,
            "forward_ms": forward_ms,
            "loss_ms": loss_ms,
            "backward_ms": backward_ms,
            "total_ms": forward_ms + loss_ms + backward_ms,
        }

    for pair in range(args.warmup_pairs):
        order = (False, True) if pair % 2 == 0 else (True, False)
        for enabled in order:
            iteration(enabled)

    measured = []
    for pair in range(args.repeat_pairs):
        order = (False, True) if pair % 2 == 0 else (True, False)
        for enabled in order:
            row = iteration(enabled)
            row["pair"] = pair
            measured.append(row)

    def summary(enabled):
        rows = [row for row in measured if row["bf16_output_dx"] == enabled]
        total_ms = median(row["total_ms"] for row in rows)
        return {
            "forward_ms": median(row["forward_ms"] for row in rows),
            "loss_ms": median(row["loss_ms"] for row in rows),
            "backward_ms": median(row["backward_ms"] for row in rows),
            "total_ms": total_ms,
            "processed_tokens_per_s": args.context * args.batch_size / (total_ms / 1000.0),
        }

    reference = summary(False)
    optimized = summary(True)
    paired = []
    for pair in range(args.repeat_pairs):
        ref = next(row for row in measured if row["pair"] == pair
                   and not row["bf16_output_dx"])
        opt = next(row for row in measured if row["pair"] == pair
                   and row["bf16_output_dx"])
        paired.append({
            "pair": pair,
            "backward_delta_ms": ref["backward_ms"] - opt["backward_ms"],
            "total_delta_ms": ref["total_ms"] - opt["total_ms"],
            "backward_improvement_percent": 100.0 * (
                ref["backward_ms"] / opt["backward_ms"] - 1.0
            ),
            "total_improvement_percent": 100.0 * (
                ref["total_ms"] / opt["total_ms"] - 1.0
            ),
        })
    pool = xp.get_default_memory_pool()
    free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    result = {
        "benchmark": "dual-resolution-output-dx-0102n-paired",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "backend": BACKEND_NAME,
        "cupy_version": xp.__version__,
        "cuda_runtime_version": int(xp.cuda.runtime.runtimeGetVersion()),
        "seed": args.seed,
        "context": args.context,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "compression_dim": args.compression_dim,
        "scored_token_rows": scored * args.batch_size,
        "warmup_pairs": args.warmup_pairs,
        "repeat_pairs": args.repeat_pairs,
        "reference": reference,
        "optimized": optimized,
        "paired_median": {
            "backward_delta_ms": median(
                row["backward_delta_ms"] for row in paired
            ),
            "total_delta_ms": median(row["total_delta_ms"] for row in paired),
            "backward_improvement_percent": median(
                row["backward_improvement_percent"] for row in paired
            ),
            "total_improvement_percent": median(
                row["total_improvement_percent"] for row in paired
            ),
        },
        "paired_results": paired,
        "memory": {
            "pool_used_mib": pool.used_bytes() / 1048576.0,
            "pool_reserved_mib": pool.total_bytes() / 1048576.0,
            "device_used_mib": (total_bytes - free_bytes) / 1048576.0,
        },
        "iterations": measured,
        "note": "Same initialized model and inputs; A/B order alternates within each pair.",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
