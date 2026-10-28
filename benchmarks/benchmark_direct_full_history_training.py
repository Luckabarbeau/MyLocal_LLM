#!/usr/bin/env python3
"""Matched training-microbatch speed probe for direct 64K head replacement."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import statistics
import time

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101_001)
    parser.add_argument("--lm-head-chunk", type=int, default=8192)
    parser.add_argument(
        "--activation-checkpoint", choices=("none", "block"),
        default=os.environ.get("MINI_LLM_ACTIVATION_CHECKPOINT", "block").lower(),
    )
    parser.add_argument(
        "--runtime-profile", choices=("gpu-fast", "consumer-gpu", "reference"),
        default="consumer-gpu",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("batch-size/repeats must be positive and warmup non-negative")
    return args


def _batch(config, batch_size, seed):
    rng = np.random.default_rng(seed)
    if config.memory_context.enabled:
        source_length = int(config.memory_context.source_input_length)
        target_length = int(config.memory_context.target_length)
        history_length = int(config.memory_context.distant_memory_length)
        inputs = rng.integers(
            0, config.vocab_size, (batch_size, source_length), dtype=np.int32
        )
        metadata = {
            "history_valid_starts": np.zeros((batch_size,), dtype=np.int64),
            "target_valid_lengths": np.full(
                (batch_size,), target_length, dtype=np.int64
            ),
        }
    else:
        target_length = int(config.context_length)
        history_length = 0
        inputs = rng.integers(
            0, config.vocab_size, (batch_size, target_length), dtype=np.int32
        )
        metadata = None
    targets = rng.integers(
        0, config.vocab_size, (batch_size, target_length), dtype=np.int32
    )
    mask = np.ones((batch_size, target_length), dtype=np.float32)
    return inputs, targets, mask, metadata, history_length


def _one_microbatch(model, batch, chunk_tokens, activation_checkpoint, cp):
    inputs, targets, loss_mask, metadata, _ = batch
    model.zero_grad()
    stream = cp.cuda.get_current_stream()
    start = cp.cuda.Event()
    after_forward = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    head_input, cache = model.forward_body(
        inputs,
        activation_checkpoint=activation_checkpoint,
        memory_metadata=metadata,
    )
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        head_input,
        targets,
        chunk_tokens=chunk_tokens,
        loss_mask=loss_mask,
        return_device_loss=True,
        target_slice=cache.get("target_slice"),
    )
    after_forward.record(stream)
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    end.record(stream)
    end.synchronize()
    result = {
        "loss": float(loss.item()),
        "forward_ms": float(cp.cuda.get_elapsed_time(start, after_forward)),
        "backward_ms": float(cp.cuda.get_elapsed_time(after_forward, end)),
        "combined_ms": float(cp.cuda.get_elapsed_time(start, end)),
    }
    del head_input, loss_cache, dx, cache, loss
    return result


def _measure(name, config, args, cp):
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    model.refresh_compute_buffers()
    batch = _batch(config, args.batch_size, args.seed + 1)
    checkpoint_blocks = args.activation_checkpoint == "block"
    for _ in range(args.warmup):
        _one_microbatch(
            model, batch, args.lm_head_chunk, checkpoint_blocks, cp
        )
    samples = [
        _one_microbatch(
            model, batch, args.lm_head_chunk, checkpoint_blocks, cp
        )
        for _ in range(args.repeats)
    ]
    tokens = args.batch_size * int(
        config.memory_context.target_length
        if config.memory_context.enabled else config.context_length
    )
    medians = {
        key: statistics.median(sample[key] for sample in samples)
        for key in ("forward_ms", "backward_ms", "combined_ms")
    }
    result = {
        "name": name,
        "parameters": sum(parameter.size for parameter in model.parameters()),
        "history_tokens_per_sample": batch[4],
        "supervised_tokens_per_microbatch": tokens,
        "activation_checkpoint": args.activation_checkpoint,
        "samples": samples,
        "median": medians,
        "supervised_tokens_per_second": tokens * 1000.0 / medians["combined_ms"],
    }
    del model, batch
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main():
    args = _parse_args()
    import cupy as cp
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.config import MemoryContextConfig, ModelConfig
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError("set MINI_LLM_BACKEND=cupy")

    direct = replace(
        ModelConfig.ensemble_1024_20l_dense_direct_full_history_64k(),
        dtype="bfloat16",
    )
    baseline = replace(
        ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k(),
        context_length=4096,
        dtype="bfloat16",
        memory_context=MemoryContextConfig(enabled=False),
        full_history_layers=(),
    )
    apply_runtime_defaults(
        args.runtime_profile, backend_name=BACKEND_NAME, config=direct,
        training=True, precision="bf16-mixed",
    )

    started = time.time()
    baseline_result = _measure("4K full head + 15 local", baseline, args, cp)
    direct_result = _measure("64K direct head + 15 local", direct, args, cp)
    report = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runtime_profile": args.runtime_profile,
        "lm_head_chunk": args.lm_head_chunk,
        "activation_checkpoint": args.activation_checkpoint,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM",
                "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD",
                "MINI_LLM_HISTORY_RECOMPUTE",
                "MINI_LLM_HISTORY_QUERY_CHUNK",
                "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS",
                "MINI_LLM_FUSED_EXPERT_GEMM",
                "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK",
                "MINI_LLM_LM_HEAD_CHUNK_TOKENS",
                "MINI_LLM_OPTIMIZER_OFFLOAD",
                "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB",
                "MINI_LLM_AR_COMPACT_TARGET",
                "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS",
            )
        },
        "baseline": baseline_result,
        "direct_full_history": direct_result,
        "combined_time_ratio": (
            direct_result["median"]["combined_ms"]
            / baseline_result["median"]["combined_ms"]
        ),
        "throughput_ratio": (
            direct_result["supervised_tokens_per_second"]
            / baseline_result["supervised_tokens_per_second"]
        ),
        "wall_seconds": time.time() - started,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
