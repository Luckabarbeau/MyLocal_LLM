#!/usr/bin/env python3
"""0059I scaling benchmark for terminal-Landmark external-memory inference.

The benchmark answers the architectural question directly: as the addressable
history grows from 4k to 64k, does autoregressive decode remain bounded by the
4k Transformer working cache plus the fixed top-k historical read?

It uses deterministic synthetic token IDs so tokenizer/text sampling overhead
cannot affect the result.  The 4k row is the ordinary sparse KV-cache baseline;
16k/32k/64k rows activate the 0058C external-memory store.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.terminal_landmark_decode import terminal_memory_decode_failure_reason


def parse_horizons(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values or any(v <= 0 for v in values):
        raise ValueError("--horizons must contain positive integers")
    return values


def load_models(checkpoint: Path):
    with open(checkpoint / "config.json", "r") as f:
        config = ModelConfig(**json.load(f))
    training = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in training.parameters()]
    loaded, _, _ = load_checkpoint(checkpoint, param_names=names, skip_optimizer=True)
    for p in training.parameters():
        if p.name in loaded:
            p.data[...] = loaded[p.name]
    inference = InferenceModel(config, dtype=config.dtype)
    inference.set_weights(training)
    return config, training, inference, len(loaded)


def synthetic_ids(vocab_size, length, seed):
    rng = np.random.default_rng(int(seed))
    low = 8 if vocab_size > 16 else 0
    host = rng.integers(low, vocab_size, size=(1, int(length)), dtype=np.int32)
    return xp.asarray(host, dtype=xp.int32)


def elapsed_call(fn):
    synchronize()
    host0 = time.perf_counter()
    if BACKEND_NAME == "cupy":
        start = xp.cuda.Event(); end = xp.cuda.Event(); start.record()
    else:
        start = end = None
    result = fn()
    if BACKEND_NAME == "cupy":
        end.record()
    synchronize()
    host_ms = (time.perf_counter() - host0) * 1e3
    gpu_ms = float(xp.cuda.get_elapsed_time(start, end)) if BACKEND_NAME == "cupy" else host_ms
    return result, gpu_ms, host_ms


def cache_mib(state):
    total = int(state.k_cache.nbytes + state.v_cache.nbytes)
    if state.router_input_cache is not None:
        total += int(state.router_input_cache.nbytes)
    return total / (1024 ** 2)


def external_mib(state):
    store = state.terminal_memory_store
    if store is None:
        return 0.0
    arrays = (
        store.history_token_proj, store.history_proj,
        store.memory_k, store.memory_v,
    )
    return sum(int(x.nbytes) for x in arrays if x is not None) / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--horizons", default="4096,16384,32768,65536")
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    config, _, model, loaded = load_models(checkpoint)
    mem = getattr(config, "memory_context", None)
    if mem is None or not mem.enabled or mem.integration_mode != "terminal_landmark":
        raise ValueError("benchmark requires a terminal_landmark checkpoint")

    horizons = parse_horizons(args.horizons)
    if max(horizons) > int(mem.memory_length):
        raise ValueError(
            f"requested horizon {max(horizons):,} exceeds trained memory horizon {mem.memory_length:,}"
        )

    print("=== 0059I terminal-memory inference scaling ===")
    print(f"Backend: {BACKEND_NAME}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Model: {config.d_model}d, {config.n_layers} layers, vocab={config.vocab_size:,}")
    print(f"Loaded arrays: {loaded}")
    print(f"Transformer working length: {int(mem.target_length):,}")
    print(f"External history capacity: {int(mem.distant_memory_length):,}")
    print(f"Memory blocks/top-k/exact tokens: {int(mem.searchable_blocks)}/{int(mem.top_k_blocks)}/{int(mem.top_k_blocks * mem.block_size):,}")
    print()
    print(
        f"{'horizon':>9}  {'mode':>8}  {'blocks':>6}  {'KV/router':>10}  {'external':>10}  "
        f"{'prefill GPU':>12}  {'decode GPU':>11}  {'GPU tok/s':>9}  {'wall tok/s':>10}"
    )

    for row, horizon in enumerate(horizons):
        state = model.create_generation_state(1, int(horizon))
        # The ordinary 4k cache is non-sliding by design, so reserve enough
        # positions for the timed decode.  Long-memory states are sliding and can
        # prefill the complete addressable horizon before generation continues.
        reserve = int(args.warmup_steps) + int(args.decode_steps)
        prompt_length = int(horizon) if state.long_memory_enabled else int(horizon) - reserve
        if prompt_length <= 0:
            raise ValueError("4k baseline horizon is too small for requested warmup/decode steps")
        ids = synthetic_ids(config.vocab_size, prompt_length, args.seed + row)
        _, prefill_gpu, prefill_host = elapsed_call(lambda: model.prefill(ids, state))

        # Decode deterministic synthetic IDs. Warmup tokens intentionally advance
        # the sliding state, exercising eviction and the external block builder.
        warm_tokens = [((args.seed + i * 17) % (config.vocab_size - 8)) + 8 for i in range(args.warmup_steps)]
        for tok in warm_tokens:
            model.decode_one(xp.asarray([[tok]], dtype=xp.int32), state)
        synchronize()

        tokens = [((args.seed + 1000 + i * 29) % (config.vocab_size - 8)) + 8 for i in range(args.decode_steps)]
        synchronize()
        host0 = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start = xp.cuda.Event(); end = xp.cuda.Event(); start.record()
        else:
            start = end = None
        for tok in tokens:
            model.decode_one(xp.asarray([[tok]], dtype=xp.int32), state)
        if BACKEND_NAME == "cupy":
            end.record()
        synchronize()
        host_s = time.perf_counter() - host0
        gpu_ms_total = float(xp.cuda.get_elapsed_time(start, end)) if BACKEND_NAME == "cupy" else host_s * 1e3
        gpu_ms = gpu_ms_total / max(len(tokens), 1)
        wall_ms = host_s * 1e3 / max(len(tokens), 1)
        blocks = state.terminal_memory_store.active_blocks if state.terminal_memory_store is not None else 0
        mode = "external" if state.long_memory_enabled else "4k-base"
        print(
            f"{horizon:9,d}  {mode:>8}  {blocks:6d}  {cache_mib(state):9.1f}M  {external_mib(state):9.1f}M  "
            f"{prefill_gpu:9.1f} ms  {gpu_ms:8.3f} ms  {1000.0/gpu_ms:9.1f}  {1000.0/wall_ms:10.1f}"
        )
        del ids, state
        if BACKEND_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()

    reason = terminal_memory_decode_failure_reason()
    if reason:
        print(f"\nTerminal-memory fused-kernel diagnostic: {reason}")
    if BACKEND_NAME == "cupy":
        pool = xp.get_default_memory_pool()
        free, total = xp.cuda.runtime.memGetInfo()
        print(f"GPU pool after benchmark: used={pool.used_bytes()/2**30:.2f} GiB, reserved={pool.total_bytes()/2**30:.2f} GiB, device_free={free/2**30:.2f}/{total/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
