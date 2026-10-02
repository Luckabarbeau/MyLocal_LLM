#!/usr/bin/env python3
"""Evaluate a Mini-LLM checkpoint on WikiText-2 raw text.

Reports token cross-entropy, perplexity, bits per byte, token/byte ratio, and
throughput. The benchmark is intentionally simple and reproducible: the raw
WikiText-2 test text is concatenated, tokenized with the checkpoint tokenizer,
and evaluated left-to-right in windows no larger than the model context.

Examples:
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/cosmopedia_medium_bf16 \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw

    # Overlapping windows: preserve more context while scoring each token once.
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/cosmopedia_medium_bf16 \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
        --stride 256
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, synchronize
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer


def load_tokenizer(checkpoint: Path):
    tokenizer_path = checkpoint / "tokenizer.json"
    if not tokenizer_path.exists():
        fallback = checkpoint.parent / "tokenizer.json"
        if fallback.exists():
            tokenizer_path = fallback
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {checkpoint / 'tokenizer.json'} "
                f"or {fallback}"
            )

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    is_simple = (
        "vocab_size" in tokenizer_config
        and "token_to_id" in tokenizer_config
    )
    tokenizer = (
        SimpleBPETokenizer.load(str(tokenizer_path))
        if is_simple
        else FastBPETokenizer.load(str(tokenizer_path))
    )
    return tokenizer, tokenizer_path


def load_model(checkpoint: Path):
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = ModelConfig(**json.load(f))

    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in model.parameters()]
    loaded, _, _ = load_checkpoint(
        checkpoint,
        param_names=names,
        skip_optimizer=True,
        skip_training=True,
    )

    missing = []
    for p in model.parameters():
        value = loaded.get(p.name)
        if value is None:
            missing.append(p.name)
        else:
            p.data[...] = value

    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} parameter arrays; "
            f"first missing parameter: {missing[0]}"
        )

    return model, config


def suffix_nll(logits, targets, suffix_start: int):
    """Return summed NLL for targets[suffix_start:] using stable FP32 math."""
    vocab = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab)[suffix_start:]
    flat_targets = targets.reshape(-1)[suffix_start:]
    n = int(flat_targets.size)
    if n == 0:
        return 0.0, 0

    work = flat_logits.astype(xp.float32, copy=True)
    max_logits = xp.max(work, axis=-1, keepdims=True)
    work -= max_logits

    rows = xp.arange(n)
    target_shifted = work[rows, flat_targets]
    xp.exp(work, out=work)
    log_normalizer = xp.log(xp.sum(work, axis=-1))
    nll = xp.sum(log_normalizer - target_shifted)

    if hasattr(nll, "get"):
        nll = float(nll.get())
    else:
        nll = float(nll)
    return nll, n


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the WikiText-2 raw test set."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to wiki.test.raw")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Number of new target tokens scored per window. Defaults to the "
            "model context length. Use a smaller value such as 256 to give "
            "most tokens more left context while still scoring each token once."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional quick-test limit on tokenizer tokens (default: full test set).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N evaluation windows (default: 50).",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"WikiText file not found: {args.data}")

    checkpoint = args.checkpoint
    tokenizer, tokenizer_path = load_tokenizer(checkpoint)
    model, config = load_model(checkpoint)

    context = int(config.context_length)
    stride = context if args.stride is None else int(args.stride)
    if stride < 1 or stride > context:
        raise ValueError(f"--stride must be in [1, {context}], got {stride}")

    raw_text = args.data.read_text(encoding="utf-8")
    raw_bytes = len(raw_text.encode("utf-8"))

    print("Tokenizing WikiText-2 test set...")
    token_ids = tokenizer.encode(raw_text)
    if args.max_tokens is not None:
        limit = max(2, int(args.max_tokens))
        token_ids = token_ids[:limit]

    if len(token_ids) < 2:
        raise RuntimeError("WikiText input produced fewer than 2 tokens")

    print("=" * 60)
    print("WikiText-2 Benchmark")
    print("=" * 60)
    print(f"Checkpoint:       {checkpoint}")
    print(f"Tokenizer:        {tokenizer_path}")
    print(f"Model dtype:      {config.dtype}")
    print(f"Context length:   {context}")
    print(f"Stride:           {stride}")
    print(f"Raw bytes:        {raw_bytes:,}")
    print(f"Tokenizer tokens: {len(token_ids):,}")
    print()

    total_nll = 0.0
    total_scored = 0
    windows = 0

    synchronize()
    start_time = time.perf_counter()

    # target_start is the global token index of the first token to score in
    # this window. Every token from index 1 onward is scored exactly once.
    for target_start in range(1, len(token_ids), stride):
        target_end = min(target_start + stride, len(token_ids))

        # Need one predecessor token plus up to context target positions.
        begin = max(0, target_end - (context + 1))
        seq = token_ids[begin:target_end]
        inputs_np = np.asarray(seq[:-1], dtype=np.int64)[None, :]
        targets_np = np.asarray(seq[1:], dtype=np.int64)[None, :]

        inputs = xp.asarray(inputs_np)
        targets = xp.asarray(targets_np)

        # Global target indices represented by targets start at begin + 1.
        suffix_start = target_start - (begin + 1)
        if suffix_start < 0 or suffix_start >= targets_np.shape[1]:
            raise RuntimeError(
                f"Internal windowing error: suffix_start={suffix_start}, "
                f"targets={targets_np.shape[1]}"
            )

        logits, cache = model.forward(inputs)
        nll, count = suffix_nll(logits, targets, suffix_start)
        total_nll += nll
        total_scored += count
        windows += 1

        # Release large forward caches before the next window.
        del logits, cache, inputs, targets

        if args.progress_every > 0 and windows % args.progress_every == 0:
            running_loss = total_nll / total_scored
            print(
                f"window {windows:5d} | scored {total_scored:9,d} tokens | "
                f"loss {running_loss:.4f} | ppl {math.exp(running_loss):.2f}"
            )

    synchronize()
    elapsed = time.perf_counter() - start_time

    loss = total_nll / total_scored
    ppl = math.exp(loss)

    # If --max-tokens is used, BPB against the full file would be misleading.
    # Report BPB only for a complete evaluation.
    bpb = None
    if args.max_tokens is None:
        bpb = total_nll / (raw_bytes * math.log(2.0))

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Scored tokens:    {total_scored:,}")
    print(f"Cross-entropy:    {loss:.6f} nats/token")
    print(f"Perplexity:       {ppl:.4f}")
    if bpb is not None:
        print(f"Bits per byte:    {bpb:.6f}")
        print(f"Tokens per byte:  {len(token_ids) / raw_bytes:.6f}")
    else:
        print("Bits per byte:    n/a (--max-tokens was used)")
    print(f"Eval windows:     {windows:,}")
    print(f"Elapsed:          {elapsed:.2f} s")
    print(f"Scored tok/s:     {total_scored / elapsed:,.1f}")


if __name__ == "__main__":
    main()
