#!/usr/bin/env python3
"""Assistant-only SFT for a pretrained MyLocal_LLM checkpoint."""

import argparse
import dataclasses
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, validate_bfloat16_backend
from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm, _array_to_float


def parse_args():
    ap = argparse.ArgumentParser(description="Assistant-only supervised fine-tuning")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--data", required=True, help="Directory produced by prepare_smoltalk.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=16)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--peak-lr", type=float, default=5e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-interval", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_streams(data_dir: Path, split: str):
    tokens = np.memmap(data_dir / f"{split}_tokens.bin", mode="r", dtype=np.uint16)
    mask = np.memmap(data_dir / f"{split}_mask.bin", mode="r", dtype=np.uint8)
    if len(tokens) != len(mask):
        raise ValueError(f"{split} token/mask stream length mismatch")
    return tokens, mask


def sample_batch(tokens, mask, batch_size, seq_len, rng):
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError("SFT stream is shorter than one context window")

    # Resample windows with no supervised assistant targets. This is uncommon
    # but possible when a long user message spans an entire random window.
    inputs = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty((batch_size, seq_len), dtype=np.int64)
    loss_mask = np.empty((batch_size, seq_len), dtype=np.float32)
    for b in range(batch_size):
        for _ in range(100):
            start = int(rng.integers(0, max_start + 1))
            m = np.asarray(mask[start + 1:start + seq_len + 1], dtype=np.float32)
            if m.sum() > 0:
                inputs[b] = tokens[start:start + seq_len]
                targets[b] = tokens[start + 1:start + seq_len + 1]
                loss_mask[b] = m
                break
        else:
            raise RuntimeError("Could not sample a window containing assistant targets")
    return inputs, targets, loss_mask


def lr_at(step, warmup, total, peak, min_ratio):
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = peak * min_ratio
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * progress))


def copy_artifacts(base: Path, out: Path, config: ModelConfig):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    tok = base / "tokenizer.json"
    if tok.exists():
        shutil.copy2(tok, out / "tokenizer.json")


def save_sft(out, model, optimizer, step, seen_tokens, config, base):
    optimizer_state = {
        "step": optimizer.step_index,
        "master_weights": {p.name: optimizer.master_weights[i] for i, p in enumerate(model.parameters())},
        "m": {p.name: optimizer.m[i] for i, p in enumerate(model.parameters())},
        "v": {p.name: optimizer.v[i] for i, p in enumerate(model.parameters())},
    }
    save_checkpoint(
        out,
        {p.name: p.data for p in model.parameters()},
        optimizer_state=optimizer_state,
        training_state={"step": step, "tokens_processed": seen_tokens, "stage": "sft"},
    )
    copy_artifacts(base, out, config)


def eval_loss(model, tokens, mask, batch_size, seq_len, batches, rng):
    vals = []
    for _ in range(batches):
        x, y, m = sample_batch(tokens, mask, batch_size, seq_len, rng)
        logits, _ = model.forward(x)
        loss, _ = model.compute_loss(logits, y, loss_mask=m)
        vals.append(loss)
    return float(sum(vals) / len(vals))


def main():
    args = parse_args()
    base = Path(args.base_checkpoint)
    data_dir = Path(args.data)
    out = Path(args.output)

    with open(base / "config.json") as f:
        config = ModelConfig(**json.load(f))
    if str(config.dtype) == "bfloat16":
        validate_bfloat16_backend()
    if args.context_length > config.context_length:
        raise ValueError(
            f"SFT context {args.context_length} exceeds pretrained context {config.context_length}"
        )

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    names = [p.name for p in model.parameters()]
    loaded, _, _ = load_checkpoint(base, param_names=names, skip_optimizer=True, skip_training=True)
    for p in model.parameters():
        p.data[...] = loaded[p.name]
    print(f"Loaded {len(loaded)} parameter arrays from {base}")

    train_tokens, train_mask = load_streams(data_dir, "train")
    test_tokens, test_mask = load_streams(data_dir, "test")
    print(f"SFT train stream: {len(train_tokens):,} tokens")
    print(f"SFT test stream:  {len(test_tokens):,} tokens")

    optimizer = AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    seen_tokens = 0
    copy_artifacts(base, out, config)

    start_time = time.time()
    for step in range(args.steps):
        losses = []
        lr = lr_at(step, args.warmup_steps, args.steps, args.peak_lr, args.min_lr_ratio)
        optimizer.lr = lr

        for _ in range(args.grad_accum_steps):
            x, y, m = sample_batch(
                train_tokens, train_mask, args.batch_size, args.context_length, train_rng
            )
            logits, cache = model.forward(x)
            loss, loss_cache = model.compute_loss(logits, y, loss_mask=m)
            d_logits = model.backward_loss(loss_cache)
            model.backward(d_logits, cache)
            losses.append(loss)
            seen_tokens += args.batch_size * args.context_length

        if args.grad_accum_steps > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad[...] /= float(args.grad_accum_steps)

        grad_norm_backend, _, finite = clip_grad_global_norm(model.parameters(), args.grad_clip)
        grad_norm = float(_array_to_float(grad_norm_backend))
        if not finite:
            optimizer.zero_grad()
            raise ValueError(f"Nonfinite SFT gradient at step {step + 1}")

        optimizer.step(lr=lr)
        optimizer.zero_grad()
        avg_loss = float(sum(losses) / len(losses))
        done = step + 1

        if done % 10 == 0 or done == 1:
            elapsed = time.time() - start_time
            print(
                f"Step {done}/{args.steps}: assistant_loss={avg_loss:.4f}, "
                f"lr={lr:.2e}, grad_norm={grad_norm:.4f}, "
                f"{done / max(elapsed, 1e-9):.3f} steps/sec"
            )
        if done % args.val_interval == 0:
            val = eval_loss(
                model, test_tokens, test_mask, args.batch_size,
                args.context_length, args.val_batches, val_rng
            )
            print(f"  SFT validation assistant loss: {val:.4f}")
        if done % args.save_interval == 0:
            save_sft(out, model, optimizer, done, seen_tokens, config, base)

    save_sft(out, model, optimizer, args.steps, seen_tokens, config, base)
    print(f"SFT complete. Chat checkpoint: {out}")


if __name__ == "__main__":
    main()
