#!/usr/bin/env python3
"""Prepare HuggingFaceTB/smol-smoltalk for assistant-only SFT.

The prepared format is deliberately simple and framework-independent:
  <split>_tokens.bin : contiguous uint16 token stream
  <split>_mask.bin   : uint8 mask aligned with tokens; 1 means that token is
                       part of an assistant answer and should be predicted.

No new tokenizer tokens are added, so a pretrained checkpoint remains exactly
compatible. Role markers are ordinary text (System:/User:/Assistant:) and the
existing EOS token ID terminates every assistant answer.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer


def load_tokenizer(path: Path):
    try:
        return FastBPETokenizer.load(str(path))
    except Exception:
        return SimpleBPETokenizer.load(str(path))


def emit_text(tokenizer, text, supervised, token_file, mask_file, stats):
    ids = tokenizer.encode(text)
    if not ids:
        return
    a = np.asarray(ids, dtype=np.uint16)
    m = np.full(a.shape, 1 if supervised else 0, dtype=np.uint8)
    a.tofile(token_file)
    m.tofile(mask_file)
    stats["tokens"] += int(a.size)
    if supervised:
        stats["assistant_tokens"] += int(a.size)


def prepare_split(dataset, split, tokenizer, out_dir, max_examples=None):
    token_path = out_dir / f"{split}_tokens.bin"
    mask_path = out_dir / f"{split}_mask.bin"
    stats = {"examples": 0, "tokens": 0, "assistant_tokens": 0}
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]

    with open(token_path, "wb") as tf, open(mask_path, "wb") as mf:
        for row in dataset:
            if max_examples is not None and stats["examples"] >= max_examples:
                break
            messages = row.get("messages") or []
            for msg in messages:
                role = str(msg.get("role", "")).strip().lower()
                content = str(msg.get("content", ""))
                if role == "assistant":
                    emit_text(tokenizer, "Assistant:\n", False, tf, mf, stats)
                    emit_text(tokenizer, content, True, tf, mf, stats)
                    np.asarray([eos_id], dtype=np.uint16).tofile(tf)
                    np.asarray([1], dtype=np.uint8).tofile(mf)
                    stats["tokens"] += 1
                    stats["assistant_tokens"] += 1
                    emit_text(tokenizer, "\n\n", False, tf, mf, stats)
                elif role == "system":
                    emit_text(tokenizer, "System:\n" + content + "\n\n", False, tf, mf, stats)
                else:
                    # Treat unknown human-like roles conservatively as user context.
                    emit_text(tokenizer, "User:\n" + content + "\n\n", False, tf, mf, stats)
            stats["examples"] += 1
            if stats["examples"] % 10000 == 0:
                print(
                    f"{split}: {stats['examples']:,} conversations | "
                    f"{stats['tokens']:,} tokens | "
                    f"{stats['assistant_tokens']:,} supervised"
                )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Base checkpoint containing tokenizer.json")
    ap.add_argument("--output", default="./sft_data/smol-smoltalk")
    ap.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--max-test-examples", type=int, default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The preparation script needs Hugging Face datasets. Install once with:\n"
            "  python -m pip install 'datasets>=3' pyarrow\n"
        ) from exc

    checkpoint = Path(args.checkpoint)
    tokenizer_path = checkpoint / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Loading {args.dataset}...")
    train_ds = load_dataset(args.dataset, split="train")
    test_ds = load_dataset(args.dataset, split="test")

    train_stats = prepare_split(
        train_ds, "train", tokenizer, out_dir, args.max_train_examples
    )
    test_stats = prepare_split(
        test_ds, "test", tokenizer, out_dir, args.max_test_examples
    )

    manifest = {
        "dataset": args.dataset,
        "format": "assistant_only_token_stream_v1",
        "token_dtype": "uint16",
        "mask_dtype": "uint8",
        "chat_template": "System:/User:/Assistant: textual prefixes; EOS after assistant answers",
        "train": train_stats,
        "test": test_stats,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for split, st in [("train", train_stats), ("test", test_stats)]:
        frac = st["assistant_tokens"] / max(1, st["tokens"])
        print(
            f"{split}: {st['examples']:,} conversations, {st['tokens']:,} tokens, "
            f"{st['assistant_tokens']:,} supervised ({frac:.1%})"
        )
    print(f"Prepared SFT data in {out_dir}")


if __name__ == "__main__":
    main()
