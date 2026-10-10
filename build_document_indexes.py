#!/usr/bin/env python3
"""Prebuild EOS document-boundary sidecars for packed token shards.

0058B can build these indexes lazily during training, but prebuilding them keeps
first-epoch data loading free of one-time shard scans.  Existing .bin shards are
never modified or retokenized.
"""

import argparse
from pathlib import Path

from mini_llm.data.token_shards import (
    document_index_sidecar_path,
    load_or_build_packed_document_index,
    map_token_shard,
)
from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer


def load_tokenizer(path: Path):
    try:
        return FastBPETokenizer.load(str(path))
    except Exception:
        return SimpleBPETokenizer.load(str(path))


def eos_id(tokenizer) -> int:
    token = tokenizer.eos_token
    mapping = getattr(tokenizer, "token_to_id", None)
    if isinstance(mapping, dict):
        value = mapping.get(token)
    elif getattr(tokenizer, "_tokenizer", None) is not None:
        value = tokenizer._tokenizer.token_to_id(token)
    else:
        value = None
    if value is None:
        raise ValueError(f"tokenizer does not contain EOS token {token!r}")
    return int(value)


def main():
    parser = argparse.ArgumentParser(
        description="Build 0058B document indexes for existing packed shards"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mixed-shard-root", type=Path)
    group.add_argument("--shard-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    root = (args.mixed_shard_root or args.shard_dir).expanduser().resolve()
    tokenizer_path = (
        args.tokenizer.expanduser().resolve()
        if args.tokenizer is not None
        else root / "tokenizer.json"
    )
    tokenizer = load_tokenizer(tokenizer_path)
    token_eos = eos_id(tokenizer)

    patterns = ["**/train_shard_*.bin"]
    if not args.train_only:
        patterns.append("**/val_shard_*.bin")
    paths = sorted({p for pattern in patterns for p in root.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no packed token shards found below {root}")

    print(f"EOS token ID: {token_eos}")
    print(f"Packed shards: {len(paths)}")
    for i, path in enumerate(paths, 1):
        sidecar = document_index_sidecar_path(path)
        existed = sidecar.exists()
        data = map_token_shard(str(path))
        if data.ndim != 1:
            print(f"[{i}/{len(paths)}] skip legacy rectangular shard: {path}")
            continue
        index = load_or_build_packed_document_index(
            path, data, token_eos, write_sidecar=True
        )
        state = "loaded" if existed else "built"
        print(
            f"[{i}/{len(paths)}] {state}: {path.name} -> "
            f"{index.document_count:,} document fragments"
        )
        mmap_obj = getattr(data, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()


if __name__ == "__main__":
    main()
