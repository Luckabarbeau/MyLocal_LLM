#!/usr/bin/env python3
"""Recover a top-level mixed-shard manifest from shards already on disk.

This is intended for interrupted preprocessing runs where most corpora completed
normally but one source stopped before its per-source ``manifest.json`` could be
written (for example because the disk filled near the end).  Existing raw packed
``uint16`` shards are not modified.

Completed sources are validated against their normal per-source manifests.
Sources named with ``--partial-source`` are reconstructed from the shard files
that actually exist on disk.  The resulting ``mixture_manifest.json`` records
those sources explicitly in ``partial_sources`` so the reduced corpus remains
visible in the experiment provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from mini_llm.data.mixed_shards import normalize_source_weights


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_uint16_token_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"empty shard cannot be recovered safely: {path}")
        if size % 2 != 0:
            raise ValueError(
                f"shard size is not a multiple of uint16 (2 bytes): {path} "
                f"({size} bytes). Rename/remove the damaged tail shard and rerun."
            )
        total += size // 2
    return total


def scan_partial_source(source_dir: Path) -> dict:
    """Describe the usable raw packed shards currently present in ``source_dir``."""

    train_paths = tuple(sorted(source_dir.glob("train_shard_*.bin")))
    val_paths = tuple(sorted(source_dir.glob("val_shard_*.bin")))
    if not train_paths:
        raise ValueError(f"partial source has no training shards: {source_dir}")
    if not val_paths:
        raise ValueError(f"partial source has no validation shards: {source_dir}")

    return {
        "train_shards": len(train_paths),
        "val_shards": len(val_paths),
        "train_tokens": _raw_uint16_token_count(train_paths),
        "val_tokens": _raw_uint16_token_count(val_paths),
    }


def _completed_source_entry(
    *,
    name: str,
    source_dir: Path,
    tokenizer_hash: str,
) -> Tuple[dict, int | None]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"source {name!r} has no manifest.json; if it is intentionally partial, "
            f"pass --partial-source {name}"
        )

    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError(f"source {name!r} uses a different tokenizer")
    if source_manifest.get("token_dtype", "uint16") != "uint16":
        raise ValueError(f"source {name!r} is not a uint16 packed-token source")

    train_paths = tuple(sorted(source_dir.glob("train_shard_*.bin")))
    val_paths = tuple(sorted(source_dir.glob("val_shard_*.bin")))
    expected_train = int(source_manifest["shards"]["train"])
    expected_val = int(source_manifest["shards"]["val"])
    if len(train_paths) != expected_train or len(val_paths) != expected_val:
        raise ValueError(
            f"completed source {name!r} does not match its manifest: "
            f"train {len(train_paths)}/{expected_train}, "
            f"val {len(val_paths)}/{expected_val}"
        )

    entry = {
        "train_shards": expected_train,
        "val_shards": expected_val,
        "train_tokens": int(source_manifest["tokens"]["train"]),
        "val_tokens": int(source_manifest["tokens"]["val"]),
    }
    vocab_size = source_manifest.get("vocab_size")
    return entry, None if vocab_size is None else int(vocab_size)


def recover_mixture_manifest(
    *,
    shard_root: Path,
    mix_config_path: Path,
    partial_sources: Iterable[str],
    output_path: Path | None = None,
) -> dict:
    """Build a trainer-compatible root manifest from existing shard directories."""

    shard_root = Path(shard_root).expanduser().resolve()
    mix_config_path = Path(mix_config_path).expanduser().resolve()
    output_path = (
        shard_root / "mixture_manifest.json"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )

    tokenizer_path = shard_root / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    if not mix_config_path.exists():
        raise FileNotFoundError(f"mix config not found: {mix_config_path}")

    mix_config = json.loads(mix_config_path.read_text())
    configured = mix_config.get("sources")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("mix config contains no sources")

    partial_set = {str(name) for name in partial_sources}
    unknown_partial = partial_set - set(configured)
    if unknown_partial:
        raise ValueError(
            "--partial-source references unknown source(s): "
            + ", ".join(sorted(unknown_partial))
        )

    tokenizer_hash = file_sha256(tokenizer_path)
    configured_weights = {
        name: float(source_cfg["pretraining_weight"])
        for name, source_cfg in configured.items()
    }
    normalized_weights = normalize_source_weights(configured_weights)

    entries: Dict[str, dict] = {}
    vocab_sizes = set()
    for name in configured:
        source_dir = shard_root / name
        if not source_dir.exists():
            raise FileNotFoundError(f"configured source directory not found: {source_dir}")

        if name in partial_set:
            raw = scan_partial_source(source_dir)
            entry = {
                **raw,
                "recovered_partial": True,
            }
        else:
            raw, vocab_size = _completed_source_entry(
                name=name,
                source_dir=source_dir,
                tokenizer_hash=tokenizer_hash,
            )
            if vocab_size is not None:
                vocab_sizes.add(vocab_size)
            entry = raw

        entry.update(
            {
                "weight": normalized_weights[name],
                "relative_shard_dir": name,
            }
        )
        entries[name] = entry

    if len(vocab_sizes) > 1:
        raise ValueError(f"completed source manifests disagree on vocab size: {vocab_sizes}")
    if not vocab_sizes:
        raise ValueError(
            "could not infer tokenizer vocabulary size from any completed source manifest"
        )
    vocab_size = next(iter(vocab_sizes))

    manifest = {
        "format_version": "weighted_packed_mixture_v1",
        "mix_name": mix_config.get("name"),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": int(vocab_size),
        "token_dtype": "uint16",
        "context_independent": True,
        "configured_sources": list(configured),
        # All configured sources are represented by usable shards.  A recovered
        # source can be shorter than the original raw corpus, which is recorded
        # separately below rather than making the training contract unusable.
        "complete": True,
        "missing_sources": [],
        "recovered": bool(partial_set),
        "partial_sources": sorted(partial_set),
        "sources": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover mixture_manifest.json from existing packed shards after an "
            "interrupted mixed-corpus preprocessing run."
        )
    )
    parser.add_argument(
        "--shard-root",
        default="./token_shards_pretraining_65280",
        help="Root containing tokenizer.json and per-source shard directories",
    )
    parser.add_argument(
        "--mix-config",
        default=None,
        help=(
            "Mixture config to use for source weights. Defaults to "
            "<shard-root>/mix_config.json."
        ),
    )
    parser.add_argument(
        "--partial-source",
        action="append",
        default=[],
        help=(
            "Source whose preprocessing was interrupted but whose existing raw "
            "shards should be included. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path; defaults to <shard-root>/mixture_manifest.json",
    )
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    mix_config = (
        Path(args.mix_config)
        if args.mix_config is not None
        else shard_root / "mix_config.json"
    )
    manifest = recover_mixture_manifest(
        shard_root=shard_root,
        mix_config_path=mix_config,
        partial_sources=args.partial_source,
        output_path=None if args.output is None else Path(args.output),
    )

    print(f"Mixture manifest written under: {Path(args.output) if args.output else shard_root}")
    print("Sources:")
    for name, entry in manifest["sources"].items():
        marker = " [RECOVERED PARTIAL]" if entry.get("recovered_partial") else ""
        print(
            f"  {name:16s} {100.0 * entry['weight']:6.2f}%  "
            f"{entry['train_shards']:4d}+{entry['val_shards']:3d} shards  "
            f"{entry['train_tokens']:,} train tokens{marker}"
        )


if __name__ == "__main__":
    main()
