#!/usr/bin/env python3
"""Train the shared 65k BPE tokenizer from a weighted pretraining corpus mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from mini_llm.data.text_corpus import (
    CorpusSampleStats,
    CorpusSourceSpec,
    TextCorpusSource,
    normalized_weights,
    weighted_text_sample,
)
from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mix_config(path: Path, data_root: Path, weight_key: str):
    with path.open() as handle:
        config = json.load(handle)
    sources = []
    for name, source in config["sources"].items():
        if weight_key not in source:
            raise ValueError(f"source {name!r} has no {weight_key!r}")
        source_path = data_root / source["relative_path"]
        sources.append(
            CorpusSourceSpec(
                name=name,
                path=source_path,
                weight=float(source[weight_key]),
                text_column=source.get("text_column"),
            )
        )
    return config, sources


def parse_text_column_overrides(values: List[str]) -> Dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"invalid --text-column {value!r}; expected SOURCE=COLUMN"
            )
        source, column = value.split("=", 1)
        result[source.strip()] = column.strip()
    return result


def with_column_overrides(specs, overrides):
    unknown = set(overrides) - {spec.name for spec in specs}
    if unknown:
        raise ValueError(f"text-column overrides reference unknown sources: {unknown}")
    return [
        CorpusSourceSpec(
            name=spec.name,
            path=spec.path,
            weight=spec.weight,
            text_column=overrides.get(spec.name, spec.text_column),
        )
        for spec in specs
    ]


def print_plan(specs, total_bytes):
    weights = normalized_weights(specs)
    print("Tokenizer corpus plan")
    print("=" * 72)
    print(f"Total sample budget: {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB)")
    print()
    for spec in specs:
        budget = total_bytes * weights[spec.name]
        print(
            f"{spec.name:16s} {100.0 * weights[spec.name]:6.2f}%  "
            f"~{budget / 1e6:8.1f} MB  {spec.path}"
        )
    print()


def inspect_sources(specs, parquet_batch_size):
    print("Corpus inspection")
    print("=" * 72)
    for spec in specs:
        source = TextCorpusSource(
            spec.name,
            spec.path,
            text_column=spec.text_column,
            parquet_batch_size=parquet_batch_size,
        )
        print(json.dumps(source.inspect(), indent=2))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from a weighted multi-corpus sample"
    )
    parser.add_argument("--data-root", default="../pretraining_data")
    parser.add_argument(
        "--mix-config",
        default="configs/pretraining_coding_mix.json",
    )
    parser.add_argument(
        "--weight-key",
        default="tokenizer_weight",
        help="Weight field from the mix config (default: tokenizer_weight)",
    )
    parser.add_argument("--vocab-size", type=int, default=65_280)
    parser.add_argument(
        "--total-bytes",
        type=int,
        default=2_000_000_000,
        help="Total UTF-8 byte budget sampled across all corpora",
    )
    parser.add_argument("--backend", choices=["fast", "simple"], default="fast")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parquet-batch-size", type=int, default=1024)
    parser.add_argument(
        "--text-column",
        action="append",
        default=[],
        metavar="SOURCE=COLUMN",
        help="Override automatic text/code-column detection for one source",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect source files/columns and exit before tokenizer training",
    )
    args = parser.parse_args()

    if not 3 <= args.vocab_size <= 65_536:
        raise ValueError("vocab size must fit the uint16 packed-token format")
    data_root = Path(args.data_root).expanduser().resolve()
    mix_path = Path(args.mix_config).expanduser().resolve()
    mix_config, specs = load_mix_config(mix_path, data_root, args.weight_key)
    specs = with_column_overrides(
        specs, parse_text_column_overrides(args.text_column)
    )

    print_plan(specs, args.total_bytes)
    inspect_sources(specs, args.parquet_batch_size)
    if args.inspect_only:
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force_retrain:
        raise FileExistsError(
            f"tokenizer already exists: {output}; use --force-retrain to replace it"
        )

    sample_stats = {spec.name: CorpusSampleStats() for spec in specs}
    texts = weighted_text_sample(
        specs,
        total_bytes=args.total_bytes,
        seed=args.seed,
        parquet_batch_size=args.parquet_batch_size,
        stats=sample_stats,
    )

    start = time.time()
    if args.backend == "fast":
        tokenizer = FastBPETokenizer(
            vocab_size=args.vocab_size, threads=args.threads
        )
        tokenizer.train(texts)
    else:
        # The reference Python tokenizer intentionally remains a small-data path.
        tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(list(texts))
    elapsed = time.time() - start

    tokenizer.save(str(output))
    actual_total = sum(item.bytes for item in sample_stats.values())
    actual_docs = sum(item.documents for item in sample_stats.values())
    normalized = normalized_weights(specs)

    metadata = {
        "file_hash": file_sha256(output),
        "vocab_size": len(tokenizer),
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
        },
        "training": {
            "backend": args.backend,
            "threads": args.threads,
            "seed": args.seed,
            "vocab_target": args.vocab_size,
            "sample_budget_bytes": args.total_bytes,
            "actual_bytes": actual_total,
            "documents_used": actual_docs,
            "training_time_seconds": elapsed,
            "mix_name": mix_config.get("name"),
            "mix_config": str(mix_path),
            "weight_key": args.weight_key,
            "sources": {
                spec.name: {
                    "path": str(spec.path),
                    "requested_weight": normalized[spec.name],
                    "documents": sample_stats[spec.name].documents,
                    "bytes": sample_stats[spec.name].bytes,
                    "actual_fraction": (
                        sample_stats[spec.name].bytes / actual_total
                        if actual_total
                        else 0.0
                    ),
                    "text_column": spec.text_column,
                }
                for spec in specs
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    metadata_path = output.with_suffix(".meta.json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nTokenizer training complete")
    print(f"  vocabulary: {len(tokenizer):,}")
    print(f"  documents:  {actual_docs:,}")
    print(f"  bytes:      {actual_total:,} ({actual_total / 1e9:.2f} GB)")
    print(f"  time:       {elapsed:.1f} s")
    print(f"  tokenizer:  {output}")
    print(f"  metadata:   {metadata_path}")
    print("\nActual source mixture")
    for spec in specs:
        item = sample_stats[spec.name]
        fraction = item.bytes / actual_total if actual_total else 0.0
        print(
            f"  {spec.name:16s} {100.0 * fraction:6.2f}%  "
            f"{item.bytes / 1e6:8.1f} MB  {item.documents:,} docs"
        )


if __name__ == "__main__":
    main()
