#!/usr/bin/env python3
"""Tokenize all configured pretraining corpora into independent packed shards.

Each source is stored separately.  The trainer later samples source directories
according to ``pretraining_weight`` from the mix config, so changing a mixture
does not require copying or regenerating token data.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from generate_packed_cosmopedia_shards import (
    PackedShardWriter,
    encode_and_write_batch,
    file_sha256,
    load_tokenizer,
    tokenizer_token_id,
)
from mini_llm.data.mixed_shards import normalize_source_weights
from mini_llm.data.text_corpus import TextCorpusSource


def parse_text_column_overrides(values: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"invalid --text-column {value!r}; expected SOURCE=COLUMN"
            )
        source, column = value.split("=", 1)
        result[source.strip()] = column.strip()
    return result


def validation_flag(text: str, source_name: str, val_ratio: float, seed: int) -> bool:
    """Stable source-aware train/validation assignment."""

    import hashlib

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if val_ratio == 0.0:
        return False
    digest = hashlib.sha256()
    digest.update(int(seed).to_bytes(8, byteorder="little", signed=True))
    digest.update(source_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    value = int.from_bytes(digest.digest()[:8], byteorder="little")
    return value / float(1 << 64) < val_ratio


def _clear_source_outputs(source_dir: Path) -> None:
    for pattern in ("train_shard_*.bin", "val_shard_*.bin"):
        for path in source_dir.glob(pattern):
            path.unlink()
    manifest = source_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def generate_source(
    *,
    source_name: str,
    source_path: Path,
    text_column: str | None,
    tokenizer,
    eos_id: int,
    tokenizer_hash: str,
    vocab_size: int,
    source_dir: Path,
    shard_size_mb: float,
    batch_documents: int,
    parquet_batch_size: int,
    val_ratio: float,
    seed: int,
    weight: float,
    max_documents: int | None,
) -> dict:
    source_dir.mkdir(parents=True, exist_ok=True)
    train_writer = PackedShardWriter(
        source_dir, "train", shard_size_mb, dtype=np.uint16
    )
    val_writer = PackedShardWriter(
        source_dir, "val", shard_size_mb, dtype=np.uint16
    )
    corpus = TextCorpusSource(
        source_name,
        source_path,
        text_column=text_column,
        parquet_batch_size=parquet_batch_size,
    )

    texts: List[str] = []
    split_flags: List[bool] = []
    documents = 0
    train_docs = 0
    val_docs = 0
    max_token_id = eos_id

    def flush_batch() -> None:
        nonlocal train_docs, val_docs, max_token_id
        if not texts:
            return
        n_train, n_val, batch_max = encode_and_write_batch(
            tokenizer,
            texts,
            split_flags,
            eos_id,
            train_writer,
            val_writer,
        )
        train_docs += n_train
        val_docs += n_val
        max_token_id = max(max_token_id, batch_max)
        texts.clear()
        split_flags.clear()

    try:
        for text in corpus.iter_texts(seed=seed, shuffle_files=True):
            if not text:
                continue
            texts.append(text)
            split_flags.append(
                validation_flag(text, source_name, val_ratio, seed)
            )
            documents += 1
            if len(texts) >= batch_documents:
                flush_batch()
                if documents % (batch_documents * 20) == 0:
                    print(
                        f"[{source_name}] {documents:,} docs | "
                        f"train {train_writer.total_tokens:,} tok | "
                        f"val {val_writer.total_tokens:,} tok"
                    )
            if max_documents is not None and documents >= max_documents:
                break
        flush_batch()
    finally:
        train_writer.close()
        val_writer.close()

    if not train_writer.paths:
        raise RuntimeError(f"source {source_name!r} produced no training shards")
    if val_ratio > 0.0 and not val_writer.paths:
        raise RuntimeError(f"source {source_name!r} produced no validation shards")
    if max_token_id >= vocab_size:
        raise RuntimeError(
            f"source {source_name!r} encoded token {max_token_id} outside "
            f"vocabulary size {vocab_size}"
        )

    manifest = {
        "format_version": "packed_token_stream_v1",
        "source_name": source_name,
        "source_path": str(source_path),
        "pretraining_weight": float(weight),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": int(vocab_size),
        "token_dtype": "uint16",
        "eos_token_id": int(eos_id),
        "preprocessing_seed": int(seed),
        "val_ratio": float(val_ratio),
        "context_independent": True,
        "text_column": text_column,
        "documents": {
            "total": int(documents),
            "train": int(train_docs),
            "val": int(val_docs),
        },
        "tokens": {
            "train": int(train_writer.total_tokens),
            "val": int(val_writer.total_tokens),
            "total": int(train_writer.total_tokens + val_writer.total_tokens),
        },
        "shards": {
            "train": len(train_writer.paths),
            "val": len(val_writer.paths),
            "target_size_mb": float(shard_size_mb),
        },
    }
    (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_root_manifest(
    output_root: Path,
    mix_config: dict,
    tokenizer_path: Path,
    vocab_size: int,
) -> dict:
    tokenizer_hash = file_sha256(tokenizer_path)
    generated = {}
    configured_weights = {
        name: float(entry["pretraining_weight"])
        for name, entry in mix_config["sources"].items()
    }
    normalized = normalize_source_weights(configured_weights)

    for name, entry in mix_config["sources"].items():
        source_manifest_path = output_root / name / "manifest.json"
        if not source_manifest_path.exists():
            continue
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError(
                f"source {name!r} was tokenized with a different tokenizer"
            )
        generated[name] = {
            "weight": normalized[name],
            "relative_shard_dir": name,
            "train_shards": int(source_manifest["shards"]["train"]),
            "val_shards": int(source_manifest["shards"]["val"]),
            "train_tokens": int(source_manifest["tokens"]["train"]),
            "val_tokens": int(source_manifest["tokens"]["val"]),
        }

    # Renormalize if the user intentionally generated only a subset of sources.
    # A normal full run includes every configured source and therefore preserves
    # exactly the config weights.
    if generated:
        generated_weights = normalize_source_weights(
            {name: entry["weight"] for name, entry in generated.items()}
        )
        for name, weight in generated_weights.items():
            generated[name]["weight"] = weight

    configured_names = list(mix_config["sources"])
    missing = [name for name in configured_names if name not in generated]
    return {
        "format_version": "weighted_packed_mixture_v1",
        "mix_name": mix_config.get("name"),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": int(vocab_size),
        "token_dtype": "uint16",
        "context_independent": True,
        "configured_sources": configured_names,
        "complete": not missing,
        "missing_sources": missing,
        "sources": generated,
    }



def parallelism_plan(
    requested_workers: int,
    total_threads: int,
    pending_sources: int,
) -> tuple[int, int]:
    """Return ``(source_workers, tokenizer_threads_per_worker)``.

    Source generation is independent, so separate sources can safely run in
    different processes.  Hugging Face ``tokenizers`` uses Rayon internally;
    its pool size is bounded per process so the aggregate CPU usage stays near
    the user-requested budget rather than multiplying ``workers * cpu_count``.
    """

    if requested_workers <= 0:
        raise ValueError("workers must be positive")
    if total_threads <= 0:
        raise ValueError("threads must be positive")
    if pending_sources <= 0:
        return 0, 0

    workers = min(int(requested_workers), int(pending_sources), int(total_threads))
    tokenizer_threads = max(1, int(total_threads) // workers)
    return workers, tokenizer_threads


def _generate_source_worker(job: dict) -> tuple[str, dict]:
    """Generate one source in a spawned process.

    ``RAYON_NUM_THREADS`` must be configured before the first tokenizer encode
    in this process.  Each worker loads its own tokenizer instance, avoiding
    cross-process tokenizer state and fork-after-Rayon hazards.
    """

    tokenizer_threads = int(job["tokenizer_threads"])
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer_path = Path(job["tokenizer_path"])
    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer)
    if vocab_size != int(job["vocab_size"]):
        raise RuntimeError(
            f"worker tokenizer vocabulary changed: {vocab_size} != "
            f"{job['vocab_size']}"
        )
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    if eos_id != int(job["eos_id"]):
        raise RuntimeError(
            f"worker tokenizer EOS changed: {eos_id} != {job['eos_id']}"
        )

    manifest = generate_source(
        source_name=job["source_name"],
        source_path=Path(job["source_path"]),
        text_column=job["text_column"],
        tokenizer=tokenizer,
        eos_id=eos_id,
        tokenizer_hash=job["tokenizer_hash"],
        vocab_size=vocab_size,
        source_dir=Path(job["source_dir"]),
        shard_size_mb=float(job["shard_size_mb"]),
        batch_documents=int(job["batch_documents"]),
        parquet_batch_size=int(job["parquet_batch_size"]),
        val_ratio=float(job["val_ratio"]),
        seed=int(job["seed"]),
        weight=float(job["weight"]),
        max_documents=job["max_documents"],
    )
    return job["source_name"], manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-source packed shards for weighted pretraining"
    )
    parser.add_argument("--data-root", default="../pretraining_data")
    parser.add_argument(
        "--mix-config", default="configs/pretraining_coding_mix.json"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--output-root", default="./token_shards_pretraining_65280"
    )
    parser.add_argument("--shard-size-mb", type=float, default=256.0)
    parser.add_argument("--batch-documents", type=int, default=1024)
    parser.add_argument("--parquet-batch-size", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of corpus sources to generate concurrently in separate "
            "processes. Each source writes only to its own directory."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget for shard generation. The budget is "
            "divided across active source workers and used to bound the "
            "Hugging Face tokenizer Rayon pool. Default: os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Generate only this source (repeatable). Default: all sources.",
    )
    parser.add_argument(
        "--text-column",
        action="append",
        default=[],
        metavar="SOURCE=COLUMN",
    )
    parser.add_argument(
        "--max-documents-per-source",
        type=int,
        default=None,
        help="Debug/smoke limit applied independently to every selected source",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_documents <= 0:
        raise ValueError("batch-documents must be positive")
    if args.parquet_batch_size <= 0:
        raise ValueError("parquet-batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    total_threads = int(args.threads or (os.cpu_count() or 1))
    if total_threads <= 0:
        raise ValueError("threads must be positive")

    # Bound the tokenizer pool even in the sequential path.  The final value
    # for parallel workers is refined below once the number of pending sources
    # is known.
    os.environ["RAYON_NUM_THREADS"] = str(total_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    data_root = Path(args.data_root).expanduser().resolve()
    mix_path = Path(args.mix_config).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mix_config = json.loads(mix_path.read_text())
    overrides = parse_text_column_overrides(args.text_column)
    configured = mix_config.get("sources", {})
    if not configured:
        raise ValueError("mix config contains no sources")
    unknown_overrides = set(overrides) - set(configured)
    if unknown_overrides:
        raise ValueError(
            f"text-column overrides reference unknown sources: {unknown_overrides}"
        )

    selected = args.source or list(configured)
    unknown = set(selected) - set(configured)
    if unknown:
        raise ValueError(f"requested unknown sources: {unknown}")

    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer)
    if not 0 < vocab_size <= 65_536:
        raise ValueError("tokenizer vocabulary must fit uint16")
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    tokenizer_hash = file_sha256(tokenizer_path)

    root_tokenizer = output_root / "tokenizer.json"
    if root_tokenizer.exists() and file_sha256(root_tokenizer) != tokenizer_hash:
        if not args.overwrite:
            raise ValueError(
                "output root already contains a different tokenizer; use a new "
                "output directory or --overwrite"
            )
    shutil.copy2(tokenizer_path, root_tokenizer)
    tokenizer_meta = tokenizer_path.with_suffix(".meta.json")
    if tokenizer_meta.exists():
        shutil.copy2(tokenizer_meta, output_root / "tokenizer.meta.json")
    shutil.copy2(mix_path, output_root / "mix_config.json")

    print("Weighted pretraining shard generation")
    print("=" * 72)
    print(f"Tokenizer vocabulary: {vocab_size:,}")
    print(f"Output root:          {output_root}")
    print(f"Sources:              {', '.join(selected)}")

    jobs = []
    for name in selected:
        source_cfg = configured[name]
        source_dir = output_root / name
        completed_manifest = source_dir / "manifest.json"
        existing_bins = (
            list(source_dir.glob("*_shard_*.bin")) if source_dir.exists() else []
        )
        if completed_manifest.exists() and not args.overwrite:
            existing = json.loads(completed_manifest.read_text())
            if existing.get("tokenizer_sha256") != tokenizer_hash:
                raise ValueError(
                    f"completed source {name!r} uses a different tokenizer"
                )
            print(f"[{name}] already complete; skipping")
            continue
        if existing_bins and not args.overwrite:
            raise RuntimeError(
                f"source {name!r} contains partial shard files but no completed "
                "manifest; rerun with --overwrite to regenerate it"
            )
        if args.overwrite:
            source_dir.mkdir(parents=True, exist_ok=True)
            _clear_source_outputs(source_dir)

        source_path = data_root / source_cfg["relative_path"]
        text_column = overrides.get(name, source_cfg.get("text_column"))
        jobs.append(
            {
                "source_name": name,
                "source_path": str(source_path),
                "text_column": text_column,
                "tokenizer_path": str(tokenizer_path),
                "eos_id": eos_id,
                "tokenizer_hash": tokenizer_hash,
                "vocab_size": vocab_size,
                "source_dir": str(source_dir),
                "shard_size_mb": args.shard_size_mb,
                "batch_documents": args.batch_documents,
                "parquet_batch_size": args.parquet_batch_size,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "weight": float(source_cfg["pretraining_weight"]),
                "max_documents": args.max_documents_per_source,
            }
        )

    source_workers, tokenizer_threads = parallelism_plan(
        args.workers, total_threads, len(jobs)
    )
    if jobs:
        print(f"CPU thread budget:    {total_threads}")
        print(f"Source workers:       {source_workers}")
        print(f"Tokenizer threads:    {tokenizer_threads} per worker")
        print(
            f"Approx. active pool:  {source_workers * tokenizer_threads} "
            "tokenizer threads"
        )
    print()

    for job in jobs:
        job["tokenizer_threads"] = tokenizer_threads
        print(f"[{job['source_name']}] source: {job['source_path']}")

    if source_workers == 1:
        # Avoid process startup overhead for the common sequential case while
        # still respecting the explicit tokenizer thread budget.
        if jobs:
            os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
        for job in jobs:
            name, manifest = _generate_source_worker(job)
            print(
                f"[{name}] complete: {manifest['tokens']['train']:,} train tokens, "
                f"{manifest['tokens']['val']:,} val tokens, "
                f"{manifest['shards']['train']}+{manifest['shards']['val']} shards"
            )
    elif jobs:
        # ``spawn`` is intentional: forking a process after a tokenizer/Rayon
        # pool has been initialized can disable tokenizers parallelism or lead
        # to unsafe inherited thread state.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=source_workers,
            mp_context=context,
        ) as executor:
            future_to_name = {
                executor.submit(_generate_source_worker, job): job["source_name"]
                for job in jobs
            }
            try:
                for future in as_completed(future_to_name):
                    name, manifest = future.result()
                    print(
                        f"[{name}] complete: "
                        f"{manifest['tokens']['train']:,} train tokens, "
                        f"{manifest['tokens']['val']:,} val tokens, "
                        f"{manifest['shards']['train']}+"
                        f"{manifest['shards']['val']} shards"
                    )
            except BaseException:
                for future in future_to_name:
                    future.cancel()
                raise

    root_manifest = build_root_manifest(
        output_root, mix_config, root_tokenizer, vocab_size
    )
    (output_root / "mixture_manifest.json").write_text(
        json.dumps(root_manifest, indent=2)
    )

    missing = set(configured) - set(root_manifest["sources"])
    print("\nMixture manifest written")
    for name, entry in root_manifest["sources"].items():
        print(
            f"  {name:16s} {100.0 * entry['weight']:6.2f}%  "
            f"{entry['train_tokens']:,} train tokens"
        )
    if missing:
        print(f"  NOTE: not yet generated: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
