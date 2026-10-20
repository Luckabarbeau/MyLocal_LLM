#!/usr/bin/env python3
"""0080BM: Shared tokenizer and thin orchestration of per-source packed shards.

The ONLY packing engine is pack_text_corpus in generate_packed_cosmopedia_shards.py.
Each input file is packed into its own shard subdirectory; data mixture weights
live solely in mixture_manifest.json, so they can change without retokenizing.

Usage (from repository root):
    python prepare_ensemble_mixture.py tokenizer --data-root ../pretraining_data \\
        --mix-config configs/pretraining_coding_mix.json \\
        --output-root ../pretraining_data/ensemble_32k_mixed \\
        --vocab-size 32768 --sample-documents 200000
    python prepare_ensemble_mixture.py shards --data-root ../pretraining_data \\
        --mix-config configs/pretraining_coding_mix.json \\
        --output-root ../pretraining_data/ensemble_32k_mixed

Full documents/EOS and document-hash train/validation split are preserved.
A bounded process pool encodes batches; the existing PackedShardWriter packs
contiguous uint16 streams separately for each raw input file. The trainer
reads all files recursively within each independently weighted source.
The mixture manifest is accepted by train_model.py --mixed-shard-root.
"""

from __future__ import annotations

import argparse
import os
import json
import math
from pathlib import Path

from generate_packed_cosmopedia_shards import (
    pack_text_corpus,
    file_sha256,
    is_validation_document,
    load_tokenizer,
    tokenizer_token_id,
)
from mini_llm.data.text_corpus import TextCorpusSource
from mini_llm.tokenizer.tokenizer import FastBPETokenizer


def sources_from_config(mix_config: Path, data_root: Path, weight_key: str,
                        include_zero: bool = False):
    settings = json.loads(mix_config.read_text(encoding="utf-8"))
    entries = settings.get("sources")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("mix configuration needs a non-empty sources dictionary")
    sources = []
    for name, entry in entries.items():
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid relative_path for {name!r}: {relative}")
        weight = float(entry[weight_key])
        if weight < 0:
            raise ValueError(f"negative {weight_key} for {name}")
        reader = TextCorpusSource(name, data_root / relative,
                                  text_column=entry.get("text_column"))
        sources.append((name, reader, weight))
    total = sum(weight for _, _, weight in sources)
    if total <= 0:
        raise ValueError(f"sum of {weight_key} must be positive")
    return [(name, reader, weight / total) for name, reader, weight in sources
            if include_zero or weight > 0]


def training_texts(sources, sample_documents, val_ratio, seed):
    """Bounded training-only sample, apportioned by tokenizer source weights."""
    targets = [round(sample_documents * weight) for _, _, weight in sources]
    targets[-1] += sample_documents - sum(targets)
    for (name, reader, _), count in zip(sources, targets):
        seen = 0
        for text in reader.iter_texts(seed=seed):
            if is_validation_document(text, val_ratio, seed):
                continue
            yield text
            seen += 1
            if seen >= count:
                break
        if seen < count:
            raise ValueError(f"{name}: only {seen} train documents available; "
                             f"requested {count}; lower --sample-documents")


def build_tokenizer(args):
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("use 0 < --val-ratio < 1 for held-out evaluation")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    target = output / "tokenizer.json"
    if target.exists():
        raise FileExistsError(f"tokenizer already exists, refusing overwrite: {target}")
    sources = sources_from_config(args.mix_config, args.data_root, "tokenizer_weight")
    if args.sample_documents < len(sources):
        raise ValueError("--sample-documents must allow at least one per source")
    # FastBPETokenizer uses the Rust/HF backend. Its ``threads`` attribute
    # alone does not configure Rayon; honor the requested CPU budget here.
    # For guaranteed early initialization, also set RAYON_NUM_THREADS in Bash.
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    tokenizer = FastBPETokenizer(vocab_size=args.vocab_size, threads=args.threads)
    print(f"Training {args.vocab_size:,}-token BPE on {args.sample_documents:,} "
          f"train-only documents from {len(sources)} corpora", flush=True)
    tokenizer.train(training_texts(sources, args.sample_documents, args.val_ratio, args.seed))
    if len(tokenizer) != args.vocab_size:
        raise RuntimeError(f"tokenizer trained {len(tokenizer)} tokens rather than "
                           f"requested {args.vocab_size}; increase corpus sample")
    tokenizer.save(str(target))
    metadata = dict(vocab_size=len(tokenizer), tokenizer_sha256=file_sha256(target),
                    sample_documents=args.sample_documents, val_ratio=args.val_ratio,
                    seed=args.seed, mix_config_sha256=file_sha256(args.mix_config),
                    source_weights={name: weight for name, _, weight in sources})
    (output / "tokenizer_training.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved tokenizer to {target}; SHA256 {metadata['tokenizer_sha256']}")


def mix_weights(mix_config: Path, weight_key: str):
    """Read sampling weights without requiring raw corpora to be mounted."""
    entries = json.loads(Path(mix_config).read_text(encoding="utf-8"))["sources"]
    weights = {name: float(entry[weight_key]) for name, entry in entries.items()}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("mixture weights must be nonnegative with a positive total")
    total = sum(weights.values())
    return {name: w / total for name, w in weights.items()}


def _write_manifest(path, manifest):
    """Publish only complete JSON; avoid truncated manifests on interruption."""
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending.replace(path)


def reweight_mixture(args):
    """Change sampling probabilities without re-reading or re-encoding text."""
    path = args.output_root / "mixture_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    weights = mix_weights(args.mix_config, "pretraining_weight")
    if set(manifest["sources"]) != set(weights):
        raise ValueError("source sets differ; reweight requires unchanged datasets")
    for name, source in manifest["sources"].items():
        source["weight"] = weights[name]
    _write_manifest(path, manifest)
    print("Updated mixture sampling weights; no shards were regenerated:", weights)


def pack_shards(args):
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("use 0 < --val-ratio < 1 for held-out evaluation")
    output = args.output_root
    tokenizer_file = output / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"train the shared tokenizer first: {tokenizer_file}")
    training_metadata = output / "tokenizer_training.json"
    if training_metadata.exists():
        metadata = json.loads(training_metadata.read_text())
        if (metadata["seed"] != args.seed or
                metadata["val_ratio"] != args.val_ratio):
            raise ValueError("split seed and validation ratio must match tokenizer "
                             "training; use a fresh output directory")
    tokenizer = load_tokenizer(str(tokenizer_file))
    vocab_size = len(tokenizer)
    if vocab_size <= 0 or vocab_size > 65_536:
        raise ValueError("tokenizer vocabulary must fit uint16")
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    all_sources = sources_from_config(args.mix_config, args.data_root,
                                      "pretraining_weight", include_zero=True)
    source_lookup = {name: (reader, weight) for name, reader, weight in all_sources}
    source_paths = {name: str(reader.path) for name, reader, _ in all_sources}
    requested = getattr(args, "source", None) or list(source_lookup)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate --source entries are not allowed")
    for name in requested:
        if name not in source_lookup:
            raise ValueError(f"unknown source {name!r}; expected {list(source_lookup)}")
    manifest_path = output / "mixture_manifest.json"
    sha = file_sha256(tokenizer_file)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete"):
            raise FileExistsError("mixture already complete; use 'reweight' to "
                                  "change only the mixture weights")
        if (manifest.get("tokenizer_sha256") != sha or
                manifest.get("preprocessing_seed") != args.seed or
                manifest.get("validation_ratio") != args.val_ratio or
                set(manifest.get("expected_sources", ())) != set(source_lookup) or
                manifest.get("source_paths") != source_paths):
            raise ValueError("partial mixture provenance differs; use a new root")
    else:
        manifest = dict(format_version="weighted_packed_mixture_v1",
                        complete=False, vocab_size=vocab_size,
                        tokenizer_sha256=sha, eos_token_id=eos_id,
                        token_dtype="uint16", preprocessing_seed=args.seed,
                        validation_ratio=args.val_ratio,
                        expected_sources=list(source_lookup),
                        source_paths=source_paths, sources={})
    # A partial dataset can resume by packing only remaining sources. Mixture
    # weights are metadata and may change without touching existing binaries.
    for done_name, done_entry in manifest["sources"].items():
        done_entry["weight"] = source_lookup[done_name][1]
    for name in requested:
        reader, weight = source_lookup[name]
        if name in manifest["sources"]:
            if getattr(args, "source", None):
                raise FileExistsError(f"{name} has already been packed")
            print(f"Skipping already completed source {name}", flush=True)
            continue
        dest = output / name
        print(f"Packing {name} independently: {reader.path} "
              f"with {getattr(args, 'workers', 1)} worker(s)", flush=True)
        result = pack_text_corpus(
            reader, output_dir=dest, tokenizer_path=tokenizer_file,
            seed=args.seed, val_ratio=args.val_ratio,
            shard_size_mb=args.shard_size_mb,
            batch_documents=args.batch_documents,
            workers=getattr(args, "workers", 1),
            pending_batches=getattr(args, "pending_batches", None),
            max_documents=getattr(args, "max_documents_per_source", None),
        )
        manifest["sources"][name] = dict(weight=weight,
                                         relative_shard_dir=name, **result)
        # Every generated source takes its sampling weight from the current
        # mixture; changes while packing must never leave mixed-era weights.
        weights = mix_weights(args.mix_config, "pretraining_weight")
        for source_name, entry in manifest["sources"].items():
            entry["weight"] = weights[source_name]
        missing = sorted(set(source_lookup) - set(manifest["sources"]))
        manifest["missing_sources"] = missing
        manifest["complete"] = not missing
        _write_manifest(manifest_path, manifest)
        print(f"  {name}: {result['train_tokens']:,} train tokens, "
              f"{result['val_tokens']:,} validation tokens", flush=True)
    if not manifest["complete"]:
        print("Partial mixture saved; remaining sources:",
              ", ".join(manifest["missing_sources"]))
    else:
        print(f"Complete mixture at {output}; per-file shards kept separate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for command in ("tokenizer", "shards", "reweight"):
        p = sub.add_parser(command)
        p.add_argument("--data-root", type=Path, default=Path("../pretraining_data"))
        p.add_argument("--mix-config", type=Path,
                       default=Path("configs/pretraining_coding_mix.json"))
        p.add_argument("--output-root", type=Path, required=True)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--val-ratio", type=float, default=0.01)
        if command == "tokenizer":
            p.add_argument("--vocab-size", type=int, default=32_768)
            p.add_argument("--sample-documents", type=int, default=200_000)
            p.add_argument("--threads", type=int, default=16,
                           help="Rayon threads; for tokenizer training set "
                                "RAYON_NUM_THREADS before Python starts")
        elif command == "shards":
            p.add_argument("--source", action="append", default=None,
                           help="only process this source (repeat for several); "
                                "each raw input file has its own shards")
            p.add_argument("--workers", type=int, default=8,
                           help="independent tokenizer processes")
            p.add_argument("--pending-batches", type=int, default=None,
                           help="max in-flight tokenization batches")
            p.add_argument("--batch-documents", type=int, default=256)
            p.add_argument("--shard-size-mb", type=float, default=256.0)
            p.add_argument("--max-documents-per-source", type=int, default=None,
                           help="Small-data smoke only; omit for production")
    args = parser.parse_args()
    if args.operation == "tokenizer":
        build_tokenizer(args)
    elif args.operation == "shards":
        pack_shards(args)
    else:
        reweight_mixture(args)


if __name__ == "__main__":
    main()
