#!/usr/bin/env python3
"""Prepare and atomically register Common Pile v0.1 filtered long-form data.

The existing ensemble tokenizer is mandatory. Documents are reconstructed where
source metadata supplies a stable parent and order, exact-deduplicated, assigned
wholly to train/validation, and written as packed uint16 streams with one EOS.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import time
import unicodedata
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from generate_packed_cosmopedia_shards import (
    PackedShardWriter,
    file_sha256,
    is_validation_document,
    load_tokenizer,
    tokenizer_token_id,
)
from mini_llm.data.mixed_shards import load_weighted_shard_sources

DEFAULT_CONFIG = Path("configs/common_pile_longform_v01.json")
PART_RE = re.compile(r"^(?P<parent>.+)/[^/]+-pt(?P<part>\d+)$")
_WORKER_TOKENIZER = None


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict
    source: str = ""


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("version") != 1 or not config.get("sources"):
        raise ValueError("unsupported or empty Common Pile long-form config")
    return config


def parse_metadata(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        return parsed if isinstance(parsed, dict) else {"raw_metadata": parsed}
    return {}


def iter_records(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
                if isinstance(value, dict):
                    value["metadata"] = parse_metadata(value.get("metadata"))
                    value["_input_file"] = path.name
                    value["_input_line"] = line_number
                    yield value


def _group_identity(record: dict, reconstruction: str, ordinal: int):
    metadata = record["metadata"]
    doc_id = str(record.get("id") or f"record-{ordinal}")
    if reconstruction == "doab_book_sections":
        key = str(metadata.get("book_id") or doc_id.rsplit(".", 1)[0])
        order = int(metadata.get("section_idx") or 0)
    elif reconstruction == "pressbooks_chapters":
        key = str(metadata.get("book_url") or metadata.get("title") or doc_id)
        order = ordinal
    elif reconstruction == "usgpo_parts":
        match = PART_RE.match(doc_id)
        key = match.group("parent") if match else doc_id
        order = int(match.group("part")) if match else 0
    else:
        key, order = doc_id, ordinal
    return key, order


def grouped_documents(records: Iterable[dict], reconstruction: str) -> Iterator[Document]:
    """Reconstruct only records sharing an explicit stable parent identifier."""
    if reconstruction == "row":
        for ordinal, record in enumerate(records):
            text = record.get("text")
            if isinstance(text, str) and text:
                metadata = dict(record["metadata"])
                metadata.update(input_file=record.get("_input_file"),
                                input_line=record.get("_input_line"),
                                component_count=1)
                yield Document(str(record.get("id") or f"record-{ordinal}"), text,
                               metadata, str(record.get("source") or ""))
        return

    current_key = None
    components = []
    seen_keys = set()
    previous_order = -1

    def emit(key, values):
        first = values[0][1]
        metadata = dict(first["metadata"])
        metadata.update(
            component_count=len(values),
            first_component_id=str(values[0][1].get("id") or ""),
            last_component_id=str(values[-1][1].get("id") or ""),
            input_file_first=values[0][1].get("_input_file"),
            input_line_first=values[0][1].get("_input_line"),
            input_file_last=values[-1][1].get("_input_file"),
            input_line_last=values[-1][1].get("_input_line"),
            reconstruction=reconstruction,
        )
        text = "\n\n".join(str(item[1].get("text") or "") for item in values).strip()
        if text:
            return Document(str(key), text, metadata, str(first.get("source") or ""))
        return None

    for ordinal, record in enumerate(records):
        if not isinstance(record.get("text"), str) or not record["text"]:
            continue
        key, order = _group_identity(record, reconstruction, ordinal)
        if current_key is None:
            current_key, previous_order = key, order
        elif key != current_key:
            document = emit(current_key, components)
            if document is not None:
                yield document
            seen_keys.add(current_key)
            if key in seen_keys:
                raise ValueError(f"non-contiguous reconstructed parent {key!r}")
            current_key, components, previous_order = key, [], order
        elif order < previous_order:
            raise ValueError(f"out-of-order components for reconstructed parent {key!r}")
        previous_order = order
        components.append((order, record))
    if components:
        document = emit(current_key, components)
        if document is not None:
            yield document


def normalized_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def length_distribution(lengths: list[int]) -> dict:
    if not lengths:
        return {"documents": 0, "total_tokens_excluding_eos": 0,
                "proportions_exceeding": {str(x): 0.0 for x in (8192, 16384, 32768, 65536)}}
    ordered = sorted(lengths)

    def percentile(q):
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    thresholds = (8192, 16384, 32768, 65536)
    return {
        "documents": len(ordered),
        "total_tokens_excluding_eos": sum(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50), "p75": percentile(0.75),
        "p90": percentile(0.90), "p95": percentile(0.95),
        "p99": percentile(0.99), "max": ordered[-1],
        "counts_exceeding": {str(x): sum(value > x for value in ordered) for x in thresholds},
        "proportions_exceeding": {
            str(x): sum(value > x for value in ordered) / len(ordered) for x in thresholds
        },
    }


def _init_worker(tokenizer_path: str):
    global _WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_path)


def _encode_worker(texts: list[str]):
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.encode_batch(texts)


def _atomic_json(path: Path, value: dict):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def _validate_raw_files(data_root: Path, source: dict) -> list[Path]:
    raw_dir = data_root / source["raw_subdir"]
    paths = [raw_dir / name for name in source["files"]]
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    partials = list(raw_dir.glob("*.part")) if raw_dir.exists() else []
    if missing or partials:
        raise FileNotFoundError(f"incomplete raw source {raw_dir}; missing={missing}, partials={partials}")
    return paths


def prepare_source(args, config: dict, name: str):
    source = config["sources"][name]
    output_root = args.output_root.resolve()
    final_dir = output_root / name
    partial_dir = output_root / f"{name}.partial"
    if (final_dir / "source_manifest.json").is_file():
        print(f"Skipping completed source {name}", flush=True)
        return
    if final_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir} or {partial_dir}; remove only the partial directory after inspection")

    tokenizer_path = output_root / "tokenizer.json"
    if file_sha256(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("existing ensemble tokenizer SHA-256 differs from long-form config")
    tokenizer = load_tokenizer(str(tokenizer_path))
    if len(tokenizer) != config["vocab_size"]:
        raise ValueError("existing tokenizer vocabulary size differs from config")
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    if eos_id != config["eos_token_id"]:
        raise ValueError("existing tokenizer EOS differs from config")
    raw_paths = _validate_raw_files(args.data_root.resolve(), source)

    partial_dir.mkdir(parents=False)
    train_writer = PackedShardWriter(partial_dir, "train", args.shard_size_mb, "uint16")
    val_writer = PackedShardWriter(partial_dir, "val", args.shard_size_mb, "uint16")
    database = sqlite3.connect(output_root / "common_pile_longform_dedup.sqlite3")
    database.execute("CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, source TEXT NOT NULL, doc_id TEXT NOT NULL)")
    database.execute("BEGIN")
    provenance = gzip.open(partial_dir / "provenance.jsonl.gz", "wt", encoding="utf-8")
    duplicates = gzip.open(partial_dir / "duplicates.jsonl.gz", "wt", encoding="utf-8")
    lengths, licenses = [], Counter()
    train_docs = val_docs = duplicate_docs = input_documents = 0
    started = time.monotonic()
    pool = None
    queue = deque()
    batch = []

    def commit(documents, encoded):
        nonlocal train_docs, val_docs
        if len(documents) != len(encoded):
            raise RuntimeError("tokenizer returned an incorrect document count")
        for document, ids in zip(documents, encoded):
            if ids and (min(ids) < 0 or max(ids) >= config["vocab_size"]):
                raise ValueError("tokenizer emitted an ID outside the configured vocabulary")
            is_val = is_validation_document(document.text, config["validation_ratio"], config["split_seed"])
            values = np.empty(len(ids) + 1, dtype=np.uint16)
            if ids:
                values[:-1] = ids
            values[-1] = eos_id
            (val_writer if is_val else train_writer).write(values)
            if is_val:
                val_docs += 1
            else:
                train_docs += 1
            lengths.append(len(ids))
            license_name = str(document.metadata.get("license") or "unspecified")
            licenses[license_name] += 1
            provenance.write(json.dumps({
                "id": document.doc_id, "source": document.source,
                "split": "val" if is_val else "train", "tokens_excluding_eos": len(ids),
                "text_sha256": normalized_text_hash(document.text), "metadata": document.metadata,
            }, ensure_ascii=False) + "\n")

    def drain_one():
        documents, future = queue.popleft()
        commit(documents, future.result())

    def submit():
        nonlocal batch
        if not batch:
            return
        documents, batch = batch, []
        texts = [document.text for document in documents]
        if pool is None:
            commit(documents, tokenizer.encode_batch(texts))
        else:
            queue.append((documents, pool.submit(_encode_worker, texts)))
            if len(queue) >= args.pending_batches:
                drain_one()

    try:
        if args.workers > 1:
            pool = ProcessPoolExecutor(max_workers=args.workers,
                                       mp_context=mp.get_context("spawn"),
                                       initializer=_init_worker,
                                       initargs=(str(tokenizer_path),))
        records = iter_records(raw_paths)
        for document in grouped_documents(records, source["reconstruction"]):
            input_documents += 1
            digest = normalized_text_hash(document.text)
            try:
                database.execute("INSERT INTO documents(hash, source, doc_id) VALUES (?, ?, ?)",
                                 (digest, name, document.doc_id))
            except sqlite3.IntegrityError:
                duplicate_docs += 1
                owner = database.execute("SELECT source, doc_id FROM documents WHERE hash=?", (digest,)).fetchone()
                duplicates.write(json.dumps({"id": document.doc_id, "text_sha256": digest,
                                             "duplicate_of_source": owner[0],
                                             "duplicate_of_id": owner[1]}, ensure_ascii=False) + "\n")
                continue
            batch.append(document)
            if len(batch) >= args.batch_documents:
                submit()
            if args.max_documents and input_documents >= args.max_documents:
                break
            if input_documents % 10000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(f"  {name}: {input_documents:,} reconstructed, {len(lengths):,} tokenized, "
                      f"{sum(lengths):,} tokens ({sum(lengths)/elapsed:,.0f} tok/s)", flush=True)
        submit()
        while queue:
            drain_one()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            pool = None
        train_writer.close(); val_writer.close(); provenance.close(); duplicates.close()
        if not train_writer.paths or not val_writer.paths:
            raise ValueError(f"{name} must produce both train and validation shards")
        stats = length_distribution(lengths)
        stats.update(input_reconstructed_documents=input_documents,
                     unique_documents=len(lengths), duplicate_documents=duplicate_docs,
                     licenses=dict(licenses))
        _atomic_json(partial_dir / "length_stats.json", stats)
        manifest = {
            "format_version": "common_pile_longform_source_v1",
            "name": name, "dataset": source["dataset"], "revision": source["revision"],
            "raw_subdir": source["raw_subdir"], "raw_files": source["files"],
            "license": source["license"], "reconstruction": source["reconstruction"],
            "tokenizer_sha256": config["tokenizer_sha256"], "vocab_size": len(tokenizer),
            "eos_token_id": eos_id, "token_dtype": "uint16",
            "validation_ratio": config["validation_ratio"], "preprocessing_seed": config["split_seed"],
            "deduplication": "SHA-256 of NFKC/newline-normalized complete text across configured long-form sources",
            "train_shards": len(train_writer.paths), "val_shards": len(val_writer.paths),
            "train_tokens": train_writer.total_tokens, "val_tokens": val_writer.total_tokens,
            "train_documents": train_docs, "val_documents": val_docs,
            "unique_documents": len(lengths), "duplicate_documents": duplicate_docs,
            "length_stats_file": "length_stats.json", "provenance_file": "provenance.jsonl.gz",
            "duplicates_file": "duplicates.jsonl.gz", "workers": args.workers,
            "production_complete": args.max_documents is None,
        }
        _atomic_json(partial_dir / "source_manifest.json", manifest)
        database.commit()
        database.close()
        partial_dir.replace(final_dir)
        print(f"Completed {name}: {len(lengths):,} unique documents, "
              f"{train_writer.total_tokens + val_writer.total_tokens:,} tokens", flush=True)
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        train_writer.close(); val_writer.close(); provenance.close(); duplicates.close()
        database.rollback(); database.close()
        raise


def stage_sources(config: dict, stage: str) -> list[str]:
    if stage == "priority1":
        return [name for name, source in config["sources"].items() if source["priority"] == 1]
    if stage == "priority2":
        return [name for name, source in config["sources"].items() if source["priority"] == 2]
    if stage == "full":
        return list(config["stages"]["full"]["weights"])
    raise ValueError(f"unknown stage {stage}")


def integrate(args, config: dict):
    stage = config["stages"][args.stage]
    weights = {name: float(weight) for name, weight in stage["weights"].items()}
    if not math.isclose(sum(weights.values()), stage["long_form_fraction"], abs_tol=1e-12):
        raise ValueError("configured long-form weights do not sum to long_form_fraction")
    root = args.output_root.resolve()
    path = root / "mixture_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("tokenizer_sha256") != config["tokenizer_sha256"]:
        raise ValueError("mixture tokenizer differs from Common Pile config")
    all_long_names = set(config["sources"])
    legacy = {name: entry for name, entry in manifest["sources"].items() if name not in all_long_names}
    legacy_total = sum(float(entry["weight"]) for entry in legacy.values())
    target_legacy = 1.0 - stage["long_form_fraction"]
    for entry in legacy.values():
        entry["weight"] = float(entry["weight"]) / legacy_total * target_legacy

    new_entries = {}
    for name, weight in weights.items():
        source_manifest_path = root / name / "source_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not source_manifest.get("production_complete"):
            raise ValueError(f"refusing to integrate capped/non-production source {name}")
        if source_manifest.get("tokenizer_sha256") != config["tokenizer_sha256"]:
            raise ValueError(f"tokenizer mismatch for {name}")
        new_entries[name] = {
            "weight": weight, "relative_shard_dir": name,
            **{key: source_manifest[key] for key in (
                "train_shards", "val_shards", "train_tokens", "val_tokens",
                "train_documents", "val_documents", "unique_documents",
                "duplicate_documents", "dataset", "revision", "license",
                "reconstruction", "length_stats_file", "provenance_file",
                "duplicates_file")},
        }
    manifest["sources"] = {**legacy, **new_entries}
    existing_expected = [name for name in manifest.get("expected_sources", []) if name not in all_long_names]
    manifest["expected_sources"] = existing_expected + list(weights)
    manifest.setdefault("source_paths", {})
    for name in all_long_names:
        manifest["source_paths"].pop(name, None)
    for name in weights:
        manifest["source_paths"][name] = str((args.data_root.resolve() / config["sources"][name]["raw_subdir"]))
    manifest["complete"] = True
    manifest["missing_sources"] = []
    manifest["long_form_mixture"] = {
        "config": str(args.config.resolve()), "config_sha256": file_sha256(args.config),
        "stage": args.stage, "target_fraction": stage["long_form_fraction"],
        "sources": weights, "integrated_at": datetime.now(timezone.utc).isoformat(),
    }
    backup = path.with_name(f"mixture_manifest.before_longform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy2(path, backup)
    _atomic_json(path, manifest)
    loaded, _ = load_weighted_shard_sources(root)
    print(f"Integrated {args.stage}: {len(loaded)} total sources; backup {backup}")
    print("Long-form weights:", json.dumps(weights, indent=2))


def report(args, config: dict):
    names = stage_sources(config, args.stage)
    combined_tokens = combined_docs = 0
    print("source\tunique_docs\ttotal_tokens\t>8K\t>16K\t>32K\t>65K")
    for name in names:
        stats = json.loads((args.output_root / name / "length_stats.json").read_text())
        manifest = json.loads((args.output_root / name / "source_manifest.json").read_text())
        total_tokens = manifest["train_tokens"] + manifest["val_tokens"]
        combined_tokens += total_tokens; combined_docs += stats["unique_documents"]
        p = stats["proportions_exceeding"]
        print(f"{name}\t{stats['unique_documents']}\t{total_tokens}\t"
              f"{p['8192']:.4%}\t{p['16384']:.4%}\t{p['32768']:.4%}\t{p['65536']:.4%}")
    print(f"TOTAL\t{combined_docs}\t{combined_tokens}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare = sub.add_parser("prepare-stage")
    prepare.add_argument("--stage", choices=("priority1", "priority2"), required=True)
    prepare.add_argument("--source", action="append", default=None)
    prepare.add_argument("--data-root", type=Path, default=Path("../pretraining_data"))
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--pending-batches", type=int, default=16)
    prepare.add_argument("--batch-documents", type=int, default=16)
    prepare.add_argument("--shard-size-mb", type=float, default=256.0)
    prepare.add_argument("--max-documents", type=int, default=None,
                         help="smoke test only; capped sources cannot be integrated")
    integrate_parser = sub.add_parser("integrate")
    integrate_parser.add_argument("--stage", choices=("priority1", "full"), required=True)
    integrate_parser.add_argument("--data-root", type=Path, default=Path("../pretraining_data"))
    integrate_parser.add_argument("--output-root", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--stage", choices=("priority1", "priority2", "full"), required=True)
    report_parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.operation == "prepare-stage":
        if args.workers < 1 or args.pending_batches < 1 or args.batch_documents < 1:
            raise ValueError("workers, pending-batches, and batch-documents must be positive")
        allowed = stage_sources(config, args.stage)
        requested = args.source or allowed
        unknown = set(requested) - set(allowed)
        if unknown:
            raise ValueError(f"sources not in {args.stage}: {sorted(unknown)}")
        for name in requested:
            prepare_source(args, config, name)
    elif args.operation == "integrate":
        integrate(args, config)
    else:
        report(args, config)


if __name__ == "__main__":
    main()
