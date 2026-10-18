#!/usr/bin/env python3
"""Create packed Cosmopedia token shards from a pre-trained tokenizer.

The output is a pair of contiguous token streams:

    train_shard_00000.bin, ...
    val_shard_00000.bin, ...

Documents are tokenized completely, terminated with EOS, and concatenated
without padding.  Because the resulting shards are context-length independent,
the same 64k-tokenizer dataset can be reused for 4k, 8k, 16k, ... context
experiments.
"""

import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer


def load_tokenizer(path: str):
    try:
        return FastBPETokenizer.load(path)
    except Exception:
        return SimpleBPETokenizer.load(path)


def tokenizer_token_id(tokenizer, token: str) -> int:
    mapping = getattr(tokenizer, "token_to_id", None)
    if isinstance(mapping, dict):
        value = mapping.get(token)
    elif getattr(tokenizer, "_tokenizer", None) is not None:
        value = tokenizer._tokenizer.token_to_id(token)
    else:
        value = None
    if value is None:
        raise ValueError(f"tokenizer does not contain required token {token!r}")
    return int(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_validation_document(text: str, val_ratio: float, seed: int) -> bool:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if val_ratio == 0.0:
        return False
    digest = hashlib.sha256()
    digest.update(int(seed).to_bytes(8, byteorder="little", signed=True))
    digest.update(text.encode("utf-8"))
    value = int.from_bytes(digest.digest()[:8], byteorder="little")
    return value / float(1 << 64) < val_ratio


class PackedShardWriter:
    """Stream token IDs into fixed-size raw binary shards without padding."""

    def __init__(self, output_dir: Path, prefix: str, shard_size_mb: float, dtype):
        if shard_size_mb <= 0:
            raise ValueError("shard_size_mb must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = str(prefix)
        self.dtype = np.dtype(dtype)
        self.tokens_per_shard = max(
            1, int(shard_size_mb * 1024 * 1024 / self.dtype.itemsize)
        )
        self.shard_index = 0
        self.tokens_in_shard = 0
        self.total_tokens = 0
        self.paths = []
        self._handle = None

    def _open_if_needed(self):
        if self._handle is None:
            path = self.output_dir / f"{self.prefix}_shard_{self.shard_index:05d}.bin"
            self._handle = open(path, "wb")
            self.paths.append(path)

    def _rotate(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.shard_index += 1
        self.tokens_in_shard = 0

    def write(self, token_ids):
        values = np.asarray(token_ids, dtype=self.dtype).reshape(-1)
        offset = 0
        while offset < values.size:
            self._open_if_needed()
            remaining = self.tokens_per_shard - self.tokens_in_shard
            take = min(remaining, values.size - offset)
            values[offset : offset + take].tofile(self._handle)
            offset += take
            self.tokens_in_shard += take
            self.total_tokens += take
            if self.tokens_in_shard == self.tokens_per_shard:
                self._rotate()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def encode_and_write_batch(
    tokenizer,
    texts,
    split_flags,
    eos_id,
    train_writer,
    val_writer,
):
    encoded = tokenizer.encode_batch(texts)
    if len(encoded) != len(texts):
        raise RuntimeError("tokenizer returned a different number of documents")
    train_docs = 0
    val_docs = 0
    max_token_id = eos_id
    for ids, is_val in zip(encoded, split_flags):
        if ids:
            max_token_id = max(max_token_id, max(ids))
            values = np.empty(len(ids) + 1, dtype=train_writer.dtype)
            values[:-1] = ids
            values[-1] = eos_id
        else:
            values = np.asarray([eos_id], dtype=train_writer.dtype)
        if is_val:
            val_writer.write(values)
            val_docs += 1
        else:
            train_writer.write(values)
            train_docs += 1
    return train_docs, val_docs, max_token_id



# Shared packed shard engine: both the original Cosmopedia CLI and the
# multi-corpus 32k ensemble preparation use the same EOS-preserving writer.
# In contrast, legacy generate_token_shards.py truncates to context_length.
_WORKER_TOKENIZER = None


def _init_packed_worker(tokenizer_path: str):
    global _WORKER_TOKENIZER
    # Multiple processes are the parallelism axis, not nested Rayon pools.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_path)


def _encode_packed_worker(texts):
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("packed shard worker tokenizer was not initialized")
    return _WORKER_TOKENIZER.encode_batch(texts)


def _write_packed_ids(encoded, flags, eos_id, vocab_size, train_writer, val_writer):
    """One canonical writer for independently encoded documents."""
    if len(encoded) != len(flags):
        raise RuntimeError("tokenizer returned incorrect document count")
    train_docs = val_docs = 0
    for ids, is_val in zip(encoded, flags):
        # Validate *before* uint16 cast, otherwise malformed IDs could wrap.
        if ids and (min(ids) < 0 or max(ids) >= vocab_size):
            raise ValueError("tokenizer emitted token ID outside vocabulary")
        values = np.empty(len(ids) + 1, dtype=np.uint16)
        if ids:
            values[:-1] = ids
        values[-1] = eos_id
        if is_val:
            val_writer.write(values)
            val_docs += 1
        else:
            train_writer.write(values)
            train_docs += 1
    return train_docs, val_docs


def pack_text_corpus(
    reader,
    *,
    output_dir: Path,
    tokenizer_path: Path,
    seed: int = 42,
    val_ratio: float = 0.01,
    shard_size_mb: float = 256,
    batch_documents: int = 256,
    workers: int = 8,
    pending_batches: int | None = None,
    max_documents: int | None = None,
):
    """Encode a source with bounded multiprocessing and per-input-file shards.

    Each discovered source file gets its own file_XXXXX/ directory. There is no
    truncation, no mixing between files or sources, and no reliance on workers
    finishing in the same order. For large *single* files, batches are still
    spread over the entire worker pool. Output is deterministic across worker
    counts, given the same tokenizer and input files.
    """
    if workers < 1 or batch_documents < 1:
        raise ValueError("workers and batch_documents must be >= 1")
    if pending_batches is None:
        pending_batches = max(2, workers * 2)
    if pending_batches < 1:
        raise ValueError("pending_batches must be >= 1")
    if not 0 < val_ratio < 1:
        raise ValueError("0 < val_ratio < 1 required for train/val shards")
    tokenizer_path = Path(tokenizer_path)
    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer)
    if not 0 < vocab_size <= 65_536:
        raise ValueError("tokenizer vocabulary must fit uint16")
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    files = reader.discover_files()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing source: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_packed_worker,
            initargs=(str(tokenizer_path),),
        )
    n_train = n_val = n_total = 0
    train_tokens = val_tokens = 0
    train_shards = val_shards = 0
    file_manifest = []
    started = time.monotonic()
    try:
        for index, source_path in enumerate(files):
            if max_documents is not None and n_total >= max_documents:
                break
            file_dir = output_dir / f"file_{index:05d}"
            train_writer = PackedShardWriter(file_dir, "train", shard_size_mb, "uint16")
            val_writer = PackedShardWriter(file_dir, "val", shard_size_mb, "uint16")
            queue = deque()
            this_train = this_val = 0
            current_batch = []

            def commit(encoded, flags):
                nonlocal this_train, this_val
                t, v = _write_packed_ids(encoded, flags, eos_id, vocab_size,
                                          train_writer, val_writer)
                this_train += t
                this_val += v

            def drain_one():
                future, flags = queue.popleft()
                commit(future.result(), flags)

            def submit_batch():
                if not current_batch:
                    return
                texts = current_batch[:]
                flags = [is_validation_document(text, val_ratio, seed) for text in texts]
                current_batch.clear()
                if pool is None:
                    commit(tokenizer.encode_batch(texts), flags)
                else:
                    queue.append((pool.submit(_encode_packed_worker, texts), flags))
                    if len(queue) >= pending_batches:
                        drain_one()

            try:
                for text in reader.iter_file_texts(source_path):
                    if not text:
                        continue
                    current_batch.append(text)
                    n_total += 1
                    if len(current_batch) >= batch_documents:
                        submit_batch()
                    if max_documents is not None and n_total >= max_documents:
                        break
                submit_batch()
                while queue:
                    drain_one()
            finally:
                # A partial failure is deliberately NOT registered as a
                # completed source; re-run in a fresh source directory.
                train_writer.close()
                val_writer.close()
            n_train += this_train
            n_val += this_val
            train_tokens += train_writer.total_tokens
            val_tokens += val_writer.total_tokens
            train_shards += len(train_writer.paths)
            val_shards += len(val_writer.paths)
            if this_train + this_val:
                file_manifest.append(dict(
                    source_file=str(source_path.relative_to(reader.path))
                    if reader.path.is_dir() else source_path.name,
                    shard_subdir=file_dir.name,
                    train_documents=this_train, val_documents=this_val,
                    train_tokens=train_writer.total_tokens,
                    val_tokens=val_writer.total_tokens,
                    train_shards=len(train_writer.paths),
                    val_shards=len(val_writer.paths),
                ))
            elapsed = max(time.monotonic() - started, 1e-9)
            print(f"  {reader.name}: {n_total:,} documents / "
                  f"{train_tokens + val_tokens:,} tokens "
                  f"({(train_tokens + val_tokens)/elapsed:,.0f} tok/s); "
                  f"file {index + 1}/{len(files)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if not train_shards or not val_shards:
        raise ValueError(f"{reader.name} needs both train and validation shards")
    return dict(train_shards=train_shards, val_shards=val_shards,
                train_tokens=train_tokens, val_tokens=val_tokens,
                train_documents=n_train, val_documents=n_val,
                source_files=file_manifest, workers=workers)


def generate(args):
    from mini_llm.data.parquet_reader import CosmopediaParquetReader

    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer)
    if vocab_size <= 0:
        raise ValueError("loaded tokenizer has an empty vocabulary")
    if vocab_size > 65_536:
        raise ValueError(
            f"vocabulary size {vocab_size} exceeds the uint16 training format"
        )
    eos_id = tokenizer_token_id(tokenizer, tokenizer.eos_token)
    dtype = np.uint16

    reader = CosmopediaParquetReader(dataset_path=Path(args.dataset_path))
    start = int(args.start_shard)
    stop = None if args.num_shards is None else start + int(args.num_shards)
    reader.shard_paths = reader.shard_paths[start:stop]
    if not reader.shard_paths:
        raise ValueError("no Parquet shards selected")

    train_writer = PackedShardWriter(
        output_dir, "train", args.shard_size_mb, dtype=dtype
    )
    val_writer = PackedShardWriter(
        output_dir, "val", args.shard_size_mb, dtype=dtype
    )

    texts = []
    split_flags = []
    train_docs = 0
    val_docs = 0
    documents = 0
    max_token_id = eos_id

    def flush_batch():
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
        for record in reader.iter_records():
            text = record.get("text", "")
            if not text:
                continue
            texts.append(text)
            split_flags.append(
                is_validation_document(text, args.val_ratio, args.seed)
            )
            documents += 1
            if len(texts) >= args.batch_documents:
                flush_batch()
                if documents % (args.batch_documents * 20) == 0:
                    print(
                        f"{documents:,} docs | "
                        f"train {train_writer.total_tokens:,} tok | "
                        f"val {val_writer.total_tokens:,} tok"
                    )
            if args.max_documents is not None and documents >= args.max_documents:
                break
        flush_batch()
    finally:
        train_writer.close()
        val_writer.close()

    if not train_writer.paths:
        raise RuntimeError("no training shards were generated")
    if args.val_ratio > 0 and not val_writer.paths:
        raise RuntimeError("no validation shards were generated")
    if max_token_id >= vocab_size:
        raise RuntimeError(
            f"encoded token ID {max_token_id} is outside tokenizer vocabulary {vocab_size}"
        )

    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    metadata_path = tokenizer_path.with_suffix(".meta.json")
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_dir / "tokenizer.meta.json")

    manifest = {
        "format_version": "packed_token_stream_v1",
        "source_dataset": "Cosmopedia-v2",
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "vocab_size": vocab_size,
        "token_dtype": "uint16",
        "eos_token_id": eos_id,
        "preprocessing_seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "context_independent": True,
        "documents": {
            "total": documents,
            "train": train_docs,
            "val": val_docs,
        },
        "tokens": {
            "train": train_writer.total_tokens,
            "val": val_writer.total_tokens,
            "total": train_writer.total_tokens + val_writer.total_tokens,
        },
        "shards": {
            "train": len(train_writer.paths),
            "val": len(val_writer.paths),
            "target_size_mb": float(args.shard_size_mb),
        },
    }
    with open(output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nPacked Cosmopedia preprocessing complete")
    print(f"  tokenizer vocab: {vocab_size:,}")
    print(f"  documents:       {documents:,}")
    print(f"  train tokens:    {train_writer.total_tokens:,}")
    print(f"  val tokens:      {val_writer.total_tokens:,}")
    print(f"  train shards:    {len(train_writer.paths)}")
    print(f"  val shards:      {len(val_writer.paths)}")
    print(f"  output:          {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate context-independent packed Cosmopedia token shards"
    )
    parser.add_argument(
        "--dataset-path", default="../cosmopedia-v2/cosmopedia-v2/"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="./token_shards_64k")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of source Parquet shards (default: all remaining)",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--batch-documents", type=int, default=1024)
    parser.add_argument("--shard-size-mb", type=float, default=256.0)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
