"""Document-preserving packed-shard preparation primitives.

The public preparation entry point is ``prepare_ensemble_mixture.py``. This
module owns its context-independent writers, deterministic document split, and
bounded parallel encoding implementation.
"""

import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
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
            assert self._handle is not None
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


# Shared packed-shard engine for the multi-corpus 32K ensemble preparation.
# Documents remain complete and EOS-delimited; no context-length truncation.
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
            current_split_keys = []

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
                split_keys = current_split_keys[:]
                flags = [is_validation_document(key, val_ratio, seed) for key in split_keys]
                current_batch.clear()
                current_split_keys.clear()
                if pool is None:
                    commit(tokenizer.encode_batch(texts), flags)
                else:
                    queue.append((pool.submit(_encode_packed_worker, texts), flags))
                    if len(queue) >= pending_batches:
                        drain_one()

            try:
                documents = (
                    reader.iter_file_documents(source_path)
                    if hasattr(reader, "iter_file_documents")
                    else ((text, text) for text in reader.iter_file_texts(source_path))
                )
                for text, split_key in documents:
                    if not text:
                        continue
                    current_batch.append(text)
                    current_split_keys.append(split_key)
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

