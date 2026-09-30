#!/usr/bin/env python3
"""Train a BPE tokenizer on a representative sample of Cosmopedia-v2.

This script trains a tokenizer ONCE and saves it to disk.
The trained tokenizer can then be reused for encoding multiple datasets.

Usage:
    # Train tokenizer with Simple (reference) backend
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --backend simple \
        --vocab-size 16384 \
        --max-documents 50000 \
        --output tokenizer_simple.json

    # Train tokenizer with Fast (production) backend
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --backend fast \
        --vocab-size 16384 \
        --max-documents 50000 \
        --threads 8 \
        --output tokenizer_fast.json

    # Quick smoke test
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --backend fast \
        --vocab-size 512 \
        --max-bytes 5000000 \
        --output tokenizer_smoke.json

    # Reuse existing tokenizer if available
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --backend fast \
        --vocab-size 16384 \
        --tokenizer-path tokenizer.json \
        --output tokenizer_new.json

"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from mini_llm.data.parquet_reader import CosmopediaParquetReader

# Import tokenizer backends
from mini_llm.tokenizer.tokenizer import (
    SimpleBPETokenizer,
    FastBPETokenizer,
    TokenizerProtocol,
)


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_sample_texts_generator(
    parquet_reader: CosmopediaParquetReader,
    max_documents: int = None,
    max_bytes: int = None,
    progress_callback=None,
) -> tuple:
    """
    Create a generator that yields text samples from the dataset.

    Args:
        parquet_reader: Parquet reader instance
        max_documents: Maximum number of documents to yield
        max_bytes: Maximum bytes of text to yield
        progress_callback: Optional callback(progress_str)

    Returns:
        Tuple of (generator, sample_info_dict)
    """
    sample_texts = []
    total_chars = 0
    documents_used = 0
    shards_scanned = 0

    for record in parquet_reader.iter_records():
        text = record.get("text", "")
        if text:
            # Check limits
            if max_documents and documents_used >= max_documents:
                break
            if max_bytes and total_chars + len(text) > max_bytes:
                break

            sample_texts.append(text)
            total_chars += len(text)
            documents_used += 1

            # Progress reporting
            if progress_callback and documents_used % 10000 == 0:
                progress_callback(f"Sampled {documents_used} documents ({total_chars / (1024*1024):.1f} MB)")

        shards_scanned = parquet_reader.num_shards

    sample_info = {
        "documents": documents_used,
        "total_chars": total_chars,
        "total_bytes": total_chars,  # For ASCII text
        "shards_scanned": shards_scanned,
    }

    return iter(sample_texts), sample_info


def train_tokenizer(
    dataset_path: str,
    output_path: str,
    vocab_size: int = 16_384,
    max_documents: int = None,
    max_bytes: int = None,
    backend: str = "fast",
    threads: int = 8,
    force_retrain: bool = False,
    tokenizer_path: str = None,
    smoke_test: bool = False,
) -> dict:
    """
    Train a BPE tokenizer on a bounded sample of documents.

    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_path: Path to save trained tokenizer
        vocab_size: Target vocabulary size
        max_documents: Maximum number of documents to use for training
        max_bytes: Maximum bytes of text to use (overrides max_documents if set)
        backend: "simple" or "fast"
        threads: Number of threads for fast backend
        force_retrain: Force retraining even if tokenizer exists
        tokenizer_path: Path to existing tokenizer to reuse
        smoke_test: If True, use minimal sample size

    Returns:
        Training statistics dictionary
    """
    dataset_path = Path(dataset_path)

    # Determine sample size based on mode
    if smoke_test:
        max_bytes = 5_000_000  # 5 MB for smoke test
        vocab_size = min(vocab_size, 512)
        max_documents = 500
        print("SMOKE TEST MODE - Using minimal sample size")
    else:
        if max_bytes is None and max_documents is None:
            # Default: reasonable sample for production
            max_bytes = 750_000_000  # 750 MB

    print(f"Training BPE tokenizer on {dataset_path}")
    print(f"Target vocab size: {vocab_size:,}")
    print(f"Backend: {backend}")
    print(f"Threads: {threads}")

    if max_bytes:
        print(f"Max bytes: {max_bytes:,} ({max_bytes / (1024*1024):.1f} MB)")
    if max_documents:
        print(f"Max documents: {max_documents:,}")
    print()

    # Check for existing tokenizer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_retrain and output_path.exists():
        print(f"Tokenizer already exists at: {output_path}")
        print("Use --force-retrain to overwrite.")
        return None

    # Start timing
    start_time = time.time()

    # Create parquet reader
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    print(f"Found {parquet_reader.num_shards} Parquet shards")

    # Progress reporting
    def progress(msg):
        print(f"  {msg}")

    # Sample texts
    print("\nSampling documents from dataset...")
    texts, sample_info = create_sample_texts_generator(
        parquet_reader,
        max_documents=max_documents,
        max_bytes=max_bytes,
        progress_callback=progress if not smoke_test else None,
    )

    elapsed = time.time() - start_time
    print(f"\n  Sampled {sample_info['documents']:,} documents in {elapsed:.1f}s")
    print(f"  Total characters: {sample_info['total_chars']:,}")
    print()

    # Train tokenizer
    print(f"Training {backend} tokenizer...")
    train_start = time.time()

    if backend == "simple":
        tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
        # Convert generator to list for simple tokenizer
        texts_list = list(texts)
        tokenizer.train(texts_list)

    elif backend == "fast":
        tokenizer = FastBPETokenizer(vocab_size=vocab_size, threads=threads)
        tokenizer.train(texts)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    train_elapsed = time.time() - train_start
    print(f"  Training completed in {train_elapsed:.1f}s")

    # Collect statistics
    stats = {
        "vocab_size": len(tokenizer),
        "num_documents_used": sample_info['documents'],
        "total_chars": sample_info['total_chars'],
        "training_time_seconds": train_elapsed,
        "initial_vocab_size": vocab_size,
        "backend": backend,
        "threads": threads,
    }

    # Print statistics
    print()
    print("=" * 60)
    print("Tokenizer Training Statistics")
    print("=" * 60)
    print(f"Documents used: {sample_info['documents']:,}")
    print(f"Characters sampled: {sample_info['total_chars']:,}")
    print(f"Initial vocabulary target: {vocab_size:,}")
    print(f"Final vocabulary size: {len(tokenizer):,}")
    if backend == "simple":
        print(f"Number of BPE merges: {len(tokenizer.merges):,}")
    print(f"Training time: {train_elapsed:.1f}s")
    print()

    # Save tokenizer
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to: {output_path}")

    # Save metadata
    meta_path = output_path.with_suffix('.meta.json')
    meta = {
        "file_hash": compute_file_hash(str(output_path)),
        "vocab_size": len(tokenizer),
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
        },
        "training": {
            "backend": backend,
            "threads": threads,
            "vocab_target": vocab_size,
            "documents_used": sample_info['documents'],
            "bytes_used": sample_info['total_chars'],
            "training_time_seconds": train_elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on Cosmopedia-v2 dataset"
    )
    parser.add_argument(
        "--dataset-path",
        default="../cosmopedia-v2/cosmopedia-v2",
        help="Path to Cosmopedia Parquet directory",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16_384,
        help="Target vocabulary size (default: 16384)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents for training",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Maximum bytes of text to use (overrides max-documents if set)",
    )
    parser.add_argument(
        "--backend",
        choices=["simple", "fast"],
        default="fast",
        help="Tokenizer backend (default: fast)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for fast backend (default: 8)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if tokenizer exists",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to existing tokenizer to reuse",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save trained tokenizer",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test with minimal sample",
    )

    args = parser.parse_args()

    train_tokenizer(
        dataset_path=args.dataset_path,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_documents=args.max_documents,
        max_bytes=args.max_bytes,
        backend=args.backend,
        threads=args.threads,
        force_retrain=args.force_retrain,
        tokenizer_path=args.tokenizer_path,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
