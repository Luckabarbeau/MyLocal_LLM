#!/usr/bin/env python3
"""Train a BPE tokenizer on a representative sample of Cosmopedia-v2.

This script trains a tokenizer ONCE and saves it to disk.
The trained tokenizer can then be reused for encoding multiple datasets.

Usage:
    # Train tokenizer on 100k documents with vocab size 16384
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --vocab-size 16384 \
        --max-documents 100000 \
        --output tokenizer.json

    # Train on a smaller sample for quick testing
    python train_tokenizer.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --vocab-size 16384 \
        --max-documents 5000 \
        --output tokenizer_test.json
"""

import argparse
import time
from pathlib import Path

from mini_llm.data.parquet_reader import CosmopediaParquetReader
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


def train_tokenizer(
    dataset_path: str,
    output_path: str,
    vocab_size: int = 16_384,
    max_documents: int = 50_000,
    max_bytes: int = None,
) -> dict:
    """
    Train a BPE tokenizer on a bounded sample of documents.
    
    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_path: Path to save trained tokenizer
        vocab_size: Target vocabulary size
        max_documents: Maximum number of documents to use for training
        max_bytes: Maximum bytes of text to use (optional, overrides max_documents)
        
    Returns:
        Training statistics dictionary
    """
    dataset_path = Path(dataset_path)
    
    print(f"Training BPE tokenizer on {dataset_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Max documents: {max_documents}")
    if max_bytes:
        print(f"Max bytes: {max_bytes:,}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Create parquet reader and sample texts
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    
    print("Sampling documents from dataset...")
    sample_texts = []
    total_chars = 0
    documents_used = 0
    
    for record in parquet_reader.iter_records():
        text = record.get("text", "")
        if text:
            # Check byte limit if specified
            if max_bytes and total_chars + len(text) > max_bytes:
                break
            
            sample_texts.append(text)
            total_chars += len(text)
            documents_used += 1
            
            if documents_used >= max_documents:
                break
        
        # Progress indicator every 10k docs
        if documents_used % 10_000 == 0 and documents_used > 0:
            print(f"  Sampled {documents_used} documents ({total_chars:,} chars)")
    
    elapsed = time.time() - start_time
    print(f"  Collected {documents_used} documents in {elapsed:.1f}s")
    print(f"  Total characters: {total_chars:,}")
    print()
    
    # Train tokenizer
    print("Training BPE tokenizer...")
    train_start = time.time()
    
    tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(sample_texts)
    
    train_elapsed = time.time() - train_start
    print(f"  Training completed in {train_elapsed:.1f}s")
    print()
    
    # Collect statistics
    stats = {
        "vocab_size": len(tokenizer),
        "num_documents_used": documents_used,
        "total_chars": total_chars,
        "training_time_seconds": train_elapsed,
        "initial_vocab_size": vocab_size,
    }
    
    # Print statistics
    print("=" * 60)
    print("Tokenizer Training Statistics")
    print("=" * 60)
    print(f"Documents used: {documents_used:,}")
    print(f"Characters sampled: {total_chars:,}")
    print(f"Initial vocabulary target: {vocab_size}")
    print(f"Final vocabulary size: {len(tokenizer):,}")
    print(f"Number of BPE merges: {len(tokenizer.merges):,}")
    print(f"Training time: {train_elapsed:.1f}s")
    print()
    
    # Save tokenizer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to: {output_path}")
    
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
        default=50_000,
        help="Maximum number of documents for training (default: 50000)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Maximum bytes of text to use (overrides max-documents if set)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save trained tokenizer JSON",
    )
    
    args = parser.parse_args()
    
    train_tokenizer(
        dataset_path=args.dataset_path,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_documents=args.max_documents,
        max_bytes=args.max_bytes,
    )


if __name__ == "__main__":
    main()
