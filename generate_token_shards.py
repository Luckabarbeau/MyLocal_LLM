#!/usr/bin/env python3
"""Generate token shards for Cosmopedia-v2 dataset.

This script processes multiple Parquet shards and generates binary token shards
for efficient training. It supports:
- Parallel processing of multiple Parquet shards
- Automatic tokenizer training on sample
- Output directory management
- Progress reporting

Usage:
    python generate_token_shards.py \\
        --dataset-path /path/to/cosmopedia-v2 \\
        --output-dir ./token_shards \\
        --num-shards 104 \\
        --documents-per-shard 10000 \\
        --context-length 512 \\
        --batch-size 100
"""

import argparse
import time
from pathlib import Path

from mini_llm.data.parquet_reader import CosmopediaParquetReader
from mini_llm.data.token_shards import TokenShardGenerator, TokenShardWriter


def generate_token_shards(
    dataset_path: str,
    output_dir: str,
    num_parquet_shards: int,
    documents_per_shard: int = 10_000,
    context_length: int = 512,
    batch_size: int = 100,
    max_shards: int = None,
    start_shard: int = 0,
) -> list:
    """
    Generate token shards from Cosmopedia Parquet data.
    
    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_dir: Output directory for binary shards
        num_parquet_shards: Number of Parquet shards to process
        documents_per_shard: Target documents per binary shard
        context_length: Maximum sequence length
        batch_size: Documents to process at once
        max_shards: Maximum binary shards to generate (None = all)
        start_shard: Starting Parquet shard index
        
    Returns:
        List of paths to generated binary shards
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating token shards from {num_parquet_shards} Parquet shards")
    print(f"Output directory: {output_dir}")
    print(f"Documents per shard: {documents_per_shard}")
    print(f"Context length: {context_length}")
    print()
    
    # Create parquet reader for a limited number of shards
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    
    # Limit to requested Parquet shards
    parquet_reader._shard_files = parquet_reader._shard_files[start_shard:start_shard + num_parquet_shards]
    print(f"Processing Parquet shards {start_shard} to {start_shard + num_parquet_shards}")
    print(f"Total Parquet shards to process: {len(parquet_reader._shard_files)}")
    print()
    
    # Train tokenizer on a sample (limited to avoid memory issues)
    print("Training tokenizer on sample...")
    start_time = time.time()
    sample_texts = []
    count = 0
    for record in parquet_reader.iter_records():
        sample_texts.append(record.get("text", ""))
        count += 1
        if count >= 500:  # Limit to 500 documents for tokenizer training
            break
    
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=16_384)
    tokenizer.train(sample_texts)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Tokenizer training took {time.time() - start_time:.1f} seconds")
    print(f"Vocabulary size: {len(tokenizer)}")
    print()
    
    # Generate shards
    generator = TokenShardGenerator(
        tokenizer=tokenizer,
        parquet_reader=parquet_reader,
        output_dir=str(output_dir),
        context_length=context_length,
        documents_per_shard=documents_per_shard,
        batch_size=batch_size,
    )
    
    shard_paths = generator.generate_shards(max_shards=max_shards)
    
    print()
    print("=" * 60)
    print("Generation complete!")
    print(f"  Binary shards written: {len(shard_paths)}")
    print(f"  Total documents processed: {sum(1 for _ in parquet_reader.iter_records())}")
    print(f"  Output directory: {output_dir}")
    
    return shard_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate token shards from Cosmopedia-v2 Parquet data"
    )
    parser.add_argument(
        "--dataset-path",
        default="../cosmopedia-v2/cosmopedia-v2",
        help="Path to Cosmopedia Parquet directory",
    )
    parser.add_argument(
        "--output-dir",
        default="./token_shards",
        help="Output directory for binary shards",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=104,
        help="Number of Parquet shards to process",
    )
    parser.add_argument(
        "--documents-per-shard",
        type=int,
        default=10_000,
        help="Documents per binary shard",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents to process at once",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum binary shards to generate",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Starting Parquet shard index",
    )
    
    args = parser.parse_args()
    
    generate_token_shards(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_parquet_shards=args.num_shards,
        documents_per_shard=args.documents_per_shard,
        context_length=args.context_length,
        batch_size=args.batch_size,
        max_shards=args.max_shards,
        start_shard=args.start_shard,
    )


if __name__ == "__main__":
    main()
