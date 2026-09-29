#!/usr/bin/env python3
"""Generate token shards from Cosmopedia-v2 dataset with parallel processing.

This script processes Parquet files and generates binary token shards.
It uses multiprocessing to accelerate tokenization across CPU cores.

## Backend

Token shard generation uses pure Python (tokenizer) and PyArrow.
It doesn't use NumPy/CuPy, so the backend setting doesn't affect this script.

Usage:
    # Generate from all 104 Parquet shards (default)
    python generate_token_shards.py --dataset-path ../cosmopedia-v2/cosmopedia-v2/

    # Process only first 5 Parquet shards for testing
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --num-shards 5 \
        --documents-per-shard 1000

    # Limit total token shards generated
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --max-shards 10

    # Use 8 parallel workers for tokenization
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --num-workers 8
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count


def worker_tokenize(args):
    """Worker function to tokenize documents in a batch."""
    batch_docs = args  # List of (text, doc_id) tuples
    tokenizer, context_length = batch_docs[0][2], batch_docs[0][3]
    
    results = []
    for text, doc_id in batch_docs:
        if text:
            ids = tokenizer.encode(text)
            # Truncate to context length
            if len(ids) > context_length:
                ids = ids[:context_length]
            if ids:  # Only add non-empty documents
                results.append((doc_id, ids))
    return results


def generate_token_shards(
    dataset_path: str,
    output_dir: str,
    num_parquet_shards: int = 104,
    documents_per_shard: int = 10_000,
    context_length: int = 512,
    batch_size: int = 100,
    max_token_shards: int = None,
    start_parquet_shard: int = 0,
    num_workers: int = 4,
) -> list:
    """
    Generate token shards from Cosmopedia Parquet data.
    
    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_dir: Output directory for binary shards
        num_parquet_shards: Number of Parquet shards to process
        documents_per_shard: Target documents per binary shard
        context_length: Maximum sequence length
        batch_size: Documents to tokenize at once per worker
        max_token_shards: Maximum token shards to generate (None = all)
        start_parquet_shard: Starting Parquet shard index
        num_workers: Number of parallel worker processes
        
    Returns:
        List of paths to generated token shards
    """
    from mini_llm.data.parquet_reader import CosmopediaParquetReader
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    from mini_llm.data.token_shards import TokenShardWriter
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating token shards from {num_parquet_shards} Parquet shards")
    print(f"Using {num_workers} parallel workers")
    print(f"Output directory: {output_dir}")
    print(f"Documents per shard: {documents_per_shard}")
    print(f"Context length: {context_length}")
    print()
    
    # Create parquet reader
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    
    # Limit to requested Parquet shards
    parquet_reader.shard_paths = parquet_reader.shard_paths[
        start_parquet_shard:start_parquet_shard + num_parquet_shards
    ]
    print(f"Processing Parquet shards {start_parquet_shard} to {start_parquet_shard + num_parquet_shards}")
    print(f"Total Parquet shards to process: {len(parquet_reader.shard_paths)}")
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
    
    tokenizer = SimpleBPETokenizer(vocab_size=16_384)
    tokenizer.train(sample_texts)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Tokenizer training took {time.time() - start_time:.1f} seconds")
    print(f"Vocabulary size: {len(tokenizer)}")
    print()
    
    # Tokenize and write shards using multiprocessing
    shard_writer = TokenShardWriter(
        str(output_dir),
        context_length=context_length,
        dtype="uint16"
    )
    
    all_shard_paths = []
    current_batch_docs: list = []
    documents_processed = 0
    token_shards_written = 0
    
    print("Processing Parquet shards with multiprocessing...")
    for parquet_idx, shard_path in enumerate(parquet_reader.shard_paths):
        print(f"  Processing Parquet shard {parquet_idx + 1}/{len(parquet_reader.shard_paths)}: {shard_path.name}")
        
        # Read this Parquet shard
        table = parquet_reader._get_table_for_shard(shard_path)
        
        # Collect documents for batch processing
        all_docs = []
        for doc_id, record in enumerate(table.to_pylist()):
            text = record.get("text", "")
            all_docs.append((text, doc_id, tokenizer, context_length))
        
        # Tokenize in parallel using worker pool
        if num_workers > 1 and len(all_docs) > 100:
            print(f"    Tokenizing {len(all_docs)} documents with {num_workers} workers...")
            start_batch_time = time.time()
            
            # Split into batches for each worker
            batch_size_per_worker = max(1, len(all_docs) // (num_workers * 4))
            batches = []
            for i in range(0, len(all_docs), batch_size_per_worker):
                batches.append(all_docs[i:i + batch_size_per_worker])
            
            # Process batches in parallel
            with Pool(processes=num_workers) as pool:
                results = pool.map(worker_tokenize, batches)
            
            # Flatten results
            for batch_result in results:
                current_batch_docs.extend(batch_result)
            
            print(f"    Tokenization took {time.time() - start_batch_time:.1f} seconds")
        else:
            # Single-threaded fallback for small datasets
            for text, doc_id in all_docs:
                if text:
                    ids = tokenizer.encode(text)
                    if len(ids) > context_length:
                        ids = ids[:context_length]
                    if ids:
                        current_batch_docs.append((doc_id, ids))
        
        documents_processed += len(all_docs)
        
        # Flush documents to token shards
        while len(current_batch_docs) >= documents_per_shard:
            shard_docs = [doc[1] for doc in current_batch_docs[:documents_per_shard]]
            current_batch_docs = current_batch_docs[documents_per_shard:]
            
            shard_path = shard_writer.write_shard(token_shards_written, shard_docs)
            if shard_path:
                all_shard_paths.append(shard_path)
                token_shards_written += 1
                print(f"    Written token shard {token_shards_written}: {shard_path.name}")
            
            if max_token_shards and token_shards_written >= max_token_shards:
                break
        
        if max_token_shards and token_shards_written >= max_token_shards:
            break
    
    # Flush remaining documents
    if current_batch_docs and (not max_token_shards or token_shards_written < max_token_shards):
        shard_docs = [doc[1] for doc in current_batch_docs]
        shard_path = shard_writer.write_shard(token_shards_written, shard_docs)
        if shard_path:
            all_shard_paths.append(shard_path)
            token_shards_written += 1
            print(f"    Written final token shard {token_shards_written}: {shard_path.name}")
    
    print()
    print("=" * 60)
    print("Generation complete!")
    print(f"  Token shards written: {len(all_shard_paths)}")
    print(f"  Documents processed: {documents_processed}")
    print(f"  Output directory: {output_dir}")
    
    return all_shard_paths


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
        help="Number of Parquet shards to process (use a small number like 1-5 for testing)",
    )
    parser.add_argument(
        "--documents-per-shard",
        type=int,
        default=10_000,
        help="Documents per binary token shard",
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
        help="Documents to tokenize at once (for single-threaded fallback)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum token shards to generate (for testing)",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Starting Parquet shard index",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for tokenization",
    )
    
    args = parser.parse_args()
    
    generate_token_shards(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_parquet_shards=args.num_shards,
        documents_per_shard=args.documents_per_shard,
        context_length=args.context_length,
        batch_size=args.batch_size,
        max_token_shards=args.max_shards,
        start_parquet_shard=args.start_shard,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
