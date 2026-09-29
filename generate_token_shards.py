#!/usr/bin/env python3
"""Generate token shards from Cosmopedia-v2 dataset using multiprocessing.

This script loads a pre-trained tokenizer and encodes documents in parallel
across multiple CPU processes to maximize throughput.

Usage:
    # Generate token shards using 8 workers
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --tokenizer ./tokenizer.json \
        --num-shards 5 \
        --documents-per-shard 1000 \
        --num-workers 8

    # Generate with single worker (for comparison/debugging)
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --tokenizer ./tokenizer.json \
        --num-shards 5 \
        --documents-per-shard 1000 \
        --num-workers 1

    # Generate all shards from full dataset
    python generate_token_shards.py \
        --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
        --tokenizer ./tokenizer.json \
        --num-shards 104 \
        --documents-per-shard 10000 \
        --num-workers 8
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer

# Global tokenizer for worker processes - DO NOT use at module level in main process
_worker_tokenizer = None


def init_worker(tokenizer_path: str, context_length: int):
    """Initialize worker process with loaded tokenizer."""
    global _worker_tokenizer
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    
    _worker_tokenizer = SimpleBPETokenizer.load(tokenizer_path)
    _worker_tokenizer.context_length = context_length


def encode_batch(batch: tuple) -> list:
    """Encode a batch of documents using the global tokenizer.
    
    Must be called from a worker process initialized with init_worker().
    Returns list of (doc_index, token_ids) tuples.
    """
    if _worker_tokenizer is None:
        raise RuntimeError("Tokenizer not initialized in worker process")
    
    doc_indices, texts = batch
    results = []
    
    for idx, text in zip(doc_indices, texts):
        if not text:
            continue
        
        # Encode
        ids = _worker_tokenizer.encode(text)
        
        # Truncate to context length
        if len(ids) > _worker_tokenizer.context_length:
            ids = ids[:_worker_tokenizer.context_length]
        
        results.append((idx, ids))
    
    return results


def generate_token_shards(
    dataset_path: str,
    output_dir: str,
    tokenizer_path: str,
    num_parquet_shards: int = 104,
    documents_per_shard: int = 10_000,
    context_length: int = 512,
    max_token_shards: int = None,
    start_parquet_shard: int = 0,
    num_workers: int = 8,
    chunksize: int = 32,
) -> list:
    """
    Generate token shards from Cosmopedia Parquet data using multiprocessing.
    
    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_dir: Output directory for binary shards
        tokenizer_path: Path to pre-trained tokenizer JSON
        num_parquet_shards: Number of Parquet shards to process
        documents_per_shard: Target documents per binary shard
        context_length: Maximum sequence length
        max_token_shards: Maximum token shards to generate (None = all)
        start_parquet_shard: Starting Parquet shard index
        num_workers: Number of parallel worker processes
        chunksize: Batch size for multiprocessing
        
    Returns:
        List of paths to generated token shards
    """
    from mini_llm.data.parquet_reader import CosmopediaParquetReader
    from mini_llm.data.token_shards import TokenShardWriter
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating token shards from {num_parquet_shards} Parquet shards")
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Workers: {num_workers}")
    print(f"Encoding backend: multiprocessing")
    print(f"Context length: {context_length}")
    print(f"Documents per shard: {documents_per_shard}")
    print(f"Chunksize: {chunksize}")
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
    
    # Tokenize and write shards using multiprocessing
    shard_writer = TokenShardWriter(
        str(output_dir),
        context_length=context_length,
        dtype="uint16"
    )
    
    all_shard_paths = []
    current_batch_docs: list = []  # (doc_id, token_ids)
    documents_processed = 0
    token_shards_written = 0
    total_tokens = 0
    
    print("Processing Parquet shards with multiprocessing...")
    for parquet_idx, shard_path in enumerate(parquet_reader.shard_paths):
        print(f"  Processing Parquet shard {parquet_idx + 1}/{len(parquet_reader.shard_paths)}: {shard_path.name}")
        
        # Read this Parquet shard
        table = parquet_reader._get_table_for_shard(shard_path)
        
        # Collect documents for batch processing
        doc_texts = []
        for record in table.to_pylist():
            text = record.get("text", "")
            doc_texts.append(text)
        
        documents_in_shard = len(doc_texts)
        
        # Encode in parallel using worker pool
        if num_workers > 1:
            print(f"    Encoding {documents_in_shard} documents with {num_workers} workers...")
            start_batch_time = time.time()
            
            # Calculate optimal batch size for reducing IPC overhead
            # Target: each worker gets ~4-8 batches total
            # For fast tokenization, use larger batches to amortize IPC overhead
            target_batches = num_workers * 2  # Fewer, larger batches
            batch_size = max(128, documents_in_shard // target_batches)
            
            doc_indices = list(range(documents_in_shard))
            batches = []
            for i in range(0, documents_in_shard, batch_size):
                end = min(i + batch_size, documents_in_shard)
                batches.append((doc_indices[i:end], doc_texts[i:end]))
            
            print(f"    Batching: {len(batches)} batches of ~{batch_size} docs each")
            task_chunksize = max(1, len(batches) // (num_workers * 2))
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=(tokenizer_path, context_length),
            ) as pool:
                # Use map with chunksize to reduce IPC overhead
                task_chunksize = max(1, len(batches) // (num_workers * 2))
                batch_results = list(pool.map(encode_batch, batches, chunksize=task_chunksize))
                print(f"    Task chunksize used: {task_chunksize}")
            
            batch_elapsed = time.time() - start_batch_time
            
            # Collect and flatten results
            for batch_result in batch_results:
                for doc_id, token_ids in batch_result:
                    if token_ids:
                        current_batch_docs.append((doc_id, token_ids))
                        total_tokens += len(token_ids)
            
            docs_per_sec = documents_in_shard / batch_elapsed if batch_elapsed > 0 else 0
            tokens_per_sec = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
            print(f"    Encoding took {batch_elapsed:.1f}s ({docs_per_sec:.0f} docs/s, {tokens_per_sec:.0f} tok/s)")
        else:
            # Single-threaded fallback for debugging
            tokenizer = SimpleBPETokenizer.load(tokenizer_path)
            tokenizer.context_length = context_length
            
            start_batch_time = time.time()
            
            for doc_id, text in enumerate(doc_texts):
                if text:
                    ids = tokenizer.encode(text)
                    if len(ids) > context_length:
                        ids = ids[:context_length]
                    if ids:
                        current_batch_docs.append((doc_id, ids))
                        total_tokens += len(ids)
            
            batch_elapsed = time.time() - start_batch_time
            docs_per_sec = documents_in_shard / batch_elapsed if batch_elapsed > 0 else 0
            print(f"    Encoding took {batch_elapsed:.1f}s ({docs_per_sec:.0f} docs/s)")
        
        documents_processed += documents_in_shard
        
        # Flush documents to token shards
        while len(current_batch_docs) >= documents_per_shard:
            shard_docs = [doc[1] for doc in current_batch_docs[:documents_per_shard]]
            current_batch_docs = current_batch_docs[documents_per_shard:]
            
            shard_path_out = shard_writer.write_shard(token_shards_written, shard_docs)
            if shard_path_out:
                all_shard_paths.append(shard_path_out)
                token_shards_written += 1
                print(f"    Written token shard {token_shards_written}: {shard_path_out.name}")
            
            if max_token_shards and token_shards_written >= max_token_shards:
                break
        
        if max_token_shards and token_shards_written >= max_token_shards:
            break
    
    # Flush remaining documents
    if current_batch_docs and (not max_token_shards or token_shards_written < max_token_shards):
        shard_docs = [doc[1] for doc in current_batch_docs]
        shard_path_out = shard_writer.write_shard(token_shards_written, shard_docs)
        if shard_path_out:
            all_shard_paths.append(shard_path_out)
            token_shards_written += 1
            print(f"    Written final token shard {token_shards_written}: {shard_path_out.name}")
    
    total_elapsed = time.time() - parquet_idx_start if 'parquet_idx_start' in locals() else time.time() - start_time
    
    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Token shards written: {len(all_shard_paths)}")
    print(f"Documents processed: {documents_processed:,}")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Output directory: {output_dir}")
    
    if documents_processed > 0:
        avg_tokens_per_doc = total_tokens / documents_processed
        print(f"Average tokens per document: {avg_tokens_per_doc:.1f}")
    
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
        "--tokenizer",
        required=True,
        help="Path to pre-trained tokenizer JSON file",
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
        default=8,
        help="Number of parallel worker processes for tokenization",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Chunk size for multiprocessing (documents per batch per worker)",
    )
    
    args = parser.parse_args()
    
    # Validate tokenizer exists
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        print("Run train_tokenizer.py first to create a tokenizer.")
        return 1
    
    generate_token_shards(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        tokenizer_path=str(tokenizer_path),
        num_parquet_shards=args.num_shards,
        documents_per_shard=args.documents_per_shard,
        context_length=args.context_length,
        max_token_shards=args.max_shards,
        start_parquet_shard=args.start_shard,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
