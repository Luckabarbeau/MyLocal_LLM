"""Quick test of Cosmopedia data pipeline without full training."""

import time
from pathlib import Path

from mini_llm.data.parquet_reader import CosmopediaParquetReader


def test_cosmopedia_pipeline():
    """Test the data pipeline with limited sampling."""
    
    print("=" * 70)
    print("COSMOPEDE-V2 PIPELINE TEST (Limited)")
    print("=" * 70)
    
    dataset_path = "../pretraining_data/cosmopedia-v2"
    
    print(f"\n=== Step 1: Load Parquet Reader ===")
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    print(f"  Dataset path: {parquet_reader.dataset_path}")
    print(f"  Shards:       {parquet_reader.num_shards}")
    print(f"  Total rows:   {parquet_reader.total_rows:,}")
    
    print(f"\n=== Step 2: Inspect Schema ===")
    print(f"  Schema:       {parquet_reader.schema}")
    
    print(f"\n=== Step 3: Peek at Records (limited to 5) ===")
    records = parquet_reader.peek_records(n=5)
    for i, record in enumerate(records):
        text_len = len(record.get("text", ""))
        prompt_len = len(record.get("prompt", ""))
        print(f"  Record {i+1}: prompt={prompt_len} chars, text={text_len} chars")
    
    print(f"\n=== Step 4: Iterate and Tokenize (limited) ===")
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    
    tokenizer = SimpleBPETokenizer(vocab_size=1000)
    
    # Train on just the peeked records
    sample_texts = [r.get("text", "") for r in records]
    tokenizer.train(sample_texts)
    print(f"  Vocabulary size: {len(tokenizer)}")
    
    # Tokenize a few more documents
    count = 0
    total_tokens = 0
    for record in parquet_reader.iter_records():
        text = record.get("text", "")
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        count += 1
        if count >= 10:
            break
    
    print(f"  Documents processed: {count}")
    print(f"  Total tokens: {total_tokens}")
    
    print(f"\n=== Step 5: Batch Iteration Test ===")
    batch_size = 3
    batch_count = 0
    for batch in parquet_reader._iter_batches():
        batch_count += 1
        print(f"  Batch {batch_count}: {len(batch)} records")
        if batch_count >= 3:
            break
    
    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    try:
        sys.exit(test_cosmopedia_pipeline())
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
