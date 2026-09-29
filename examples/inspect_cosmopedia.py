"""Inspect the Cosmopedia-v2 dataset.

This script demonstrates using the Parquet reader to explore the dataset.
It shows schema, metadata, and sample records without loading everything into memory.
"""

import sys

from mini_llm.data.parquet_reader import CosmopediaParquetReader


def main():
    print("=" * 80)
    print("Cosmopedia-v2 Dataset Inspection")
    print("=" * 80)
    
    # Initialize reader with default path
    try:
        reader = CosmopediaParquetReader()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is available at: ../cosmopedia-v2/cosmopedia-v2/")
        return 1
    
    print(f"\nResolved dataset directory:")
    print(f"  {reader.dataset_path}")
    
    print(f"\nDataset overview:")
    print(f"  Number of shards: {reader.num_shards}")
    print(f"  Total rows: {reader.total_rows:,}")
    
    print(f"\nAvailable columns ({len(reader.available_columns)}):")
    for col in reader.available_columns:
        print(f"  - {col}")
    
    print(f"\nSchema:")
    print(f"  {reader.schema}")
    
    # Peek at first few records
    print(f"\nFirst 3 records (sampled):")
    records = reader.peek_records(n=3)
    
    for i, record in enumerate(records, 1):
        print(f"\n--- Record {i} ---")
        
        # Get text length for display
        if "text" in record:
            text = record["text"]
            text_len = len(text)
            # Truncate for display
            truncated = text[:200] + "..." if len(text) > 200 else text
            print(f"Text (first 200 chars):")
            print(f"  {truncated}")
            print(f"Text length: {text_len} characters")
        
        # Show other fields
        for key, value in record.items():
            if key != "text":
                print(f"{key}: {value}")
    
    # Demonstrate batch iteration
    print(f"\n\nDemonstrating batch iteration:")
    reader_batched = CosmopediaParquetReader(batch_size=10)
    
    batch_count = 0
    total_records = 0
    for batch in reader_batched._iter_batches():
        batch_count += 1
        total_records += len(batch)
        if batch_count <= 3:
            print(f"  Batch {batch_count}: {len(batch)} records")
            # Show first record from batch
            pylist = batch.to_pylist()
            if pylist:
                rec = pylist[0]
                text_len = len(rec.get("text", ""))
                print(f"    First record: prompt_len={len(rec.get('prompt', ''))}, "
                      f"text_len={text_len}")
        if batch_count >= 5:
            break
    
    print(f"\n  ... (batch {batch_count} shown, iteration continues)")
    print(f"\nTotal batches read in sample: {batch_count}")
    print(f"Total records in sample: {total_records}")
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
