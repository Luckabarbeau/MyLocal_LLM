"""Generates a small, dummy token shard dataset for testing the PyTorch backend
without requiring the full Cosmopedia Parquet corpus."""

import argparse
import os
import numpy as np
from pathlib import Path


def generate_dummy_shards(output_dir, num_shards, seq_length, vocab_size, num_docs_per_shard):
    """Generates .bin files following the packed token stream format."""
    print(f"Generating {num_shards} dummy shards in {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_shards):
        shard_path = Path(output_dir) / f"train_shard_{i:04d}.bin"
        
        # A packed shard is a 1D array of token IDs.
        # Each document is: [tokens...], [EOS]
        # We'll generate num_docs_per_shard documents.
        
        all_tokens = []
        for _ in range(num_docs_per_shard):
            # Random length between seq_length/2 and seq_length
            doc_len = np.random.randint(seq_length // 2, seq_length)
            tokens = np.random.randint(0, vocab_size, size=doc_len, dtype=np.uint16)
            # Use a fixed EOS token (e.g., 0 or vocab_size-1). Let's say 0.
            # But we need to ensure 0 isn't used for content, so let's use 1.
            # Actually, let's just use uint16 and assume tokens < vocab_size.
            # Let's use 0 as EOS and ensure tokens start from 1.
            tokens = np.random.randint(1, vocab_size, size=doc_len, dtype=np.uint16)
            all_tokens.append(tokens)
            all_tokens.append(np.array([0], dtype=np.uint16)) # EOS

        packed_tokens = np.concatenate(all_tokens).astype(np.uint16)
        packed_tokens.tofile(shard_path)
        print(f"  Created shard {i} ({len(packed_tokens)} tokens)")

    print("Dummy dataset generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./token_shards")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--docs-per-shard", type=int, default=10)
    args = parser.parse_args()

    generate_dummy_shards(
        args.output_dir, 
        args.num_shards, 
        args.seq_length, 
        args.vocab_size, 
        args.docs_per_shard
    )
