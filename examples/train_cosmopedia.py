"""Test training on actual Cosmopedia-v2 data with tokenizer."""

import time
from pathlib import Path

from mini_llm.config import ModelConfig
from mini_llm.data.parquet_reader import CosmopediaParquetReader
from mini_llm.data.token_shards import generate_token_shards
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train import MiniTrainer, load_token_shard


def train_on_cosmopedia_sample():
    """Train on a small sample of Cosmopedia data."""
    
    print("=" * 70)
    print("TRAINING ON COSMOPEDE-V2 (Sample)")
    print("=" * 70)
    
    # Dataset config
    dataset_path = "../cosmopedia-v2/cosmopedia-v2"
    output_dir = "cosmopedia_tokens_sample"
    num_shards_to_generate = 1  # Just 1 shard for quick test
    
    print(f"\n=== Step 1: Token Generation ===")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output:  {output_dir}")
    
    start_time = time.time()
    shard_paths = generate_token_shards(
        dataset_path=dataset_path,
        output_dir=output_dir,
        context_length=64,          # Short for quick test
        documents_per_shard=5000,   # Small for quick test
        max_shards=num_shards_to_generate,
    )
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Shards generated: {len(shard_paths)}")
    
    if not shard_paths:
        print("  Warning: No shards generated. Skipping training.")
        return
    
    # Verify token shards
    print(f"\n=== Step 2: Verify Token Shards ===")
    for path in shard_paths:
        data = load_token_shard(str(path))
        print(f"  {path.name}: {data.shape}")
    
    # Load tokenizer info
    tokenizer_path = Path(output_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
        print(f"  Vocabulary size: {len(tokenizer)}")
    
    # Create tiny model
    print(f"\n=== Step 3: Create Tiny Model ===")
    config = ModelConfig(
        vocab_size=16_384,          # Full vocab from tokenizer
        context_length=64,
        n_layers=1,
        d_model=32,                 # Small but reasonable
        n_q_heads=2,
        n_kv_heads=1,
        d_head=16,
        d_ff=128,
        n_experts=2,
        top_k=1,
    )
    
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    
    # Train
    print(f"\n=== Step 4: Training ===")
    trainer = MiniTrainer(
        model=model,
        shard_paths=[str(p) for p in shard_paths],
        batch_size=4,
        seq_length=32,
        warmup_steps=50,
        total_steps=100,
        peak_lr=1e-3,
        grad_clip=1.0,
    )
    
    print(f"  Batch size:       4")
    print(f"  Sequence length:  32")
    print(f"  Total steps:      100")
    
    losses = trainer.train(num_steps=100, log_interval=25)
    
    print(f"\n=== Results ===")
    print(f"  Initial loss:     {losses[0]:.4f}")
    print(f"  Final loss:       {losses[-1]:.4f}")
    print(f"  Loss reduction:   {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")


if __name__ == "__main__":
    import sys
    try:
        sys.exit(train_on_cosmopedia_sample())
    except Exception as e:
        print(f"\nError: {e}")
        print("This may be due to missing Cosmopedia dataset or tokenizer issues.")
        sys.exit(1)
