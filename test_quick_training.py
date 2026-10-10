"""Quick test script to verify training works end-to-end."""

import os
import sys

# Set backend (change to 'cupy' for GPU)
os.environ["MINI_LLM_BACKEND"] = "numpy"

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.data.token_shards import load_token_shard

import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("Quick Training Test")
    print("=" * 60)
    
    # Use quickstart shards for quick test
    shard_path = "quickstart_shards/shard_00000.bin"
    
    if not Path(shard_path).exists():
        print(f"Error: Shard not found at {shard_path}")
        return
    
    # Load shard data
    shard_data = load_token_shard(shard_path)
    print(f"Loaded shard: {shard_data.shape}")
    
    # Create model
    config = ModelConfig.mini()
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float16")
    
    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    print(f"Model: {total_params:,} parameters, dtype=float16")
    
    # Create trainer with just one shard for quick test
    trainer = ExtendedTrainer(
        model=model,
        train_shard_paths=[shard_path],
        val_shard_paths=[shard_path],  # Use same for quick test
        batch_size=4,
        seq_length=512,
        grad_accum_steps=1,
        total_steps=100,
        warmup_steps=10,
        peak_lr=3e-4,
        grad_clip=1.0,
        weight_decay=0.1,
        checkpoint_dir="./checkpoints/quicktest",
        log_file="./logs/quicktest.csv",
        val_interval=50,
        val_steps=2,
        save_interval=50,
        loss_scale=1.0,
        rng_seed=42,
    )
    
    print()
    print("Starting training...")
    print(f"  Steps: 100")
    print(f"  Batch size: 4")
    print(f"  Sequence length: 512")
    print(f"  Backend: {trainer.model.dtype}")
    print()
    
    # Train
    losses = trainer.train(num_steps=100, log_interval=10)
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Avg loss: {np.mean(losses):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
