"""Minimal micro-training example with tiny model and small dataset."""

import tempfile
from pathlib import Path

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train import MiniTrainer, load_token_shard


def create_micro_shard(path: Path, num_docs: int = 10, context_len: int = 64):
    """Create a tiny micro-shard for ultra-fast testing."""
    with open(path, "wb") as f:
        f.write(np.int64(num_docs).tobytes())
        f.write(np.int64(context_len).tobytes())
        # Create repetitive patterns to make learning easy
        # This helps verify the training loop works
        data = np.tile(
            np.arange(context_len, dtype=np.uint16) % 64,
            (num_docs, 1)
        )
        f.write(data.tobytes())


def main():
    print("=" * 70)
    print("MICRO TRAINING: Tiny Model + Tiny Dataset")
    print("=" * 70)
    
    # Create the smallest possible model for testing
    config = ModelConfig(
        vocab_size=64,          # Very small vocab for micro test
        context_length=64,
        n_layers=1,             # Single layer
        d_model=8,              # Tiny embedding dimension
        n_q_heads=1,            # Single head
        n_kv_heads=1,
        d_head=8,
        d_ff=16,                # Tiny FFN
        n_experts=2,            # Minimal MoE
        top_k=1,
        init_std=0.1,           # Larger init for faster initial learning
    )
    
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    
    print(f"\n=== Micro Model Configuration ===")
    print(f"  Vocab size:      {config.vocab_size}")
    print(f"  Model dim (d):   {config.d_model}")
    print(f"  Layers:          {config.n_layers}")
    print(f"  Heads:           {config.n_q_heads}")
    print(f"  FFN dim:         {config.d_ff}")
    print(f"  MoE experts:     {config.n_experts}")
    print(f"  Context length:  {config.context_length}")
    
    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    trainable_params = sum(p.data.size for p in model.parameters() if p.decay)
    print(f"\n=== Parameter Count ===")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Create tiny synthetic shard
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "train.bin"
        create_micro_shard(shard_path, num_docs=10, context_len=64)
        
        print(f"\n=== Tiny Dataset ===")
        print(f"  Shard path:       {shard_path}")
        print(f"  Documents:        10")
        print(f"  Context length:   64")
        
        # Verify shard loading
        data = load_token_shard(str(shard_path))
        print(f"  Loaded shape:     {data.shape}")
        print(f"  Data range:       [{data.min()}, {data.max()}]")
        
        # Configure training for micro test
        trainer = MiniTrainer(
            model=model,
            shard_paths=[str(shard_path)],
            batch_size=2,           # Tiny batch
            seq_length=32,          # Half context for micro test
            warmup_steps=5,         # Very short warmup
            total_steps=20,         # Just a few steps
            peak_lr=1e-2,           # Higher LR for fast initial learning
            grad_clip=1.0,
        )
        
        print(f"\n=== Training Configuration ===")
        print(f"  Batch size:       2")
        print(f"  Sequence length:  32")
        print(f"  Warmup steps:     5")
        print(f"  Total steps:      20")
        print(f"  Peak LR:          1e-2")
        
        # Train
        print(f"\n=== Training ===")
        losses = trainer.train(num_steps=20, log_interval=5)
        
        print(f"\n=== Results ===")
        print(f"  Initial loss:     {losses[0]:.4f}")
        print(f"  Final loss:       {losses[-1]:.4f}")
        print(f"  Loss reduction:   {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")
        
        if losses[-1] < losses[0]:
            print(f"\n✓ Training successful - loss decreased!")
        else:
            print(f"\n⚠ Warning: Loss did not decrease (may need more steps or higher LR)")
    
    print("\n" + "=" * 70)
    print("MICRO TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    sys.exit(main())
