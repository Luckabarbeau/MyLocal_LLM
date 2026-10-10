"""Example: Small-scale training with synthetic data.

This demonstrates training the model on synthetic tokenized data.
"""

import tempfile
from pathlib import Path

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train import MiniTrainer, load_token_shard


def create_synthetic_shard(path: Path, num_docs: int = 100, context_len: int = 256):
    """Create a synthetic token shard for testing."""
    with open(path, "wb") as f:
        f.write(np.int64(num_docs).tobytes())
        f.write(np.int64(context_len).tobytes())
        # Create some repetitive token patterns
        data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
        f.write(data.tobytes())


def main():
    print("=" * 60)
    print("Small-Scale Training Example")
    print("=" * 60)
    
    # Create a tiny model for testing
    config = ModelConfig.tiny_inspection()
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    
    print(f"\nModel configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    
    # Create synthetic shard
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "train.bin"
        create_synthetic_shard(shard_path, num_docs=100, context_len=256)
        
        print(f"\nCreated synthetic shard at {shard_path}")
        print(f"  Documents: 100")
        print(f"  Context length: 256")
        
        # Verify shard loading
        data = load_token_shard(str(shard_path))
        print(f"  Loaded shape: {data.shape}")
        
        # Create trainer
        trainer = MiniTrainer(
            model=model,
            shard_paths=[str(shard_path)],
            batch_size=4,
            seq_length=64,  # Shorter sequence for faster testing
            warmup_steps=100,
            total_steps=500,
            peak_lr=1e-3,
            grad_clip=1.0,
        )
        
        print(f"\nStarting training...")
        print(f"  Batch size: 4")
        print(f"  Sequence length: 64")
        print(f"  Total steps: 50")
        
        # Train for a few steps to demonstrate
        losses = trainer.train(num_steps=50, log_interval=10)
        
        print(f"\nFinal loss: {losses[-1]:.4f}")
        print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
