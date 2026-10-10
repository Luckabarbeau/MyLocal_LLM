"""Micro-training example with float16 to test numerical stability."""

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
        data = np.tile(
            np.arange(context_len, dtype=np.uint16) % 64,
            (num_docs, 1)
        )
        f.write(data.tobytes())


def check_finite(value, name: str):
    """Check if value is finite and return min/max for reporting."""
    if isinstance(value, np.ndarray):
        has_nan = np.any(np.isnan(value))
        has_inf = np.any(np.isinf(value))
        if has_nan or has_inf:
            return False, None, None
        return True, float(np.min(value)), float(np.max(value))
    return True, None, None


def main():
    print("=" * 70)
    print("MICRO TRAINING: Float16 Mode (with Mixed Precision)")
    print("=" * 70)
    
    # Create the smallest possible model for testing
    config = ModelConfig(
        vocab_size=64,
        context_length=64,
        n_layers=1,
        d_model=8,
        n_q_heads=1,
        n_kv_heads=1,
        d_head=8,
        d_ff=16,
        n_experts=2,
        top_k=1,
        init_std=0.1,
    )
    
    # Use float16 for model weights
    print(f"\n=== Float16 Configuration ===")
    print(f"  Model dtype:    float16 (weights)")
    print(f"  Model dim (d):  {config.d_model}")
    print(f"  Layers:         {config.n_layers}")
    
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float16")
    
    print(f"\n=== Micro Model Configuration ===")
    print(f"  Vocab size:     {config.vocab_size}")
    print(f"  Heads:          {config.n_q_heads}")
    print(f"  FFN dim:        {config.d_ff}")
    
    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    trainable_params = sum(p.data.size for p in model.parameters() if p.decay)
    print(f"\n=== Parameter Count ===")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Check dtype of model weights
    sample_param = model.embedding.W
    print(f"\n=== Weight Dtypes ===")
    print(f"  Embedding: {sample_param.data.dtype}")
    print(f"  Output proj: {model.output_proj.W.data.dtype}")
    
    # Create tiny synthetic shard
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "train.bin"
        create_micro_shard(shard_path, num_docs=10, context_len=64)
        
        print(f"\n=== Tiny Dataset ===")
        print(f"  Documents:        10")
        print(f"  Context length:   64")
        
        # Verify shard loading
        data = load_token_shard(str(shard_path))
        print(f"  Loaded shape:     {data.shape}")
        
        # Configure training for micro test with static loss scaling
        trainer = MiniTrainer(
            model=model,
            shard_paths=[str(shard_path)],
            batch_size=2,
            seq_length=32,
            warmup_steps=5,
            total_steps=20,
            peak_lr=1e-2,  # Higher LR now works with mixed precision
            grad_clip=1.0,
        )
        
        print(f"\n=== Training Configuration ===")
        print(f"  Batch size:       2")
        print(f"  Sequence length:  32")
        print(f"  Warmup steps:     5")
        print(f"  Total steps:      20")
        print(f"  Peak LR:          1e-2 (higher now works with mixed precision)")
        print(f"\n  Mixed Precision Implementation:")
        print(f"    - AdamW master weights: float32")
        print(f"    - AdamW moments (m, v): float32")
        print(f"    - RMSNorm reductions: float32")
        print(f"    - Cross entropy: float32")
        print(f"    - Gradient norm: float32")
        
        # Train
        print(f"\n=== Training ===")
        losses = trainer.train(num_steps=20, log_interval=5)
        
        print(f"\n=== Results ===")
        print(f"  Initial loss:     {losses[0]:.4f}")
        print(f"  Final loss:       {losses[-1]:.4f}")
        print(f"  Loss reduction:   {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")
        
        if losses[-1] < losses[0]:
            print(f"\n✓ Training successful with float16!")
        else:
            print(f"\n⚠ Warning: Loss did not decrease significantly")
        
        # Verify model weights are still finite
        all_finite = True
        for p in model.parameters():
            is_finite, min_val, max_val = check_finite(p.data, p.name)
            if not is_finite:
                print(f"  WARNING: {p.name} has NaN/Inf!")
                all_finite = False
            else:
                print(f"  {p.name}: finite range [{min_val:.4f}, {max_val:.4f}]")
        
        if all_finite:
            print(f"\n✓ All model weights remain finite after training")
    
    print("\n" + "=" * 70)
    print("FLOAT16 MICRO TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    sys.exit(main())
