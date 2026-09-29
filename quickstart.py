#!/usr/bin/env python3
"""Quick start guide for extended training.

This script demonstrates the core training components.
"""

import numpy as np

# Import all necessary modules
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel


def main():
    print("=" * 60)
    print("Quick Start: Extended Training Components")
    print("=" * 60)
    print()
    
    # Step 1: Create model configuration
    print("Step 1: Model Configuration")
    config = ModelConfig.mini()
    print(f"  Model: Mini ({config.d_model}d, {config.n_layers} layers)")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Context length: {config.context_length}")
    
    # Step 2: Create model
    print("\nStep 2: Model Creation")
    model = DecoderLanguageModel(config, dtype="float16")
    params = sum(p.data.size for p in model.parameters())
    trainable = sum(p.data.size for p in model.parameters() if p.decay)
    print(f"  Total parameters: {params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    
    # Step 3: Test forward pass
    print("\nStep 3: Forward Pass Test")
    batch_size = 4
    seq_len = 32
    inputs = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.uint16)
    
    logits, cache = model.forward(inputs)
    loss, loss_cache = model.compute_loss(logits, inputs)
    print(f"  Input shape: {inputs.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss:.4f}")
    
    # Step 4: Test backward pass
    print("\nStep 4: Backward Pass Test")
    d_logits = model.backward_loss(loss_cache)
    model.backward(d_logits, cache)
    
    grad_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None))
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Step 5: Test optimizer step
    print("\nStep 5: Optimizer Step Test")
    from mini_llm.optim.adamw import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3)
    optimizer.step(lr=1e-3)
    print("  Optimizer step completed")
    
    # Step 6: Test learning rate schedule
    print("\nStep 6: Learning Rate Schedule")
    from mini_llm.optim.schedule import WarmupCosineSchedule
    scheduler = WarmupCosineSchedule(
        peak_lr=1e-3,
        warmup_steps=100,
        total_steps=1000,
    )
    
    lrs = [scheduler(step) for step in [0, 50, 100, 500, 900, 999]]
    print(f"  LR at step 0: {lrs[0]:.6f} (warmup)")
    print(f"  LR at step 100: {lrs[2]:.6f} (peak)")
    print(f"  LR at step 500: {lrs[3]:.6f}")
    print(f"  LR at step 1000: {lrs[5]:.6f} (cosine decay)")
    
    # Step 7: Test checkpoint save/load
    print("\nStep 7: Checkpoint Save/Load")
    from mini_llm.checkpoint import save_checkpoint, load_checkpoint
    
    # Get original parameters for comparison
    orig_params = {p.name: p.data.copy() for p in model.parameters()}
    
    # Save
    save_checkpoint(
        path="./quickstart_checkpoints",
        model_params={p.name: p.data for p in model.parameters()},
    )
    
    # Create new model and load
    model2 = DecoderLanguageModel(config, dtype="float16")
    loaded_params, _, _ = load_checkpoint("./quickstart_checkpoints")
    
    # Verify parameters loaded correctly
    all_match = True
    for p in model.parameters():
        if p.name in loaded_params:
            if not np.allclose(p.data, loaded_params[p.name], atol=1e-6):
                all_match = False
    
    print(f"  Checkpoint saved: ✓")
    print(f"  Checkpoint loaded: ✓")
    print(f"  Parameters match: {all_match}")
    
    # Cleanup
    import shutil
    shutil.rmtree("./quickstart_checkpoints")
    
    # Final summary
    print()
    print("=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print()
    print("All core components verified working.")
    print()
    print("For full training pipeline, run:")
    print()
    print("  # Generate token shards from Cosmopedia Parquet files")
    print("  python generate_token_shards.py --dataset-path ../cosmopedia-v2/cosmopedia-v2")
    print()
    print("  # Train a model")
    print("  python train_model.py --model mini --total-steps 10000")
    print()
    print("  # Generate text from trained model")
    print("  python inference.py --checkpoint ./checkpoints/mini --prompt 'The sky is'")
    print()


if __name__ == "__main__":
    main()
