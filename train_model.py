#!/usr/bin/env python3
"""Extended training script for Cosmopedia-v2.

This script demonstrates extended training with:
- Model configuration selection (mini/small/medium)
- Training/validation split generation
- Gradient accumulation
- Checkpointing
- CSV logging

Usage:
    # Train mini model on first 10 Parquet shards
    python train_model.py \\
        --model mini \\
        --num-parquet-shards 10 \\
        --batch-size 16 \\
        --grad-accum-steps 2 \\
        --seq-length 512 \\
        --total-steps 10000 \\
        --checkpoint-dir ./checkpoints/mini_10shards \\
        --log-file ./logs/mini_10shards.csv

    # Resume from checkpoint
    python train_model.py \\
        --resume-from ./checkpoints/mini_10shards \\
        --total-steps 20000
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, validate_bfloat16_backend
from mini_llm.config import ModelConfig
from mini_llm.data.parquet_reader import CosmopediaParquetReader
from mini_llm.data.token_shards import TokenShardGenerator, load_token_shard
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a decoder-only language model on Cosmopedia-v2"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        choices=["micro", "mini", "small", "medium","large"],
        default="mini",
        help="Model size configuration",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "bf16-mixed", "float32"],
        default="float16",
        help=(
            "Training precision policy. bf16-mixed uses BF16 parameters/GEMMs "
            "with FP32 residuals, reductions, gradients, and optimizer state."
        ),
    )
    
    # Data configuration
    parser.add_argument(
        "--dataset-path",
        default="../cosmopedia-v2/cosmopedia-v2",
        help="Path to Cosmopedia Parquet directory",
    )
    parser.add_argument(
        "--num-parquet-shards",
        type=int,
        default=104,
        help="Number of Parquet shards to use",
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per step",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)",
    )
    
    # Training configuration
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total training steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps for learning rate",
    )
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW",
    )
    
    # Checkpointing and logging
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume training from checkpoint directory",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="CSV log file path",
    )
    
    # Validation
    parser.add_argument(
        "--val-interval",
        type=int,
        default=500,
        help="Steps between validation checks",
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=10,
        help="Number of validation steps per check",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2000,
        help="Steps between checkpoint saves",
    )
    
    # Validation split
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Ratio of data to use for validation",
    )
    
    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging",
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Numerical debugging
    parser.add_argument(
        "--numerical-debug",
        action="store_true",
        help="Enable numerical stability monitoring (checks for Inf/NaN at each tensor)",
    )
    
    # Tokenizer configuration
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help="Tokenizer vocabulary size (default: 512 for small models, 8192 for larger)",
    )
    parser.add_argument(
        "--tokenizer-backend",
        choices=["simple", "fast"],
        default="fast",
        help="Tokenizer backend (default: fast)",
    )
    
    return parser.parse_args()


def setup_model(config: ModelConfig, dtype: str = "float16") -> DecoderLanguageModel:
    """Create and initialize model."""
    print(f"Creating {config.d_model}d model with {config.n_layers} layers...")
    model = DecoderLanguageModel(config, rng_seed=42, dtype=dtype)
    
    # Count parameters
    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    trainable_params = sum(p.data.size for p in params if p.decay)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model dtype: {dtype}")
    
    return model


def generate_shards_for_training(
    dataset_path: str,
    output_dir: str,
    num_parquet_shards: int,
    documents_per_shard: int,
    context_length: int,
    val_ratio: float = 0.01,
) -> tuple:
    """
    Generate token shards and split into train/val.
    
    Args:
        dataset_path: Path to Parquet directory
        output_dir: Output directory for shards
        num_parquet_shards: Number of Parquet shards to process
        documents_per_shard: Documents per binary shard
        context_length: Maximum sequence length
        val_ratio: Ratio for validation split
        
    Returns:
        Tuple of (train_shard_paths, val_shard_paths, tokenizer_path)
    """
    from mini_llm.data.token_shards import (
        TokenShardGenerator,
        CosmopediaParquetReader,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create parquet reader
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    
    # Limit to requested Parquet shards
    if len(parquet_reader.shard_paths) > num_parquet_shards:
        parquet_reader.shard_paths = parquet_reader.shard_paths[:num_parquet_shards]
    print(f"Processing {len(parquet_reader.shard_paths)} Parquet shards")
    
    # Use a reasonable vocab size for the smoke test or small model
    # Large models should use pre-trained tokenizers
    if context_length <= 64:
        vocab_size = 512  # Small model needs smaller vocab
    else:
        vocab_size = 8192  # Reasonable default for most models

    print(f"Training tokenizer on sample (vocab_size={vocab_size}, backend=fast)...")
    sample_texts = []
    count = 0
    for record in parquet_reader.iter_records():
        sample_texts.append(record.get("text", ""))
        count += 1
        if count >= 500:
            break

    # Use FastBPETokenizer for training (faster, production-ready)
    try:
        from mini_llm.tokenizer.tokenizer import FastBPETokenizer
        tokenizer = FastBPETokenizer(vocab_size=vocab_size, threads=8)
        tokenizer.train(sample_texts)
    except Exception as e:
        print(f"Fast tokenizer failed ({e}), falling back to SimpleBPETokenizer")
        from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
        tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
        tokenizer.train(sample_texts)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Generate all token shards
    generator = TokenShardGenerator(
        tokenizer=tokenizer,
        parquet_reader=parquet_reader,
        output_dir=str(output_dir),
        context_length=context_length,
        documents_per_shard=documents_per_shard,
        batch_size=100,
    )
    
    all_shards = generator.generate_shards()
    
    if not all_shards:
        raise RuntimeError("No token shards generated!")
    
    # Split into train/validation
    num_val = max(1, int(len(all_shards) * val_ratio))
    val_shards = all_shards[-num_val:]
    train_shards = all_shards[:-num_val]
    
    print()
    print("=" * 60)
    print(f"Data split:")
    print(f"  Training shards: {len(train_shards)}")
    print(f"  Validation shards: {len(val_shards)}")
    print(f"  Total shards: {len(all_shards)}")
    
    # Return tokenizer vocab size for model config update
    tokenizer_vocab_size = len(tokenizer)
    return train_shards, val_shards, tokenizer_path, tokenizer_vocab_size


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    xp.random.seed(args.seed)

    if args.precision == "bf16-mixed":
        validate_bfloat16_backend()
        model_dtype = "bfloat16"
    elif args.precision == "float16":
        model_dtype = "float16"
    else:
        model_dtype = "float32"
    
    print("=" * 60)
    print("Extended Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Precision: {args.precision}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Parquet shards: {args.num_parquet_shards}")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}x")
    print(f"Effective batch: {args.batch_size * args.grad_accum_steps}")
    print(f"Training steps: {args.total_steps}")
    print(f"Learning rate: {args.peak_lr}")
    print(f"Warmup: {args.warmup_steps} steps")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Documents per shard: {args.documents_per_shard}")
    print()
    
    # Load or create model
    if args.resume_from:
        # Resume from checkpoint
        checkpoint_path = Path(args.resume_from)
        # Load config
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            # Use args.model config
            if args.model == "micro":
                config = ModelConfig.micro_debug()
            elif args.model == "mini":
                config = ModelConfig.mini()
            elif args.model == "small":
                config = ModelConfig.small()
            elif args.model == "medium":
                config = ModelConfig.medium()
            else:
                config = ModelConfig.large()
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load model
        config = dataclasses.replace(config, dtype=model_dtype)
        model = setup_model(config, dtype=model_dtype)
        
        # Load checkpoint
        from mini_llm.checkpoint import load_checkpoint
        param_names = [p.name for p in model.parameters()]
        loaded_params, optimizer_state, training_state = load_checkpoint(
            checkpoint_path,
            param_names=param_names,
        )
        
        # Apply loaded parameters
        for p in model.parameters():
            if p.name in loaded_params:
                p.data[...] = loaded_params[p.name]
        
        # Store optimizer state for later restoration after trainer creation
        stored_optimizer_state = optimizer_state
        
        start_step = training_state.get("step", 0) if training_state else 0
        
        # Issue #16: Restore tokens_processed
        if training_state and "tokens_processed" in training_state:
            tokens_processed = training_state["tokens_processed"]
        elif training_state:
            tokens_processed = start_step * args.batch_size * args.context_length
        else:
            tokens_processed = 0
        
        # Issue #12: Restore RNG states
        if training_state and "train_rng_state" in training_state:
            # Note: xp is imported at module level
            # Note: We'll restore this after creating the trainer
            rng_states = {
                "train": training_state["train_rng_state"],
                "val": training_state["val_rng_state"],
            }
        else:
            rng_states = None
        
    else:
        # Create new model
        shard_dir = Path("./token_shards")
        stored_optimizer_state = None
        
        if args.model == "micro":
            config = ModelConfig.micro_debug()
        elif args.model == "mini":
            config = ModelConfig.mini()
        elif args.model == "small":
            config = ModelConfig.small()
        elif args.model == "medium":
            config = ModelConfig.medium()
        else:
            config = ModelConfig.large()
        
        # Load tokenizer to get actual vocab size (if existing shards exist)
        existing_tokenizer_path = shard_dir / "tokenizer.json"
        if existing_tokenizer_path.exists():
            from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer
            try:
                tokenizer = FastBPETokenizer.load(str(existing_tokenizer_path))
            except Exception:
                tokenizer = SimpleBPETokenizer.load(str(existing_tokenizer_path))
            
            # Update config vocab size before creating model
            config = dataclasses.replace(config, tokenizer_vocab_size=len(tokenizer))
            print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        config = dataclasses.replace(config, dtype=model_dtype)
        model = setup_model(config, dtype=model_dtype)
        start_step = 0
        tokens_processed = 0
        rng_states = None
        
        # Save config
        if args.checkpoint_dir:
            config_path = Path(args.checkpoint_dir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(vars(config), f, indent=2)
    
    # Adjust context length in config if needed
    if config.context_length != args.context_length:
        print(f"Info: Context length set to {args.context_length} for this run (model config: {config.context_length})")
    
    # Generate or use existing token shards
    shard_dir = Path("./token_shards")
    
    if not any(shard_dir.glob("shard_*.bin")):
        print("\nGenerating token shards...")
        train_shards, val_shards, _, tokenizer_vocab_size = generate_shards_for_training(
            dataset_path=args.dataset_path,
            output_dir=str(shard_dir),
            num_parquet_shards=args.num_parquet_shards,
            documents_per_shard=args.documents_per_shard,
            context_length=args.context_length,
            val_ratio=args.val_ratio,
        )
        
        # Update config vocab size using dataclasses.replace for frozen dataclass
        config = dataclasses.replace(config, tokenizer_vocab_size=tokenizer_vocab_size)
        print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    else:
        # Use existing shards
        all_shards = sorted(shard_dir.glob("shard_*.bin"))
        num_val = max(1, int(len(all_shards) * args.val_ratio))
        val_shards = all_shards[-num_val:]
        train_shards = all_shards[:-num_val]
        
        print(f"Found existing shards:")
        print(f"  Training: {len(train_shards)}")
        print(f"  Validation: {len(val_shards)}")
        
        # Load tokenizer from shards directory to get actual vocab size
        existing_tokenizer_path = shard_dir / "tokenizer.json"
        if existing_tokenizer_path.exists():
            from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer
            try:
                tokenizer = FastBPETokenizer.load(str(existing_tokenizer_path))
            except Exception:
                tokenizer = SimpleBPETokenizer.load(str(existing_tokenizer_path))
            
            # Update config vocab size using dataclasses.replace for frozen dataclass
            config = dataclasses.replace(config, tokenizer_vocab_size=len(tokenizer))
            print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Create trainer
    trainer = ExtendedTrainer(
        model=model,
        train_shard_paths=[str(p) for p in train_shards],
        val_shard_paths=[str(p) for p in val_shards],
        batch_size=args.batch_size,
        seq_length=args.context_length,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.peak_lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        val_interval=args.val_interval,
        val_steps=args.val_steps,
        save_interval=args.save_interval,
        loss_scale=1.0,  # Can increase for mixed precision
        numerical_debug=args.numerical_debug,
    )
    
    # Issue #12: Restore trainer state including RNGs and tokens_processed
    trainer.step = start_step
    trainer.tokens_processed = tokens_processed
    
    # Issue #14: Restore optimizer state after trainer creation
    if stored_optimizer_state is not None and "m" in stored_optimizer_state:
        trainer.optimizer.step_index = int(
            stored_optimizer_state.get("step", start_step)
        )
    
        m_dict = stored_optimizer_state["m"]
        v_dict = stored_optimizer_state.get("v", {})
        
        # Build a mapping from parameter name to optimizer index
        param_to_idx = {p.name: i for i, p in enumerate(trainer.model.parameters())}
        
        restored_count = 0
        for p in trainer.model.parameters():
            if p.name in m_dict and p.name in v_dict:
                idx = param_to_idx[p.name]
                m_arr = xp.asarray(m_dict[p.name])
                v_arr = xp.asarray(v_dict[p.name])
                
                if m_arr.shape == trainer.optimizer.m[idx].shape:
                    trainer.optimizer.m[idx][...] = m_arr
                    trainer.optimizer.v[idx][...] = v_arr
                    restored_count += 1
                else:
                    print(f"WARNING: Shape mismatch for {p.name}: stored={m_arr.shape}, current={trainer.optimizer.m[idx].shape}")
        
        print(f"Restored optimizer state for {restored_count} parameters")
        
        # Check for missing parameters in optimizer state
        missing = [p.name for p in trainer.model.parameters() if p.name not in m_dict]
        if missing:
            print(f"WARNING: {len(missing)} parameters not found in saved optimizer state: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Issue #12: Restore RNG states after trainer creation
    if rng_states is not None:
        trainer.train_rng.bit_generator.state = rng_states["train"]
        trainer.val_rng.bit_generator.state = rng_states["val"]
    
    print()
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Train
    losses = trainer.train(
        num_steps=args.total_steps - start_step,
        log_interval=args.log_interval,
    )
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    
    if args.checkpoint_dir:
        print(f"Final checkpoint saved to: {Path(args.checkpoint_dir) / 'final'}")
        # Save final checkpoint
        trainer.save()


if __name__ == "__main__":
    main()
