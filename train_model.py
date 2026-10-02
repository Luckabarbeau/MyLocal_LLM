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
        choices=["micro", "mini", "small", "medium", "medium-context-4k", "large"],
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
        "--shard-dir",
        default="./token_shards",
        help=(
            "Directory containing token shards. Packed preprocessing uses "
            "train_shard_*.bin / val_shard_*.bin."
        ),
    )
    parser.add_argument(
        "--mixed-shard-root",
        default=None,
        help=(
            "Root produced by generate_mixed_pretraining_shards.py. When set, "
            "training samples corpora using mixture_manifest.json weights instead "
            "of treating all shards as one uniform pool."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Tokenizer JSON used to create the shards. Defaults to "
            "<shard-dir>/tokenizer.json when present."
        ),
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
        default=None,
        help="Training sequence length (default: selected model preset)",
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


def model_config_from_name(name: str) -> ModelConfig:
    if name == "micro":
        return ModelConfig.micro_debug()
    if name == "mini":
        return ModelConfig.mini()
    if name == "small":
        return ModelConfig.small()
    if name == "medium":
        return ModelConfig.medium()
    if name == "medium-context-4k":
        return ModelConfig.medium_context_4k()
    if name == "large":
        return ModelConfig.large()
    raise ValueError(f"unknown model preset: {name}")


def load_tokenizer_file(path: Path):
    try:
        return FastBPETokenizer.load(str(path))
    except Exception:
        return SimpleBPETokenizer.load(str(path))


def discover_existing_shards(shard_dir: Path, val_ratio: float):
    """Discover canonical packed shards first, then legacy shard names."""
    train = sorted(shard_dir.glob("train_shard_*.bin"))
    val = sorted(shard_dir.glob("val_shard_*.bin"))
    if train:
        if not val:
            raise RuntimeError(
                f"found packed training shards in {shard_dir} but no val_shard_*.bin"
            )
        return train, val

    all_shards = sorted(shard_dir.glob("shard_*.bin"))
    if not all_shards:
        return [], []
    num_val = max(1, int(len(all_shards) * val_ratio))
    if len(all_shards) <= num_val:
        raise RuntimeError("not enough legacy shards to create train/validation sets")
    return all_shards[:-num_val], all_shards[-num_val:]


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

    np.random.seed(args.seed)
    xp.random.seed(args.seed)

    if args.precision == "bf16-mixed":
        validate_bfloat16_backend()
        model_dtype = "bfloat16"
    elif args.precision == "float16":
        model_dtype = "float16"
    else:
        model_dtype = "float32"

    mixed_shard_root = (
        Path(args.mixed_shard_root).expanduser().resolve()
        if args.mixed_shard_root is not None
        else None
    )
    shard_dir = Path(args.shard_dir)
    if mixed_shard_root is None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        default_tokenizer_path = shard_dir / "tokenizer.json"
    else:
        default_tokenizer_path = mixed_shard_root / "tokenizer.json"
    tokenizer_path = (
        Path(args.tokenizer_path)
        if args.tokenizer_path is not None
        else default_tokenizer_path
    )

    # Resolve the architecture first so its native context length can be used
    # when --context-length is omitted.
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = ModelConfig(**json.load(f))
        else:
            config = model_config_from_name(args.model)
    else:
        checkpoint_path = None
        config = model_config_from_name(args.model)

    run_context_length = (
        int(args.context_length)
        if args.context_length is not None
        else int(config.context_length)
    )

    # Prefer context-independent packed shards.  Mixed pretraining keeps each
    # source physically separate and samples source names according to explicit
    # weights in mixture_manifest.json.
    mixed_sources = None
    mixed_manifest = None
    if mixed_shard_root is not None:
        from mini_llm.data.mixed_shards import load_weighted_shard_sources

        mixed_sources, mixed_manifest = load_weighted_shard_sources(
            mixed_shard_root
        )
        train_shards = [
            path for source in mixed_sources for path in source.train_paths
        ]
        val_shards = [
            path for source in mixed_sources for path in source.val_paths
        ]
        tokenizer_vocab_size = int(mixed_manifest["vocab_size"])
    else:
        train_shards, val_shards = discover_existing_shards(
            shard_dir, args.val_ratio
        )
        if not train_shards:
            if args.model == "medium-context-4k":
                raise RuntimeError(
                    "medium-context-4k requires pre-generated packed shards. "
                    "Run generate_packed_cosmopedia_shards.py or use "
                    "--mixed-shard-root."
                )
            print("\nGenerating legacy token shards...")
            train_shards, val_shards, generated_tokenizer, tokenizer_vocab_size = (
                generate_shards_for_training(
                    dataset_path=args.dataset_path,
                    output_dir=str(shard_dir),
                    num_parquet_shards=args.num_parquet_shards,
                    documents_per_shard=args.documents_per_shard,
                    context_length=run_context_length,
                    val_ratio=args.val_ratio,
                )
            )
            tokenizer_path = Path(generated_tokenizer)
        else:
            tokenizer_vocab_size = None

    if tokenizer_path.exists():
        tokenizer = load_tokenizer_file(tokenizer_path)
        loaded_vocab_size = len(tokenizer)
        if loaded_vocab_size <= 0:
            raise RuntimeError(f"tokenizer at {tokenizer_path} has empty vocabulary")
        if (
            tokenizer_vocab_size is not None
            and int(tokenizer_vocab_size) != int(loaded_vocab_size)
        ):
            raise RuntimeError(
                "tokenizer vocabulary does not match dataset manifest: "
                f"{loaded_vocab_size} != {tokenizer_vocab_size}"
            )
        tokenizer_vocab_size = loaded_vocab_size
    elif tokenizer_vocab_size is None:
        raise FileNotFoundError(
            f"tokenizer not found at {tokenizer_path}; pass --tokenizer-path explicitly"
        )

    config = dataclasses.replace(
        config,
        tokenizer_vocab_size=int(tokenizer_vocab_size),
        context_length=run_context_length,
        dtype=model_dtype,
    )

    print("=" * 60)
    print("Extended Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Precision: {args.precision}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Tokenizer vocab: {tokenizer_vocab_size:,}")
    if mixed_shard_root is not None:
        print(f"Mixed shard root: {mixed_shard_root}")
        print("Pretraining mixture:")
        for source in mixed_sources:
            print(
                f"  {source.name:16s} {100.0 * source.weight:6.2f}%  "
                f"{len(source.train_paths)} train / {len(source.val_paths)} val shards"
            )
    else:
        print(f"Shard directory: {shard_dir}")
    print(f"Training shards: {len(train_shards)}")
    print(f"Validation shards: {len(val_shards)}")
    print(f"Context length: {run_context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}x")
    print(f"Effective batch: {args.batch_size * args.grad_accum_steps}")
    print(f"Training steps: {args.total_steps}")
    print(f"Learning rate: {args.peak_lr}")
    print(f"Warmup: {args.warmup_steps} steps")
    print()

    model = setup_model(config, dtype=model_dtype)

    if args.resume_from:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        from mini_llm.checkpoint import load_checkpoint

        param_names = [p.name for p in model.parameters()]
        loaded_params, optimizer_state, training_state = load_checkpoint(
            checkpoint_path,
            param_names=param_names,
        )
        for p in model.parameters():
            if p.name in loaded_params:
                p.data[...] = loaded_params[p.name]

        stored_optimizer_state = optimizer_state
        start_step = training_state.get("step", 0) if training_state else 0
        if training_state and "tokens_processed" in training_state:
            tokens_processed = training_state["tokens_processed"]
        elif training_state:
            tokens_processed = (
                start_step * args.batch_size * run_context_length
            )
        else:
            tokens_processed = 0

        if training_state and "train_rng_state" in training_state:
            rng_states = {
                "train": training_state["train_rng_state"],
                "val": training_state["val_rng_state"],
            }
        else:
            rng_states = None
    else:
        stored_optimizer_state = None
        start_step = 0
        tokens_processed = 0
        rng_states = None

    if args.checkpoint_dir:
        config_path = Path(args.checkpoint_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

        # Keep the exact tokenizer next to checkpoints for reproducible decode
        # and later mixed-dataset training.
        checkpoint_tokenizer = Path(args.checkpoint_dir) / "tokenizer.json"
        if tokenizer_path.resolve() != checkpoint_tokenizer.resolve():
            import shutil
            shutil.copy2(tokenizer_path, checkpoint_tokenizer)
        if mixed_shard_root is not None:
            import shutil
            shutil.copy2(
                mixed_shard_root / "mixture_manifest.json",
                Path(args.checkpoint_dir) / "mixture_manifest.json",
            )
            mix_config_path = mixed_shard_root / "mix_config.json"
            if mix_config_path.exists():
                shutil.copy2(
                    mix_config_path, Path(args.checkpoint_dir) / "mix_config.json"
                )

    if mixed_sources is not None:
        from mini_llm.data.mixed_shards import (
            source_paths_by_split,
            source_weight_dict,
        )
        train_source_shards = source_paths_by_split(mixed_sources, "train")
        val_source_shards = source_paths_by_split(mixed_sources, "val")
        source_weights = source_weight_dict(mixed_sources)
    else:
        train_source_shards = None
        val_source_shards = None
        source_weights = None

    trainer = ExtendedTrainer(
        model=model,
        train_shard_paths=[str(p) for p in train_shards],
        val_shard_paths=[str(p) for p in val_shards],
        batch_size=args.batch_size,
        seq_length=run_context_length,
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
        loss_scale=1.0,
        numerical_debug=args.numerical_debug,
        train_source_shards=train_source_shards,
        val_source_shards=val_source_shards,
        source_weights=source_weights,
    )

    trainer.step = start_step
    trainer.tokens_processed = tokens_processed
    if args.resume_from and training_state:
        trainer.current_train_shard_idx = int(
            training_state.get("current_train_shard_idx", 0)
        )
        trainer.current_val_shard_idx = int(
            training_state.get("current_val_shard_idx", 0)
        )
        if trainer.train_source_shards is not None:
            for name, value in training_state.get(
                "current_train_source_shard_idx", {}
            ).items():
                if name in trainer.current_train_source_shard_idx:
                    trainer.current_train_source_shard_idx[name] = int(value)
            for name, value in training_state.get(
                "current_val_source_shard_idx", {}
            ).items():
                if name in trainer.current_val_source_shard_idx:
                    trainer.current_val_source_shard_idx[name] = int(value)
            for name, value in training_state.get(
                "train_source_batch_counts", {}
            ).items():
                if name in trainer.train_source_batch_counts:
                    trainer.train_source_batch_counts[name] = int(value)
            for name, value in training_state.get(
                "val_source_batch_counts", {}
            ).items():
                if name in trainer.val_source_batch_counts:
                    trainer.val_source_batch_counts[name] = int(value)

    if stored_optimizer_state is not None and "m" in stored_optimizer_state:
        trainer.optimizer.step_index = int(
            stored_optimizer_state.get("step", start_step)
        )
        m_dict = stored_optimizer_state["m"]
        v_dict = stored_optimizer_state.get("v", {})
        param_to_idx = {
            p.name: i for i, p in enumerate(trainer.model.parameters())
        }
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
                    print(
                        f"WARNING: Shape mismatch for {p.name}: "
                        f"stored={m_arr.shape}, current={trainer.optimizer.m[idx].shape}"
                    )
        print(f"Restored optimizer state for {restored_count} parameters")
        missing = [
            p.name for p in trainer.model.parameters() if p.name not in m_dict
        ]
        if missing:
            suffix = "..." if len(missing) > 5 else ""
            print(
                f"WARNING: {len(missing)} parameters not found in saved optimizer "
                f"state: {missing[:5]}{suffix}"
            )

    if rng_states is not None:
        trainer.train_rng.bit_generator.state = rng_states["train"]
        trainer.val_rng.bit_generator.state = rng_states["val"]

    print()
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

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
        trainer.save()


if __name__ == "__main__":
    main()
