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
import os
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, validate_bfloat16_backend
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer
from mini_llm.runtime_defaults import apply_runtime_defaults, format_runtime_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a decoder-only language model on Cosmopedia-v2"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        choices=[
            "micro", "mini", "small", "medium", "medium-context-4k",
            "moe-525m-context-4k", "moe-1b-context-4k",
            "wide-500m-context-4k", "wide-500m-context-8k",
            "wide-500m-context-16k", "wide-500m-context-32k",
            "wide-500m-context-64k",
            "wide-500m-memory-16k", "wide-500m-memory-32k",
            "wide-500m-memory-64k", "large",
        ],
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
    parser.add_argument(
        "--runtime-profile",
        choices=["auto", "reference", "gpu-fast", "consumer-gpu"],
        default=None,
        help=(
            "Runtime defaults profile. Default: MINI_LLM_RUNTIME_PROFILE or auto. "
            "auto selects reference on NumPy, gpu-fast on CuPy, and consumer-gpu "
            "for large CuPy training models. Explicit MINI_LLM_* variables always win."
        ),
    )
    
    # Progressive-depth training (0064B)
    parser.add_argument(
        "--progressive-depth",
        action="store_true",
        help=(
            "Enable residual-gated progressive Transformer depth. New runs begin "
            "with --initial-active-layers blocks; inactive blocks are skipped "
            "entirely until activated."
        ),
    )
    parser.add_argument(
        "--initial-active-layers",
        type=int,
        default=None,
        help=(
            "Initial executed depth for a new --progressive-depth run "
            "(default: 1). Ignored on resume, where checkpoint state wins."
        ),
    )
    parser.add_argument(
        "--progressive-growth-steps",
        default=None,
        help=(
            "Comma-separated completed optimizer steps at which to activate one "
            "additional block before the following update, e.g. "
            "'1000,3000,7000'. On resume, the checkpointed schedule is reused "
            "when this option is omitted."
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
        "--init-from",
        default=None,
        help=(
            "Initialize matching model weights from a checkpoint but start a "
            "fresh optimizer/training run. Intended for loading a trained 4k "
            "backbone into a 0058C memory preset."
        ),
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
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help=(
            "Profile the first N executed optimizer steps with synchronized "
            "coarse GPU timings (default: 0/off). Use 1-3 for diagnosis."
        ),
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
    if name == "moe-525m-context-4k":
        return ModelConfig.moe_525m_context_4k()
    if name == "moe-1b-context-4k":
        return ModelConfig.moe_1b_context_4k()
    if name == "wide-500m-context-4k":
        return ModelConfig.wide_500m_context_4k()
    if name == "wide-500m-context-8k":
        return ModelConfig.wide_500m_context_8k()
    if name == "wide-500m-context-16k":
        return ModelConfig.wide_500m_context_16k()
    if name == "wide-500m-context-32k":
        return ModelConfig.wide_500m_context_32k()
    if name == "wide-500m-context-64k":
        return ModelConfig.wide_500m_context_64k()
    if name == "wide-500m-memory-16k":
        return ModelConfig.wide_500m_memory_16k()
    if name == "wide-500m-memory-32k":
        return ModelConfig.wide_500m_memory_32k()
    if name == "wide-500m-memory-64k":
        return ModelConfig.wide_500m_memory_64k()
    if name == "large":
        return ModelConfig.large()
    raise ValueError(f"unknown model preset: {name}")


def parse_progressive_growth_steps(raw):
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return []
    try:
        steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "--progressive-growth-steps must be a comma-separated list of integers"
        ) from exc
    if any(step <= 0 for step in steps):
        raise ValueError("progressive growth steps must be positive")
    if steps != sorted(set(steps)):
        raise ValueError("progressive growth steps must be unique and strictly increasing")
    return steps


def load_tokenizer_file(path: Path):
    try:
        return FastBPETokenizer.load(str(path))
    except Exception:
        return SimpleBPETokenizer.load(str(path))


def tokenizer_eos_id(tokenizer) -> int:
    token = getattr(tokenizer, "eos_token", None)
    if token is None:
        raise ValueError("loaded tokenizer does not expose eos_token")
    mapping = getattr(tokenizer, "token_to_id", None)
    if isinstance(mapping, dict):
        value = mapping.get(token)
    elif getattr(tokenizer, "_tokenizer", None) is not None:
        value = tokenizer._tokenizer.token_to_id(token)
    else:
        value = None
    if value is None:
        raise ValueError(f"tokenizer does not contain EOS token {token!r}")
    return int(value)


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
    if getattr(model, "progressive_depth_enabled", False):
        optimized = sum(p.data.size for p in model.optimization_parameters())
        print(
            f"  Active Transformer depth: {model.active_layers}/{model.max_layers}"
        )
        print(f"  Initial optimizer parameters: {optimized:,}")
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
    if args.resume_from and args.init_from:
        raise ValueError("--resume-from and --init-from are mutually exclusive")

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

    # 0058/0060: the long historical store is searched outside the deep
    # Transformer.  Routed-prefix mode reopens only a bounded 2k subset before
    # the continuous 4k working window.
    # Canonical values live in ModelConfig; environment overrides are provided
    # for benchmark/curriculum sweeps without changing parameter shapes.
    memory_cfg = config.memory_context
    memory_overrides = {}
    override_map = {
        "MINI_LLM_MEMORY_LENGTH": "memory_length",
        "MINI_LLM_MEMORY_BLOCK_SIZE": "block_size",
        "MINI_LLM_MEMORY_TOP_K_BLOCKS": "top_k_blocks",
        "MINI_LLM_MEMORY_RECENT_TOKENS": "recent_length",
        "MINI_LLM_MEMORY_TARGET_TOKENS": "target_length",
        "MINI_LLM_MEMORY_QUERY_TOKENS": "router_query_length",
        "MINI_LLM_MEMORY_ROUTER_DIM": "router_dim",
        "MINI_LLM_MEMORY_READ_HEADS": "read_heads",
        "MINI_LLM_MEMORY_READ_KV_HEADS": "read_kv_heads",
        "MINI_LLM_MEMORY_READ_QUERY_CHUNK": "read_query_chunk",
        "MINI_LLM_MEMORY_ATTENTION_LAYER": "memory_attention_layer",
        "MINI_LLM_MEMORY_MIN_ROUTER_HISTORY_BLOCKS": "min_router_history_blocks",
        "MINI_LLM_MEMORY_ROUTER_TEMPERATURE_ANNEAL_STEPS": "router_temperature_anneal_steps",
    }
    for env_name, field_name in override_map.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            memory_overrides[field_name] = int(raw)
    residual_scale_raw = os.environ.get("MINI_LLM_MEMORY_READER_RESIDUAL_SCALE")
    if residual_scale_raw is not None:
        memory_overrides["reader_residual_scale"] = float(residual_scale_raw)
    float_override_map = {
        "MINI_LLM_MEMORY_ROUTER_TEMPERATURE": "router_temperature",
        "MINI_LLM_MEMORY_ROUTER_TEMPERATURE_MIN": "router_temperature_min",
        "MINI_LLM_MEMORY_ROUTER_SURROGATE_SCALE": "router_surrogate_scale",
        "MINI_LLM_MEMORY_RETRIEVAL_BATCH_PROBABILITY": "retrieval_batch_probability",
    }
    for env_name, field_name in float_override_map.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            memory_overrides[field_name] = float(raw)
    gumbel_raw = os.environ.get("MINI_LLM_MEMORY_ROUTER_GUMBEL_NOISE")
    if gumbel_raw is not None:
        memory_overrides["router_gumbel_noise"] = gumbel_raw.strip().lower() not in {
            "0", "false", "off", "no", ""
        }
    integration_raw = os.environ.get("MINI_LLM_MEMORY_INTEGRATION")
    if integration_raw is not None:
        memory_overrides["integration_mode"] = integration_raw.strip().lower()
    training_raw = os.environ.get("MINI_LLM_MEMORY_TRAINING")
    if training_raw is not None:
        memory_overrides["memory_training"] = training_raw.strip().lower()
    enable_raw = os.environ.get("MINI_LLM_HIERARCHICAL_MEMORY")
    if enable_raw is not None:
        memory_overrides["enabled"] = enable_raw.strip().lower() not in {
            "0", "false", "off", "no", ""
        }
    if memory_overrides:
        memory_cfg = dataclasses.replace(memory_cfg, **memory_overrides)
        config = dataclasses.replace(config, memory_context=memory_cfg)

    # 0064B: progressive-depth is an architecture/checkpoint property, not a
    # runtime kernel flag.  New runs opt in explicitly.  Resume always trusts
    # the checkpoint config so parameter layout cannot silently change.
    if args.resume_from:
        if args.progressive_depth and not config.progressive_depth:
            raise ValueError(
                "cannot enable progressive depth while resuming a non-progressive checkpoint"
            )
        if (
            args.initial_active_layers is not None
            and int(args.initial_active_layers) != int(config.progressive_initial_layers)
        ):
            raise ValueError(
                "--initial-active-layers cannot change the checkpoint's progressive architecture"
            )
    else:
        if args.progressive_depth:
            initial_layers = (
                1 if args.initial_active_layers is None
                else int(args.initial_active_layers)
            )
            config = dataclasses.replace(
                config,
                progressive_depth=True,
                progressive_initial_layers=initial_layers,
            )
        elif args.initial_active_layers is not None:
            raise ValueError(
                "--initial-active-layers requires --progressive-depth on a new run"
            )

    memory_cfg = config.memory_context
    requested_growth_steps = parse_progressive_growth_steps(
        args.progressive_growth_steps
    )
    if requested_growth_steps and not config.progressive_depth:
        raise ValueError(
            "--progressive-growth-steps requires a progressive-depth model"
        )

    if memory_cfg.enabled:
        if args.context_length is not None:
            raise ValueError(
                "--context-length is a direct-sequence control and is ambiguous for "
                "hierarchical-memory models. Use a wide-500m-memory-* preset or "
                "MINI_LLM_MEMORY_LENGTH instead."
            )
        # The loader samples a historical store plus the dense current window;
        # config.context_length records only the bounded deep active length.
        run_context_length = int(memory_cfg.source_input_length)
        config = dataclasses.replace(
            config, context_length=int(memory_cfg.active_length)
        )
    else:
        run_context_length = (
            int(args.context_length)
            if args.context_length is not None
            else int(config.context_length)
        )

    runtime_profile = apply_runtime_defaults(
        args.runtime_profile,
        backend_name=BACKEND_NAME,
        config=config,
        training=True,
        precision=args.precision,
    )
    print(format_runtime_profile(runtime_profile))

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
            if args.model in {
                "medium-context-4k",
                "moe-525m-context-4k",
                "moe-1b-context-4k",
                "wide-500m-context-4k",
                "wide-500m-context-8k",
                "wide-500m-context-16k",
                "wide-500m-context-32k",
                "wide-500m-context-64k",
                "wide-500m-memory-16k",
                "wide-500m-memory-32k",
                "wide-500m-memory-64k",
            }:
                raise RuntimeError(
                    f"{args.model} requires pre-generated packed shards. "
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

    eos_token_id = None
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
        eos_token_id = tokenizer_eos_id(tokenizer)
    elif tokenizer_vocab_size is None:
        raise FileNotFoundError(
            f"tokenizer not found at {tokenizer_path}; pass --tokenizer-path explicitly"
        )

    replace_kwargs = {
        "tokenizer_vocab_size": int(tokenizer_vocab_size),
        "dtype": model_dtype,
    }
    if not config.memory_context.enabled:
        replace_kwargs["context_length"] = run_context_length
    config = dataclasses.replace(config, **replace_kwargs)

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
    if config.memory_context.enabled:
        mc = config.memory_context
        print(f"Memory horizon: {mc.memory_length:,}")
        print(f"External searchable history: {mc.distant_memory_length:,}")
        print(f"Memory integration: {mc.integration_mode}")
        if mc.integration_mode == "terminal_landmark":
            layer = mc.memory_attention_layer if mc.memory_attention_layer >= 0 else config.n_layers + mc.memory_attention_layer
            print(f"Memory-aware Transformer layer: {layer}")
            print("Router-supervised positions/sample: 1 (terminal target only)")
        print(f"Active Transformer length: {mc.active_length:,}")
        print(f"Source sample input length: {run_context_length:,}")
    else:
        print(f"Context length: {run_context_length}")
    if config.progressive_depth:
        print(
            "Progressive depth: enabled "
            f"(initial {config.progressive_initial_layers}/{config.n_layers} layers)"
        )
        if requested_growth_steps is not None:
            print(
                "Requested growth steps: "
                + (
                    ", ".join(str(x) for x in requested_growth_steps)
                    if requested_growth_steps else "manual only"
                )
            )
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}x")
    effective_sequences = args.batch_size * args.grad_accum_steps
    source_tokens_per_microbatch = args.batch_size * run_context_length
    source_tokens_per_optimizer_step = effective_sequences * run_context_length
    print(f"Effective batch: {effective_sequences} sequences")
    if config.memory_context.enabled:
        print(f"Source capacity tokens/microbatch: {source_tokens_per_microbatch:,}")
        print(
            f"Active tokens/microbatch: "
            f"{args.batch_size * config.memory_context.active_length:,}"
        )
        router_only = config.memory_context.memory_training == "router_only"
        supervised_per_sequence = 1 if router_only else config.memory_context.target_length
        print(
            f"Supervised tokens/microbatch: "
            f"{args.batch_size * supervised_per_sequence:,}"
        )
        print(f"Source capacity tokens/optimizer step: {source_tokens_per_optimizer_step:,}")
        print(
            f"Supervised tokens/optimizer step: "
            f"{effective_sequences * supervised_per_sequence:,}"
        )
    else:
        print(f"Tokens/microbatch: {source_tokens_per_microbatch:,}")
        print(f"Tokens/optimizer step: {source_tokens_per_optimizer_step:,}")
    print(f"Training steps: {args.total_steps}")
    print(f"Learning rate: {args.peak_lr}")
    print(f"Warmup: {args.warmup_steps} steps")
    if args.profile_steps:
        print(f"Performance profiling: first {args.profile_steps} optimizer step(s)")
    print()

    estimated_params = config.estimated_parameter_count()
    print(
        f"Preset parameter estimate: {estimated_params:,} "
        f"({estimated_params / 1e6:.1f}M)"
    )
    if config.memory_context.enabled:
        disabled_memory = dataclasses.replace(config.memory_context, enabled=False)
        base_params = dataclasses.replace(
            config, memory_context=disabled_memory
        ).estimated_parameter_count()
        print(f"  Base Transformer parameters: {base_params:,}")
        print(f"  External-memory parameters: {estimated_params - base_params:,}")
    if args.precision == "bf16-mixed":
        gib = 1024 ** 3
        optimized_estimate = (
            config.estimated_parameter_count(
                active_layers=config.progressive_initial_layers
            )
            if config.progressive_depth
            else estimated_params
        )
        if (
            config.memory_context.enabled
            and config.memory_context.memory_training == "router_only"
        ):
            disabled_memory = dataclasses.replace(config.memory_context, enabled=False)
            base_estimate = dataclasses.replace(
                config, memory_context=disabled_memory
            ).estimated_parameter_count()
            optimized_estimate = estimated_params - base_estimate
        # Parameter currently allocates an FP32 gradient buffer for every model
        # tensor at construction time, including frozen router-only backbone
        # weights.  Router-only saves optimizer state/GEMMs and activation caches,
        # but do not under-report this persistent gradient-buffer allocation.
        gpu_param_grad = estimated_params * (2 + 4) / gib
        print(
            "  BF16 weights + allocated FP32 gradients: "
            f"~{gpu_param_grad:.2f} GiB GPU before activations/workspaces"
        )
        if optimized_estimate != estimated_params:
            print(
                f"  Router-only optimized parameters: {optimized_estimate:,} "
                "(optimizer state allocated only for this subset)"
            )
        offload_mode = os.environ.get(
            "MINI_LLM_OPTIMIZER_OFFLOAD", "none"
        ).strip().lower()
        if offload_mode == "full":
            host_state = optimized_estimate * 12 / gib
            print(
                "  Full optimizer offload state: "
                f"~{host_state:.2f} GiB host RAM"
            )
        elif offload_mode == "moments":
            host_state = optimized_estimate * 8 / gib
            gpu_master = optimized_estimate * 4 / gib
            print(
                "  Moment offload state: "
                f"~{host_state:.2f} GiB host + {gpu_master:.2f} GiB GPU master"
            )

    model = setup_model(config, dtype=model_dtype)

    if args.init_from:
        init_path = Path(args.init_from)
        print(f"Initializing matching weights from checkpoint: {init_path}")
        from mini_llm.checkpoint import initialize_matching_parameters

        loaded_names, missing, loaded_elements = initialize_matching_parameters(
            init_path, model.parameters()
        )
        memory_missing = [
            name for name in missing
            if name.startswith("memory_router.") or ".terminal_memory." in name
        ]
        unexpected_missing = [name for name in missing if name not in memory_missing]
        print(
            f"  Loaded {len(loaded_names)} parameter tensors "
            f"({loaded_elements:,} elements) using streaming initialization"
        )
        if memory_missing:
            print(
                f"  Initialized {len(memory_missing)} new 0058C memory tensors "
                "from the requested model preset"
            )
        if unexpected_missing:
            preview = ", ".join(unexpected_missing[:8])
            raise ValueError(
                "--init-from checkpoint is missing non-memory backbone parameters: "
                + preview
            )
        get_pool = getattr(xp, "get_default_memory_pool", None)
        if get_pool is not None:
            get_pool().free_all_blocks()

    if args.resume_from:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        from mini_llm.checkpoint import load_checkpoint

        param_names = [p.name for p in model.parameters()]
        loaded_params, optimizer_state, training_state = load_checkpoint(
            checkpoint_path,
            param_names=param_names,
        )
        # Copy checkpoint parameters into the already-allocated model and
        # immediately drop each temporary backend copy.  On CuPy,
        # load_checkpoint() materializes checkpoint arrays on the GPU, so
        # retaining loaded_params would otherwise keep a second complete model
        # resident throughout resumed training.
        for p in model.parameters():
            loaded = loaded_params.pop(p.name, None)
            if loaded is not None:
                p.data[...] = loaded
                del loaded
        loaded_params.clear()
        del loaded_params

        # Return now-unreferenced CuPy blocks to the device before constructing
        # the trainer.  NumPy has no memory-pool API, so this is a no-op there.
        get_pool = getattr(xp, "get_default_memory_pool", None)
        if get_pool is not None:
            get_pool().free_all_blocks()

        if config.progressive_depth:
            restored_active_layers = int(
                (training_state or {}).get(
                    "progressive_active_layers", config.progressive_initial_layers
                )
            )
            model.set_active_layers(restored_active_layers)
            print(
                "Restored progressive active depth: "
                f"{model.active_layers}/{model.max_layers}"
            )

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

    if requested_growth_steps is None:
        progressive_growth_steps = (
            list((training_state or {}).get("progressive_growth_steps", []))
            if args.resume_from
            else []
        )
    else:
        progressive_growth_steps = list(requested_growth_steps)

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
        profile_steps=args.profile_steps,
        train_source_shards=train_source_shards,
        val_source_shards=val_source_shards,
        source_weights=source_weights,
        eos_token_id=eos_token_id,
        progressive_growth_steps=progressive_growth_steps,
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

    if stored_optimizer_state is not None:
        trainer.optimizer.step_index = int(
            stored_optimizer_state.get("step", start_step)
        )
        optimizer_params = list(trainer.optimizer.parameters)
        param_to_idx = {p.name: i for i, p in enumerate(optimizer_params)}

        # 0055C: restore exact FP32 master weights as well as moments.  When
        # optimizer offload is active, load_checkpoint keeps these arrays on
        # the host so resume never materializes a second optimizer-sized GPU
        # copy.
        master_dict = stored_optimizer_state.get("master_weights", {})
        restored_masters = 0
        for p in optimizer_params:
            value = master_dict.pop(p.name, None)
            if value is None:
                continue
            idx = param_to_idx[p.name]
            if trainer.optimizer.restore_master_weight(idx, value):
                restored_masters += 1
            else:
                print(
                    f"WARNING: Master-weight shape mismatch for {p.name}: "
                    f"stored={value.shape}, "
                    f"current={trainer.optimizer.master_weights[idx].shape}"
                )
            del value
        if master_dict is not None:
            master_dict.clear()
        if restored_masters:
            print(f"Restored exact FP32 master weights for {restored_masters} parameters")

        if "m" in stored_optimizer_state:
            m_dict = stored_optimizer_state["m"]
            v_dict = stored_optimizer_state.get("v", {})
            saved_m_names = set(m_dict)
            restored_count = 0
            for p in optimizer_params:
                if p.name in m_dict and p.name in v_dict:
                    idx = param_to_idx[p.name]
                    m_value = m_dict.pop(p.name)
                    v_value = v_dict.pop(p.name)
                    if trainer.optimizer.restore_moments(idx, m_value, v_value):
                        restored_count += 1
                    else:
                        print(
                            f"WARNING: Shape mismatch for {p.name}: "
                            f"stored={m_value.shape}, current={trainer.optimizer.m[idx].shape}"
                        )
                    del m_value, v_value
            print(f"Restored optimizer state for {restored_count} parameters")
            missing = [
                p.name for p in optimizer_params if p.name not in saved_m_names
            ]
            if missing:
                suffix = "..." if len(missing) > 5 else ""
                print(
                    f"WARNING: {len(missing)} parameters not found in saved optimizer "
                    f"state: {missing[:5]}{suffix}"
                )
            m_dict.clear()
            v_dict.clear()

        birth_state = (training_state or {}).get(
            "optimizer_parameter_birth_steps", {}
        )
        restored_births = trainer.optimizer.restore_parameter_birth_steps(
            birth_state
        )
        if restored_births:
            print(
                f"Restored Adam birth steps for {restored_births} parameters"
            )

        # No checkpoint optimizer arrays are needed after the in-place restore.
        stored_optimizer_state.clear()
        stored_optimizer_state = None
        optimizer_state = None
        get_pool = getattr(xp, "get_default_memory_pool", None)
        if get_pool is not None:
            get_pool().free_all_blocks()

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
