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
import math
import warnings
import os
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, validate_bfloat16_backend
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import (
    AdamW, DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2, DEFAULT_ADAM_EPS,
)
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer
from mini_llm.runtime_defaults import (
    apply_runtime_defaults, format_runtime_profile, format_runtime_fast_paths,
)


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
            "wide-500m-memory-64k", "deep-500m-memory-64k",
            "wide24-500m-memory-64k", "wide24-500m-dense-memory-64k",
            "ensemble-deep-512-dense-memory-64k",
            "ensemble-wide-768-dense-memory-64k",
            "ensemble-deep-512-dense-historical-64k",
            "ensemble-wide-768-dense-historical-64k",
            "ensemble-deep-512-dense-historical-hybrid-64k",
            "ensemble-wide-768-dense-historical-hybrid-64k",
            "ensemble-intermediate-768-20l-dense-historical-hybrid-64k",
            "ensemble-1024-16l-dense-historical-hybrid-64k",
            "ensemble-1024-20l-dense-historical-hybrid-64k",
            "dual-resolution-1024-20l",
            "dual-resolution-matrix-pool-1024-20l",
            "dual-resolution-matrix-pool-sparse-1024-20l",
            "ensemble-1024-20l-dense-direct-full-history-64k",
            "ensemble-1024-22l-dense-historical-hybrid-64k",
            "ensemble-1024-24l-dense-historical-hybrid-64k",
            "ensemble-large-1536-36l-dense-historical-hybrid-64k",
            "ensemble-max-1792-32l-dense-historical-hybrid-64k",
            "ensemble-medium-wide-768-dense-memory-64k",
            "large",
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
    
    # 0080N: optional shared production objective. None preserves the original
    # AR CLI; a resumed 0080 checkpoint selects its stored objective by default.
    parser.add_argument("--objective", choices=["ar", "block-proposal"], default=None,
                        help="Prediction objective (default: AR, or saved checkpoint objective)")
    parser.add_argument("--block-proposal-k", type=int, default=None)
    parser.add_argument("--block-proposal-head-width", type=int, default=None)
    parser.add_argument("--block-proposal-residual-scale", type=float, default=None)
    parser.add_argument("--block-proposal-decoder-width", type=int, default=None)
    parser.add_argument("--block-proposal-decoder-layers", type=int, default=None)
    parser.add_argument("--block-proposal-decoder-heads", type=int, default=None)
    parser.add_argument("--block-proposal-decoder-ff-multiplier", type=int, default=None,
                        help="Small decoder SwiGLU inner-width multiplier (default: 4)")
    parser.add_argument("--block-proposal-lambda", type=float, default=None)
    # 0080AE: explicit future soft-interface contract. Only the legacy
    # expectation/off combination is executable in this patch; experimental
    # selections fail before model allocation until the kernels are complete.
    parser.add_argument("--block-proposal-soft-interface", choices=["expectation", "full_vocab", "probability_embedding", "probability_embedding_direct", "proposal_latents_direct"],
                        default=None)
    parser.add_argument("--block-proposal-prob-embedding-dim", type=int, default=None,
                        help="Trainable full-probability embedding dimension (v3 only)")
    parser.add_argument("--block-proposal-vocab-read-source", choices=["own", "all"], default=None)
    parser.add_argument("--block-proposal-vocab-read-heads", type=int, default=None)
    parser.add_argument("--block-proposal-vocab-read-layer", type=int, default=None)
    parser.add_argument("--block-proposal-vocab-read-temperature", type=float, default=None)
    parser.add_argument("--block-proposal-vocab-read-scale", type=float, default=None)
    parser.add_argument("--block-proposal-boundary-condition", choices=["off", "dec_residual"],
                        default=None)
    parser.add_argument("--block-proposal-vocab-read-algorithm",
                        choices=["source_loop", "source_mixture", "batched_mixture", "fused_mixture"],
                        default=None)
    parser.add_argument("--block-proposal-vocab-read-tile", type=int, default=None)
    parser.add_argument("--block-proposal-vocab-read-block-tile", type=int, default=None)
    parser.add_argument("--block-proposal-init-from-v1", type=str, default=None,
                        help="Full-model v1 weights-only transfer; new v2 optimizer and checkpoint directory")
    parser.add_argument("--block-proposal-group-size", type=int, default=8,
                        help="Bound number of K-block plans decoded together (default: 8)")

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
    # 0077A: optional bounded streaming training sources; never added to validation.
    parser.add_argument(
        "--stream-registry", default=None,
        help="JSON registry of pinned remote training streams (version 1)",
    )
    parser.add_argument(
        "--stream-segment-tokens", type=int, default=524288,
        help="Target tokens per temporary remote segment, default 524288",
    )
    parser.add_argument(
        "--stream-reuse-batches", type=int, default=64,
        help="Training microbatches using each streamed segment before rotation",
    )
    parser.add_argument(
        "--reset-remote-cursors", action="store_true",
        help="One-time recorded recovery if old HF remote cursor cannot replay its segment",
    )
    parser.add_argument(
        "--stream-cache-dir", default=None,
        help="Directory for temporary stream segments; files cleaned up on exit",
    )
    parser.add_argument(
        "--allow-streaming-fork", action="store_true",
        help="Deprecated: data-mixture changes now start a recorded phase automatically",
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
    
    # 0074C: None means "inherit from resumed checkpoint"; for new runs,
    # defaults remain the original AdamW coefficients.
    parser.add_argument(
        "--adam-beta1", type=float, default=None,
        help="AdamW first-moment decay (new-run default: 0.9; resume inherits checkpoint)",
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=None,
        help="AdamW second-moment decay (new-run default: 0.95; resume inherits checkpoint)",
    )
    parser.add_argument(
        "--adam-eps", type=float, default=None,
        help="AdamW denominator epsilon (new-run default: 1e-8; resume inherits checkpoint)",
    )

    # 0076A: Moonlight-scaled Muon for hidden matrices, AdamW elsewhere.
    # Omitted optimizer kind inherits the saved checkpoint kind.
    parser.add_argument(
        "--optimizer", choices=["adamw", "hybrid-muon"], default=None,
        help="Optimizer backend (default: AdamW, or inherit checkpoint on resume)",
    )
    parser.add_argument(
        "--muon-momentum", type=float, default=None,
        help="Muon first-moment decay (default: 0.95)",
    )
    parser.add_argument(
        "--muon-ns-steps", type=int, default=None,
        help="Muon Newton-Schulz iterations (default: 5)",
    )
    parser.add_argument(
        "--muon-matmul-precision", choices=["bf16", "fp32"], default=None,
        help="Muon orthogonalization GEMM precision (default: bf16 on GPU)",
    )
    parser.add_argument(
        "--full-history-only-steps", type=int, default=None,
        help=(
            "On a full-history checkpoint resume, update only independent "
            "full-history-head parameters for this many optimizer steps, then "
            "automatically unfreeze the preserved backbone. The absolute phase "
            "boundary is checkpointed for exact interrupted resume."
        ),
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
    
    parser.add_argument(
        "--benchmark-suite", type=str, default=None,
        help="0092: Optional JSON benchmark suite evaluated in-process at supervised-token milestones",
    )
    parser.add_argument(
        "--benchmark-supervised-tokens-offset", type=float, default=None,
        help="0092: Explicit cumulative supervised-token count for legacy checkpoints lacking the counter",
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
        "--val-per-source-steps", type=int, default=0,
        help=("Fixed held-out batches per local validation source at each "
              "--val-interval (0 disables; 2 recommended for a pilot)"),
    )
    parser.add_argument(
        "--val-source-seed", type=int, default=78078,
        help="Fixed per-source validation seed; keep identical across checkpoints",
    )
    parser.add_argument(
        "--val-representative-steps", type=int, default=0,
        help=("0078C: fixed source-token-sampled validation batches per local "
              "source, without a minimum-history constraint (0 disables)"),
    )
    parser.add_argument(
        "--val-representative-seed", type=int, default=78078,
        help="Fixed seed for representative per-source evaluation",
    )
    parser.add_argument(
        "--val-fixed-mixed-steps", type=int, default=0,
        help="0078F: reproducible document-first, target-weighted mixed validation batches (0 disables)",
    )
    parser.add_argument(
        "--val-fixed-mixed-seed", type=int, default=78078,
        help="0078F: fixed mixed validation seed; hold constant across checkpoints",
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
    
    parser.add_argument(
        "--max-run-steps", type=int, default=None,
        help=("Run at most N optimizer updates without changing the LR "
              "scheduler's --total-steps horizon (for smoke tests)."),
    )

    # 0077K: token-normalized accumulation across variable-length microbatches.
    parser.add_argument(
        "--gradient-accumulation-weighting",
        choices=["microbatch", "supervised_tokens"],
        default=None,
        help=("Gradient objective: 'microbatch' averages microbatch mean losses "
              "(legacy); 'supervised_tokens' computes mean loss over all valid "
              "targets in the accumulated optimizer step. On resume, inherit "
              "the checkpoint mode unless explicitly overridden."),
    )

    # 0074A: observational gradient diagnostics; no change to AdamW updates.
    parser.add_argument("--gradient-noise-monitor", action="store_true",
                        help="Enable read-only disjoint-half gradient diagnostics")
    parser.add_argument("--gradient-noise-interval", type=int, default=100,
                        help="Measure every N completed optimizer steps (default: 100)")
    parser.add_argument("--gradient-noise-output", default="./logs/gradient_noise.csv",
                        help="Separate gradient diagnostic CSV file")
    parser.add_argument("--gradient-noise-accum-points", default="8,16,32",
                        help="Even nested microbatch counts, <= grad accumulation")
    parser.add_argument("--gradient-noise-parameter-groups",
                        default="attention,moe,moe_router,memory,head",
                        help="Selected parameter categories, comma separated")

    # 0084E: norm plateau against a random-direction null during accumulation.
    parser.add_argument("--gradient-convergence-monitor", action="store_true",
                        help="Read-only partial-gradient convergence vs random-sign reference")
    parser.add_argument("--gradient-convergence-interval", type=int, default=100,
                        help="Measure every N completed optimizer updates")
    parser.add_argument("--gradient-convergence-points", default="auto",
                        help="Microbatch counts; final accumulation point added automatically")
    parser.add_argument("--gradient-convergence-groups", default="attention,ffn",
                        help="Selected matrix groups: attention,ffn,moe,memory,head")
    parser.add_argument("--gradient-convergence-output",
                        default="./logs/gradient_convergence.csv",
                        help="Separate observational convergence CSV")

    # 0074B: read-only counterfactual AdamW direction diagnostics.
    parser.add_argument("--adamw-update-monitor", action="store_true",
                        help="Compare hypothetical AdamW updates at gradient-noise accumulation points")
    parser.add_argument("--adamw-update-output", default="./logs/adamw_update.csv",
                        help="Separate CSV for observational AdamW update diagnostics")

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


def resolve_adamw_hyperparameters(args, training_state=None):
    """Resolve CLI overrides, saved checkpoint settings, and legacy defaults.

    Old checkpoints predate 0074C and do not record hyperparameters; their
    original AdamW defaults are used unless explicitly overridden. Overrides
    retain the saved moments/ages, so they are not an exact optimizer resume.
    """
    defaults = {
        "beta1": DEFAULT_ADAM_BETA1,
        "beta2": DEFAULT_ADAM_BETA2,
        "eps": DEFAULT_ADAM_EPS,
    }
    saved = (training_state or {}).get("adamw_hyperparameters") or {}
    resolved = {}
    changed = []
    for name, default in defaults.items():
        checkpoint_value = float(saved.get(name, default))
        value = getattr(args, f"adam_{name}")
        effective = checkpoint_value if value is None else float(value)
        if name.startswith("beta"):
            valid = math.isfinite(effective) and 0.0 <= effective < 1.0
            required = "finite and in [0, 1)"
        else:
            valid = math.isfinite(effective) and effective > 0.0
            required = "finite and > 0"
        if not valid:
            raise ValueError(f"--adam-{name} must be {required}; got {effective!r}")
        if value is not None and training_state is not None and effective != checkpoint_value:
            changed.append(f"{name}: {checkpoint_value:g} -> {effective:g}")
        resolved[name] = effective
    if changed:
        warnings.warn(
            "AdamW hyperparameters changed on checkpoint resume ("
            + ", ".join(changed)
            + "). Existing moments and per-parameter ages are retained. "
            + "This is an optimizer transition, not an exact resume. "
            + ("Legacy checkpoint hyperparameters were assumed to use the "
               "original defaults. " if not saved else ""),
            UserWarning,
            stacklevel=2,
        )
    return resolved


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
    if name == "deep-500m-memory-64k":
        return ModelConfig.deep_500m_memory_64k()
    if name == "wide24-500m-memory-64k":
        return ModelConfig.wide24_500m_memory_64k()
    if name == "wide24-500m-dense-memory-64k":
        return ModelConfig.wide24_500m_dense_memory_64k()
    if name == "ensemble-deep-512-dense-memory-64k":
        return ModelConfig.ensemble_deep_512_dense_memory_64k()
    if name == "ensemble-wide-768-dense-memory-64k":
        return ModelConfig.ensemble_wide_768_dense_memory_64k()
    if name == "ensemble-deep-512-dense-historical-64k":
        return ModelConfig.ensemble_deep_512_dense_historical_64k()
    if name == "ensemble-wide-768-dense-historical-64k":
        return ModelConfig.ensemble_wide_768_dense_historical_64k()
    if name == "ensemble-deep-512-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_deep_512_dense_historical_hybrid_64k()
    if name == "ensemble-wide-768-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_wide_768_dense_historical_hybrid_64k()
    if name == "ensemble-intermediate-768-20l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_intermediate_768_20l_dense_historical_hybrid_64k()
    if name == "ensemble-1024-16l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_1024_16l_dense_historical_hybrid_64k()
    if name == "ensemble-1024-20l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    if name == "dual-resolution-1024-20l":
        return ModelConfig.dual_resolution_1024_20l()
    if name == "dual-resolution-matrix-pool-1024-20l":
        return ModelConfig.dual_resolution_matrix_pool_1024_20l()
    if name == "dual-resolution-matrix-pool-sparse-1024-20l":
        return ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    if name == "ensemble-1024-20l-dense-direct-full-history-64k":
        return ModelConfig.ensemble_1024_20l_dense_direct_full_history_64k()
    if name == "ensemble-1024-22l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_1024_22l_dense_historical_hybrid_64k()
    if name == "ensemble-1024-24l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_1024_24l_dense_historical_hybrid_64k()
    if name == "ensemble-large-1536-36l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_large_1536_36l_dense_historical_hybrid_64k()
    if name == "ensemble-max-1792-32l-dense-historical-hybrid-64k":
        return ModelConfig.ensemble_max_1792_32l_dense_historical_hybrid_64k()
    if name == "ensemble-medium-wide-768-dense-memory-64k":
        return ModelConfig.ensemble_medium_wide_768_dense_memory_64k()
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



def plan_data_mixture_resume(*, resume_from, stored_provenance, stream_specs,
                             segment_tokens, reuse_batches, mixture_sha):
    """Resume exact data state when unchanged; otherwise start a new data phase.

    A deliberate change to sources, revisions, weights, local manifest or
    stream-segment policy is never an error.  It starts fresh remote cursors,
    preserving model/optimizer state and adding an auditable phase boundary.
    Tokenizer incompatibility is checked separately and remains fatal.
    """
    if not resume_from:
        return [], False
    prior = stored_provenance or {}
    old_stream = prior.get("streaming")
    old_manifest_sha = prior.get("local_mixture_manifest_sha256")
    import hashlib
    from mini_llm.data.streaming_sources import _canonical
    new_registry_fp = (
        hashlib.sha256(_canonical([spec.__dict__ for spec in stream_specs]).encode()).hexdigest()
        if stream_specs else None
    )
    reasons = []
    if bool(old_stream) != bool(stream_specs):
        reasons.append("remote sources enabled/disabled")
    elif old_stream and stream_specs:
        if old_stream.get("registry_fingerprint") != new_registry_fp:
            reasons.append("remote dataset registry/revision/weights changed")
        if (int(old_stream.get("segment_tokens", -1)) != int(segment_tokens)
                or int(old_stream.get("reuse_batches", -1)) != int(reuse_batches)):
            reasons.append("remote segment/reuse policy changed")
    if old_manifest_sha is not None and old_manifest_sha != mixture_sha:
        reasons.append("local shard mixture manifest changed")
    exact_stream_resume = bool(old_stream and stream_specs and not reasons)
    return reasons, exact_stream_resume



def restore_compatible_stream_cursors(cache, stored_state, *, checkpoint_dir=None):
    """Keep exact cursors for sources whose identity and segment policy match.

    Mixture weights may change without replaying remote data.  New sources and
    sources with changed revision/content schema start fresh.  Checkpoint
    source fingerprints are still verified when regenerated on first use.
    """
    if not stored_state:
        return []
    if (stored_state.get("segment_tokens") != cache.segment_tokens or
            stored_state.get("reuse_batches") != cache.reuse_batches):
        return []
    prior_specs = {s["name"]: s for s in stored_state.get("registry", [])}
    restored = []
    for name, spec in cache.specs.items():
        prior = prior_specs.get(name)
        if prior is None or name not in stored_state.get("sources", {}):
            continue
        current_identity = {k: v for k, v in spec.__dict__.items() if k != "weight"}
        previous_identity = {k: v for k, v in prior.items() if k != "weight"}
        if current_identity == previous_identity:
            if checkpoint_dir is None:
                cache.sources[name].load_state_dict(stored_state["sources"][name])
            else:
                cache.sources[name].load_state_dict(
                    stored_state["sources"][name], checkpoint_dir=checkpoint_dir,
                )
            restored.append(name)
    return restored


def populate_v1_warmstart_options(args):
    """Resolve unchanged K/head/decoder dimensions directly from the source v1 manifest.

    New v2 fields remain independently selected. Refuse a contradictory CLI
    flag before allocating the 500M model or touching source checkpoint data.
    """
    source=getattr(args,'block_proposal_init_from_v1',None)
    if not source:
        return None
    from mini_llm.block_proposal.checkpoint import ARCHITECTURE
    source_path=Path(source).expanduser().resolve()
    with (source_path/'training_state.json').open(encoding='utf8') as handle:
        stored=json.load(handle)
    architecture=stored.get('architecture') or {}
    if architecture.get('type')!=ARCHITECTURE:
        raise ValueError('weights-only v2 initialization requires a complete v1 block checkpoint')
    decoder_width=int(architecture['decoder_width'])
    ff_width=int(architecture['decoder_ff_width'])
    if ff_width%decoder_width:
        raise ValueError('source v1 decoder FF width is not divisible by decoder width')
    shared={
        'block_proposal_k':int(architecture['block_size']),
        'block_proposal_head_width':int(architecture['proposal_hidden_width']),
        'block_proposal_residual_scale':float(architecture['proposal_residual_scale']),
        'block_proposal_decoder_width':decoder_width,
        'block_proposal_decoder_layers':int(architecture['decoder_layers']),
        'block_proposal_decoder_heads':int(architecture['decoder_heads']),
        'block_proposal_decoder_ff_multiplier':ff_width//decoder_width,
        'block_proposal_lambda':float(architecture['lambda_proposal']),
    }
    for option,value in shared.items():
        current=getattr(args,option,None)
        if current is not None and current!=value:
            raise ValueError('v1 warm-start architecture option mismatch: '+option)
        setattr(args,option,value)
    return architecture


def build_training_objective(model, args, *, checkpoint_state=None):
    """Construct only the 0080-specific objective, reusing production model.

    A resume must agree *exactly* with serialized 0080 architecture. The
    existing --model/config factory remains the source of all trunk settings.
    """
    from mini_llm.block_proposal.checkpoint import ARCHITECTURE, ARCHITECTURE_V2, ARCHITECTURE_V3, ARCHITECTURE_V4, ARCHITECTURE_V5
    saved = checkpoint_state or {}
    old = saved.get("prediction_objective", "block-proposal" if
                    (saved.get("architecture") or {}).get("type") in (ARCHITECTURE,ARCHITECTURE_V2,ARCHITECTURE_V3,ARCHITECTURE_V4,ARCHITECTURE_V5)
                    else "ar")
    requested = args.objective or old
    if checkpoint_state is not None and requested != old:
        raise ValueError("cannot change checkpoint prediction objective on resume")
    if requested == "ar":
        explicit = [name for name in (
            "block_proposal_k", "block_proposal_head_width", "block_proposal_residual_scale",
            "block_proposal_decoder_width", "block_proposal_decoder_layers",
            "block_proposal_decoder_heads", "block_proposal_decoder_ff_multiplier",
            "block_proposal_lambda", "block_proposal_soft_interface",
            "block_proposal_prob_embedding_dim",
            "block_proposal_vocab_read_source", "block_proposal_vocab_read_heads",
            "block_proposal_vocab_read_layer", "block_proposal_vocab_read_temperature",
            "block_proposal_vocab_read_scale", "block_proposal_boundary_condition",
            "block_proposal_vocab_read_algorithm", "block_proposal_vocab_read_tile",
            "block_proposal_vocab_read_block_tile", "block_proposal_init_from_v1")
            if getattr(args, name, None) is not None]
        if explicit:
            raise ValueError("block proposal options require --objective block-proposal")
        return None
    if requested != "block-proposal":
        raise ValueError("unknown objective")
    previous = (saved.get("architecture") or {}) if checkpoint_state is not None else {}
    if checkpoint_state is not None and previous.get("type") not in (ARCHITECTURE,ARCHITECTURE_V2,ARCHITECTURE_V3,ARCHITECTURE_V4,ARCHITECTURE_V5):
        raise ValueError("0080 resume requires a saved two-stage architecture")
    if model.config.progressive_depth:
        raise ValueError("0080 does not yet support progressive depth")
    if model.config.memory_context.enabled and (
        model.config.memory_context.integration_mode != "routed_prefix"
        or model.config.memory_context.memory_training != "joint"
    ):
        raise ValueError("0080 requires joint routed-prefix learned memory")

    def choice(flag, key, default):
        value = getattr(args, flag, None)
        saved_value = previous.get(key)
        if saved_value is not None and value is not None and value != saved_value:
            raise ValueError("0080 checkpoint option mismatch: " + flag)
        return saved_value if value is None and saved_value is not None else (
            default if value is None else value)

    k = int(choice("block_proposal_k", "block_size", 8))
    head_width = int(choice("block_proposal_head_width", "proposal_hidden_width",
                            model.config.d_model))
    residual = float(choice("block_proposal_residual_scale", "proposal_residual_scale", .1))
    width = int(choice("block_proposal_decoder_width", "decoder_width", 384))
    layers = int(choice("block_proposal_decoder_layers", "decoder_layers", 3))
    heads = int(choice("block_proposal_decoder_heads", "decoder_heads", 6))
    # FF width was always checkpointed, but previously only configurable in
    # the Python class, not the production training CLI.
    saved_ff = previous.get("decoder_ff_width")
    if saved_ff is not None:
        if int(saved_ff) % width:
            raise ValueError("checkpoint decoder FF width is not an integer multiple of width")
        saved_multiplier = int(saved_ff) // width
    else:
        saved_multiplier = 4
    explicit_multiplier = getattr(args, "block_proposal_decoder_ff_multiplier", None)
    if explicit_multiplier is not None and checkpoint_state is not None and (
            int(explicit_multiplier) != saved_multiplier):
        raise ValueError("0080 checkpoint option mismatch: block_proposal_decoder_ff_multiplier")
    multiplier = (saved_multiplier if explicit_multiplier is None
                  else int(explicit_multiplier))
    if multiplier < 1:
        raise ValueError("block-proposal decoder FF multiplier must be positive")
    coefficient = float(choice("block_proposal_lambda", "lambda_proposal", .5))
    if k < 1 or k > 64 or not 0 < coefficient < float("inf"):
        raise ValueError("0080 requires 1 <= K <= 64 and finite positive lambda")
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    field_flags = {
        'mode':'block_proposal_soft_interface',
        'source_scope':'block_proposal_vocab_read_source',
        'read_heads':'block_proposal_vocab_read_heads',
        'read_layer':'block_proposal_vocab_read_layer',
        'temperature':'block_proposal_vocab_read_temperature',
        'read_scale':'block_proposal_vocab_read_scale',
        'boundary_condition':'block_proposal_boundary_condition',
        'probability_embedding_dim':'block_proposal_prob_embedding_dim',
    }
    defaults=SoftInterfaceConfig()
    stored_interface=previous.get('soft_interface') or {}
    values={}
    for fld,flag in field_flags.items():
        requested_value=getattr(args,flag,None)
        saved_value=stored_interface.get(fld,getattr(defaults,fld))
        if checkpoint_state is not None and requested_value is not None and requested_value!=saved_value:
            raise ValueError('v2 soft interface resume mismatch: '+flag)
        values[fld]=(saved_value if checkpoint_state is not None else
                     (getattr(defaults,fld) if requested_value is None else requested_value))
    # In the direct mode the embedding width is structurally tied to d_model.
    # A fresh v4 run needs no independent probability-dimension flag.
    if checkpoint_state is None and values['mode']=='probability_embedding_direct' and (
            getattr(args,'block_proposal_prob_embedding_dim',None) is None):
        values['probability_embedding_dim']=int(model.config.d_model)
    interface=SoftInterfaceConfig(**values).validate(
        model_width=model.config.d_model,decoder_layers=layers,block_size=k)
    v2=not interface.is_v1_baseline
    read_state=previous.get('vocab_read_execution') or {}
    algorithm=choice('block_proposal_vocab_read_algorithm','_unused',
                     'fused_mixture' if interface.mode=='full_vocab' and interface.source_scope=='all' else 'source_mixture')
    vocab_tile=int(choice('block_proposal_vocab_read_tile','_unused_tile',2048))
    block_tile=int(choice('block_proposal_vocab_read_block_tile','_unused_block_tile',8))
    if checkpoint_state is not None and read_state:
        for flag,value,key in (
            ('block_proposal_vocab_read_algorithm',algorithm,'algorithm'),
            ('block_proposal_vocab_read_tile',vocab_tile,'vocab_tile'),
            ('block_proposal_vocab_read_block_tile',block_tile,'block_tile')):
            requested_value=getattr(args,flag,None)
            saved_value=read_state[key]
            if requested_value is not None and requested_value!=saved_value:
                raise ValueError('v2 vocabulary read resume mismatch: '+flag)
            if key=='algorithm': algorithm=saved_value
            elif key=='vocab_tile': vocab_tile=int(saved_value)
            else: block_tile=int(saved_value)
    if not v2:
        interface.require_implemented()
        if any(getattr(args,flag,None) is not None for flag in (
            'block_proposal_vocab_read_algorithm','block_proposal_vocab_read_tile',
            'block_proposal_vocab_read_block_tile')):
            raise ValueError('vocab read algorithm/tile flags require v2')
    if v2 and interface.mode!='full_vocab' and read_state:
        raise ValueError('non-full-vocabulary variant cannot own vocab read metadata')
    if interface.mode not in ('probability_embedding','probability_embedding_direct') and getattr(args,'block_proposal_prob_embedding_dim',None) is not None:
        raise ValueError('probability embedding dimension requires probability_embedding interface')
    if v2 and interface.mode!='full_vocab' and any(
            getattr(args,flag,None) is not None for flag in (
            'block_proposal_vocab_read_algorithm','block_proposal_vocab_read_tile',
            'block_proposal_vocab_read_block_tile')):
        raise ValueError('vocabulary read execution flags require full_vocab interface')
    if v2 and interface.mode=='full_vocab' and interface.source_scope!='all' and algorithm!='source_mixture':
        raise ValueError('own-source requires source_mixture algorithm')
    from mini_llm.backend import BACKEND_NAME
    if v2 and BACKEND_NAME!='cupy' and model.config.vocab_size>512:
        raise NotImplementedError('v2 large-vocabulary experiments require CuPy')
    if v2 and BACKEND_NAME!='cupy' and algorithm=='fused_mixture':
        algorithm='source_mixture' if checkpoint_state is None else algorithm
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.block_proposal.objective import BlockProposalObjective
    head = HorizonProposalHead(model, block_size=k, hidden_width=head_width,
                               residual_scale=residual)
    decoder = BlockPrefixDecoder(model, k, width=width, n_layers=layers,
                                 n_heads=heads, ff_multiplier=multiplier,
                                 soft_interface_config=interface,
                                 experimental_gpu_full_vocab=(v2 and BACKEND_NAME=='cupy'),
                                 experimental_cpu_full_vocab=(v2 and BACKEND_NAME!='cupy'),
                                 vocab_read_algorithm=algorithm,
                                 vocab_read_tile=vocab_tile,vocab_read_block_tile=block_tile)
    net = TwoStageBlockModel(head, decoder, lambda_proposal=coefficient)
    objective = BlockProposalObjective(net, block_group_size=args.block_proposal_group_size)
    if checkpoint_state is not None and objective.checkpoint_metadata() != previous:
        raise ValueError("0080 checkpoint architecture mismatch (model configuration or decoder FF width)")
    return objective


def main():
    args = parse_args()
    if sum(bool(x) for x in (args.resume_from,args.init_from,args.block_proposal_init_from_v1))>1:
        raise ValueError('--resume-from, --init-from and --block-proposal-init-from-v1 are mutually exclusive')

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
    if runtime_profile.resolved != "reference" and BACKEND_NAME == "cupy":
        print(format_runtime_fast_paths(config))

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
                "deep-500m-memory-64k",
                "wide24-500m-memory-64k",
                "wide24-500m-dense-memory-64k",
                "ensemble-deep-512-dense-memory-64k",
                "ensemble-wide-768-dense-memory-64k",
                "ensemble-medium-wide-768-dense-memory-64k",
                "ensemble-deep-512-dense-historical-64k",
                "ensemble-wide-768-dense-historical-64k",
                "ensemble-deep-512-dense-historical-hybrid-64k",
                "ensemble-wide-768-dense-historical-hybrid-64k",
                "ensemble-intermediate-768-20l-dense-historical-hybrid-64k",
                "ensemble-1024-16l-dense-historical-hybrid-64k",
                "ensemble-1024-20l-dense-historical-hybrid-64k",
                "ensemble-1024-20l-dense-direct-full-history-64k",
                "ensemble-1024-22l-dense-historical-hybrid-64k",
                "ensemble-1024-24l-dense-historical-hybrid-64k",
                "ensemble-large-1536-36l-dense-historical-hybrid-64k",
                "ensemble-max-1792-32l-dense-historical-hybrid-64k",
                
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

    if args.model in {
        "ensemble-deep-512-dense-memory-64k",
        "ensemble-wide-768-dense-memory-64k",
        "ensemble-medium-wide-768-dense-memory-64k",
                "ensemble-deep-512-dense-historical-64k",
                "ensemble-wide-768-dense-historical-64k",
                "ensemble-deep-512-dense-historical-hybrid-64k",
                "ensemble-wide-768-dense-historical-hybrid-64k",
                "ensemble-intermediate-768-20l-dense-historical-hybrid-64k",
                "ensemble-1024-16l-dense-historical-hybrid-64k",
                "ensemble-1024-20l-dense-historical-hybrid-64k",
                "ensemble-1024-20l-dense-direct-full-history-64k",
                "ensemble-1024-22l-dense-historical-hybrid-64k",
                "ensemble-1024-24l-dense-historical-hybrid-64k",
                "ensemble-large-1536-36l-dense-historical-hybrid-64k",
                "ensemble-max-1792-32l-dense-historical-hybrid-64k",
    } and int(tokenizer_vocab_size) != 32_768:
        raise ValueError(
            "0080BL ensemble presets require their shared 32,768-token vocabulary; "
            f"loaded {tokenizer_vocab_size} tokens from {tokenizer_path}"
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
        disabled_memory = dataclasses.replace(
            config.memory_context,
            enabled=False,
            integration_mode=(
                "pretransformer_read"
                if config.memory_context.integration_mode == "direct_full_history"
                else config.memory_context.integration_mode
            ),
        )
        base_params = dataclasses.replace(
            config, memory_context=disabled_memory, full_history_layers=()
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
    saved_objective_state = None
    if args.resume_from:
        resume_metadata = checkpoint_path / "training_state.json"
        if not resume_metadata.is_file():
            raise ValueError("checkpoint missing training_state.json")
        with resume_metadata.open(encoding="utf8") as handle:
            saved_objective_state = json.load(handle)
    if args.block_proposal_init_from_v1:
        populate_v1_warmstart_options(args)
    training_objective = build_training_objective(
        model, args, checkpoint_state=saved_objective_state)
    if training_objective is not None:
        # Never overwrite a v1 or previously completed v2 run, including for
        # completely fresh v2 initialization (not only v1 warm-start).
        from mini_llm.block_proposal.checkpoint import ARCHITECTURE_V2,ARCHITECTURE_V3,ARCHITECTURE_V4,ARCHITECTURE_V5
        if (not args.resume_from and
                training_objective.checkpoint_metadata()['type'] in (ARCHITECTURE_V2,ARCHITECTURE_V3,ARCHITECTURE_V4,ARCHITECTURE_V5)):
            if not args.checkpoint_dir:
                raise ValueError('new v2 training requires a checkpoint directory')
            dest=Path(args.checkpoint_dir).resolve()
            if dest.exists() and any(dest.iterdir()):
                raise ValueError('v2 checkpoint directory must be NEW and empty; never overwrite an existing run')
        print("Prediction objective: probabilistic K-block proposal + prefix decoder")
        print("  K:", training_objective.net.proposal_head.block_size,
              "decoder groups:", training_objective.block_group_size)

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

    if args.block_proposal_init_from_v1:
        if training_objective is None:
            raise ValueError('--block-proposal-init-from-v1 requires block-proposal v2 objective')
        if args.checkpoint_dir is None:
            raise ValueError('v2 warm start requires a NEW checkpoint directory')
        source_path=Path(args.block_proposal_init_from_v1).resolve()
        target_path=Path(args.checkpoint_dir).resolve()
        if source_path==target_path or source_path in target_path.parents:
            raise ValueError('never save v2 checkpoints over/inside a source v1 checkpoint')
        if target_path.exists() and any(target_path.iterdir()):
            raise ValueError('v2 checkpoint destination must be new and empty')
        from mini_llm.block_proposal.checkpoint import warmstart_v2_from_v1
        identity=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        detail=warmstart_v2_from_v1(source_path,training_objective.net,
                                    expected_tokenizer_hash=identity)
        print('v1 -> v2 complete weights-only transfer:',detail)
        get_pool=getattr(xp,'get_default_memory_pool',None)
        if get_pool is not None:
            get_pool().free_all_blocks()

    if args.resume_from:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        from mini_llm.checkpoint import load_checkpoint

        restore_parameters = (list(model.parameters()) if training_objective is None
                              else training_objective.parameters())
        param_names = [p.name for p in restore_parameters]
        loaded_params, optimizer_state, training_state = load_checkpoint(
            checkpoint_path,
            param_names=param_names,
        )
        # Resume is an exact continuation, never an architecture warm start.
        # Partial loading remains available only through the explicit --init-from
        # path. Check the on-disk set too because load_checkpoint(param_names=...)
        # intentionally filters unexpected arrays.
        expected_names = set(param_names)
        disk_names = {
            path.stem for path in (checkpoint_path / "model_params").glob("*.npy")
        }
        if disk_names != expected_names or set(loaded_params) != expected_names:
            missing = sorted(expected_names - disk_names)
            extra = sorted(disk_names - expected_names)
            raise ValueError(
                "resume checkpoint parameter set differs from current architecture; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        if optimizer_state is None or training_state is None:
            raise ValueError("resume requires complete optimizer and training state")
        saved_optimizer_kind = training_state.get("optimizer_kind")
        if saved_optimizer_kind not in {"adamw", "hybrid-muon"}:
            raise ValueError("unknown or missing checkpoint optimizer kind")
        if saved_optimizer_kind == "hybrid-muon":
            from mini_llm.optim.hybrid_muon import muon_eligible
            muon_names = {p.name for p in restore_parameters if muon_eligible(p)}
            adam_names = expected_names - muon_names
            expected_components = {
                "master_weights": expected_names, "m": adam_names,
                "v": adam_names, "muon_m": muon_names,
            }
        else:
            expected_components = {
                "master_weights": expected_names,
                "m": expected_names, "v": expected_names,
            }
        for component, names in expected_components.items():
            if set(optimizer_state.get(component) or {}) != names:
                raise ValueError(
                    "incomplete resume optimizer checkpoint state: " + component
                )
        # Copy checkpoint parameters into the already-allocated model and
        # immediately drop each temporary backend copy.  On CuPy,
        # load_checkpoint() materializes checkpoint arrays on the GPU, so
        # retaining loaded_params would otherwise keep a second complete model
        # resident throughout resumed training.
        for p in restore_parameters:
            loaded = loaded_params.pop(p.name, None)
            if loaded is not None:
                p.data[...] = loaded
                del loaded
        loaded_params.clear()
        del loaded_params
        for block in model.blocks:
            if hasattr(block, "refresh_compute_buffers"):
                block.refresh_compute_buffers()
        if training_objective is not None:
            for layer in training_objective.net.decoder.layers:
                if hasattr(layer.attn, "refresh_compute_buffers"):
                    layer.attn.refresh_compute_buffers()

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
        training_state = None  # New run: there is no checkpoint metadata.
        start_step = 0
        tokens_processed = 0
        rng_states = None

    # Data provenance must identify immutable source revisions and tokenizer.
    import hashlib
    from mini_llm.data.streaming_sources import load_registry
    stream_specs = load_registry(args.stream_registry) if args.stream_registry else []
    if stream_specs and mixed_shard_root is None:
        raise ValueError("--stream-registry requires --mixed-shard-root")
    manifest_path = (mixed_shard_root / "mixture_manifest.json") if mixed_shard_root else None
    mixture_sha = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path is not None and manifest_path.is_file() else None
    )
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    stored_provenance = ((training_state or {}).get("data_provenance") or {})
    stored_stream = stored_provenance.get("streaming")
    reasons, resume_stream_exactly = plan_data_mixture_resume(
        resume_from=args.resume_from,
        stored_provenance=stored_provenance,
        stream_specs=stream_specs,
        segment_tokens=args.stream_segment_tokens,
        reuse_batches=args.stream_reuse_batches,
        mixture_sha=mixture_sha,
    )
    if args.reset_remote_cursors:
        if not args.resume_from or not stored_stream or not stream_specs:
            raise ValueError("--reset-remote-cursors requires a streamed checkpoint resume")
        reasons.append("explicit_remote_cursor_reset_after_unverifiable_legacy_segment")
        resume_stream_exactly = False
        print("WARNING: starting a new REMOTE stream cursor phase; model, optimizer, "
              "local cursors, and source statistics are preserved. Remote documents "
              "may repeat.")
    # The source checkpoint may have trained with 0077E/I packing. Restarting
    # with the streaming-only code is a distinct, auditable training-data phase.
    old_packing = stored_provenance.get("short_document_packing") or {}
    if old_packing.get("enabled"):
        reasons.append("short_document_packing_disabled_streaming_only_rollback")
    # A source change is a normal continuation, not a special CLI mode.
    # Keep a permanent transition log *inside* subsequent checkpoints.
    mixture_history = list(stored_provenance.get("mixture_history") or [])
    fork_provenance = stored_provenance.get("fork")
    phase_start_source_counts = dict(
        stored_provenance.get("mixture_phase_start_source_counts") or {}
    )
    phase_start_batch_counts = dict(
        stored_provenance.get("mixture_phase_start_batch_counts") or {}
    )
    if reasons:
        state_path = checkpoint_path / "training_state.json"
        new_sources = [spec.__dict__.copy() for spec in stream_specs]
        event = {
            "kind": "data_mixture_changed",
            "start_step": int(start_step),
            "start_tokens_processed": int(tokens_processed),
            "reasons": reasons,
            "parent_checkpoint": str(checkpoint_path.resolve()),
            "parent_training_state_sha256": (
                hashlib.sha256(state_path.read_bytes()).hexdigest()
                if state_path.is_file() else None
            ),
            "previous": {
                "local_mixture_manifest_sha256": stored_provenance.get(
                    "local_mixture_manifest_sha256"
                ),
                "effective_weights": stored_provenance.get("effective_weights"),
                "remote_registry": (stored_stream or {}).get("registry", []),
                "remote_registry_fingerprint": (stored_stream or {}).get(
                    "registry_fingerprint"
                ),
                # Includes segment hashes and cursors: preserve the previous
                # phase's audit trail after the active stream is replaced.
                "remote_stream_state": stored_stream,
                "segment_tokens": (stored_stream or {}).get("segment_tokens"),
                "reuse_batches": (stored_stream or {}).get("reuse_batches"),
                "source_supervised_tokens_at_end": dict(
                    (training_state or {}).get("train_source_supervised_tokens", {})
                ),
                "source_microbatches_at_end": dict(
                    (training_state or {}).get("train_source_batch_counts", {})
                ),
                "sample_trace_sha256_at_end": stored_provenance.get(
                    "sample_trace_sha256"
                ),
                "short_document_packing": old_packing or None,
            },
            "new": {
                "local_mixture_manifest_sha256": mixture_sha,
                "remote_registry": new_sources,
                "segment_tokens": args.stream_segment_tokens if stream_specs else None,
                "reuse_batches": args.stream_reuse_batches if stream_specs else None,
                "short_document_packing": {"enabled": False},
            },
        }
        mixture_history.append(event)
        phase_start_source_counts = dict(
            (training_state or {}).get("train_source_supervised_tokens", {})
        )
        phase_start_batch_counts = dict(
            (training_state or {}).get("train_source_batch_counts", {})
        )
        if fork_provenance is None:
            fork_provenance = event
        print("DATA MIXTURE UPDATED AUTOMATICALLY at step "
              f"{start_step}: {', '.join(reasons)}")
        print("  Model/optimizer states and historical source counts retained; "
              "unchanged remote sources keep their stream cursors.")
        if (args.checkpoint_dir and
                Path(args.checkpoint_dir).resolve() == checkpoint_path.resolve()):
            print("  NOTE: checkpoint output equals the resumed directory; "
                  "future saves overwrite the previous weights. "
                  "The mixture transition is recorded in training_state.json.")
    if (args.resume_from and stored_provenance.get("tokenizer_sha256") is not None
            and stored_provenance["tokenizer_sha256"] != tokenizer_sha):
        raise ValueError("training tokenizer differs from checkpoint provenance")

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

    # Resolve after reading the checkpoint so omitted CLI options inherit the
    # saved configuration rather than silently returning to the defaults.
    adamw_hparams = resolve_adamw_hyperparameters(
        args, training_state if args.resume_from else None,
    )
    saved_kind = (training_state or {}).get("optimizer_kind", "adamw")
    if saved_kind not in {"adamw", "hybrid-muon"}:
        raise ValueError("unknown optimizer kind in checkpoint: " + str(saved_kind))
    optimizer_kind = args.optimizer or (saved_kind if args.resume_from else "adamw")
    if saved_kind == "hybrid-muon" and optimizer_kind != "hybrid-muon":
        raise ValueError("Hybrid-Muon checkpoints must resume with hybrid-muon; "
                         "use --init-from for weights-only AdamW initialization")
    if optimizer_kind == "hybrid-muon" and args.adamw_update_monitor:
        raise ValueError("--adamw-update-monitor is only valid for --optimizer adamw")
    saved_muon = ((training_state or {}).get("muon_hyperparameters") or {})
    muon_momentum = args.muon_momentum if args.muon_momentum is not None else saved_muon.get("momentum", 0.95)
    muon_ns_steps = args.muon_ns_steps if args.muon_ns_steps is not None else saved_muon.get("ns_steps", 5)
    muon_precision = args.muon_matmul_precision or saved_muon.get("matmul_precision", "bf16")
    if not math.isfinite(muon_momentum) or not 0 <= muon_momentum < 1:
        raise ValueError("--muon-momentum must be finite and in [0,1)")
    if not 1 <= muon_ns_steps <= 20:
        raise ValueError("--muon-ns-steps must lie in [1,20]")
    if saved_kind == "hybrid-muon" and args.resume_from and (
        (args.muon_momentum is not None and muon_momentum != saved_muon.get("momentum"))
        or (args.muon_ns_steps is not None and muon_ns_steps != saved_muon.get("ns_steps"))
        or (args.muon_matmul_precision is not None and muon_precision != saved_muon.get("matmul_precision"))
    ):
        raise ValueError("Muon hyperparameters cannot change on optimizer resume; "
                         "use --init-from to start new optimizer states")
    print("AdamW: beta1={beta1:g}, beta2={beta2:g}, eps={eps:g}".format(**adamw_hparams))
    if optimizer_kind == "hybrid-muon":
        print(f"Hybrid Muon: momentum={muon_momentum:g}, ns_steps={muon_ns_steps}, "
              f"matmul_precision={muon_precision}, RMS update scale=0.2")
        if args.resume_from and saved_kind == "adamw":
            print("WARNING: AdamW -> hybrid-Muon optimizer transition: "
                  "preserving AdamW moments only for AdamW parameters, "
                  "resetting Muon momentum to zero; optimizer dynamics will change")

    saved_weighting = (training_state or {}).get(
        "gradient_accumulation_weighting", "microbatch"
    )
    gradient_weighting = args.gradient_accumulation_weighting or (
        "supervised_tokens" if training_objective is not None and not args.resume_from
        else saved_weighting)
    if training_objective is not None and gradient_weighting != "supervised_tokens":
        raise ValueError("0080 requires --gradient-accumulation-weighting supervised_tokens")
    gradient_weighting_history = list(
        (training_state or {}).get("gradient_weighting_history", [])
    )
    if args.resume_from and gradient_weighting != saved_weighting:
        print(
            "GRADIENT OBJECTIVE CHANGED at step "
            f"{start_step}: {saved_weighting} -> {gradient_weighting}; "
            "optimizer moments preserved, not an exact training continuation"
        )
        gradient_weighting_history.append({
            "step": int(start_step),
            "from": saved_weighting,
            "to": gradient_weighting,
            "parent_checkpoint": str(args.resume_from),
        })
    print(f"Gradient accumulation weighting: {gradient_weighting}")

    saved_phase_start = (training_state or {}).get("full_history_only_start_step")
    saved_phase_end = (training_state or {}).get("full_history_only_until_step")
    if (saved_phase_start is None) != (saved_phase_end is None):
        raise ValueError("checkpoint contains an incomplete full-history-only schedule")
    full_history_only_start_step = None
    full_history_only_until_step = None
    if args.full_history_only_steps is not None:
        if args.full_history_only_steps <= 0:
            raise ValueError("--full-history-only-steps must be positive")
        if not args.resume_from:
            raise ValueError("--full-history-only-steps requires --resume-from")
        if not config.full_history_layers:
            raise ValueError("--full-history-only-steps requires full-history heads")
        if saved_phase_end is None:
            full_history_only_start_step = int(start_step)
            full_history_only_until_step = (
                full_history_only_start_step + int(args.full_history_only_steps)
            )
        else:
            assert saved_phase_start is not None
            full_history_only_start_step = int(saved_phase_start)
            full_history_only_until_step = int(saved_phase_end)
            saved_duration = (
                full_history_only_until_step - full_history_only_start_step
            )
            if saved_duration != int(args.full_history_only_steps):
                raise ValueError(
                    "--full-history-only-steps differs from the checkpointed "
                    f"schedule ({args.full_history_only_steps} != {saved_duration})"
                )
    elif saved_phase_end is not None:
        assert saved_phase_start is not None
        full_history_only_start_step = int(saved_phase_start)
        full_history_only_until_step = int(saved_phase_end)

    if full_history_only_until_step is not None:
        assert full_history_only_start_step is not None
        if int(start_step) < full_history_only_start_step:
            raise ValueError("checkpoint step precedes its full-history-only schedule")
        if int(start_step) < full_history_only_until_step:
            print(
                "Full-history-only phase: steps "
                f"{full_history_only_start_step + 1}-"
                f"{full_history_only_until_step} update only the new heads"
            )
        else:
            print(
                "Full-history-only phase already complete at step "
                f"{full_history_only_until_step}; all parameters will update"
            )

    trainer = ExtendedTrainer(
        model=model,
        train_shard_paths=[str(p) for p in train_shards],
        val_shard_paths=[str(p) for p in val_shards],
        batch_size=args.batch_size,
        seq_length=run_context_length,
        grad_accum_steps=args.grad_accum_steps,
        gradient_accumulation_weighting=gradient_weighting,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.peak_lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        val_interval=args.val_interval,
        val_steps=args.val_steps,
        val_per_source_steps=args.val_per_source_steps,
        val_source_seed=args.val_source_seed,
        val_representative_steps=args.val_representative_steps,
        val_representative_seed=args.val_representative_seed,
        val_fixed_mixed_steps=args.val_fixed_mixed_steps,
        val_fixed_mixed_seed=args.val_fixed_mixed_seed,
        save_interval=args.save_interval,
        loss_scale=1.0,
        numerical_debug=args.numerical_debug,
        profile_steps=args.profile_steps,
        train_source_shards=train_source_shards,
        val_source_shards=val_source_shards,
        source_weights=source_weights,
        eos_token_id=eos_token_id,
        progressive_growth_steps=progressive_growth_steps,
        gradient_convergence_monitor=args.gradient_convergence_monitor,
        gradient_convergence_interval=args.gradient_convergence_interval,
        gradient_convergence_points=args.gradient_convergence_points,
        gradient_convergence_groups=args.gradient_convergence_groups,
        gradient_convergence_output=args.gradient_convergence_output,
        gradient_noise_monitor=args.gradient_noise_monitor,
        gradient_noise_interval=args.gradient_noise_interval,
        gradient_noise_output=args.gradient_noise_output,
        gradient_noise_accum_points=args.gradient_noise_accum_points,
        gradient_noise_parameter_groups=args.gradient_noise_parameter_groups,
        adamw_update_monitor=args.adamw_update_monitor,
        adamw_update_output=args.adamw_update_output,
        adam_beta1=adamw_hparams["beta1"],
        adam_beta2=adamw_hparams["beta2"],
        adam_eps=adamw_hparams["eps"],
        optimizer_kind=optimizer_kind,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        muon_matmul_precision=muon_precision,
        full_history_only_start_step=full_history_only_start_step,
        full_history_only_until_step=full_history_only_until_step,
        stream_specs=stream_specs,
        stream_tokenizer=tokenizer if stream_specs else None,
        stream_segment_tokens=args.stream_segment_tokens,
        stream_reuse_batches=args.stream_reuse_batches,
        stream_cache_dir=args.stream_cache_dir,
        local_mixture_manifest_sha256=mixture_sha,
        training_objective=training_objective,
    )
    if training_objective is not None and args.resume_from:
        trainer.validate_block_proposal_resume_contract(training_state or {})
    trainer.gradient_weighting_history = gradient_weighting_history
    trainer.tokenizer_sha256 = tokenizer_sha
    trainer.data_fork = fork_provenance
    trainer.mixture_history = mixture_history
    trainer.mixture_phase_start_source_counts = phase_start_source_counts
    trainer.mixture_phase_start_batch_counts = phase_start_batch_counts
    if args.resume_from and stored_provenance:
        trainer.train_source_supervised_tokens = dict(
            (training_state or {}).get("train_source_supervised_tokens", {})
        )
        if trainer.streaming_cache is not None and stored_stream is not None:
            if args.reset_remote_cursors:
                print("  Started new remote cursors; previous stream state is "
                      "preserved in data_provenance.mixture_history")
            elif resume_stream_exactly:
                trainer.streaming_cache.load_state_dict(
                    stored_stream, checkpoint_dir=checkpoint_path,
                )
            else:
                restored = restore_compatible_stream_cursors(
                    trainer.streaming_cache, stored_stream,
                    checkpoint_dir=checkpoint_path,
                )
                if restored:
                    print("  Retained remote cursors for unchanged sources: "
                          + ", ".join(restored))
        # Preserve the cumulative sample audit chain even across mixture changes.
        if (trainer.streaming_cache is not None or training_objective is not None) and stored_provenance.get("sample_trace_sha256"):
            trainer.sample_trace_sha256 = stored_provenance["sample_trace_sha256"]
            trainer.sample_trace_microbatches = int(
                stored_provenance.get("sample_trace_microbatches", 0)
            )

    # 0078A: observational source metrics are separate from legacy counters.
    # Older checkpoints have no source_metrics key: start a new measured epoch.
    if args.resume_from and training_state:
        trainer.restore_source_metrics(training_state.get("source_metrics"))
        trainer.restore_routing_mode_stats(training_state.get("routing_mode_metrics"))

    trainer.step = start_step
    trainer.tokens_processed = tokens_processed
    if training_state and "supervised_tokens_total" in training_state:
        trainer.supervised_tokens_total = float(training_state["supervised_tokens_total"])
        trainer.supervised_tokens_provenance = str(
            training_state.get("supervised_tokens_provenance", "measured")
        )
        if (args.benchmark_supervised_tokens_offset is not None
                and abs(trainer.supervised_tokens_total - args.benchmark_supervised_tokens_offset) > 0.1):
            raise ValueError("Explicit token offset disagrees with checkpoint counter")
    elif args.benchmark_supervised_tokens_offset is not None:
        if args.benchmark_supervised_tokens_offset < 0:
            raise ValueError("benchmark token offset must be nonnegative")
        trainer.supervised_tokens_total = float(args.benchmark_supervised_tokens_offset)
        trainer.supervised_tokens_provenance = "user_supplied_legacy_offset"
    elif training_state and start_step:
        # An old checkpoint has no authoritative supervised-token total.
        # Do not silently substitute source capacity or sampled counters.
        trainer.supervised_tokens_provenance = "since_0092_only"
        print("WARNING: Legacy checkpoint lacks cumulative supervised tokens; "
              "recording newly measured tokens from zero. Pass "
              "--benchmark-supervised-tokens-offset to align absolute milestones.")
    if training_objective is not None and training_state:
        trainer._block_proposal_supervised_tokens = int(training_state.get("supervised_tokens", 0))
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
                # Keep removed-source totals for long-term provenance too.
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

        if optimizer_kind == "hybrid-muon":
            from mini_llm.optim.hybrid_muon import HybridMuonAdamW
            assert isinstance(trainer.optimizer, HybridMuonAdamW)
            if saved_kind == "hybrid-muon":
                expected_groups = set(saved_muon.get("group_names", []))
                if expected_groups != trainer.optimizer.muon_names:
                    raise ValueError("saved Muon parameter groups differ from current architecture")
            masters, adam_restored, muon_restored = trainer.optimizer.restore_checkpoint(
                stored_optimizer_state, from_hybrid=(saved_kind == "hybrid-muon"),
            )
            print(f"Restored hybrid masters={masters}, AdamW moments={adam_restored}, "
                  f"Muon momenta={muon_restored}")
            birth_state = (training_state or {}).get("optimizer_parameter_birth_steps", {})
            trainer.optimizer.restore_parameter_birth_steps(birth_state)
            stored_optimizer_state.clear()
            stored_optimizer_state = None
            optimizer_state = None
            # Match the AdamW resume path below: checkpoint tensors were
            # materialized on the GPU and can leave optimizer-sized cached or
            # fragmented pool blocks after their contents have been restored.
            get_pool = getattr(xp, "get_default_memory_pool", None)
            if get_pool is not None:
                get_pool().free_all_blocks()

        if optimizer_kind == "adamw":
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
            if training_objective is not None and restored_masters != len(optimizer_params):
                raise ValueError("0080 resume did not restore every FP32 master weight")
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
                if training_objective is not None and restored_count != len(optimizer_params):
                    raise ValueError("0080 resume did not restore every AdamW moment")
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

    if args.max_run_steps is not None and args.max_run_steps < 1:
        raise ValueError("--max-run-steps must be >= 1")

    print()
    if args.benchmark_suite:
        if training_objective is not None:
            raise ValueError("0092 live benchmark runner currently supports only AR objective")
        if tokenizer_path is None or not Path(tokenizer_path).is_file():
            raise ValueError("0092 requires an existing tokenizer file")
        from benchmarks.live_suite import LiveBenchmarkScheduler
        trainer.live_benchmark_scheduler = LiveBenchmarkScheduler.from_file(
            Path(args.benchmark_suite), tokenizer, trainer,
        )

    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    try:
        losses = trainer.train(
            num_steps=(min(args.total_steps - start_step, args.max_run_steps)
                       if args.max_run_steps is not None
                       else args.total_steps - start_step),
            log_interval=args.log_interval,
        )
        # A short invocation may finish between periodic checkpoint saves.
        # Snapshot active streamed segments before closing their backing cache.
        if args.checkpoint_dir and trainer.step % trainer.save_interval != 0:
            trainer.save()
    finally:
        if trainer.streaming_cache is not None:
            trainer.streaming_cache.close()

    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")

    if args.checkpoint_dir:
        print(f"Final checkpoint saved to: {Path(args.checkpoint_dir)}")


if __name__ == "__main__":
    # 0084B: first-class online probability fusion, using frozen teacher
    # forwards directly from mixed shards. Keep AR/BlockProposal parsing intact.
    import sys
    if len(sys.argv) > 2 and any(
        sys.argv[i] == "--objective" and sys.argv[i + 1] == "fusion"
        for i in range(1, len(sys.argv) - 1)
    ):
        from mini_llm.collective.online_training import main as fusion_main
        argv = list(sys.argv[1:])
        pos = argv.index("--objective")
        del argv[pos:pos + 2]
        fusion_main(argv)
    else:
        main()
