#!/usr/bin/env python3
"""Debug training script for quick verification."""

import argparse
import json
from pathlib import Path

import numpy as np

from mini_llm.backend import xp
from mini_llm.config import ModelConfig
from mini_llm.data.packed_dataset import PackedDatasetGenerator, DatasetManifest
from mini_llm.data.token_shards import load_token_shard, create_minibatch
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.train_extended import ExtendedTrainer


def create_synthetic_packed_dataset(output_dir: str = "./debug_shards", num_docs: int = 100):
    """Create a tiny synthetic packed dataset for debugging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Use existing small tokenizer if available
    tokenizer_path = output_dir / "tokenizer.json"
    
    if tokenizer_path.exists():
        from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
    else:
        # Create a simple character-level tokenizer for debug
        from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
        tokenizer = SimpleBPETokenizer(vocab_size=256)
        
        # Generate some synthetic text
        texts = []
        for i in range(num_docs):
            doc = f"document number {i} with some sample text content here. " * 10
            texts.append(doc)
        
        tokenizer.train(texts[:50])  # Train on subset
        tokenizer.save(str(tokenizer_path))
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Get token IDs
    eos_id = tokenizer.token_to_id.get(tokenizer.eos_token, 0)
    
    # Generate all documents
    all_texts = []
    for i in range(num_docs):
        doc = f"document number {i} with some sample text content here. " * 20
        all_texts.append(doc)
    
    # Tokenize completely (no truncation)
    all_token_ids = []
    for text in all_texts:
        ids = tokenizer.encode(text)
        if ids and ids[-1] != eos_id:
            ids.append(eos_id)
        all_token_ids.extend(ids)
    
    print(f"Total tokens: {len(all_token_ids)}")
    
    # Write as packed shard
    shard_path = output_dir / "train_shard_00000.bin"
    arr = np.array(all_token_ids, dtype=np.uint16)
    arr.tofile(shard_path)
    
    # Create manifest
    manifest = DatasetManifest(
        format_version="1.0",
        tokenizer_hash="debug_hash",
        vocab_size=len(tokenizer),
        token_dtype="uint16",
        eos_token_id=eos_id,
        total_train_tokens=len(all_token_ids),
        total_val_tokens=0,
        train_document_count=num_docs,
        val_document_count=0,
        train_shard_count=1,
        val_shard_count=0,
        source_dataset_name="synthetic_debug",
        preprocessing_seed=42,
        context_length=512,
    )
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    print(f"Created debug dataset at {output_dir}")
    return str(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug training with micro model")
    
    parser.add_argument(
        "--model",
        choices=["micro_debug", "tiny_inspection"],
        default="micro_debug",
        help="Model size configuration",
    )
    
    parser.add_argument(
        "--dataset-path",
        default="./debug_shards",
        help="Path to debug dataset (auto-created if doesn't exist)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step",
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=32,
        help="Sequence length for training",
    )
    
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50,
        help="Total training steps",
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for learning rate",
    )
    
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=1e-3,
        help="Peak learning rate",
    )
    
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints/debug_micro",
        help="Directory for saving checkpoints",
    )
    
    parser.add_argument(
        "--log-file",
        default="./logs/debug_micro.csv",
        help="CSV log file path",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Debug Training with Micro Model")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    
    # Create synthetic dataset if needed
    if not Path(args.dataset_path).exists():
        print(f"\nCreating synthetic debug dataset at {args.dataset_path}...")
        args.dataset_path = create_synthetic_packed_dataset(
            output_dir=args.dataset_path,
            num_docs=50,
        )
    
    # Load or create model
    if args.model == "micro_debug":
        config = ModelConfig.micro_debug()
    else:
        config = ModelConfig.tiny_inspection()
    
    print(f"\nCreating {args.model} model...")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  context_length: {config.context_length}")
    
    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype="float16")
    
    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    trainable_params = sum(p.data.size for p in params if p.decay)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Find shard files
    shard_paths = sorted(Path(args.dataset_path).glob("train_shard_*.bin"))
    
    if not shard_paths:
        raise FileNotFoundError(f"No train shards found in {args.dataset_path}")
    
    val_shard_paths = shard_paths[:1]  # Use same for validation (just for debug)
    
    print(f"\nUsing shards:")
    for p in shard_paths:
        print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")
    
    # Create trainer
    trainer = ExtendedTrainer(
        model=model,
        train_shard_paths=[str(p) for p in shard_paths],
        val_shard_paths=[str(p) for p in val_shard_paths],
        batch_size=args.batch_size,
        seq_length=args.context_length,
        grad_accum_steps=1,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.peak_lr,
        grad_clip=args.grad_clip,
        weight_decay=0.1,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        val_interval=25,
        val_steps=2,
        save_interval=25,
        loss_scale=1.0,
        rng_seed=args.seed,
    )
    
    print()
    print("=" * 60)
    print("Starting Debug Training")
    print("=" * 60)
    print(f"  Steps: {args.total_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.context_length}")
    print(f"  Learning rate: {args.peak_lr}")
    print()
    
    # Train
    losses = trainer.train(num_steps=args.total_steps, log_interval=10)
    
    print()
    print("=" * 60)
    print("Debug Training Complete!")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Average loss: {np.mean(losses):.4f}")
    print(f"  Checkpoint saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
