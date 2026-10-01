#!/usr/bin/env python3
"""Performance benchmark for training and inference.

This module provides coarse timing measurements for:
- Training: forward, loss, backward, gradient norm/clipping, optimizer, total step, tokens/sec
- Inference: prefill, decode_one, sampling, decode ms/token, tokens/sec

GPU operations are asynchronous, so timing synchronizes correctly using xp.cuda.Stream.null.synchronize().

Usage:
    python benchmarks/benchmark_performance.py --mode both --model medium --iterations 50 --warmup 10
"""

import argparse
import time
import numpy as np
from pathlib import Path

from mini_llm.backend import xp, synchronize, asnumpy, BACKEND_NAME
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from mini_llm.train import MiniTrainer
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm
from mini_llm.optim.schedule import WarmupCosineSchedule
from mini_llm.data.token_shards import create_minibatch


# Configuration for the medium model as specified in the task
MEDIUM_MODEL_CONFIG = {
    "d_model": 512,
    "n_layers": 8,
    "context_length": 512,
    "batch_size": 8,
    "experts": 6,
    "top_k": 2,
    "dtype": "float16",
}


def generate_synthetic_shard(num_docs: int = 100, context_length: int = 512) -> np.ndarray:
    """Generate synthetic token data for benchmarking."""
    # Generate random tokens
    tokens = xp.random.randint(0, 8192, size=(num_docs, context_length), dtype=xp.uint16)
    return asnumpy(tokens)


def create_dummy_shard_file(path: str, num_docs: int = 100, context_length: int = 512, vocab_size: int = 16384):
    """Create a dummy shard file for benchmarking."""
    import struct
    
    # Shard context length needs to be at least seq_length + 1 for shifted targets
    # We use a larger context to ensure we have enough tokens
    actual_context = context_length + 16  # Add buffer
    tokens = xp.random.randint(0, vocab_size, size=(num_docs, actual_context), dtype=xp.uint16)
    
    with open(path, "wb") as f:
        # Write header: (num_docs, actual_context) as int64 (little-endian)
        f.write(struct.pack("<qq", num_docs, actual_context))
        # Write data as uint16
        f.write(tokens.tobytes())


def benchmark_training(
    model: DecoderLanguageModel,
    shard_path: str,
    batch_size: int,
    seq_length: int,
    iterations: int,
    warmup_iterations: int,
) -> dict:
    """
    Benchmark training performance with fine-grained timing.
    
    Measures:
        - forward
        - loss
        - backward
        - gradient norm/clipping
        - optimizer
        - total step
        - tokens/sec
    
    Args:
        model: DecoderLanguageModel instance
        shard_path: Path to token shard file
        batch_size: Batch size
        seq_length: Sequence length
        iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations before benchmarking
        
    Returns:
        Dictionary with timing statistics
    """
    print("\n" + "=" * 70)
    print("TRAINING BENCHMARK")
    print("=" * 70)
    
    # Load shard data
    from mini_llm.train import load_token_shard
    shard_data = load_token_shard(shard_path)
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-4)
    # Use large warmup to ensure we're in the linear ramp during benchmark
    total_training_steps = 10000
    scheduler = WarmupCosineSchedule(
        peak_lr=3e-4,
        warmup_steps=1000,
        total_steps=total_training_steps,
    )
    
    # Statistics containers
    stats = {
        "forward": [],
        "loss": [],
        "backward": [],
        "grad_norm": [],
        "optimizer": [],
        "total_step": [],
        "tokens_per_second": [],
    }
    
    print(f"\nConfiguration:")
    print(f"  Model: {model.config.d_model}d, {model.config.n_layers} layers")
    print(f"  Context: {model.config.context_length} (shard)")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Backend: {BACKEND_NAME}")
    print(f"  dtype: {model.dtype}")
    print(f"  Parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"\nBenchmarking:")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Benchmark iterations: {iterations}")
    
    # Warmup phase - don't measure these
    print("\nRunning warmup iterations...")
    for step in range(warmup_iterations):
        lr = scheduler(step)
        optimizer.lr = lr
        
        inputs, targets = create_minibatch(shard_data, batch_size, seq_length)
        
        # Forward pass
        logits, cache = model.forward(inputs)
        
        # Loss
        loss, loss_cache = model.compute_loss(logits, targets)
        
        # Backward pass
        d_logits = model.backward_loss(loss_cache)
        model.backward(d_logits, cache)
        
        # Gradient clipping
        clip_grad_global_norm(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step(lr=lr)
        optimizer.zero_grad()
    
    print("Warmup complete. Starting benchmark...\n")
    
    # Benchmark phase
    for step in range(iterations):
        lr = scheduler(warmup_iterations + step)
        optimizer.lr = lr
        
        inputs, targets = create_minibatch(shard_data, batch_size, seq_length)
        
        # Timing start
        synchronize()
        t0 = time.perf_counter()
        
        # Forward pass
        logits, cache = model.forward(inputs)
        synchronize()
        forward_time = time.perf_counter() - t0
        
        # Loss computation
        t0 = time.perf_counter()
        loss, loss_cache = model.compute_loss(logits, targets)
        synchronize()
        loss_time = time.perf_counter() - t0
        
        # Backward pass
        t0 = time.perf_counter()
        d_logits = model.backward_loss(loss_cache)
        model.backward(d_logits, cache)
        synchronize()
        backward_time = time.perf_counter() - t0
        
        # Gradient clipping
        t0 = time.perf_counter()
        grad_norm = clip_grad_global_norm(model.parameters(), max_norm=1.0)
        synchronize()
        grad_norm_time = time.perf_counter() - t0
        
        # Optimizer step
        t0 = time.perf_counter()
        optimizer.step(lr=lr)
        optimizer.zero_grad()
        synchronize()
        optimizer_time = time.perf_counter() - t0
        
        # Total step time
        total_step_time = forward_time + loss_time + backward_time + grad_norm_time + optimizer_time
        
        # Tokens per second (total tokens processed per second)
        tokens_per_sec = (batch_size * seq_length) / total_step_time
        
        # Record statistics
        stats["forward"].append(forward_time)
        stats["loss"].append(loss_time)
        stats["backward"].append(backward_time)
        stats["grad_norm"].append(grad_norm_time)
        stats["optimizer"].append(optimizer_time)
        stats["total_step"].append(total_step_time)
        stats["tokens_per_second"].append(tokens_per_sec)
    
    # Compute statistics
    def compute_stats(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    results: dict = {k: compute_stats(v) for k, v in stats.items()}
    # Add tokens_per_sec separately (this is a scalar, not a stats dict)
    results["tokens_per_sec"] = float(np.mean(stats["tokens_per_second"]))
    
    # Print results
    print("\n" + "-" * 70)
    print("TRAINING BENCHMARK RESULTS")
    print("-" * 70)
    
    print(f"\nForward pass:")
    print(f"  Mean: {results['forward']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['forward']['std']*1000:.2f}ms")
    print(f"  Min:  {results['forward']['min']*1000:.2f}ms")
    print(f"  Max:  {results['forward']['max']*1000:.2f}ms")
    
    print(f"\nLoss computation:")
    print(f"  Mean: {results['loss']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['loss']['std']*1000:.2f}ms")
    print(f"  Min:  {results['loss']['min']*1000:.2f}ms")
    print(f"  Max:  {results['loss']['max']*1000:.2f}ms")
    
    print(f"\nBackward pass:")
    print(f"  Mean: {results['backward']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['backward']['std']*1000:.2f}ms")
    print(f"  Min:  {results['backward']['min']*1000:.2f}ms")
    print(f"  Max:  {results['backward']['max']*1000:.2f}ms")
    
    print(f"\nGradient norm/clipping:")
    print(f"  Mean: {results['grad_norm']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['grad_norm']['std']*1000:.2f}ms")
    print(f"  Min:  {results['grad_norm']['min']*1000:.2f}ms")
    print(f"  Max:  {results['grad_norm']['max']*1000:.2f}ms")
    
    print(f"\nOptimizer step:")
    print(f"  Mean: {results['optimizer']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['optimizer']['std']*1000:.2f}ms")
    print(f"  Min:  {results['optimizer']['min']*1000:.2f}ms")
    print(f"  Max:  {results['optimizer']['max']*1000:.2f}ms")
    
    print(f"\nTotal step:")
    print(f"  Mean: {results['total_step']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['total_step']['std']*1000:.2f}ms")
    print(f"  Min:  {results['total_step']['min']*1000:.2f}ms")
    print(f"  Max:  {results['total_step']['max']*1000:.2f}ms")
    
    print(f"\nTokens per second:")
    print(f"  Mean: {results['tokens_per_sec']:.1f} tokens/sec")
    
    print("\n" + "=" * 70)
    
    return results


def benchmark_inference(
    inference_model: InferenceModel,
    max_length: int = 512,
    prompt_length: int = 64,
    max_new_tokens: int = 32,
    iterations: int = 50,
    warmup_iterations: int = 10,
) -> dict:
    """
    Benchmark inference performance with fine-grained timing.
    
    Measures:
        - prefill
        - decode_one
        - sampling
        - decode ms/token
        - tokens/sec
    
    Args:
        inference_model: InferenceModel instance
        max_length: Maximum sequence length
        prompt_length: Length of prompt for prefill
        max_new_tokens: Number of tokens to generate in decode phase
        iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations before benchmarking
        
    Returns:
        Dictionary with timing statistics
    """
    print("\n" + "=" * 70)
    print("INFERENCE BENCHMARK")
    print("=" * 70)
    
    batch_size = 1
    
    # Statistics containers
    stats = {
        "prefill": [],
        "decode_one": [],
        "sampling": [],
    }
    
    # Create generation state
    state = inference_model.create_generation_state(batch_size=batch_size, max_length=max_length)
    
    # Generate random input for benchmarking
    np.random.seed(42)
    prompt_ids = xp.random.randint(0, inference_model.vocab_size, size=(batch_size, prompt_length), dtype=xp.int32)
    
    # Limit prompt length to available context
    actual_prompt_len = min(prompt_length, max_length - max_new_tokens - 1)
    
    print(f"\nConfiguration:")
    print(f"  Model: {inference_model.d_model}d, {inference_model.n_layers} layers")
    print(f"  Vocab size: {inference_model.vocab_size}")
    print(f"  Context: {max_length}")
    print(f"  Prompt length: {prompt_length}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Backend: {BACKEND_NAME}")
    print(f"  dtype: {inference_model.dtype}")
    print(f"\nBenchmarking:")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Benchmark iterations: {iterations}")
    
    # Warmup phase - don't measure these
    print("\nRunning warmup iterations...")
    for _ in range(warmup_iterations):
        warmup_state = inference_model.create_generation_state(batch_size=batch_size, max_length=max_length)
        inference_model.prefill(prompt_ids, warmup_state)
        synchronize()
        for _ in range(max_new_tokens):
            logits = inference_model.decode_one(prompt_ids[:, :1], warmup_state)
            synchronize()
            # Sampling (simple greedy decode for timing)
            t0 = time.perf_counter()
            next_id = xp.argmax(logits, axis=-1)
            synchronize()
            sampling_time = time.perf_counter() - t0
    print("Warmup complete. Starting benchmark...\n")
    
    # Benchmark phase
    for _ in range(iterations):
        # Prefill timing
        prefill_state = inference_model.create_generation_state(batch_size=batch_size, max_length=max_length)
        
        synchronize()
        t0 = time.perf_counter()
        inference_model.prefill(prompt_ids, prefill_state)
        synchronize()
        prefill_time = time.perf_counter() - t0
        
        # Decode timing
        decode_times = []
        sampling_times = []
        
        for _ in range(max_new_tokens):
            # Decode one token
            synchronize()
            t0 = time.perf_counter()
            logits = inference_model.decode_one(prompt_ids[:, :1], prefill_state)
            synchronize()
            decode_time = time.perf_counter() - t0
            
            # Sampling (greedy decode)
            t0 = time.perf_counter()
            next_id = xp.argmax(logits, axis=-1)
            synchronize()
            sampling_time = time.perf_counter() - t0
            
            decode_times.append(decode_time)
            sampling_times.append(sampling_time)
        
        stats["prefill"].append(prefill_time)
        stats["decode_one"].append(np.mean(decode_times))
        stats["sampling"].append(np.mean(sampling_times))
    
    # Compute statistics
    def compute_stats(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    results = {k: compute_stats(v) for k, v in stats.items()}
    
    # Total decode time (sum of means times max_new_tokens)
    total_decode_per_token = results["decode_one"]["mean"] + results["sampling"]["mean"]
    total_decode_mean = total_decode_per_token * max_new_tokens
    
    # Compute overall metrics
    total_prefill_mean = results["prefill"]["mean"]
    total_time_mean = total_prefill_mean + total_decode_mean
    total_tokens = prompt_length + max_new_tokens
    tokens_per_sec = total_tokens / total_time_mean
    
    # Print results
    print("\n" + "-" * 70)
    print("INFERENCE BENCHMARK RESULTS")
    print("-" * 70)
    
    print(f"\nPrefill:")
    print(f"  Mean: {results['prefill']['mean']*1000:.2f}ms")
    print(f"  Std:  {results['prefill']['std']*1000:.2f}ms")
    print(f"  Min:  {results['prefill']['min']*1000:.2f}ms")
    print(f"  Max:  {results['prefill']['max']*1000:.2f}ms")
    
    print(f"\nDecode one token:")
    print(f"  Mean: {results['decode_one']['mean']*1000:.4f}ms")
    print(f"  Std:  {results['decode_one']['std']*1000:.4f}ms")
    print(f"  Min:  {results['decode_one']['min']*1000:.4f}ms")
    print(f"  Max:  {results['decode_one']['max']*1000:.4f}ms")
    
    print(f"\nSampling (greedy decode):")
    print(f"  Mean: {results['sampling']['mean']*1000:.4f}ms")
    print(f"  Std:  {results['sampling']['std']*1000:.4f}ms")
    print(f"  Min:  {results['sampling']['min']*1000:.4f}ms")
    print(f"  Max:  {results['sampling']['max']*1000:.4f}ms")
    
    print(f"\nOverall decode (including prefill):")
    print(f"  Total time: {total_time_mean*1000:.2f}ms")
    print(f"  Decode ms/token: {(total_decode_mean/max_new_tokens)*1000:.4f}ms")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    
    print("\n" + "=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both",
                        help="Which benchmark to run")
    parser.add_argument("--model", choices=["micro", "tiny", "mini", "medium", "small"], default="micro",
                        help="Model size configuration")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for JSON results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create dummy shard but don't run full benchmark")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nBackend: {BACKEND_NAME}")
    
    # Create temporary directory for shard
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    shard_path = str(temp_dir / "benchmark_shard.bin")
    
    try:
        # Generate configuration based on model size
        if args.model == "micro":
            config = ModelConfig.micro_debug()
            batch_size = 4
            seq_length = 32  # Smaller context for micro model
        elif args.model == "tiny":
            config = ModelConfig.tiny_inspection()
            batch_size = 2
            seq_length = 16
        elif args.model == "medium":
            config = ModelConfig.medium()
            batch_size = 8
            seq_length = 512
        elif args.model == "mini":
            config = ModelConfig.mini()
            batch_size = 8
            seq_length = 512
        else:  # small
            config = ModelConfig.small()
            batch_size = 4  # Reduced for memory efficiency
            seq_length = 256  # Reduced context to fit in GPU memory
        
        print(f"\nModel configuration ({args.model}):")
        print(f"  d_model: {config.d_model}")
        print(f"  n_layers: {config.n_layers}")
        print(f"  context_length: {config.context_length}")
        print(f"  batch_size: {batch_size}")
        print(f"  experts: {config.n_experts}")
        print(f"  top_k: {config.top_k}")
        print(f"  dtype: {config.dtype}")
        print(f"  seq_length (training): {seq_length}")
        
        # Create synthetic shard with correct vocab size
        print("\nGenerating synthetic token shard...")
        create_dummy_shard_file(shard_path, num_docs=100, context_length=config.context_length, vocab_size=config.vocab_size)
        print(f"  Created: {shard_path}")
        
        if args.dry_run:
            print("\nDry run complete. Exiting.")
            return 0
        
        # Create model
        print(f"\nCreating model...")
        model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
        print(f"  Parameters: {sum(p.data.size for p in model.parameters()):,}")
        
        # Initialize results to None (only set when mode allows)
        train_results: dict | None = None
        inference_results: dict | None = None
        
        if args.mode in ["train", "both"]:
            train_results = benchmark_training(
                model=model,
                shard_path=shard_path,
                batch_size=batch_size,
                seq_length=seq_length,
                iterations=args.iterations,
                warmup_iterations=args.warmup,
            )
        
        # For inference, we need a fresh inference model
        print(f"\nCreating inference model...")
        inference_model = InferenceModel(config, dtype=config.dtype)
        inference_model.set_weights(model)
        
        if args.mode in ["inference", "both"]:
            # Use smaller prompt for micro/tiny models
            if args.model in ["micro", "tiny"]:
                prompt_len = min(32, config.context_length - 10)
                max_new = min(16, config.context_length - prompt_len - 1)
            else:
                prompt_len = 64
                max_new = 32
            inference_results = benchmark_inference(
                inference_model=inference_model,
                max_length=config.context_length,
                prompt_length=prompt_len,
                max_new_tokens=max_new,
                iterations=args.iterations,
                warmup_iterations=args.warmup,
            )
        
        # Output JSON if requested
        if args.output:
            import json
            results = {
                "model": args.model,
                "iterations": args.iterations,
                "warmup": args.warmup,
            }
            if args.mode in ["train", "both"]:
                results["training"] = train_results
            if args.mode in ["inference", "both"]:
                results["inference"] = inference_results
            
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
