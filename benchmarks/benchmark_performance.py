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


def _sample_generated_token(
    logits,
    strategy: str = "greedy",
    top_k: int = 40,
    temperature: float = 0.8,
) -> int:
    """Sample one token using production-like host semantics.

    The returned Python ``int`` intentionally introduces the same device->host
    scalar dependency that the real generation loop has.  This helper is used
    only by the end-to-end generation diagnostic, not by pure model timing.
    """
    flat_logits = logits.reshape(-1)

    if strategy == "greedy":
        token = xp.argmax(flat_logits)
        return int(asnumpy(token).reshape(()))

    if strategy != "top-k":
        raise ValueError(f"Unsupported inference benchmark strategy: {strategy}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for top-k sampling")

    vocab_size = int(flat_logits.shape[0])
    k = max(1, min(int(top_k), vocab_size))

    if k == vocab_size:
        indices = xp.arange(vocab_size, dtype=xp.int32)
    else:
        kth = vocab_size - k
        indices = xp.argpartition(flat_logits, kth)[kth:]

    # Only the selected logits need FP32 softmax-like work.
    top_logits = flat_logits[indices].astype(xp.float32, copy=True)
    top_logits /= float(temperature)
    top_logits -= xp.max(top_logits)
    xp.exp(top_logits, out=top_logits)

    cdf = xp.cumsum(top_logits)
    draw = xp.random.random() * cdf[-1]
    selected_slot = xp.searchsorted(cdf, draw, side="right")
    token = indices[selected_slot]
    return int(asnumpy(token).reshape(()))


def benchmark_decode_position_scaling(
    inference_model: InferenceModel,
    max_length: int,
    positions: list[int],
    iterations: int = 5,
    warmup_iterations: int = 1,
) -> list[dict]:
    """Measure one pure ``decode_one`` call at known KV-cache lengths.

    Prefill is performed before the timed region.  The reported cache position
    is the number of tokens already present in the KV cache when ``decode_one``
    begins.  Sampling is deliberately excluded.
    """
    print("\n" + "-" * 70)
    print("DECODE POSITION SCALING (MODEL ONLY)")
    print("-" * 70)
    print("  Prefill/setup is excluded from each timed decode.")
    print("  Sampling is excluded; each decode consumes a fixed token id.")
    print()
    print(f"{'cache length':>12}  {'mean ms/token':>14}  {'std ms':>10}  {'tok/s':>10}")

    fixed_token = xp.asarray([[4]], dtype=xp.int32)
    results: list[dict] = []

    for requested_position in positions:
        position = int(requested_position)
        if position < 1 or position >= max_length:
            print(f"{position:>12}  {'SKIP':>14}  {'-':>10}  {'-':>10}")
            continue

        prompt = xp.random.randint(
            0,
            inference_model.vocab_size,
            size=(1, position),
            dtype=xp.int32,
        )

        for _ in range(max(0, warmup_iterations)):
            state = inference_model.create_generation_state(batch_size=1, max_length=max_length)
            inference_model.prefill(prompt, state)
            synchronize()
            inference_model.decode_one(fixed_token, state)
            synchronize()

        samples = []
        for _ in range(max(1, iterations)):
            state = inference_model.create_generation_state(batch_size=1, max_length=max_length)
            inference_model.prefill(prompt, state)
            synchronize()

            t0 = time.perf_counter()
            inference_model.decode_one(fixed_token, state)
            synchronize()
            samples.append(time.perf_counter() - t0)

        arr = np.asarray(samples, dtype=np.float64)
        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr))
        tok_s = 1.0 / mean_s if mean_s > 0 else float("inf")
        result = {
            "cache_length": position,
            "mean_seconds": mean_s,
            "std_seconds": std_s,
            "tokens_per_second": tok_s,
        }
        results.append(result)
        print(f"{position:12d}  {mean_s * 1000:14.4f}  {std_s * 1000:10.4f}  {tok_s:10.1f}")

    return results


def benchmark_full_context_model_decode(
    inference_model: InferenceModel,
    max_length: int,
    prompt_length: int = 4,
    iterations: int = 3,
    warmup_iterations: int = 1,
) -> dict:
    """Measure pure model decode while sweeping from a short prompt to max context.

    The prompt is prefetched before timing.  A fixed token is repeatedly sent to
    ``decode_one`` so sampling cannot hide model/KV-cache scaling.  This sweeps
    cache lengths from ``prompt_length`` through ``max_length - 1``.
    """
    prompt_length = max(1, min(int(prompt_length), max_length - 1))
    decode_calls = max_length - prompt_length
    prompt = xp.random.randint(
        0,
        inference_model.vocab_size,
        size=(1, prompt_length),
        dtype=xp.int32,
    )
    fixed_token = xp.asarray([[4]], dtype=xp.int32)

    def run_once() -> float:
        state = inference_model.create_generation_state(batch_size=1, max_length=max_length)
        inference_model.prefill(prompt, state)
        synchronize()

        t0 = time.perf_counter()
        for _ in range(decode_calls):
            inference_model.decode_one(fixed_token, state)
        synchronize()
        return time.perf_counter() - t0

    for _ in range(max(0, warmup_iterations)):
        run_once()

    elapsed = np.asarray([run_once() for _ in range(max(1, iterations))], dtype=np.float64)
    mean_s = float(np.mean(elapsed))
    std_s = float(np.std(elapsed))
    ms_per_token = (mean_s / decode_calls) * 1000.0
    tok_s = decode_calls / mean_s

    result = {
        "prompt_length": prompt_length,
        "decode_calls": decode_calls,
        "mean_seconds": mean_s,
        "std_seconds": std_s,
        "ms_per_token": ms_per_token,
        "tokens_per_second": tok_s,
    }

    print("\n" + "-" * 70)
    print("FULL-CONTEXT MODEL-ONLY DECODE SWEEP")
    print("-" * 70)
    print(f"  Cache range: {prompt_length} -> {max_length - 1}")
    print(f"  Decode calls: {decode_calls}")
    print(f"  Mean total: {mean_s * 1000:.2f}ms")
    print(f"  Mean decode: {ms_per_token:.4f}ms/token")
    print(f"  Pure model decode: {tok_s:.1f} tokens/sec")

    return result


def benchmark_full_context_generation(
    inference_model: InferenceModel,
    max_length: int,
    prompt_length: int = 4,
    iterations: int = 3,
    warmup_iterations: int = 1,
    strategy: str = "greedy",
    top_k: int = 40,
    temperature: float = 0.8,
) -> dict:
    """Benchmark a production-like autoregressive loop to context capacity.

    Prompt prefill is excluded from generation timing, matching ``inference.py``.
    The first generated token is sampled from prefill logits.  Therefore a run
    generating ``max_length - prompt_length`` tokens performs one fewer
    ``decode_one`` calls, just like the production loop.
    """
    prompt_length = max(1, min(int(prompt_length), max_length - 1))
    generated_tokens = max_length - prompt_length
    decode_calls = max(0, generated_tokens - 1)
    prompt = xp.random.randint(
        0,
        inference_model.vocab_size,
        size=(1, prompt_length),
        dtype=xp.int32,
    )

    def run_once() -> float:
        state = inference_model.create_generation_state(batch_size=1, max_length=max_length)
        logits = inference_model.prefill(prompt, state)
        synchronize()

        t0 = time.perf_counter()
        for step in range(generated_tokens):
            next_id = _sample_generated_token(
                logits,
                strategy=strategy,
                top_k=top_k,
                temperature=temperature,
            )
            if step == generated_tokens - 1:
                break
            next_token = xp.asarray([[next_id]], dtype=xp.int32)
            logits = inference_model.decode_one(next_token, state)
        synchronize()
        return time.perf_counter() - t0

    for _ in range(max(0, warmup_iterations)):
        run_once()

    elapsed = np.asarray([run_once() for _ in range(max(1, iterations))], dtype=np.float64)
    mean_s = float(np.mean(elapsed))
    std_s = float(np.std(elapsed))
    ms_per_generated = (mean_s / generated_tokens) * 1000.0
    tok_s = generated_tokens / mean_s

    result = {
        "prompt_length": prompt_length,
        "generated_tokens": generated_tokens,
        "decode_calls": decode_calls,
        "strategy": strategy,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "mean_seconds": mean_s,
        "std_seconds": std_s,
        "ms_per_generated_token": ms_per_generated,
        "generated_tokens_per_second": tok_s,
    }

    print("\n" + "-" * 70)
    print("FULL-CONTEXT PRODUCTION-LIKE GENERATION")
    print("-" * 70)
    print(f"  Prompt length: {prompt_length}")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  decode_one calls: {decode_calls}")
    print(f"  Sampling: {strategy}" + (f" (k={top_k}, T={temperature})" if strategy == "top-k" else ""))
    print(f"  Mean total generation: {mean_s * 1000:.2f}ms")
    print(f"  Mean generation: {ms_per_generated:.4f}ms/token")
    print(f"  Generated-token throughput: {tok_s:.1f} tokens/sec")

    return result


def benchmark_inference(
    inference_model: InferenceModel,
    max_length: int = 512,
    prompt_length: int = 64,
    max_new_tokens: int = 32,
    iterations: int = 50,
    warmup_iterations: int = 10,
) -> dict:
    """Benchmark short-window inference with unambiguous throughput metrics.

    This benchmark intentionally keeps the historical 64-token prefill + short
    decode window, but it no longer counts prompt tokens as generated tokens.
    It reports prompt-processing throughput, pure decode throughput, decode plus
    sampling throughput, and request-level generated-token throughput separately.
    """
    print("\n" + "=" * 70)
    print("INFERENCE BENCHMARK")
    print("=" * 70)

    batch_size = 1
    max_new_tokens = max(1, min(int(max_new_tokens), max_length - 1))
    prompt_length = max(1, min(int(prompt_length), max_length - max_new_tokens))

    stats = {
        "prefill": [],
        "decode_one": [],
        "sampling": [],
    }

    prompt_ids = xp.random.randint(
        0,
        inference_model.vocab_size,
        size=(batch_size, prompt_length),
        dtype=xp.int32,
    )
    fixed_token = prompt_ids[:, :1]

    print(f"\nConfiguration:")
    print(f"  Model: {inference_model.d_model}d, {inference_model.n_layers} layers")
    print(f"  Vocab size: {inference_model.vocab_size}")
    print(f"  Context: {max_length}")
    print(f"  Prompt length: {prompt_length}")
    print(f"  Decode window: {max_new_tokens} tokens")
    print(f"  Backend: {BACKEND_NAME}")
    print(f"  dtype: {inference_model.dtype}")
    print(f"\nBenchmarking:")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Benchmark iterations: {iterations}")

    print("\nRunning warmup iterations...")
    for _ in range(warmup_iterations):
        warmup_state = inference_model.create_generation_state(batch_size=batch_size, max_length=max_length)
        inference_model.prefill(prompt_ids, warmup_state)
        synchronize()
        for _ in range(max_new_tokens):
            logits = inference_model.decode_one(fixed_token, warmup_state)
            synchronize()
            xp.argmax(logits, axis=-1)
            synchronize()
    print("Warmup complete. Starting benchmark...\n")

    for _ in range(iterations):
        state = inference_model.create_generation_state(batch_size=batch_size, max_length=max_length)

        synchronize()
        t0 = time.perf_counter()
        inference_model.prefill(prompt_ids, state)
        synchronize()
        prefill_time = time.perf_counter() - t0

        decode_times = []
        sampling_times = []
        for _ in range(max_new_tokens):
            synchronize()
            t0 = time.perf_counter()
            logits = inference_model.decode_one(fixed_token, state)
            synchronize()
            decode_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            xp.argmax(logits, axis=-1)
            synchronize()
            sampling_times.append(time.perf_counter() - t0)

        stats["prefill"].append(prefill_time)
        stats["decode_one"].append(float(np.mean(decode_times)))
        stats["sampling"].append(float(np.mean(sampling_times)))

    def compute_stats(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    results = {k: compute_stats(v) for k, v in stats.items()}

    decode_model_s = results["decode_one"]["mean"]
    sampling_s = results["sampling"]["mean"]
    decode_plus_sampling_s = decode_model_s + sampling_s
    total_decode_s = decode_plus_sampling_s * max_new_tokens
    prefill_s = results["prefill"]["mean"]
    total_request_s = prefill_s + total_decode_s

    results["metrics"] = {
        "prefill_tokens_per_second": prompt_length / prefill_s,
        "model_decode_tokens_per_second": 1.0 / decode_model_s,
        "decode_plus_sampling_tokens_per_second": 1.0 / decode_plus_sampling_s,
        "request_generated_tokens_per_second_including_prefill": max_new_tokens / total_request_s,
        "decode_ms_per_token": decode_model_s * 1000.0,
        "decode_plus_sampling_ms_per_token": decode_plus_sampling_s * 1000.0,
    }

    print("\n" + "-" * 70)
    print("INFERENCE BENCHMARK RESULTS")
    print("-" * 70)

    print(f"\nPrefill ({prompt_length} prompt tokens in parallel):")
    print(f"  Mean: {prefill_s * 1000:.2f}ms")
    print(f"  Prompt-processing throughput: {results['metrics']['prefill_tokens_per_second']:.1f} tokens/sec")

    print(f"\nDecode one token (model only, cache ~{prompt_length}..{prompt_length + max_new_tokens}):")
    print(f"  Mean: {decode_model_s * 1000:.4f}ms")
    print(f"  Pure model decode: {results['metrics']['model_decode_tokens_per_second']:.1f} tokens/sec")

    print(f"\nSampling (greedy argmax):")
    print(f"  Mean: {sampling_s * 1000:.4f}ms")

    print(f"\nAutoregressive decode + greedy sampling:")
    print(f"  Mean: {decode_plus_sampling_s * 1000:.4f}ms/generated token")
    print(f"  Generated-token throughput: {results['metrics']['decode_plus_sampling_tokens_per_second']:.1f} tokens/sec")

    print(f"\nWhole request including prefill (generated tokens only in numerator):")
    print(f"  Total mean: {total_request_s * 1000:.2f}ms")
    print(f"  Generated-token throughput: {results['metrics']['request_generated_tokens_per_second_including_prefill']:.1f} tokens/sec")
    print("  NOTE: prompt tokens are NOT counted as generated tokens.")

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
    parser.add_argument(
        "--decode-scaling",
        action="store_true",
        help=(
            "Run diagnostic decode scaling: fixed-position decode timings plus "
            "full-context model-only and production-like generation sweeps"
        ),
    )
    parser.add_argument(
        "--decode-positions",
        type=str,
        default="32,64,128,256,384,500",
        help="Comma-separated KV-cache lengths for --decode-scaling",
    )
    parser.add_argument(
        "--position-iterations",
        type=int,
        default=5,
        help="Measured iterations at each decode position",
    )
    parser.add_argument(
        "--full-context-prompt-length",
        type=int,
        default=4,
        help="Prompt length for the full-context decode/generation sweep",
    )
    parser.add_argument(
        "--inference-strategy",
        choices=["greedy", "top-k"],
        default="top-k",
        help="Sampling used by the production-like full-context generation diagnostic",
    )
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k value for --inference-strategy top-k")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for --inference-strategy top-k")
    
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

            if args.decode_scaling:
                try:
                    decode_positions = [
                        int(value.strip())
                        for value in args.decode_positions.split(",")
                        if value.strip()
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid --decode-positions value: {args.decode_positions!r}"
                    ) from exc

                # Keep only valid positions, preserving user order and removing duplicates.
                seen_positions = set()
                decode_positions = [
                    pos for pos in decode_positions
                    if 1 <= pos < config.context_length
                    and not (pos in seen_positions or seen_positions.add(pos))
                ]

                position_results = benchmark_decode_position_scaling(
                    inference_model=inference_model,
                    max_length=config.context_length,
                    positions=decode_positions,
                    iterations=args.position_iterations,
                    warmup_iterations=min(args.warmup, 2),
                )
                full_model_results = benchmark_full_context_model_decode(
                    inference_model=inference_model,
                    max_length=config.context_length,
                    prompt_length=args.full_context_prompt_length,
                    iterations=max(1, min(args.position_iterations, 5)),
                    warmup_iterations=1,
                )
                full_generation_results = benchmark_full_context_generation(
                    inference_model=inference_model,
                    max_length=config.context_length,
                    prompt_length=args.full_context_prompt_length,
                    iterations=max(1, min(args.position_iterations, 5)),
                    warmup_iterations=1,
                    strategy=args.inference_strategy,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

                inference_results["decode_position_scaling"] = position_results
                inference_results["full_context_model_decode"] = full_model_results
                inference_results["full_context_generation"] = full_generation_results
        
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
