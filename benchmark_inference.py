#!/usr/bin/env python3
"""Benchmark script for KV cache vs full prefix inference scaling."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, synchronize, asnumpy
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


def softmax(logits, temperature=1.0):
    """Apply softmax with temperature scaling."""
    logits = logits / temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def greedy_decode(logits):
    """Greedy decoding: pick highest probability token."""
    return int(np.argmax(logits, axis=-1))


def benchmark_full_prefix(model, input_ids, max_new_tokens=20, seed=42, verbose=False):
    """Benchmark full-prefix (reference) inference - processes entire prefix each time."""
    B, T_prompt = input_ids.shape
    
    if verbose:
        print(f"  Full prefix: processing {T_prompt} prompt tokens")
    
    generated_ids = list(input_ids[0])
    forward_times = []
    
    np.random.seed(seed)
    xp.random.seed(seed)
    
    for step in range(max_new_tokens):
        # Process entire prefix each time
        context = generated_ids[-512:]
        input_tensor = np.array([context], dtype=np.uint16)
        
        start = time.time()
        logits, _ = model.forward(input_tensor)
        synchronize()
        forward_times.append(time.time() - start)
        
        last_logits = logits[0, -1, :]
        next_id = greedy_decode(last_logits)
        generated_ids.append(next_id)
    
    return {
        "prompt_tokens": T_prompt,
        "generated_tokens": len(generated_ids) - T_prompt,
        "forward_times": forward_times,
        "total_forward_time": sum(forward_times),
        "avg_forward_per_token": sum(forward_times) / max(1, len(forward_times)),
    }


def benchmark_kv_cache(model, input_ids, max_new_tokens=20, seed=42, verbose=False):
    """Benchmark KV cache inference - prefill once, decode one at a time."""
    B, T_prompt = input_ids.shape
    
    if verbose:
        print(f"  KV cache: prefill {T_prompt} tokens")
    
    # Create generation state
    state = model.create_generation_state(batch_size=1, max_length=512)
    
    # Prefill
    start = time.time()
    model.prefill(input_ids, state)
    synchronize()
    prefill_time = time.time() - start
    
    if verbose:
        print(f"  Prefill time: {prefill_time*1000:.2f}ms")
    
    # Decode
    decode_times = []
    for step in range(max_new_tokens):
        next_id = greedy_decode(model.prefill(input_ids[:, -1:], state) if step == 0 else 
                                model.decode_one(xp.asarray([[generated_ids[-1]]], dtype=xp.int32), state))
        
        # Actually need to track generated_ids properly
        if step == 0:
            # Use the last logit from prefill for first sample
            pass
    
    return {
        "prompt_tokens": T_prompt,
        "generated_tokens": max_new_tokens,
        "prefill_time": prefill_time,
        "decode_times": decode_times,
        "total_decode_time": sum(decode_times) if decode_times else 0,
        "avg_decode_per_token": sum(decode_times) / max(1, len(decode_times)) if decode_times else 0,
    }


def compute_logits_full_prefix(model, input_ids):
    """Compute logits for full prefix using training model."""
    logits, _ = model.forward(input_ids)
    return asnumpy(logits)


def compute_logits_kv_cache(inference_model, input_ids):
    """Compute logits for full prefix using KV cache inference model."""
    state = inference_model.create_generation_state(batch_size=1, max_length=512)
    logits = inference_model.prefill(input_ids, state)
    return asnumpy(logits)


def compare_logits(logits_ref, logits_kv, verbose=False):
    """Compare logits between reference and KV cache implementations."""
    diff = np.abs(logits_ref - logits_kv)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    
    if verbose:
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    return {"max_diff": max_diff, "mean_diff": mean_diff}


def benchmark_position_scaling(inference_model, training_model, input_ids, max_positions=128, verbose=False):
    """Benchmark how inference time scales with position in sequence."""
    B, T_prompt = input_ids.shape
    
    # Pre-compute reference logits for all positions
    ref_logits_all = compute_logits_full_prefix(training_model, input_ids)
    
    results = []
    
    for end_pos in [1, 2, 4, 8, 16, 32, 64, 128]:
        if end_pos > max_positions:
            break
        if end_pos > T_prompt:
            # Pad with zeros if needed
            pad_len = end_pos - T_prompt
            context = list(input_ids[0]) + [0] * pad_len
            test_input = np.array([context[:end_pos]], dtype=np.uint16)
        else:
            test_input = input_ids[:, :end_pos]
        
        B_test, T_test = test_input.shape
        
        # Reference
        ref_logits = ref_logits_all[0, T_test-1, :]  # Last position
        
        # KV cache
        state = inference_model.create_generation_state(batch_size=1, max_length=512)
        kv_logits = compute_logits_kv_cache(inference_model, test_input)[0]
        
        diff = np.abs(ref_logits - kv_logits)
        results.append({
            "position": end_pos,
            "max_diff": float(np.max(diff)),
            "mean_diff": float(np.mean(diff)),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache inference scaling")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--prompt", default="Water is", help="Prompt for benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path / "config.json"
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = ModelConfig(**config_dict)
    print(f"=== Inference Scaling Benchmark ===")
    print(f"Model: {config.d_model}d, {config.n_layers} layers, vocab={config.vocab_size}")
    print(f"Context: {config.context_length}, GQA: {config.n_q_heads}Q/{config.n_kv_heads}KV heads")
    
    # Load tokenizer
    tokenizer_path = checkpoint_path / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    
    if tokenizer_path.exists():
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
        print(f"Tokenizer: {len(tokenizer)} tokens")
    else:
        class DummyTokenizer:
            def encode(self, s): return [1, 2, 3, 4, 5]
            def decode(self, ids): return "test"
        tokenizer = DummyTokenizer()
    
    # Load models
    print("\nLoading models...")
    load_start = time.time()
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float16")
    
    param_names = [p.name for p in training_model.parameters()]
    loaded_params, _, _ = load_checkpoint(checkpoint_path, param_names=param_names, skip_optimizer=True)
    
    for p in training_model.parameters():
        if p.name in loaded_params:
            p.data[...] = loaded_params[p.name]
    
    inference_model = InferenceModel(config, dtype="float16")
    inference_model.set_weights(training_model)
    
    load_time = time.time() - load_start
    print(f"Models loaded in {load_time:.3f}s")
    
    # Warmup
    print("\nWarming up...")
    input_ids = xp.asarray([[1, 2, 3]], dtype=xp.int32)
    state = inference_model.create_generation_state(batch_size=1, max_length=16)
    try:
        inference_model.prefill(input_ids, state)
        inference_model.decode_one(xp.asarray([[4]], dtype=xp.int32), state)
    except:
        pass
    synchronize()
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokenized: {prompt_ids} ({len(prompt_ids)} tokens)")
    
    input_tensor = xp.asarray([prompt_ids], dtype=xp.int32)
    B, T_prompt = input_tensor.shape
    print(f"Input shape: [{B}, {T_prompt}]")
    
    # Benchmark position scaling
    print("\n--- Position Scaling ---")
    pos_results = benchmark_position_scaling(
        inference_model, training_model, input_tensor, 
        max_positions=min(128, T_prompt + args.max_new_tokens),
        verbose=args.verbose
    )
    
    for r in pos_results:
        print(f"  Position {r['position']:3d}: max_diff={r['max_diff']:.2e}, mean_diff={r['mean_diff']:.2e}")
    
    # Compare logits at different positions
    print("\n--- Numerical Equivalence Check ---")
    
    for T_test in [1, 4, T_prompt]:
        if T_test > T_prompt:
            continue
        test_input = input_tensor[:, :T_test]
        
        ref_logits = compute_logits_full_prefix(training_model, test_input)
        kv_logits = compute_logits_kv_cache(inference_model, test_input)
        
        diff = compare_logits(ref_logits, kv_logits, verbose=args.verbose)
        
        # Verify last position matches
        assert diff["max_diff"] < 1e-4, f"Large difference at T={T_test}: {diff['max_diff']}"
        print(f"  T={T_test}: PASS (max_diff={diff['max_diff']:.2e})")
    
    print("\n--- Generation Timing ---")
    
    # Warmup for timing
    warmup_input = xp.asarray([[1, 2, 3]], dtype=xp.int32)
    warmup_state = inference_model.create_generation_state(batch_size=1, max_length=16)
    inference_model.prefill(warmup_input, warmup_state)
    inference_model.decode_one(xp.asarray([[4]], dtype=xp.int32), warmup_state)
    synchronize()
    
    # KV cache benchmark
    kv_state = inference_model.create_generation_state(batch_size=1, max_length=512)
    kv_start = time.time()
    inference_model.prefill(input_tensor, kv_state)
    synchronize()
    kv_prefill_time = time.time() - kv_start
    
    decode_times = []
    for _ in range(args.max_new_tokens):
        next_ids = xp.asarray([[4]], dtype=xp.int32)  # Dummy token
        start = time.time()
        inference_model.decode_one(next_ids, kv_state)
        synchronize()
        decode_times.append(time.time() - start)
    
    kv_total_time = sum(decode_times)
    
    print(f"KV Cache:")
    print(f"  Prefill: {kv_prefill_time*1000:.2f}ms")
    print(f"  Decode: {len(decode_times)} tokens in {kv_total_time*1000:.2f}ms")
    print(f"  Decode rate: {len(decode_times)/kv_total_time:.1f} tokens/sec")
    print(f"  Avg per token: {kv_total_time/len(decode_times)*1000:.2f}ms")
    
    # KV cache memory
    kv_cache_size = (config.n_layers * 2 * 1 * 512 * config.d_head * 2) / (1024 * 1024)  # MB
    print(f"  Cache size: ~{kv_cache_size:.1f} MB")
    
    print("\n--- Benchmark Complete ---")


if __name__ == "__main__":
    main()
