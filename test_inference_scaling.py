#!/usr/bin/env python3
"""Simple test for KV cache inference scaling."""

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from mini_llm.backend import xp, synchronize, asnumpy


def softmax(logits, temperature=1.0):
    """Apply softmax with temperature scaling."""
    logits = logits / temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def greedy_decode(logits):
    """Greedy decoding: pick highest probability token."""
    return int(np.argmax(logits, axis=-1))


def main():
    print("=== KV Cache Inference Scaling Test ===\n")
    
    # Use tiny_inspection config for testing
    config = ModelConfig.tiny_inspection()
    print(f"Model: {config.d_model}d, {config.n_layers} layers, vocab={config.vocab_size}")
    print(f"Context: {config.context_length}, GQA: {config.n_q_heads}Q/{config.n_kv_heads}KV heads")
    print()
    
    # Create models
    print("Creating models...")
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    inference_model = InferenceModel(config, dtype="float32")
    inference_model.set_weights(training_model)
    print("Models created.\n")
    
    # Test 1: Numerical equivalence at different positions
    print("--- Test 1: Numerical Equivalence ---")
    
    for T_test in [1, 4, 8]:
        input_ids = xp.arange(T_test, dtype=xp.int32)[None, :]
        
        ref_logits, _ = training_model.forward(input_ids)
        ref_last = ref_logits[0, -1, :]
        
        state = inference_model.create_generation_state(batch_size=1, max_length=16)
        kv_logits = inference_model.prefill(input_ids, state)
        kv_last = kv_logits[0]
        
        diff = np.abs(asnumpy(ref_last) - asnumpy(kv_last))
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        status = "PASS" if max_diff < 1e-4 else "FAIL"
        print(f"  T={T_test:2d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} [{status}]")
    
    print()
    
    # Test 2: Generation scaling
    print("--- Test 2: Generation Scaling ---")
    
    # Warmup
    warmup_ids = xp.asarray([[1, 2, 3]], dtype=xp.int32)
    warmup_state = inference_model.create_generation_state(batch_size=1, max_length=16)
    inference_model.prefill(warmup_ids, warmup_state)
    inference_model.decode_one(xp.asarray([[4]], dtype=xp.int32), warmup_state)
    synchronize()
    
    # Run benchmark
    prompt_ids = xp.arange(4, dtype=xp.int32)[None, :]  # 4 token prompt
    state = inference_model.create_generation_state(batch_size=1, max_length=32)  # 32 tokens capacity
    
    # Prefill
    import time
    prefill_start = time.time()
    inference_model.prefill(prompt_ids, state)
    synchronize()
    prefill_time = time.time() - prefill_start
    
    # Decode
    decode_times = []
    for _ in range(20):
        token = xp.asarray([[4]], dtype=xp.int32)
        start = time.time()
        inference_model.decode_one(token, state)
        synchronize()
        decode_times.append(time.time() - start)
    
    print(f"  Prefill: 4 tokens in {prefill_time*1000:.2f}ms ({4/prefill_time:.1f} tokens/sec)")
    print(f"  Decode:  20 tokens in {sum(decode_times)*1000:.2f}ms ({20/sum(decode_times):.1f} tokens/sec)")
    print(f"  Avg decode: {np.mean(decode_times)*1000:.2f}ms/token")
    
    # Test 3: Position scaling
    print("\n--- Test 3: Position Scaling ---")
    
    for pos in [1, 4, 8]:
        input_ids = xp.arange(pos, dtype=xp.int32)[None, :]
        
        ref_logits, _ = training_model.forward(input_ids)
        ref_last = ref_logits[0, -1, :]
        
        state = inference_model.create_generation_state(batch_size=1, max_length=16)
        kv_logits = inference_model.prefill(input_ids, state)
        kv_last = kv_logits[0]
        
        diff = np.abs(asnumpy(ref_last) - asnumpy(kv_last))
        print(f"  Position {pos:3d}: max_diff={np.max(diff):.2e}, mean_diff={np.mean(diff):.2e}")
    
    # Test 4: Reset functionality
    print("\n--- Test 4: State Reset ---")
    
    state = inference_model.create_generation_state(batch_size=1, max_length=32)
    input_ids = xp.arange(4, dtype=xp.int32)[None, :]
    
    inference_model.prefill(input_ids, state)
    print(f"  After prefill: length={state.length}")
    
    inference_model.decode_one(xp.asarray([[4]], dtype=xp.int32), state)
    print(f"  After decode 1: length={state.length}")
    
    state.reset()
    print(f"  After reset: length={state.length}")
    
    k_cache_ptr = state.k_cache.__array_interface__['data'][0]
    v_cache_ptr = state.v_cache.__array_interface__['data'][0]
    
    inference_model.prefill(input_ids, state)
    k_cache_ptr_after = state.k_cache.__array_interface__['data'][0]
    v_cache_ptr_after = state.v_cache.__array_interface__['data'][0]
    
    cache_unchanged = (k_cache_ptr == k_cache_ptr_after) and (v_cache_ptr == v_cache_ptr_after)
    print(f"  Cache unchanged after reset: {cache_unchanged}")
    
    print("\n=== All Tests Complete ===")


if __name__ == "__main__":
    main()
