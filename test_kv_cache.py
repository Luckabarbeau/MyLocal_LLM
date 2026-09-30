#!/usr/bin/env python3
"""Test script for KV-cached inference implementation."""

import numpy as np

from mini_llm.backend import xp, synchronize
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel


def test_inference_model_basic():
    """Test basic inference model functionality."""
    print("=== Test: Basic Inference Model ===")
    
    # Create a tiny config for testing
    config = ModelConfig.tiny_inspection()
    
    # Create training model
    print("Creating training model...")
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    
    # Create inference model
    print("Creating inference model...")
    inference_model = InferenceModel(config, dtype="float32")
    
    # Set weights
    print("Setting weights from training model...")
    inference_model.set_weights(training_model)
    
    # Create generation state
    print("Creating generation state...")
    state = inference_model.create_generation_state(batch_size=1, max_length=16)
    
    # Test prefill
    print("Testing prefill...")
    input_ids = xp.asarray([[0, 1, 2, 3]], dtype=xp.int32)  # 4 tokens
    logits = inference_model.prefill(input_ids, state)
    
    print(f"Prefill output shape: {logits.shape}")
    assert logits.shape == (1, config.vocab_size), f"Expected (1, {config.vocab_size}), got {logits.shape}"
    assert state.length == 4, f"Expected length 4, got {state.length}"
    print("Prefill OK")
    
    # Test decode_one
    print("Testing decode_one...")
    for i in range(3):
        next_ids = xp.asarray([5], dtype=xp.int32)  # Single token
        logits = inference_model.decode_one(next_ids, state)
        print(f"  Decode {i+1}: output shape {logits.shape}, state length {state.length}")
        assert logits.shape == (1, config.vocab_size), f"Expected (1, {config.vocab_size}), got {logits.shape}"
        assert state.length == 5 + i, f"Expected length {5+i}, got {state.length}"
    
    print("Decode one OK")
    
    # Test reset
    print("Testing state reset...")
    state.reset()
    assert state.length == 0, f"Expected length 0 after reset, got {state.length}"
    print("Reset OK")
    
    print("\n=== All basic tests passed! ===\n")


def test_gqa_attention():
    """Test GQA attention inference implementation."""
    print("=== Test: GQA Attention Inference ===")
    
    config = ModelConfig.tiny_inspection()
    
    # Create models
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    inference_model = InferenceModel(config, dtype="float32")
    inference_model.set_weights(training_model)
    
    # Get attention layer
    att_train = training_model.blocks[0].attention
    att_inf = inference_model.blocks[0].attention
    
    print(f"Training attention: Q heads={att_train.n_q_heads}, KV heads={att_train.n_kv_heads}")
    print(f"Inference attention: Q heads={att_inf.n_q_heads}, KV heads={att_inf.n_kv_heads}")
    
    # Test with small input
    B, T, D = 1, 4, config.d_model
    x = xp.random.randn(B, T, D).astype(xp.float32)
    
    # Training forward
    y_train, _ = att_train.forward(x)
    
    # Create cache
    state = inference_model.create_generation_state(batch_size=1, max_length=16)
    
    # Inference prefill
    y_inf = att_inf.prefill(x, state.k_cache[0], state.v_cache[0], 0)
    
    print(f"Training output shape: {y_train.shape}")
    print(f"Inference output shape: {y_inf.shape}")
    
    # They should be the same (last position only)
    assert y_inf.shape == (B, D), f"Expected ({B}, {D}), got {y_inf.shape}"
    
    print("GQA attention test OK\n")


def test_cache_allocation():
    """Test that cache is preallocated and reused."""
    print("=== Test: Cache Allocation ===")
    
    config = ModelConfig.tiny_inspection()
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    inference_model = InferenceModel(config, dtype="float32")
    inference_model.set_weights(training_model)
    
    state = inference_model.create_generation_state(batch_size=1, max_length=128)
    
    # Get cache addresses before generation
    # Handle both NumPy and CuPy array interfaces
    if hasattr(state.k_cache, '__cuda_array_interface__'):
        k_cache_ptr = state.k_cache.__cuda_array_interface__['data'][0]
        v_cache_ptr = state.v_cache.__cuda_array_interface__['data'][0]
    else:
        k_cache_ptr = state.k_cache.__array_interface__['data'][0]
        v_cache_ptr = state.v_cache.__array_interface__['data'][0]
    
    # Do some generations
    input_ids = xp.asarray([[0, 1, 2]], dtype=xp.int32)
    inference_model.prefill(input_ids, state)
    
    for _ in range(5):
        next_ids = xp.asarray([5], dtype=xp.int32)
        inference_model.decode_one(next_ids, state)
    
    # Cache addresses should be the same
    if hasattr(state.k_cache, '__cuda_array_interface__'):
        k_cache_ptr_after = state.k_cache.__cuda_array_interface__['data'][0]
        v_cache_ptr_after = state.v_cache.__cuda_array_interface__['data'][0]
    else:
        k_cache_ptr_after = state.k_cache.__array_interface__['data'][0]
        v_cache_ptr_after = state.v_cache.__array_interface__['data'][0]
    
    assert k_cache_ptr == k_cache_ptr_after, "K cache was reallocated!"
    assert v_cache_ptr == v_cache_ptr_after, "V cache was reallocated!"
    
    print(f"Cache pointers match: K={k_cache_ptr}, V={v_cache_ptr}")
    print("Cache allocation test OK\n")


def test_numerical_equivalence():
    """Test that cached inference matches full prefix."""
    print("=== Test: Numerical Equivalence ===")
    
    config = ModelConfig.tiny_inspection()
    training_model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    inference_model = InferenceModel(config, dtype="float32")
    inference_model.set_weights(training_model)
    
    # Test with a 4-token sequence
    input_ids = xp.asarray([[10, 20, 30, 40]], dtype=xp.int32)
    
    # Full prefix forward (training model)
    logits_full, _ = training_model.forward(input_ids)
    logits_last_full = logits_full[:, -1, :]  # Last position only
    
    # KV-cached prefill
    state = inference_model.create_generation_state(batch_size=1, max_length=16)
    logits_cached = inference_model.prefill(input_ids, state)
    
    # Compare (last position logits)
    diff = xp.abs(logits_last_full - logits_cached)
    max_diff = xp.max(diff)
    mean_diff = xp.mean(diff)
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Should be very close (same computation)
    assert max_diff < 1e-5, f"Large difference: {max_diff}"
    
    print("Numerical equivalence test OK\n")


if __name__ == "__main__":
    test_inference_model_basic()
    test_gqa_attention()
    test_cache_allocation()
    test_numerical_equivalence()
    
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
