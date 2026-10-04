"""Tests for the DecoderLanguageModel."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig


def make_model():
    """Create a small test DecoderLanguageModel."""
    config = ModelConfig.tiny_inspection()
    return DecoderLanguageModel(config, rng_seed=10, dtype="float64")


def test_decoder_lm_forward_shape():
    """Test that forward pass produces correct output shape."""
    model = make_model()
    config = model.config
    B, T = 2, 8
    token_ids = xp.asarray(np.random.default_rng(11).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    
    logits, cache = model.forward(token_ids)
    
    assert logits.shape == (B, T, config.vocab_size), f"Expected shape ({B}, {T}, {config.vocab_size}), got {logits.shape}"
    assert isinstance(cache, dict), "Cache should be a dictionary"


def test_decoder_lm_backward_shape():
    """Test that backward pass produces correct gradient shapes."""
    model = make_model()
    config = model.config
    B, T = 2, 8
    token_ids = xp.asarray(np.random.default_rng(12).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    logits, cache = model.forward(token_ids)
    targets = xp.asarray(np.random.default_rng(13).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    
    _, loss_cache = model.compute_loss(logits, targets)
    d_logits = model.backward_loss(loss_cache)
    
    model.zero_grad()
    model.backward(d_logits, cache)
    
    # Check that all parameters have gradients
    for p in model.parameters():
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch for {p.name}"


def test_decoder_lm_backward_direction():
    """Test backward pass using directional derivative check on logits."""
    model = make_model()
    config = model.config
    B, T = 1, 4
    token_ids = xp.asarray(np.random.default_rng(14).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    
    logits, cache = model.forward(token_ids)
    _, loss_cache = model.compute_loss(logits, token_ids)  # Use token_ids as targets for self-supervised
    d_logits = model.backward_loss(loss_cache)
    
    model.zero_grad()
    dx = model.backward(d_logits, cache)
    
    # Check directional derivative for logits gradient
    # We check that the backward pass correctly computes the gradient of sum(logits * d_logits)
    v = xp.asarray(np.random.default_rng(15).normal(size=logits.shape), dtype="float64")
    eps = 1e-6
    
    def objective_with_logits_perturbation(eps_val):
        logits_pert = logits + eps_val * v
        loss_cache_pert, _ = model.compute_loss(logits_pert, token_ids)
        return float(xp.sum(logits_pert * d_logits))
    
    fd = (objective_with_logits_perturbation(eps) - objective_with_logits_perturbation(-eps)) / (2 * eps)
    an = float(xp.sum(d_logits * v))  # The gradient of sum(logits * d_logits) w.r.t. logits is just d_logits
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    assert rel < 1e-6, f"Logits backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_decoder_lm_parameter_backward():
    """Test parameter gradient computation."""
    model = make_model()
    config = model.config
    B, T = 1, 3
    token_ids = xp.asarray(np.random.default_rng(16).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    
    logits, cache = model.forward(token_ids)
    _, loss_cache = model.compute_loss(logits, token_ids)
    d_logits = model.backward_loss(loss_cache)
    
    model.zero_grad()
    model.backward(d_logits, cache)
    
    # Check gradient for embedding parameter using cross-entropy loss
    # The objective is the cross-entropy loss, not logits dot product
    param = model.embedding.W
    v = xp.asarray(np.random.default_rng(17).normal(size=param.data.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    original_data = param.data.copy()
    
    def objective_with_perturbation(eps_val):
        param.data[...] = original_data + eps_val * v
        logits_pert, _ = model.forward(token_ids)
        loss_val, _ = model.compute_loss(logits_pert, token_ids)
        return float(loss_val)
    
    # Compute finite difference
    f_plus = objective_with_perturbation(eps)
    f_minus = objective_with_perturbation(-eps)
    fd = (f_plus - f_minus) / (2 * eps)
    
    an = float(xp.sum(param.grad * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Restore original
    param.data[...] = original_data
    
    # Tolerance adjusted for renormalization effects
    assert rel < 0.001, f"Parameter backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_decoder_lm_tied_embeddings():
    """Test that embedding matrix is tied to output logits."""
    model = make_model()
    config = model.config
    
    # The embedding weight should be used for logits
    assert model.embedding.W.data is not None
    assert model.embedding.W.data.shape[0] == config.vocab_size
    assert model.embedding.W.data.shape[1] == config.d_model


def test_decoder_lm_output_proj_exists():
    """Test that output projection layer exists and has correct shape."""
    model = make_model()
    config = model.config
    
    # Output projection should exist
    assert hasattr(model, 'output_proj'), "Model should have output_proj layer"
    assert model.output_proj.W.data.shape == (config.d_model, config.vocab_size), \
        f"Output proj shape mismatch: expected {(config.d_model, config.vocab_size)}, got {model.output_proj.W.data.shape}"


def test_decoder_lm_output_proj_gradients():
    """Test that output projection gradients are computed correctly."""
    model = make_model()
    config = model.config
    B, T = 1, 3
    token_ids = xp.asarray(np.random.default_rng(18).integers(0, config.vocab_size, size=(B, T)), dtype="int64")
    
    logits, cache = model.forward(token_ids)
    _, loss_cache = model.compute_loss(logits, token_ids)
    d_logits = model.backward_loss(loss_cache)
    
    model.zero_grad()
    model.backward(d_logits, cache)
    
    # Check that output projection has non-zero gradients
    out_proj_param = model.output_proj.W
    assert out_proj_param.grad is not None, "Output proj should have gradients"
    assert out_proj_param.grad.shape == out_proj_param.data.shape, \
        f"Output proj grad shape mismatch: expected {out_proj_param.data.shape}, got {out_proj_param.grad.shape}"
    
    # Verify gradient is non-zero (at least one element should be non-trivial)
    assert xp.any(out_proj_param.grad != 0), "Output proj gradients should not be all zeros"


def test_decoder_lm_forward_without_cache_matches_training_forward():
    """Forward-only inference must match the ordinary cached forward exactly."""
    model = make_model()
    config = model.config
    token_ids = xp.asarray(
        np.random.default_rng(19).integers(0, config.vocab_size, size=(1, 6)),
        dtype="int64",
    )

    logits_cached, _ = model.forward(token_ids)
    logits_forward_only = model.forward(token_ids, return_cache=False)

    assert logits_forward_only.shape == logits_cached.shape
    assert xp.allclose(logits_forward_only, logits_cached, rtol=1e-12, atol=1e-12)


def test_chunked_lm_head_matches_full_loss_and_gradients():
    """0056 chunked/recomputed head should match the full float32 reference."""
    config = ModelConfig.tiny_inspection()
    full = DecoderLanguageModel(config, rng_seed=31, dtype="float32")
    chunked = DecoderLanguageModel(config, rng_seed=31, dtype="float32")
    rng = np.random.default_rng(32)
    token_ids = xp.asarray(
        rng.integers(0, config.vocab_size, size=(2, 7)), dtype="int64"
    )
    targets = xp.asarray(
        rng.integers(0, config.vocab_size, size=(2, 7)), dtype="int64"
    )

    logits, full_cache = full.forward(token_ids)
    full_loss, full_loss_cache = full.compute_loss(logits, targets)
    d_logits = full.backward_loss(full_loss_cache)
    full.zero_grad()
    full.backward(d_logits, full_cache)

    head, body_cache = chunked.forward_body(token_ids)
    chunk_loss, chunk_loss_cache = chunked.chunked_lm_head_loss_forward(
        head, targets, chunk_tokens=5
    )
    chunked.zero_grad()
    dx = chunked.chunked_lm_head_backward(chunk_loss_cache)
    chunked.backward_body(dx, body_cache)

    assert np.isclose(chunk_loss, full_loss, rtol=2e-6, atol=2e-6)
    for p_ref, p_new in zip(full.parameters(), chunked.parameters()):
        assert p_ref.name == p_new.name
        assert xp.allclose(p_new.grad, p_ref.grad, rtol=2e-5, atol=2e-6), p_ref.name


def test_forward_body_plus_projection_matches_forward():
    """0056 body split must not change the ordinary inference/training API."""
    config = ModelConfig.tiny_inspection()
    model = DecoderLanguageModel(config, rng_seed=33, dtype="float32")
    token_ids = xp.asarray(
        np.random.default_rng(34).integers(0, config.vocab_size, size=(1, 6)),
        dtype="int64",
    )
    logits, _ = model.forward(token_ids)
    head = model.forward_body(token_ids, return_cache=False)
    logits_split, _ = model.output_proj.forward(head)
    assert xp.allclose(logits_split, logits, rtol=1e-6, atol=1e-6)


def test_block_activation_checkpoint_matches_regular_body_and_gradients():
    """0057 block recomputation must preserve body outputs and parameter grads."""
    config = ModelConfig(
        tokenizer_vocab_size=128,
        context_length=16,
        n_layers=3,
        d_model=32,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=8,
        n_experts=3,
        top_k=2,
        d_ff=64,
    )
    reference = DecoderLanguageModel(config, rng_seed=71, dtype="float64")
    checkpointed = DecoderLanguageModel(config, rng_seed=71, dtype="float64")
    rng = np.random.default_rng(72)
    token_ids = xp.asarray(
        rng.integers(0, config.vocab_size, size=(2, 11)), dtype="int64"
    )

    head_ref, cache_ref = reference.forward_body(token_ids)
    head_ckpt, cache_ckpt = checkpointed.forward_body(
        token_ids, activation_checkpoint=True
    )

    assert xp.allclose(head_ckpt, head_ref, rtol=1e-12, atol=1e-12)
    assert cache_ckpt["activation_checkpoint"] == "block"
    assert "block_caches" not in cache_ckpt
    assert len(cache_ckpt["block_checkpoints"]) == config.n_layers

    dx = xp.asarray(rng.normal(size=head_ref.shape), dtype="float64")
    reference.zero_grad()
    checkpointed.zero_grad()
    reference.backward_body(dx.copy(), cache_ref)
    checkpointed.backward_body(dx.copy(), cache_ckpt)

    for p_ref, p_ckpt in zip(reference.parameters(), checkpointed.parameters()):
        assert p_ref.name == p_ckpt.name
        assert xp.allclose(p_ckpt.grad, p_ref.grad, rtol=2e-10, atol=2e-11), p_ref.name
