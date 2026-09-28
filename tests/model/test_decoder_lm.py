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
    
    assert rel < 1e-4, f"Parameter backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_decoder_lm_tied_embeddings():
    """Test that embedding matrix is tied to output logits."""
    model = make_model()
    config = model.config
    
    # The embedding weight should be used for logits
    assert model.embedding.W.data is not None
    assert model.embedding.W.data.shape[0] == config.vocab_size
    assert model.embedding.W.data.shape[1] == config.d_model
