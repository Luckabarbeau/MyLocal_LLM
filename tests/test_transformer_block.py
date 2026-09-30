"""Tests for the Transformer block."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.transformer_block import TransformerBlock


def make_block():
    """Create a small test TransformerBlock."""
    rng = RandomStream(10)
    return TransformerBlock(
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        d_ff=16,
        n_experts=3,
        top_k=2,
        input_std=0.08,
        output_std=0.04,
        rope_base=10_000.0,
        rng=rng,
        dtype="float64"
    )


def test_transformer_block_forward_shape():
    """Test that forward pass preserves shape."""
    block = make_block()
    x = xp.asarray(np.random.default_rng(11).normal(size=(2, 3, 8)), dtype="float64")
    
    y, cache = block.forward(x)
    
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert isinstance(cache, dict), "Cache should be a dictionary"


def test_transformer_block_backward_shape():
    """Test that backward pass produces correct gradient shapes."""
    block = make_block()
    x = xp.asarray(np.random.default_rng(12).normal(size=(2, 3, 8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(13).normal(size=(2, 3, 8)), dtype="float64")
    
    _, cache = block.forward(x)
    block.zero_grad()
    dx = block.backward(dy, cache)
    
    assert dx.shape == x.shape, f"Expected gradient shape {x.shape}, got {dx.shape}"
    
    # Check that all parameters have gradients
    for p in block.parameters():
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch for {p.name}"


def test_transformer_block_backward_direction():
    """Test backward pass using directional derivative check."""
    block = make_block()
    x = xp.asarray(np.random.default_rng(14).normal(size=(1, 4, 8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(15).normal(size=(1, 4, 8)), dtype="float64")
    
    _, cache = block.forward(x)
    block.zero_grad()
    dx = block.backward(dy, cache)
    
    # Check directional derivative for input
    v = xp.asarray(np.random.default_rng(16).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    def objective(z):
        out, _ = block.forward(z)
        return float(xp.sum(out * dy))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    # Tolerance adjusted for numerical stability
    # Tolerance adjusted for sparse expert dispatch
    assert rel < 1e-3, f"Input backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_transformer_block_parameter_backward():
    """Test parameter gradient computation."""
    block = make_block()
    x = xp.asarray(np.random.default_rng(17).normal(size=(1, 3, 8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(18).normal(size=(1, 3, 8)), dtype="float64")
    
    _, cache = block.forward(x)
    block.zero_grad()
    block.backward(dy, cache)
    
    # Check gradient for first parameter
    param = block.parameters()[0]
    v = xp.asarray(np.random.default_rng(19).normal(size=param.data.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    original_data = param.data.copy()
    
    def objective_with_perturbation(eps_val):
        param.data[...] = original_data + eps_val * v
        out, _ = block.forward(x)
        return float(xp.sum(out * dy))
    
    # Restore and compute finite difference
    f_plus = objective_with_perturbation(eps)
    f_minus = objective_with_perturbation(-eps)
    fd = (f_plus - f_minus) / (2 * eps)
    
    an = float(xp.sum(param.grad * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Restore original
    param.data[...] = original_data
    
    assert rel < 2e-5, f"Parameter backward direction check failed: fd={fd}, an={an}, rel={rel}"
