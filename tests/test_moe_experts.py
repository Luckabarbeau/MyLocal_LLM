"""Tests for the MoE experts."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.experts import Experts, ExpertFFN


def make_experts():
    """Create a small test Experts instance."""
    rng = RandomStream(1)
    return Experts(
        d_model=4,
        d_ff=8,
        n_experts=3,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )


def test_experts_forward_shape():
    """Test that forward pass produces correct shapes."""
    experts = make_experts()
    x = xp.asarray(np.random.default_rng(2).normal(size=(2, 3, 4)), dtype="float64")
    weights = xp.asarray(np.random.default_rng(3).normal(size=(2, 3, 2)), dtype="float64")
    weights = xp.abs(weights) / (xp.sum(xp.abs(weights), axis=-1, keepdims=True) + 1e-8)  # Normalize
    expert_indices = xp.asarray([[0, 1], [1, 2], [2, 0], [0, 1], [1, 2], [2, 0]], dtype=xp.int64).reshape(2, 3, 2)
    
    y, cache = experts.forward(x, weights, expert_indices)
    
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert isinstance(cache, dict), "Cache should be a dictionary"


def test_experts_backward_shape():
    """Test that backward pass produces correct gradient shapes."""
    experts = make_experts()
    x = xp.asarray(np.random.default_rng(4).normal(size=(1, 2, 4)), dtype="float64")
    weights = xp.asarray(np.random.default_rng(5).normal(size=(1, 2, 2)), dtype="float64")
    weights = xp.abs(weights) / (xp.sum(xp.abs(weights), axis=-1, keepdims=True) + 1e-8)
    expert_indices = xp.asarray([[0, 1], [1, 2]], dtype=xp.int64).reshape(1, 2, 2)
    
    _, cache = experts.forward(x, weights, expert_indices)
    experts.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(6).normal(size=x.shape), dtype="float64")
    dx = experts.backward(dy, cache)
    
    assert dx.shape == x.shape, f"Expected gradient shape {x.shape}, got {dx.shape}"
    # Check that all expert parameters have gradients
    for p in experts.parameters():
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch for {p.name}"


def test_experts_backward_direction():
    """Test backward pass using directional derivative check."""
    rng = RandomStream(7)
    experts = Experts(
        d_model=4,
        d_ff=8,
        n_experts=3,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(8).normal(size=(1, 3, 4)), dtype="float64")
    weights = xp.asarray([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]], dtype=xp.float64).reshape(1, 3, 2)
    expert_indices = xp.asarray([[0, 1], [1, 2], [0, 2]], dtype=xp.int64).reshape(1, 3, 2)
    
    _, cache = experts.forward(x, weights, expert_indices)
    experts.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(9).normal(size=x.shape), dtype="float64")
    dx = experts.backward(dy, cache)
    
    # Check directional derivative for input
    v = xp.asarray(np.random.default_rng(10).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    def objective(z):
        w = xp.abs(weights) / (xp.sum(xp.abs(weights), axis=-1, keepdims=True) + 1e-8)
        out, _ = experts.forward(z, w, expert_indices)
        return float(xp.sum(out * dy))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    assert rel < 1e-6, f"Input backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_experts_parameter_backward():
    """Test parameter gradient computation."""
    rng = RandomStream(11)
    experts = Experts(
        d_model=4,
        d_ff=8,
        n_experts=3,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(12).normal(size=(1, 2, 4)), dtype="float64")
    weights = xp.asarray([[0.6, 0.4], [0.3, 0.7]], dtype=xp.float64).reshape(1, 2, 2)
    expert_indices = xp.asarray([[0, 1], [1, 2]], dtype=xp.int64).reshape(1, 2, 2)
    
    _, cache = experts.forward(x, weights, expert_indices)
    experts.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(13).normal(size=x.shape), dtype="float64")
    dx = experts.backward(dy, cache)
    
    # Check gradient for first parameter
    param = experts.parameters()[0]
    v = xp.asarray(np.random.default_rng(14).normal(size=param.data.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    original_data = param.data.copy()
    
    def objective_with_perturbation(eps_val):
        param.data[...] = original_data + eps_val * v
        w = xp.abs(weights) / (xp.sum(xp.abs(weights), axis=-1, keepdims=True) + 1e-8)
        out, _ = experts.forward(x, w, expert_indices)
        return float(xp.sum(out * dy))
    
    f_plus = objective_with_perturbation(eps)
    f_minus = objective_with_perturbation(-eps)
    fd = (f_plus - f_minus) / (2 * eps)
    
    an = float(xp.sum(param.grad * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Restore original
    param.data[...] = original_data
    
    assert rel < 1e-5, f"Parameter backward direction check failed: fd={fd}, an={an}, rel={rel}"


def test_experts_zero_grad():
    """Test that zero_grad works correctly."""
    experts = make_experts()
    x = xp.asarray(np.random.default_rng(15).normal(size=(1, 2, 4)), dtype="float64")
    weights = xp.asarray([[0.6, 0.4], [0.3, 0.7]], dtype=xp.float64).reshape(1, 2, 2)
    expert_indices = xp.asarray([[0, 1], [1, 2]], dtype=xp.int64).reshape(1, 2, 2)
    
    _, cache = experts.forward(x, weights, expert_indices)
    dy = xp.asarray(np.random.default_rng(16).normal(size=x.shape), dtype="float64")
    dx = experts.backward(dy, cache)
    
    # Check that gradients are non-zero
    for p in experts.parameters():
        assert xp.any(p.grad != 0), "Gradients should be non-zero after backward"
    
    experts.zero_grad()
    
    # Check that gradients are zero
    for p in experts.parameters():
        assert xp.all(p.grad == 0), "Gradients should be zero after zero_grad"


def test_experts_multiple_selections():
    """Test that expert can be selected multiple times."""
    rng = RandomStream(17)
    experts = Experts(
        d_model=4,
        d_ff=8,
        n_experts=2,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(18).normal(size=(1, 3, 4)), dtype="float64")
    # All positions select expert 0
    weights = xp.ones((1, 3, 1), dtype=xp.float64)
    expert_indices = xp.zeros((1, 3, 1), dtype=xp.int64)
    
    y, cache = experts.forward(x, weights, expert_indices)
    
    # All outputs should be from expert 0
    assert y.shape == x.shape


def test_experts_forward_without_cache_matches_cached_forward():
    experts = make_experts()
    x = xp.asarray(np.random.default_rng(22).normal(size=(1, 4, 4)), dtype="float64")
    weights = xp.asarray(
        [[[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.5, 0.5]]],
        dtype="float64",
    )
    expert_indices = xp.asarray(
        [[[0, 1], [1, 2], [2, 0], [0, 2]]], dtype=xp.int64
    )

    y_cached, _ = experts.forward(x, weights, expert_indices)
    y_forward_only = experts.forward(
        x, weights, expert_indices, return_cache=False
    )

    assert xp.allclose(y_forward_only, y_cached, rtol=1e-12, atol=1e-12)


def test_expert_ffn_training_cache_recomputes_elementwise_intermediates():
    rng = RandomStream(23)
    expert = ExpertFFN(
        d_model=4,
        d_ff=8,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64",
    )
    x = xp.asarray(np.random.default_rng(24).normal(size=(5, 4)), dtype="float64")

    y, cache = expert.forward(x)

    assert set(cache) == {"x", "g", "u"}
    assert "a" not in cache
    assert "h" not in cache

    dy = xp.asarray(np.random.default_rng(25).normal(size=y.shape), dtype="float64")
    expert.zero_grad()
    dx = expert.backward(dy, cache)
    assert dx.shape == x.shape
    assert xp.all(xp.isfinite(dx))
