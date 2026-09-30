"""Tests for the MoE router."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.router import Router


def test_router_forward_shape():
    """Test that forward pass produces correct shapes."""
    rng = RandomStream(1)
    router = Router(4, 6, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(2).normal(size=(2, 3, 4)), dtype="float64")
    
    weights, expert_indices, cache = router.forward(x)
    
    assert weights.shape == (2, 3, 2), f"Expected weights shape (2, 3, 2), got {weights.shape}"
    assert expert_indices.shape == (2, 3, 2), f"Expected indices shape (2, 3, 2), got {expert_indices.shape}"
    assert weights.ndim == 3
    assert xp.all(weights >= 0) and xp.all(weights <= 1), "Weights should be in [0, 1]"
    # Weights are renormalized softmax probabilities for the top-k selected experts
    # They sum to 1 across the k selected experts
    weight_sums = xp.sum(weights, axis=-1)
    assert xp.allclose(weight_sums, 1.0, atol=1e-6), "Selected weights should sum to 1"


def test_router_forward_deterministic():
    """Test that forward pass is deterministic with same input."""
    rng = RandomStream(3)
    router = Router(4, 6, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(4).normal(size=(1, 2, 4)), dtype="float64")
    
    weights1, expert_indices1, _ = router.forward(x)
    weights2, expert_indices2, _ = router.forward(x)
    
    assert xp.allclose(weights1, weights2), "Forward pass should be deterministic"
    assert xp.all(expert_indices1 == expert_indices2), "Expert indices should be the same"


def test_router_backward_shape():
    """Test that backward pass produces correct gradient shapes."""
    rng = RandomStream(5)
    router = Router(4, 6, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(6).normal(size=(1, 2, 4)), dtype="float64")
    
    _, _, cache = router.forward(x)
    
    # Gradient w.r.t. output weights
    dweights = xp.asarray(np.random.default_rng(7).normal(size=(1, 2, 2)), dtype="float64")
    dx = router.backward(dweights, cache)
    
    assert dx.shape == x.shape, f"Expected gradient shape {x.shape}, got {dx.shape}"
    # Check that parameters have gradients
    for p in router.parameters():
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch for {p.name}"


def test_router_backward_direction():
    """Test backward pass using directional derivative check."""
    rng = RandomStream(8)
    router = Router(4, 6, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(9).normal(size=(1, 3, 4)), dtype="float64")
    
    weights, expert_indices, cache = router.forward(x)
    
    # Use a linear objective: sum(w * c) where c is a constant
    # This creates a simpler gradient flow that's easier to verify
    v = xp.asarray(np.random.default_rng(10).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    # Use a constant gradient direction (not dependent on x)
    c = xp.asarray([[0.5, -0.3]], dtype="float64")  # Shape (1, 2) for k=2
    
    def objective(z):
        w, _, _ = router.forward(z)
        return float(xp.sum(w * c))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    
    # Compute analytical gradient
    _, _, cache = router.forward(x)
    # Expand c to match (B, T, k) shape
    dweights = xp.tile(c, (1, 3, 1))
    dx = router.backward(dweights, cache)
    an = float(xp.sum(dx * v))
    
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Run convergence study with multiple epsilon values
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for e in epsilons:
        fd_e = (objective(x + e * v) - objective(x - e * v)) / (2 * e)
        err = abs(fd_e - an)
        errors.append((e, err))
    
    # The straight-through estimator for top-k selection introduces some error
    # but should still be reasonably close
    assert rel < 0.1, f"Backward direction check failed: fd={fd}, an={an}, rel={rel}"
    
    # Print convergence diagnostics (only visible when test fails or with -s)
    print(f"Router backward convergence: rel_error={rel:.6f}")
    for e, err in errors:
        print(f"  eps={e:.0e}, fd_err={err:.2e}")


def test_router_top_k_selection():
    """Test that top-k selection works correctly."""
    rng = RandomStream(11)
    router = Router(4, 5, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(12).normal(size=(1, 1, 4)), dtype="float64")
    
    weights, expert_indices, _ = router.forward(x)
    
    # Check that exactly k experts are selected
    assert expert_indices.shape[-1] == 2, "Should select 2 experts"
    # Check that indices are in valid range
    assert xp.all((expert_indices >= 0) & (expert_indices < 5)), "Expert indices should be in [0, n_experts)"


def test_router_parameter_gradients():
    """Test that router parameter gradients are computed correctly."""
    rng = RandomStream(13)
    router = Router(4, 6, 2, 0.1, rng, dtype="float64")
    x = xp.asarray(np.random.default_rng(14).normal(size=(1, 2, 4)), dtype="float64")
    
    weights, expert_indices, cache = router.forward(x)
    router.zero_grad()
    
    # Use a non-trivial gradient input with non-uniform values
    # to avoid the zero-gradient case of uniform dweights through softmax
    np_dweights = np.random.default_rng(15).uniform(0.1, 1.0, size=weights.shape)
    dweights = xp.asarray(np_dweights, dtype="float64")
    dx = router.backward(dweights, cache)
    
    # Check that parameter gradients are non-zero
    for p in router.parameters():
        assert xp.any(p.grad != 0), f"Gradient for {p.name} should not be zero"
