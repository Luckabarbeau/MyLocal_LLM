"""Tests for the MoE block."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.moe_block import MoE


def make_moe():
    """Create a small test MoE instance."""
    rng = RandomStream(1)
    return MoE(
        d_model=4,
        d_ff=8,
        n_experts=3,
        k=2,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )


def test_moe_forward_shape():
    """Test that forward pass produces correct shapes."""
    moe = make_moe()
    x = xp.asarray(np.random.default_rng(2).normal(size=(2, 3, 4)), dtype="float64")
    
    y, cache = moe.forward(x)
    
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert isinstance(cache, dict), "Cache should be a dictionary"


def test_moe_backward_shape():
    """Test that backward pass produces correct gradient shapes."""
    moe = make_moe()
    x = xp.asarray(np.random.default_rng(3).normal(size=(1, 2, 4)), dtype="float64")
    
    _, cache = moe.forward(x)
    moe.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(4).normal(size=x.shape), dtype="float64")
    dx = moe.backward(dy, cache)
    
    assert dx.shape == x.shape, f"Expected gradient shape {x.shape}, got {dx.shape}"
    # Check that all parameters have gradients
    for p in moe.parameters():
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch for {p.name}"


def test_moe_backward_direction():
    """Test backward pass using directional derivative check."""
    rng = RandomStream(5)
    moe = MoE(
        d_model=4,
        d_ff=8,
        n_experts=3,
        k=2,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(6).normal(size=(1, 3, 4)), dtype="float64")
    
    _, cache = moe.forward(x)
    moe.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(7).normal(size=x.shape), dtype="float64")
    dx = moe.backward(dy, cache)
    
    # Check directional derivative for input
    v = xp.asarray(np.random.default_rng(8).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    def objective(z):
        out, _ = moe.forward(z)
        return float(xp.sum(out * dy))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Run convergence study with multiple epsilon values
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for e in epsilons:
        fd_e = (objective(x + e * v) - objective(x - e * v)) / (2 * e)
        err = abs(fd_e - an)
        errors.append((e, err))
    
    # Analytical gradient should match finite difference very closely.
    # Away from top-k boundaries, the selected path is smooth and should check accurately.
    # Target relative error is on the order of 1e-5 to 1e-6 for float64.
    assert rel < 1e-3, f"Input backward direction check failed: fd={fd}, an={an}, rel={rel}"
    
    # Print convergence diagnostics (only visible when test fails or with -s)
    print(f"MoE block input backward convergence: rel_error={rel:.6f}")
    for e, err in errors:
        print(f"  eps={e:.0e}, fd_err={err:.2e}")


def test_moe_parameter_backward():
    """Test parameter gradient computation."""
    rng = RandomStream(9)
    moe = MoE(
        d_model=4,
        d_ff=8,
        n_experts=3,
        k=2,
        input_std=0.1,
        output_std=0.05,
        rng=rng,
        dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(10).normal(size=(1, 2, 4)), dtype="float64")
    
    _, cache = moe.forward(x)
    moe.zero_grad()
    
    dy = xp.asarray(np.random.default_rng(11).normal(size=x.shape), dtype="float64")
    dx = moe.backward(dy, cache)
    
    # Check gradient for first parameter (router.W_router_param)
    param = moe.parameters()[0]
    v = xp.asarray(np.random.default_rng(12).normal(size=param.data.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    original_data = param.data.copy()
    
    def objective_with_perturbation(eps_val):
        param.data[...] = original_data + eps_val * v
        out, _ = moe.forward(x)
        return float(xp.sum(out * dy))
    
    f_plus = objective_with_perturbation(eps)
    f_minus = objective_with_perturbation(-eps)
    fd = (f_plus - f_minus) / (2 * eps)
    
    an = float(xp.sum(param.grad * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    # Run convergence study with multiple epsilon values
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for e in epsilons:
        f_plus_e = objective_with_perturbation(e)
        f_minus_e = objective_with_perturbation(-e)
        fd_e = (f_plus_e - f_minus_e) / (2 * e)
        err = abs(fd_e - an)
        errors.append((e, err))
    
    # Restore original
    param.data[...] = original_data
    
    # Analytical gradient should match finite difference very closely.
    # Away from top-k boundaries, the selected path is smooth and should check accurately.
    assert rel < 1e-3, f"Parameter backward direction check failed: fd={fd}, an={an}, rel={rel}"
    
    # Print convergence diagnostics (only visible when test fails or with -s)
    print(f"MoE block parameter backward convergence: rel_error={rel:.6f}")
    for e, err in errors:
        print(f"  eps={e:.0e}, fd_err={err:.2e}")


def test_moe_zero_grad():
    """Test that zero_grad works correctly."""
    moe = make_moe()
    x = xp.asarray(np.random.default_rng(13).normal(size=(1, 2, 4)), dtype="float64")
    
    _, cache = moe.forward(x)
    dy = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
    dx = moe.backward(dy, cache)
    
    # Check that at least some gradients are non-zero (some experts might not be selected)
    total_grad_norm = 0.0
    for p in moe.parameters():
        total_grad_norm += float(xp.sqrt(xp.sum(p.grad**2)))
    assert total_grad_norm > 0, "At least some gradients should be non-zero after backward"
    
    moe.zero_grad()
    
    # Check that all gradients are zero after zero_grad
    for p in moe.parameters():
        assert xp.all(p.grad == 0), f"Gradients should be zero after zero_grad for {p.name}"


def test_moe_parameters_count():
    """Test that MoE has correct number of parameters."""
    moe = make_moe()
    params = moe.parameters()
    
    # Router: 2 parameters (W_router, b_router)
    # Experts: 3 experts * 3 parameters each (W_gate, W_up, W_down) = 9
    # Total: 2 + 9 = 11
    assert len(params) == 11, f"Expected 11 parameters, got {len(params)}"
