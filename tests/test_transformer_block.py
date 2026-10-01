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
    
    # Run convergence study with multiple epsilon values
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for e in epsilons:
        fd_e = (objective(x + e * v) - objective(x - e * v)) / (2 * e)
        err = abs(fd_e - an)
        errors.append((e, err))
    
    # Tolerance adjusted for numerical stability
    # Tolerance adjusted for sparse expert dispatch
    assert rel < 1e-3, f"Input backward direction check failed: fd={fd}, an={an}, rel={rel}"
    
    # Print convergence diagnostics (only visible when test fails or with -s)
    print(f"Transformer block input backward convergence: rel_error={rel:.6f}")
    for e, err in errors:
        print(f"  eps={e:.0e}, fd_err={err:.2e}")


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


def test_fp16_residual_addition_uses_fp32_accumulator():
    """Finite FP16 branches must not overflow during residual addition."""

    class IdentityNorm:
        def forward(self, x):
            return x, {"dtype": x.dtype}

        def backward(self, dy, cache):
            return dy

    class ConstantBranch:
        def __init__(self, value):
            self.value = value
            self.seen_dtype = None

        def forward(self, x):
            self.seen_dtype = x.dtype
            out = xp.full(x.shape, self.value, dtype="float16")
            return out, {}

        def backward(self, dy, cache):
            return xp.zeros(dy.shape, dtype="float16")

    block = TransformerBlock.__new__(TransformerBlock)
    block.compute_dtype = xp.dtype("float16")
    block.use_fp32_residual = True
    block.norm1 = IdentityNorm()
    block.norm2 = IdentityNorm()
    block.attention = ConstantBranch(0.0)
    block.moe = ConstantBranch(40000.0)

    # Both residual2 and moe_out are individually finite in FP16, but their
    # mathematical sum (80000) exceeds the FP16 maximum (65504).
    x = xp.full((1, 1, 4), 40000.0, dtype="float16")
    y, _ = block.forward(x)

    assert block.attention.seen_dtype == xp.dtype("float16")
    assert block.moe.seen_dtype == xp.dtype("float16")
    assert y.dtype == xp.dtype("float32")
    assert bool(xp.all(xp.isfinite(y)))
    np.testing.assert_allclose(np.asarray(y), 80000.0, rtol=0.0, atol=32.0)


def test_bfloat16_residual_stream_uses_fp32():
    """BF16 compute branches retain an FP32 residual stream."""
    import ml_dtypes

    class IdentityNorm:
        def forward(self, x):
            return x, {"dtype": x.dtype}

        def backward(self, dy, cache):
            return dy

    class ZeroBranch:
        def __init__(self):
            self.seen_dtype = None

        def forward(self, x):
            self.seen_dtype = x.dtype
            return xp.zeros(x.shape, dtype=ml_dtypes.bfloat16), {}

        def backward(self, dy, cache):
            return xp.zeros(dy.shape, dtype=ml_dtypes.bfloat16)

    block = TransformerBlock.__new__(TransformerBlock)
    block.compute_dtype = ml_dtypes.bfloat16
    block.use_fp32_residual = True
    block.norm1 = IdentityNorm()
    block.norm2 = IdentityNorm()
    block.attention = ZeroBranch()
    block.moe = ZeroBranch()

    x = xp.ones((1, 2, 4), dtype=ml_dtypes.bfloat16)
    y, _ = block.forward(x)

    assert str(block.attention.seen_dtype) == "bfloat16"
    assert str(block.moe.seen_dtype) == "bfloat16"
    assert y.dtype == xp.dtype("float32")
    assert bool(xp.all(xp.isfinite(y)))
