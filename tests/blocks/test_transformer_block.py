import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks import TransformerBlock
from mini_llm.config import ModelConfig


def make_transformer_block():
    """Create a transformer block with small dimensions for testing."""
    rng = RandomStream(10)
    return TransformerBlock(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4, n_experts=3, top_k=2, d_ff=16,
        input_std=0.08, output_std=0.04, residual_init_std=0.01, rng=rng, dtype="float64"
    )


def test_transformer_block_forward():
    """Test transformer block forward pass."""
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(11).normal(size=(2, 5, 8)), dtype="float64")
    y, cache = block.forward(x)
    
    assert y.shape == (2, 5, 8), f"Output shape should be (2,5,8), got {y.shape}"
    assert "x_original" in cache, "Cache should contain x_original"
    assert "attn_norm" in cache, "Cache should contain attn_norm"
    assert "attention" in cache, "Cache should contain attention"
    assert "ffn_norm" in cache, "Cache should contain ffn_norm"
    assert "swiglu" in cache, "Cache should contain swiglu"
    
    print("✓ Transformer block forward pass test passed")


def test_transformer_block_backward():
    """Test transformer block backward pass with gradient checking."""
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(12).normal(size=(1, 3, 8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(13).normal(size=(1, 3, 8)), dtype="float64")
    block.zero_grad()
    
    y, cache = block.forward(x)
    block.zero_grad()
    dx = block.backward(dy, cache)
    
    assert dx.shape == x.shape, f"Input gradient shape should match input shape ({x.shape}), got {dx.shape}"
    
    # Simple gradient check using finite differences
    v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
    v = v / xp.sqrt(xp.sum(v * v))
    eps = 1e-6
    
    def objective(z):
        y_out, _ = block.forward(z)
        return float(xp.sum(y_out * dy))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    print(f"  FD: {fd:.6e}, AD: {an:.6e}, rel: {rel:.6e}")
    assert rel < 1e-5, f"Gradient check failed: relative error {rel} exceeds threshold"
    
    print("✓ Transformer block backward pass test passed")


def test_transformer_block_zero_grad():
    """Test that zero_grad clears gradients."""
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(15).normal(size=(1, 3, 8)), dtype="float64")
    y, cache = block.forward(x)
    block.zero_grad()
    
    # Check that gradients are zero before backward
    for param in block.parameters():
        grad_norm = xp.sum(param.grad * param.grad)
        assert float(grad_norm) == 0.0, "Gradient should be zero after zero_grad"
    
    print("✓ Transformer block zero_grad test passed")


def test_transformer_block_residual_connections():
    """Test that residual connections are properly handled."""
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(16).normal(size=(1, 3, 8)), dtype="float64")
    
    # Test with cache
    y1, cache1 = block.forward(x)
    y2, cache2 = block.forward(x)
    
    # Should be identical due to deterministic forward pass
    assert xp.allclose(y1, y2, rtol=1e-10), "Forward passes should be deterministic"
    
    # Check that cache contains the right components
    assert "x_original" in cache1
    assert xp.allclose(cache1["x_original"], x), "Cached x_original should match input"
    
    print("✓ Transformer block residual connections test passed")


if __name__ == "__main__":
    test_transformer_block_forward()
    test_transformer_block_backward()
    test_transformer_block_zero_grad()
    test_transformer_block_residual_connections()
    print("\nAll transformer block tests passed!")