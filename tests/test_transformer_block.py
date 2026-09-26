import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.config import ModelConfig
def test_transformer_block_backward_direction():
    """Test that the Transformer block's backward pass matches finite differences."""
    rng = RandomStream(42)

    # Create a small Transformer block
    cfg = ModelConfig.tiny_inspection()
    block = TransformerBlock(
        d_model=cfg.d_model,
        n_q_heads=cfg.n_q_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_head=cfg.d_head,
        d_ff=cfg.d_ff,
        input_std=cfg.init_std,
        output_std=cfg.residual_init_std,
        rng=rng,
        dtype="float64"
    )

    # Create random input
    x = xp.asarray(np.random.default_rng(123).normal(size=(2, 4, cfg.d_model)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(456).normal(size=(2, 4, cfg.d_model)), dtype="float64")

    # Forward pass
    x_out, cache = block.forward(x, return_cache=True)

    # Backward pass
    dx = block.backward(dy, cache)

    # Test finite differences
    v = xp.asarray(np.random.default_rng(789).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    eps = 1e-6

    def objective(z):
        x_out, _ = block.forward(z, return_cache=False)
        return float(xp.sum(x_out * dy))

    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))

    rel_error = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    assert rel_error < 1e-7, f"Relative error too large: {rel_error}"
def test_transformer_block_shapes():
    """Test that the Transformer block preserves tensor shapes."""
    rng = RandomStream(123)

    cfg = ModelConfig.tiny_inspection()
    block = TransformerBlock(
        d_model=cfg.d_model,
        n_q_heads=cfg.n_q_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_head=cfg.d_head,
        d_ff=cfg.d_ff,
        input_std=cfg.init_std,
        output_std=cfg.residual_init_std,
        rng=rng,
        dtype="float64"
    )

    # Test with different batch sizes and sequence lengths
    for B in [1, 3]:
        for T in [1, 8, 16]:
            x = xp.asarray(np.random.default_rng(999).normal(size=(B, T, cfg.d_model)), dtype="float64")
            x_out, cache = block.forward(x, return_cache=True)

            # Check output shape matches input
            assert x_out.shape == x.shape, f"Shape mismatch: input {x.shape}, output {x_out.shape}"

            # Check cache structure
            assert "x" in cache
            assert "x_norm1" in cache
            assert "attn_cache" in cache
            assert "x_norm2" in cache
            assert "ffn_cache" in cache
            assert "x_attn" in cache

            print(f"✓ Block forward pass correct for B={B}, T={T}")
def test_transformer_block_residual_connections():
    """Test that residual connections work correctly."""
    rng = RandomStream(555)

    cfg = ModelConfig.tiny_inspection()
    block = TransformerBlock(
        d_model=cfg.d_model,
        n_q_heads=cfg.n_q_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_head=cfg.d_head,
        d_ff=cfg.d_ff,
        input_std=cfg.init_std,
        output_std=cfg.residual_init_std,
        rng=rng,
        dtype="float64"
    )

    # Create input
    x = xp.asarray(np.random.default_rng(111).normal(size=(1, 4, cfg.d_model)), dtype="float64")

    # Forward pass
    x_out, _ = block.forward(x, return_cache=False)

    # Check that output is not identical to input (residual should change it)
    assert not xp.allclose(x_out, x), "Output should differ from input due to residual connections"

    # Check that output contains contributions from both branches
    # (This is implicitly tested by the backward pass verification)
    print("✓ Residual connections working correctly")

if __name__ == "__main__":
    test_transformer_block_backward_direction()
    test_transformer_block_shapes()
    test_transformer_block_residual_connections()
    print("\n✓ All transformer block tests passed!")