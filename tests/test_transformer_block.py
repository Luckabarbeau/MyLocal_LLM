import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.attention import GQAAttention
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.config import ModelConfig
def make_transformer_block(d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4, d_ff=16):
    rng = RandomStream(10)
    return TransformerBlock(
        d_model=d_model,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_ff=d_ff,
        rms_eps=1e-6,
        rope_base=10_000.0,
        init_std=0.08,
        residual_init_std=0.04,
        dtype="float64",
        name="test_block"
    )
def test_transformer_block_forward():
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(11).normal(size=(1, 4, 8)), dtype="float64")
    y, cache = block.forward(x)

    assert y.shape == x.shape
    assert isinstance(cache, dict)
    assert "x" in cache
    assert "y" in cache or "attn_out" in cache
    print("✓ Transformer block forward pass works")
def test_transformer_block_backward():
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(12).normal(size=(1, 3, 8)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(13).normal(size=(1, 3, 8)), dtype="float64")
    _, cache = block.forward(x)
    block.zero_grad()
    dx = block.backward(dy, cache)

    assert dx.shape == x.shape
    assert isinstance(dx, xp.ndarray)

    # Numerical gradient check
    v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v*v))
    eps = 1e-6

    def objective(z):
        out, _ = block.forward(z)
        return float(xp.sum(out*dy))

    fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
    an = float(xp.sum(dx*v))
    rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)
    assert rel < 2e-5, f"Gradient mismatch: rel={rel}"

    print("✓ Transformer block backward pass works")
def test_transformer_block_parameters():
    block = make_transformer_block()
    params = block.parameters()

    assert len(params) > 0
    for p in params:
        assert hasattr(p, 'data')
        assert hasattr(p, 'grad')
        assert hasattr(p, 'zero_grad')

    print("✓ Transformer block parameters are valid")
def test_transformer_block_residual_connections():
    block = make_transformer_block()
    x = xp.asarray(np.random.default_rng(15).normal(size=(1, 3, 8)), dtype="float64")
    y, cache = block.forward(x)

    # Check that residual connections are working by examining cache
    assert "residual1" in cache
    assert "residual2" in cache

    # Residual should be scaled version of input
    assert xp.allclose(cache["residual1"], x * block.resid_scale1.data)
    assert xp.allclose(cache["residual2"], cache["x1"] * block.resid_scale2.data)

    print("✓ Transformer block residual connections work")
def test_transformer_block_causal_mask():
    block = make_transformer_block(d_model=16, n_q_heads=4, n_kv_heads=2, d_head=4, d_ff=32)
    x = xp.asarray(np.random.default_rng(16).normal(size=(1, 5, 16)), dtype="float64")
    y, cache = block.forward(x)

    # The attention mechanism should be causal (future tokens not attended to)
    attn_cache = cache.get("cache2", {})
    if "probs" in attn_cache:
        p = attn_cache["probs"]
        # Check causal mask: for each position t, positions > t should have zero attention
        for t in range(5):
            if t + 1 < 5:
                assert float(xp.max(xp.abs(p[0, :, t, t+1:]))) == 0.0

    print("✓ Transformer block causal attention works")