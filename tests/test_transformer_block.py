import numpy as np
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.backend import RandomStream
from mini_llm.backend import xp


def test_transformer_block_forward_backward():
    # deterministic RNG
    rng = RandomStream(1)
    # small dimensions for a quick test
    d_model = 8
    n_q_heads = 2
    n_kv_heads = 2
    d_head = 4
    d_ff = 32
    block = TransformerBlock(
        d_model=d_model,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_ff=d_ff,
        rng=rng,
    )

    # random input and upstream gradient
    B, T = 2, 3
    x = xp.asarray(
        np.random.default_rng(2).normal(size=(B, T, d_model)), dtype="float64"
    )
    dy = xp.asarray(
        np.random.default_rng(3).normal(size=(B, T, d_model)), dtype="float64"
    )

    # forward pass
    y, cache = block.forward(x)

    # backward pass
    dx = block.backward(dy, cache)

    # gradient check via finite differences
    # random direction vector, normalized
    v = xp.asarray(
        np.random.default_rng(4).normal(size=x.shape), dtype="float64"
    )
    v = v / xp.sqrt(xp.sum(v * v))
    eps = 1e-6

    def objective(z):
        y_fwd, _ = block.forward(z)
        # scalar objective: dot of y and upstream gradient dy
        return float(xp.sum(y_fwd * dy))

    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2.0 * eps)
    an = float(xp.sum(dx * v))
    assert abs(fd - an) < 1e-5, f"Finite diff {fd} vs analytic {an} mismatch"

    # ensure zero_grad works (no errors)
    block.zero_grad()
