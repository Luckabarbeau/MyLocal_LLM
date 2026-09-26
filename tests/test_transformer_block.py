import numpy as np
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.backend import RandomStream


def test_transformer_block_forward_backward():
    rng = RandomStream(42)
    cfg = dict(
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        d_ff=16,
        std=0.02,
    )
    blk = TransformerBlock(**cfg, rng=rng, name="testblk")
    B, T, D = 1, 4, cfg["d_model"]
    x = rng.normal((B, T, D), std=0.1, dtype="float32")

    # Forward
    y, cache = blk.forward(x)
    assert y.shape == (B, T, D)

    # Finite difference on input
    eps = 1e-4
    dy = rng.normal((B, T, D), std=0.1, dtype="float32")
    def objective(z):
        return float((blk.forward(z)[0] * dy).sum())

    f_plus = objective(x + eps)
    f_minus = objective(x - eps)
    fd = (f_plus - f_minus) / (2 * eps)

    # Backward
    blk.zero_grad()
    dx = blk.backward(dy, cache)
    an = float((dx * dy).sum())
    rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
    assert rel < 1e-4, f"Gradient mismatch: rel={rel:.2e}"

    # Finite difference on parameters (Wq as example)
    blk.zero_grad()
    orig = blk.attn.Wq.data.copy()
    v = rng.normal(blk.attn.Wq.data.shape, std=0.1, dtype="float32")
    blk.attn.Wq.data[...] = orig + eps * v
    f_plus = objective(x)
    blk.attn.Wq.data[...] = orig - eps * v
    f_minus = objective(x)
    blk.attn.Wq.data[...] = orig
    fd = (f_plus - f_minus) / (2 * eps)
    # analytic grad via backward
    blk.zero_grad()
    _ = blk.forward(x)[0]  # compute forward to set cache
    # use same dy
    blk.backward(dy, cache)
    analytic = np.sum(blk.attn.Wq.grad * v)
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
    assert rel < 1e-4, f"Param grad mismatch: rel={rel:.2e}"


if __name__ == "__main__":
    test_transformer_block_forward_backward()
