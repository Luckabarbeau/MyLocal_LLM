"""Debug the finite difference computation."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.transformer_block import TransformerBlock

# Create a tiny block
rng = RandomStream(10)
block = TransformerBlock(
    d_model=2,
    n_q_heads=1,
    n_kv_heads=1,
    d_head=2,
    d_ff=4,
    input_std=0.08,
    output_std=0.04,
    rng=rng,
    dtype="float64"
)

# Input and gradient
x = xp.asarray([[[-1.0, 1.0]]], dtype="float64")
dy = xp.asarray([[[-1.0, 1.0]]], dtype="float64")

# Compute the original loss
y, _ = block.forward(x)
original_loss = float(xp.sum(y * dy))
print(f"Original loss: {original_loss}")

# FD check with a small v
v = xp.asarray(np.random.default_rng(100).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))

print(f"\nv: {v}")
print(f"v norm: {xp.sqrt(xp.sum(v * v))}")

eps = 1e-6
x_plus = x + eps * v
x_minus = x - eps * v

y_plus, _ = block.forward(x_plus)
y_minus, _ = block.forward(x_minus)

loss_plus = float(xp.sum(y_plus * dy))
loss_minus = float(xp.sum(y_minus * dy))

print(f"\nloss_plus: {loss_plus}")
print(f"loss_minus: {loss_minus}")
print(f"loss_plus - loss_minus: {loss_plus - loss_minus}")

fd = (loss_plus - loss_minus) / (2 * eps)
print(f"FD: {fd}")

# Now compute the analytical gradient
dx, _ = block.backward(dy, _)  # This won't work - need proper cache
