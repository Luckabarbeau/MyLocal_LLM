"""Debug simple case: just one component."""

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

# Input and gradient - use simple values
x = xp.asarray([[[-1.0, 1.0]]], dtype="float64")  # B=1, T=1, D=2
dy = xp.asarray([[[-1.0, 1.0]]], dtype="float64")

print(f"x = {x}")
print(f"dy = {dy}")

# Forward pass
y, cache = block.forward(x)
print(f"\ny = {y}")

# Loss
loss = float(xp.sum(y * dy))
print(f"Loss = sum(y * dy) = {loss}")

# Backward pass
block.zero_grad()
dx = block.backward(dy, cache)
print(f"\ndx = {dx}")

# FD check
eps = 1e-6
v = xp.asarray([[[-0.5, 0.5]]], dtype="float64")
v /= xp.sqrt(xp.sum(v * v))

def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

f_plus = objective(x + eps * v)
f_minus = objective(x - eps * v)
fd = (f_plus - f_minus) / (2 * eps)

an = float(xp.sum(dx * v))

print(f"\nFD = {fd}")
print(f"AN = {an}")
print(f"Relative error = {abs(fd - an) / (abs(fd) + abs(an) + 1e-12)}")

# Check individual parameters
print("\n--- Parameter gradients ---")
for p in block.parameters():
    print(f"{p.name}: grad norm = {xp.sqrt(xp.sum(p.grad * p.grad))}")
