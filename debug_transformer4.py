"""Debug TransformerBlock backward pass - check FD computation."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.transformer_block import TransformerBlock

# Create a small block
rng = RandomStream(10)
block = TransformerBlock(
    d_model=8,
    n_q_heads=2,
    n_kv_heads=1,
    d_head=4,
    d_ff=16,
    input_std=0.08,
    output_std=0.04,
    rng=rng,
    dtype="float64"
)

# Input and gradient
x = xp.asarray(np.random.default_rng(12).normal(size=(1, 3, 8)), dtype="float64")
dy = xp.asarray(np.random.default_rng(13).normal(size=(1, 3, 8)), dtype="float64")

# Forward pass
y, cache = block.forward(x)

print(f"Original output sum(y*dy): {float(xp.sum(y * dy))}")

# Check FD at x
eps = 1e-6
v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))

def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

y_plus, _ = block.forward(x + eps * v)
y_minus, _ = block.forward(x - eps * v)

f_plus = float(xp.sum(y_plus * dy))
f_minus = float(xp.sum(y_minus * dy))

print(f"\nFoward check:")
print(f"f(x+eps*v) = {f_plus}")
print(f"f(x-eps*v) = {f_minus}")
print(f"FD = (f_plus - f_minus) / (2*eps) = {(f_plus - f_minus) / (2 * eps)}")

# Now compute the analytical gradient
block.zero_grad()
dx = block.backward(dy, cache)
an = float(xp.sum(dx * v))
print(f"\nAnalytical gradient: {an}")
print(f"Analytical sign: {'+' if an > 0 else '-'}")

# Check individual components of the gradient
print(f"\nGradient breakdown:")
print(f"dx[0,0,:] = {dx[0,0,:]}")
print(f"v[0,0,:] = {v[0,0,:]}")
print(f"dx*v[0,0,:] = {dx[0,0,:] * v[0,0,:]}")
print(f"sum(dx*v) = {xp.sum(dx * v)}")

# Also check what happens if I change the sign
print(f"\nIf AN was positive: {abs(an)}")
print(f"FD is positive: {f_plus > f_minus}")
