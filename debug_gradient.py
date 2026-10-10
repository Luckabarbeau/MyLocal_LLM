"""Debug the gradient computation."""

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

# Forward pass
y, cache = block.forward(x)

# Compute dx manually
norm2_input = cache["norm2_cache"]["x"]
ffn_out, ffn_cache_full = block.ffn.forward(norm2_input)
dnorm2_out = block.ffn.backward(dy, ffn_cache_full)
dx_norm2 = block.norm2.backward(dnorm2_out, cache["norm2_cache"])

norm1_input = cache["norm1_cache"]["x"]
attn_out, attn_cache_full = block.attention.forward(norm1_input)
dnorm1_out = block.attention.backward(dx_norm2, attn_cache_full)
dx_norm1 = block.norm1.backward(dnorm1_out, cache["norm1_cache"])

dresidual1 = dx_norm2
dx_total = dresidual1 + dx_norm1

print(f"dx_total: {dx_total}")
print(f"dx_total[0,0,:]: {dx_total[0,0,:]}")

# Random vector v
v = xp.asarray(np.random.default_rng(100).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))

print(f"\nv: {v}")
print(f"v[0,0,:]: {v[0,0,:]}")

# Compute sum(dx * v)
result = xp.sum(dx_total * v)
print(f"\nsum(dx_total * v): {result}")

# Check element-wise
print(f"\ndx_total * v: {dx_total * v}")
print(f"sum(dx_total * v) = {dx_total[0,0,0] * v[0,0,0] + dx_total[0,0,1] * v[0,0,1]}")

# FD check
eps = 1e-6
def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
print(f"\nFD: {fd}")

# Now compute AN using the actual dx from my backward pass
dx_my = block.backward(dy, cache)
print(f"dx_my: {dx_my}")
an_my = float(xp.sum(dx_my * v))
print(f"AN (my backward): {an_my}")
