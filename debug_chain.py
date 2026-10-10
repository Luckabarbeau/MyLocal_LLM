"""Debug the chain of backward passes."""

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
print(f"Output y: {y}")
print(f"Loss (sum(y*dy)): {float(xp.sum(y * dy))}")

# Manually compute the gradient through norm2 and ffn
norm2_input = cache["norm2_cache"]["x"]
ffn_out, ffn_cache_full = block.ffn.forward(norm2_input)

print(f"\n--- Forward through norm2 and ffn ---")
print(f"norm2_input: {norm2_input}")
print(f"ffn_out: {ffn_out}")

# Backward through ffn
dnorm2_out = block.ffn.backward(dy, ffn_cache_full)
print(f"\n--- After FFN backward ---")
print(f"dnorm2_out (gradient into norm2): {dnorm2_out}")

# Backward through norm2
dx_norm2 = block.norm2.backward(dnorm2_out, cache["norm2_cache"])
print(f"\n--- After norm2 backward ---")
print(f"dx_norm2 (gradient into first residual): {dx_norm2}")

# Now compute the gradient through norm1 and attention
norm1_input = cache["norm1_cache"]["x"]
attn_out, attn_cache_full = block.attention.forward(norm1_input)

print(f"\n--- Forward through norm1 and attn ---")
print(f"norm1_input: {norm1_input}")
print(f"attn_out: {attn_out}")

# Backward through attention
dnorm1_out = block.attention.backward(dx_norm2, attn_cache_full)
print(f"\n--- After attention backward ---")
print(f"dnorm1_out (gradient into norm1): {dnorm1_out}")

# Backward through norm1
dx_norm1 = block.norm1.backward(dnorm1_out, cache["norm1_cache"])
print(f"\n--- After norm1 backward ---")
print(f"dx_norm1 (gradient into x from norm1/attn): {dx_norm1}")

# Combine gradients
dresidual1 = dx_norm2  # Gradient from second residual through first residual
dx_total = dresidual1 + dx_norm1

print(f"\n--- Final gradient ---")
print(f"dresidual1: {dresidual1}")
print(f"dx_norm1: {dx_norm1}")
print(f"dx_total (dresidual1 + dx_norm1): {dx_total}")

# Compare with my backward pass
dx_my = block.backward(dy, cache)
print(f"\nMy backward dx: {dx_my}")

# Check if they match
print(f"\nMatch: {xp.allclose(dx_total, dx_my)}")
if not xp.allclose(dx_total, dx_my):
    print(f"Difference: {xp.abs(dx_total - dx_my)}")

# FD check
eps = 1e-6
v = xp.asarray(np.random.default_rng(100).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))

def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
an_my = float(xp.sum(dx_my * v))
an_total = float(xp.sum(dx_total * v))

print(f"\n--- FD check ---")
print(f"FD: {fd}")
print(f"AN (my backward): {an_my}")
print(f"AN (total): {an_total}")
