"""Debug TransformerBlock backward pass - step by step."""

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

print(f"Forward output (first token, first dim): {y[0,0,0]}")

# Check if loss is computed correctly
loss = float(xp.sum(y * dy))
print(f"Loss (sum of y*dy): {loss}")

# Backward pass
block.zero_grad()
dx = block.backward(dy, cache)

print(f"\nBackward dx (first token, first dim): {dx[0,0,0]}")

# Check individual components
print("\n--- Checking individual components ---")

# 1. Check FFN backward
_, ffn_cache = cache["ffn_cache"], None
norm2_out = cache["norm2_cache"]["x"]  # This is the input to FFN

# Manually compute what should happen through FFN
ffn_out, ffn_cache_full = block.ffn.forward(norm2_out)
print(f"FFN output (first token, first dim): {ffn_out[0,0,0]}")

# Compute gradient through FFN
dnorm2_out_manual = block.ffn.backward(dy, ffn_cache_full)
print(f"Gradient into FFN input (first token, first dim): {dnorm2_out_manual[0,0,0]}")

# 2. Check norm2 backward
norm2_out_val, norm2_cache_full = block.norm2.forward(norm2_out)
print(f"\nNorm2 output (first token, first dim): {norm2_out_val[0,0,0]}")

dx_norm2_manual = block.norm2.backward(dnorm2_out_manual, norm2_cache_full)
print(f"Gradient into norm2 input (first token, first dim): {dx_norm2_manual[0,0,0]}")

# 3. Check attention backward
attn_out, attn_cache_full = block.attention.forward(cache["norm1_cache"]["x"])
print(f"\nAttention output (first token, first dim): {attn_out[0,0,0]}")

# The gradient flowing into attention should be dx_norm2_manual (which is dy for the first residual)
dattn_out = dx_norm2_manual
dnorm1_out_manual = block.attention.backward(dattn_out, attn_cache_full)
print(f"Gradient into attention input (first token, first dim): {dnorm1_out_manual[0,0,0]}")

# 4. Check norm1 backward
norm1_out_val, norm1_cache_full = block.norm1.forward(x)
print(f"\nNorm1 output (first token, first dim): {norm1_out_val[0,0,0]}")

dx_norm1_manual = block.norm1.backward(dnorm1_out_manual, norm1_cache_full)
print(f"Gradient into norm1 input (first token, first dim): {dx_norm1_manual[0,0,0]}")

# Total gradient should be dresidual1 + dx_norm1
# Since x = residual1 + attn_out, and residual1 = x, then dresidual1 = dx_norm2_manual
dresidual1 = dx_norm2_manual
dx_total = dresidual1 + dx_norm1_manual
print(f"\nTotal gradient (first token, first dim): {dx_total[0,0,0]}")
print(f"My backward dx (first token, first dim): {dx[0,0,0]}")

# Check if they match
print(f"\nMatch: {xp.allclose(dx_total, dx)}")
