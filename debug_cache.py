"""Debug TransformerBlock cache."""

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

print(f"Input x: {x}")

# Forward pass
y, cache = block.forward(x)
print(f"\nOutput y: {y}")

# Check cache contents
print("\n--- Cache contents ---")
print(f"residual1 (should be x): {cache['residual1']}")
print(f"norm1_cache keys: {cache['norm1_cache'].keys()}")
print(f"norm1_cache['x']: {cache['norm1_cache']['x']}")

# Backward pass
block.zero_grad()
dx = block.backward(dy, cache)

print(f"\nBackward dx: {dx}")

# Now let's manually compute the gradient step by step
print("\n--- Manual backward computation ---")

# Start with dy
current_grad = dy.copy()
print(f"1. Starting with dy: {current_grad}")

# Through second residual: y = residual2 + ffn_out
# Both paths get dy
ffn_grad = current_grad.copy()
residual2_grad = current_grad.copy()
print(f"2. After second residual: ffn_grad = {ffn_grad}")

# Through FFN
norm2_input = cache["norm2_cache"]["x"]
ffn_out, ffn_cache_full = block.ffn.forward(norm2_input)
print(f"3. FFN input: {norm2_input}")
print(f"   FFN output: {ffn_out}")

# Backward through FFN
dnorm2_out = block.ffn.backward(ffn_grad, ffn_cache_full)
print(f"4. Gradient into norm2 input: {dnorm2_out}")

# Through norm2
norm2_out_val, norm2_cache_full = block.norm2.forward(norm2_input)
print(f"5. Norm2 output: {norm2_out_val}")

dx_norm2 = block.norm2.backward(dnorm2_out, norm2_cache_full)
print(f"6. Gradient into first residual (x1): {dx_norm2}")

# Through first residual: x1 = x + attn_out
attn_grad = dx_norm2.copy()
residual1_grad = dx_norm2.copy()
print(f"7. After first residual: attn_grad = {attn_grad}")

# Through attention
norm1_input = cache["norm1_cache"]["x"]
attn_out, attn_cache_full = block.attention.forward(norm1_input)
print(f"8. Attention input: {norm1_input}")
print(f"   Attention output: {attn_out}")

# Backward through attention
dnorm1_out = block.attention.backward(attn_grad, attn_cache_full)
print(f"9. Gradient into norm1 input: {dnorm1_out}")

# Through norm1
norm1_out_val, norm1_cache_full = block.norm1.forward(norm1_input)
print(f"10. Norm1 output: {norm1_out_val}")

dx_norm1 = block.norm1.backward(dnorm1_out, norm1_cache_full)
print(f"11. Gradient into x (from norm1): {dx_norm1}")

# Combine gradients
dx_manual = residual1_grad + dx_norm1
print(f"\n12. Final gradient (residual1_grad + dx_norm1): {dx_manual}")
print(f"    My backward dx: {dx}")
print(f"    Match: {xp.allclose(dx_manual, dx)}")
