"""Debug TransformerBlock backward pass."""

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
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Cache keys: {cache.keys()}")

# Check if x is in the cache
print(f"\nCache norm1_cache keys: {cache['norm1_cache'].keys()}")
print(f"Cache attn_cache keys: {cache['attn_cache'].keys()}")
print(f"Cache norm2_cache keys: {cache['norm2_cache'].keys()}")
print(f"Cache ffn_cache keys: {cache['ffn_cache'].keys()}")

# Backward pass
block.zero_grad()
dx = block.backward(dy, cache)
print(f"\nBackward dx shape: {dx.shape}")

# Check gradients for a parameter
param = block.parameters()[0]
print(f"\nParameter name: {param.name}")
print(f"Parameter shape: {param.data.shape}")
print(f"Gradient shape: {param.grad.shape}")
print(f"Gradient sum: {xp.sum(xp.abs(param.grad))}")

# Try finite difference check manually
v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))
eps = 1e-6

def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
an = float(xp.sum(dx * v))
rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)

print(f"\nFinite difference check:")
print(f"FD: {fd}")
print(f"AN: {an}")
print(f"Relative error: {rel}")

# Also check parameter gradient
param = block.parameters()[0]
v_param = xp.asarray(np.random.default_rng(15).normal(size=param.data.shape), dtype="float64")
v_param /= xp.sqrt(xp.sum(v_param * v_param))

original_data = param.data.copy()

def objective_with_perturbation(eps_val):
    param.data[...] = original_data + eps_val * v_param
    out, _ = block.forward(x)
    return float(xp.sum(out * dy))

f_plus = objective_with_perturbation(eps)
f_minus = objective_with_perturbation(-eps)
fd_param = (f_plus - f_minus) / (2 * eps)
an_param = float(xp.sum(param.grad * v_param))
rel_param = abs(fd_param - an_param) / (abs(fd_param) + abs(an_param) + 1e-12)

# Restore
param.data[...] = original_data

print(f"\nParameter gradient check:")
print(f"FD param: {fd_param}")
print(f"AN param: {an_param}")
print(f"Relative error: {rel_param}")
