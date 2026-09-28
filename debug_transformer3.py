"""Debug TransformerBlock backward pass - directional derivative."""

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

# Backward pass
block.zero_grad()
dx = block.backward(dy, cache)

# Check directional derivative
v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v * v))
eps = 1e-6

def objective(z):
    out, _ = block.forward(z)
    return float(xp.sum(out * dy))

fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
an = float(xp.sum(dx * v))

print(f"Finite difference: {fd}")
print(f"Analytical: {an}")
print(f"Relative error: {abs(fd - an) / (abs(fd) + abs(an) + 1e-12)}")

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

param.data[...] = original_data

print(f"\nParameter gradient check:")
print(f"FD param: {fd_param}")
print(f"AN param: {an_param}")
print(f"Relative error: {abs(fd_param - an_param) / (abs(fd_param) + abs(an_param) + 1e-12)}")

# Print the actual gradients
print(f"\nActual FD: {fd}")
print(f"Actual AN: {an}")
print(f"FD sign: {'+' if fd > 0 else '-'}")
print(f"AN sign: {'+' if an > 0 else '-'}")
