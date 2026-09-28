"""Comprehensive debug of TransformerBlock backward pass."""

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

# Use a small eps for finite differences
eps = 1e-6

# Test with different inputs
for i in range(3):
    print(f"\n{'='*60}")
    print(f"Test {i+1}")
    print('='*60)
    
    # Input and gradient
    x = xp.asarray(np.random.default_rng(100+i).normal(size=(1, 2, 2)), dtype="float64")
    dy = xp.asarray(np.random.default_rng(200+i).normal(size=(1, 2, 2)), dtype="float64")
    
    # Forward pass
    y, cache = block.forward(x)
    
    # Backward pass
    block.zero_grad()
    dx = block.backward(dy, cache)
    
    # FD check for input
    v = xp.asarray(np.random.default_rng(300+i).normal(size=x.shape), dtype="float64")
    v /= xp.sqrt(xp.sum(v * v))
    
    def objective(z):
        out, _ = block.forward(z)
        return float(xp.sum(out * dy))
    
    fd = (objective(x + eps * v) - objective(x - eps * v)) / (2 * eps)
    an = float(xp.sum(dx * v))
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
    
    print(f"Input gradient check:")
    print(f"  FD: {fd:.6e}")
    print(f"  AN: {an:.6e}")
    print(f"  Rel error: {rel:.2e}")
    print(f"  Sign match: {'YES' if (fd > 0) == (an > 0) else 'NO'}")
    
    # FD check for first parameter
    param = block.parameters()[0]
    v_param = xp.asarray(np.random.default_rng(400+i).normal(size=param.data.shape), dtype="float64")
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
    
    param.data[...] = original_data
    
    print(f"Parameter '{param.name}' gradient check:")
    print(f"  FD: {fd_param:.6e}")
    print(f"  AN: {an_param:.6e}")
    print(f"  Rel error: {rel_param:.2e}")
    print(f"  Sign match: {'YES' if (fd_param > 0) == (an_param > 0) else 'NO'}")
