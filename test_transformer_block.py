import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.blocks import TransformerBlock

def test_transformer_block_basic():
    """Test basic forward and backward pass of TransformerBlock."""
    rng = RandomStream(42)
    
    # Create a small Transformer block
    block = TransformerBlock(
        d_model=16,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=8,
        d_ff=32,
        input_std=0.02,
        output_std=0.02,
        rng=rng,
        dtype="float64"
    )
    
    # Create test input
    x = rng.normal((2, 5, 16), std=0.25, dtype="float64")
    
    # Forward pass
    y = block.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{xp.min(y):.6f}, {xp.max(y):.6f}]")
    
    # Test backward pass
    dy = rng.normal((2, 5, 16), std=0.1, dtype="float64")
    
    # Need to capture cache for backward
    y, cache = block.forward(x, return_cache=True)
    block.zero_grad()
    
    dx = block.backward(dy, cache)
    
    print(f"Gradient shape: {dx.shape}")
    print(f"Gradient range: [{xp.min(dx):.6f}, {xp.max(dx):.6f}]")
    
    # Test numerical gradient for a parameter
    params = block.parameters()
    if params:
        p = params[0]
        orig = p.data.copy()
        
        # Compute numerical gradient using finite differences
        eps = 1e-6
        v = rng.normal(p.shape, std=0.1, dtype="float64")
        v = v / xp.sqrt(xp.sum(v*v))
        
        # Forward with perturbed parameter
        p.data[...] = orig + eps * v
        y_plus, _ = block.forward(x, return_cache=True)
        loss_plus = xp.sum(y_plus * dy)
        
        # Forward with original parameter
        p.data[...] = orig
        y_orig, _ = block.forward(x, return_cache=True)
        loss_orig = xp.sum(y_orig * dy)
        
        # Forward with perturbed parameter
        p.data[...] = orig - eps * v
        y_minus, _ = block.forward(x, return_cache=True)
        loss_minus = xp.sum(y_minus * dy)
        
        p.data[...] = orig
        
        # Compute analytical and numerical gradients
        ana = p.grad
        num = (loss_plus - loss_minus) / (2 * eps)
        dot = xp.sum(ana * v)
        
        print(f"Parameter {p.name}:")
        print(f"  Analytical gradient dot: {dot:.8f}")
        print(f"  Numerical gradient: {num:.8f}")
        print(f"  Relative error: {abs(dot - num)/(abs(dot)+abs(num)+1e-12):.2e}")
        
        # Check if numerical gradient matches analytical
        if abs(dot - num) / (abs(dot) + abs(num) + 1e-12) < 1e-4:
            print("✓ Numerical gradient test PASSED")
        else:
            print("✗ Numerical gradient test FAILED")
    
    print("\nBasic TransformerBlock test completed successfully!")

if __name__ == "__main__":
    test_transformer_block_basic()