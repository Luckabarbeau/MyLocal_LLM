"""Dense Transformer block with residual streams."""

from mini_llm.backend import xp
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.swiglu import SwiGLU


class TransformerBlock:
    """
    Dense Transformer block for decoder-only language models.
    
    Structure:
        X → RMSNorm → GQAAttention → +X → RMSNorm → SwiGLU → +Y
        
    Where Y is the final output.
    
    All operations preserve shape B×T×d_model.
    """
    
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        input_std: float,
        output_std: float,
        rng,
        eps: float = 1e-6,
        name: str = "block",
        dtype: str = "float32"
    ):
        """Initialize the Transformer block."""
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        
        # First RMSNorm (input to attention)
        self.norm1 = RMSNorm(d_model, eps=eps, name=f"{name}.norm1", dtype=dtype)
        
        # Attention sublayer
        self.attention = GQAAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            input_std=input_std,
            output_std=output_std,
            rng=rng,
            dtype=dtype
        )
        
        # Second RMSNorm (input to FFN)
        self.norm2 = RMSNorm(d_model, eps=eps, name=f"{name}.norm2", dtype=dtype)
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            input_std=input_std,
            output_std=output_std,
            rng=rng,
            name=f"{name}.ffn",
            dtype=dtype
        )
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.ffn.parameters())
        return params
    
    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, x):
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary containing intermediate values for backward pass
        """
        # First residual branch: attention
        residual1 = x
        
        # Norm → Attention → residual
        norm1_out, norm1_cache = self.norm1.forward(x)
        attn_out, attn_cache = self.attention.forward(norm1_out)
        x = residual1 + attn_out
        
        # Second residual branch: FFN
        residual2 = x
        
        # Norm → FFN → residual
        norm2_out, norm2_cache = self.norm2.forward(x)
        ffn_out, ffn_cache = self.ffn.forward(norm2_out)
        y = residual2 + ffn_out
        
        cache = {
            "residual1": residual1,
            "norm1_cache": norm1_cache,
            "attn_cache": attn_cache,
            "residual2": residual2,
            "norm2_cache": norm2_cache,
            "ffn_cache": ffn_cache,
        }
        
        return y, cache
    
    def backward(self, dy, cache):
        """
        Backward pass through the Transformer block.
        
        Args:
            dy: Gradient of loss w.r.t. output y, shape (B, T, d_model)
            cache: Dictionary from forward pass containing intermediates
            
        Returns:
            dx: Gradient of loss w.r.t. input x, shape (B, T, d_model)
        """
        residual2 = cache["residual2"]
        norm2_cache = cache["norm2_cache"]
        ffn_cache = cache["ffn_cache"]
        
        # Backward through second residual: y = residual2 + ffn_out
        # The gradient flows through both paths: direct (x1) and through FFN
        dfn_out = dy  # Gradient w.r.t. ffn_out
        dresidual2 = dy  # Gradient w.r.t. residual2 (which is x1)
        
        # Backward through FFN
        dnorm2_out = self.ffn.backward(dfn_out, ffn_cache)
        
        # Backward through second RMSNorm - this gives gradient through FFN path
        dx_norm2_through_ffn = self.norm2.backward(dnorm2_out, norm2_cache)
        
        # Total gradient through first residual is sum of direct and FFN paths
        # Since x1 = residual2 + ffn_out, the gradient w.r.t. x1 is dy (from direct path)
        # Plus the gradient from the FFN path
        dx_norm2 = dresidual2 + dx_norm2_through_ffn
        
        # Backward through first residual: x = residual1 + attn_out
        dattn_out = dx_norm2  # Gradient flows through addition
        dresidual1 = dx_norm2
        
        # Backward through attention
        dnorm1_out = self.attention.backward(dattn_out, cache["attn_cache"])
        
        # Backward through first RMSNorm
        dx_norm1 = self.norm1.backward(dnorm1_out, cache["norm1_cache"])
        
        # Combine gradients for residual1 and norm1
        dx = dresidual1 + dx_norm1
        
        return dx
