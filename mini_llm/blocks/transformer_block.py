from mini_llm.backend import xp
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.swiglu import SwiGLU
class TransformerBlock:
    """
    Standard dense Transformer block with two residual branches:
    1. Attention branch: RMSNorm → Attention → residual connection
    2. Feed-forward branch: RMSNorm → SwiGLU → residual connection
    
    Formula: Y = X + Attention(RMSNorm1(X)) + SwiGLU(RMSNorm2(X))
    """

    def __init__(
        self,
        d_model,
        n_q_heads,
        n_kv_heads,
        d_head,
        d_ff,
        input_std,
        output_std,
        rng,
        rope_base=10_000.0,
        rms_eps=1e-6,
        name="block",
        dtype="float32"
    ):
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")

        self.d_model = d_model
        self.name = name
        self.dtype = dtype

        # First RMSNorm for attention branch
        self.norm1 = RMSNorm(d_model, eps=rms_eps, name=f"{name}.norm1", dtype=dtype)
        
        # Attention layer
        self.attn = GQAAttention(
            d_model, n_q_heads, n_kv_heads, d_head,
            input_std, output_std, rng, rope_base, dtype
        )

        # Second RMSNorm for feed-forward branch
        self.norm2 = RMSNorm(d_model, eps=rms_eps, name=f"{name}.norm2", dtype=dtype)
        
        # SwiGLU feed-forward network
        self.ffn = SwiGLU(
            d_model, d_ff, input_std, output_std, rng, name=f"{name}.ffn", dtype=dtype
        )

    def parameters(self):
        """Return all trainable parameters in the block."""
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.ffn.parameters())
        return params

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x, return_cache=False):
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape [B, T, d_model]
            return_cache: Whether to return attention cache for debugging
            
        Returns:
            y: Output tensor of same shape as input
            cache: Attention cache if return_cache=True, otherwise None
        """
        if x.ndim != 3:
            raise ValueError("TransformerBlock input must have shape [B,T,D].")
        b, t, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Input final dimension ({d}) != d_model ({self.d_model}).")

        # Attention branch: RMSNorm → Attention
        norm1_out, cache1 = self.norm1.forward(x)
        attn_out, attn_cache = self.attn.forward(norm1_out, return_cache=True)
        
        # Feed-forward branch: RMSNorm → SwiGLU
        norm2_out, cache2 = self.norm2.forward(x)
        ffn_out, cache3 = self.ffn.forward(norm2_out)

        # Residual connections
        y = x + attn_out + ffn_out

        if not return_cache:
            return y
        
        return y, {
            "norm1_cache": cache1,
            "norm2_cache": cache2,
            "attn_cache": attn_cache,
            "ffn_cache": cache3,
            "x": x,
            "norm1_out": norm1_out,
            "norm2_out": norm2_out,
        }

    def backward(self, dy, cache):
        """
        Backward pass through the Transformer block.
        
        Args:
            dy: Gradient of loss w.r.t. output y
            cache: Cache from forward pass
            
        Returns:
            dx: Gradient of loss w.r.t. input x
        """
        if cache is None:
            raise ValueError("Backward pass requires cache from forward pass.")

        x = cache["x"]
        norm1_out = cache["norm1_out"]
        norm2_out = cache["norm2_out"]

        # Attention branch: compute gradient through attention → norm1
        dattn = self.attn.backward(dy, cache["attn_cache"])
        dx1 = self.norm1.backward(dattn, cache["norm1_cache"])
        
        # Feed-forward branch: compute gradient through swiglu → norm2
        dffn = self.ffn.backward(dy, cache["ffn_cache"])
        dx2 = self.norm2.backward(dffn, cache["norm2_cache"])
        
        # Residual connection: dy flows directly to x
        dx = dx1 + dx2 + dy

        return dx