from ..backend import xp, RandomStream
from ..ops.rmsnorm import RMSNorm
from ..ops.attention import GQAAttention
from ..ops.swiglu import SwiGLU
class TransformerBlock:
    """
    Standard Transformer block with two residual branches:
    - Attention branch: RMSNorm → Attention → +X
    - Feed-forward branch: RMSNorm → SwiGLU → +X
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        d_ff, input_std, output_std, rng,
        rms_eps=1e-6, name="transformer_block", dtype=None
    ):
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff

        if dtype is None:
            dtype = "float32"

        # Attention branch components
        self.attention_norm = RMSNorm(d_model, eps=rms_eps, name=f"{name}.attn_norm", dtype=dtype)
        self.attention = GQAAttention(
            d_model, n_q_heads, n_kv_heads, d_head,
            input_std, output_std, rng, name=f"{name}.attention", dtype=dtype
        )

        # Feed-forward branch components
        self.ffn_norm = RMSNorm(d_model, eps=rms_eps, name=f"{name}.ffn_norm", dtype=dtype)
        self.ffn = SwiGLU(
            d_model, d_ff, input_std, output_std, rng, name=f"{name}.ffn", dtype=dtype
        )

    def parameters(self):
        """Return all trainable parameters of the block."""
        params = []
        params.extend(self.attention_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.ffn.parameters())
        return params

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x, return_cache=True):
        """
        Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape [B, T, d_model]
            return_cache: Whether to return cache for backward pass

        Returns:
            If return_cache is True: (y, cache) tuple
            If return_cache is False: (y, None) tuple for consistency with tests
        """
        # Attention branch
        x_norm1 = self.attention_norm.forward_no_cache(x)
        x_attn, attn_cache = self.attention.forward(x_norm1, return_cache=True)
        x_attn_norm = x_norm1 + x_attn

        # Feed-forward branch
        x_norm2 = self.ffn_norm.forward_no_cache(x_attn_norm)
        x_ffn, ffn_cache = self.ffn.forward(x_norm2, return_cache=True)
        y = x_norm2 + x_ffn

        if not return_cache:
            return y, None

        cache = {
            "x": x,
            "x_norm1": x_norm1,
            "attn_cache": attn_cache,
            "x_norm2": x_norm2,
            "ffn_cache": ffn_cache,
            "x_attn": x_attn_norm,
        }
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through the Transformer block.

        Args:
            dy: Gradient of loss with respect to output
            cache: Cache from forward pass

        Returns:
            dx: Gradient of loss with respect to input
        """
        x = cache["x"]
        x_norm1 = cache["x_norm1"]
        attn_cache = cache["attn_cache"]
        x_norm2 = cache["x_norm2"]
        ffn_cache = cache["ffn_cache"]
        x_attn = cache["x_attn"]

        # Backward through feed-forward branch
        dff = self.ffn.backward(dy, ffn_cache)
        df_norm2 = dff + dy

        # Backward through first RMSNorm (FFN branch)
        dx_norm2 = self.ffn_norm.backward(df_norm2, {"x": x_norm2, "x_hat": x_norm2, "inv_rms": 1.0 / xp.sqrt(xp.mean(x_norm2 * x_norm2) + self.ffn_norm.eps)})

        # Backward through attention branch
        dattn = dx_norm2 + df_norm2  # Residual connection
        dq = self.attention.backward(dattn, attn_cache)

        # Backward through first RMSNorm (attention branch)
        dx_norm1 = self.attention_norm.backward(dq, {"x": x_norm1, "x_hat": x_norm1, "inv_rms": 1.0 / xp.sqrt(xp.mean(x_norm1 * x_norm1) + self.attention_norm.eps)})

        # Residual connection for attention
        dx = dx_norm1 + dq

        return dx