from ..backend import xp
from ..init import matrix_parameter
from ..ops import (
    RMSNorm, GQAAttention, SwiGLU, cross_entropy_forward, cross_entropy_backward
)
class TransformerBlock:
    """
    Standard dense Transformer block with two residual branches:
    1. Attention branch: RMSNorm → Attention → +X
    2. Feed-forward branch: RMSNorm → SwiGLU → +X
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head, n_experts, top_k, d_ff,
        input_std, output_std, residual_init_std, rng,
        rope_base=10_000.0, rms_eps=1e-6,
        name="transformer_block", dtype="float32"
    ):
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.rope_base = float(rope_base)
        self.rms_eps = float(rms_eps)

        # Attention branch
        self.attn_norm = RMSNorm(d_model, eps=rms_eps, name=f"{name}.attn_norm", dtype=dtype)
        self.attention = GQAAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            input_std=input_std,
            output_std=output_std,
            rng=rng,
            rope_base=rope_base,
            name=f"{name}.attention",
            dtype=dtype
        )

        # Feed-forward branch
        self.ffn_norm = RMSNorm(d_model, eps=rms_eps, name=f"{name}.ffn_norm", dtype=dtype)
        self.swiglu = SwiGLU(
            d_model, 
            d_ff, 
            input_std=input_std, 
            output_std=output_std, 
            rng=rng, 
            name=f"{name}.swiglu", 
            dtype=dtype
        )

    def parameters(self):
        return (
            self.attn_norm.parameters() +
            self.attention.parameters() +
            self.ffn_norm.parameters() +
            self.swiglu.parameters()
        )

    def forward(self, x, return_cache=True):
        if x.ndim != 3:
            raise ValueError("input must have shape [B,T,D].")
        b, t, d = x.shape
        if d != self.d_model:
            raise ValueError(f"input final dimension ({d}) != d_model ({self.d_model}).")

        # Store original input for residual connections
        x_original = x

        # Attention branch
        x_attn, attn_norm_cache = self.attn_norm.forward(x)
        if return_cache:
            x_attn, attention_cache = self.attention.forward(x_attn, return_cache=return_cache)
            x = x_original + x_attn  # residual connection
            # Feed-forward branch
            x_ffn, ffn_norm_cache = self.ffn_norm.forward(x)
            x_ffn, swiglu_cache = self.swiglu.forward(x_ffn)
            x = x + x_ffn  # residual connection
            return x, {
                "x_original": x_original,
                "attn_norm": attn_norm_cache,
                "attention": attention_cache,
                "ffn_norm": ffn_norm_cache,
                "swiglu": swiglu_cache
            }
        else:
            x_attn, _ = self.attention.forward(x_attn, return_cache=return_cache)
            x = x_original + x_attn  # residual connection
            x_ffn, _ = self.swiglu.forward(x)
            x = x + x_ffn  # residual connection
            return x, {"x": x}

    def backward(self, dy, cache):
        # Extract caches
        x_original = cache["x_original"]
        attn_norm_cache = cache["attn_norm"]
        attention_cache = cache["attention"]
        ffn_norm_cache = cache["ffn_norm"]
        swiglu_cache = cache["swiglu"]

        # Backprop through attention branch (in reverse order)
        # First, gradient through attention norm (pass entire cache)
        dx_norm_attn = self.attn_norm.backward(dy, attn_norm_cache)
        
        # Then gradient through attention
        dx = self.attention.backward(dx_norm_attn, attention_cache)
        
        # Add attention branch residual
        dx = dx + x_original
        
        # Backprop through feed-forward branch
        dx_ffn = self.swiglu.backward(dx, swiglu_cache)
        dx_norm_ffn = self.ffn_norm.backward(dx_ffn, ffn_norm_cache)
        
        # Add feed-forward branch residual
        dx = dx_norm_ffn + dx

        return dx

    def zero_grad(self):
        for p in self.parameters():
            if hasattr(p, 'zero_grad'):
                p.zero_grad()