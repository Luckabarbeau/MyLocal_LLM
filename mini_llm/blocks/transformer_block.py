from ..ops.rmsnorm import RMSNorm
from ..ops.attention import GQAAttention
from ..ops.swiglu import SwiGLU
from ..init import matrix_parameter
from ..backend import xp, RandomStream
from ..config import ModelConfig
from ..parameter import Parameter
import numpy as np

# Global random stream for parameter initialization
_global_rng = RandomStream(42)
class TransformerBlock:
    """
    Standard dense Transformer block with residual connections.

    Architecture:
    X → RMSNorm → Attention → +X → RMSNorm → SwiGLU → +X

    Where "+X" denotes residual connection.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        rms_eps: float = 1e-6,
        rope_base: float = 10_000.0,
        init_std: float = 0.02,
        residual_init_std: float = 0.02,
        dtype: str = "float32",
        name: str = "block"
    ):
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff

        # Components
        self.norm1 = RMSNorm(d_model, eps=rms_eps, name=f"{name}.norm1", dtype=dtype)
        self.attn = GQAAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            input_std=init_std,
            output_std=residual_init_std,
            rng=_global_rng,
            rope_base=rope_base,
            dtype=dtype,
            name=f"{name}.attn"
        )
        self.norm2 = RMSNorm(d_model, eps=rms_eps, name=f"{name}.norm2", dtype=dtype)
        self.swiglu = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            input_std=init_std,
            output_std=residual_init_std,
            rng=_global_rng,
            dtype=dtype,
            name=f"{name}.swiglu"
        )

        # Layer normalization scales for residual connections
        self.resid_scale1 = matrix_parameter(
            (d_model,), residual_init_std, _global_rng, f"{name}.resid_scale1",
            dtype=dtype, decay=False
        )
        self.resid_scale2 = matrix_parameter(
            (d_model,), residual_init_std, _global_rng, f"{name}.resid_scale2",
            dtype=dtype, decay=False
        )

    def parameters(self):
        """Return all learnable parameters in this block."""
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.swiglu.parameters())
        params.extend([self.resid_scale1, self.resid_scale2])
        return params

    def forward(self, x):
        """
        Forward pass through Transformer block.

        Input:
            x: [B, T, D] tensor

        Returns:
            y: [B, T, D] tensor
            cache: dictionary containing intermediate values for backward pass
        """
        # First residual block: X → RMSNorm → Attention → +X
        x_norm1, cache1 = self.norm1.forward(x)
        attn_out, cache2 = self.attn.forward(x_norm1)

        # Residual connection: attn_out + x (scaled by resid_scale1)
        residual1 = x * self.resid_scale1.data
        x1 = attn_out + residual1

        # Second residual block: X1 → RMSNorm → SwiGLU → +X1
        x_norm2, cache3 = self.norm2.forward(x1)
        swiglu_out, cache4 = self.swiglu.forward(x_norm2)

        # Residual connection: swiglu_out + x1 (scaled by resid_scale2)
        residual2 = x1 * self.resid_scale2.data
        y = swiglu_out + residual2

        cache = {
            "x": x,
            "x_norm1": x_norm1,
            "attn_out": attn_out,
            "residual1": residual1,
            "x1": x1,
            "x_norm2": x_norm2,
            "swiglu_out": swiglu_out,
            "residual2": residual2,
            "cache1": cache1,
            "cache2": cache2,
            "cache3": cache3,
            "cache4": cache4,
        }

        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through Transformer block.

        Input:
            dy: [B, T, D] gradient of loss w.r.t. output y
            cache: cache from forward pass

        Returns:
            dx: [B, T, D] gradient of loss w.r.t. input x
        """
        # Unpack cache
        x = cache["x"]
        x_norm1 = cache["x_norm1"]
        attn_out = cache["attn_out"]
        residual1 = cache["residual1"]
        x1 = cache["x1"]
        x_norm2 = cache["x_norm2"]
        swiglu_out = cache["swiglu_out"]
        residual2 = cache["residual2"]
        cache1 = cache["cache1"]
        cache2 = cache["cache2"]
        cache3 = cache["cache3"]
        cache4 = cache["cache4"]

        # Second block backward: dL/d(x_norm2) from SwiGLU backward
        dx_norm2 = self.swiglu.backward(dy - residual2, cache4)

        # Second norm backward: dL/d(x1) from RMSNorm backward
        dx1 = self.norm2.backward(dx_norm2, cache3)

        # First residual gradient: dL/d(attn_out) + dL/d(residual2) * scale
        d_attn_out = dx1 - residual2 * self.resid_scale2.data

        # First norm backward: dL/d(x_norm1) from RMSNorm backward
        dx_norm1 = self.norm1.backward(d_attn_out, cache1)

        # Attention backward: dL/d(x_norm1) from Attention backward
        dx_norm1_from_attn = self.attn.backward(dx_norm1, cache2)

        # First residual gradient: dL/d(attn_out) + dL/d(residual1) * scale
        dx = dx_norm1_from_attn - residual1 * self.resid_scale1.data

        return dx

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()