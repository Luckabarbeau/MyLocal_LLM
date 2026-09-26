"""Transformer block implementation.

This module provides a minimal implementation of a dense decoder-only Transformer block
using the primitives defined in :mod:`mini_llm.ops`.

The block follows the architecture described in the repository plan:

```
X -> RMSNorm -> Attention -> +X -> RMSNorm -> SwiGLU -> +X
```

The implementation uses explicit forward/backward passes and stores intermediate
values in a cache dictionary for use during back‑propagation.
"""

from __future__ import annotations

from typing import Any, Dict

from ..ops.rmsnorm import RMSNorm
from ..ops.attention import GQAAttention
from ..ops.swiglu import SwiGLU


class TransformerBlock:
    """A dense decoder‑only Transformer block.

    Parameters
    ----------
    d_model: int
        Dimension of the model (hidden size).
    n_q_heads: int
        Number of query heads.
    n_kv_heads: int
        Number of key/value heads.
    d_head: int
        Size of each head.
    d_ff: int
        Size of the feed‑forward (SwiGLU) hidden layer.
    std: float
        Standard deviation for weight initialization.
    rng: RandomStream
        Random number generator used for weight init.
    name: str, optional
        Base name for the sub‑components.
    dtype: str, optional
        Data type for parameters.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        std: float,
        rng,
        name: str = "transformer_block",
        dtype: str = "float32",
    ):
        self.rms1 = RMSNorm(d_model, eps=1e-6, name=f"{name}.rms1", dtype=dtype)
        self.attn = GQAAttention(
            d_model,
            n_q_heads,
            n_kv_heads,
            d_head,
            input_std=std,
            output_std=std,
            rng=rng,
            name=f"{name}.attn",
            dtype=dtype,
        )
        self.rms2 = RMSNorm(d_model, eps=1e-6, name=f"{name}.rms2", dtype=dtype)
        self.swin = SwiGLU(
            d_model,
            d_ff,
            input_std=std,
            output_std=std,
            rng=rng,
            name=f"{name}.swiglu",
            dtype=dtype,
        )

    def parameters(self):
        return (
            self.rms1.parameters()
            + self.attn.parameters()
            + self.rms2.parameters()
            + self.swin.parameters()
        )

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Any, return_cache: bool = True):
        """Forward pass.

        Parameters
        ----------
        x: array of shape [B, T, D]
            Input tensor.
        return_cache: bool
            Whether to return a cache for backward.

        Returns
        -------
        y: array
            Output tensor of the same shape as ``x``.
        cache: dict
            Cached intermediate values (if ``return_cache``).
        """
        # 1. RMSNorm 1
        y1, cache1 = self.rms1.forward(x)
        # 2. Attention
        y2, cache2 = self.attn.forward(y1)
        # 3. Residual 1
        res1 = x + y2
        # 4. RMSNorm 2
        y3, cache3 = self.rms2.forward(res1)
        # 5. SwiGLU
        y4, cache4 = self.swin.forward(y3)
        # 6. Residual 2
        y = res1 + y4
        if not return_cache:
            return y
        return y, {
            "cache1": cache1,
            "cache2": cache2,
            "cache3": cache3,
            "cache4": cache4,
            "res1": res1,
        }

    def backward(self, dy: Any, cache: Dict[str, Any]):
        """Backward pass.

        Parameters
        ----------
        dy: array
            Gradient of the loss w.r.t. the block output.
        cache: dict
            Cache produced by :meth:`forward`.

        Returns
        -------
        dx: array
            Gradient w.r.t. the block input.
        """
        cache1 = cache["cache1"]
        cache2 = cache["cache2"]
        cache3 = cache["cache3"]
        cache4 = cache["cache4"]

        # 1. Backward through the final residual addition: y = res1 + y4
        dy4 = dy
        # Residual gradient from output to res1
        dres1_residual = dy
        
        # Backward through SwiGLU
        dy3 = self.swin.backward(dy4, cache4)
        # Backward through RMSNorm 2 (y3 -> res1)
        dres1_from_rms2 = self.rms2.backward(dy3, cache3)
        # Total gradient to res1
        dres1 = dres1_residual + dres1_from_rms2
        
        # Backward through the attention module
        dy1 = self.attn.backward(dres1, cache2)
        
        # Backward through RMSNorm 1
        dx = self.rms1.backward(dy1, cache1)
        return dx

# End of file
