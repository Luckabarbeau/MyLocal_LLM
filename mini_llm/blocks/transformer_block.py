"""Transformer block implementation using existing primitive ops.

Implements the computation graph:
    x -> RMSNorm -> GQAAttention -> + -> RMSNorm -> SwiGLU -> + -> y
where the '+' denotes residual connections. This mirrors the design in
`repository_plan.tex`.

Provides:
- `parameters()` – flat list of all Parameter objects.
- `zero_grad()` – clears gradients.
- `forward(x)` – returns (y, cache).
- `backward(dy, cache)` – back‑propagates and updates grads.
"""

from ..backend import xp
from ..ops.rmsnorm import RMSNorm
from ..ops.attention import GQAAttention
from ..ops.swiglu import SwiGLU


class TransformerBlock:
    """Dense transformer block.

    The forward path is:
        x -> RMSNorm -> GQAAttention -> + (residual) -> RMSNorm -> SwiGLU -> + (residual) -> y
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, d_head: int,
                 d_ff: int = None, eps: float = 1e-6, name: str = "transformer_block",
                 dtype: str = "float32", rng=None):
        """Create a transformer block.

        Args:
            d_model: Model dimension.
            n_q_heads: Number of query heads (must be divisible by n_kv_heads).
            n_kv_heads: Number of key/value heads.
            d_head: Dimension per head.
            d_ff: Hidden size of the feed‑forward network (defaults to 4*d_model).
            eps: Epsilon for RMSNorm.
            name: Prefix for parameter names.
            dtype: Parameter dtype.
            rng: Optional RandomStream for weight init.
        """
        if d_ff is None:
            d_ff = 4 * d_model

        # Primitive components
        self.norm1 = RMSNorm(d_model, eps=eps, name=f"{name}.norm1")
        self.attn = GQAAttention(
            d_model, n_q_heads, n_kv_heads, d_head,
            input_std=0.02, output_std=0.02,
            rng=rng,
            name=f"{name}.attn",
        )
        self.norm2 = RMSNorm(d_model, eps=eps, name=f"{name}.norm2")
        self.swi = SwiGLU(
            d_model, d_ff,
            input_std=0.02, output_std=0.02,
            rng=rng,
            name=f"{name}.swi",
        )

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def parameters(self):
        """Return a flat list of all Parameter objects used by the block."""
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.swi.parameters()
        )

    def zero_grad(self):
        """Zero gradients of all parameters."""
        for p in self.parameters():
            p.zero_grad()

    # ---------------------------------------------------------------------
    # Forward / backward
    # ---------------------------------------------------------------------
    def forward(self, x):
        """Forward pass.

        Returns (y, cache) where ``y`` has shape [B, T, D] and ``cache`` stores
        intermediate values needed for the backward pass.
        """
        # First RMSNorm
        x_norm, cache_norm1 = self.norm1.forward(x)
        # Attention
        attn_out, cache_attn = self.attn.forward(x_norm, return_cache=True)
        # First residual
        y1 = x + attn_out
        # Second RMSNorm
        y1_norm, cache_norm2 = self.norm2.forward(y1)
        # SwiGLU
        ff_out, cache_swi = self.swi.forward(y1_norm)
        # Second residual
        y = y1 + ff_out
        cache = {
            "x": x,
            "norm1": cache_norm1,
            "attn": cache_attn,
            "y1": y1,
            "norm2": cache_norm2,
            "swi": cache_swi,
        }
        return y, cache

    def backward(self, dy, cache):
        """Backward pass.

        Args:
            dy: Gradient of loss w.r.t. block output.
            cache: Cache from forward.
        Returns:
            Gradient w.r.t. block input.
        """
        # Unpack cache
        x = cache["x"]
        norm1_cache = cache["norm1"]
        attn_cache = cache["attn"]
        y1 = cache["y1"]
        norm2_cache = cache["norm2"]
        swi_cache = cache["swi"]

        # Backprop through second residual
        # Gradient splits to y1 and through SwiGLU
        dff = self.swi.backward(dy, swi_cache)
        dnorm2 = self.norm2.backward(dff, norm2_cache)
        # Combine gradients for y1
        dy1 = dy + dnorm2
        # Backprop through first residual (x + attn_out)
        self.attn.backward(dy1, attn_cache)
        dnorm1 = self.norm1.backward(dy1, norm1_cache)
        # Return gradient w.r.t. input x
        return dnorm1
"

from ..backend import xp

# Import the primitive modules
from ..ops.rmsnorm import RMSNorm
from ..ops.attention import GQAAttention
from ..ops.swiglu import SwiGLU


class TransformerBlock:
    """Dense transformer block.

    The block implements the following computation graph (B = batch, T = sequence length, D = model dimension)::

        x ──► RMSNorm ──► GQAAttention ──► + ──► RMSNorm ──► SwiGLU ──► + ──► y

    The first residual adds the original ``x`` to the attention output. The second residual
    adds the output of the first residual to the output of the feed‑forward network.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int = None,
        eps: float = 1e-6,
        name: str = "transformer_block",
        dtype: str = "float32",
        rng=None,
    ):
        """Create a new transformer block.

        Args:
            d_model: Model dimension.
            n_q_heads: Number of query heads (must be divisible by n_kv_heads).
            n_kv_heads: Number of key/value heads.
            d_head: Dimension per head.
            d_ff: Hidden size of the feed‑forward network.
            eps: Epsilon for RMSNorm.
            name: Prefix for parameter names.
            dtype: Parameter dtype.
            rng: RandomStream used for weight initialization (optional).
        """
        if d_ff is None:
            d_ff = 4 * d_model

        # Primitive components
        self.norm1 = RMSNorm(d_model, eps=eps, name=f"{name}.norm1")
        # Pass the provided rng to the attention module
        self.attn = GQAAttention(
            d_model,
            n_q_heads,
            n_kv_heads,
            d_head,
            input_std=0.02,
            output_std=0.02,
            rng=rng,
            name=f"{name}.attn",
        )
        self.norm2 = RMSNorm(d_model, eps=eps, name=f"{name}.norm2")
        # Pass rng to SwiGLU as well
        self.swi = SwiGLU(
            d_model,
            d_ff,
            input_std=0.02,
            output_std=0.02,
            rng=rng,
            name=f"{name}.swi",
        )


    # ---------------------------------------------------------------------
    # Helper methods for the optimiser / training loop
    # ---------------------------------------------------------------------
    def parameters(self):
        """Return a flat list of all Parameter objects used by the block."""
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.swi.parameters()
        )

    def zero_grad(self):
        """Zero the gradients of all parameters.

        This mirrors the pattern used in other modules (e.g. ``Linear.backward``).
        """
        for p in self.parameters():
            p.zero_grad()

    # ---------------------------------------------------------------------
    # Forward / backward passes
    # ---------------------------------------------------------------------
    def forward(self, x):
        """Forward pass.

        Args:
+        x: Input tensor of shape ``[B, T, D]``.
+
+        Returns:
+        ``(y, cache)`` where ``y`` has shape ``[B, T, D]`` and ``cache`` stores
+        all intermediate values required for ``backward``.
+        """
        # First RMSNorm
        x_norm, cache_norm1 = self.norm1.forward(x)

        # Attention
        attn_out, cache_attn = self.attn.forward(x_norm, return_cache=True)

        # First residual connection
        y1 = x + attn_out

        # Second RMSNorm (applied to the output of the first residual)
+        y1_norm, cache_norm2 = self.norm2.forward(y1)
+
+        # Feed‑forward (SwiGLU)
+        ff_out, cache_swi = self.swi.forward(y1_norm)
+
+        # Second residual connection
+        y = y1 + ff_out
+
+        # Cache everything we will need in the backward pass.
+        cache = {
+            "x": x,
+            "norm1": cache_norm1,
+            "attn": cache_attn,
+            "y1": y1,
+            "norm2": cache_norm2,
+            "swi": cache_swi,
+        }
+        return y, cache
+
+    def backward(self, dy, cache):
+        """Backward pass.
+
+        Args:
+            dy: Gradient of the loss w.r.t. the block output ``y``.
+            cache: The cache produced by the matching ``forward`` call.
+
+        Returns:
+            Gradient of the loss w.r.t. the block input ``x``.
+        """
+        # Unpack cache
+        x = cache["x"]
+        norm1_cache = cache["norm1"]
+        attn_cache = cache["attn"]
+        y1 = cache["y1"]
+        norm2_cache = cache["norm2"]
+        swi_cache = cache["swi"]
+
+        # The gradient flowing into the second residual split into two parts:
+        #   * direct gradient from the loss (dy)
+        #   * gradient that flows back from later layers (via norm2 and SwiGLU)
+        # Compute the contribution from the feed‑forward path first.
+        # -----------------------------------------------------------------
+        # Backprop through SwiGLU
+        dff = self.swi.backward(dy, swi_cache)
+        # Backprop through the second RMSNorm (its input is y1)
+        dnorm2 = self.norm2.backward(dff, norm2_cache)
+
+        # Now we have the total gradient w.r.t. y1 (the output of the first residual)
+        # It consists of the direct path (dy) plus the contribution that arrived
+        # via the RMSNorm/FFN path.
+        dy1_total = dy + dnorm2
+
+        # Backprop through the first residual (x + attn_out). The gradient splits
+        # equally to both branches.
+        # Gradient for the attention branch
+        self.attn.backward(dy1_total, attn_cache)
+        # Gradient for the RMSNorm branch – we need to propagate through the
+        # RMSNorm that preceded the attention.
+        dnorm1 = self.norm1.backward(dy1_total, norm1_cache)
+        # The RMSNorm backward returns the gradient w.r.t. its input (the original x).
+        return dnorm1
+
+    # ---------------------------------------------------------------------
+    # Convenience: expose a single ``zero_grad`` call that also clears the
+    # internal caches of any sub‑modules (they store no state beyond the cache).
+    # ---------------------------------------------------------------------
+    # No extra methods needed – the ``zero_grad`` method defined above already
+    # iterates over all parameters.
+
+    # The class deliberately does not implement ``__call__``; the surrounding
+    # training loop should invoke ``forward`` / ``backward`` explicitly, mirroring
+    # the pattern used throughout the codebase.
+
"