"""Decoder-only language model.

Implements a minimal decoder LM using the explicit primitives defined in the
project (embedding, transformer block, and a tied output projection). The model
structure follows the plan in ``repository_plan.tex`` for Milestone 3:

* ``Embedding`` maps token ids to vectors.
* A stack of ``TransformerBlock`` layers processes the sequence.
* The final logits are produced by a linear projection that re‑uses the embedding
  matrix (weight‑tying).

Only the core forward/backward logic is provided – training loops, loss
functions, etc. are handled elsewhere in the codebase.
"""

from ..init import matrix_parameter
from ..backend import xp
from ..ops.embedding import Embedding
from ..ops.attention import GQAAttention  # imported for type hint only
from ..ops.swiglu import SwiGLU
from ..ops.rmsnorm import RMSNorm
from ..blocks.transformer_block import TransformerBlock


class DecoderLM:
    """Simple decoder‑only language model.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    d_model : int
        Model dimension (hidden size).
    n_layers : int
        Number of transformer blocks to stack.
    n_q_heads : int
    n_kv_heads : int
    d_head : int
        Head configuration – passed to each ``TransformerBlock``.
    d_ff : int, optional
        Hidden dimension of the feed‑forward network; if omitted ``4 * d_model`` is used.
    rng : RandomStream, optional
        Random stream for weight initialisation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int = None,
        rng=None,
    ):
        # Embedding (weights are a Parameter created via ``matrix_parameter``)
        self.embed = Embedding(vocab_size, d_model, std=0.02, rng=rng, name="decoder.embed")

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                d_ff=d_ff,
                name=f"decoder.block{i}",
                rng=rng,
            )
            for i in range(n_layers)
        ]

        # Output projection – weight‑tied to the embedding matrix (transpose).
        # We store a reference to the embedding weight for convenience.
        self.output_weight = self.embed.W  # tie weight

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def parameters(self):
        """Return a flat list of all parameters (including tied ones).

        The embedding weight appears only once; the transformer blocks expose their
        own parameters via their ``parameters()`` method.
        """
        params = [self.embed.W]
        for block in self.blocks:
            params.extend(block.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients in the model."""
        self.embed.W.zero_grad()
        for block in self.blocks:
            block.zero_grad()

    # ---------------------------------------------------------------------
    # Forward / backward passes
    # ---------------------------------------------------------------------
    def forward(self, token_ids):
        """Forward pass.

        Args:
            token_ids : array of shape ``[B, T]`` containing integer token ids.

        Returns:
                (logits, cache) where ``logits`` has shape ``[B, T, vocab_size]``.
        """
        # Embedding lookup – shape (B, T, d_model)
        x, embed_cache = self.embed.forward(token_ids)

        # Sequentially apply transformer blocks
        cache_blocks = []
        for block in self.blocks:
            x, block_cache = block.forward(x)
            cache_blocks.append(block_cache)

        # Final projection – use the tied embedding matrix (transpose)
        # x shape: (B, T, d_model), weight: (vocab, d_model)
        logits = x @ self.output_weight.data.T

        cache = {
            "embed": embed_cache,
            "blocks": cache_blocks,
            "token_ids": token_ids,
        }
        return logits, cache

    def backward(self, dlogits, cache):
        """Backward pass.

        Args:
            dlogits : gradient w.r.t. the logits output, shape ``[B, T, vocab_size]``.
            cache : cache dict returned by ``forward``.
        Returns:
            gradient w.r.t. the input token ids (None, as token ids are not differentiable).
        """
        # Backprop through the tied output projection – accumulate gradient on the
        # embedding weight (transpose of the projection).
        # dlogits @ W -> gradient w.r.t. x
        dx = dlogits @ self.output_weight.data
        # Gradient for the embedding weight – we need to add the contribution to the
        # embedding matrix. This mirrors the pattern used in ``Embedding.backward``.
        # ``Embedding.backward`` expects the gradient w.r.t. the embedding output,
        # which is exactly ``dx``.
        self.embed.backward(dx, cache["embed"])

        # Backprop through transformer blocks in reverse order
        grad = dx
        for block, block_cache in zip(reversed(self.blocks), reversed(cache["blocks"])):
            # each block's backward returns gradient for its input
            grad = block.backward(grad, block_cache)

        # No gradient w.r.t. token ids (they are integers), so we simply return None
        return None

    # ---------------------------------------------------------------------
    # Convenience for training loops
    # ---------------------------------------------------------------------
    def train_step(self, token_ids, targets, loss_fn):
        """One training step returning loss value.

        ``loss_fn`` should accept ``logits`` and ``targets`` and return a scalar
        loss and its gradient w.r.t. logits (e.g. cross‑entropy). This helper is not
        used by the tests but provides a useful entry point for experimentation.
        """
        logits, cache = self.forward(token_ids)
        loss, dlogits = loss_fn(logits, targets)
        self.backward(dlogits, cache)
        return loss
"