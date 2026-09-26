"""Decoder‑only language model.

This module provides a very small, explicit implementation of a decoder‑only
Transformer that can be trained on toy data.  The implementation is kept
minimal to illustrate the design and to satisfy the repository plan.

The model consists of:

* ``Embedding`` – token‑to‑vector lookup.
* ``n_layers`` instances of :class:`~mini_llm.blocks.transformer_block.TransformerBlock`.
* ``lm_head`` – a linear projection from hidden state to vocabulary
  logits.  The projection weights are tied to the embedding matrix.

The public API mirrors the simple ``forward`` / ``backward`` interface
used in the primitive tests.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..backend import xp
from ..config import ModelConfig
from ..init import matrix_parameter
from ..ops.embedding import Embedding
from ..ops.loss import cross_entropy_backward
from . import DecoderLM

# Forward and backward pass utilities are defined in this module for
# clarity.  They operate on NumPy/CuPy arrays directly.


class DecoderLM:
    """Decoder‑only Transformer language model.

    Parameters
    ----------
    cfg: ModelConfig
        Configuration defining model hyper‑parameters.
    rng: RandomStream
        Random number generator for weight initialisation.
    """

    def __init__(self, cfg: ModelConfig, rng):
        self.cfg = cfg
        self.embedding = Embedding(
            cfg.vocab_size,
            cfg.d_model,
            cfg.init_std,
            rng,
            name="embedding",
            dtype=cfg.dtype,
        )
        # Block stack
        self.blocks: List[Any] = []
        for i in range(cfg.n_layers):
            blk = __import__("..blocks.transformer_block", globals={}, locals={}, fromlist=["TransformerBlock"], level=1).TransformerBlock(
                d_model=cfg.d_model,
                n_q_heads=cfg.n_q_heads,
                n_kv_heads=cfg.n_kv_heads,
                d_head=cfg.d_head,
                d_ff=cfg.d_ff,
                std=cfg.init_std,
                rng=rng,
                name=f"block{i}",
            )
            self.blocks.append(blk)
        # LM head tied to embedding
        self.lm_head = self.embedding.W

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def parameters(self):
        params = [self.embedding.W]
        for blk in self.blocks:
            params.extend(blk.parameters())
        return params

    def zero_grad(self):
        self.embedding.W.zero_grad()
        for blk in self.blocks:
            blk.zero_grad()

    # ---------------------------------------------------------------------
    # forward / backward
    # ---------------------------------------------------------------------
    def forward(self, token_ids: Any, return_cache: bool = True):
        """Forward pass.

        Parameters
        ----------
        token_ids: array of shape [B, T]
            Integer token ids.
        return_cache: bool
            Whether to return a cache for backward.

        Returns
        -------
        logits: array of shape [B, T, vocab]
            Unnormalised log‑probabilities.
        cache: dict
            Cached intermediates if ``return_cache``.
        """
        # 1. Embed
        h, cache_embed = self.embedding.forward(token_ids)
        # 2. Pass through blocks
        caches_blocks: List[Dict] = []
        for blk in self.blocks:
            h, blk_cache = blk.forward(h)
            caches_blocks.append(blk_cache)
        # 3. LM head (tied)
        logits = h @ self.lm_head.data.T
        if not return_cache:
            return logits
        return logits, {
            "cache_embed": cache_embed,
            "caches_blocks": caches_blocks,
            "h": h,  # final hidden state
        }

    def backward(self, dy: Any, cache: Dict):
        """Backward pass.

        Parameters
        ----------
        dy: array
            Gradient of loss w.r.t. logits.
        cache: dict
            Cache from ``forward``.

        Returns
        -------
        dx: array
            Gradient w.r.t. token ids – not used for training, but
            returned for API consistency.
        """
        # LM head gradient (tied to embedding)
        self.lm_head.grad += dy @ self.embedding.W.data
        # Backprop through blocks (reverse order)
        dh = dy @ self.lm_head.data.T
        for blk, blk_cache in reversed(list(zip(self.blocks, cache["caches_blocks"]))):
            dh = blk.backward(dh, blk_cache)
        # Backprop through embedding
        dx = self.embedding.backward(dh, cache["cache_embed"])
        return dx

    # ---------------------------------------------------------------------
    # simple loss
    # ---------------------------------------------------------------------
    def loss_and_grad(self, token_ids: Any):
        """Compute cross‑entropy loss and perform a backward step.

        Returns the loss scalar.
        """
        logits, cache = self.forward(token_ids)
        # Shift targets one step ahead for next‑token prediction
        targets = token_ids[:, 1:]
        logits = logits[:, :-1, :]
        loss, loss_cache = cross_entropy_forward(logits, targets)
        grads = cross_entropy_backward(loss_cache)
        self.backward(grads, cache)
        return loss

# expose for imports
__all__ = ["DecoderLM"]
"""
