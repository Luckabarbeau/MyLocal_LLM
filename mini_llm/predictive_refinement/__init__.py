"""Opt-in research primitives for block-causal continuous predictive refinement.

Nothing in the existing autoregressive model imports this package implicitly.
"""

from .soft_embeddings import (
    FuturePositionEmbeddings,
    SoftTokenExpectation,
    stable_softmax,
    softmax_vjp,
)

__all__ = [
    "FuturePositionEmbeddings",
    "SoftTokenExpectation",
    "stable_softmax",
    "softmax_vjp",
]
