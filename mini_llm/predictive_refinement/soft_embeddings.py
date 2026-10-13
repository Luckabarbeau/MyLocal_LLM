"""0079A: exact full-vocabulary soft-token feedback and manual derivatives.

Reference mathematics (no top-k, no detached feedback)::

    P = softmax(L / T)
    S = P @ E
    dP = dS @ E.T
    dE += P.T @ dS
    dL = P * (dP - sum(P * dP, axis=-1)) / T

All saved contexts are per-*call*, not mutable layer state, so the same module
can participate in multiple differentiable refinement passes. The 2-D GEMM
paths follow the repository's CuPy BF16 limitation on N-D matmul.
"""

from dataclasses import dataclass

from mini_llm.backend import xp, is_low_precision_dtype
from mini_llm.parameter import Parameter


def _acc_dtype(dtype):
    return xp.float32 if is_low_precision_dtype(dtype) else dtype


def stable_softmax(logits, temperature=1.0):
    """Full-vocabulary softmax with FP32 reductions for FP16/BF16 inputs."""
    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("feedback temperature must be finite and positive")
    if logits.ndim < 2:
        raise ValueError("expected logits [..., vocab]")
    z = logits.astype(_acc_dtype(logits.dtype), copy=False) / float(temperature)
    z = z - xp.max(z, axis=-1, keepdims=True)
    exp_z = xp.exp(z)
    return exp_z / xp.sum(exp_z, axis=-1, keepdims=True)


def softmax_vjp(probs, upstream, temperature=1.0):
    """Jacobian-vector product without allocating a V-by-V Jacobian."""
    if probs.shape != upstream.shape:
        raise ValueError("probability and upstream gradient shapes must match")
    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("feedback temperature must be finite and positive")
    dtype = _acc_dtype(probs.dtype)
    p = probs.astype(dtype, copy=False)
    g = upstream.astype(dtype, copy=False)
    return p * (g - xp.sum(p * g, axis=-1, keepdims=True)) / float(temperature)


@dataclass(frozen=True)
class ExpectationCache:
    probs: object
    embedding_weight: Parameter
    temperature: float
    original_shape: tuple


class SoftTokenExpectation:
    """Shared token-embedding expectation over **all** vocabulary entries.

    ``embedding_weight`` is the original ``model.embedding.W`` Parameter.
    Its ``grad`` is updated in-place in backward and therefore also collects
    gradients from ordinary fixed-token lookups and tied projections, if any.

    Precision: FP64/FP32 uses normal GEMM. On the GPU low-precision path we
    cast only the *small* [num_active_positions, V] probabilities/gradients to
    the embedding compute dtype, not the large embedding matrix to FP32.
    The FP32 probability/JVP accumulation is retained for softmax stability.
    """

    def __init__(self, embedding_weight: Parameter):
        if len(embedding_weight.shape) != 2:
            raise ValueError("embedding_weight must have shape [vocab, d_model]")
        self.embedding_weight = embedding_weight

    def forward(self, logits, *, temperature=1.0):
        if logits.shape[-1] != self.embedding_weight.shape[0]:
            raise ValueError("logit vocabulary dimension does not match embedding")
        p = stable_softmax(logits, temperature)
        out_shape = (*p.shape[:-1], self.embedding_weight.shape[1])
        p2 = p.reshape(-1, p.shape[-1])
        e = self.embedding_weight.data
        p_compute = (
            p2.astype(e.dtype, copy=False)
            if is_low_precision_dtype(e.dtype) else p2
        )
        s = (p_compute @ e).reshape(out_shape)
        return s, ExpectationCache(
            probs=p,
            embedding_weight=self.embedding_weight,
            temperature=float(temperature),
            original_shape=tuple(logits.shape),
        )

    def backward(self, upstream, cache: ExpectationCache):
        """Return gradient w.r.t. input logits; add weight gradient in-place."""
        weight = cache.embedding_weight
        if weight is not self.embedding_weight:
            raise ValueError("backward cache belongs to another embedding")
        if tuple(upstream.shape) != (
            *cache.original_shape[:-1], weight.shape[1]
        ):
            raise ValueError("superposition gradient shape mismatch")
        p2 = cache.probs.reshape(-1, weight.shape[0])
        dy2 = upstream.reshape(-1, weight.shape[1])
        e = weight.data
        if is_low_precision_dtype(e.dtype):
            # Two regular 2D BF16/FP16 GEMMs. FP32 gradient accumulation is
            # performed by Parameter.grad; never materialize E.astype(FP32).
            p_mat = p2.astype(e.dtype, copy=False)
            dy_mat = dy2.astype(e.dtype, copy=False)
        else:
            p_mat, dy_mat = p2, dy2
        weight.grad += p_mat.T @ dy_mat
        dp = (dy_mat @ e.T).reshape(cache.original_shape)
        return softmax_vjp(cache.probs, dp, cache.temperature)


class FuturePositionEmbeddings:
    """Separate learned unknown/soft/fixed state vectors for an active block.

    No additional absolute positional embedding is introduced: RoPE remains
    the sole positional mechanism in the existing Transformer.
    """

    def __init__(self, d_model, rng, *, std=0.02, dtype="float32"):
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        self.unknown = Parameter(
            rng.normal((int(d_model),), std=std, dtype=dtype),
            name="predictive_refinement.unknown", decay=False,
        )
        self.soft = Parameter(
            xp.zeros((int(d_model),), dtype=dtype),
            name="predictive_refinement.type_soft", decay=False,
        )
        self.fixed = Parameter(
            xp.zeros((int(d_model),), dtype=dtype),
            name="predictive_refinement.type_fixed", decay=False,
        )

    def parameters(self):
        return [self.unknown, self.soft, self.fixed]

    def unknown_block(self, batch_size, block_size):
        if batch_size <= 0 or block_size <= 0:
            raise ValueError("block dimensions must be positive")
        return xp.broadcast_to(
            self.unknown.data[None, None, :],
            (batch_size, block_size, self.unknown.shape[0]),
        ).copy()

    def soft_block(self, expected_embeddings):
        if expected_embeddings.shape[-1] != self.soft.shape[0]:
            raise ValueError("soft embedding dimension mismatch")
        return expected_embeddings + self.soft.data

    def fixed_block(self, token_ids, embedding_weight: Parameter):
        if len(embedding_weight.shape) != 2 or embedding_weight.shape[1] != self.fixed.shape[0]:
            raise ValueError("token embedding dimension mismatch")
        return embedding_weight.data[token_ids] + self.fixed.data

    def backward_unknown(self, dy):
        self.unknown.grad += xp.sum(dy.astype(self.unknown.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))

    def backward_soft(self, dy):
        self.soft.grad += xp.sum(dy.astype(self.soft.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))
        return dy

    def backward_fixed(self, dy, token_ids, embedding_weight: Parameter):
        self.fixed.grad += xp.sum(dy.astype(self.fixed.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))
        xp.add.at(
            embedding_weight.grad,
            token_ids.reshape(-1),
            dy.reshape(-1, dy.shape[-1]),
        )
