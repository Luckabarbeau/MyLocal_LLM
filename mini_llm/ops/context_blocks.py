"""Block primitives for sparse long-context retrieval.

Compressed block representations are used for *search only*.  Once blocks are
selected, :func:`gather_full_resolution_blocks` reopens the original hidden
states (and later the original K/V tensors) at full resolution.
"""

import math

from ..backend import xp, resolve_dtype, is_low_precision_dtype
from ..parameter import Parameter


def complete_block_count(seq_len, block_size):
    """Number of complete non-overlapping history blocks in ``seq_len``."""
    seq_len = int(seq_len)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return seq_len // block_size


def block_token_indices(block_indices, block_size):
    """Expand block IDs to contiguous token IDs.

    Args:
        block_indices: Integer tensor of arbitrary shape.
        block_size: Number of tokens per block.

    Returns:
        Tensor of shape ``block_indices.shape + (block_size,)``.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    offsets = xp.arange(block_size, dtype=block_indices.dtype)
    return block_indices[..., None] * block_size + offsets


class HistoryBlockPooler:
    """Compress complete history blocks to one ``d_model`` vector per block."""

    def __init__(
        self,
        d_model,
        block_size,
        strategy="mean",
        rng=None,
        input_std=0.02,
        name="history_pool",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.block_size = int(block_size)
        self.strategy = str(strategy)
        if self.d_model <= 0 or self.block_size <= 0:
            raise ValueError("d_model and block_size must be positive")
        if self.strategy not in {"mean", "first", "last", "learned"}:
            raise ValueError("unsupported history pooling strategy")

        self.score_param = None
        if self.strategy == "learned":
            if rng is None:
                raise ValueError("learned history pooling requires rng")
            data = xp.asarray(
                rng.normal((self.d_model,), std=input_std, dtype=dtype)
            )
            self.score_param = Parameter(data, name=f"{name}.score")

    def parameters(self):
        return [] if self.score_param is None else [self.score_param]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, x):
        """Pool every complete block in ``x``.

        Args:
            x: ``(batch, seq_len, d_model)`` hidden states.

        Returns:
            pooled: ``(batch, n_complete_blocks, d_model)``.
            cache: Explicit backward cache.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (batch, seq_len, d_model)")

        batch, seq_len, _ = x.shape
        n_blocks = complete_block_count(seq_len, self.block_size)
        usable = n_blocks * self.block_size
        blocks = x[:, :usable, :].reshape(
            batch, n_blocks, self.block_size, self.d_model
        )

        if self.strategy == "mean":
            pooled = xp.mean(blocks, axis=2)
            cache = {"x_shape": x.shape, "n_blocks": n_blocks}
        elif self.strategy == "first":
            pooled = blocks[:, :, 0, :]
            cache = {"x_shape": x.shape, "n_blocks": n_blocks}
        elif self.strategy == "last":
            pooled = blocks[:, :, -1, :]
            cache = {"x_shape": x.shape, "n_blocks": n_blocks}
        else:
            scale = 1.0 / math.sqrt(self.d_model)
            score_vector = self.score_param.data
            blocks_2d = blocks.reshape(-1, self.d_model)
            scores = (blocks_2d @ score_vector).reshape(
                batch, n_blocks, self.block_size
            ) * scale
            scores_work = (
                scores.astype("float32", copy=False)
                if is_low_precision_dtype(scores.dtype)
                else scores
            )
            scores_work = scores_work - xp.max(scores_work, axis=2, keepdims=True)
            exp_scores = xp.exp(scores_work)
            alpha_work = exp_scores / xp.sum(exp_scores, axis=2, keepdims=True)
            alpha = (
                alpha_work.astype(x.dtype, copy=False)
                if is_low_precision_dtype(x.dtype)
                else alpha_work
            )
            pooled = xp.sum(blocks * alpha[..., None], axis=2)
            cache = {
                "x_shape": x.shape,
                "n_blocks": n_blocks,
                "blocks": blocks,
                "alpha_work": alpha_work,
                "scale": scale,
            }

        return pooled, cache

    def backward(self, dpooled, cache):
        """Explicit backward for block pooling."""
        x_shape = tuple(cache["x_shape"])
        batch, _, d_model = x_shape
        n_blocks = int(cache["n_blocks"])
        if dpooled.shape != (batch, n_blocks, d_model):
            raise ValueError("dpooled has incompatible shape")

        dx = xp.zeros(x_shape, dtype=dpooled.dtype)
        usable = n_blocks * self.block_size
        if n_blocks == 0:
            return dx

        dx_blocks = dx[:, :usable, :].reshape(
            batch, n_blocks, self.block_size, self.d_model
        )

        if self.strategy == "mean":
            dx_blocks[...] = dpooled[:, :, None, :] / self.block_size
            return dx
        if self.strategy == "first":
            dx_blocks[:, :, 0, :] = dpooled
            return dx
        if self.strategy == "last":
            dx_blocks[:, :, -1, :] = dpooled
            return dx

        blocks = cache["blocks"]
        alpha_work = cache["alpha_work"]
        scale = cache["scale"]

        # y = sum_t alpha_t h_t.  The first term is the direct value path.
        alpha_compute = (
            alpha_work.astype(dpooled.dtype, copy=False)
            if is_low_precision_dtype(dpooled.dtype)
            else alpha_work
        )
        dx_blocks += alpha_compute[..., None] * dpooled[:, :, None, :]

        # dalpha_t = dy dot h_t, followed by the softmax Jacobian.
        dpooled_work = dpooled.astype("float32", copy=False)
        blocks_work = blocks.astype("float32", copy=False)
        dalpha = xp.sum(blocks_work * dpooled_work[:, :, None, :], axis=-1)
        correction = xp.sum(alpha_work * dalpha, axis=2, keepdims=True)
        dscores = alpha_work * (dalpha - correction)

        score_vector_work = self.score_param.data.astype("float32", copy=False)
        score_path = dscores[..., None] * score_vector_work * scale
        dx_blocks += score_path.astype(dx_blocks.dtype, copy=False)

        grad_score = xp.sum(dscores[..., None] * blocks_work, axis=(0, 1, 2)) * scale
        self.score_param.grad += grad_score
        return dx


def gather_full_resolution_blocks(x, selected_blocks, block_size):
    """Gather complete selected blocks from a batch of sequences.

    ``selected_blocks`` has shape ``(B, ..., K)`` and the output has shape
    ``(B, ..., K * block_size, D)``.  Repeated block selections are allowed;
    the corresponding backward therefore uses scatter-add.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, seq_len, width)")
    if selected_blocks.ndim < 2:
        raise ValueError("selected_blocks must have shape (batch, ..., k)")
    if selected_blocks.shape[0] != x.shape[0]:
        raise ValueError("batch dimension mismatch")

    batch, seq_len, width = x.shape
    token_indices = block_token_indices(selected_blocks, block_size)
    if token_indices.size:
        if bool(xp.any(token_indices < 0)) or bool(xp.any(token_indices >= seq_len)):
            raise IndexError("selected block extends outside the source sequence")

    leading_shape = selected_blocks.shape[1:-1]
    k = selected_blocks.shape[-1]
    flat_tokens = token_indices.reshape(batch, -1)
    batch_ids = xp.arange(batch)[:, None]
    gathered = x[batch_ids, flat_tokens, :]
    gathered = gathered.reshape(
        (batch,) + leading_shape + (k * int(block_size), width)
    )
    cache = {
        "x_shape": x.shape,
        "flat_tokens": flat_tokens,
        "leading_shape": leading_shape,
        "k": int(k),
        "block_size": int(block_size),
    }
    return gathered, cache


def scatter_full_resolution_blocks(dgathered, cache):
    """Scatter-add a gathered-block gradient back to the source sequence."""
    x_shape = tuple(cache["x_shape"])
    batch, _, width = x_shape
    flat_tokens = cache["flat_tokens"]
    expected = (
        (batch,)
        + tuple(cache["leading_shape"])
        + (cache["k"] * cache["block_size"], width)
    )
    if dgathered.shape != expected:
        raise ValueError(f"dgathered shape {dgathered.shape} != expected {expected}")

    dx = xp.zeros(x_shape, dtype=dgathered.dtype)
    values = dgathered.reshape(batch, -1, width)
    batch_ids = xp.broadcast_to(xp.arange(batch)[:, None], flat_tokens.shape)
    xp.add.at(dx, (batch_ids, flat_tokens), values)
    return dx
