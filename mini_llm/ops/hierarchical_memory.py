"""Top-level hierarchical memory selector used before the Transformer trunk.

The long historical horizon is searched once using cheap block summaries.  Only
selected full-resolution historical tokens are reopened and concatenated with a
recent contiguous prefix and the bounded training target.  This module owns no
Transformer attention; it is deliberately a shallow model-boundary router.
"""

import math
from dataclasses import dataclass

from ..backend import xp, is_low_precision_dtype
from ..parameter import Parameter
from ..performance_profiler import performance_scope
from .context_blocks import HistoryBlockPooler
from .context_router import CausalQueryPooler
from .topk import selected_topk_softmax_forward, selected_topk_softmax_backward


@dataclass
class ActiveContext:
    """Bounded Transformer input assembled from long addressable memory."""

    embeddings: object
    token_ids: object
    position_ids: object
    source_indices: object
    target_start: int
    target_end: int
    selected_blocks: object
    route_weights: object


class HierarchicalMemoryRouter:
    """Select distant history blocks from a query formed only from recent context.

    Selection identities are discrete and fixed during the local backward, just
    like the repository's existing ContextRouter/MoE top-k convention.  Gradient
    flows through the selected softmax weights.  The model boundary uses those
    weights as a differentiable gate on reopened full-resolution embeddings.
    """

    def __init__(
        self,
        d_model,
        config,
        rng,
        input_std=0.02,
        name="memory_router",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.config = config
        self.router_dim = int(config.router_dim)
        self.top_k_blocks = int(config.top_k_blocks)

        self.query_pooler = CausalQueryPooler(
            d_model=self.d_model,
            query_window=int(config.router_query_length),
            strategy=config.query_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.query_pool",
            dtype=dtype,
        )
        self.history_pooler = HistoryBlockPooler(
            d_model=self.d_model,
            block_size=int(config.block_size),
            strategy=config.history_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.history_pool",
            dtype=dtype,
        )

        self.W_query = Parameter(
            xp.asarray(
                rng.normal(
                    (self.d_model, self.router_dim),
                    std=input_std,
                    dtype=dtype,
                )
            ),
            name=f"{name}.W_query",
        )
        self.W_history = Parameter(
            xp.asarray(
                rng.normal(
                    (self.d_model, self.router_dim),
                    std=input_std,
                    dtype=dtype,
                )
            ),
            name=f"{name}.W_history",
        )

    def parameters(self):
        return (
            [self.W_query, self.W_history]
            + self.query_pooler.parameters()
            + self.history_pooler.parameters()
        )

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, history_x, recent_x):
        """Return chronologically sorted selected blocks and their route weights.

        ``recent_x`` contains only pre-target tokens.  No target representation
        is accepted by this API, making the top-level causality rule structural
        rather than merely a masking convention.
        """
        if history_x.ndim != 3 or history_x.shape[-1] != self.d_model:
            raise ValueError("history_x must have shape (batch, history, d_model)")
        if recent_x.ndim != 3 or recent_x.shape[-1] != self.d_model:
            raise ValueError("recent_x must have shape (batch, recent, d_model)")
        if int(history_x.shape[0]) != int(recent_x.shape[0]):
            raise ValueError("history_x and recent_x batch dimensions must match")
        if int(recent_x.shape[1]) < int(self.config.router_query_length):
            raise ValueError("recent context is shorter than router_query_length")

        batch = int(history_x.shape[0])
        with performance_scope("memory_router.history_pool.forward"):
            history_pooled, history_cache = self.history_pooler.forward(history_x)
        n_blocks = int(history_pooled.shape[1])
        if n_blocks < self.top_k_blocks:
            raise ValueError(
                "distant history contains fewer complete blocks than top_k_blocks"
            )

        # Exactly one routing boundary: the end of the recent pre-target region.
        route_starts = xp.asarray([int(recent_x.shape[1])], dtype=xp.int64)
        with performance_scope("memory_router.query_pool.forward"):
            query_pooled_3d, query_cache = self.query_pooler.forward(
                recent_x, route_starts
            )
        query_pooled = query_pooled_3d[:, 0, :]

        with performance_scope("memory_router.projection.forward"):
            query_proj = query_pooled @ self.W_query.data
            history_proj = (
                history_pooled.reshape(-1, self.d_model) @ self.W_history.data
            ).reshape(batch, n_blocks, self.router_dim)

        # The score tensor is only [B, num_blocks].  As in ContextRouter, keep
        # low-precision projections but perform the tiny score product in FP32.
        with performance_scope("memory_router.score.forward"):
            if is_low_precision_dtype(query_proj.dtype):
                q_score = query_proj.astype("float32", copy=False)
                h_score = history_proj.astype("float32", copy=False)
            else:
                q_score = query_proj
                h_score = history_proj
            scores = xp.sum(h_score * q_score[:, None, :], axis=-1)
            scores = scores / math.sqrt(self.router_dim)

        with performance_scope("memory_router.topk.forward"):
            weights, selected, topk_cache = selected_topk_softmax_forward(
                scores,
                self.top_k_blocks,
                output_dtype=history_x.dtype,
            )
            # Top-k returns score order; the active sequence must instead be
            # chronological so compact tensor order agrees with causal order.
            sort_order = xp.argsort(selected, axis=-1)
            selected_sorted = xp.take_along_axis(selected, sort_order, axis=-1)
            weights_sorted = xp.take_along_axis(weights, sort_order, axis=-1)
            inverse_sort_order = xp.argsort(sort_order, axis=-1)

        cache = {
            "history_pooled": history_pooled,
            "query_pooled": query_pooled,
            "history_proj": history_proj,
            "query_proj": query_proj,
            "history_cache": history_cache,
            "query_cache": query_cache,
            "topk_cache": topk_cache,
            "scores": scores,
            "inverse_sort_order": inverse_sort_order,
        }
        return weights_sorted, selected_sorted, cache

    def backward(self, dweights_sorted, cache):
        """Backward through selected route weights into history/recent embeddings."""
        inverse_sort_order = cache["inverse_sort_order"]
        dweights = xp.take_along_axis(
            dweights_sorted, inverse_sort_order, axis=-1
        )

        with performance_scope("memory_router.topk.backward"):
            dscores = selected_topk_softmax_backward(
                dweights, cache["topk_cache"]
            )

        query_proj = cache["query_proj"]
        history_proj = cache["history_proj"]
        scale = 1.0 / math.sqrt(self.router_dim)
        work_dtype = dscores.dtype
        query_work = query_proj.astype(work_dtype, copy=False)
        history_work = history_proj.astype(work_dtype, copy=False)

        with performance_scope("memory_router.score.backward"):
            dquery_proj_work = xp.sum(
                dscores[..., None] * history_work, axis=1
            ) * scale
            dhistory_proj_work = (
                dscores[..., None] * query_work[:, None, :]
            ) * scale

        compute_dtype = cache["query_pooled"].dtype
        dquery_proj = dquery_proj_work.astype(compute_dtype, copy=False)
        dhistory_proj = dhistory_proj_work.astype(compute_dtype, copy=False)

        with performance_scope("memory_router.projection.backward"):
            query_pooled = cache["query_pooled"]
            history_pooled = cache["history_pooled"]
            self.W_query.grad += query_pooled.T @ dquery_proj
            self.W_history.grad += (
                history_pooled.reshape(-1, self.d_model).T
                @ dhistory_proj.reshape(-1, self.router_dim)
            )
            dquery_pooled = dquery_proj @ self.W_query.data.T
            dhistory_pooled = (
                dhistory_proj.reshape(-1, self.router_dim) @ self.W_history.data.T
            ).reshape(history_pooled.shape)

        with performance_scope("memory_router.query_pool.backward"):
            drecent = self.query_pooler.backward(
                dquery_pooled[:, None, :], cache["query_cache"]
            )
        with performance_scope("memory_router.history_pool.backward"):
            dhistory = self.history_pooler.backward(
                dhistory_pooled, cache["history_cache"]
            )
        return dhistory, drecent


def router_weight_gate(weights, top_k, scale):
    """Return the differentiable scalar gate for each selected exact block.

    ``gate = 1 + scale * (K*w - 1)`` keeps equal routing weights exactly neutral
    (gate=1).  At the default scale=1 it is simply ``K*w``.  This provides the
    selected-route learning signal required by hard top-k while preserving full
    token resolution; block summaries are never substituted for token content.
    """
    return 1.0 + float(scale) * (int(top_k) * weights - 1.0)


def router_weight_gate_backward(dgate, top_k, scale):
    return dgate * (float(scale) * int(top_k))
