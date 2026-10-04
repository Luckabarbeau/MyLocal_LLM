"""Causal per-position external memory for the 0058A hierarchy.

The expensive Transformer processes only the dense neighboring training window.
For every prediction row, a cheap router summarizes the causal recent history,
selects only blocks older than that recent window, and an indexed cross-attention
read reopens the selected exact historical tokens.  Different prediction rows
may therefore fetch different blocks without concatenating their union into the
Transformer sequence.
"""

import math
from dataclasses import dataclass

from ..backend import xp, is_low_precision_dtype
from ..parameter import Parameter
from ..performance_profiler import performance_scope
from .attention_selection import KeySelectionPlan
from .context_blocks import HistoryBlockPooler, block_token_indices
from .context_router import CausalQueryPooler
from .indexed_attention import indexed_attention_forward, indexed_attention_backward
from .rope import rope_forward, rope_backward
from .topk import selected_topk_softmax_forward, selected_topk_softmax_backward


@dataclass
class ActiveContext:
    """Dense Transformer input plus causal external-memory routing metadata."""

    embeddings: object
    token_ids: object
    position_ids: object
    source_indices: object
    target_start: int
    target_end: int
    selected_blocks: object
    route_weights: object
    selected_valid: object
    route_valid: object


class HierarchicalMemoryRouter:
    """Vectorized causal block router for every row in a dense training window.

    ``history_x`` contains the fixed historical store preceding the training
    window. ``query_source_x`` contains enough tail history plus the current
    window to form each causal recent-history query.  ``candidate_cutoffs`` are
    absolute sample-local source positions: a block is eligible for route ``r``
    only when ``block_end <= candidate_cutoffs[r]``.

    Top-k identities are discrete.  Gradients flow through selected softmax
    weights, matching the repository's established retrieval/MoE convention.
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

    def forward(
        self, history_x, query_source_x, route_starts, candidate_cutoffs,
        *, history_valid_starts=None, query_valid_starts=None,
    ):
        if history_x.ndim != 3 or history_x.shape[-1] != self.d_model:
            raise ValueError("history_x must have shape (batch, history, d_model)")
        if query_source_x.ndim != 3 or query_source_x.shape[-1] != self.d_model:
            raise ValueError(
                "query_source_x must have shape (batch, query_source, d_model)"
            )
        if int(history_x.shape[0]) != int(query_source_x.shape[0]):
            raise ValueError("history/query batch dimensions must match")
        route_starts = xp.asarray(route_starts, dtype=xp.int64)
        candidate_cutoffs = xp.asarray(candidate_cutoffs, dtype=xp.int64)
        if route_starts.ndim != 1 or candidate_cutoffs.shape != route_starts.shape:
            raise ValueError("route_starts/candidate_cutoffs must be matching 1D arrays")
        if route_starts.size and (
            bool(xp.any(route_starts <= 0))
            or bool(xp.any(route_starts > query_source_x.shape[1]))
        ):
            raise ValueError("route starts must lie inside the causal query source")

        batch = int(history_x.shape[0])
        if history_valid_starts is None:
            history_valid_starts = xp.zeros((batch,), dtype=xp.int64)
        else:
            history_valid_starts = xp.asarray(history_valid_starts, dtype=xp.int64)
            if history_valid_starts.shape != (batch,):
                raise ValueError("history_valid_starts must have shape (batch,)")
            if bool(xp.any(history_valid_starts < 0)) or bool(
                xp.any(history_valid_starts > history_x.shape[1])
            ):
                raise ValueError("history_valid_starts lie outside history_x")

        with performance_scope("memory_router.history_pool.forward"):
            history_pooled, history_cache = self.history_pooler.forward(history_x)
        n_blocks = int(history_pooled.shape[1])
        if n_blocks < self.top_k_blocks:
            raise ValueError("history contains fewer complete blocks than top_k_blocks")

        with performance_scope("memory_router.query_pool.forward"):
            query_pooled, query_cache = self.query_pooler.forward(
                query_source_x, route_starts, valid_starts=query_valid_starts
            )
        n_routes = int(query_pooled.shape[1])

        with performance_scope("memory_router.projection.forward"):
            query_proj = (
                query_pooled.reshape(-1, self.d_model) @ self.W_query.data
            ).reshape(batch, n_routes, self.router_dim)
            history_proj = (
                history_pooled.reshape(-1, self.d_model) @ self.W_history.data
            ).reshape(batch, n_blocks, self.router_dim)

        with performance_scope("memory_router.score.forward"):
            if is_low_precision_dtype(query_proj.dtype):
                q_score = query_proj.astype("float32", copy=False)
                h_score = history_proj.astype("float32", copy=False)
            else:
                q_score = query_proj
                h_score = history_proj
            scores = xp.matmul(q_score, xp.swapaxes(h_score, 1, 2))
            scores = scores / math.sqrt(self.router_dim)

        # The mask is the temporal definition of the architecture.  The recent
        # causal window determines the query but is never itself retrievable.
        block_ids = xp.arange(n_blocks, dtype=xp.int64)
        block_starts = block_ids * int(self.config.block_size)
        block_ends = (block_ids + 1) * int(self.config.block_size)
        old_enough = (
            block_ends[None, :, None]
            <= candidate_cutoffs[None, None, :]
        )
        old_enough = xp.swapaxes(old_enough, 1, 2)  # [1,R,N]
        inside_document = (
            block_starts[None, :] >= history_valid_starts[:, None]
        )[:, None, :]
        candidate_mask = old_enough & inside_document
        candidate_counts = xp.sum(candidate_mask, axis=2)
        route_valid = candidate_counts > 0

        with performance_scope("memory_router.topk.forward"):
            weights, selected, topk_cache = selected_topk_softmax_forward(
                scores,
                self.top_k_blocks,
                output_dtype=history_x.dtype,
                candidate_mask=candidate_mask,
            )
            selected_valid = topk_cache["selected_valid"]

        cache = {
            "history_pooled": history_pooled,
            "query_pooled": query_pooled,
            "history_proj": history_proj,
            "query_proj": query_proj,
            "history_cache": history_cache,
            "query_cache": query_cache,
            "topk_cache": topk_cache,
            "candidate_mask": candidate_mask,
            "route_valid": route_valid,
            "selected_valid": selected_valid,
            "scores": scores,
        }
        return weights, selected, selected_valid, route_valid, query_pooled, cache

    def backward(self, dweights, cache, dquery_pooled_extra=None):
        with performance_scope("memory_router.topk.backward"):
            dscores = selected_topk_softmax_backward(dweights, cache["topk_cache"])
            dscores = xp.where(cache["candidate_mask"], dscores, 0.0)
            dscores = xp.where(
                cache["route_valid"][..., None], dscores, 0.0
            )

        query_proj = cache["query_proj"]
        history_proj = cache["history_proj"]
        scale = 1.0 / math.sqrt(self.router_dim)
        work_dtype = dscores.dtype
        query_work = query_proj.astype(work_dtype, copy=False)
        history_work = history_proj.astype(work_dtype, copy=False)

        with performance_scope("memory_router.score.backward"):
            dquery_proj_work = xp.matmul(dscores, history_work) * scale
            dhistory_proj_work = (
                xp.swapaxes(dscores, 1, 2) @ query_work
            ) * scale

        compute_dtype = cache["query_pooled"].dtype
        dquery_proj = dquery_proj_work.astype(compute_dtype, copy=False)
        dhistory_proj = dhistory_proj_work.astype(compute_dtype, copy=False)

        with performance_scope("memory_router.projection.backward"):
            query_pooled = cache["query_pooled"]
            history_pooled = cache["history_pooled"]
            self.W_query.grad += (
                query_pooled.reshape(-1, self.d_model).T
                @ dquery_proj.reshape(-1, self.router_dim)
            )
            self.W_history.grad += (
                history_pooled.reshape(-1, self.d_model).T
                @ dhistory_proj.reshape(-1, self.router_dim)
            )
            dquery_pooled = (
                dquery_proj.reshape(-1, self.router_dim) @ self.W_query.data.T
            ).reshape(query_pooled.shape)
            if dquery_pooled_extra is not None:
                if dquery_pooled_extra.shape != dquery_pooled.shape:
                    raise ValueError("dquery_pooled_extra has incompatible shape")
                dquery_pooled = dquery_pooled + dquery_pooled_extra.astype(
                    dquery_pooled.dtype, copy=False
                )
            dhistory_pooled = (
                dhistory_proj.reshape(-1, self.router_dim) @ self.W_history.data.T
            ).reshape(history_pooled.shape)

        with performance_scope("memory_router.query_pool.backward"):
            dquery_source = self.query_pooler.backward(
                dquery_pooled, cache["query_cache"]
            )
        with performance_scope("memory_router.history_pool.backward"):
            dhistory = self.history_pooler.backward(
                dhistory_pooled, cache["history_cache"]
            )
        return dhistory, dquery_source


def build_external_memory_plan(
    selected_blocks,
    selected_weights,
    selected_valid,
    *,
    block_size,
    weight_scale=1.0,
    weight_eps=1e-8,
):
    """Expand per-position block choices into a shared-head exact-token plan.

    The plan stores indices once with head dimension one.  Indexed attention
    broadcasts it over memory-read query heads without duplicating the index
    tensor.  Invalid early routes keep an in-range placeholder but receive an
    all-false validity mask and therefore exactly zero probability/gradient.
    """
    if selected_blocks.shape != selected_weights.shape or selected_blocks.ndim != 3:
        raise ValueError("selected blocks/weights must have shape (B,T,Kblocks)")
    batch, n_routes, n_selected = selected_blocks.shape
    selected_valid = xp.asarray(selected_valid, dtype=bool)
    if selected_valid.shape != selected_blocks.shape:
        raise ValueError("selected_valid must match selected block shape")
    block_size = int(block_size)
    token_indices = block_token_indices(selected_blocks, block_size).reshape(
        batch, n_routes, n_selected * block_size
    )
    key_indices = token_indices[:, None, :, :]
    token_valid = xp.repeat(selected_valid, block_size, axis=-1)
    valid_mask = token_valid[:, None, :, :]

    weights_work = (
        selected_weights.astype("float32", copy=False)
        if is_low_precision_dtype(selected_weights.dtype)
        else selected_weights
    )
    block_bias = float(weight_scale) * xp.log(weights_work + float(weight_eps))
    token_bias = xp.repeat(block_bias, block_size, axis=-1)[:, None, :, :]
    token_bias = xp.where(valid_mask, token_bias, 0.0)
    logit_bias = token_bias.astype(selected_weights.dtype, copy=False)
    return KeySelectionPlan(key_indices, valid_mask, logit_bias)


def external_memory_bias_backward(
    dlogit_bias,
    selected_weights,
    selected_valid,
    *,
    block_size,
    weight_scale=1.0,
    weight_eps=1e-8,
):
    """Reduce exact-token/read-head logit-bias gradients to router weights."""
    if dlogit_bias is None:
        return xp.zeros(selected_weights.shape, dtype="float32")
    batch, n_routes, n_selected = selected_weights.shape
    block_size = int(block_size)
    if dlogit_bias.ndim != 4 or dlogit_bias.shape[0] != batch:
        raise ValueError("dlogit_bias has incompatible shape")
    if int(dlogit_bias.shape[2]) != n_routes:
        raise ValueError("dlogit_bias route dimension mismatch")
    if int(dlogit_bias.shape[3]) != n_selected * block_size:
        raise ValueError("dlogit_bias key dimension mismatch")

    # Sum over memory-read query heads and over the exact tokens belonging to
    # each selected block.  The selected logit prior is alpha*log(w+eps).
    dbias = dlogit_bias.reshape(
        batch,
        int(dlogit_bias.shape[1]),
        n_routes,
        n_selected,
        block_size,
    ).sum(axis=(1, 4))
    weights_work = selected_weights.astype(dbias.dtype, copy=False)
    dweights = dbias * (float(weight_scale) / (weights_work + float(weight_eps)))
    dweights = xp.where(selected_valid, dweights, 0.0)
    return dweights


class ExternalMemoryReader:
    """One sparse exact-token cross-attention read before the deep trunk."""

    def __init__(
        self,
        d_model,
        d_head,
        config,
        rng,
        input_std=0.02,
        output_std=None,
        rope_base=10_000.0,
        name="memory_reader",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.d_head = int(d_head)
        self.config = config
        self.rope_base = float(rope_base)
        self.n_q_heads = int(config.read_heads)
        self.n_kv_heads = int(config.read_kv_heads)
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("memory read heads must be divisible by KV heads")
        self.q_width = self.n_q_heads * self.d_head
        self.kv_width = self.n_kv_heads * self.d_head
        output_std = input_std if output_std is None else float(output_std)

        def param(shape, std, suffix):
            return Parameter(
                xp.asarray(rng.normal(shape, std=std, dtype=dtype)),
                name=f"{name}.{suffix}",
            )

        self.W_q = param((self.d_model, self.q_width), input_std, "W_q")
        self.W_k = param((self.d_model, self.kv_width), input_std, "W_k")
        self.W_v = param((self.d_model, self.kv_width), input_std, "W_v")
        self.W_o = param((self.q_width, self.d_model), output_std, "W_o")

    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(
        self,
        history_x,
        current_x,
        selected_blocks,
        selected_weights,
        selected_valid,
        *,
        history_position_ids=None,
        query_position_ids=None,
        return_cache=True,
    ):
        """Read exact routed history for every current autoregressive row.

        The coarse router query and the fine memory-read query are deliberately
        distinct.  Routing summarizes the recent history; exact-token attention
        queries from the current token representation.  RoPE uses the original
        source positions so selected historical tokens retain temporal distance.
        """
        batch, history_length, _ = history_x.shape
        q_batch, query_length, _ = current_x.shape
        if q_batch != batch:
            raise ValueError("history/current batch dimensions must match")
        if selected_blocks.shape[:2] != (batch, query_length):
            raise ValueError("selected block routes must match current rows")
        if selected_valid.shape != selected_blocks.shape:
            raise ValueError("selected_valid must match selected block routes")
        if history_position_ids is None:
            history_position_ids = xp.arange(history_length, dtype=xp.int64)
        if query_position_ids is None:
            query_position_ids = xp.arange(query_length, dtype=xp.int64)

        with performance_scope("memory_reader.qkv.forward"):
            q_raw = (current_x.reshape(-1, self.d_model) @ self.W_q.data).reshape(
                batch, query_length, self.n_q_heads, self.d_head
            )
            k_raw = (history_x.reshape(-1, self.d_model) @ self.W_k.data).reshape(
                batch, history_length, self.n_kv_heads, self.d_head
            )
            v = (history_x.reshape(-1, self.d_model) @ self.W_v.data).reshape(
                batch, history_length, self.n_kv_heads, self.d_head
            )
        with performance_scope("memory_reader.rope.forward"):
            q, q_rope_cache = rope_forward(
                q_raw, base=self.rope_base, position_ids=query_position_ids
            )
            k, k_rope_cache = rope_forward(
                k_raw, base=self.rope_base, position_ids=history_position_ids
            )
        del q_raw, k_raw

        with performance_scope("memory_reader.plan.forward"):
            plan = build_external_memory_plan(
                selected_blocks,
                selected_weights,
                selected_valid,
                block_size=int(self.config.block_size),
                weight_scale=float(self.config.router_weight_scale),
                weight_eps=float(self.config.reader_weight_eps),
            )

        with performance_scope("memory_reader.attention.forward"):
            if return_cache:
                context, attention_cache = indexed_attention_forward(
                    q,
                    k,
                    v,
                    plan,
                    return_cache=True,
                    query_chunk_size=int(self.config.read_query_chunk),
                )
            else:
                context = indexed_attention_forward(
                    q,
                    k,
                    v,
                    plan,
                    return_cache=False,
                    query_chunk_size=int(self.config.read_query_chunk),
                )
                attention_cache = None

        with performance_scope("memory_reader.output.forward"):
            context_flat = context.reshape(-1, self.q_width)
            context_compute = (
                context_flat.astype(self.W_o.data.dtype, copy=False)
                if is_low_precision_dtype(self.W_o.data.dtype)
                else context_flat
            )
            output = (context_compute @ self.W_o.data).reshape(
                batch, query_length, self.d_model
            )

        if not return_cache:
            return output
        cache = {
            "attention_cache": attention_cache,
            "current_x": current_x,
            "context": context,
            "selected_weights": selected_weights,
            "selected_valid": selected_valid,
            "q_rope_cache": q_rope_cache,
            "k_rope_cache": k_rope_cache,
            "history_shape": tuple(history_x.shape),
        }
        return output, cache

    def backward(self, doutput, cache, history_x_replay):
        batch, query_length, _ = doutput.shape
        context = cache.pop("context")
        context_flat = context.reshape(-1, self.q_width)
        doutput_compute = (
            doutput.astype(self.W_o.data.dtype, copy=False)
            if is_low_precision_dtype(self.W_o.data.dtype)
            else doutput
        )
        doutput_flat = doutput_compute.reshape(-1, self.d_model)

        with performance_scope("memory_reader.output.backward"):
            self.W_o.grad += context_flat.astype(
                self.W_o.data.dtype, copy=False
            ).T @ doutput_flat
            dcontext = (doutput_flat @ self.W_o.data.T).reshape(context.shape)

        with performance_scope("memory_reader.attention.backward"):
            dq_rot, dk_rot, dv, dlogit_bias = indexed_attention_backward(
                dcontext, cache.pop("attention_cache")
            )
        with performance_scope("memory_reader.rope.backward"):
            dq = rope_backward(dq_rot, cache.pop("q_rope_cache"))
            dk = rope_backward(dk_rot, cache.pop("k_rope_cache"))

        current_x = cache.pop("current_x")
        with performance_scope("memory_reader.qkv.backward"):
            dq2 = dq.reshape(-1, self.q_width).astype(
                self.W_q.data.dtype, copy=False
            )
            dk2 = dk.reshape(-1, self.kv_width).astype(
                self.W_k.data.dtype, copy=False
            )
            dv2 = dv.reshape(-1, self.kv_width).astype(
                self.W_v.data.dtype, copy=False
            )
            current2 = current_x.reshape(-1, self.d_model)
            history2 = history_x_replay.reshape(-1, self.d_model)
            self.W_q.grad += current2.T @ dq2
            self.W_k.grad += history2.T @ dk2
            self.W_v.grad += history2.T @ dv2
            dcurrent = (dq2 @ self.W_q.data.T).reshape(current_x.shape)
            dhistory = (
                dk2 @ self.W_k.data.T + dv2 @ self.W_v.data.T
            ).reshape(history_x_replay.shape)

        with performance_scope("memory_reader.plan.backward"):
            dweights = external_memory_bias_backward(
                dlogit_bias,
                cache.pop("selected_weights"),
                cache.pop("selected_valid"),
                block_size=int(self.config.block_size),
                weight_scale=float(self.config.router_weight_scale),
                weight_eps=float(self.config.reader_weight_eps),
            )
        return dhistory, dcurrent, dweights
