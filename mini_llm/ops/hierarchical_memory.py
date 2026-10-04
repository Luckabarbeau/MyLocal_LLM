"""Hierarchical external-memory primitives used by 0058A through 0058C.

The retained 0058A/B path supports causal per-position pre-Transformer memory
reads. 0058C adds the production terminal-Landmark path: one recent-context
router selects old same-document blocks, exact selected tokens stay K/V-only,
and the deep Transformer still processes only the dense working window.
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
from .rmsnorm import RMSNorm
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
    terminal_memory: object = None


@dataclass
class TerminalMemoryContext:
    """Exact historical memory exposed only to the terminal Transformer query.

    The object is deliberately small: it retains only the selected exact token
    embeddings (normally 16x128 rows), their true source positions, and the raw
    selected router gate logits.  Attention backward deposits gradients here so
    the decoder can finish router/embedding backward after the deep trunk.
    """

    selected_embeddings: object
    selected_token_ids: object
    selected_position_ids: object
    selected_blocks: object
    gate_scores: object
    route_weights: object
    selected_valid: object
    route_valid: object
    terminal_rows: object
    # Reserved for the complementary deterministic/dilated external-history
    # sampler planned after the learned router baseline is profiled. Keeping it
    # in the context contract avoids another attention API rewrite later.
    deterministic_token_indices: object = None
    # Lightweight diagnostics from the actual grouped attention, distinct from
    # router top-k probabilities. These make it possible to tell whether the
    # Transformer truly used routed memory rather than merely selecting blocks.
    attention_history_mass: object = None
    attention_block_probs: object = None
    attention_max_token_prob: object = None
    d_selected_embeddings: object = None
    d_gate_scores: object = None

    def clear_backward(self):
        self.d_selected_embeddings = None
        self.d_gate_scores = None


class HierarchicalMemoryRouter:
    """Reusable causal block router for hierarchical memory.

    ``forward`` retains the 0058A/B vectorized per-position route. 0058C uses
    ``forward_terminal`` exactly once from the complete current working window.
    ``candidate_cutoffs`` always define the causal searchable region; document
    masks independently exclude left padding and earlier packed documents.

    Top-k identities are discrete. Gradients flow through selected gate scores,
    matching the repository's established retrieval/MoE convention.
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

    def forward_terminal(
        self, history_x, recent_x, *, history_valid_starts=None, terminal_rows=None
    ):
        """Route once from the complete recent working window.

        This is the 0058C training/inference semantic: the last valid causal
        working state asks which *older* same-document blocks should be opened
        for the next-token prediction.  The selected raw score is retained as
        the Landmark block gate; top-k softmax weights are diagnostic only.
        """
        batch = int(recent_x.shape[0])
        if terminal_rows is None:
            terminal_rows = xp.full(
                (batch,), int(recent_x.shape[1]) - 1, dtype=xp.int64
            )
        else:
            terminal_rows = xp.asarray(terminal_rows, dtype=xp.int64)
            if terminal_rows.shape != (batch,):
                raise ValueError("terminal_rows must have shape (batch,)")

        # CausalQueryPooler currently shares route boundaries across the batch.
        # Router-valid long-memory examples always have a complete dense working
        # window; short-document rows have no old memory and are masked below.
        route_starts = xp.asarray([int(recent_x.shape[1])], dtype=xp.int64)
        cutoffs = xp.asarray([int(history_x.shape[1])], dtype=xp.int64)
        weights, selected, selected_valid, route_valid, pooled, cache = self.forward(
            history_x, recent_x, route_starts, cutoffs,
            history_valid_starts=history_valid_starts,
            query_valid_starts=None,
        )
        selected_scores = xp.take_along_axis(cache["scores"], selected, axis=-1)
        selected_scores = xp.where(selected_valid, selected_scores, 0.0)
        cache["terminal_selected"] = selected
        cache["terminal_selected_valid"] = selected_valid
        cache["terminal_rows"] = terminal_rows
        return (
            weights[:, 0, :], selected[:, 0, :], selected_scores[:, 0, :],
            selected_valid[:, 0, :], route_valid[:, 0], cache,
        )

    def backward_terminal(self, dselected_scores, cache):
        """Backward raw selected Landmark gate logits into router parameters."""
        selected = cache["terminal_selected"]
        selected_valid = cache["terminal_selected_valid"]
        ds = xp.asarray(dselected_scores)[:, None, :]
        if ds.shape != selected.shape:
            raise ValueError("terminal gate-score gradient has incompatible shape")
        scores = cache["scores"]
        dscores = xp.zeros(scores.shape, dtype=(
            xp.float32 if is_low_precision_dtype(scores.dtype) else scores.dtype
        ))
        safe_selected = xp.where(selected_valid, selected, 0)
        # Only one route is used by 0058C.  scatter-add keeps tie/duplicate
        # semantics explicit even though top-k normally returns unique blocks.
        for slot in range(int(selected.shape[-1])):
            idx = safe_selected[:, 0, slot]
            vals = xp.where(selected_valid[:, 0, slot], ds[:, 0, slot], 0.0)
            xp.add.at(dscores[:, 0, :], (xp.arange(scores.shape[0]), idx), vals)
        return self._backward_scores(dscores, cache)

    def _backward_scores(self, dscores, cache, dquery_pooled_extra=None):
        dscores = xp.where(cache["candidate_mask"], dscores, 0.0)
        dscores = xp.where(cache["route_valid"][..., None], dscores, 0.0)

        query_proj = cache["query_proj"]
        history_proj = cache["history_proj"]
        scale = 1.0 / math.sqrt(self.router_dim)
        work_dtype = dscores.dtype
        query_work = query_proj.astype(work_dtype, copy=False)
        history_work = history_proj.astype(work_dtype, copy=False)
        with performance_scope("memory_router.score.backward"):
            dquery_proj_work = xp.matmul(dscores, history_work) * scale
            dhistory_proj_work = xp.swapaxes(dscores, 1, 2) @ query_work * scale

        compute_dtype = cache["query_pooled"].dtype
        dquery_proj = dquery_proj_work.astype(compute_dtype, copy=False)
        dhistory_proj = dhistory_proj_work.astype(compute_dtype, copy=False)
        with performance_scope("memory_router.projection.backward"):
            query_pooled = cache["query_pooled"]
            history_pooled = cache["history_pooled"]
            self.W_query.grad += query_pooled.reshape(-1, self.d_model).T @ dquery_proj.reshape(-1, self.router_dim)
            self.W_history.grad += history_pooled.reshape(-1, self.d_model).T @ dhistory_proj.reshape(-1, self.router_dim)
            dquery_pooled = (dquery_proj.reshape(-1, self.router_dim) @ self.W_query.data.T).reshape(query_pooled.shape)
            if dquery_pooled_extra is not None:
                dquery_pooled = dquery_pooled + dquery_pooled_extra.astype(dquery_pooled.dtype, copy=False)
            dhistory_pooled = (dhistory_proj.reshape(-1, self.router_dim) @ self.W_history.data.T).reshape(history_pooled.shape)
        with performance_scope("memory_router.query_pool.backward"):
            dquery_source = self.query_pooler.backward(dquery_pooled, cache["query_cache"])
        with performance_scope("memory_router.history_pool.backward"):
            dhistory = self.history_pooler.backward(dhistory_pooled, cache["history_cache"])
        return dhistory, dquery_source

    def backward(self, dweights, cache, dquery_pooled_extra=None):
        with performance_scope("memory_router.topk.backward"):
            dscores = selected_topk_softmax_backward(dweights, cache["topk_cache"])
            dscores = xp.where(cache["candidate_mask"], dscores, 0.0)
            dscores = xp.where(
                cache["route_valid"][..., None], dscores, 0.0
            )

        return self._backward_scores(
            dscores, cache, dquery_pooled_extra=dquery_pooled_extra
        )


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


class TerminalLandmarkAttention:
    """Landmark-gated exact historical K/V for one terminal query.

    The normal mixed-attention kernels compute every working-window row first.
    This module surgically replaces only the terminal row of the configured
    retrieval heads when a valid old-memory route exists.  Local/current K/V
    and selected historical blocks share a top-level normalization; each block
    then owns an independent token-level softmax:

        P(history token i in block b) = P(block b) * P(i | b).

    Historical tokens are K/V only.  They never become Transformer query rows,
    so MoE/RMSNorm/residual cost remains fixed by the 4k working window.
    """

    def __init__(
        self, d_model, d_head, n_query_heads, config, rng,
        input_std=0.02, rope_base=10_000.0, name="terminal_memory", dtype="float32",
    ):
        self.d_model = int(d_model)
        self.d_head = int(d_head)
        self.n_query_heads = int(n_query_heads)
        self.n_kv_heads = int(config.read_kv_heads)
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError("terminal memory query heads must divide memory KV heads")
        self.group_size = self.n_query_heads // self.n_kv_heads
        self.config = config
        self.rope_base = float(rope_base)
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.norm = RMSNorm(
            self.d_model, name=f"{name}.norm", dtype=dtype
        )
        kv_width = self.n_kv_heads * self.d_head
        self.W_k = Parameter(
            xp.asarray(rng.normal((self.d_model, kv_width), std=input_std, dtype=dtype)),
            name=f"{name}.W_k",
        )
        self.W_v = Parameter(
            xp.asarray(rng.normal((self.d_model, kv_width), std=input_std, dtype=dtype)),
            name=f"{name}.W_v",
        )

    def parameters(self):
        return self.norm.parameters() + [self.W_k, self.W_v]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    @staticmethod
    def _softmax_masked(scores, mask=None, axis=-1):
        work = scores.astype("float32", copy=False) if is_low_precision_dtype(scores.dtype) else scores
        if mask is not None:
            work = xp.where(mask, work, -xp.inf)
        row_max = xp.max(work, axis=axis, keepdims=True)
        finite = xp.isfinite(row_max)
        shifted = xp.where(finite, work - row_max, -xp.inf)
        expv = xp.exp(shifted)
        if mask is not None:
            expv = xp.where(mask, expv, 0.0)
        denom = xp.sum(expv, axis=axis, keepdims=True)
        return xp.where(denom > 0, expv / xp.maximum(denom, 1e-30), 0.0)

    def forward(self, q_subset, local_k, local_v, kv_head_indices, memory, *, return_cache=True):
        bsz, seq_len, n_heads, d_head = q_subset.shape
        if n_heads != self.n_query_heads or d_head != self.d_head:
            raise ValueError("terminal memory retrieval-head shape mismatch")
        if len(kv_head_indices) != n_heads:
            raise ValueError("kv_head_indices must match terminal memory query heads")

        route_valid = xp.asarray(memory.route_valid, dtype=bool)
        active_batches = xp.nonzero(route_valid)[0]
        if int(active_batches.size) == 0:
            return (None, None) if return_cache else None

        selected_x = memory.selected_embeddings
        batch, n_tokens, _ = selected_x.shape
        n_blocks = int(memory.selected_blocks.shape[-1])
        block_size = int(self.config.block_size)
        if n_tokens != n_blocks * block_size:
            raise ValueError("selected exact-token tensor does not match block layout")

        with performance_scope("memory.terminal.history_kv_project"):
            normed, norm_cache = self.norm.forward(selected_x)
            flat = normed.reshape(-1, self.d_model)
            mem_k_pre = (flat @ self.W_k.data).reshape(
                batch, n_tokens, self.n_kv_heads, self.d_head
            )
            mem_v = (flat @ self.W_v.data).reshape(
                batch, n_tokens, self.n_kv_heads, self.d_head
            )
        with performance_scope("memory.terminal.history_rope"):
            mem_k, mem_rope_cache = rope_forward(
                mem_k_pre, base=self.rope_base,
                position_ids=memory.selected_position_ids,
            )

        mem_map = xp.arange(n_heads, dtype=xp.int64) // self.group_size
        mem_k_h = mem_k[:, :, mem_map, :].transpose(0, 2, 1, 3)
        mem_v_h = mem_v[:, :, mem_map, :].transpose(0, 2, 1, 3)
        local_map = xp.asarray(kv_head_indices, dtype=xp.int64)
        local_k_h = local_k[:, :, local_map, :].transpose(0, 2, 1, 3)
        local_v_h = local_v[:, :, local_map, :].transpose(0, 2, 1, 3)

        terminal_rows = xp.asarray(memory.terminal_rows, dtype=xp.int64)
        batch_ids = xp.arange(batch, dtype=xp.int64)
        q_term = q_subset[batch_ids, terminal_rows, :, :]  # [B,H,D]

        # The terminal route is used only for complete working windows in the
        # production sampler.  A causal row mask keeps the primitive correct for
        # tiny validators and short-document edge cases.
        key_positions = xp.arange(seq_len, dtype=xp.int64)[None, None, :]
        local_valid = key_positions <= terminal_rows[:, None, None]
        local_valid = xp.broadcast_to(local_valid, (batch, n_heads, seq_len))

        with performance_scope("memory.terminal.local_logits"):
            q_work = q_term.astype("float32", copy=False)
            lk_work = local_k_h.astype("float32", copy=False)
            local_logits = xp.sum(q_work[:, :, None, :] * lk_work, axis=-1) * self.scale

        with performance_scope("memory.terminal.block_token_logits"):
            mk_work = mem_k_h.astype("float32", copy=False)
            token_logits = xp.sum(q_work[:, :, None, :] * mk_work, axis=-1) * self.scale
            token_logits = token_logits.reshape(batch, n_heads, n_blocks, block_size)
            token_valid = xp.broadcast_to(
                memory.selected_valid[:, None, :, None], token_logits.shape
            )
            inner_probs = self._softmax_masked(token_logits, token_valid, axis=-1)
            mv_work = mem_v_h.astype("float32", copy=False).reshape(
                batch, n_heads, n_blocks, block_size, self.d_head
            )
            block_values = xp.sum(inner_probs[..., None] * mv_work, axis=3)

        with performance_scope("memory.terminal.grouped_softmax"):
            gate_logits = (
                memory.gate_scores.astype("float32", copy=False)
                * float(self.config.router_weight_scale)
            )
            gates_h = xp.broadcast_to(gate_logits[:, None, :], (batch, n_heads, n_blocks))
            combined = xp.concatenate((local_logits, gates_h), axis=-1)
            combined_mask = xp.concatenate((
                local_valid,
                xp.broadcast_to(memory.selected_valid[:, None, :], (batch, n_heads, n_blocks)),
            ), axis=-1)
            top_probs = self._softmax_masked(combined, combined_mask, axis=-1)
            local_probs = top_probs[:, :, :seq_len]
            block_probs = top_probs[:, :, seq_len:]
            memory.attention_history_mass = xp.sum(block_probs, axis=-1)
            memory.attention_block_probs = block_probs
            memory.attention_max_token_prob = xp.max(inner_probs, axis=-1)

        with performance_scope("memory.terminal.output"):
            lv_work = local_v_h.astype("float32", copy=False)
            local_context = xp.sum(local_probs[..., None] * lv_work, axis=2)
            history_context = xp.sum(block_probs[..., None] * block_values, axis=2)
            terminal_context = local_context + history_context
            out_dtype = q_subset.dtype
            terminal_context = terminal_context.astype(out_dtype, copy=False)

        if not return_cache:
            return terminal_context
        cache = {
            "q_term": q_term,
            "local_k_h": local_k_h,
            "local_v_h": local_v_h,
            "mem_k_h": mem_k_h,
            "mem_v_h": mem_v_h,
            "local_probs": local_probs,
            "block_probs": block_probs,
            "inner_probs": inner_probs,
            "block_values": block_values,
            "local_valid": local_valid,
            "token_valid": token_valid,
            "normed": normed,
            "norm_cache": norm_cache,
            "mem_rope_cache": mem_rope_cache,
            "mem_map": mem_map,
            "local_map": local_map,
            "n_local_kv_heads": int(local_k.shape[2]),
            "terminal_rows": terminal_rows,
            "route_valid": route_valid,
        }
        return terminal_context, cache

    def backward(self, dterminal, cache, memory):
        """Backward the terminal grouped attention and accumulate memory grads."""
        q_term = cache["q_term"].astype("float32", copy=False)
        local_k = cache["local_k_h"].astype("float32", copy=False)
        local_v = cache["local_v_h"].astype("float32", copy=False)
        mem_k = cache["mem_k_h"].astype("float32", copy=False)
        mem_v = cache["mem_v_h"].astype("float32", copy=False)
        local_p = cache["local_probs"]
        block_p = cache["block_probs"]
        inner_p = cache["inner_probs"]
        block_values = cache["block_values"]
        dctx = dterminal.astype("float32", copy=False)
        batch, n_heads, seq_len, _ = local_k.shape
        n_blocks = int(block_p.shape[-1])
        block_size = int(self.config.block_size)

        # Top-level softmax over local exact tokens and historical block gates.
        dlocal_p = xp.sum(dctx[:, :, None, :] * local_v, axis=-1)
        dblock_p = xp.sum(dctx[:, :, None, :] * block_values, axis=-1)
        top_p = xp.concatenate((local_p, block_p), axis=-1)
        dtop_p = xp.concatenate((dlocal_p, dblock_p), axis=-1)
        dot = xp.sum(dtop_p * top_p, axis=-1, keepdims=True)
        dtop_logits = top_p * (dtop_p - dot)
        dlocal_logits = dtop_logits[:, :, :seq_len]
        dgate = dtop_logits[:, :, seq_len:].sum(axis=1) * float(self.config.router_weight_scale)
        dgate = xp.where(memory.selected_valid, dgate, 0.0)

        # Local value/logit paths.
        dlocal_v = local_p[..., None] * dctx[:, :, None, :]
        dq = xp.sum(dlocal_logits[..., None] * local_k, axis=2) * self.scale
        dlocal_k = dlocal_logits[..., None] * q_term[:, :, None, :] * self.scale

        # Block-value path followed by token softmax inside each selected block.
        dblock_values = block_p[..., None] * dctx[:, :, None, :]
        mem_v_blocks = mem_v.reshape(batch, n_heads, n_blocks, block_size, self.d_head)
        dinner_p = xp.sum(dblock_values[:, :, :, None, :] * mem_v_blocks, axis=-1)
        dinner_dot = xp.sum(dinner_p * inner_p, axis=-1, keepdims=True)
        dinner_logits = inner_p * (dinner_p - dinner_dot)
        dinner_logits = xp.where(cache["token_valid"], dinner_logits, 0.0)
        dmem_v_blocks = inner_p[..., None] * dblock_values[:, :, :, None, :]
        mem_k_blocks = mem_k.reshape(batch, n_heads, n_blocks, block_size, self.d_head)
        dq += xp.sum(
            dinner_logits[..., None] * mem_k_blocks, axis=(2, 3)
        ) * self.scale
        dmem_k_blocks = (
            dinner_logits[..., None] * q_term[:, :, None, None, :] * self.scale
        )
        dmem_k_h = dmem_k_blocks.reshape(batch, n_heads, n_blocks * block_size, self.d_head)
        dmem_v_h = dmem_v_blocks.reshape(batch, n_heads, n_blocks * block_size, self.d_head)

        # Reduce query-head memory gradients back into compact memory KV heads.
        dmem_k = xp.zeros(
            (batch, n_blocks * block_size, self.n_kv_heads, self.d_head), dtype="float32"
        )
        dmem_v = xp.zeros_like(dmem_k)
        for h in range(n_heads):
            kvh = int(cache["mem_map"][h])
            dmem_k[:, :, kvh, :] += dmem_k_h[:, h, :, :]
            dmem_v[:, :, kvh, :] += dmem_v_h[:, h, :, :]

        with performance_scope("memory.terminal.history_rope.backward"):
            dmem_k_pre = rope_backward(dmem_k, cache["mem_rope_cache"])
        normed = cache["normed"]
        flat_normed = normed.reshape(-1, self.d_model)
        dk2 = dmem_k_pre.reshape(-1, self.n_kv_heads * self.d_head).astype(
            self.W_k.data.dtype, copy=False
        )
        dv2 = dmem_v.reshape(-1, self.n_kv_heads * self.d_head).astype(
            self.W_v.data.dtype, copy=False
        )
        with performance_scope("memory.terminal.history_kv_project.backward"):
            self.W_k.grad += flat_normed.T @ dk2
            self.W_v.grad += flat_normed.T @ dv2
            dnorm = (dk2 @ self.W_k.data.T + dv2 @ self.W_v.data.T).reshape(normed.shape)
            dselected = self.norm.backward(dnorm, cache["norm_cache"])

        # Current-window q/k/v gradients are returned to GQAAttention. Historical
        # embedding/router gradients are deliberately side-banded to the decoder.
        dq_full = xp.zeros(
            (batch, seq_len, n_heads, self.d_head), dtype="float32"
        )
        dk_full = xp.zeros(
            (batch, seq_len, int(cache["n_local_kv_heads"]), self.d_head),
            dtype="float32",
        )
        dv_full = xp.zeros_like(dk_full)
        batch_ids = xp.arange(batch, dtype=xp.int64)
        rows = cache["terminal_rows"]
        dq_full[batch_ids, rows, :, :] = dq
        for h in range(n_heads):
            kvh = int(cache["local_map"][h])
            dk_full[:, :, kvh, :] += dlocal_k[:, h, :, :]
            dv_full[:, :, kvh, :] += dlocal_v[:, h, :, :]

        memory.d_selected_embeddings = dselected
        memory.d_gate_scores = dgate
        return dq_full, dk_full, dv_full
