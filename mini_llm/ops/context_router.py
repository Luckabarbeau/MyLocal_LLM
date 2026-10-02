"""Learned causal block router for long-context retrieval.

The router answers a deliberately narrow question:

    Given the recent hidden-state context available *before* a routing
    boundary, which complete distant history blocks are most relevant?

Compressed block vectors are only used to search.  The selected block IDs are
later expanded back to full-resolution tokens/KV by ``context_blocks``.
"""

import math

from ..backend import xp, is_low_precision_dtype
from ..parameter import Parameter
from .context_blocks import HistoryBlockPooler, complete_block_count
from .topk import selected_topk_softmax_forward, selected_topk_softmax_backward


def eligible_route_starts(seq_len, config):
    """Return routing boundaries with enough causally eligible history.

    A route beginning at token ``s`` may only use query tokens ``< s`` and may
    only select history blocks whose end satisfies

        block_end <= s - exclude_recent_tokens.

    Boundaries with fewer than ``top_k_blocks`` candidates are omitted.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return xp.zeros((0,), dtype=xp.int64)

    starts = xp.arange(
        int(config.routing_stride), seq_len, int(config.routing_stride),
        dtype=xp.int64,
    )
    if starts.size == 0:
        return starts

    usable_history = starts - int(config.exclude_recent_tokens)
    candidate_counts = xp.maximum(usable_history, 0) // int(config.history_block_size)
    return starts[candidate_counts >= int(config.top_k_blocks)]


class CausalQueryPooler:
    """Pool a recent causal window immediately preceding each route start."""

    def __init__(
        self,
        d_model,
        query_window,
        strategy="mean",
        rng=None,
        input_std=0.02,
        name="query_pool",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.query_window = int(query_window)
        self.strategy = str(strategy)
        if self.d_model <= 0 or self.query_window <= 0:
            raise ValueError("d_model and query_window must be positive")
        if self.strategy not in {"mean", "last", "learned"}:
            raise ValueError("unsupported query pooling strategy")

        self.score_param = None
        if self.strategy == "learned":
            if rng is None:
                raise ValueError("learned query pooling requires rng")
            data = xp.asarray(
                rng.normal((self.d_model,), std=input_std, dtype=dtype)
            )
            self.score_param = Parameter(data, name=f"{name}.score")

    def parameters(self):
        return [] if self.score_param is None else [self.score_param]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def _window_indices(self, route_starts):
        offsets = xp.arange(self.query_window, dtype=route_starts.dtype)
        raw = route_starts[:, None] - self.query_window + offsets[None, :]
        valid = raw >= 0
        clipped = xp.maximum(raw, 0)
        return clipped, valid

    def forward(self, x, route_starts):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (batch, seq_len, d_model)")
        if route_starts.ndim != 1:
            raise ValueError("route_starts must be one-dimensional")
        if route_starts.size:
            if bool(xp.any(route_starts <= 0)) or bool(xp.any(route_starts > x.shape[1])):
                raise ValueError("route starts must satisfy 0 < start <= seq_len")

        batch = x.shape[0]
        n_routes = int(route_starts.shape[0])
        if n_routes == 0:
            pooled = xp.zeros((batch, 0, self.d_model), dtype=x.dtype)
            return pooled, {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "indices": xp.zeros((0, self.query_window), dtype=xp.int64),
                "valid": xp.zeros((0, self.query_window), dtype=bool),
            }

        indices, valid = self._window_indices(route_starts)
        windows = x[:, indices, :]  # (B, R, W, D)
        valid_f = valid.astype(windows.dtype, copy=False)

        if self.strategy == "last":
            pooled = x[:, route_starts - 1, :]
            cache = {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "indices": indices,
                "valid": valid,
            }
            return pooled, cache

        if self.strategy == "mean":
            counts = xp.sum(valid_f, axis=1).reshape(1, n_routes, 1)
            pooled = xp.sum(windows * valid_f[None, :, :, None], axis=2) / counts
            cache = {
                "x_shape": x.shape,
                "route_starts": route_starts,
                "indices": indices,
                "valid": valid,
                "counts": counts,
            }
            return pooled, cache

        scale = 1.0 / math.sqrt(self.d_model)
        windows_2d = windows.reshape(-1, self.d_model)
        scores = (windows_2d @ self.score_param.data).reshape(
            batch, n_routes, self.query_window
        ) * scale
        scores_work = (
            scores.astype("float32", copy=False)
            if is_low_precision_dtype(scores.dtype)
            else scores
        )
        neg_inf = xp.asarray(-xp.inf, dtype=scores_work.dtype)
        scores_work = xp.where(valid[None, :, :], scores_work, neg_inf)
        max_scores = xp.max(scores_work, axis=2, keepdims=True)
        exp_scores = xp.where(
            valid[None, :, :], xp.exp(scores_work - max_scores), 0.0
        )
        alpha_work = exp_scores / xp.sum(exp_scores, axis=2, keepdims=True)
        alpha = (
            alpha_work.astype(x.dtype, copy=False)
            if is_low_precision_dtype(x.dtype)
            else alpha_work
        )
        pooled = xp.sum(windows * alpha[..., None], axis=2)
        cache = {
            "x_shape": x.shape,
            "x": x,
            "route_starts": route_starts,
            "indices": indices,
            "valid": valid,
            # Re-gather the query windows during backward instead of keeping
            # the large [B,R,W,D] tensor alive for the whole layer backward.
            "alpha_work": alpha_work,
            "scale": scale,
        }
        return pooled, cache

    def backward(self, dpooled, cache):
        x_shape = tuple(cache["x_shape"])
        batch, _, d_model = x_shape
        route_starts = cache["route_starts"]
        n_routes = int(route_starts.shape[0])
        if dpooled.shape != (batch, n_routes, d_model):
            raise ValueError("dpooled has incompatible shape")

        grad_dtype = (
            "float32" if is_low_precision_dtype(dpooled.dtype) else dpooled.dtype
        )
        dx = xp.zeros(x_shape, dtype=grad_dtype)
        if n_routes == 0:
            return dx

        if self.strategy == "last":
            batch_ids = xp.broadcast_to(
                xp.arange(batch)[:, None], (batch, n_routes)
            )
            token_ids = xp.broadcast_to(
                (route_starts - 1)[None, :], (batch, n_routes)
            )
            xp.add.at(
                dx,
                (batch_ids, token_ids),
                dpooled.astype(grad_dtype, copy=False),
            )
            return dx

        indices = cache["indices"]
        valid = cache["valid"]
        if self.strategy == "mean":
            counts = cache["counts"]
            dwindow = dpooled[:, :, None, :] / counts[:, :, None, :]
            dwindow = dwindow * valid[None, :, :, None]
        else:
            windows = cache["x"][:, indices, :]
            alpha_work = cache["alpha_work"]
            scale = cache["scale"]
            alpha_compute = (
                alpha_work.astype(dpooled.dtype, copy=False)
                if is_low_precision_dtype(dpooled.dtype)
                else alpha_work
            )
            dwindow = alpha_compute[..., None] * dpooled[:, :, None, :]

            dpooled_work = dpooled.astype("float32", copy=False)
            windows_work = windows.astype("float32", copy=False)
            dalpha = xp.sum(windows_work * dpooled_work[:, :, None, :], axis=-1)
            correction = xp.sum(alpha_work * dalpha, axis=2, keepdims=True)
            dscores = alpha_work * (dalpha - correction)
            dscores = xp.where(valid[None, :, :], dscores, 0.0)

            score_vector_work = self.score_param.data.astype("float32", copy=False)
            score_path = dscores[..., None] * score_vector_work * scale
            dwindow += score_path.astype(dwindow.dtype, copy=False)
            grad_score = xp.sum(
                dscores[..., None] * windows_work, axis=(0, 1, 2)
            ) * scale
            self.score_param.grad += grad_score

        flat_indices = xp.broadcast_to(
            indices[None, :, :], (batch, n_routes, self.query_window)
        ).reshape(batch, -1)
        values = dwindow.reshape(batch, -1, d_model)
        batch_ids = xp.broadcast_to(xp.arange(batch)[:, None], flat_indices.shape)
        xp.add.at(
            dx,
            (batch_ids, flat_indices),
            values.astype(grad_dtype, copy=False),
        )
        return dx


class ContextRouter:
    """Select distant history blocks from the current causal working state."""

    def __init__(
        self,
        d_model,
        config,
        rng,
        input_std=0.02,
        name="context_router",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.config = config
        self.router_dim = int(config.router_dim)
        self.num_queries = int(config.num_queries)

        self.query_pooler = CausalQueryPooler(
            d_model=self.d_model,
            query_window=config.query_window,
            strategy=config.query_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.query_pool",
            dtype=dtype,
        )
        self.history_pooler = HistoryBlockPooler(
            d_model=self.d_model,
            block_size=config.history_block_size,
            strategy=config.history_pooling,
            rng=rng,
            input_std=input_std,
            name=f"{name}.history_pool",
            dtype=dtype,
        )

        query_data = xp.asarray(
            rng.normal(
                (self.d_model, self.num_queries * self.router_dim),
                std=input_std,
                dtype=dtype,
            )
        )
        history_data = xp.asarray(
            rng.normal(
                (self.d_model, self.router_dim), std=input_std, dtype=dtype
            )
        )
        self.W_query = Parameter(query_data, name=f"{name}.W_query")
        self.W_history = Parameter(history_data, name=f"{name}.W_history")

    def parameters(self):
        return (
            [self.W_query, self.W_history]
            + self.query_pooler.parameters()
            + self.history_pooler.parameters()
        )

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (batch, seq_len, d_model)")

        batch, seq_len, _ = x.shape
        route_starts = eligible_route_starts(seq_len, self.config)
        query_pooled, query_cache = self.query_pooler.forward(x, route_starts)
        history_pooled, history_cache = self.history_pooler.forward(x)
        n_routes = int(route_starts.shape[0])
        n_blocks = int(history_pooled.shape[1])

        if n_routes == 0:
            shape = (batch, 0, self.num_queries, self.config.top_k_blocks)
            weights = xp.zeros(shape, dtype=x.dtype)
            selected = xp.zeros(shape, dtype=xp.int64)
            cache = {
                "x": x,
                "route_starts": route_starts,
                "query_pooled": query_pooled,
                "history_pooled": history_pooled,
                "query_cache": query_cache,
                "history_cache": history_cache,
                "scores": xp.zeros(
                    (batch, 0, self.num_queries, n_blocks), dtype=x.dtype
                ),
                "candidate_mask": xp.zeros((0, n_blocks), dtype=bool),
                "topk_cache": None,
            }
            return weights, selected, route_starts, cache

        query_proj = (query_pooled.reshape(-1, self.d_model) @ self.W_query.data).reshape(
            batch, n_routes, self.num_queries, self.router_dim
        )
        history_proj = (
            history_pooled.reshape(-1, self.d_model) @ self.W_history.data
        ).reshape(batch, n_blocks, self.router_dim)

        query_3d = query_proj.reshape(
            batch, n_routes * self.num_queries, self.router_dim
        )
        history_t = xp.swapaxes(history_proj, 1, 2)

        # CuPy's generic N-D matmul path does not currently understand BF16
        # (ml_dtypes dtype code ``E``), even though its 2-D BF16 GEMM path is
        # supported.  Router score tensors are tiny compared with attention
        # activations, so perform this batched query/history product in FP32
        # for *all* low-precision model dtypes.  This also keeps Top-K logits
        # numerically stable and matches the existing MoE routing convention.
        if is_low_precision_dtype(query_3d.dtype):
            query_score = query_3d.astype("float32", copy=False)
            history_score_t = history_t.astype("float32", copy=False)
        else:
            query_score = query_3d
            history_score_t = history_t
        scores = xp.matmul(query_score, history_score_t).reshape(
            batch, n_routes, self.num_queries, n_blocks
        ) / math.sqrt(self.router_dim)

        block_ends = (
            xp.arange(n_blocks, dtype=route_starts.dtype) + 1
        ) * int(self.config.history_block_size)
        candidate_mask = block_ends[None, :] <= (
            route_starts[:, None] - int(self.config.exclude_recent_tokens)
        )

        scores_work = (
            scores.astype("float32", copy=False)
            if is_low_precision_dtype(scores.dtype)
            else scores
        )
        masked_scores = xp.where(
            candidate_mask[None, :, None, :], scores_work, -xp.inf
        )
        weights, selected, topk_cache = selected_topk_softmax_forward(
            masked_scores,
            int(self.config.top_k_blocks),
            output_dtype=x.dtype,
        )

        cache = {
            "x": x,
            "route_starts": route_starts,
            "query_pooled": query_pooled,
            "history_pooled": history_pooled,
            "query_proj": query_proj,
            "history_proj": history_proj,
            "query_cache": query_cache,
            "history_cache": history_cache,
            "scores": scores,
            "candidate_mask": candidate_mask,
            "topk_cache": topk_cache,
        }
        return weights, selected, route_starts, cache

    def backward(self, dweights, cache, dscores_extra=None):
        """Backward through selected routing weights and router representations.

        ``dscores_extra`` is an optional full-score gradient intended for
        auxiliary retrieval losses.  Invalid/non-causal candidate entries are
        always masked to zero.
        """
        x = cache["x"]
        batch = x.shape[0]
        route_starts = cache["route_starts"]
        n_routes = int(route_starts.shape[0])
        history_pooled = cache["history_pooled"]
        n_blocks = int(history_pooled.shape[1])

        if n_routes == 0:
            return xp.zeros_like(x)

        dscores = selected_topk_softmax_backward(dweights, cache["topk_cache"])
        if dscores_extra is not None:
            if dscores_extra.shape != dscores.shape:
                raise ValueError("dscores_extra must have the same shape as full scores")
            dscores = dscores + dscores_extra.astype(dscores.dtype, copy=False)
        dscores = xp.where(
            cache["candidate_mask"][None, :, None, :], dscores, 0.0
        )

        query_proj = cache["query_proj"]
        history_proj = cache["history_proj"]
        rq = n_routes * self.num_queries
        scale = 1.0 / math.sqrt(self.router_dim)

        dscores_3d = dscores.reshape(batch, rq, n_blocks)
        query_work = query_proj.reshape(batch, rq, self.router_dim).astype(
            dscores.dtype, copy=False
        )
        history_work = history_proj.astype(dscores.dtype, copy=False)
        dquery_proj_work = (dscores_3d @ history_work) * scale
        dhistory_proj_work = (
            xp.swapaxes(dscores_3d, 1, 2) @ query_work
        ) * scale

        compute_dtype = x.dtype
        dquery_proj = dquery_proj_work.astype(compute_dtype, copy=False).reshape(
            batch, n_routes, self.num_queries * self.router_dim
        )
        dhistory_proj = dhistory_proj_work.astype(compute_dtype, copy=False)

        query_pooled = cache["query_pooled"]
        query_2d = query_pooled.reshape(-1, self.d_model)
        dq_2d = dquery_proj.reshape(-1, self.num_queries * self.router_dim)
        self.W_query.grad += query_2d.T @ dq_2d
        dquery_pooled = (dq_2d @ self.W_query.data.T).reshape(query_pooled.shape)

        history_2d = history_pooled.reshape(-1, self.d_model)
        dh_2d = dhistory_proj.reshape(-1, self.router_dim)
        self.W_history.grad += history_2d.T @ dh_2d
        dhistory_pooled = (dh_2d @ self.W_history.data.T).reshape(
            history_pooled.shape
        )

        dx_query = self.query_pooler.backward(dquery_pooled, cache["query_cache"])
        dx_history = self.history_pooler.backward(
            dhistory_pooled, cache["history_cache"]
        )
        return dx_query + dx_history
