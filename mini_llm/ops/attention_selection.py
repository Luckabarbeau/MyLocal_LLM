"""Explicit key-visibility plans for modular sparse attention.

A :class:`KeySelectionPlan` answers only one question for an attention kernel:

    Which exact key/value token positions are visible to each query head/token?

It deliberately contains no Q/K/V projection logic and no attention math.  This
keeps sparse topology independent from the numerical attention kernel and makes
manual, deterministic, and learned retrieval interchangeable.
"""

from dataclasses import dataclass
import math

from ..backend import xp, is_low_precision_dtype
from .context_blocks import block_token_indices


@dataclass
class KeySelectionPlan:
    """Exact key indices and validity mask for indexed attention.

    ``key_indices`` and ``valid_mask`` use shape ``(B, Hplan, Tq, K)``.  The
    plan may contain either one shared head (``Hplan == 1``) or one plan per
    query head.  ``logit_bias`` is optional and follows the same broadcast
    convention; it is added to the selected attention logits before softmax.

    Invalid entries must still contain an in-range placeholder key index.  They
    receive exactly zero probability and zero gradient.
    """

    key_indices: object
    valid_mask: object
    logit_bias: object = None

    def validate(self, batch_size, n_q_heads, query_length, key_length):
        if self.key_indices.ndim != 4:
            raise ValueError("key_indices must have shape (B,H,Tq,K)")
        if self.valid_mask.shape != self.key_indices.shape:
            raise ValueError("valid_mask must have the same shape as key_indices")
        if int(self.key_indices.shape[0]) != int(batch_size):
            raise ValueError("key selection batch dimension mismatch")
        if int(self.key_indices.shape[2]) != int(query_length):
            raise ValueError("key selection query-length mismatch")
        if int(self.key_indices.shape[1]) not in {1, int(n_q_heads)}:
            raise ValueError("plan head dimension must be 1 or n_q_heads")
        if int(self.key_indices.shape[3]) <= 0:
            raise ValueError("indexed attention requires at least one key slot")
        if self.logit_bias is not None:
            if self.logit_bias.shape != self.key_indices.shape:
                raise ValueError("logit_bias must have the same shape as key_indices")

        if self.key_indices.size:
            if bool(xp.any(self.key_indices < 0)):
                raise IndexError("key_indices must be non-negative")
            if bool(xp.any(self.key_indices >= int(key_length))):
                raise IndexError("key index extends outside the key/value sequence")


def build_block_retrieval_plan(
    selected_blocks,
    route_starts,
    seq_len,
    block_size,
    routing_stride,
    exclude_recent_tokens=0,
):
    """Expand manually selected history blocks into an exact token-key plan.

    Args:
        selected_blocks: ``(B,R,Kb)`` or ``(B,R,H,Kb)`` integer block IDs.
            The optional ``H`` dimension lets different retrieval heads use
            different block selections.
        route_starts: ``(R,)`` causal routing boundaries.  Selection ``r`` is
            reused for query tokens ``[route_starts[r], start+routing_stride)``.
        seq_len: Query/key sequence length.
        block_size: Number of full-resolution tokens reopened per history block.
        routing_stride: Number of query tokens controlled by one route.
        exclude_recent_tokens: Optional minimum gap used as an additional
            correctness assertion for manual selections.

    Returns:
        :class:`KeySelectionPlan` with full-resolution token indices.

    The function validates causality independently from the learned router.  A
    selected block must end no later than ``route_start-exclude_recent_tokens``.
    This makes manual retrieval a useful numerical reference path.
    """
    if selected_blocks.ndim == 3:
        selected = selected_blocks[:, :, None, :]
    elif selected_blocks.ndim == 4:
        selected = selected_blocks
    else:
        raise ValueError("selected_blocks must have shape (B,R,K) or (B,R,H,K)")

    if route_starts.ndim != 1:
        raise ValueError("route_starts must be one-dimensional")
    batch, n_routes, n_heads, n_blocks = selected.shape
    if int(route_starts.shape[0]) != int(n_routes):
        raise ValueError("selected_blocks route dimension must match route_starts")

    seq_len = int(seq_len)
    block_size = int(block_size)
    routing_stride = int(routing_stride)
    exclude_recent_tokens = int(exclude_recent_tokens)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size <= 0 or routing_stride <= 0:
        raise ValueError("block_size and routing_stride must be positive")
    if exclude_recent_tokens < 0:
        raise ValueError("exclude_recent_tokens must be non-negative")
    if n_blocks <= 0:
        raise ValueError("selected_blocks must contain at least one block slot")

    if n_routes:
        if bool(xp.any(route_starts < 0)) or bool(xp.any(route_starts >= seq_len)):
            raise ValueError("route start lies outside the sequence")
        if n_routes > 1 and bool(xp.any(route_starts[1:] <= route_starts[:-1])):
            raise ValueError("route_starts must be strictly increasing")
        if bool(xp.any(selected < 0)):
            raise ValueError("selected block IDs must be non-negative")

        allowed_end = route_starts[None, :, None, None] - exclude_recent_tokens
        selected_end = (selected + 1) * block_size
        if bool(xp.any(selected_end > allowed_end)):
            raise ValueError(
                "manual block selection violates the causal/recent-history boundary"
            )

    key_slots = n_blocks * block_size
    if seq_len == 0:
        empty_indices = xp.zeros(
            (batch, n_heads, 0, key_slots), dtype=xp.int64
        )
        return KeySelectionPlan(empty_indices, xp.zeros_like(empty_indices, dtype=bool))

    token_positions = xp.arange(seq_len, dtype=xp.int64)

    if n_routes == 0:
        indices = xp.zeros(
            (batch, n_heads, seq_len, key_slots), dtype=xp.int64
        )
        valid = xp.zeros(indices.shape, dtype=bool)
        return KeySelectionPlan(indices, valid)

    # Map each query token to the latest route boundary at or before it.  Gaps
    # larger than routing_stride intentionally remain invalid rather than
    # silently reusing a stale retrieval decision.
    route_for_token = xp.searchsorted(route_starts, token_positions, side="right") - 1
    safe_route = xp.maximum(route_for_token, 0)
    controlling_start = route_starts[safe_route]
    token_has_route = (route_for_token >= 0) & (
        token_positions < controlling_start + routing_stride
    )

    # selected[:, safe_route] -> [B,T,H,Kb], then expose head before token.
    token_blocks = selected[:, safe_route, :, :].transpose(0, 2, 1, 3)
    token_keys = block_token_indices(token_blocks, block_size).reshape(
        batch, n_heads, seq_len, key_slots
    )
    valid = xp.broadcast_to(
        token_has_route[None, None, :, None], token_keys.shape
    ).copy()

    # Invalid rows still need an in-range placeholder for the gather.  The
    # masked softmax guarantees these positions have zero probability/gradient.
    token_keys = xp.where(valid, token_keys, 0).astype(xp.int64, copy=False)
    return KeySelectionPlan(token_keys, valid)


def build_weighted_block_retrieval_plan(
    selected_blocks,
    selected_weights,
    route_starts,
    seq_len,
    block_size,
    routing_stride,
    exclude_recent_tokens=0,
    weight_scale=1.0,
    weight_eps=1e-8,
):
    """Build exact retrieval keys plus a differentiable per-block logit prior.

    ``selected_blocks`` and ``selected_weights`` use shape ``(B,R,Kb)`` or
    ``(B,R,H,Kb)``.  Every selected block is reopened at full token resolution.
    Its normalized router weight ``p`` contributes the same additive attention
    bias to each token in that block::

        bias = weight_scale * log(p + weight_eps)

    The returned cache is consumed by
    :func:`weighted_block_retrieval_bias_backward` to reduce the expanded
    attention-bias gradient back to one gradient per selected router weight.
    """
    if selected_blocks.shape != selected_weights.shape:
        raise ValueError("selected_blocks and selected_weights must have the same shape")
    if selected_blocks.ndim == 3:
        weights = selected_weights[:, :, None, :]
        router_had_head_axis = False
    elif selected_blocks.ndim == 4:
        weights = selected_weights
        router_had_head_axis = True
    else:
        raise ValueError("selected tensors must have shape (B,R,K) or (B,R,H,K)")

    weight_scale = float(weight_scale)
    weight_eps = float(weight_eps)
    if not math.isfinite(weight_scale):
        raise ValueError("weight_scale must be finite")
    if not math.isfinite(weight_eps) or weight_eps <= 0:
        raise ValueError("weight_eps must be positive and finite")
    if bool(xp.any(~xp.isfinite(weights))):
        raise ValueError("selected_weights must be finite")
    if bool(xp.any(weights < 0)):
        raise ValueError("selected_weights must be non-negative")

    plan = build_block_retrieval_plan(
        selected_blocks,
        route_starts,
        seq_len=seq_len,
        block_size=block_size,
        routing_stride=routing_stride,
        exclude_recent_tokens=exclude_recent_tokens,
    )

    batch, n_routes, n_heads, n_selected = weights.shape
    seq_len = int(seq_len)
    block_size = int(block_size)
    routing_stride = int(routing_stride)

    token_positions = xp.arange(seq_len, dtype=xp.int64)
    if n_routes == 0:
        logit_bias = xp.zeros(plan.key_indices.shape, dtype=selected_weights.dtype)
        plan.logit_bias = logit_bias
        cache = {
            "weights": weights,
            "safe_route": xp.zeros((seq_len,), dtype=xp.int64),
            "token_has_route": xp.zeros((seq_len,), dtype=bool),
            "block_size": block_size,
            "weight_scale": weight_scale,
            "weight_eps": weight_eps,
            "router_had_head_axis": router_had_head_axis,
        }
        return plan, cache

    route_for_token = xp.searchsorted(route_starts, token_positions, side="right") - 1
    safe_route = xp.maximum(route_for_token, 0)
    controlling_start = route_starts[safe_route]
    token_has_route = (route_for_token >= 0) & (
        token_positions < controlling_start + routing_stride
    )

    # [B,T,H,Kb] -> [B,H,T,Kb].  One block weight is repeated over all exact
    # token keys reopened from that block.
    token_weights = weights[:, safe_route, :, :].transpose(0, 2, 1, 3)
    token_weights_work = (
        token_weights.astype("float32", copy=False)
        if is_low_precision_dtype(token_weights.dtype)
        else token_weights
    )
    token_bias = weight_scale * xp.log(token_weights_work + weight_eps)
    logit_bias = xp.repeat(token_bias, block_size, axis=-1)
    logit_bias = xp.where(plan.valid_mask, logit_bias, 0.0)
    plan.logit_bias = logit_bias.astype(selected_weights.dtype, copy=False)

    cache = {
        "weights": weights,
        "safe_route": safe_route,
        "token_has_route": token_has_route,
        "block_size": block_size,
        "weight_scale": weight_scale,
        "weight_eps": weight_eps,
        "router_had_head_axis": router_had_head_axis,
    }
    return plan, cache


def weighted_block_retrieval_bias_backward(dlogit_bias, cache):
    """Reduce exact-token attention-bias gradients to selected router weights.

    ``indexed_attention_backward`` returns bias gradients after any shared plan
    head has been broadcast across query heads.  When the context router used a
    single query (head dimension one), those query-head contributions are summed
    before reducing tokens back to routes/blocks.
    """
    weights = cache["weights"]
    batch, n_routes, n_router_heads, n_selected = weights.shape
    block_size = int(cache["block_size"])

    if dlogit_bias.ndim != 4 or int(dlogit_bias.shape[0]) != batch:
        raise ValueError("dlogit_bias must have shape (B,H,T,K)")
    if int(dlogit_bias.shape[-1]) != n_selected * block_size:
        raise ValueError("dlogit_bias key dimension does not match selected blocks")

    if n_routes == 0:
        result = xp.zeros(weights.shape, dtype=dlogit_bias.dtype)
        if cache["router_had_head_axis"]:
            return result
        return result[:, :, 0, :]

    if int(dlogit_bias.shape[1]) == n_router_heads:
        dbias = dlogit_bias
    elif n_router_heads == 1:
        dbias = xp.sum(dlogit_bias, axis=1, keepdims=True)
    else:
        raise ValueError(
            "attention bias head dimension must equal router heads, unless the "
            "router has one shared query"
        )

    # Sum the repeated full-resolution token bias within each selected block.
    dbias_blocks = dbias.reshape(
        batch,
        n_router_heads,
        dlogit_bias.shape[2],
        n_selected,
        block_size,
    ).sum(axis=-1)

    safe_route = cache["safe_route"]
    token_has_route = cache["token_has_route"]
    if int(dbias_blocks.shape[2]) != int(token_has_route.shape[0]):
        raise ValueError("dlogit_bias query length does not match forward cache")

    token_weights = weights[:, safe_route, :, :].transpose(0, 2, 1, 3)
    token_weights = token_weights.astype(dlogit_bias.dtype, copy=False)
    derivative = cache["weight_scale"] / (token_weights + cache["weight_eps"])
    contributions = dbias_blocks * derivative
    contributions = xp.where(
        token_has_route[None, None, :, None], contributions, 0.0
    )

    scatter_dtype = (
        "float32"
        if is_low_precision_dtype(contributions.dtype)
        else contributions.dtype
    )
    dweights = xp.zeros(weights.shape, dtype=scatter_dtype)
    contributions = contributions.astype(scatter_dtype, copy=False)
    if n_routes:
        b_ids = xp.broadcast_to(
            xp.arange(batch, dtype=xp.int64)[:, None, None, None],
            contributions.shape,
        )
        h_ids = xp.broadcast_to(
            xp.arange(n_router_heads, dtype=xp.int64)[None, :, None, None],
            contributions.shape,
        )
        r_ids = xp.broadcast_to(
            safe_route[None, None, :, None], contributions.shape
        )
        k_ids = xp.broadcast_to(
            xp.arange(n_selected, dtype=xp.int64)[None, None, None, :],
            contributions.shape,
        )
        xp.add.at(dweights, (b_ids, r_ids, h_ids, k_ids), contributions)

    if cache["router_had_head_axis"]:
        return dweights
    return dweights[:, :, 0, :]


def build_local_causal_plan(batch_size, seq_len, window):
    """Build a shared exact causal sliding-window key plan.

    The plan has one topology head and is broadcast across whichever query-head
    subset consumes it. Query token ``t`` may see
    ``max(0, t-window+1) ... t``. The effective key-slot count is clipped to
    ``seq_len`` so asking for a window larger than the sequence does not create
    unnecessary padding.
    """
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    window = int(window)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    key_slots = min(window, seq_len)
    queries = xp.arange(seq_len, dtype=xp.int64)[:, None]
    offsets = xp.arange(key_slots, dtype=xp.int64)[None, :]
    raw = queries - (key_slots - 1) + offsets
    valid = raw >= 0
    indices = xp.maximum(raw, 0)

    indices = xp.broadcast_to(
        indices[None, None, :, :], (batch_size, 1, seq_len, key_slots)
    ).copy()
    valid = xp.broadcast_to(valid[None, None, :, :], indices.shape).copy()
    return KeySelectionPlan(indices, valid)


def build_dilated_causal_plan(batch_size, seq_len, window, dilation, offset=0):
    """Build a shared causal fixed-phase dilated key plan.

    Query token ``t`` sees exact key positions

        ``t - offset - n*dilation``

    that fall inside the trailing ``window``-token span.  Keys are returned in
    chronological order.  ``offset=0`` includes the current token, while
    offsets ``1 ... dilation-1`` expose complementary phases without changing
    the attention kernel.

    The number of key slots is approximately ``window / dilation`` rather than
    ``window``.  Early query positions may have no valid key for non-zero
    offsets; indexed attention defines those rows to have zero context.
    """
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    window = int(window)
    dilation = int(dilation)
    offset = int(offset)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if offset < 0 or offset >= dilation:
        raise ValueError("offset must satisfy 0 <= offset < dilation")
    if offset >= window:
        raise ValueError("offset must be smaller than window")

    # Largest admissible lag is window-1.  The selected lags are
    # offset, offset+dilation, ... <= window-1.
    key_slots = ((window - 1 - offset) // dilation) + 1
    queries = xp.arange(seq_len, dtype=xp.int64)[:, None]
    steps = xp.arange(key_slots - 1, -1, -1, dtype=xp.int64)[None, :]
    lags = offset + steps * dilation
    raw = queries - lags
    valid = raw >= 0
    indices = xp.maximum(raw, 0)

    indices = xp.broadcast_to(
        indices[None, None, :, :], (batch_size, 1, seq_len, key_slots)
    ).copy()
    valid = xp.broadcast_to(valid[None, None, :, :], indices.shape).copy()
    return KeySelectionPlan(indices, valid)


def build_global_sparse_causal_plan(
    batch_size, seq_len, stride, offset=0, include_current=True
):
    """Build a shared causal whole-prefix fixed-anchor key plan.

    Global anchors are absolute token positions

        ``offset, offset + stride, offset + 2*stride, ...``

    and query token ``t`` may use only anchors ``<= t``.  When
    ``include_current`` is true, ``t`` is appended as an exact key whenever it
    is not already one of the global anchors.  This gives every query a local
    self path while preserving ``O(T / stride)`` whole-history connectivity.

    Different heads can use complementary ``offset`` values without changing
    the indexed-attention kernel.  ``stride=1, offset=0`` is exactly ordinary
    full causal visibility.
    """
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    stride = int(stride)
    offset = int(offset)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if offset < 0 or offset >= stride:
        raise ValueError("offset must satisfy 0 <= offset < stride")
    if not isinstance(include_current, (bool, xp.bool_)):
        raise TypeError("include_current must be a bool")

    anchors = xp.arange(offset, seq_len, stride, dtype=xp.int64)
    n_anchors = int(anchors.shape[0])
    extra_current = bool(include_current and stride != 1)
    key_slots = n_anchors + int(extra_current)

    # KeySelectionPlan intentionally requires at least one physical slot even
    # when a head has no visible anchor yet.  Keep an invalid in-range
    # placeholder for that rare configuration.
    if key_slots == 0:
        indices = xp.zeros((batch_size, 1, seq_len, 1), dtype=xp.int64)
        valid = xp.zeros(indices.shape, dtype=bool)
        return KeySelectionPlan(indices, valid)

    queries = xp.arange(seq_len, dtype=xp.int64)[:, None]
    if n_anchors:
        anchor_indices = xp.broadcast_to(anchors[None, :], (seq_len, n_anchors))
        anchor_valid = anchor_indices <= queries
    else:
        anchor_indices = xp.empty((seq_len, 0), dtype=xp.int64)
        anchor_valid = xp.empty((seq_len, 0), dtype=bool)

    if extra_current:
        current = xp.arange(seq_len, dtype=xp.int64)[:, None]
        # Avoid counting the same token twice when the current position itself
        # lies on this global anchor phase.
        on_phase = (current >= offset) & (((current - offset) % stride) == 0)
        current_valid = ~on_phase
        indices_2d = xp.concatenate((anchor_indices, current), axis=1)
        valid_2d = xp.concatenate((anchor_valid, current_valid), axis=1)
    else:
        indices_2d = anchor_indices
        valid_2d = anchor_valid

    indices = xp.broadcast_to(
        indices_2d[None, None, :, :],
        (batch_size, 1, seq_len, key_slots),
    ).copy()
    valid = xp.broadcast_to(valid_2d[None, None, :, :], indices.shape).copy()
    return KeySelectionPlan(indices, valid)
