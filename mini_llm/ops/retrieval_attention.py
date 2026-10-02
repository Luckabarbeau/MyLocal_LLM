"""Learned block retrieval connected to full-resolution indexed attention.

This module intentionally owns no Q/K/V projection.  It composes two already
independently testable pieces:

    recent/history hidden states -> ContextRouter -> selected history blocks
    selected history blocks      -> indexed_attention over exact K/V tokens

The selected router probabilities may be injected as an additive log-probability
prior on the exact retrieved-token attention logits.  Top-K identities remain
fixed during the local backward pass, matching the explicit MoE-router
convention used elsewhere in the repository.
"""

from ..backend import xp
from .attention_selection import (
    build_block_retrieval_plan,
    build_weighted_block_retrieval_plan,
    weighted_block_retrieval_bias_backward,
)
from .context_router import ContextRouter
from .indexed_attention import indexed_attention_forward, indexed_attention_backward


class ContextRetrievalAttention:
    """Learned history-block retrieval followed by exact indexed attention.

    The module consumes already-projected/position-encoded ``q``, ``k``, and
    ``v`` tensors.  ``router_input`` is the layer representation used to form
    recent-context queries and historical block summaries.  Keeping these two
    roles separate lets the future Transformer attention module share packed
    Q/K/V projections across local, dilated, global, and retrieval heads.
    """

    def __init__(
        self,
        d_model,
        config,
        rng,
        input_std=0.02,
        name="context_retrieval",
        dtype="float32",
    ):
        self.d_model = int(d_model)
        self.config = config
        self.router = ContextRouter(
            d_model=self.d_model,
            config=config,
            rng=rng,
            input_std=input_std,
            name=f"{name}.router",
            dtype=dtype,
        )

    def parameters(self):
        return self.router.parameters()

    def zero_grad(self):
        self.router.zero_grad()

    def forward(
        self,
        router_input,
        q,
        k,
        v,
        kv_head_indices=None,
        return_cache=True,
    ):
        """Route distant blocks and attend to their exact full-resolution K/V.

        Training/full-sequence mode currently requires ``router_input`` and
        ``q`` to span the same sequence.  Incremental cache-based inference is
        deliberately deferred to a later patch.
        """
        if router_input.ndim != 3 or router_input.shape[-1] != self.d_model:
            raise ValueError(
                "router_input must have shape (batch, seq_len, d_model)"
            )
        if q.ndim != 4:
            raise ValueError("q must have shape (batch, seq_len, n_q_heads, d_head)")
        if int(router_input.shape[0]) != int(q.shape[0]) or int(
            router_input.shape[1]
        ) != int(q.shape[1]):
            raise ValueError("router_input and q must share batch and sequence dimensions")

        n_q_heads = int(q.shape[2])
        if self.router.num_queries not in {1, n_q_heads}:
            raise ValueError(
                "context router num_queries must be 1 or equal the number of "
                "retrieval query heads"
            )

        weights, selected, route_starts, router_cache = self.router.forward(
            router_input
        )

        if self.config.router_weight_mode == "logit_bias":
            plan, plan_cache = build_weighted_block_retrieval_plan(
                selected,
                weights,
                route_starts,
                seq_len=q.shape[1],
                block_size=self.config.history_block_size,
                routing_stride=self.config.routing_stride,
                exclude_recent_tokens=self.config.exclude_recent_tokens,
                weight_scale=self.config.router_weight_scale,
                weight_eps=self.config.router_weight_eps,
            )
        elif self.config.router_weight_mode == "none":
            plan = build_block_retrieval_plan(
                selected,
                route_starts,
                seq_len=q.shape[1],
                block_size=self.config.history_block_size,
                routing_stride=self.config.routing_stride,
                exclude_recent_tokens=self.config.exclude_recent_tokens,
            )
            plan_cache = None
        else:  # Config validation should make this unreachable.
            raise ValueError("unsupported router_weight_mode")

        if return_cache:
            context, attention_cache = indexed_attention_forward(
                q,
                k,
                v,
                plan,
                kv_head_indices=kv_head_indices,
                return_cache=True,
            )
        else:
            context = indexed_attention_forward(
                q,
                k,
                v,
                plan,
                kv_head_indices=kv_head_indices,
                return_cache=False,
            )
            attention_cache = None

        routing = {
            "weights": weights,
            "selected_blocks": selected,
            "route_starts": route_starts,
        }
        if not return_cache:
            return context, routing

        cache = {
            "router_cache": router_cache,
            "attention_cache": attention_cache,
            "plan_cache": plan_cache,
            "weights_shape": weights.shape,
        }
        return context, routing, cache

    def backward(self, dcontext, cache, dscores_extra=None):
        """Explicit backward from exact attention into the context router.

        Returns four independent gradients:

        ``drouter_input, dq, dk, dv``.

        A future Transformer integration patch will add ``drouter_input`` to
        the Q/K/V projection path because both originate from the same layer
        hidden state there.
        """
        dq, dk, dv, dlogit_bias = indexed_attention_backward(
            dcontext, cache["attention_cache"]
        )

        if self.config.router_weight_mode == "logit_bias":
            if dlogit_bias is None:
                raise RuntimeError("weighted retrieval expected a logit-bias gradient")
            dweights = weighted_block_retrieval_bias_backward(
                dlogit_bias, cache["plan_cache"]
            )
        else:
            # With discrete selection only, the LM objective has no local
            # differentiable path through Top-K identities.  Auxiliary score
            # losses can still train the router through dscores_extra.
            dweights = xp.zeros(cache["weights_shape"], dtype=dcontext.dtype)

        drouter_input = self.router.backward(
            dweights,
            cache["router_cache"],
            dscores_extra=dscores_extra,
        )
        return drouter_input, dq, dk, dv
