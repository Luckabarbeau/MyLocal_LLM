"""Inference-only external token store for 0060B routed-prefix memory.

The training path never sends the full long history through the Transformer.
Instead it mean-pools 128-token historical blocks, scores them with one query
from the current dense window, reopens exactly ``top_k`` real blocks, sorts
those blocks chronologically, and prepends them to the 4k working window.

This module keeps the long history in the smallest representation needed to
reproduce that selector during autoregressive generation:

* raw token IDs in a bounded ring, so selected blocks can be reopened exactly;
* one ``d_model -> router_dim`` projection per old token, so block means can be
  reconstructed without storing d_model-sized embeddings;
* absolute position of the oldest token, so selected rows retain their real
  RoPE distance from the current working window.

No deep-Transformer K/V or activations are stored for the long history.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from mini_llm.backend import xp, is_low_precision_dtype


@dataclass
class RoutedPrefixInferenceRoute:
    """One deterministic top-level memory decision used to rebuild deep cache."""

    selected_blocks: object
    selected_token_ids: object
    selected_position_ids: object
    route_valid: object
    first_token_offset: int = 0


class RoutedPrefixInferenceStore:
    """Bounded old-token ring plus cheap router projections."""

    def __init__(
        self, config, memory_router, embedding_weight, *, external_capacity_tokens=None,
    ):
        self.config = config
        self.router = memory_router
        self.embedding_weight = embedding_weight
        self.block_size = int(config.block_size)
        configured_external = int(config.distant_memory_length)
        requested_external = (
            configured_external
            if external_capacity_tokens is None
            else max(0, min(int(external_capacity_tokens), configured_external))
        )
        self.external_capacity_tokens = int(requested_external)
        self.capacity_blocks = self.external_capacity_tokens // self.block_size
        self.top_k = int(config.top_k_blocks)
        self.router_dim = int(config.router_dim)
        self.d_model = int(memory_router.d_model)
        self.dtype = embedding_weight.dtype
        self.projection_chunk = max(
            self.block_size,
            int(os.environ.get("MINI_LLM_ROUTED_PREFIX_PREFILL_CHUNK", "4096")),
        )

        if str(config.history_pooling) != "mean":
            raise NotImplementedError(
                "routed-prefix inference currently requires mean history pooling"
            )
        if str(config.query_pooling) != "mean":
            raise NotImplementedError(
                "routed-prefix inference currently requires mean query pooling"
            )

        self.batch_size = 0
        self.token_ids = None
        self.history_token_proj = None
        self.token_count = 0
        self.ring_start = 0
        self.oldest_abs_pos = 0
        self._logical_offsets = None

    @property
    def active_blocks(self):
        return min(self.capacity_blocks, int(self.token_count) // self.block_size)

    @property
    def first_block_token_offset(self):
        n_blocks = self.active_blocks
        return int(self.token_count) - n_blocks * self.block_size

    def _allocate(self, batch):
        self.batch_size = int(batch)
        cap = int(self.external_capacity_tokens)
        self.token_ids = xp.empty((batch, cap), dtype=xp.int32)
        self.history_token_proj = xp.empty(
            (batch, cap, self.router_dim), dtype=self.dtype
        )
        self._logical_offsets = xp.arange(cap, dtype=xp.int64)
        self.token_count = 0
        self.ring_start = 0
        self.oldest_abs_pos = 0

    def _logical_slots(self, offset=0, count=None):
        if count is None:
            count = int(self.token_count) - int(offset)
        count = max(0, int(count))
        if count == 0:
            return self._logical_offsets[:0]
        cap = int(self.external_capacity_tokens)
        logical = self._logical_offsets[int(offset): int(offset) + count]
        return (logical + int(self.ring_start)) % cap

    def _project_embeddings(self, embeddings):
        batch, n_tokens, _ = embeddings.shape
        return (embeddings.reshape(-1, self.d_model) @ self.router.W_history.data).reshape(
            batch, n_tokens, self.router_dim
        )

    def initialize(self, external_ids, *, absolute_start=0):
        """Initialize from the prompt portion preceding the current working window."""
        external_ids = xp.asarray(external_ids, dtype=xp.int32)
        if external_ids.ndim != 2:
            raise ValueError("external_ids must have shape [B,T]")
        batch, length = external_ids.shape
        self._allocate(batch)
        cap = int(self.external_capacity_tokens)
        absolute_start = int(absolute_start)
        if int(length) == 0 or cap == 0:
            self.oldest_abs_pos = absolute_start + int(length)
            return

        if int(length) > cap:
            drop = int(length) - cap
            external_ids = external_ids[:, drop:]
            absolute_start += drop
            length = cap
        length = int(length)
        self.oldest_abs_pos = absolute_start

        # Project in chunks so a 60k x d_model embedding temporary never exists.
        for start in range(0, length, self.projection_chunk):
            end = min(length, start + self.projection_chunk)
            ids = external_ids[:, start:end]
            embeddings = self.embedding_weight[ids]
            projected = self._project_embeddings(embeddings)
            self.token_ids[:, start:end] = ids
            self.history_token_proj[:, start:end, :] = projected
        self.token_count = length
        self.ring_start = 0

    def append_evicted(self, token_ids, absolute_position):
        """Append one token that just aged out of the 4k working window."""
        token_ids = xp.asarray(token_ids, dtype=xp.int32).reshape(self.batch_size)
        cap = int(self.external_capacity_tokens)
        if cap == 0:
            return

        embeddings = self.embedding_weight[token_ids][:, None, :]
        projected = self._project_embeddings(embeddings)[:, 0, :]
        absolute_position = int(absolute_position)

        if int(self.token_count) < cap:
            if int(self.token_count) == 0:
                self.oldest_abs_pos = absolute_position
            slot = (int(self.ring_start) + int(self.token_count)) % cap
            self.token_count += 1
        else:
            slot = int(self.ring_start)
            self.ring_start = (int(self.ring_start) + 1) % cap
            self.oldest_abs_pos += 1

        self.token_ids[:, slot] = token_ids
        self.history_token_proj[:, slot, :] = projected

    def _block_history_projection(self):
        """Return right-aligned complete block projections [B,N,Dr]."""
        n_blocks = self.active_blocks
        if n_blocks <= 0:
            return self.history_token_proj[:, :0, :].reshape(
                self.batch_size, 0, self.router_dim
            )
        usable = n_blocks * self.block_size
        first = self.first_block_token_offset
        slots = self._logical_slots(first, usable)
        token_proj = self.history_token_proj[:, slots, :]
        work = token_proj.astype(xp.float32, copy=False).reshape(
            self.batch_size, n_blocks, self.block_size, self.router_dim
        )
        return xp.mean(work, axis=2)


    def direct_prefix(self, max_tokens=None):
        """Return exact recent external history without invoking the router.

        This is the correct policy while all available history fits inside the
        retrieved-token budget: there is no selection problem to learn or solve.
        If ``max_tokens`` is smaller than the stored history, the newest exact
        tokens are kept so the method also covers the narrow pre-routing
        transition where block alignment has not yet produced K candidates.
        """
        if int(self.token_count) <= 0:
            return None
        limit = int(self.config.retrieved_length) if max_tokens is None else int(max_tokens)
        count = min(int(self.token_count), max(0, limit))
        if count <= 0:
            return None
        offset = int(self.token_count) - count
        slots = self._logical_slots(offset, count)
        selected_ids = self.token_ids[:, slots]
        # _logical_slots is shared across the batch, so absolute positions are too.
        selected_abs_1d = (
            xp.arange(offset, offset + count, dtype=xp.int64)
            + int(self.oldest_abs_pos)
        )
        selected_abs = xp.broadcast_to(
            selected_abs_1d[None, :], (self.batch_size, count)
        )
        return RoutedPrefixInferenceRoute(
            selected_blocks=xp.empty((self.batch_size, 0), dtype=xp.int64),
            selected_token_ids=xp.ascontiguousarray(
                selected_ids.astype(xp.int32, copy=False)
            ),
            selected_position_ids=xp.ascontiguousarray(
                selected_abs.astype(xp.int64, copy=False)
            ),
            route_valid=xp.ones((self.batch_size,), dtype=bool),
            first_token_offset=offset,
        )

    def route(self, working_embedding_sum, working_count):
        """Deterministically select K complete historical blocks.

        The query is exactly the mean raw embedding of the current working
        window followed by ``W_query``. History uses mean token projection,
        equivalent to ``mean(embedding) @ W_history`` because the projection is
        linear. Selected blocks are sorted by historical order before their real
        token IDs and absolute positions are returned.
        """
        n_blocks = self.active_blocks
        if n_blocks < self.top_k:
            return None

        count = max(int(working_count), 1)
        pooled_work = working_embedding_sum.astype(xp.float32, copy=False) / float(count)
        pooled_query = (
            pooled_work.astype(self.dtype, copy=False)
            if is_low_precision_dtype(self.dtype)
            else pooled_work
        )
        qproj = pooled_query @ self.router.W_query.data
        history_proj = self._block_history_projection()

        qscore = qproj.astype(xp.float32, copy=False)
        hscore = history_proj.astype(xp.float32, copy=False)
        scores = xp.sum(qscore[:, None, :] * hscore, axis=-1) / math.sqrt(
            self.router_dim
        )

        selected = xp.argsort(-scores, axis=-1)[:, : self.top_k]
        selected = xp.sort(selected, axis=-1)

        offsets = xp.arange(self.block_size, dtype=xp.int64)
        first = int(self.first_block_token_offset)
        logical_positions = (
            first
            + selected[..., None] * self.block_size
            + offsets[None, None, :]
        ).reshape(self.batch_size, -1)
        cap = int(self.external_capacity_tokens)
        slots = (logical_positions + int(self.ring_start)) % cap
        batch_ids = xp.arange(self.batch_size, dtype=xp.int64)[:, None]
        selected_ids = self.token_ids[batch_ids, slots]
        selected_abs = logical_positions + int(self.oldest_abs_pos)

        return RoutedPrefixInferenceRoute(
            selected_blocks=xp.ascontiguousarray(selected.astype(xp.int64, copy=False)),
            selected_token_ids=xp.ascontiguousarray(selected_ids.astype(xp.int32, copy=False)),
            selected_position_ids=xp.ascontiguousarray(selected_abs.astype(xp.int64, copy=False)),
            route_valid=xp.ones((self.batch_size,), dtype=bool),
            first_token_offset=first,
        )
