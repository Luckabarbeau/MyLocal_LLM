"""Inference-only external store for 0058C terminal-Landmark memory.

The 0058C training representation always ends in a dense Transformer working
window and right-aligns any older same-document history immediately before it.
History blocks are therefore defined *relative to the current working boundary*.
During autoregressive generation that boundary advances every token, so a
correct streaming implementation must let the logical 128-token block phase
advance as well; pinning blocks to global absolute positions would slowly drift
away from the training semantics.

This store keeps old tokens in a compact ring:

* one router projection per historical token (``d_model -> router_dim``),
* one terminal-memory adapter K/V pair per historical token,
* no deep-Transformer activations for those old tokens.

At routing time the token-projection ring is repartitioned into the same
right-aligned complete blocks used by training.  Only ``top_k * block_size``
exact historical K/V rows are subsequently opened by terminal attention.
The deep Transformer K/V cache remains bounded by the 4k working window.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from mini_llm.backend import xp, is_low_precision_dtype
from mini_llm.ops.rope import rope_forward
from mini_llm.ops.topk import selected_topk_softmax_forward


@dataclass
class TerminalMemoryRoute:
    selected_blocks: object
    gate_scores: object
    route_weights: object
    selected_valid: object
    route_valid: object
    # Logical offset of the first token belonging to block zero.  This is
    # normally zero once the external ring is full, but is non-zero for a
    # shorter right-aligned history whose oldest fragment does not fill a block.
    first_token_offset: int = 0


class TerminalMemoryInferenceStore:
    """Token-ring history with router projections and preprojected memory K/V."""

    def __init__(
        self, config, memory_router, terminal_adapter, embedding_weight, *,
        external_capacity_tokens=None,
    ):
        self.config = config
        self.router = memory_router
        self.adapter = terminal_adapter
        self.embedding_weight = embedding_weight
        self.block_size = int(config.block_size)
        configured_external = int(config.distant_memory_length)
        requested_external = (
            configured_external if external_capacity_tokens is None
            else max(0, min(int(external_capacity_tokens), configured_external))
        )
        self.external_capacity_tokens = int(requested_external)
        self.capacity_blocks = self.external_capacity_tokens // self.block_size
        # Keep the router candidate dimension identical to training even when a
        # caller requests a shorter addressable horizon. Invalid tail blocks are
        # masked, exactly like left padding in document-aware training.
        self.router_blocks = int(config.searchable_blocks)
        self.max_blocks = self.capacity_blocks  # physical blocks inside requested horizon
        self.top_k = int(config.top_k_blocks)
        self.router_dim = int(config.router_dim)
        self.d_model = int(memory_router.d_model)
        self.n_mem_kv_heads = int(config.read_kv_heads)
        self.d_head = int(terminal_adapter.d_head)
        self.dtype = embedding_weight.dtype
        self.projection_chunk = max(
            self.block_size,
            int(os.environ.get("MINI_LLM_TERMINAL_MEMORY_PREFILL_CHUNK", "4096")),
        )

        if str(config.history_pooling) != "mean":
            raise NotImplementedError(
                "terminal-memory inference currently requires mean history pooling"
            )
        if str(config.query_pooling) != "mean":
            raise NotImplementedError(
                "terminal-memory inference currently requires mean query pooling"
            )

        self.batch_size = 0
        self.history_token_proj = None
        # FP32 block summaries avoid cumulative BF16 drift while preserving the
        # mathematically equivalent mean-then-linear router calculation.
        self.history_proj = None
        self.memory_k = None
        self.memory_v = None
        self.token_count = 0
        self.ring_start = 0
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
        self.history_token_proj = xp.empty(
            (batch, cap, self.router_dim), dtype=self.dtype
        )
        self.history_proj = xp.zeros(
            (batch, self.router_blocks, self.router_dim), dtype=xp.float32
        )
        self.memory_k = xp.empty(
            (batch, self.n_mem_kv_heads, cap, self.d_head), dtype=self.dtype
        )
        self.memory_v = xp.empty_like(self.memory_k)
        self._logical_offsets = xp.arange(cap, dtype=xp.int64)
        self.token_count = 0
        self.ring_start = 0

    def _project_tokens(self, embeddings, position_ids):
        """Project raw token embeddings into router and terminal-memory spaces."""
        batch, n_tokens, _ = embeddings.shape
        flat = embeddings.reshape(-1, self.d_model)
        token_proj = (flat @ self.router.W_history.data).reshape(
            batch, n_tokens, self.router_dim
        )

        normed, _ = self.adapter.norm.forward(embeddings)
        normed_flat = normed.reshape(-1, self.d_model)
        mem_k_pre = (normed_flat @ self.adapter.W_k.data).reshape(
            batch, n_tokens, self.n_mem_kv_heads, self.d_head
        )
        mem_v = (normed_flat @ self.adapter.W_v.data).reshape(
            batch, n_tokens, self.n_mem_kv_heads, self.d_head
        )
        mem_k, _ = rope_forward(
            mem_k_pre,
            base=float(self.adapter.rope_base),
            position_ids=position_ids,
        )
        return (
            token_proj,
            mem_k.transpose(0, 2, 1, 3),
            mem_v.transpose(0, 2, 1, 3),
        )

    def initialize(self, external_ids, *, absolute_start=0):
        """Build the old-token ring from the prompt portion before the 4k window."""
        external_ids = xp.asarray(external_ids, dtype=xp.int32)
        if external_ids.ndim != 2:
            raise ValueError("external_ids must have shape [B,T]")
        batch, length = external_ids.shape
        self._allocate(batch)
        cap = int(self.external_capacity_tokens)
        if int(length) == 0 or cap == 0:
            return

        absolute_start = int(absolute_start)
        if int(length) > cap:
            drop = int(length) - cap
            external_ids = external_ids[:, drop:]
            absolute_start += drop
            length = cap
        length = int(length)

        # Chunking avoids a 61k x d_model temporary during 64k prompt prefill.
        for start in range(0, length, self.projection_chunk):
            end = min(length, start + self.projection_chunk)
            ids = external_ids[:, start:end]
            embeddings = self.embedding_weight[ids]
            pos1d = xp.arange(
                absolute_start + start,
                absolute_start + end,
                dtype=xp.int64,
            )
            positions = xp.broadcast_to(pos1d[None, :], (batch, end - start))
            tproj, mk, mv = self._project_tokens(embeddings, positions)
            self.history_token_proj[:, start:end, :] = tproj
            self.memory_k[:, :, start:end, :] = mk
            self.memory_v[:, :, start:end, :] = mv
        self.token_count = length
        self.ring_start = 0

    def _logical_slots(self, offset=0, count=None):
        """Physical ring slots for a contiguous logical old-history span."""
        if count is None:
            count = int(self.token_count) - int(offset)
        count = max(0, int(count))
        if count == 0:
            return self._logical_offsets[:0]
        cap = int(self.external_capacity_tokens)
        logical = self._logical_offsets[int(offset): int(offset) + count]
        return (logical + int(self.ring_start)) % cap

    def _refresh_history_proj(self):
        """Recreate training-equivalent right-aligned block summaries.

        The router's history projection is linear, so averaging per-token
        ``embedding @ W_history`` values is mathematically identical to the
        training ``mean(embedding) @ W_history`` calculation while avoiding a
        d_model-sized external store.
        """
        n_blocks = self.active_blocks
        if n_blocks <= 0:
            return 0
        usable = n_blocks * self.block_size
        first = self.first_block_token_offset
        slots = self._logical_slots(first, usable)
        token_proj = self.history_token_proj[:, slots, :]
        work = token_proj.astype(xp.float32, copy=False).reshape(
            self.batch_size, n_blocks, self.block_size, self.router_dim
        )
        pooled = xp.mean(work, axis=2)
        self.history_proj[:, :n_blocks, :] = pooled
        return n_blocks

    def append_evicted(self, token_ids, absolute_position):
        """Append one token that just aged out of the dense working window."""
        token_ids = xp.asarray(token_ids, dtype=xp.int32).reshape(self.batch_size)
        cap = int(self.external_capacity_tokens)
        if cap == 0:
            return

        embeddings = self.embedding_weight[token_ids][:, None, :]
        positions = xp.full(
            (self.batch_size, 1), int(absolute_position), dtype=xp.int64
        )
        tproj, mk, mv = self._project_tokens(embeddings, positions)

        if int(self.token_count) < cap:
            slot = (int(self.ring_start) + int(self.token_count)) % cap
            self.token_count += 1
        else:
            # Overwrite the oldest token; after advancing ring_start the same
            # physical slot becomes the newest logical token.
            slot = int(self.ring_start)
            self.ring_start = (int(self.ring_start) + 1) % cap
        self.history_token_proj[:, slot, :] = tproj[:, 0, :]
        self.memory_k[:, :, slot, :] = mk[:, :, 0, :]
        self.memory_v[:, :, slot, :] = mv[:, :, 0, :]

    def route(self, working_embedding_sum, working_count):
        """Route the current recent window to training-equivalent old blocks."""
        batch = int(working_embedding_sum.shape[0])
        n_blocks = self._refresh_history_proj()
        min_blocks = int(self.config.min_router_history_blocks)
        if n_blocks < min_blocks:
            return None

        count = max(int(working_count), 1)
        pooled_work = working_embedding_sum.astype(xp.float32, copy=False) / float(count)
        pooled = (
            pooled_work.astype(self.dtype, copy=False)
            if is_low_precision_dtype(self.dtype)
            else pooled_work
        )
        qproj = pooled @ self.router.W_query.data
        qscore = qproj.astype(xp.float32, copy=False)
        scores = xp.sum(
            qscore[:, None, :] * self.history_proj, axis=-1
        ) / math.sqrt(self.router_dim)

        block_ids = xp.arange(self.router_blocks, dtype=xp.int64)
        candidate = xp.broadcast_to(
            (block_ids[None, :] < int(n_blocks)), (batch, self.router_blocks)
        )
        weights, selected, cache = selected_topk_softmax_forward(
            scores,
            self.top_k,
            output_dtype=self.dtype,
            candidate_mask=candidate,
        )
        selected_valid = cache["selected_valid"]
        gate_scores = xp.take_along_axis(scores, selected, axis=-1)
        gate_scores = xp.where(selected_valid, gate_scores, 0.0).astype(
            xp.float32, copy=False
        )
        return TerminalMemoryRoute(
            selected_blocks=xp.ascontiguousarray(selected.astype(xp.int64, copy=False)),
            gate_scores=xp.ascontiguousarray(gate_scores),
            route_weights=xp.ascontiguousarray(weights),
            selected_valid=xp.ascontiguousarray(selected_valid),
            route_valid=xp.ones((batch,), dtype=bool),
            first_token_offset=int(self.first_block_token_offset),
        )
