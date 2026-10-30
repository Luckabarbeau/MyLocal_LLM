"""Inference model with KV cache for autoregressive generation.

This module provides a KV-cached inference implementation that:
- Preallocates K/V caches to avoid reallocation during generation
- Supports prefill (batched prompt processing) and decode (single-token)
- Uses absolute RoPE positions for correct positional encoding
- Implements grouped-query attention with proper KV sharing

Usage:
    model = InferenceModel(config)
    model.set_weights(training_model)  # Share weights
    
    state = model.create_generation_state(batch_size=1, max_length=512)
    
    # Prefill with prompt
    logits = model.prefill(input_ids, state)
    
    # Decode tokens one at a time
    for _ in range(max_new_tokens):
        next_id = sample(logits)
        logits = model.decode_one(next_id, state)
"""

import os
from dataclasses import replace

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.config import ModelConfig
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.rmsnorm_inference import RMSNormInference
from mini_llm.blocks.transformer_block_inference import TransformerBlockInference
from mini_llm.inference_state import GenerationState
from mini_llm.ops.terminal_memory_inference import TerminalMemoryInferenceStore
from mini_llm.ops.routed_prefix_inference import RoutedPrefixInferenceStore


class InferenceModel:
    """
    Inference-only model with KV cache.
    
    Shares weights with a training DecoderLanguageModel but uses
    preallocated K/V caches for fast autoregressive generation.
    """

    def __init__(self, config: ModelConfig, dtype: str = None):
        """
        Initialize inference model.
        
        Args:
            config: ModelConfig with all hyperparameters
            dtype: Data type (overrides config if provided)
        """
        self.config = config
        self.dtype = dtype if dtype is not None else config.dtype
        self.dual_resolution_config = getattr(config, "dual_resolution", None)
        self.dual_resolution_enabled = bool(
            self.dual_resolution_config is not None
            and getattr(self.dual_resolution_config, "enabled", False)
        )
        self.sparse_attention = (
            config.attention_layers is not None and not self.dual_resolution_enabled
        )
        memory_cfg = getattr(config, "memory_context", None)
        self.historical_read_memory = bool(
            memory_cfg is not None
            and memory_cfg.enabled
            and memory_cfg.integration_mode == "historical_read"
        )
        self.terminal_landmark_memory = bool(
            memory_cfg is not None
            and memory_cfg.enabled
            and memory_cfg.integration_mode == "terminal_landmark"
        )
        self.routed_prefix_memory = bool(
            memory_cfg is not None
            and memory_cfg.enabled
            and memory_cfg.integration_mode == "routed_prefix"
        )
        self.memory_working_length = (
            int(memory_cfg.target_length)
            if (self.terminal_landmark_memory or self.routed_prefix_memory)
            else None
        )
        self.memory_config = (
            memory_cfg
            if (self.terminal_landmark_memory or self.routed_prefix_memory)
            else None
        )
        self.historical_reader = None
        self.full_history_enabled = bool(config.full_history_layers)
        self.historical_refresh_tokens = int(
            os.environ.get("MINI_LLM_HISTORICAL_REFRESH_TOKENS", "32")
        )
        if self.historical_read_memory and self.historical_refresh_tokens < 1:
            raise ValueError("MINI_LLM_HISTORICAL_REFRESH_TOKENS must be positive")
        self.memory_router = None
        self.memory_attention_layer = None
        self.terminal_memory_adapter = None
        self.routed_prefix_refresh_tokens = None
        if self.routed_prefix_memory:
            self.routed_prefix_refresh_tokens = int(
                os.environ.get(
                    "MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS",
                    str(int(memory_cfg.inference_route_refresh_tokens)),
                )
            )
            if self.routed_prefix_refresh_tokens <= 0:
                raise ValueError(
                    "MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS must be positive"
                )
        # Short-horizon states remain ordinary KV inference.  When the requested
        # horizon exceeds the working window, hierarchical-memory modes keep the
        # long history outside the deep Transformer cache: 0059I uses terminal
        # Landmark K/V and 0060B uses an exact-token routed prefix.
        
        # Create inference components
        self.n_layers = config.n_layers
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_head
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Embedding will be set from training model
        self.embedding = None
        
        # Create transformer blocks for inference
        self.blocks = []
        for layer_idx in range(config.n_layers):
            attention_config = (
                None if config.attention_layers is None
                else config.attention_layers[layer_idx]
            )
            block = TransformerBlockInference(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                n_experts=config.n_experts,
                top_k=config.top_k,
                ffn_mode=getattr(config, "ffn_mode", "moe"),
                dtype=self.dtype,
                max_context=config.context_length,
                rope_base=config.rope_base,
                attention_config=attention_config,
                layer_idx=layer_idx,
                rms_eps=config.rms_eps,
                dual_resolution_config=(
                    replace(
                        self.dual_resolution_config,
                        sparse_retrieval_phase=(0, 8, 4, 12, 2, 10, 6, 14,
                                                1, 9, 5, 13, 3, 11, 7, 15)[
                            self.dual_resolution_config.sparse_retrieval_layers.index(layer_idx)
                        ],
                    )
                    if (self.dual_resolution_config.sparse_retrieval_stride
                        and layer_idx in self.dual_resolution_config.sparse_retrieval_layers)
                    else self.dual_resolution_config
                ),
            )
            self.blocks.append(block)
        
        # RMSNorm for inference (weights shared with training model)
        self.final_norm = RMSNormInference(config.d_model, eps=config.rms_eps, dtype=self.dtype)
        
        # LM head - will be set from training model
        self.lm_head = None

    def create_generation_state(
        self, batch_size: int, max_length: int, eos_token_id: int = None,
    ) -> GenerationState:
        """
        Create a new generation state with preallocated K/V caches.
        
        Args:
            batch_size: Batch size for generation
            max_length: Maximum sequence length (context + generated tokens)
            eos_token_id: Optional token ending a row's current document. The EOS
                token remains in that document; its row resets before the next token.
            
        Returns:
            GenerationState with preallocated caches
        """
        max_length = int(max_length)
        batch_size = int(batch_size)
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        if eos_token_id is not None:
            eos_token_id = int(eos_token_id)
            if not 0 <= eos_token_id < self.vocab_size:
                raise ValueError("eos_token_id lies outside the vocabulary")
        # Independent full-history heads require their own current K/V state even
        # for short prompts. Route those checkpoints through historical_read with
        # an empty historical prefix rather than silently omitting the branch.
        long_memory = bool(
            not self.dual_resolution_enabled and (
                self.full_history_enabled or (
                    (self.historical_read_memory or self.terminal_landmark_memory
                     or self.routed_prefix_memory)
                    and max_length > int(self.config.memory_context.target_length)
                )
            )
        )
        if long_memory and max_length > int(self.config.memory_context.memory_length):
            raise ValueError(
                f"requested horizon {max_length:,} exceeds the checkpoint's "
                f"trained memory horizon {self.config.memory_context.memory_length:,}"
            )
        if self.dual_resolution_enabled:
            cache_length = int(self.dual_resolution_config.local_window)
            if (
                getattr(self.dual_resolution_config, "compression_mode", "transformer")
                == "matrix_pool"
                and int(getattr(self.dual_resolution_config, "long_local_heads", 0)) > 0
            ):
                cache_length = max(
                    cache_length,
                    int(self.dual_resolution_config.long_local_window),
                )
        elif long_memory and self.routed_prefix_memory:
            cache_length = int(self.memory_config.active_length)
        elif long_memory:
            cache_length = int(self.config.memory_context.target_length)
        else:
            cache_length = max_length
        k_cache = xp.empty(
            (self.n_layers, batch_size, self.n_kv_heads, cache_length, self.head_dim),
            dtype=self.dtype,
        )
        v_cache = xp.empty_like(k_cache)
        router_input_cache = None
        if self.sparse_attention:
            router_input_cache = xp.empty(
                (self.n_layers, batch_size, cache_length, self.d_model),
                dtype=self.dtype,
            )

        dual_state = {}
        if self.dual_resolution_enabled:
            block_size = int(self.dual_resolution_config.block_size)
            max_blocks = int(self.dual_resolution_config.max_blocks)
            matrix_pool = (
                getattr(self.dual_resolution_config, "compression_mode", "transformer")
                == "matrix_pool"
            )
            summary_heads = (
                int(self.dual_resolution_config.global_heads)
                if matrix_pool else self.n_kv_heads
            )
            summary_width = (
                int(self.dual_resolution_config.global_head_dim)
                if matrix_pool else self.head_dim
            )
            summary_capacity = max_blocks * (
                int(self.dual_resolution_config.summaries_per_block)
                if matrix_pool else 1
            )
            summary_shape = (
                self.n_layers, batch_size, summary_heads,
                summary_capacity, summary_width,
            )
            layer_rows = (self.n_layers, batch_size)
            sparse_stride = int(getattr(self.dual_resolution_config, "sparse_retrieval_stride", 0))
            sparse_capacity = (max_length + sparse_stride - 1) // sparse_stride + 1 if sparse_stride else 0
            sparse_shape = (self.n_layers, batch_size, 1, sparse_capacity, self.head_dim)
            dual_state = dict(
                dual_resolution_enabled=True,
                dual_sparse_k_cache=xp.empty(sparse_shape, dtype=self.dtype) if sparse_stride else None,
                dual_sparse_v_cache=xp.empty(sparse_shape, dtype=self.dtype) if sparse_stride else None,
                dual_sparse_count=xp.zeros(layer_rows, dtype=xp.int32) if sparse_stride else None,
                dual_summary_k_cache=xp.empty(summary_shape, dtype=self.dtype),
                dual_summary_v_cache=xp.empty(summary_shape, dtype=self.dtype),
                dual_incomplete_hidden=xp.empty(
                    (self.n_layers, batch_size, block_size, self.d_model),
                    dtype=self.dtype,
                ),
                dual_local_start=xp.zeros(layer_rows, dtype=xp.int32),
                dual_local_count=xp.zeros(layer_rows, dtype=xp.int32),
                dual_summary_start=xp.zeros(layer_rows, dtype=xp.int32),
                dual_summary_count=xp.zeros(layer_rows, dtype=xp.int32),
                dual_incomplete_count=xp.zeros(layer_rows, dtype=xp.int32),
                dual_document_positions=xp.zeros((batch_size,), dtype=xp.int64),
                dual_reset_pending=xp.zeros((batch_size,), dtype=xp.bool_),
                dual_host_counters=(
                    [dict(
                        document_position=0, local_start=0, local_count=0,
                        summary_start=0, summary_count=0,
                        incomplete_count=0, sparse_count=0,
                    ) for _ in range(self.n_layers)]
                    if matrix_pool and batch_size == 1 else None
                ),
            )

        for block in self.blocks:
            block.attention.ensure_rope_capacity(max_length)

        state = GenerationState(
            k_cache=k_cache, v_cache=v_cache, length=0, max_length=max_length,
            batch_size=batch_size, router_input_cache=router_input_cache,
            long_memory_enabled=long_memory, working_capacity=cache_length,
            eos_token_id=eos_token_id, **dual_state,
        )
        if long_memory and self.routed_prefix_memory:
            state.working_capacity = int(self.memory_working_length)
            state.routed_prefix_refresh_tokens = int(self.routed_prefix_refresh_tokens)
        return state

    def _embedding_lookup(self, input_ids):
        """Inference embedding lookup kept as a separately benchmarkable region."""
        x = self.embedding.W.data[input_ids]
        if BACKEND_NAME == "cupy" and not hasattr(x, "__cuda_array_interface__"):
            x = xp.asarray(x)
        return x

    def _lm_head_forward(self, last_hidden):
        """Project one or more hidden rows to vocabulary logits.

        Keeping this operation behind a small method makes the very large
        single-token vocabulary projection independently benchmarkable without
        changing its numerical path.
        """
        if BACKEND_NAME == "cupy" and not hasattr(last_hidden, "__cuda_array_interface__"):
            last_hidden = xp.asarray(last_hidden)
        # 0080E: expose the latest final-normalized causal boundary vector
        # before the shared vocabulary head; multi-horizon proposals consume
        # this representation, never vocab logits inverted back to features.
        self.last_hidden = last_hidden
        return last_hidden @ self.lm_head

    def set_weights(self, training_model):
        """
        Set weights from training model.
        
        Args:
            training_model: Trained DecoderLanguageModel instance
        """
        # Share the embedding reference directly
        self.embedding = training_model.embedding
        
        # Set inference block weights from the executed training prefix only.
        # Progressive checkpoints may intentionally stop before max depth.
        active_layers = int(
            getattr(training_model, "active_layers", len(training_model.blocks))
        )
        if not 1 <= active_layers <= len(self.blocks):
            raise ValueError("training model active depth is outside inference depth")
        self.blocks = self.blocks[:active_layers]
        self.n_layers = active_layers
        for i, (inf_block, train_block) in enumerate(
            zip(self.blocks, training_model.blocks[:active_layers])
        ):
            inf_block.set_weights(train_block)
        
        # Set final norm weights
        self.final_norm.set_weights(training_model.final_norm.gamma.data)
        
        # Set LM head (output projection)
        # output_proj.W is [d_model, vocab], so we use it directly
        self.lm_head = training_model.output_proj.W.data  # [d_model, vocab]
        if self.historical_read_memory:
            self.historical_reader = training_model.historical_memory
            if self.historical_reader is None:
                raise ValueError("historical_read checkpoint has no trained reader")
            for layer_idx, block in enumerate(self.blocks):
                block.attention.historical_reader = self.historical_reader
                block.attention._historical_layer = layer_idx
        if self.terminal_landmark_memory or self.routed_prefix_memory:
            self.memory_router = training_model.memory_router
        if self.terminal_landmark_memory:
            self.memory_attention_layer = int(training_model.memory_attention_layer)
            self.terminal_memory_adapter = (
                training_model.blocks[self.memory_attention_layer].attention.terminal_memory
            )

    @staticmethod
    def _logical_working_ids(state):
        """Return working token IDs in oldest->newest logical order."""
        count = int(state.working_count)
        cap = int(state.working_capacity)
        start = int(state.cache_start)
        if count <= 0:
            return state.working_token_ids[:, :0]
        if count < cap and start == 0:
            return state.working_token_ids[:, :count]
        logical = (xp.arange(count, dtype=xp.int64) + start) % cap
        return state.working_token_ids[:, logical]

    def _refresh_historical_read(self, state):
        """Build read-only encoded history, keeping all 64K tokens OUT of deep KV.

        The block encoder uses independent 32-token groups. Historical source
        slots retain the fixed training horizon and are right-aligned when the
        available history is shorter. Learned routing is sensitive to absolute
        RoPE coordinates; each refresh restores the training coordinates.
        """
        history = state.historical_token_ids
        count = int(history.shape[1])
        if count == 0:
            state.historical_memory_state = None
        else:
            cfg = self.config.memory_context
            limit = int(cfg.distant_memory_length)
            if count > limit:
                history = history[:, -limit:]
                count = limit
                state.historical_token_ids = history
            # Historical sweep phases are defined over the FIXED 61,440-token
            # source horizon, not over the number of available old tokens.
            # Right-align partial history just like training's document-first
            # masks; HistoricalReadMemory.open crops the common invalid prefix
            # before expensive block encoding, so padding remains inexpensive.
            padding = limit - count
            if padding:
                padded = xp.concatenate((
                    xp.zeros((history.shape[0], padding), dtype=xp.int32), history
                ), axis=1)
            else:
                padded = history
            # IMPORTANT: historical learned routing compares RoPE-rotated Q
            # against *unrotated* block_keys. Global RoPE translation is NOT
            # invariant for this route. Keep the training source coordinates
            # [0, H) and the current-window coordinates [H, H+W).
            offset = 0
            starts = xp.full((history.shape[0],), padding, dtype=xp.int64)
            state.historical_memory_state = self.historical_reader.open(
                padded, starts, position_offset=offset, inference=True
            )
        state.historical_reader_routes.clear()
        state.full_history_historical_kv.clear()
        state.historical_refresh_count += 1
        state.historical_tokens_since_refresh = 0

    def _prefill_historical_long(self, input_ids, state):
        """Prefill only 4K current tokens; block-encode older prompt once."""
        batch, total = input_ids.shape
        cap = int(self.config.memory_context.target_length)
        past = max(0, total - cap)
        working = input_ids[:, past:]
        state.historical_token_ids = input_ids[:, :past].copy()
        state.working_token_ids = xp.empty((batch, cap), dtype=xp.int32)
        state.working_token_ids[:, :working.shape[1]] = working
        state.working_count = int(working.shape[1])
        state.working_capacity = cap
        state.cache_start = 0
        state.full_history_current_kv.clear()
        state.full_history_historical_kv.clear()
        state.working_start_abs = past
        state.next_abs_pos = total
        state.historical_origin = past
        self._refresh_historical_read(state)
        x = self._embedding_lookup(working)
        for i, block in enumerate(self.blocks):
            router = (None if state.router_input_cache is None
                      else state.router_input_cache[i])
            x = block.prefill(
                x, state.k_cache[i], state.v_cache[i], 0,
                router_input_cache=router,
                retrieval_route_cache=state.retrieval_routes,
                rope_start_pos=int(self.config.memory_context.distant_memory_length),
                historical_state=state.historical_memory_state,
                full_history_state=state,
                full_history_rope_start=int(self.config.memory_context.distant_memory_length),
            )
        state.length = total
        x = self.final_norm.forward(x)
        return self._lm_head_forward(x[:, -1, :].astype(self.dtype, copy=False))

    def _rebuild_historical_working_cache(self, state):
        """Rebase the 4K Transformer cache to the training window coordinates.

        At the refresh boundary we recompute each layer's *current* 4K rows
        against the refreshed history. This is required for parity: simply
        replacing old history K/V while retaining their downstream working
        hidden states would mix incompatible historical selections. There is
        no 64K-token deep attention or 64K KV allocation here.
        """
        count = int(state.working_count)
        working_ids = self._logical_working_ids(state).copy()
        state.working_token_ids[:, :count] = working_ids
        state.cache_start = 0
        state.full_history_current_kv.clear()
        state.historical_origin = int(state.working_start_abs)
        state.retrieval_routes.clear()
        state.historical_reader_routes.clear()
        x = self._embedding_lookup(working_ids)
        for i, block in enumerate(self.blocks):
            router = (None if state.router_input_cache is None
                      else state.router_input_cache[i])
            x = block.prefill(
                x, state.k_cache[i], state.v_cache[i], 0,
                router_input_cache=router,
                retrieval_route_cache=state.retrieval_routes,
                rope_start_pos=int(self.config.memory_context.distant_memory_length),
                historical_state=state.historical_memory_state,
                full_history_state=state,
                full_history_rope_start=int(self.config.memory_context.distant_memory_length),
            )
        x = self.final_norm.forward(x)
        return self._lm_head_forward(x[:, -1, :].astype(self.dtype, copy=False))

    def _decode_historical_long(self, next_ids, state):
        """Ring-cache decode; refresh external history on a bounded cadence.

        Strict sliding-window parity requires rebuilding both the block
        encoder and all current-layer K/V at each new token. By default this
        work is amortized over a 32-token refresh interval. BETWEEN refreshes
        the changing historical source is not yet visible to read heads, and
        deep hidden states are not recomputed. At refresh, both are rebuilt.
        Use MINI_LLM_HISTORICAL_REFRESH_TOKENS=1 for reference parity.
        """
        cap = int(state.working_capacity)
        abs_pos = int(state.next_abs_pos)
        rope_pos = (int(self.config.memory_context.distant_memory_length)
                    + abs_pos - int(state.historical_origin))
        slid = int(state.working_count) == cap
        refreshed = False
        if slid:
            slot = int(state.cache_start)
            evicted = state.working_token_ids[:, slot:slot+1].copy()
            state.historical_token_ids = xp.concatenate(
                (state.historical_token_ids, evicted), axis=1
            )
            state.cache_start = (slot + 1) % cap
            state.working_start_abs += 1
            state.historical_tokens_since_refresh += 1
            physical = (state.cache_start + cap - 1) % cap
            logical_pos = cap - 1
            # Sliding invalidates the per-window local retrieval indices.
            state.retrieval_routes.clear()
            # When a previously memoryless prompt first crosses 4K, build
            # the historical index immediately (not after 32 missing reads).
            if (state.historical_memory_state is None or
                    state.historical_tokens_since_refresh >= self.historical_refresh_tokens):
                self._refresh_historical_read(state)
                refreshed = True
        else:
            logical_pos = int(state.working_count)
            physical = (int(state.cache_start) + logical_pos) % cap
            state.working_count += 1
        state.working_token_ids[:, physical] = next_ids[:, 0]
        if refreshed:
            state.next_abs_pos += 1
            state.length += 1
            return self._rebuild_historical_working_cache(state)
        new_x = self._embedding_lookup(next_ids)
        # Distinguish historical route-cache epoch and 128-token query boundary.
        epoch = (state.historical_refresh_count, abs_pos // 128)
        for i, block in enumerate(self.blocks):
            router = (None if state.router_input_cache is None
                      else state.router_input_cache[i])
            new_x = block.decode_one(
                new_x, state.k_cache[i], state.v_cache[i], logical_pos,
                router_input_cache=router,
                retrieval_route_cache=state.retrieval_routes,
                cache_position=physical, rope_position=rope_pos,
                cache_start=state.cache_start, working_count=state.working_count,
                route_clock=abs_pos, working_start_abs=state.working_start_abs,
                working_capacity=cap,
                historical_state=state.historical_memory_state,
                historical_routes=state.historical_reader_routes,
                historical_clock=abs_pos, historical_refresh_id=epoch,
                full_history_state=state,
            )
        state.next_abs_pos += 1
        state.length += 1
        new_x = self.final_norm.forward(new_x)
        return self._lm_head_forward(new_x[:, -1, :].astype(self.dtype, copy=False))

    def _refresh_routed_prefix_cache(self, state):
        """Reroute history and rebuild the bounded deep active sequence.

        This is the only operation that changes which top-level historical
        blocks are visible to the Transformer. Between refreshes the selected
        prefix stays fixed and only the 4k working segment rotates.
        """
        cfg = self.memory_config
        working_count = int(state.working_count)
        working_ids = self._logical_working_ids(state)

        # Normalize the token ring whenever the deep cache is rebuilt.  This
        # makes the next decode interval start from a simple contiguous layout.
        if working_count:
            state.working_token_ids[:, :working_count] = working_ids
        state.cache_start = 0

        route = None
        if working_count == int(cfg.target_length):
            history_tokens = int(state.routed_prefix_store.token_count)
            # Before routing has any real compression choice to make, expose
            # all exact historical tokens directly.  The small transition band
            # above the token budget also keeps the newest exact budget until K
            # complete aligned blocks are available.
            if history_tokens > 0:
                route_threshold = (
                    int(cfg.retrieved_length) + int(cfg.block_size) - 1
                )
                if history_tokens < route_threshold:
                    route = state.routed_prefix_store.direct_prefix(
                        int(cfg.retrieved_length)
                    )
                elif state.routed_prefix_store.active_blocks >= int(cfg.top_k_blocks):
                    route = state.routed_prefix_store.route(
                        state.working_embedding_sum, working_count
                    )
                else:
                    route = state.routed_prefix_store.direct_prefix(
                        int(cfg.retrieved_length)
                    )

        if route is None:
            prefix_ids = None
            prefix_positions = None
            prefix_length = 0
            active_ids = working_ids
        else:
            prefix_ids = route.selected_token_ids
            prefix_positions = route.selected_position_ids
            prefix_length = int(prefix_ids.shape[1])
            active_ids = xp.concatenate((prefix_ids, working_ids), axis=1)

        batch = int(active_ids.shape[0])
        working_positions = xp.broadcast_to(
            xp.arange(
                int(state.working_start_abs),
                int(state.working_start_abs) + working_count,
                dtype=xp.int64,
            )[None, :],
            (batch, working_count),
        )
        if route is None:
            active_positions = working_positions
        else:
            active_positions = xp.concatenate(
                (prefix_positions, working_positions), axis=1
            )

        active_x = self._embedding_lookup(active_ids)
        state.retrieval_routes.clear()
        x = active_x
        for i, block in enumerate(self.blocks):
            router_cache = (
                None if state.router_input_cache is None
                else state.router_input_cache[i]
            )
            x = block.prefill(
                x,
                state.k_cache[i],
                state.v_cache[i],
                0,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
                position_ids=active_positions,
            )

        x = self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        state.routed_prefix_route = route
        state.routed_prefix_length = int(prefix_length)
        state.routed_prefix_tokens_since_refresh = 0
        state.routed_prefix_refresh_count += 1
        return self._lm_head_forward(last_hidden)

    def _prefill_routed_prefix_long(self, input_ids, state):
        """0060B long-prompt prefill for deterministic routed-prefix memory."""
        cfg = self.memory_config
        batch, prompt_length = input_ids.shape
        working_capacity = int(cfg.target_length)
        working_count = min(int(prompt_length), working_capacity)
        external_count = max(0, int(prompt_length) - working_count)
        external_ids = input_ids[:, :external_count]
        working_ids = input_ids[:, external_count:]

        store = RoutedPrefixInferenceStore(
            cfg,
            self.memory_router,
            self.embedding.W.data,
            external_capacity_tokens=max(0, int(state.max_length) - working_capacity),
        )
        store.initialize(external_ids, absolute_start=0)
        state.routed_prefix_store = store

        state.working_token_ids = xp.empty(
            (batch, working_capacity), dtype=xp.int32
        )
        if working_count:
            state.working_token_ids[:, :working_count] = working_ids
        working_x = self._embedding_lookup(working_ids)
        state.working_embedding_sum = xp.sum(
            working_x.astype(xp.float32, copy=False), axis=1
        )
        state.working_count = int(working_count)
        state.cache_start = 0
        state.working_start_abs = int(external_count)
        state.next_abs_pos = int(prompt_length)
        state.length = int(prompt_length)
        return self._refresh_routed_prefix_cache(state)

    def _decode_routed_prefix_long(self, next_ids, state):
        """Decode one token with a fixed prefix and rotating 4k working cache."""
        cfg = self.memory_config
        batch = int(next_ids.shape[0])
        cap = int(state.working_capacity)
        absolute_position = int(state.next_abs_pos)
        new_embedding = self._embedding_lookup(next_ids)
        new_row = new_embedding[:, 0, :]
        slid = False

        if int(state.working_count) >= cap:
            slid = True
            old_slot = int(state.cache_start)
            old_ids = state.working_token_ids[:, old_slot]
            state.routed_prefix_store.append_evicted(
                old_ids, int(state.working_start_abs)
            )
            old_embedding = self.embedding.W.data[old_ids].astype(
                xp.float32, copy=False
            )
            state.working_embedding_sum -= old_embedding
            state.cache_start = (old_slot + 1) % cap
            state.working_start_abs += 1
            logical_work_position = cap - 1
            physical_work_slot = (
                int(state.cache_start) + logical_work_position
            ) % cap
            state.working_count = cap
            state.routed_prefix_tokens_since_refresh += 1
        else:
            logical_work_position = int(state.working_count)
            physical_work_slot = (
                int(state.cache_start) + logical_work_position
            ) % cap
            state.working_count += 1

        state.working_token_ids[:, physical_work_slot] = next_ids[:, 0]
        state.working_embedding_sum += new_row.astype(xp.float32, copy=False)
        state.next_abs_pos += 1
        state.length += 1

        can_prefix = bool(
            int(state.working_count) == cap
            and int(state.routed_prefix_store.token_count) > 0
        )
        route_active = state.routed_prefix_route is not None
        need_refresh = bool(
            (can_prefix and not route_active)
            or (
                route_active
                and int(state.routed_prefix_tokens_since_refresh)
                >= int(state.routed_prefix_refresh_tokens)
            )
        )
        if need_refresh:
            return self._refresh_routed_prefix_cache(state)

        prefix_length = int(state.routed_prefix_length)
        logical_position = prefix_length + int(state.working_count) - 1
        cache_position = prefix_length + int(physical_work_slot)
        active_count = prefix_length + int(state.working_count)

        # Once the 4k ring slides, compact working indices refer to different
        # tokens. Recompute the small per-layer learned retrieval routes rather
        # than reusing block IDs from the previous logical window.
        if slid:
            state.retrieval_routes.clear()

        x = new_embedding
        for i, block in enumerate(self.blocks):
            router_cache = (
                None if state.router_input_cache is None
                else state.router_input_cache[i]
            )
            x = block.decode_one(
                x,
                state.k_cache[i],
                state.v_cache[i],
                logical_position,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
                cache_position=cache_position,
                rope_position=absolute_position,
                cache_start=state.cache_start,
                working_count=active_count,
                route_clock=absolute_position,
                working_start_abs=state.working_start_abs,
                fixed_prefix_length=prefix_length,
                working_capacity=cap,
            )

        x = self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        return self._lm_head_forward(last_hidden)

    def _prefill_terminal_long(self, input_ids, state):
        cfg = self.memory_config
        B, T_prompt = input_ids.shape
        working_capacity = int(cfg.target_length)
        working_count = min(int(T_prompt), working_capacity)
        external_count = max(0, int(T_prompt) - working_count)
        external_ids = input_ids[:, :external_count]
        working_ids = input_ids[:, external_count:]

        # Use one monotonic RoPE coordinate across external + working tokens.
        # RoPE is translation invariant, while the external store itself handles
        # the right-aligned 128-token block phase used by training.
        working_start_abs = external_count

        store = TerminalMemoryInferenceStore(
            cfg, self.memory_router, self.terminal_memory_adapter, self.embedding.W.data,
            external_capacity_tokens=max(0, int(state.max_length) - working_capacity),
        )
        store.initialize(external_ids, absolute_start=0)
        state.terminal_memory_store = store

        working_x = self._embedding_lookup(working_ids)
        state.working_token_ids = xp.empty(
            (B, working_capacity), dtype=xp.int32
        )
        state.working_token_ids[:, :working_count] = working_ids
        state.working_embedding_sum = xp.sum(
            working_x.astype(xp.float32, copy=False), axis=1
        )
        state.working_count = working_count
        state.cache_start = 0
        state.working_start_abs = int(working_start_abs)
        state.next_abs_pos = int(working_start_abs + working_count)
        state.terminal_memory_route = store.route(
            state.working_embedding_sum, working_count
        )

        x = working_x
        for i, block in enumerate(self.blocks):
            router_cache = None if state.router_input_cache is None else state.router_input_cache[i]
            use_memory = i == self.memory_attention_layer
            x = block.prefill(
                x, state.k_cache[i], state.v_cache[i], 0,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
                rope_start_pos=working_start_abs,
                terminal_memory_store=store if use_memory else None,
                terminal_memory_route=state.terminal_memory_route if use_memory else None,
            )

        x = self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        state.length = int(T_prompt)
        return self._lm_head_forward(last_hidden)

    def _decode_terminal_long(self, next_ids, state):
        B = int(next_ids.shape[0])
        new_embedding = self._embedding_lookup(next_ids)
        new_embedding_row = new_embedding[:, 0, :]
        cap = int(state.working_capacity)

        if int(state.working_count) >= cap:
            old_slot = int(state.cache_start)
            old_ids = state.working_token_ids[:, old_slot]
            state.terminal_memory_store.append_evicted(
                old_ids, state.working_start_abs
            )
            old_embedding = self.embedding.W.data[old_ids].astype(xp.float32, copy=False)
            state.working_embedding_sum -= old_embedding
            state.cache_start = (old_slot + 1) % cap
            state.working_start_abs += 1
            logical_position = cap - 1
            cache_position = (state.cache_start + logical_position) % cap
            state.working_count = cap
        else:
            logical_position = int(state.working_count)
            cache_position = (int(state.cache_start) + logical_position) % cap
            state.working_count += 1

        state.working_token_ids[:, cache_position] = next_ids[:, 0]
        state.working_embedding_sum += new_embedding_row.astype(xp.float32, copy=False)
        rope_position = int(state.next_abs_pos)
        state.next_abs_pos += 1
        state.terminal_memory_route = state.terminal_memory_store.route(
            state.working_embedding_sum, state.working_count
        )

        x = new_embedding
        for i, block in enumerate(self.blocks):
            router_cache = None if state.router_input_cache is None else state.router_input_cache[i]
            use_memory = i == self.memory_attention_layer
            x = block.decode_one(
                x, state.k_cache[i], state.v_cache[i], logical_position,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
                cache_position=cache_position, rope_position=rope_position,
                cache_start=state.cache_start, working_count=state.working_count,
                route_clock=rope_position, working_start_abs=state.working_start_abs,
                terminal_memory_store=state.terminal_memory_store if use_memory else None,
                terminal_memory_route=state.terminal_memory_route if use_memory else None,
            )

        x = self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        state.length += 1
        return self._lm_head_forward(last_hidden)

    def _dual_incremental_token(self, token_ids, state):
        """Advance all dual-resolution rows by one token without prior recompute."""
        if token_ids.ndim != 2 or tuple(token_ids.shape) != (state.batch_size, 1):
            raise ValueError("dual-resolution token input must be [batch_size,1]")
        if not state.has_capacity(1):
            raise ValueError("No capacity remaining in generation cache.")

        pending = state.dual_reset_pending
        host_counters = state.dual_host_counters
        if host_counters is not None:
            if state.dual_host_reset_pending:
                state.dual_local_start[:, 0] = 0
                state.dual_local_count[:, 0] = 0
                if state.dual_sparse_count is not None:
                    state.dual_sparse_count[:, 0] = 0
                state.dual_summary_start[:, 0] = 0
                state.dual_summary_count[:, 0] = 0
                state.dual_incomplete_count[:, 0] = 0
                state.dual_document_positions[0] = 0
                pending[0] = False
                for counters in host_counters:
                    counters.update(
                        document_position=0, local_start=0, local_count=0,
                        summary_start=0, summary_count=0,
                        incomplete_count=0, sparse_count=0,
                    )
                state.dual_host_reset_pending = False
        elif bool(xp.any(pending)):
            rows = xp.nonzero(pending)[0]
            state.dual_local_start[:, rows] = 0
            state.dual_local_count[:, rows] = 0
            if state.dual_sparse_count is not None:
                state.dual_sparse_count[:, rows] = 0
            state.dual_summary_start[:, rows] = 0
            state.dual_summary_count[:, rows] = 0
            state.dual_incomplete_count[:, rows] = 0
            state.dual_document_positions[rows] = 0
            pending[rows] = False

        x = self._embedding_lookup(token_ids)
        chain_norms = bool(self.blocks) and self.blocks[0].use_fp32_residual
        normalized_input = None
        for i, block in enumerate(self.blocks):
            next_norm = None
            if chain_norms:
                next_norm = (
                    self.blocks[i + 1].norm1
                    if i + 1 < len(self.blocks)
                    else self.final_norm
                )
            block_output = block.decode_one(
                x, state.k_cache[i], state.v_cache[i], state.length,
                normalized_input=normalized_input, next_norm=next_norm,
                dual_summary_k=state.dual_summary_k_cache[i],
                dual_summary_v=state.dual_summary_v_cache[i],
                dual_incomplete_hidden=state.dual_incomplete_hidden[i],
                dual_local_start=state.dual_local_start[i],
                dual_local_count=state.dual_local_count[i],
                dual_summary_start=state.dual_summary_start[i],
                dual_summary_count=state.dual_summary_count[i],
                dual_incomplete_count=state.dual_incomplete_count[i],
                dual_document_positions=state.dual_document_positions,
                dual_sparse_k=None if state.dual_sparse_k_cache is None else state.dual_sparse_k_cache[i],
                dual_sparse_v=None if state.dual_sparse_v_cache is None else state.dual_sparse_v_cache[i],
                dual_sparse_count=None if state.dual_sparse_count is None else state.dual_sparse_count[i],
                dual_host_counters=None if host_counters is None else host_counters[i],
            )
            if chain_norms:
                x, normalized_input = block_output
            else:
                x = block_output

        x = normalized_input if chain_norms else self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        logits = self._lm_head_forward(last_hidden)
        state.dual_document_positions += 1
        if host_counters is not None:
            for counters in host_counters:
                counters["document_position"] += 1
        if state.eos_token_id is not None:
            eos_rows = token_ids[:, 0] == int(state.eos_token_id)
            state.dual_reset_pending |= eos_rows
            if host_counters is not None:
                state.dual_host_reset_pending = bool(eos_rows[0].item())
        state.length += 1
        return logits

    def prefill(self, input_ids: xp.ndarray, state: GenerationState) -> xp.ndarray:
        """
        Prefill KV cache for prompt.
        
        Processes the entire prompt in one forward pass, writing
        all K/V to the cache. Returns logits for the last position only.
        
        Args:
            input_ids: Input token IDs, shape [B, T_prompt]
            state: Generation state with preallocated caches
            
        Returns:
            logits: Output logits [B, vocab_size] (only last position)
        """
        # Ensure input_ids is the correct backend type and dtype
        if not isinstance(input_ids, xp.ndarray):
            input_ids = xp.asarray(input_ids, dtype=xp.int32)
        else:
            # Convert to the correct dtype if needed
            input_ids = input_ids.astype(xp.int32, copy=False)
        
        B, T_prompt = input_ids.shape
        if B != state.batch_size:
            raise ValueError("input batch size does not match generation state")
        if T_prompt <= 0:
            raise ValueError("prefill requires at least one token")

        if state.dual_resolution_enabled:
            if not state.has_capacity(T_prompt):
                raise ValueError(
                    f"Prompt length {T_prompt} exceeds remaining capacity "
                    f"{state.remaining_capacity()}."
                )
            from mini_llm.native_cuda_graph_decode import (
                enabled as native_graph_enabled,
                strict as native_graph_strict,
            )
            existing_native_graph = getattr(state, "_native_cuda_graph", None)
            native_graph = None
            if native_graph_enabled():
                try:
                    from mini_llm.native_dual_cuda_graph_decode import (
                        NativeDualBF16CudaGraphDecoder,
                    )
                    if (
                        isinstance(existing_native_graph, NativeDualBF16CudaGraphDecoder)
                        and existing_native_graph.engine is self
                        and existing_native_graph.state is state
                    ):
                        native_graph = existing_native_graph
                    else:
                        xp.cuda.get_current_stream().synchronize()
                        native_graph = NativeDualBF16CudaGraphDecoder(self, state)
                except Exception as exc:
                    state._native_cuda_graph_failure = f"{type(exc).__name__}: {exc}"
                    if native_graph_strict():
                        raise
            if native_graph is not None:
                state._native_cuda_graph = native_graph
                state._native_cuda_graph_failure = None
                logits = None
                for offset in range(T_prompt):
                    logits = native_graph.decode_one(
                        input_ids[:, offset:offset + 1], state
                    )
                return logits

            state._native_cuda_graph = None
            if not native_graph_enabled():
                state._native_cuda_graph_failure = None
            logits = None
            for offset in range(T_prompt):
                logits = self._dual_incremental_token(
                    input_ids[:, offset:offset + 1], state
                )
            return logits

        if state.long_memory_enabled:
            if T_prompt > int(state.max_length):
                input_ids = input_ids[:, -int(state.max_length):]
            if self.historical_read_memory:
                return self._prefill_historical_long(input_ids, state)
            if self.routed_prefix_memory:
                return self._prefill_routed_prefix_long(input_ids, state)
            return self._prefill_terminal_long(input_ids, state)
        
        if not state.has_capacity(T_prompt):
            raise ValueError(
                f"Prompt length {T_prompt} exceeds remaining capacity "
                f"{state.remaining_capacity()}."
            )
        
        # Embed tokens.
        x = self._embedding_lookup(input_ids)  # [B, T_prompt, D]
    
        # Process through transformer blocks
        start_pos = 0
        for i, block in enumerate(self.blocks):
            router_cache = (
                None if state.router_input_cache is None
                else state.router_input_cache[i]
            )
            x = block.prefill(
                x, state.k_cache[i], state.v_cache[i], start_pos,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
            )
        
        # Final normalization follows the training model's FP32-residual /
        # low-precision-head policy.
        x = self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        
        # LM head - return only last position logits.
        logits = self._lm_head_forward(last_hidden)  # [B, vocab_size]
        
        # Update state length
        state.length += T_prompt

        # Native BF16 graph decode is deliberately opt-in and narrowly gated.
        # Unsupported checkpoints retain this established allocating path.
        from mini_llm.native_cuda_graph_decode import (
            NativeBF16CudaGraphDecoder,
            enabled as native_graph_enabled,
            strict as native_graph_strict,
        )
        state._native_cuda_graph = None
        state._native_cuda_graph_failure = None
        if native_graph_enabled():
            try:
                xp.cuda.get_current_stream().synchronize()
                state._native_cuda_graph = NativeBF16CudaGraphDecoder(self, state)
            except Exception as exc:
                state._native_cuda_graph_failure = f"{type(exc).__name__}: {exc}"
                if native_graph_strict():
                    raise

        return logits

    def forward_provisional_block(self, active_embeddings, state):
        """0079D: K active provisional queries with reused committed-context KV.

        Only available for direct, dense GQA. This does not mutate the cache;
        provisional bidirectional states are never reusable causal entries.
        """
        if self.full_history_enabled:
            raise NotImplementedError(
                "predictive refinement does not support independent full-history heads"
            )
        if self.sparse_attention or state.long_memory_enabled:
            raise NotImplementedError(
                "predictive refinement cache supports direct dense attention only"
            )
        if self.embedding is None or self.lm_head is None:
            raise RuntimeError("load training weights before provisional inference")
        if active_embeddings.ndim != 3 or active_embeddings.shape[0] != state.batch_size or active_embeddings.shape[-1] != self.d_model:
            raise ValueError("active_embeddings must be [B,K,d_model]")
        k = int(active_embeddings.shape[1])
        if k <= 0 or not state.has_capacity(k):
            raise ValueError("provisional block exceeds KV horizon")
        x = active_embeddings
        for i, block in enumerate(self.blocks):
            x = block.forward_provisional_block(
                x, state.k_cache[i], state.v_cache[i], state.length
            )
        x = self.final_norm.forward(x)
        head = x.astype(self.dtype, copy=False).reshape(-1, self.d_model)
        return self._lm_head_forward(head).reshape(state.batch_size, k, self.vocab_size)

    def append_causal_block(self, committed_ids, state):
        """0079D: append actual tokens causally; never reuse provisional KV.

        A single prefix-append prefill processes the K fixed tokens together,
        with the existing triangular attention over new tokens and the
        immutable preceding cache. State length advances only on success.
        """
        if self.full_history_enabled:
            raise NotImplementedError(
                "predictive refinement cache append does not support independent "
                "full-history heads"
            )
        if self.sparse_attention or state.long_memory_enabled:
            raise NotImplementedError(
                "predictive refinement cache append supports direct dense attention only"
            )
        ids = xp.asarray(committed_ids, dtype=xp.int32)
        if ids.ndim != 2 or ids.shape[0] != state.batch_size:
            raise ValueError("committed_ids must be [B,K]")
        k = int(ids.shape[1])
        if k == 0:
            return
        if not state.has_capacity(k):
            raise ValueError("committed block exceeds KV capacity")
        x = self._embedding_lookup(ids)
        start = int(state.length)
        for i, block in enumerate(self.blocks):
            x = block.prefill(x, state.k_cache[i], state.v_cache[i], start)
        state.length += k
        # 0080E: the just-committed block becomes the causal history for the
        # NEXT proposal. Preserve its boundary state without a redundant
        # Transformer traversal or an extra large vocabulary projection.
        norm = self.final_norm.forward(x[:, -1:, :])
        self.last_hidden = norm[:, -1, :].astype(self.dtype, copy=False)

    def decode_one(self, next_ids: xp.ndarray, state: GenerationState) -> xp.ndarray:
        """
        Decode one token using cached K/V.
        
        Args:
            next_ids: Next token IDs, shape [B] or [B, 1]
            state: Generation state with cached K/V
            
        Returns:
            logits: Output logits [B, vocab_size]
        """
        # Ensure next_ids is the correct backend type and dtype
        if not isinstance(next_ids, xp.ndarray):
            next_ids = xp.asarray(next_ids, dtype=xp.int32)
        else:
            # Convert to the correct dtype if needed
            next_ids = next_ids.astype(xp.int32, copy=False)
        
        # Ensure shape [B, 1]
        if next_ids.ndim == 1:
            next_ids = next_ids[:, None]
        
        B, T_new = next_ids.shape
        assert T_new == 1, "decode_one processes exactly one token"
        if B != state.batch_size:
            raise ValueError("input batch size does not match generation state")

        if not state.has_capacity(1):
            raise ValueError("No capacity remaining in generation cache.")

        native_graph = getattr(state, "_native_cuda_graph", None)
        if native_graph is not None:
            return native_graph.decode_one(next_ids, state)

        if state.dual_resolution_enabled:
            return self._dual_incremental_token(next_ids, state)

        if state.long_memory_enabled:
            if self.historical_read_memory:
                return self._decode_historical_long(next_ids, state)
            if self.routed_prefix_memory:
                return self._decode_routed_prefix_long(next_ids, state)
            return self._decode_terminal_long(next_ids, state)
        
        # Embed token.
        x = self._embedding_lookup(next_ids)  # [B, 1, D]
        
        # Process through transformer blocks. Low-precision models chain each
        # FFN residual add into the next block's RMSNorm (and the last one into
        # final_norm), avoiding separate cast/multiply/add/norm launches.
        start_pos = state.length
        chain_norms = bool(self.blocks) and self.blocks[0].use_fp32_residual
        normalized_input = None
        for i, block in enumerate(self.blocks):
            router_cache = (
                None if state.router_input_cache is None
                else state.router_input_cache[i]
            )
            next_norm = None
            if chain_norms:
                next_norm = (
                    self.blocks[i + 1].norm1
                    if i + 1 < len(self.blocks)
                    else self.final_norm
                )
            block_output = block.decode_one(
                x, state.k_cache[i], state.v_cache[i], start_pos,
                router_input_cache=router_cache,
                retrieval_route_cache=state.retrieval_routes,
                normalized_input=normalized_input,
                next_norm=next_norm,
            )
            if chain_norms:
                x, normalized_input = block_output
            else:
                x = block_output

        # Final normalization follows the training model's FP32-residual /
        # low-precision-head policy. The chained path computed it with the last
        # residual addition.
        x = normalized_input if chain_norms else self.final_norm.forward(x)
        last_hidden = x[:, -1, :].astype(self.dtype, copy=False)
        
        # LM head - return only last position logits.
        logits = self._lm_head_forward(last_hidden)  # [B, vocab_size]
        
        # Update state length
        state.length += 1
        
        return logits
