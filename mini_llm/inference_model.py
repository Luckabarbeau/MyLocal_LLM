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
        self.sparse_attention = config.attention_layers is not None
        memory_cfg = getattr(config, "memory_context", None)
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
                dtype=self.dtype,
                max_context=config.context_length,
                rope_base=config.rope_base,
                attention_config=attention_config,
                layer_idx=layer_idx,
            )
            self.blocks.append(block)
        
        # RMSNorm for inference (weights shared with training model)
        self.final_norm = RMSNormInference(config.d_model, eps=config.rms_eps, dtype=self.dtype)
        
        # LM head - will be set from training model
        self.lm_head = None

    def create_generation_state(self, batch_size: int, max_length: int) -> GenerationState:
        """
        Create a new generation state with preallocated K/V caches.
        
        Args:
            batch_size: Batch size for generation
            max_length: Maximum sequence length (context + generated tokens)
            
        Returns:
            GenerationState with preallocated caches
        """
        max_length = int(max_length)
        long_memory = bool(
            (self.terminal_landmark_memory or self.routed_prefix_memory)
            and max_length > int(self.memory_working_length)
        )
        if long_memory and max_length > int(self.memory_config.memory_length):
            raise ValueError(
                f"requested horizon {max_length:,} exceeds the checkpoint's "
                f"trained memory horizon {self.memory_config.memory_length:,}"
            )
        if long_memory and self.routed_prefix_memory:
            cache_length = int(self.memory_config.active_length)
        elif long_memory:
            cache_length = int(self.memory_working_length)
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

        for block in self.blocks:
            block.attention.ensure_rope_capacity(max_length)

        state = GenerationState(
            k_cache=k_cache, v_cache=v_cache, length=0, max_length=max_length,
            batch_size=batch_size, router_input_cache=router_input_cache,
            long_memory_enabled=long_memory, working_capacity=cache_length,
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
            input_ids = input_ids.astype(xp.int32)
        
        B, T_prompt = input_ids.shape

        if state.long_memory_enabled:
            if T_prompt > int(state.max_length):
                input_ids = input_ids[:, -int(state.max_length):]
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
        
        return logits

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
            next_ids = next_ids.astype(xp.int32)
        
        # Ensure shape [B, 1]
        if next_ids.ndim == 1:
            next_ids = next_ids[:, None]
        
        B, T_new = next_ids.shape
        assert T_new == 1, "decode_one processes exactly one token"

        if state.long_memory_enabled:
            if self.routed_prefix_memory:
                return self._decode_routed_prefix_long(next_ids, state)
            return self._decode_terminal_long(next_ids, state)
        
        if not state.has_capacity(1):
            raise ValueError("No capacity remaining in generation cache.")
        
        # Embed token.
        x = self._embedding_lookup(next_ids)  # [B, 1, D]
        
        # Process through transformer blocks
        start_pos = state.length
        for i, block in enumerate(self.blocks):
            router_cache = (
                None if state.router_input_cache is None
                else state.router_input_cache[i]
            )
            x = block.decode_one(
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
        state.length += 1
        
        return logits
