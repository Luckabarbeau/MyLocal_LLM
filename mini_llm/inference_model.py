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

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.config import ModelConfig
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.rmsnorm_inference import RMSNormInference
from mini_llm.blocks.transformer_block_inference import TransformerBlockInference
from mini_llm.inference_state import GenerationState
from mini_llm.ops.terminal_memory_inference import TerminalMemoryInferenceStore


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
        self.memory_working_length = (
            int(memory_cfg.target_length) if self.terminal_landmark_memory else None
        )
        self.memory_config = memory_cfg if self.terminal_landmark_memory else None
        self.memory_router = None
        self.memory_attention_layer = None
        self.terminal_memory_adapter = None
        # Short prompts remain ordinary sparse 4k inference.  When the requested
        # horizon exceeds the working window, 0059I adds the separate external
        # terminal-Landmark block store while keeping these deep caches bounded.
        
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
            self.terminal_landmark_memory
            and max_length > int(self.memory_working_length)
        )
        if long_memory and max_length > int(self.memory_config.memory_length):
            raise ValueError(
                f"requested horizon {max_length:,} exceeds the checkpoint's "
                f"trained terminal-memory horizon {self.memory_config.memory_length:,}"
            )
        cache_length = int(self.memory_working_length) if long_memory else max_length
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

        return GenerationState(
            k_cache=k_cache, v_cache=v_cache, length=0, max_length=max_length,
            batch_size=batch_size, router_input_cache=router_input_cache,
            long_memory_enabled=long_memory, working_capacity=cache_length,
        )

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
        
        # Set inference block weights from training blocks
        for i, (inf_block, train_block) in enumerate(zip(self.blocks, training_model.blocks)):
            inf_block.set_weights(train_block)
        
        # Set final norm weights
        self.final_norm.set_weights(training_model.final_norm.gamma.data)
        
        # Set LM head (output projection)
        # output_proj.W is [d_model, vocab], so we use it directly
        self.lm_head = training_model.output_proj.W.data  # [d_model, vocab]
        if self.terminal_landmark_memory:
            self.memory_router = training_model.memory_router
            self.memory_attention_layer = int(training_model.memory_attention_layer)
            self.terminal_memory_adapter = (
                training_model.blocks[self.memory_attention_layer].attention.terminal_memory
            )

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
