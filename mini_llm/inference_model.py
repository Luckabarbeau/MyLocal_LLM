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
        # 0059C short-history fallback: a terminal-Landmark checkpoint is an
        # ordinary sparse 4k Transformer whenever no tokens exist outside the
        # working window.  Do not reject such checkpoints at construction.
        # The true >working-window external-memory cache remains a separate
        # inference feature and is guarded in create_generation_state() so we
        # never silently substitute incorrect long-memory semantics.
        
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
        if (
            self.terminal_landmark_memory
            and max_length > int(self.memory_working_length)
        ):
            raise NotImplementedError(
                "0058C terminal-Landmark KV inference currently supports the "
                f"short-history fallback through {self.memory_working_length:,} "
                "tokens. A larger cache requires the external-history Landmark "
                "store/router inference path; use --max-context at or below the "
                "working-window length until that path is enabled."
            )
        k_cache = xp.empty(
            (
                self.n_layers,
                batch_size,
                self.n_kv_heads,
                max_length,
                self.head_dim,
            ),
            dtype=self.dtype,
        )
        
        v_cache = xp.empty_like(k_cache)
        router_input_cache = None
        if self.sparse_attention:
            router_input_cache = xp.empty(
                (
                    self.n_layers, batch_size, max_length, self.d_model,
                ),
                dtype=self.dtype,
            )

        # RoPE tables are shared through the module-level cache, so the first
        # block may create/grow the table and subsequent blocks reuse it.
        for block in self.blocks:
            block.attention.ensure_rope_capacity(max_length)
        
        return GenerationState(
            k_cache=k_cache,
            v_cache=v_cache,
            length=0,
            max_length=max_length,
            batch_size=batch_size,
            router_input_cache=router_input_cache,
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
