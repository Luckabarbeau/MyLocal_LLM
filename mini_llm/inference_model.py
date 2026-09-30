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
            block = TransformerBlockInference(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                n_experts=config.n_experts,
                top_k=config.top_k,
                dtype=self.dtype
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
        
        return GenerationState(
            k_cache=k_cache,
            v_cache=v_cache,
            length=0,
            max_length=max_length,
            batch_size=batch_size,
        )

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
        
        # Embed tokens - ensure we stay on the correct backend
        W_data = self.embedding.W.data
        
        x = W_data[input_ids]  # [B, T_prompt, D]
        
        # Ensure x is the correct backend type (CuPy if CuPy backend)
        # CuPy arrays should have __cuda_array_interface__ attribute
        if BACKEND_NAME == "cupy" and not hasattr(x, '__cuda_array_interface__'):
            x = xp.asarray(x)
    
        # Process through transformer blocks
        start_pos = 0
        for i, block in enumerate(self.blocks):
            x = block.prefill(x, state.k_cache[i], state.v_cache[i], start_pos)
        
        # Final normalization
        x = self.final_norm.forward(x)
        
        # LM head - return only last position logits
        last_hidden = x[:, -1, :]  # [B, d_model]
        
        # Ensure last_hidden is on the correct backend for matmul with lm_head
        # CuPy arrays should have __cuda_array_interface__ attribute
        if BACKEND_NAME == "cupy" and not hasattr(last_hidden, '__cuda_array_interface__'):
            last_hidden = xp.asarray(last_hidden)
        
        logits = last_hidden @ self.lm_head  # [B, vocab_size]
        
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
        
        # Embed token
        x = self.embedding.W.data[next_ids]  # [B, 1, D]
        
        # Ensure x is the correct backend type (CuPy if CuPy backend)
        # CuPy arrays should have __cuda_array_interface__ attribute
        if BACKEND_NAME == "cupy" and not hasattr(x, '__cuda_array_interface__'):
            x = xp.asarray(x)
        
        # Process through transformer blocks
        start_pos = state.length
        for i, block in enumerate(self.blocks):
            x = block.decode_one(x, state.k_cache[i], state.v_cache[i], start_pos)
        
        # Final normalization
        x = self.final_norm.forward(x)
        
        # LM head - return only last position logits
        last_hidden = x[:, -1, :]  # [B, d_model]
        
        # Ensure last_hidden is on the correct backend for matmul with lm_head
        # CuPy arrays should have __cuda_array_interface__ attribute
        if BACKEND_NAME == "cupy" and not hasattr(last_hidden, '__cuda_array_interface__'):
            last_hidden = xp.asarray(last_hidden)
        
        logits = last_hidden @ self.lm_head  # [B, vocab_size]
        
        # Update state length
        state.length += 1
        
        return logits
