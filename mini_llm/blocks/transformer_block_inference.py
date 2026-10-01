"""Inference-only Transformer block with KV cache."""


import numpy as _np
from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.ops.attention_inference import GQAAttentionInference
from mini_llm.ops.rmsnorm_inference import RMSNormInference


class TransformerBlockInference:
    """
    Transformer block for inference with KV cache.
    
    Structure:
        X → RMSNorm → Attention (with KV cache) → +X → RMSNorm → MoE → +Y
        
    This is the inference counterpart to TransformerBlock, using
    preallocated KV cache instead of recomputing full prefixes.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        n_experts: int,
        top_k: int,
        dtype: str = "float32",
        max_context: int = None
    ):
        """
        Initialize inference Transformer block.
        
        Args:
            d_model: Model dimension
            n_q_heads: Number of query heads
            n_kv_heads: Number of key/value heads
            d_head: Dimension per head
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            top_k: Number of top experts to use
            dtype: Data type
        """
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        
        # RMSNorms (weights shared with training)
        self.norm1 = RMSNormInference(d_model, dtype=dtype)
        self.norm2 = RMSNormInference(d_model, dtype=dtype)
        
        # Attention (weights shared with training)
        self.attention = GQAAttentionInference(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            dtype=dtype,
            max_context=max_context
        )
        
        # MoE (weights shared with training)
        from mini_llm.blocks.moe_block_inference import MoEInference
        self.moe = MoEInference(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            k=top_k,
            dtype=dtype
        )

    def set_weights(self, training_block):
        """
        Set weights from training TransformerBlock.
        
        Args:
            training_block: Training TransformerBlock instance
        """
        self.norm1.set_weights(training_block.norm1.gamma.data)
        self.norm2.set_weights(training_block.norm2.gamma.data)
        self.attention.set_weights(
            training_block.attention.Wq.data,
            training_block.attention.Wk.data,
            training_block.attention.Wv.data,
            training_block.attention.Wo.data
        )
        self.moe.set_weights(training_block.moe)

    def prefill(self, x, k_cache, v_cache, start_pos):
        """
        Prefill through this block.
        
        Args:
            x: Input [B, T_prompt, D]
            k_cache: K cache for this layer
            v_cache: V cache for this layer
            start_pos: Starting position
            
        Returns:
            y: Output [B, T_prompt, D]
        """
        # First residual branch: attention (return all positions)
        residual1 = x
        h = self.norm1.forward(x)
        
        # Ensure h is on the correct backend for attention operations
        if BACKEND_NAME == "cupy" and isinstance(h, _np.ndarray):
            h = xp.asarray(h)
        
        a = self.attention.prefill(h, k_cache, v_cache, start_pos, return_all=True)
        x = residual1 + a
        
        # Second residual branch: MoE
        residual2 = x
        h = self.norm2.forward(x)
        
        # Ensure h is on the correct backend for MoE operations
        if BACKEND_NAME == "cupy" and isinstance(h, _np.ndarray):
            h = xp.asarray(h)
        
        m = self.moe.prefill(h)
        y = residual2 + m
        
        return y

    def decode_one(self, x, k_cache, v_cache, start_pos):
        """
        Decode one token through this block.
        
        Args:
            x: Input [B, 1, D]
            k_cache: K cache for this layer
            v_cache: V cache for this layer
            start_pos: Current position
            
        Returns:
            y: Output [B, 1, D]
        """
        # First residual branch: attention
        h = self.norm1.forward(x)
        
        # Ensure h is on the correct backend for attention operations
        if BACKEND_NAME == "cupy" and isinstance(h, _np.ndarray):
            h = xp.asarray(h)
        
        a = self.attention.decode_one(h, k_cache, v_cache, start_pos)
        
        x = x + a
        
        # Second residual branch: MoE
        h = self.norm2.forward(x)
        
        # Ensure h is on the correct backend for MoE operations
        if BACKEND_NAME == "cupy" and isinstance(h, _np.ndarray):
            h = xp.asarray(h)
        
        m = self.moe.decode_one(h)
        
        y = x + m
        
        return y
