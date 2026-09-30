"""Inference-only attention with preallocated KV cache.

This module provides an inference-specific attention implementation that:
- Uses a preallocated K/V cache instead of recomputing prefixes
- Supports both prefill (batched prompt processing) and decode (single-token)
- Maintains absolute position RoPE for correct positional encoding
- Implements grouped-query attention with proper KV sharing
"""

import numpy as _np
from mini_llm.backend import xp, BACKEND_NAME


class GQAAttentionInference:
    """
    Grouped-query attention for autoregressive inference with KV cache.
    
    During inference, keys and values from previous tokens are cached and reused.
    The cache is preallocated to avoid reallocation during generation.
    
    Cache layout (for all layers):
        K: [n_layers, batch, n_kv_heads, max_context, head_dim]
        V: [n_layers, batch, n_kv_heads, max_context, head_dim]
    
    During prefill:
        - Process the entire prompt in one forward pass
        - Write K/V to cache positions [0, prompt_length)
        
    During decode:
        - Process only the new token (T_new = 1)
        - Append K/V to cache at current position

    RoPE uses absolute positions based on cache position, not relative
    positions within the current input.
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        rope_base=10_000.0, dtype="float32"
    ):
        """
        Initialize inference attention (weights shared with training version).
        
        Args:
            d_model: Model dimension
            n_q_heads: Number of query heads
            n_kv_heads: Number of key/value heads
            d_head: Dimension per head
            rope_base: RoPE base parameter
            dtype: Data type for computations
        """
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_q_heads // n_kv_heads
        self.scale = 1.0 / (d_head ** 0.5)
        self.rope_base = float(rope_base)
        self.dtype = dtype

        # These will be set externally to share weights with training model
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

    def set_weights(self, Wq, Wk, Wv, Wo):
        """Set attention weights (shared with training model)."""
        # Convert weights to CuPy if using CuPy backend
        self.Wq = xp.asarray(Wq) if BACKEND_NAME == "cupy" else Wq
        self.Wk = xp.asarray(Wk) if BACKEND_NAME == "cupy" else Wk
        self.Wv = xp.asarray(Wv) if BACKEND_NAME == "cupy" else Wv
        self.Wo = xp.asarray(Wo) if BACKEND_NAME == "cupy" else Wo

    def _expand_kv(self, x):
        """Expand KV heads for GQA: [B,Hkv,T,D] -> [B,Hq,T,D]."""
        return xp.repeat(x, self.group_size, axis=1)

    def _apply_rope_absolute(self, x, positions):
        """
        Apply RoPE with absolute positions.
        
        Args:
            x: Input tensor [B, T_new, H, D]
            positions: Absolute positions [T_new] or scalar
            
        Returns:
            x_rope: RoPE-rotated tensor
        """
        B, T_new, H, D = x.shape
        
        if xp.isscalar(positions):
            positions = xp.array([positions], dtype=xp.float32)
        
        # Compute RoPE factors for these positions
        # i ranges over even indices: 0, 2, 4, ..., D-2
        i = xp.arange(0, D, 2, dtype="float32")  # [D/2]
        inv_freq = 1.0 / (self.rope_base ** (i / float(D)))  # [D/2]
        
        # theta[n, j] = position[n] * inv_freq[j]
        theta = positions[:, None] * inv_freq[None, :]  # [T_new, D/2]
        
        cos = xp.cos(theta).astype(x.dtype)  # [T_new, D/2]
        sin = xp.sin(theta).astype(x.dtype)  # [T_new, D/2]
        
        # Reshape for broadcasting
        cos = cos.reshape(B, T_new, 1, D // 2)  # [B, T_new, 1, D/2]
        sin = sin.reshape(B, T_new, 1, D // 2)  # [B, T_new, 1, D/2]
        
        # Split into even/odd components
        x0 = x[..., 0::2]  # [B, T_new, H, D/2]
        x1 = x[..., 1::2]  # [B, T_new, H, D/2]
        
        # RoPE: [x0*cos - x1*sin, x0*sin + x1*cos]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        
        # Interleave back
        y = xp.empty_like(x)
        y[..., 0::2] = y0
        y[..., 1::2] = y1
        
        return y

    def prefill(self, x, k_cache, v_cache, start_pos, return_all=False):
        """
        Prefill KV cache for prompt.
        
        Args:
            x: Input [B, T_prompt, D]
            k_cache: K cache [n_layers, B, H_kv, max_context, D_head]
            v_cache: V cache [n_layers, B, H_kv, max_context, D_head]
            start_pos: Starting position in cache (usually 0)
            return_all: If True, return all positions; if False, only last
            
        Returns:
            y: Output [B, T_prompt, D] if return_all else [B, D] (last position)
        """
        B, T_prompt, D = x.shape
        
        # Cache layout passed in: [B, H_kv, max_context, D_head]
        cache_capacity = k_cache.shape[2]
        if start_pos + T_prompt > cache_capacity:
            raise ValueError(
                f"Prompt length {T_prompt} at position {start_pos} "
                f"exceeds cache capacity {cache_capacity}."
            )
        
        # Ensure x is on the correct backend for matmul with weight matrices
        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)
        
        # Ensure weights are on the correct backend for matmul with x
        # For CuPy backend, self.Wq is a CuPy array (already converted in set_weights)
        # For NumPy backend, self.Wq is the raw data (array or Parameter.data)
        if BACKEND_NAME == "cupy":
            Wq_data = self.Wq
            Wk_data = self.Wk
            Wv_data = self.Wv
            Wo_data = self.Wo
        else:
            # NumPy backend: self.Wq might be a Parameter, use .data
            Wq_data = self.Wq.data if hasattr(self.Wq, 'data') else self.Wq
            Wk_data = self.Wk.data if hasattr(self.Wk, 'data') else self.Wk
            Wv_data = self.Wv.data if hasattr(self.Wv, 'data') else self.Wv
            Wo_data = self.Wo.data if hasattr(self.Wo, 'data') else self.Wo
        
        # Project to Q, K, V
        q_pre = (x @ Wq_data).reshape(B, T_prompt, self.n_q_heads, self.d_head)
        k_pre = (x @ Wk_data).reshape(B, T_prompt, self.n_kv_heads, self.d_head)
        v = (x @ Wv_data).reshape(B, T_prompt, self.n_kv_heads, self.d_head)
        
        # Apply RoPE with absolute positions
        q_positions = xp.arange(start_pos, start_pos + T_prompt, dtype=xp.float32)
        q = self._apply_rope_absolute(q_pre, q_positions)
        k = self._apply_rope_absolute(k_pre, q_positions)
        
        # Write K/V to cache
        # Cache layout passed in: [B, H_kv, max_context, D_head]
        # K/V from forward: [B, T, H_kv, D_head]
        # Need to transpose to match: [B, H_kv, T, D_head]
        end_pos = start_pos + T_prompt
        k_cache[:, :, start_pos:end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, start_pos:end_pos, :] = v.transpose(0, 2, 1, 3)
        
        # Expand KV for GQA attention
        # k_cache is [B, H_kv, max_context, D], take first end_pos positions
        k_all = self._expand_kv(k_cache[:, :, :end_pos, :])  # [B, H_q, end_pos, D]
        v_all = self._expand_kv(v_cache[:, :, :end_pos, :])  # [B, H_q, end_pos, D]
        
        # Reshape Q for attention
        q_attn = q.transpose(0, 2, 1, 3)  # [B, H_q, T_prompt, D]
        
        # Compute attention scores with causal mask
        scores = xp.einsum("bhtd,bhsd->bhts", q_attn, k_all) * self.scale  # [B, H_q, T_prompt, end_pos]
        
        # Causal mask: query can only attend to keys at or before its position
        q_positions_2d = q_positions[:, None]  # [T_prompt, 1]
        k_positions_2d = xp.arange(end_pos, dtype=xp.float32)[None, :]  # [1, end_pos]
        causal_mask = k_positions_2d <= q_positions_2d  # [T_prompt, end_pos] - True means allowed
        
        # Expand mask to broadcast with scores: [B, H_q, T_prompt, end_pos]
        causal_mask_expanded = causal_mask.astype(scores.dtype)[None, None, :, :]
        scores_masked = xp.where(causal_mask_expanded, scores, -xp.inf)
        
        # Softmax in FP32 for numerical stability
        scores_f32 = scores_masked.astype(xp.float32)
        scores_f32 = scores_f32 - xp.max(scores_f32, axis=-1, keepdims=True)
        probs = xp.exp(scores_f32)
        probs = probs / xp.sum(probs, axis=-1, keepdims=True)
        probs = probs.astype(self.dtype)
        
        # Compute context
        context = xp.einsum("bhts,bhsd->bhtd", probs, v_all)  # [B, H_q, T_prompt, D]
        
        # Merge heads and project output
        context_merged = context.transpose(0, 2, 1, 3).reshape(B, T_prompt, self.n_q_heads * self.d_head)
        y = context_merged @ Wo_data
        
        if return_all:
            return y
        else:
            return y[:, -1, :]

    def decode_one(self, x, k_cache, v_cache, start_pos):
        """
        Decode one token using cached K/V.
        
        Args:
            x: Input [B, 1, D] (single token)
            k_cache: K cache [n_layers, B, H_kv, max_context, D_head]
            v_cache: V cache [n_layers, B, H_kv, max_context, D_head]
            start_pos: Current position in cache
            
        Returns:
            logits: Output logits [B, vocab_size]
        """
        B, T_new, D = x.shape
        assert T_new == 1, "decode_one processes exactly one token"
        
        # Ensure x is on the correct backend for matmul with weight matrices
        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)
        
        # Ensure weights are on the correct backend for matmul with x
        # For CuPy backend, self.Wq is a CuPy array (already converted in set_weights)
        # For NumPy backend, self.Wq is the raw data (array or Parameter.data)
        if BACKEND_NAME == "cupy":
            Wq_data = self.Wq
            Wk_data = self.Wk
            Wv_data = self.Wv
            Wo_data = self.Wo
        else:
            # NumPy backend: self.Wq might be a Parameter, use .data
            Wq_data = self.Wq.data if hasattr(self.Wq, 'data') else self.Wq
            Wk_data = self.Wk.data if hasattr(self.Wk, 'data') else self.Wk
            Wv_data = self.Wv.data if hasattr(self.Wv, 'data') else self.Wv
            Wo_data = self.Wo.data if hasattr(self.Wo, 'data') else self.Wo
        
        # Project to Q, K, V
        q_pre = (x @ Wq_data).reshape(B, T_new, self.n_q_heads, self.d_head)
        k_pre = (x @ Wk_data).reshape(B, T_new, self.n_kv_heads, self.d_head)
        v = (x @ Wv_data).reshape(B, T_new, self.n_kv_heads, self.d_head)
        
        # Apply RoPE at absolute position
        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)
        
        # Write K/V to cache at current position
        # Cache layout passed in: [B, H_kv, max_context, D_head]
        # K/V from forward: [B, 1, H_kv, D_head]
        # Need to transpose to match: [B, H_kv, 1, D_head]
        k_cache[:, :, start_pos:start_pos + 1, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, start_pos:start_pos + 1, :] = v.transpose(0, 2, 1, 3)
        
        # All cached keys/values for attention
        # k_cache is [B, H_kv, max_context, D], take first start_pos+1 positions
        k_all = self._expand_kv(k_cache[:, :, :start_pos + 1, :])  # [B, H_q, start_pos+1, D]
        v_all = self._expand_kv(v_cache[:, :, :start_pos + 1, :])  # [B, H_q, start_pos+1, D]
        
        # Reshape Q for attention
        q_attn = q.transpose(0, 2, 1, 3)  # [B, H_q, 1, D]
        
        # Compute attention scores (no causal mask needed - single query)
        scores = xp.einsum("bhtd,bhsd->bhts", q_attn, k_all) * self.scale  # [B, H_q, 1, start_pos+1]
        
        # Softmax in FP32
        scores_f32 = scores.astype(xp.float32)
        scores_f32 = scores_f32 - xp.max(scores_f32, axis=-1, keepdims=True)
        probs = xp.exp(scores_f32)
        probs = probs / xp.sum(probs, axis=-1, keepdims=True)
        probs = probs.astype(self.dtype)
        
        # Compute context
        context = xp.einsum("bhts,bhsd->bhtd", probs, v_all)  # [B, H_q, 1, D]
        
        # Merge heads and project output
        context_merged = context.transpose(0, 2, 1, 3).reshape(B, 1, self.n_q_heads * self.d_head)
        y = context_merged @ Wo_data
        
        return y[:, -1, :]  # [B, D]
