"""Router for inference - no backward caches needed."""


from mini_llm.backend import xp, is_low_precision_dtype


class RouterInference:
    """
    Router for MoE during inference.
    
    Same computation as training but without caching for backprop.
    """

    def __init__(self, d_model, n_experts, k, dtype="float32"):
        """
        Initialize router for inference.
        
        Args:
            d_model: Model dimension
            n_experts: Number of experts
            k: Number of top experts to select
            dtype: Data type
        """
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = min(k, n_experts)
        
        # Will be set externally
        self.W_router = None
        self.b_router = None

    def set_weights(self, router):
        """Set weights from training Router."""
        self.W_router = router.W_router_param.data
        self.b_router = router.b_router_param.data

    def forward(self, x):
        """
        Forward pass through router.
        
        Args:
            x: Input [B, T, D]
            
        Returns:
            weights: Output weights [B, T, k]
            expert_indices: Expert indices [B, T, k]
        """
        batch_size, seq_len, _ = x.shape
        
        # Keep the projection on a true 2-D GEMM path. CuPy's generic N-D
        # matmul path does not support BF16 reliably.
        x2 = x.reshape(batch_size * seq_len, self.d_model)
        logits = (x2 @ self.W_router + self.b_router).reshape(
            batch_size, seq_len, self.n_experts
        )

        # CuPy/Thrust cannot argsort BF16. Router logits are tiny
        # (n_experts values/token), so promote only this selection/reduction
        # work to FP32.
        logits_work = (
            logits.astype(xp.float32, copy=False)
            if is_low_precision_dtype(logits.dtype) else logits
        )

        expert_indices = xp.argsort(
            -logits_work, axis=-1
        )[..., :self.k]

        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]

        logits_flat = logits_work.reshape(-1, self.n_experts)
        selected_logits = logits_flat[batch_idx, flat_indices].reshape(
            batch_size, seq_len, self.k
        )

        selected_logits_max = xp.max(selected_logits, axis=-1, keepdims=True)
        exp_selected = xp.exp(selected_logits - selected_logits_max)
        selected_sums = xp.sum(exp_selected, axis=-1, keepdims=True)
        output_weights_f32 = exp_selected / selected_sums
        output_weights = (
            output_weights_f32.astype(x.dtype, copy=False)
            if is_low_precision_dtype(x.dtype) else output_weights_f32
        )

        return output_weights, expert_indices
