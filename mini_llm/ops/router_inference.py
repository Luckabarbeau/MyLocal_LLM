"""Router for inference - no backward caches needed."""


from mini_llm.backend import xp


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
        
        # Compute logits
        logits = x @ self.W_router + self.b_router  # [B, T, n_experts]
        
        # Select top-k experts
        expert_indices = xp.argsort(-logits, axis=-1)[..., :self.k]  # [B, T, k]
        
        # Gather selected logits
        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]
        
        logits_flat = logits.reshape(-1, self.n_experts)
        selected_logits = logits_flat[batch_idx, flat_indices].reshape(batch_size, seq_len, self.k)
        
        # Apply softmax
        selected_logits_max = xp.max(selected_logits, axis=-1, keepdims=True)
        exp_selected = xp.exp(selected_logits - selected_logits_max)
        selected_sums = xp.sum(exp_selected, axis=-1, keepdims=True)
        output_weights = exp_selected / selected_sums
        
        return output_weights, expert_indices
