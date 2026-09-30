"""Experts for inference - no backward caches needed."""


from mini_llm.backend import xp
from mini_llm.ops.silu import silu


class ExpertFFNInference:
    """
    Single expert FFN for inference.
    
    Same computation as training but without caching intermediates for backprop.
    """

    def __init__(self, d_model, d_ff, dtype="float32"):
        """
        Initialize expert FFN for inference.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        
        # Will be set externally
        self.W_gate = None
        self.W_up = None
        self.W_down = None

    def set_weights(self, expert):
        """Set weights from training ExpertFFN."""
        self.W_gate = expert.W_gate.data
        self.W_up = expert.W_up.data
        self.W_down = expert.W_down.data

    def forward(self, x):
        """
        Forward pass through expert.
        
        Args:
            x: Input [B, T, D] or [N, D]
            
        Returns:
            y: Output [B, T, D] or [N, D]
        """
        g = x @ self.W_gate  # [..., d_ff]
        u = x @ self.W_up    # [..., d_ff]
        a = silu(g)
        h = a * u
        y = h @ self.W_down  # [..., d_model]
        
        return y


class ExpertsInference:
    """
    Mixture of Expert FFN layers for inference with sparse evaluation.
    
    Only evaluates the selected experts, not all experts.
    """

    def __init__(self, d_model, d_ff, n_experts, dtype="float32"):
        """
        Initialize experts for inference.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        
        # Create inference experts
        self.experts = []
        for i in range(n_experts):
            expert = ExpertFFNInference(
                d_model=d_model,
                d_ff=d_ff,
                dtype=dtype
            )
            self.experts.append(expert)

    def set_weights(self, experts):
        """Set weights from training Experts."""
        for i in range(self.n_experts):
            self.experts[i].set_weights(experts.experts[i])

    def forward(self, x, weights, expert_indices):
        """
        Forward pass through experts with sparse token-to-expert dispatch.
        
        Args:
            x: Input [B, T, D]
            weights: Expert weights [B, T, k]
            expert_indices: Expert indices [B, T, k]
            
        Returns:
            y: Output [B, T, D]
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len
        
        # Flatten inputs
        x_flat = x.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)
        
        # Prepare output
        y_flat = xp.zeros((N, d_model), dtype=x.dtype)
        
        # Group by expert
        for exp_idx in range(self.n_experts):
            expert_selected = (expert_indices_flat == exp_idx)
            
            if xp.any(expert_selected):
                flat_mask = expert_selected.flatten()
                all_token_indices = xp.repeat(xp.arange(N), k)
                token_indices = all_token_indices[flat_mask]
                
                all_weights = weights_flat.flatten()
                expert_weights = all_weights[flat_mask]
                
                expert_x = x_flat[token_indices]
                expert_out = self.experts[exp_idx].forward(expert_x)
                
                weighted_out = expert_weights[:, xp.newaxis] * expert_out
                xp.add.at(y_flat, token_indices, weighted_out)
        
        y = y_flat.reshape(batch_size, seq_len, d_model)
        
        return y
