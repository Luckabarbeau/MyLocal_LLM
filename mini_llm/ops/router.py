"""Router mechanism for Mixture of Experts."""

from ..backend import xp


class Router:
    """
    Router for Mixture of Experts.
    
    Computes expert weights from input and selects top-k experts.
    Uses straight-through estimator for gradient through top-k selection.
    """

    def __init__(
        self, d_model, n_experts, k, input_std, rng, name="router", dtype="float32"
    ):
        """
        Initialize the router.
        
        Args:
            d_model: Model dimension
            n_experts: Number of experts
            k: Number of top experts to select
            input_std: Standard deviation for weight initialization
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = min(k, n_experts)  # Ensure k <= n_experts
        
        # Router weights: map d_model -> n_experts
        std = input_std
        self.W_router = xp.asarray(
            rng.normal((d_model, n_experts), std=std, dtype=dtype)
        )
        self.b_router = xp.zeros((n_experts,), dtype=dtype)
        
        # Gradient accumulators (as Parameter objects for consistency)
        from ..parameter import Parameter
        self.W_router_param = Parameter(self.W_router.copy(), name=f"{name}.W_router")
        self.b_router_param = Parameter(self.b_router.copy(), name=f"{name}.b_router")

    def parameters(self):
        """Return trainable parameters."""
        return [self.W_router_param, self.b_router_param]

    def zero_grad(self):
        """Zero out gradients."""
        self.W_router_param.zero_grad()
        self.b_router_param.zero_grad()

    def forward(self, x):
        """
        Forward pass through the router.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            output_weights: Weights for top-k experts, shape (B, T, k)
            expert_indices: Indices of top-k experts, shape (B, T, k)
            cache: Dictionary for backward pass
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute logits: (B, T, d_model) @ (d_model, n_experts) -> (B, T, n_experts)
        logits = x @ self.W_router_param.data + self.b_router_param.data
        
        # Compute softmax probabilities with numerical stability
        logits_max = xp.max(logits, axis=-1, keepdims=True)
        exp_logits = xp.exp(logits - logits_max)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)
        
        # Select top-k experts using argsort for deterministic selection
        expert_indices = xp.argsort(-probs, axis=-1)[..., :self.k]  # (B, T, k)
        
        # Get weights for top-k experts
        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]
        
        probs_flat = probs.reshape(-1, self.n_experts)
        output_weights = probs_flat[batch_idx, flat_indices].reshape(batch_size, seq_len, self.k)
        
        cache = {
            "x": x,
            "logits": logits,
            "probs": probs,
            "expert_indices": expert_indices,
            "output_weights": output_weights,
        }
        
        return output_weights, expert_indices, cache

    def backward(self, dweights, cache):
        """
        Backward pass through the router.
        
        Uses straight-through estimator for top-k selection.
        
        Args:
            dweights: Gradient w.r.t. output weights, shape (B, T, k)
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input x, shape (B, T, d_model)
        """
        x = cache["x"]
        probs = cache["probs"]
        expert_indices = cache["expert_indices"]
        
        batch_size, seq_len, _ = x.shape
        
        # Create gradient w.r.t. probabilities (straight-through for top-k)
        dprobs_flat = xp.zeros((batch_size * seq_len, self.n_experts), dtype=xp.float64)
        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]
        
        # Gradient flows only through selected experts
        dprobs_flat[batch_idx, flat_indices] = dweights.reshape(-1, self.k)
        dprobs = dprobs_flat.reshape(batch_size, seq_len, self.n_experts)
        
        # Gradient through softmax: dL/dlogits = probs * (dL/dprobs - sum(probs * dL/dprobs))
        dprobs_sum = xp.sum(dprobs * probs, axis=-1, keepdims=True)
        dlogits = probs * (dprobs - dprobs_sum)
        
        # Gradient through logits = x @ W + b
        dx = dlogits @ self.W_router_param.data.T
        
        # Accumulate parameter gradients
        x_2d = x.reshape(-1, self.d_model)
        dlogits_2d = dlogits.reshape(-1, self.n_experts)
        self.W_router_param.grad += x_2d.T @ dlogits_2d
        self.b_router_param.grad += xp.sum(dlogits, axis=(0, 1))
        
        return dx
