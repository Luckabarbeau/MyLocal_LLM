"""Mixture of Experts (MoE) block for Transformer models."""

from ..backend import xp
from ..ops.router import Router
from ..ops.experts import Experts


class MoE:
    """
    Mixture of Experts (MoE) block replacing dense FFN.
    
    Structure:
        Input → Router → Top-k Experts → Weighted Combination → Output
        
    The MoE block takes the same interface as SwiGLU for easy replacement.
    """

    def __init__(
        self, d_model, d_ff, n_experts, k, input_std, output_std, rng,
        name="moe", dtype="float32"
    ):
        """
        Initialize the MoE block.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Total number of experts
            k: Number of top experts to select and use
            input_std: Standard deviation for weight initialization
            output_std: Standard deviation for down projections
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = min(k, n_experts)  # Ensure k <= n_experts
        
        # Router for expert selection
        self.router = Router(
            d_model=d_model,
            n_experts=n_experts,
            k=k,
            input_std=input_std,
            rng=rng,
            name=f"{name}.router",
            dtype=dtype
        )
        
        # Expert FFN layers
        self.experts = Experts(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            input_std=input_std,
            output_std=output_std,
            rng=rng,
            name=f"{name}.experts",
            dtype=dtype
        )

    def parameters(self):
        """Return all trainable parameters."""
        params = []
        # Add router parameters (Router uses Parameter objects internally)
        params.extend(self.router.parameters())
        # Add expert parameters
        params.extend(self.experts.parameters())
        return params

    def zero_grad(self):
        """Zero out all gradients."""
        self.router.zero_grad()
        self.experts.zero_grad()

    def forward(self, x):
        """
        Forward pass through the MoE block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary containing intermediate values for backward pass
        """
        # Router computes expert weights and selects top-k
        weights, expert_indices, router_cache = self.router.forward(x)
        
        # Experts compute weighted combination
        y, experts_cache = self.experts.forward(x, weights, expert_indices)
        
        cache = {
            "router_cache": router_cache,
            "experts_cache": experts_cache,
        }
        
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through the MoE block.
        
        Args:
            dy: Gradient of loss w.r.t. output y, shape (B, T, d_model)
            cache: Dictionary from forward pass containing intermediates
            
        Returns:
            dx: Gradient of loss w.r.t. input x, shape (B, T, d_model)
        """
        # Get cached values
        router_cache = cache["router_cache"]
        experts_cache = cache["experts_cache"]
        
        # Backward through experts - this computes dx and parameter gradients for experts
        dx = self.experts.backward(dy, experts_cache)
        
        # Backward through router to get router parameter gradients
        # The gradient for router is the weights (since y = sum_i weight[i] * expert_output[i])
        # So dL/dweights = expert_output weighted by dy
        weights = experts_cache["weights"]
        expert_indices = experts_cache["expert_indices"]
        all_outputs = experts_cache["all_outputs"]
        
        batch_size, seq_len, _ = dx.shape
        k = weights.shape[-1]
        
        # Compute gradient w.r.t. router output weights
        # dL/dweights[b,t,i] = dy[b,t] @ expert_output[selected_expert[b,t,i]][b,t]
        dweights = xp.zeros_like(weights)
        
        for i in range(k):
            expert_idx = expert_indices[..., i]
            for b in range(batch_size):
                for t in range(seq_len):
                    exp_idx = expert_idx[b, t]
                    # Gradient flows: dy * expert_output
                    dweights[b, t, i] = xp.dot(dy[b, t], all_outputs[exp_idx, b, t])
        
        # Backward through router
        dx_router = self.router.backward(dweights, router_cache)
        dx += dx_router  # Add router gradient to input gradient
        
        return dx
