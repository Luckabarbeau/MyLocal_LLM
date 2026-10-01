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

    def reset_forward_count(self):
        """Reset forward call counters for router and experts."""
        # Reset experts forward counts
        if hasattr(self.experts, 'reset_forward_count'):
            self.experts.reset_forward_count()

    def get_expert_forward_count(self):
        """Get the number of expert forward calls."""
        return self.experts.get_forward_count()

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
        # We now reuse expert outputs from the forward pass instead of recomputing them.
        
        x = experts_cache["x"]
        weights = experts_cache["weights"]
        expert_indices = experts_cache["expert_indices"]
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len
        
        # Flatten for easier indexing
        x_flat = x.reshape(N, d_model)
        dy_flat = dy.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)
        
        # Get cached expert outputs from forward pass
        expert_outputs = experts_cache.get("expert_outputs", {})
        
        # For each (token, slot) pair, get expert output and compute dweights
        # dweights[n,i] = dot(dy[n], expert_output[expert_idx[n,i], n])
        
        dweights_flat = xp.zeros((N, k))
        
        for exp_idx in range(self.n_experts):
            # Find tokens where this expert was selected
            expert_selected = (expert_indices_flat == exp_idx)  # [N, k]
            
            if xp.any(expert_selected):
                # Get token indices and slot indices
                flat_mask = expert_selected.flatten()  # [N*k]
                all_token_indices = xp.repeat(xp.arange(N), k)  # [N*k]
                all_slot_indices = xp.tile(xp.arange(k), N)  # [N*k]
                
                token_indices = all_token_indices[flat_mask]  # [n_assigned]
                slot_indices = all_slot_indices[flat_mask]  # [n_assigned]
                
                # Gather dy and weights
                expert_dy = dy_flat[token_indices]
                expert_weights = weights_flat.flatten()[flat_mask]  # [n_assigned]
                
                # Use cached expert outputs instead of recomputing
                if exp_idx in expert_outputs:
                    # Reuse cached outputs - NO FORWARD CALL!
                    cached_data = expert_outputs[exp_idx]
                    cached_token_indices = cached_data["token_indices"]
                    cached_outputs = cached_data["outputs"]
                    
                    # Create mapping from token index to output in cached array
                    # We need to match token_indices with cached_token_indices
                    dweights_for_tokens = xp.zeros(len(token_indices))
                    
                    for i, tok_idx in enumerate(token_indices):
                        # Find position of tok_idx in cached_token_indices
                        pos = xp.where(cached_token_indices == tok_idx)[0]
                        if len(pos) > 0:
                            dweights_for_tokens[i] = xp.sum(expert_dy[i] * cached_outputs[pos[0]])
                    
                    # Accumulate to dweights_flat
                    xp.add.at(dweights_flat, (token_indices, slot_indices), dweights_for_tokens)
                else:
                    # Fallback: forward through expert (shouldn't happen with new cache)
                    expert_x = x_flat[token_indices]
                    expert_out, _ = self.experts.experts[exp_idx].forward(expert_x)
                    
                    # dweights[n,i] = dot(dy[n], expert_output)
                    # Sum over d_model dimension
                    dweights_for_tokens = xp.sum(expert_dy * expert_out, axis=-1)  # [n_assigned]
                    
                    # Accumulate to dweights_flat
                    xp.add.at(dweights_flat, (token_indices, slot_indices), dweights_for_tokens)
        
        dweights = dweights_flat.reshape(batch_size, seq_len, k)
        
        # Backward through router
        dx_router = self.router.backward(dweights, router_cache)
        dx += dx_router  # Add router gradient to input gradient
        
        return dx
