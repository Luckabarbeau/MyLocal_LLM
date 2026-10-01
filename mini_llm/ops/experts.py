"""Expert FFN layers for Mixture of Experts.

Two implementations:
1. Dense (reference): Runs all selected experts on full batch
2. Sparse (optimized): True token-to-expert dispatch
"""

from ..backend import xp
from .routing_plan import RoutingPlan


def silu(x):
    """SiLU activation: x * sigmoid(x).
    
    Numerically stable implementation. Uses different formulations based on
    input value to avoid overflow/underflow:
    - For x >= 0: x / (1 + exp(-x))
    - For x < 0: x * exp(x) / (1 + exp(x))
    """
    # Numerically stable SiLU implementation
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    result = xp.zeros_like(x)
    
    # For x >= 0: x / (1 + exp(-x))
    pos_x = x[pos_mask]
    if pos_x.size > 0:
        result[pos_mask] = pos_x / (1.0 + xp.exp(-pos_x))
    
    # For x < 0: x * exp(x) / (1 + exp(x))
    neg_x = x[neg_mask]
    if neg_x.size > 0:
        exp_neg_x = xp.exp(neg_x)
        result[neg_mask] = neg_x * exp_neg_x / (1.0 + exp_neg_x)
    
    return result


def silu_prime(x):
    """Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = 1.0 / (1.0 + xp.exp(-x))
    return s + x * s * (1.0 - s)


class ExpertFFN:
    """
    Single expert FFN using SwiGLU structure.
    
    This is a standalone expert that can be composed into a Mixture of Experts.
    """

    def __init__(
        self, d_model, d_ff, input_std, output_std, rng, name="expert", dtype="float32"
    ):
        """
        Initialize the expert FFN.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension
            input_std: Standard deviation for weight initialization
            output_std: Standard deviation for down projection
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        from ..parameter import Parameter
        
        std = input_std
        self.W_gate = Parameter(
            xp.asarray(rng.normal((d_model, d_ff), std=std, dtype=dtype)),
            name=f"{name}.W_gate"
        )
        self.W_up = Parameter(
            xp.asarray(rng.normal((d_model, d_ff), std=std, dtype=dtype)),
            name=f"{name}.W_up"
        )
        self.W_down = Parameter(
            xp.asarray(rng.normal((d_ff, d_model), std=output_std, dtype=dtype)),
            name=f"{name}.W_down"
        )

    def parameters(self):
        """Return trainable parameters."""
        return [self.W_gate, self.W_up, self.W_down]

    def zero_grad(self):
        """Zero out gradients."""
        self.W_gate.zero_grad()
        self.W_up.zero_grad()
        self.W_down.zero_grad()

    @classmethod
    def reset_forward_count(cls):
        """Reset the forward call counter."""
        cls._forward_call_count = 0

    @classmethod
    def get_forward_count(cls):
        """Get the total number of forward calls since last reset."""
        return cls._forward_call_count

    _forward_call_count = 0  # Class-level counter for profiling

    def forward(self, x):
        """
        Forward pass through the expert.
        
        G = X W_gate
        U = X W_up
        A = SiLU(G)
        H = A * U
        Y = H W_down
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary for backward pass (contains minimal intermediates)
        """
        # Increment forward call counter
        ExpertFFN._forward_call_count += 1
        
        g = x @ self.W_gate.data
        u = x @ self.W_up.data
        a = silu(g)
        h = a * u
        y = h @ self.W_down.data
        
        # Cache only the minimum needed: x, g, u (h can be reconstructed as silu(g) * u)
        cache = {
            "x": x,
            "g": g,
            "u": u,
            "y": y,
        }
        
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through the expert.
        
        Args:
            dy: Gradient w.r.t. output, same shape as y
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input, same shape as x
        """
        x = cache["x"]
        g = cache["g"]
        u = cache["u"]
        
        dy_2d = dy.reshape(-1, dy.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        
        # Reconstruct h from cached g and u: h = silu(g) * u
        a = silu(g)
        h = a * u
        
        h_2d = h.reshape(-1, h.shape[-1])
        
        self.W_down.grad += h_2d.T @ dy_2d
        
        dh = dy_2d @ self.W_down.data.T
        da = dh * u.reshape(-1, u.shape[-1])
        du = dh * a.reshape(-1, a.shape[-1])
        dg = da * silu_prime(g.reshape(-1, g.shape[-1]))
        
        self.W_gate.grad += x_2d.T @ dg
        self.W_up.grad += x_2d.T @ du
        
        dx = dg @ self.W_gate.data.T + du @ self.W_up.data.T
        dx = dx.reshape(dy.shape)
        
        return dx


class Experts:
    """
    Mixture of Expert FFN layers with sparse expert evaluation.
    
    Only evaluates the unique experts that are selected, not all experts.
    Uses vectorized operations for each expert's full batch computation.
    """

    def __init__(
        self, d_model, d_ff, n_experts, input_std, output_std, rng, name="experts", dtype="float32"
    ):
        """
        Initialize the experts.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            input_std: Standard deviation for weight initialization
            output_std: Standard deviation for down projections
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        
        # Create n_experts ExpertFFN layers
        self.experts = []
        for i in range(n_experts):
            expert = ExpertFFN(
                d_model=d_model,
                d_ff=d_ff,
                input_std=input_std,
                output_std=output_std,
                rng=rng,
                name=f"{name}.expert_{i}",
                dtype=dtype
            )
            self.experts.append(expert)

    def parameters(self):
        """Return all trainable parameters from all experts."""
        params = []
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def zero_grad(self):
        """Zero out gradients for all experts."""
        for expert in self.experts:
            expert.zero_grad()

    def reset_forward_count(self):
        """Reset forward call counters for all experts."""
        for expert in self.experts:
            ExpertFFN.reset_forward_count()

    def get_forward_count(self):
        """Get the total number of forward calls from all experts."""
        # The counter is shared across all instances via class variable
        return ExpertFFN.get_forward_count()

    def forward(self, x, weights, expert_indices, routing_plan: RoutingPlan = None, n_experts: int = None):
        """
        Forward pass through experts with true sparse token-to-expert dispatch.
        
        Flattens tokens and dispatches each token to its selected experts.
        Each expert only processes tokens routed to it (sparse computation).
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            weights: Expert weights from router, shape (B, T, k)
            expert_indices: Indices of selected experts, shape (B, T, k)
            routing_plan: Pre-computed routing plan (optional, creates if not provided)
            n_experts: Total number of experts (uses self.n_experts if not provided)
            
        Returns:
            y: Weighted combination of expert outputs, shape (B, T, d_model)
            cache: Dictionary for backward pass
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len  # Total tokens
        
        # Flatten inputs: [B, T, D] -> [N, D]
        x_flat = x.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)
        
        # For each (token, slot) pair, we have: token_idx, expert_idx, weight
        # Total assignments = N * k
        
        # Prepare output: [N, D]
        y_flat = xp.zeros((N, d_model), dtype=x.dtype)
        
        # Cache for expert outputs and caches (to avoid recomputation in backward)
        expert_outputs = {}  # exp_idx -> {"outputs": [...], "token_indices": [...], "weights": [...]}
        expert_caches = {}   # exp_idx -> cache from expert.forward()
        
        # Determine n_experts
        if n_experts is None:
            n_experts = self.n_experts
        
        # If routing plan not provided, create it
        if routing_plan is None:
            routing_plan = RoutingPlan.from_router_outputs(expert_indices, weights, k, n_experts=self.n_experts)
        
        # Group assignments by expert using the routing plan
        for exp_idx in range(self.n_experts):
            # Get token, slot, and weight indices for this expert from plan
            token_indices, slot_indices, expert_weights = routing_plan.get_expert_assignments(exp_idx)
            
            if len(token_indices) > 0:
                # Gather tokens: [n_assigned, D]
                expert_x = x_flat[token_indices]
                
                # Forward through this expert
                expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)
                
                # Cache the outputs and cache for later use in backward
                expert_outputs[exp_idx] = {
                    "outputs": expert_out,
                    "token_indices": token_indices,
                    "weights": expert_weights,
                }
                expert_caches[exp_idx] = expert_cache
                
                # Weight and accumulate to output
                weighted_out = expert_weights[:, xp.newaxis] * expert_out
                xp.add.at(y_flat, token_indices, weighted_out)
        
        # Reshape: [N, D] -> [B, T, D]
        y = y_flat.reshape(batch_size, seq_len, d_model)
        
        # Store cache for backward - includes expert outputs to avoid recomputation
        cache = {
            "x": x,
            "weights": weights,
            "expert_indices": expert_indices,
            "y": y,
            "expert_outputs": expert_outputs,  # For router gradient computation
            "expert_caches": expert_caches,    # For expert backward pass
            "routing_plan": routing_plan,      # For backward pass
        }
        
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through experts with true sparse token-to-expert dispatch.
        
        For each expert, gathers gradients from tokens that were routed to it,
        computes gradients, and accumulates to input gradient.
        
        Args:
            dy: Gradient w.r.t. output, shape (B, T, d_model)
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input, shape (B, T, d_model)
        """
        x = cache["x"]
        weights = cache["weights"]
        expert_indices = cache["expert_indices"]
        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        k = weights.shape[-1]
        
        # Flatten inputs
        dy_flat = dy.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)
        
        # Initialize gradient w.r.t. input
        dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
        
        # Get cached expert outputs and caches from forward pass
        expert_outputs = cache.get("expert_outputs", {})
        expert_caches = cache.get("expert_caches", {})
        
        # If routing plan not available (from forward pass)
        routing_plan = cache.get("routing_plan")
        
        # Determine n_experts from cache or use self.n_experts
        expert_indices_cache = cache.get("expert_indices")
        if routing_plan is None and expert_indices_cache is not None:
            # Compute n_experts from expert_indices
            n_exp_from_indices = int(xp.max(expert_indices_cache).item()) + 1
        else:
            n_exp_from_indices = self.n_experts
        
        # For each expert, gather tokens and accumulate gradients
        for exp_idx in range(self.n_experts):
            # Get token indices from routing plan (or compute if not available)
            if routing_plan is not None:
                # Use routing plan - no mask reconstruction needed!
                token_indices, slot_indices, expert_weights = routing_plan.get_expert_assignments(exp_idx)
                # Check if this expert has any assignments
                if len(token_indices) == 0:
                    continue
            else:
                # Fallback: reconstruct by finding tokens where expert was selected
                expert_selected = (expert_indices_flat == exp_idx)  # [N, k]
                if xp.any(expert_selected):
                    flat_mask = expert_selected.flatten()  # [N*k]
                    all_token_indices = xp.repeat(xp.arange(N), k)  # [N*k]
                    token_indices = all_token_indices[flat_mask]  # [n_assigned]
                    all_weights = weights_flat.flatten()  # [N*k]
                    expert_weights = all_weights[flat_mask]  # [n_assigned]
                else:
                    token_indices = xp.empty(0, dtype=xp.int32)
                    expert_weights = xp.empty(0, dtype=weights.dtype)
            
            if len(token_indices) > 0:
                # Gather dy and weight: [n_assigned, D]
                expert_dy = dy_flat[token_indices] * expert_weights[:, xp.newaxis]
                
                # Use cached expert outputs and caches if available
                if exp_idx in expert_outputs and exp_idx in expert_caches:
                    # Reuse cached outputs - NO FORWARD CALL!
                    expert_out = expert_outputs[exp_idx]["outputs"]
                    expert_cache = expert_caches[exp_idx]
                else:
                    # Fallback: gather x and forward again (shouldn't happen with new cache)
                    x_flat = x.reshape(N, d_model)
                    expert_x = x_flat[token_indices]
                    expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)
                
                # Backward through this expert
                expert_dx = self.experts[exp_idx].backward(expert_dy, expert_cache)
                
                # Accumulate to input gradient
                xp.add.at(dx_flat, token_indices, expert_dx)
        
        # Reshape: [N, D] -> [B, T, D]
        dx = dx_flat.reshape(batch_size, seq_len, d_model)
        
        return dx
