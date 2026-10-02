"""Expert FFN layers for Mixture of Experts.

Two implementations:
1. Dense (reference): Runs all selected experts on full batch
2. Sparse (optimized): True token-to-expert dispatch
"""

from ..backend import xp
from .routing_plan import RoutingPlan


from .silu import silu, silu_prime


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

    def forward(self, x, return_cache=True):
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
        
        if not return_cache:
            return y

        # Speed-oriented training cache.  Keeping a and h avoids recomputing
        # SiLU(g) and a*u during backward.  The routed expert output y is
        # cached once by Experts.forward() for the router-weight gradient.
        cache = {
            "x": x,
            "g": g,
            "u": u,
            "a": a,
            "h": h,
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
        a = cache["a"]
        h = cache["h"]
        
        dy_2d = dy.reshape(-1, dy.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
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

    def forward(self, x, weights, expert_indices, routing_plan: RoutingPlan = None, n_experts: int = None, return_cache=True):
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
                
                # Forward through this expert.  Reference inference does not
                # retain activations or routed outputs needed only by backward.
                if return_cache:
                    expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)
                    expert_outputs[exp_idx] = {
                        "outputs": expert_out,
                        "token_indices": token_indices,
                        "weights": expert_weights,
                    }
                    expert_caches[exp_idx] = expert_cache
                else:
                    expert_out = self.experts[exp_idx].forward(
                        expert_x, return_cache=False
                    )
                
                # Weight and accumulate to output.  token_indices are unique
                # within a single expert because top-k cannot select the same
                # expert twice for one token, so atomics are unnecessary.
                weighted_out = expert_weights[:, xp.newaxis] * expert_out
                y_flat[token_indices] += weighted_out
        
        # Reshape: [N, D] -> [B, T, D]
        y = y_flat.reshape(batch_size, seq_len, d_model)
        
        if not return_cache:
            return y

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

    def backward(self, dy, cache, return_dweights=False):
        """
        Backward pass through sparsely-routed experts.

        The routing plan and expert outputs cached in forward are reused here.
        When ``return_dweights`` is true, the router-weight gradient is
        computed in this SAME expert loop, avoiding the second dispatch pass
        that previously lived in ``MoE.backward``.

        Args:
            dy: Gradient w.r.t. MoE output, shape (B, T, d_model).
            cache: Dictionary returned by ``Experts.forward``.
            return_dweights: Also return dL/d(selected routing weights).

        Returns:
            dx, or ``(dx, dweights)`` when ``return_dweights=True``.
        """
        x = cache["x"]
        weights = cache["weights"]
        expert_indices = cache["expert_indices"]
        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        k = weights.shape[-1]

        dy_flat = dy.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)

        dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
        dweights_flat = (
            xp.zeros((N, k), dtype=dy.dtype) if return_dweights else None
        )

        expert_outputs = cache.get("expert_outputs", {})
        expert_caches = cache.get("expert_caches", {})
        routing_plan = cache.get("routing_plan")

        # Fallback metadata is built lazily only for legacy caches.  The normal
        # optimized path always receives a RoutingPlan from forward.
        fallback_token_ids = None
        fallback_slot_ids = None
        fallback_weights = None

        for exp_idx in range(self.n_experts):
            if routing_plan is not None:
                token_indices, slot_indices, expert_weights = (
                    routing_plan.get_expert_assignments(exp_idx)
                )
                if routing_plan.get_assignment_count(exp_idx) == 0:
                    continue
            else:
                expert_selected = (expert_indices_flat == exp_idx)
                if fallback_token_ids is None:
                    fallback_token_ids = xp.repeat(xp.arange(N), k)
                    fallback_slot_ids = xp.tile(xp.arange(k), N)
                    fallback_weights = weights_flat.reshape(-1)

                flat_mask = expert_selected.reshape(-1)
                token_indices = fallback_token_ids[flat_mask]
                slot_indices = fallback_slot_ids[flat_mask]
                expert_weights = fallback_weights[flat_mask]
                if len(token_indices) == 0:
                    continue

            # Raw dy is needed for dL/d(router weight).  The expert parameter
            # gradient receives dy multiplied by the selected routing weight.
            raw_expert_dy = dy_flat[token_indices]

            if exp_idx in expert_outputs and exp_idx in expert_caches:
                expert_out = expert_outputs[exp_idx]["outputs"]
                expert_cache = expert_caches[exp_idx]
            else:
                # Legacy-cache fallback only.  New forward passes always cache
                # both expert output and activation cache.
                expert_x = x.reshape(N, d_model)[token_indices]
                expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)

            if return_dweights:
                # For y = sum_s w_s E_s(x):
                #   dL/dw_s = dot(dL/dy, E_s(x)).
                # The cached output order is exactly the routing-plan order for
                # this expert, so no token search / xp.where is necessary.
                dweight_values = xp.sum(
                    raw_expert_dy * expert_out, axis=-1
                )
                dweights_flat[token_indices, slot_indices] = dweight_values

            expert_dy = raw_expert_dy * expert_weights[:, xp.newaxis]
            expert_dx = self.experts[exp_idx].backward(
                expert_dy, expert_cache
            )

            # top-k selection contains each expert at most once per token, so
            # token_indices are unique within this expert.  The expert loop is
            # serialized on the same stream; atomics are unnecessary here.
            dx_flat[token_indices] += expert_dx

        dx = dx_flat.reshape(batch_size, seq_len, d_model)

        if return_dweights:
            return dx, dweights_flat.reshape(batch_size, seq_len, k)
        return dx
