"""Expert FFN layers for Mixture of Experts - Sparse evaluation with vectorization."""

from ..backend import xp


def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    return x / (1.0 + xp.exp(-x))


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
            cache: Dictionary for backward pass
        """
        g = x @ self.W_gate.data
        u = x @ self.W_up.data
        a = silu(g)
        h = a * u
        y = h @ self.W_down.data
        
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
        x, g, u, a, h = (
            cache["x"], cache["g"], cache["u"], cache["a"], cache["h"]
        )
        
        dy_2d = dy.reshape(-1, dy.shape[-1])
        h_2d = h.reshape(-1, h.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        
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

    def forward(self, x, weights, expert_indices):
        """
        Forward pass through experts with sparse evaluation.
        
        Only evaluates unique experts that are selected, not all experts.
        Each selected expert processes the full batch once.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            weights: Expert weights from router, shape (B, T, k)
            expert_indices: Indices of selected experts, shape (B, T, k)
            
        Returns:
            y: Weighted combination of expert outputs, shape (B, T, d_model)
            cache: Dictionary for backward pass
        """
        batch_size, seq_len, _ = x.shape
        k = weights.shape[-1]
        
        # Find all unique experts that need to be evaluated
        unique_experts = xp.unique(expert_indices)
        
        # Evaluate only the unique selected experts on the full batch
        computed_outputs = {}  # exp_idx -> (output, cache)
        for exp_idx in unique_experts:
            out, cache = self.experts[exp_idx].forward(x)
            computed_outputs[exp_idx] = (out, cache)
        
        # Create output by selecting and weighting
        y = xp.zeros_like(x)
        
        batch_idx, seq_idx = xp.meshgrid(
            xp.arange(batch_size), xp.arange(seq_len), indexing='ij'
        )
        flat_batch = batch_idx.flatten()
        flat_seq = seq_idx.flatten()
        
        # Pre-compute all outputs stack (only for selected experts)
        all_outputs_list = []
        for exp_idx in range(self.n_experts):
            if exp_idx in computed_outputs:
                all_outputs_list.append(computed_outputs[exp_idx][0])
            else:
                all_outputs_list.append(xp.zeros_like(x))
        
        all_outputs_stack = xp.stack(all_outputs_list, axis=0)
        
        for i in range(k):
            expert_idx = expert_indices[..., i]  # (B, T)
            flat_expert_idx = expert_idx.flatten()
            
            # Get expert output using advanced indexing
            expert_out = all_outputs_stack[flat_expert_idx, flat_batch, flat_seq]
            expert_out = expert_out.reshape(batch_size, seq_len, -1)
            
            # Weight it
            y += weights[..., i:i+1] * expert_out
        
        # Store caches for backward pass
        all_caches = [computed_outputs[exp_idx][1] if exp_idx in computed_outputs else None 
                      for exp_idx in range(self.n_experts)]
        
        cache = {
            "x": x,
            "weights": weights,
            "expert_indices": expert_indices,
            "computed_outputs": computed_outputs,
            "all_caches": all_caches,
            "y": y,
        }
        
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through experts with sparse evaluation.
        
        For each expert that was selected, accumulates gradients from all
        positions where it was selected, then calls backward once.
        
        Args:
            dy: Gradient w.r.t. output, shape (B, T, d_model)
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input, shape (B, T, d_model)
        """
        x = cache["x"]
        weights = cache["weights"]
        expert_indices = cache["expert_indices"]
        all_caches = cache["all_caches"]
        
        batch_size, seq_len, _ = x.shape
        k = weights.shape[-1]
        
        # Initialize gradient w.r.t. input
        dx = xp.zeros_like(x)
        
        # For each expert, accumulate gradients from positions where it was selected
        for exp_idx in range(self.n_experts):
            # Find all positions where this expert was selected
            expert_selected = (expert_indices == exp_idx)  # (B, T, k)
            
            if xp.any(expert_selected):
                # Accumulate weighted dy for this expert
                # Shape: (B, T, d_model)
                expert_dy = xp.zeros_like(x)
                
                for i in range(k):
                    # Positions where this expert is selected at position i
                    selected_at_i = expert_selected[..., i]
                    expert_dy += selected_at_i[..., xp.newaxis] * (
                        weights[..., i:i+1] * dy
                    )
                
                # Backward through this expert with accumulated gradient
                expert_cache = all_caches[exp_idx]
                expert_dx = self.experts[exp_idx].backward(expert_dy, expert_cache)
                
                # Accumulate to input gradient
                dx += expert_dx
        
        return dx
