"""Router mechanism for Mixture of Experts.

Implements the selected-top-k softmax with proper gradient computation.
For a token y = sum_{i=1}^{k} w_i E_i(x), the gradient with respect to
selected router logits z_i is:

    dL/dz_i = w_i * (dL/dw_i - sum_j(w_j * dL/dw_j))

where w = softmax(z_selected) and top-k selection is treated with a
straight-through estimator (fixed selection during gradient computation).
"""

from ..backend import xp, resolve_dtype, is_low_precision_dtype


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
        self.b_router = xp.zeros((n_experts,), dtype=resolve_dtype(dtype))
        
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
        
        Uses selected-top-k softmax:
        1. Compute full logits for all experts
        2. Select top-k indices (fixed during backward)
        3. Apply softmax only to selected logits
        4. Return selected weights that sum to 1 across k experts
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            output_weights: Weights for top-k experts, shape (B, T, k)
            expert_indices: Indices of top-k experts, shape (B, T, k)
            cache: Dictionary for backward pass
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten leading dimensions so BF16 uses CuPy's supported 2-D GEMM
        # path rather than its generic N-D batched tensordot implementation.
        x_2d = x.reshape(-1, self.d_model)
        logits = (x_2d @ self.W_router_param.data).reshape(
            batch_size, seq_len, self.n_experts
        ) + self.b_router_param.data
        
        # CuPy/Thrust does not currently implement argsort for BF16 arrays.
        # Router logits are tiny compared with the expert activations
        # (n_experts is typically single-digit), so promote them to FP32 for
        # expert selection and router softmax. The projection GEMM above and
        # all expert GEMMs remain in BF16.
        logits_work = (
            logits.astype("float32", copy=False)
            if is_low_precision_dtype(logits.dtype) else logits
        )

        # Select top-k experts using argsort for deterministic selection.
        expert_indices = xp.argsort(-logits_work, axis=-1)[..., :self.k]

        # Gather selected logits for softmax computation
        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]

        logits_flat = logits_work.reshape(-1, self.n_experts)
        selected_logits = logits_flat[batch_idx, flat_indices].reshape(batch_size, seq_len, self.k)
        
        # Router softmax/reductions run in FP32 for FP16/BF16 inputs.  Cast the
        # selected weights back to the branch dtype for sparse expert GEMMs and
        # weighted accumulation.
        selected_work = (
            selected_logits.astype("float32", copy=False)
            if is_low_precision_dtype(selected_logits.dtype) else selected_logits
        )
        selected_logits_max = xp.max(selected_work, axis=-1, keepdims=True)
        exp_selected = xp.exp(selected_work - selected_logits_max)
        selected_sums = xp.sum(exp_selected, axis=-1, keepdims=True)
        output_weights_f32 = exp_selected / selected_sums
        output_weights = (
            output_weights_f32.astype(x.dtype, copy=False)
            if is_low_precision_dtype(x.dtype) else output_weights_f32
        )
        
        cache = {
            "x": x,
            "logits": logits,
            "expert_indices": expert_indices,
            "output_weights": output_weights,
            "output_weights_f32": output_weights_f32,
            # Store selected logits for backward pass (needed for gradient computation)
            "selected_logits": selected_logits,
            "selected_expert_indices": flat_indices,
            "batch_idx": batch_idx,
        }
        
        return output_weights, expert_indices, cache

    def backward(self, dweights, cache):
        """
        Backward pass through the router.
        
        Uses selected-top-k softmax with proper Jacobian computation.
        Top-k selection is treated with straight-through estimator (fixed).
        
        The output weights are: w = softmax(z_selected)
        where z_selected are the logits for the top-k experts.
        
        Gradient for selected logits:
            dL/dz_i = w_i * (dL/dw_i - sum_j(w_j * dL/dw_j))
        
        Args:
            dweights: Gradient w.r.t. output weights, shape (B, T, k)
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input x, shape (B, T, d_model)
        """
        x = cache["x"]
        expert_indices = cache["expert_indices"]
        output_weights = cache["output_weights"]
        output_weights_f32 = cache.get("output_weights_f32", output_weights)
        selected_logits = cache["selected_logits"]
        batch_idx = cache["batch_idx"]
        
        batch_size, seq_len, _ = x.shape
        N = batch_size * seq_len
        
        # dL/dw = dweights (gradient w.r.t. output weights)
        # For softmax: dw/dz = diag(w) - w * w^T
        # So: dL/dz = w * (dL/dw - sum_j(w_j * dL/dw_j))
        
        # Compute the correction term: sum_j(w_j * dL/dw_j)
        dweights_work = dweights.astype("float32", copy=False)
        weights_work = output_weights_f32.astype("float32", copy=False)
        w_dweights_sum = xp.sum(weights_work * dweights_work, axis=-1, keepdims=True)

        # Gradient w.r.t. selected logits in FP32, then cast only for GEMMs.
        dselected_logits = weights_work * (dweights_work - w_dweights_sum)
        
        # Scatter selected gradients back to full logits
        dlogits_flat = xp.zeros((N, self.n_experts), dtype=dselected_logits.dtype)
        dselected_logits_flat = dselected_logits.reshape(-1, self.k)
        dlogits_flat[batch_idx, expert_indices.reshape(-1, self.k)] = dselected_logits_flat
        dlogits = dlogits_flat.reshape(batch_size, seq_len, self.n_experts)
        dlogits_compute = (
            dlogits.astype(x.dtype, copy=False)
            if is_low_precision_dtype(x.dtype) else dlogits
        )

        # Gradient through logits = x @ W + b. Keep the GEMM strictly 2-D for
        # CuPy BF16 compatibility, then restore the original token layout.
        dlogits_2d = dlogits_compute.reshape(-1, self.n_experts)
        dx = (dlogits_2d @ self.W_router_param.data.T).reshape(x.shape)

        # Accumulate parameter gradients. GEMM stays in branch compute dtype;
        # Parameter.grad itself is FP32 for low-precision parameters.
        x_2d = x.reshape(-1, self.d_model)
        self.W_router_param.grad += x_2d.T @ dlogits_2d
        self.b_router_param.grad += xp.sum(dlogits, axis=(0, 1))
        
        return dx
