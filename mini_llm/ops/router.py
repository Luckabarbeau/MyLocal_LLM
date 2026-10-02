"""Router mechanism for Mixture of Experts.

Implements the selected-top-k softmax with proper gradient computation.
For a token y = sum_{i=1}^{k} w_i E_i(x), the gradient with respect to
selected router logits z_i is:

    dL/dz_i = w_i * (dL/dw_i - sum_j(w_j * dL/dw_j))

where w = softmax(z_selected) and top-k selection is treated with a
straight-through estimator (fixed selection during gradient computation).
"""

from ..backend import xp, resolve_dtype, is_low_precision_dtype
from .topk import selected_topk_softmax_forward, selected_topk_softmax_backward


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
        
        # Top-K selection and selected softmax are shared with the context
        # router.  The helper preserves the original deterministic argsort and
        # FP32 routing math for FP16/BF16 inputs.
        output_weights, expert_indices, topk_cache = selected_topk_softmax_forward(
            logits, self.k, output_dtype=x.dtype
        )
        output_weights_f32 = topk_cache["weights_work"]
        selected_logits = xp.take_along_axis(
            logits.astype("float32", copy=False)
            if is_low_precision_dtype(logits.dtype) else logits,
            expert_indices,
            axis=-1,
        )
        flat_indices = expert_indices.reshape(-1, self.k)
        batch_idx = xp.arange(batch_size * seq_len)[:, None]

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
            "topk_cache": topk_cache,
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
        
        # Reuse the same selected-softmax Jacobian/scatter primitive used by
        # the context router.  Top-K identities remain fixed locally.
        dlogits = selected_topk_softmax_backward(dweights, cache["topk_cache"])
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
