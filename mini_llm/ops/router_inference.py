"""Router for inference - no backward caches needed."""


from mini_llm.backend import xp
from mini_llm.ops.topk import selected_topk_softmax_forward


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
        
        # Keep the projection on a true 2-D GEMM path. CuPy's generic N-D
        # matmul path does not support BF16 reliably.
        x2 = x.reshape(batch_size * seq_len, self.d_model)
        logits = (x2 @ self.W_router + self.b_router).reshape(
            batch_size, seq_len, self.n_experts
        )

        # Use the exact same top-k + selected-softmax primitive as training.
        # Besides avoiding duplicate logic, this guarantees identical FP32
        # promotion, deterministic tie handling, reduction order and cast-back
        # behavior for BF16/FP16 router outputs.
        output_weights, expert_indices, _ = selected_topk_softmax_forward(
            logits,
            self.k,
            output_dtype=x.dtype,
        )

        return output_weights, expert_indices
