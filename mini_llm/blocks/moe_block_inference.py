"""Inference-only MoE block with KV cache support."""


from mini_llm.backend import xp
from mini_llm.ops.router_inference import RouterInference
from mini_llm.ops.experts_inference import ExpertsInference


class MoEInference:
    """
    Mixture of Experts for inference - no backward caches.
    
    Runs the same computation as training MoE but without
    storing activations needed for backprop.
    """

    def __init__(
        self, d_model, d_ff, n_experts, k, dtype="float32"
    ):
        """
        Initialize MoE for inference.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            k: Number of top experts to use
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = min(k, n_experts)
        
        # Router and experts (weights shared with training)
        self.router = RouterInference(
            d_model=d_model,
            n_experts=n_experts,
            k=k,
            dtype=dtype
        )
        self.experts = ExpertsInference(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            dtype=dtype
        )

    def set_weights(self, moe):
        """
        Set weights from training MoE.
        
        Args:
            moe: Training MoE instance
        """
        self.router.set_weights(moe.router)
        self.experts.set_weights(moe.experts)

    def prefill(self, x):
        """
        Forward through MoE for prefill.
        
        Args:
            x: Input [B, T_prompt, D]
            
        Returns:
            y: Output [B, T_prompt, D]
        """
        weights, expert_indices = self.router.forward(x)
        y = self.experts.forward(x, weights, expert_indices)
        return y

    def decode_one(self, x):
        """
        Forward through MoE for single token.
        
        Args:
            x: Input [B, 1, D]
            
        Returns:
            y: Output [B, 1, D]
        """
        weights, expert_indices = self.router.forward(x)
        y = self.experts.forward(x, weights, expert_indices)
        return y
