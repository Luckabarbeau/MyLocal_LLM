"""Mixture of Experts (MoE) block for Transformer models."""

from ..backend import xp
from ..ops.router import Router
from ..ops.experts import Experts
from ..ops.routing_plan import RoutingPlan
from ..performance_profiler import moe_detail_scope


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

    def forward(self, x, return_cache=True):
        """
        Forward pass through the MoE block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary containing intermediate values for backward pass
        """
        # Router computes expert weights and selects top-k.  Fine-grained MoE
        # profiling uses CUDA events and therefore does not synchronize each
        # sub-operation when MINI_LLM_MOE_DETAIL_PROFILE=1.
        with moe_detail_scope("moe.router.forward"):
            weights, expert_indices, router_cache = self.router.forward(x)

        # Create routing plan once for all expert dispatch.
        k = weights.shape[-1]
        with moe_detail_scope("moe.plan.build"):
            routing_plan = RoutingPlan.from_router_outputs(
                expert_indices, weights, k, n_experts=self.n_experts
            )

        # Experts compute weighted combination.
        if return_cache:
            with moe_detail_scope("moe.experts.forward"):
                y, experts_cache = self.experts.forward(
                    x, weights, expert_indices, routing_plan
                )
            cache = {
                "router_cache": router_cache,
                "experts_cache": experts_cache,
            }
            return y, cache

        with moe_detail_scope("moe.experts.forward"):
            return self.experts.forward(
                x, weights, expert_indices, routing_plan, return_cache=False
            )

    def backward(self, dy, cache):
        """Backward pass through experts and router.

        Expert input/parameter gradients and the selected router-weight
        gradients are produced in a single expert dispatch pass.
        """
        router_cache = cache["router_cache"]
        experts_cache = cache["experts_cache"]

        # Reuse the routing plan, expert outputs, and activation caches from
        # forward.  This avoids both expert recomputation and a second expert
        # dispatch loop solely for dL/d(router weights).
        with moe_detail_scope("moe.experts.backward"):
            dx_experts, dweights = self.experts.backward(
                dy, experts_cache, return_dweights=True
            )

        with moe_detail_scope("moe.router.backward"):
            dx_router = self.router.backward(dweights, router_cache)
        return dx_experts + dx_router
