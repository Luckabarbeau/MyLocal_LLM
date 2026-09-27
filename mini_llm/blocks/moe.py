from ..backend import xp
from ..init import matrix_parameter
from ..ops import RMSNorm, SwiGLU
class MoEBlock:
    """
    Mixture of Experts block for sparse feed-forward replacement.
    """

    def __init__(
        self, d_model, n_experts, top_k, d_ff,
        input_std, output_std, router_std, rng,
        name="moe", dtype="float32"
    ):
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.router_std = float(router_std)

        # Router (shared across all tokens)
        self.router = matrix_parameter(
            (d_model, n_experts), router_std, rng,
            f"{name}.router", dtype=dtype, decay=False
        )

        # Experts (experts are grouped)
        self.experts = []
        for i in range(n_experts):
            expert = SwiGLU(
                d_model, d_ff, input_std, output_std, rng,
                name=f"{name}.expert{i}", dtype=dtype
            )
            self.experts.append(expert)

    def parameters(self):
        params = [self.router]
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def forward(self, x, return_cache=True):
        if x.ndim != 3:
            raise ValueError("input must have shape [B,T,D].")
        b, t, d = x.shape
        if d != self.d_model:
            raise ValueError(f"input final dimension ({d}) != d_model ({self.d_model}).")

        # Compute router logits
        router_logits = x @ self.router.data  # [B,T,n_experts]

        # Apply softmax to get router probabilities
        router_probs = xp.exp(router_logits - xp.max(router_logits, axis=-1, keepdims=True))
        router_probs = router_probs / xp.sum(router_probs, axis=-1, keepdims=True)  # [B,T,n_experts]

        # Get top-k experts for each token
        top_k_probs, top_k_indices = xp.topk(router_probs, self.top_k, axis=-1)  # [B,T,top_k], [B,T,top_k]

        # Compute expert outputs
        expert_outputs = []
        expert_caches = []
        for i in range(self.n_experts):
            expert = self.experts[i]
            # For each expert, compute output for all tokens where this expert is selected
            expert_out, expert_cache = expert.forward(x)
            expert_outputs.append(expert_out)
            expert_caches.append(expert_cache)

        # Combine expert outputs based on top-k selection
        # This is a simplified implementation - in a real MoE you'd need scatter/gather
        y = xp.sum(expert_outputs[0], axis=0)  # Simplified for now

        if return_cache:
            return y, {
                "router_logits": router_logits,
                "router_probs": router_probs,
                "top_k_indices": top_k_indices,
                "expert_caches": expert_caches
            }
        return y, {"x": x}

    def backward(self, dy, cache):
        # Extract caches
        router_logits = cache["router_logits"]
        router_probs = cache["router_probs"]
        top_k_indices = cache["top_k_indices"]
        expert_caches = cache["expert_caches"]

        # Compute gradient through router logits
        # This is a simplified implementation
        dx = dy  # Simplified for now

        return dx

    def zero_grad(self):
        for p in self.parameters():
            if hasattr(p, 'zero_grad'):
                p.zero_grad()