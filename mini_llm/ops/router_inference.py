"""MoE expert router used at inference, sharing its weighting rule with training."""

from mini_llm.ops.router import expert_topk_forward


class RouterInference:
    """Full-softmax-then-top-k router; no backward cache required."""

    def __init__(self, d_model, n_experts, k, dtype="float32"):
        self.d_model = d_model
        self.n_experts = n_experts
        if not (1 <= int(k) <= int(n_experts)):
            raise ValueError("expert router expects 1 <= k <= n_experts")
        self.k = int(k)
        self.W_router = None
        self.b_router = None

    def set_weights(self, router):
        self.W_router = router.W_router_param.data
        self.b_router = router.b_router_param.data

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x2 = x.reshape(batch_size * seq_len, self.d_model)
        logits = (x2 @ self.W_router + self.b_router).reshape(
            batch_size, seq_len, self.n_experts
        )
        weights, indices, _ = expert_topk_forward(
            logits, self.k, output_dtype=x.dtype, return_cache=False,
        )
        return weights, indices
