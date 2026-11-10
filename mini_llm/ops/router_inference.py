"""MoE expert router used at inference, sharing its weighting rule with training."""

import math

from mini_llm.backend import xp
from mini_llm.ops.router import expert_topk_forward


class RouterInference:
    """Full-softmax-then-top-k router; optional inference-only renormalization.

    The default retains native training/inference selected probability mass.
    External adapters may explicitly normalize selected probabilities instead.
    """

    def __init__(self, d_model, n_experts, k, dtype="float32", *,
                 renormalize_selected=False, normalization_floor=1e-30):
        self.d_model = d_model
        self.n_experts = n_experts
        if not (1 <= int(k) <= int(n_experts)):
            raise ValueError("expert router expects 1 <= k <= n_experts")
        self.k = int(k)
        if not isinstance(renormalize_selected, bool):
            raise TypeError("renormalize_selected must be bool")
        if not math.isfinite(normalization_floor) or normalization_floor <= 0:
            raise ValueError("normalization_floor must be finite and positive")
        self.renormalize_selected = renormalize_selected
        self.normalization_floor = float(normalization_floor)
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
            logits, self.k,
            output_dtype=None if self.renormalize_selected else x.dtype,
            return_cache=False,
        )
        if self.renormalize_selected:
            # Normalize in the softmax compute dtype before low-precision storage.
            total = xp.sum(weights, axis=-1, keepdims=True)
            weights = (weights / xp.maximum(total, self.normalization_floor)).astype(
                x.dtype, copy=False)
        return weights, indices
