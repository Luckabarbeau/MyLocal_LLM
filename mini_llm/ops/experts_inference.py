"""Experts for inference - no backward caches needed."""


import os

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.ops.experts import _fused_swiglu_forward
from mini_llm.ops.routing_plan import RoutingPlan
from mini_llm.ops.silu import silu


class ExpertFFNInference:
    """
    Single expert FFN for inference.
    
    Same computation as training but without caching intermediates for backprop.
    """

    def __init__(self, d_model, d_ff, dtype="float32"):
        """
        Initialize expert FFN for inference.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        
        # Will be set externally
        self.W_gate = None
        self.W_up = None
        self.W_down = None
        self.W_gate_up = None

    def set_weight_arrays(self, gate, up, down):
        """Attach inference weights and persistently pack FP16 gate/up once."""
        enabled = os.environ.get(
            "MINI_LLM_PACKED_INFERENCE_FFN", "1"
        ).strip().lower() not in {"0", "false", "off", "no", ""}
        if (
            enabled
            and BACKEND_NAME == "cupy"
            and gate.dtype == xp.float16
            and up.dtype == xp.float16
        ):
            self.W_gate_up = xp.concatenate((gate, up), axis=1)
            self.W_gate = self.W_gate_up[:, :self.d_ff]
            self.W_up = self.W_gate_up[:, self.d_ff:]
        else:
            self.W_gate_up = None
            self.W_gate = gate
            self.W_up = up
        self.W_down = down

    def set_weights(self, expert):
        """Set weights from training ExpertFFN."""
        self.set_weight_arrays(
            expert.W_gate.data, expert.W_up.data, expert.W_down.data
        )

    def forward(self, x):
        """
        Forward pass through expert.
        
        Args:
            x: Input [B, T, D] or [N, D]
            
        Returns:
            y: Output [B, T, D] or [N, D]
        """
        if self.W_gate_up is None:
            g = x @ self.W_gate  # [..., d_ff]
            u = x @ self.W_up    # [..., d_ff]
        else:
            gate_up = x @ self.W_gate_up
            g = gate_up[..., :self.d_ff]
            u = gate_up[..., self.d_ff:]
        h = _fused_swiglu_forward(g, u)
        if h is None:
            a = silu(g)
            h = a * u
        y = h @ self.W_down  # [..., d_model]
        
        return y


class ExpertsInference:
    """
    Mixture of Expert FFN layers for inference with sparse evaluation.
    
    Only evaluates the selected experts, not all experts.
    """

    def __init__(self, d_model, d_ff, n_experts, dtype="float32"):
        """
        Initialize experts for inference.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        
        # Create inference experts.  The sparse objects remain the prefill
        # implementation/reference path; decode additionally uses persistent
        # stacked weights so all experts can be evaluated in batched matmuls
        # without any GPU->CPU expert-id synchronization.
        self.experts = []
        for i in range(n_experts):
            expert = ExpertFFNInference(
                d_model=d_model,
                d_ff=d_ff,
                dtype=dtype
            )
            self.experts.append(expert)

    def set_weights(self, experts):
        """Set weights from the training Experts module."""
        for i in range(self.n_experts):
            self.experts[i].set_weights(experts.experts[i])

    def forward(self, x, weights, expert_indices):
        """Sparse inference dispatch matching the training forward semantics.

        Both prompt prefill and one-token decode use the same RoutingPlan-based
        sparse top-k dispatch.  BF16 decode deliberately avoids the previous
        all-expert FP32 path because that changes the numerical path relative
        to the BF16 training forward.
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len

        x_flat = x.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        # Group assignments once, exactly as in the training dispatcher.
        routing_plan = RoutingPlan.from_router_outputs(
            expert_indices, weights, k, n_experts=self.n_experts
        )
        for exp_idx in range(self.n_experts):
            if routing_plan.get_assignment_count(exp_idx) == 0:
                continue

            token_indices, _, expert_weights = (
                routing_plan.get_expert_assignments(exp_idx)
            )
            expert_out = self.experts[exp_idx].forward(x_flat[token_indices])
            y_flat[token_indices] += expert_weights[:, xp.newaxis] * expert_out

        return y_flat.reshape(batch_size, seq_len, d_model)
