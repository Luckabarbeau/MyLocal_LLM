"""Experts for inference - no backward caches needed."""


from mini_llm.backend import xp
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

    def set_weights(self, expert):
        """Set weights from training ExpertFFN."""
        self.W_gate = expert.W_gate.data
        self.W_up = expert.W_up.data
        self.W_down = expert.W_down.data

    def forward(self, x):
        """
        Forward pass through expert.
        
        Args:
            x: Input [B, T, D] or [N, D]
            
        Returns:
            y: Output [B, T, D] or [N, D]
        """
        g = x @ self.W_gate  # [..., d_ff]
        u = x @ self.W_up    # [..., d_ff]
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

        self.W_gate_stack = None
        self.W_up_stack = None
        self.W_down_stack = None

    def set_weights(self, experts):
        """Set weights and build persistent stacked decode matrices once."""
        for i in range(self.n_experts):
            self.experts[i].set_weights(experts.experts[i])

        self.W_gate_stack = xp.stack(
            [expert.W_gate for expert in self.experts], axis=0
        )
        self.W_up_stack = xp.stack(
            [expert.W_up for expert in self.experts], axis=0
        )
        self.W_down_stack = xp.stack(
            [expert.W_down for expert in self.experts], axis=0
        )

    def forward(self, x, weights, expert_indices):
        """Sparse inference dispatch with a special single-token decode path.

        Decode avoids the old ``for every expert -> xp.any(mask)`` pattern,
        which forced up to ``n_experts`` device/host synchronizations per
        layer.  Only the two selected expert ids are copied to the host once.
        Prefill reuses the same grouped RoutingPlan used by training.
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len

        x_flat = x.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        if N == 1:
            # GPU-only decode experiment: evaluate all experts together rather
            # than synchronizing selected expert ids to Python.  This performs
            # more arithmetic than sparse top-k, but converts many tiny dynamic
            # operations into three regular batched matmuls.  Benchmark on the
            # target GPU before keeping this path.
            x_experts = x_flat[xp.newaxis, :, :]  # [1,1,D]
            g = xp.matmul(x_experts, self.W_gate_stack)  # [E,1,Dff]
            u = xp.matmul(x_experts, self.W_up_stack)    # [E,1,Dff]
            h = silu(g) * u
            all_outputs = xp.matmul(
                h, self.W_down_stack
            )  # [E,1,D]

            selected = all_outputs[
                expert_indices.reshape(-1), 0, :
            ]  # [k,D], GPU gather only after expert computation
            combined = xp.sum(
                selected * weights_flat[0, :, xp.newaxis],
                axis=0,
            )
            y_flat[0, :] = combined
            return y_flat.reshape(batch_size, seq_len, d_model)

        # Prompt/prefill path: group assignments once instead of performing
        # one device-synchronizing xp.any() check for every expert.
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
