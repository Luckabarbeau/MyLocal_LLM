"""Experts for inference - no backward caches needed."""


from mini_llm.backend import xp, is_bfloat16_dtype
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

        # CuPy 14.x cannot execute generic N-D BF16 matmul.  Single-token
        # decode evaluates all experts as one batched operation, so retain
        # FP32 mirrors of those stacked matrices only when the model weights
        # are BF16.  Prompt/prefill still uses the original BF16 2-D GEMMs.
        self.W_gate_stack_f32 = None
        self.W_up_stack_f32 = None
        self.W_down_stack_f32 = None

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

        if is_bfloat16_dtype(self.W_gate_stack.dtype):
            self.W_gate_stack_f32 = self.W_gate_stack.astype(
                xp.float32, copy=False
            )
            self.W_up_stack_f32 = self.W_up_stack.astype(
                xp.float32, copy=False
            )
            self.W_down_stack_f32 = self.W_down_stack.astype(
                xp.float32, copy=False
            )
        else:
            self.W_gate_stack_f32 = None
            self.W_up_stack_f32 = None
            self.W_down_stack_f32 = None

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
            # than synchronizing selected expert ids to Python.  CuPy 14.x
            # cannot run its generic N-D matmul path with BF16 operands.  For
            # BF16 models, use persistent FP32 mirrors for these *batched*
            # single-token expert products.  The much larger prompt/prefill
            # expert products remain true 2-D BF16 GEMMs below.
            bf16_decode = self.W_gate_stack_f32 is not None
            if bf16_decode:
                x_experts = x_flat.astype(
                    xp.float32, copy=False
                )[xp.newaxis, :, :]
                W_gate = self.W_gate_stack_f32
                W_up = self.W_up_stack_f32
                W_down = self.W_down_stack_f32
            else:
                x_experts = x_flat[xp.newaxis, :, :]
                W_gate = self.W_gate_stack
                W_up = self.W_up_stack
                W_down = self.W_down_stack

            g = xp.matmul(x_experts, W_gate)  # [E,1,Dff]
            u = xp.matmul(x_experts, W_up)    # [E,1,Dff]
            h = silu(g) * u
            all_outputs = xp.matmul(h, W_down)  # [E,1,D]

            selected = all_outputs[
                expert_indices.reshape(-1), 0, :
            ]  # [k,D], GPU gather only after expert computation
            combine_weights = weights_flat[0, :, xp.newaxis]
            if bf16_decode:
                combine_weights = combine_weights.astype(
                    xp.float32, copy=False
                )
            combined = xp.sum(
                selected * combine_weights,
                axis=0,
            )

            # Return the branch in the model compute dtype so the next block
            # continues to use BF16 2-D projection GEMMs.
            if bf16_decode:
                combined = combined.astype(x.dtype, copy=False)
            return combined.reshape(batch_size, seq_len, d_model)

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
