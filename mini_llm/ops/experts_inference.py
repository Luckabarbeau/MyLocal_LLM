"""Experts for inference - no backward caches needed."""


import os
from contextlib import nullcontext

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

    def __init__(self, d_model, d_ff, n_experts, dtype="float32", *,
                 expert_executor=None, layer_id=0):
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
        self.expert_executor = expert_executor
        self.layer_id = int(layer_id)

        # Create inference experts.  The sparse objects remain the prefill
        # implementation/reference path; decode additionally uses persistent
        # stacked weights so all experts can be evaluated in batched matmuls
        # without any GPU->CPU expert-id synchronization.
        self.experts = []
        for i in range(n_experts if expert_executor is None else 0):
            expert = ExpertFFNInference(
                d_model=d_model,
                d_ff=d_ff,
                dtype=dtype
            )
            self.experts.append(expert)

    def set_weights(self, experts):
        """Set weights from the training Experts module."""
        if self.expert_executor is not None:
            raise RuntimeError("Managed experts load from their host provider; do not set resident weights")
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
        # Protect selected GPU experts from eviction while their launches and
        # routed reductions are pending. The optional executor records a single
        # CUDA completion event for the MoE layer when this scope exits.
        scope = (self.expert_executor.dispatch_scope(self.layer_id, routing_plan.active_expert_ids())
                 if self.expert_executor is not None else nullcontext())
        with scope:
            if self.expert_executor is not None and getattr(
                self.expert_executor, "overlaps_cpu_gpu", False
            ):
                # Submit every selected CPU expert before waiting for any result.
                # Resident GPU experts enqueue kernels on the caller's stream
                # while CPU workers concurrently run offloaded experts. The
                # final scatter remains ordered by expert id, preserving the
                # deterministic reduction order of the synchronous dispatcher.
                jobs = []
                try:
                    for exp_idx in routing_plan.active_expert_ids():
                        token_indices, _, expert_weights = (
                            routing_plan.get_expert_assignments(exp_idx)
                        )
                        pending = self.expert_executor.submit(
                            self.layer_id, exp_idx, x_flat[token_indices]
                        )
                        jobs.append((token_indices, expert_weights, pending))
                except Exception:
                    # A provider or GPU kernel may fail after some workers were
                    # submitted. Drain them before allowing their inputs to die.
                    for _, _, pending in jobs:
                        try:
                            self.expert_executor.resolve(pending)
                        except Exception:
                            pass
                    raise
                # Drain all started workers, including on exceptions, so a failed
                # expert never leaves outstanding tasks writing to discarded GPU
                # buffers or reading activations from the next inference step.
                error = None
                for token_indices, expert_weights, pending in jobs:
                    try:
                        expert_out = self.expert_executor.resolve(pending)
                        if error is None:
                            y_flat[token_indices] += expert_weights[:, xp.newaxis] * expert_out
                    except Exception as exc:
                        if error is None:
                            error = exc
                if error is not None:
                    raise error
            else:
                for exp_idx in routing_plan.active_expert_ids():
                    token_indices, _, expert_weights = (
                        routing_plan.get_expert_assignments(exp_idx)
                    )
                    if self.expert_executor is None:
                        expert_out = self.experts[exp_idx].forward(x_flat[token_indices])
                    else:
                        expert_out = self.expert_executor.forward(
                            self.layer_id, exp_idx, x_flat[token_indices]
                        )
                    y_flat[token_indices] += expert_weights[:, xp.newaxis] * expert_out

        return y_flat.reshape(batch_size, seq_len, d_model)
