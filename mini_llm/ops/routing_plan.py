"""Routing plan for Mixture of Experts.

Creates a plan for token-to-expert assignments that can be reused
during both forward and backward passes, avoiding redundant routing computation.
"""

from ..backend import xp


class RoutingPlan:
    """
    Pre-computed routing plan for MoE dispatch.
    
    Contains all information needed to efficiently dispatch tokens to experts
    and collect results, without recomputing masks during backward.
    
    For N tokens and top_k = k, we have N*k assignments.
    
    Attributes:
        order: Indices that would sort expert_ids
        token_indices: Token IDs sorted by expert (for forward)
        slot_indices: Slot IDs sorted by expert (for forward)
        expert_indices: Expert IDs sorted (for forward)
        weights: Weights sorted by expert (for forward)
        expert_offsets: Start index for each expert in sorted arrays
        expert_sizes: Number of assignments per expert
        n_experts: Total number of experts
        k: Top-k value
        N: Total number of tokens (batch * seq_len)
    """

    def __init__(
        self,
        token_indices: xp.ndarray,
        slot_indices: xp.ndarray,
        expert_indices: xp.ndarray,
        weights: xp.ndarray,
        order: xp.ndarray,
        expert_offsets: xp.ndarray,
        expert_sizes: xp.ndarray,
        n_experts: int,
        k: int,
        N: int,
    ):
        """
        Initialize routing plan.
        
        Args:
            token_indices: Token IDs sorted by expert, shape [N*k]
            slot_indices: Slot IDs sorted by expert, shape [N*k]
            expert_indices: Expert IDs sorted, shape [N*k]
            weights: Weights sorted by expert, shape [N*k]
            order: Indices that sort expert_ids
            expert_offsets: Start index for each expert in sorted arrays
            expert_sizes: Number of assignments per expert
            n_experts: Total number of experts
            k: Top-k value
            N: Total number of tokens
        """
        self.token_indices = token_indices
        self.slot_indices = slot_indices
        self.expert_indices = expert_indices
        self.weights = weights
        self.order = order
        self.expert_offsets = expert_offsets
        self.expert_sizes = expert_sizes
        self.n_experts = n_experts
        self.k = k
        self.N = N

    @classmethod
    def from_router_outputs(cls, expert_indices: xp.ndarray, weights: xp.ndarray, k: int, n_experts: int = None):
        """
        Create routing plan from router outputs.
        
        Args:
            expert_indices: Expert indices from router, shape [B, T, k]
            weights: Weights from router, shape [B, T, k]
            k: Top-k value
            n_experts: Total number of experts (computed if not provided)
            
        Returns:
            RoutingPlan instance
        """
        batch_size, seq_len, _ = expert_indices.shape
        N = batch_size * seq_len
        
        # Determine n_experts if not provided
        if n_experts is None:
            n_experts = int(xp.max(expert_indices).item()) + 1
        
        # Flatten all arrays
        expert_indices_flat = expert_indices.reshape(N, k)
        weights_flat = weights.reshape(N, k)
        
        # Create token_ids and slot_ids arrays
        # token_ids: [0, 0, ..., 1, 1, ..., N-1, N-1, ...] (k times each)
        token_ids = xp.repeat(xp.arange(N), k)  # [N*k]
        
        # slot_ids: [0, 1, ..., k-1, 0, 1, ..., k-1, ...] (N times)
        slot_ids = xp.tile(xp.arange(k), N)  # [N*k]
        
        # Flatten expert indices and weights
        expert_ids_flat = expert_indices_flat.flatten()  # [N*k]
        weights_flat_sorted = weights_flat.flatten()  # [N*k]
        
        # Sort by expert id to group assignments by expert
        order = xp.argsort(expert_ids_flat)
        expert_sorted = expert_ids_flat[order]  # [N*k]
        token_sorted = token_ids[order]  # [N*k]
        slot_sorted = slot_ids[order]  # [N*k]
        weights_sorted = weights_flat_sorted[order]  # [N*k]
        
        # Compute expert offsets (start index for each expert)
        # For efficient lookup: given expert_id, where does its assignments start?
        expert_offsets = xp.empty(n_experts + 1, dtype=xp.int32)
        expert_offsets[0] = 0
        
        for exp_idx in range(n_experts):
            # Count assignments for this expert
            count = xp.sum(expert_sorted == exp_idx).item()
            expert_offsets[exp_idx + 1] = expert_offsets[exp_idx] + count
        
        # Compute expert sizes (number of assignments per expert)
        expert_sizes = expert_offsets[1:] - expert_offsets[:-1]
        
        return cls(
            token_indices=token_sorted,
            slot_indices=slot_sorted,
            expert_indices=expert_sorted,
            weights=weights_sorted,
            order=order,
            expert_offsets=expert_offsets,
            expert_sizes=expert_sizes,
            n_experts=n_experts,
            k=k,
            N=N,
        )

    def get_expert_assignments(self, expert_idx: int) -> tuple:
        """
        Get token, slot, and weight indices for a specific expert.
        
        Args:
            expert_idx: Expert ID
            
        Returns:
            Tuple of (token_indices, slot_indices, weights) for this expert
        """
        start = self.expert_offsets[expert_idx]
        end = self.expert_offsets[expert_idx + 1]
        return (
            self.token_indices[start:end],
            self.slot_indices[start:end],
            self.weights[start:end],
        )

    def get_assignment_count(self, expert_idx: int) -> int:
        """Get number of token assignments to a specific expert."""
        return self.expert_sizes[expert_idx].item()


def create_routing_plan(expert_indices: xp.ndarray, weights: xp.ndarray, k: int, n_experts: int) -> RoutingPlan:
    """
    Convenience function to create a routing plan.
    
    Args:
        expert_indices: Expert indices from router, shape [B, T, k]
        weights: Weights from router, shape [B, T, k]
        k: Top-k value
        n_experts: Total number of experts
        
    Returns:
        RoutingPlan instance
    """
    return RoutingPlan.from_router_outputs(expert_indices, weights, k)
