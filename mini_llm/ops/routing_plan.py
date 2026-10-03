"""Routing plan for Mixture of Experts.

Creates a plan for token-to-expert assignments that can be reused during
forward and backward without rebuilding masks for every expert.

The hot-path arrays stay on the selected backend.  Only the tiny per-expert
assignment counts are copied to the host once when the plan is created so
expert slices use ordinary Python integer offsets instead of GPU scalar
indices (which would synchronize repeatedly under CuPy).
"""

from ..backend import xp, asnumpy
from ..performance_profiler import moe_detail_scope


class RoutingPlan:
    """Pre-computed token-to-expert dispatch information."""

    def __init__(
        self,
        token_indices,
        slot_indices,
        expert_indices,
        weights,
        order,
        expert_offsets,
        expert_sizes,
        n_experts: int,
        k: int,
        N: int,
    ):
        self.token_indices = token_indices
        self.slot_indices = slot_indices
        self.expert_indices = expert_indices
        self.weights = weights
        self.order = order

        # Keep these on the host.  They contain only n_experts (+1) integers
        # and are used for Python slicing in every expert dispatch.  Keeping
        # them as backend scalars would force GPU->CPU synchronization every
        # time get_expert_assignments() is called with CuPy.
        self.expert_offsets = tuple(int(v) for v in expert_offsets)
        self.expert_sizes = tuple(int(v) for v in expert_sizes)
        self.n_experts = int(n_experts)
        self.k = int(k)
        self.N = int(N)

    @classmethod
    def from_router_outputs(
        cls,
        expert_indices,
        weights,
        k: int,
        n_experts: int = None,
    ):
        """Create a routing plan from router top-k outputs.

        There are N*k assignments.  We sort the flattened assignments once by
        expert id and retain the same order for token ids, top-k slot ids, and
        routing weights.  The sorted arrays are then reused by both forward and
        backward.
        """
        batch_size, seq_len, _ = expert_indices.shape
        N = batch_size * seq_len

        if n_experts is None:
            # Compatibility fallback.  Normal MoE code should always provide
            # n_experts so this host synchronization is not needed.
            n_experts = int(asnumpy(xp.max(expert_indices))) + 1

        expert_ids_flat = expert_indices.reshape(-1)
        weights_flat = weights.reshape(-1)

        # Group assignments once.  The flattened assignment index already
        # contains both token and top-k slot information:
        #   token = assignment // k
        #   slot  = assignment % k
        # so avoid constructing repeat()/tile() arrays before the sort.
        with moe_detail_scope("moe.plan.sort"):
            order = xp.argsort(expert_ids_flat)
        with moe_detail_scope("moe.plan.permute"):
            expert_sorted = expert_ids_flat[order]
            token_sorted = order // k
            slot_sorted = order % k
            weights_sorted = weights_flat[order]

        # Compute all expert sizes in one backend operation, then perform ONE
        # tiny host transfer.  The old implementation called .item() once per
        # expert, causing repeated CuPy synchronization.
        with moe_detail_scope("moe.plan.counts"):
            counts = xp.bincount(expert_ids_flat, minlength=n_experts)
        with moe_detail_scope("moe.plan.host_counts"):
            counts_host = asnumpy(counts)

        offsets = [0]
        sizes = []
        running = 0
        for count in counts_host:
            count_i = int(count)
            sizes.append(count_i)
            running += count_i
            offsets.append(running)

        return cls(
            token_indices=token_sorted,
            slot_indices=slot_sorted,
            expert_indices=expert_sorted,
            weights=weights_sorted,
            order=order,
            expert_offsets=offsets,
            expert_sizes=sizes,
            n_experts=n_experts,
            k=k,
            N=N,
        )

    def get_expert_assignments(self, expert_idx: int):
        """Return token ids, top-k slot ids, and weights for one expert."""
        start = self.expert_offsets[expert_idx]
        end = self.expert_offsets[expert_idx + 1]
        return (
            self.token_indices[start:end],
            self.slot_indices[start:end],
            self.weights[start:end],
        )

    def get_assignment_count(self, expert_idx: int) -> int:
        """Return the number of assignments routed to one expert."""
        return self.expert_sizes[expert_idx]


def create_routing_plan(expert_indices, weights, k: int, n_experts: int) -> RoutingPlan:
    """Convenience wrapper preserving the explicit expert count."""
    return RoutingPlan.from_router_outputs(
        expert_indices,
        weights,
        k,
        n_experts=n_experts,
    )
