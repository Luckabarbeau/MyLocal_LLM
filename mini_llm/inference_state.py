"""Generation state for KV-cached autoregressive inference."""


from dataclasses import dataclass, field
from typing import Optional

from mini_llm.backend import xp


@dataclass
class GenerationState:
    """
    State for KV-cached autoregressive generation.
    
    The K/V caches are preallocated and reused across generation steps.
    No backward caches are stored - this is purely for inference.
    
    Cache layout (per layer):
        K cache: [batch_size, n_kv_heads, max_length, head_dim]
        V cache: [batch_size, n_kv_heads, max_length, head_dim]
    
    The cache stores only the actual KV heads (not expanded for GQA).
    During attention, KV is expanded to match Q heads count.
    """
    
    k_cache: xp.ndarray
    v_cache: xp.ndarray
    length: int
    max_length: int
    batch_size: int
    # For configurable sparse/retrieval attention, each layer keeps the
    # normalized attention input used by ContextRouter.  This is inference
    # state, not a backward cache; BF16 storage keeps the cost modest and lets
    # route decisions be reconstructed exactly at routing boundaries.
    router_input_cache: Optional[xp.ndarray] = None
    # Latest learned retrieval decision per (layer, group).  A decision is
    # reused for routing_stride consecutive decode positions.
    retrieval_routes: dict = field(default_factory=dict)

    # 0059I terminal-Landmark long-memory inference state.  The deep
    # Transformer cache remains bounded by ``working_capacity`` while
    # ``max_length`` is the total addressable horizon.  ``cache_start`` is the
    # physical ring slot corresponding to logical working position zero.
    long_memory_enabled: bool = False
    working_capacity: int = 0
    working_count: int = 0
    cache_start: int = 0
    working_start_abs: int = 0
    next_abs_pos: int = 0
    working_token_ids: Optional[xp.ndarray] = None
    working_embedding_sum: Optional[xp.ndarray] = None
    terminal_memory_store: object = None
    terminal_memory_route: object = None

    # 0060B routed-prefix long-memory state.  The top-level memory store keeps
    # only old token IDs and cheap router projections.  Deep K/V remains bounded
    # by [retrieved prefix | working window].  ``cache_start`` is the ring start
    # *inside the working segment*; the prefix segment itself never rotates.
    routed_prefix_store: object = None
    routed_prefix_route: object = None
    routed_prefix_length: int = 0
    routed_prefix_refresh_tokens: int = 0
    routed_prefix_tokens_since_refresh: int = 0
    routed_prefix_refresh_count: int = 0

    def reset(self):
        """Reset generation state without deallocating cache."""
        self.length = 0
        self.retrieval_routes.clear()
        self.terminal_memory_route = None
        self.routed_prefix_route = None
        self.routed_prefix_tokens_since_refresh = 0
        self.routed_prefix_refresh_count = 0
    
    def remaining_capacity(self) -> int:
        """Remaining tokens that can be generated."""
        if self.long_memory_enabled:
            # The terminal-memory state is a sliding horizon. Generation can
            # continue after the addressable horizon fills because old external
            # blocks are evicted from the memory ring.
            return 2**63 - 1
        return self.max_length - self.length
    
    def has_capacity(self, n_tokens: int = 1) -> bool:
        """Check if there's capacity for n more tokens."""
        if self.long_memory_enabled:
            return True
        return self.length + n_tokens <= self.max_length
