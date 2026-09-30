"""Generation state for KV-cached autoregressive inference."""


from dataclasses import dataclass
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
    
    def reset(self):
        """Reset generation state without deallocating cache."""
        self.length = 0
    
    def remaining_capacity(self) -> int:
        """Remaining tokens that can be generated."""
        return self.max_length - self.length
    
    def has_capacity(self, n_tokens: int = 1) -> bool:
        """Check if there's capacity for n more tokens."""
        return self.length + n_tokens <= self.max_length
