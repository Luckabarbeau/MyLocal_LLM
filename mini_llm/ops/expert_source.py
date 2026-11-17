"""Canonical expert sources, independent of execution and residency."""

from dataclasses import dataclass
import numpy as np

from .expert_residency import ExpertKey
from .quantized_weight import PackedGGUFWeight


@dataclass(frozen=True, eq=False)
class ExpertBundle:
    """Complete gate/up/down handles retaining their canonical source owner.

    Read-only views are accepted without changing caller-owned array flags.
    The owner is responsible for not mutating writable aliases, if any.
    """

    key: ExpertKey
    weights: tuple
    owner: object
    source_tier: str

    def __post_init__(self):
        if not isinstance(self.key, ExpertKey):
            raise TypeError("key must be ExpertKey")
        if self.source_tier not in {"mmap", "host"}:
            raise ValueError("source_tier must be 'mmap' or 'host'")
        if self.owner is None:
            raise ValueError("source owner is required")
        weights = tuple(self.weights)
        if len(weights) != 3:
            raise ValueError("ExpertBundle requires gate, up and down")
        packed = [isinstance(w, PackedGGUFWeight) for w in weights]
        if any(packed) != all(packed):
            raise ValueError("Mixed dense/packed projections are unsupported")
        for w in weights:
            data = w.data if all(packed) else w
            if not isinstance(data, np.ndarray) or data.flags.writeable:
                raise ValueError("Canonical provider weights must be read-only arrays")
            if len(w.shape) != 2 or min(w.shape) <= 0:
                raise ValueError("Expert projections must be positive matrices")
        g, u, d = weights
        if g.shape != u.shape or d.shape != g.shape[::-1]:
            raise ValueError("Incompatible gate/up/down shapes")
        object.__setattr__(self, "weights", weights)

    @property
    def nbytes(self):
        return sum(w.nbytes for w in self.weights)
