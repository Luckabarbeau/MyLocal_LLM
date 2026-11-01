"""0103-0: backend and kernel capability contracts (no numerical rewrites).

The registry describes reusable operator implementations; it is intentionally
separate from the architecture and has no implicit environment-variable
fallbacks. Registering does not claim an implementation is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .model_contracts import ArchitectureSpec, ExecutionReport, OperatorSpec


@runtime_checkable
class ExecutionEngine(Protocol):
    """Base protocol for a backend-bound model. Optional operations are separate."""

    @property
    def spec(self) -> ArchitectureSpec: ...

    def execution_info(self) -> ExecutionReport: ...


@runtime_checkable
class GenerationEngine(Protocol):
    """Inference-specific contract; implementation may use fused kernels."""

    def new_state(self, *, batch_size: int, max_length: int) -> Any: ...

    def prefill(self, token_ids: Any, state: Any) -> Any: ...

    def decode_one(self, token_ids: Any, state: Any) -> Any: ...


@runtime_checkable
class TrainingEngine(Protocol):
    """Training-specific contract; CuPy may use handwritten backward."""

    def forward_loss(self, batch: Any) -> Any: ...

    def backward(self, loss_cache: Any) -> Any: ...


@dataclass(frozen=True)
class KernelImplementation:
    """Descriptor for an implementation, not a performance benchmark result."""

    name: str
    operation: OperatorSpec
    backend: str
    mode: str  # forward, backward, prefill, decode
    verified: bool = False


class KernelRegistry:
    """Exact-match registry; no guessed compatibility or silent fallbacks.

    Matching constraints can be expanded in 0103D once full-model runtime
    benchmarks establish shape, dtype and hardware requirements.
    """

    def __init__(self):
        self._implementations: dict[tuple[str, int, str, str, str], KernelImplementation] = {}

    def register(self, implementation: KernelImplementation) -> None:
        if not implementation.name or not implementation.operation.name:
            raise ValueError("Kernel and operation names must not be empty")
        if implementation.backend not in {"numpy", "cupy", "torch"}:
            raise ValueError("Unknown kernel backend")
        if implementation.mode not in {"forward", "backward", "prefill", "decode"}:
            raise ValueError("Unknown kernel execution mode")
        key = (implementation.operation.name, implementation.operation.version,
               implementation.backend, implementation.mode, implementation.name)
        if key in self._implementations:
            raise ValueError(f"Duplicate kernel registration: {implementation.name}")
        self._implementations[key] = implementation

    def eligible(self, operation: OperatorSpec, *, backend: str, mode: str,
                 require_verified: bool = True) -> tuple[KernelImplementation, ...]:
        """Report registered matches; does not choose a fastest kernel yet."""
        matches = (value for key, value in self._implementations.items()
                   if key[0] == operation.name and key[1] == operation.version
                   and key[2] == backend and key[3] == mode
                   and (not require_verified or value.verified))
        return tuple(sorted(matches, key=lambda impl: impl.name))
