"""0103-0: framework-neutral, immutable contracts for the model-centric API.

These types describe a model or an observed execution.  They deliberately
contain no NumPy/CuPy/PyTorch tensors and do not choose a numerical kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class UnsupportedModelOperation(RuntimeError):
    """The selected model/provider adapter cannot perform this operation."""


class UnsupportedExecutionBackend(RuntimeError):
    """The requested backend/device is unavailable in this execution process."""


class UnverifiedExecutionPath(RuntimeError):
    """A strict performance request lacks verified acceleration evidence."""


@dataclass(frozen=True)
class Capabilities:
    """Operations *currently wired through this API*, not theoretical abilities."""

    inspect: bool = True
    generate: bool = False
    train: bool = False
    score: bool = False
    streaming: bool = False
    persistent_sessions: bool = False
    accessible_weights: bool = False
    tools: bool = False


@dataclass(frozen=True)
class OperatorSpec:
    """Semantic operation identifier, independent of its optimized implementation."""

    name: str
    version: int = 1


@dataclass(frozen=True)
class ArchitectureSpec:
    """Portable description of one architecture; NOT a materialized model."""

    family: str
    layers: int
    hidden_size: int
    ffn_size: int
    vocabulary_size: int
    context_length: int
    dtype: str
    operators: tuple[OperatorSpec, ...]
    block_size: int | None = None
    summaries_per_block: int | None = None
    local_window: int | None = None
    long_local_window: int | None = None
    sparse_stride: int = 0
    sparse_layers: tuple[int, ...] = ()
    sparse_head: int | None = None

    @classmethod
    def from_config(cls, config: Any) -> "ArchitectureSpec":
        dual = getattr(config, "dual_resolution", None)
        enabled = bool(dual is not None and dual.enabled)
        matrix = bool(enabled and dual.compression_mode == "matrix_pool")
        stride = int(dual.sparse_retrieval_stride) if matrix else 0
        family = (
            "dual_resolution_matrix_pool_sparse" if stride else
            "dual_resolution_matrix_pool" if matrix else
            "dual_resolution_transformer" if enabled else
            "decoder_language_model"
        )
        ops = [OperatorSpec(name) for name in
               ("embedding", "rmsnorm", "grouped_attention", "rope", "ffn", "lm_head")]
        if matrix:
            ops.extend((OperatorSpec("completed_block_matrix_pool"),
                        OperatorSpec("compressed_global_attention")))
        if stride:
            ops.append(OperatorSpec("sparse_stride_history"))
        return cls(
            family=family,
            layers=int(config.n_layers),
            hidden_size=int(config.d_model),
            ffn_size=int(config.d_ff),
            vocabulary_size=int(config.vocab_size),
            context_length=int(config.context_length),
            dtype=str(config.dtype),
            operators=tuple(ops),
            block_size=int(dual.block_size) if enabled else None,
            summaries_per_block=int(dual.summaries_per_block) if enabled else None,
            local_window=int(dual.local_window) if enabled else None,
            long_local_window=int(dual.long_local_window) if enabled else None,
            sparse_stride=stride,
            sparse_layers=tuple(dual.sparse_retrieval_layers) if stride else (),
            sparse_head=int(dual.sparse_retrieval_head) if stride else None,
        )


@dataclass(frozen=True)
class ExecutionReport:
    """Observed execution information; acceleration is *unverified* until measured."""

    backend: str
    device: str
    dtype: str
    acceleration_verified: bool = False
    active_optimizations: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def require_fast(self) -> None:
        if not self.acceleration_verified or self.fallbacks:
            raise UnverifiedExecutionPath(
                "Fast execution is not verified for this model/session. "
                "Do not report this run as an accelerated benchmark."
            )


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    architecture: ArchitectureSpec
    parameter_count: int | None
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    capabilities: Capabilities
    checkpoint: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    timing: dict[str, Any]
    execution: ExecutionReport
