"""0103D: one model-aware inference policy and *observed* execution audit.

Legacy kernels still read process-wide MINI_LLM_* variables. This is an
incremental bridge until 0103E makes execution options session-local. Never
infer that a CUDA Graph ran merely because its environment flag was enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .model_contracts import ExecutionReport, UnverifiedExecutionPath

POLICIES = ("auto", "reference", "balanced", "fast", "strict-fast")
_FALSE = frozenset(("", "0", "false", "no", "off"))


def is_enabled(name: str) -> bool:
    return str(os.environ.get(name, "0")).strip().lower() not in _FALSE


def is_model_b_graph_compatible(config: Any, *, backend: str, dtype: Any = None) -> bool:
    """Narrow architecture contract of NativeDualBF16CudaGraphDecoder.

    The constructor remains the final authority (including device capability,
    weight layout, runtime allocation and stream capture errors).
    """
    dual = getattr(config, "dual_resolution", None)
    actual_dtype = str(dtype or getattr(config, "dtype", "")).lower()
    if backend != "cupy" or actual_dtype not in ("bf16", "bfloat16") or dual is None:
        return False
    return bool(
        getattr(dual, "enabled", False)
        and getattr(dual, "compression_mode", None) == "matrix_pool"
        and int(getattr(dual, "block_size", 0)) == 128
        and int(getattr(dual, "summaries_per_block", 0)) == 4
        and int(getattr(dual, "local_window", 0)) == 512
        and int(getattr(dual, "long_local_window", 0)) == 4096
        and int(getattr(dual, "long_local_heads", 0)) == 1
        and int(getattr(dual, "global_heads", 0)) == 2
        and int(getattr(dual, "global_head_dim", 0)) == 64
        and bool(getattr(dual, "global_enabled", False))
        and int(getattr(dual, "sparse_retrieval_stride", 0)) == 16
        and int(getattr(dual, "sparse_retrieval_head", -1)) == 1
        and int(getattr(config, "d_model", 0)) == 1024
        and int(getattr(config, "n_layers", 0)) == 20
        and int(getattr(config, "n_q_heads", 0)) == 16
        and int(getattr(config, "n_kv_heads", 0)) == 4
        and int(getattr(config, "d_head", 0)) == 64
    )


@dataclass(frozen=True)
class PolicyResult:
    requested: str
    resolved: str
    graph_requested: bool
    graph_configured: bool
    model_b_graph_eligible: bool
    changed: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]


def configure_inference(config: Any, *, backend: str, dtype: Any = None,
                        policy: str = "auto", model_format: str = "native") -> PolicyResult:
    """Select known-good defaults *before* constructing the numerical engine.

    Only the native Model B graph decoder is automatically enabled here.
    Other native and external architectures keep their existing fast profiles,
    but graph capture remains unverified or explicitly opt-in. A strict-fast
    request rejects configurations without an eligible decoder.
    """
    requested = str(policy).strip().lower().replace("_", "-")
    if requested not in POLICIES:
        raise ValueError(f"execution policy must be one of: {', '.join(POLICIES)}")
    if model_format not in ("native", "gguf", "safetensors"):
        raise ValueError(f"Unsupported in-process model format: {model_format}")
    resolved = ("fast" if backend == "cupy" else "reference") if requested == "auto" else requested
    if requested == "balanced":
        resolved = "fast" if backend == "cupy" else "reference"
    if resolved in ("fast", "strict-fast") and backend != "cupy":
        raise UnverifiedExecutionPath(f"{requested} requires the CuPy GPU backend; use auto for CPU reference")

    eligible = (model_format == "native" and is_model_b_graph_compatible(
        config, backend=backend, dtype=dtype))
    if requested == "strict-fast" and not eligible:
        raise UnverifiedExecutionPath(
            "strict-fast currently requires native BF16 Model B with stride-16 sparse history; "
            "other model acceleration cannot yet be verified"
        )

    changes: list[tuple[str, str]] = []
    warnings: list[str] = []
    if resolved in ("fast", "strict-fast"):
        from .runtime_defaults import apply_runtime_defaults
        prior = dict(os.environ)
        apply_runtime_defaults("gpu-fast", backend_name=backend, config=config,
                               training=False, precision=str(dtype or getattr(config, "dtype", "")))
        changes.extend((k, os.environ[k]) for k in sorted(set(os.environ) - set(prior))
                       if k.startswith("MINI_LLM_"))
        if eligible:
            # Explicit environment overrides remain visible and authoritative.
            for key, value in (("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1"),
                               ("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT", "1" if requested == "strict-fast" else "0")):
                if key not in os.environ:
                    os.environ[key] = value
                    changes.append((key, value))
            if not is_enabled("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE"):
                warnings.append("Model B CUDA Graph disabled by explicit environment override")
                if requested == "strict-fast":
                    raise UnverifiedExecutionPath(warnings[-1])
    elif requested == "reference":
        if is_enabled("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE"):
            raise UnverifiedExecutionPath(
                "reference mode conflicts with process-global CUDA Graph override; "
                "run in a fresh process with MINI_LLM_NATIVE_CUDA_GRAPH_DECODE=0"
            )
    return PolicyResult(
        requested=requested, resolved=resolved,
        graph_requested=eligible and resolved in ("fast", "strict-fast"),
        graph_configured=eligible and is_enabled("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE"),
        model_b_graph_eligible=eligible, changed=tuple(changes), warnings=tuple(warnings),
    )


def observe_execution(engine: Any, state: Any, *, policy: str = "auto") -> ExecutionReport:
    """Only a *live captured graph attached to this state* counts as verified.

    The report verifies graph activation, not mathematical equivalence or speed.
    Full CUDA numerical parity and performance measurements are separate gates.
    """
    from .backend import BACKEND_NAME
    graph = getattr(state, "_native_cuda_graph", None)
    external_graph = getattr(state, "_external_cuda_graph", None)
    active = graph if graph is not None else external_graph
    actual_count = int(getattr(state, "_cuda_graph_decode_count", 0))
    reasons = [str(v) for v in (
        getattr(state, "_native_cuda_graph_failure", None),
        getattr(state, "_external_cuda_graph_failure", None),
    ) if v]
    expected = is_enabled("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE") and bool(
        getattr(engine, "dual_resolution_enabled", False))
    if expected and not graph and not reasons:
        reasons.append("requested native CUDA Graph not active in generation state")
    optimizations = (type(active).__name__,) if active is not None and actual_count else ()
    return ExecutionReport(
        backend=BACKEND_NAME,
        device="cuda" if BACKEND_NAME == "cupy" else "cpu",
        dtype=str(getattr(engine, "dtype", "unknown")),
        acceleration_verified=bool(active is not None and actual_count > 0),
        active_optimizations=optimizations,
        fallbacks=tuple(reasons),
        notes=((f"CUDA Graph actually executed for {actual_count} tokens; numerical parity and speed remain separate checks",)
               if active is not None and actual_count else
               ("CUDA Graph captured but not executed yet",) if active is not None else
               ("No captured CUDA Graph observed; other GPU kernels may still be accelerated",)),
    )
