"""Benchmark-only planning and capability helpers for PATCH 0086C.

This module deliberately has no production imports.  It describes the two
regular-GQA strategies without relabelling original query heads and probes an
optional cuDNN Frontend installation without making it a project dependency.
"""

from __future__ import annotations

import ctypes.util
import importlib
import importlib.metadata
import importlib.util
import sys
from typing import Any


N_Q_HEADS = 16
N_KV_HEADS = 4
D_HEAD = 64
ATTENTION_SCALE = 1.0 / 8.0
LOCAL_WINDOW = 512

_HEAD_PATTERNS = {
    "15": tuple(range(1, 16)),
    "13": tuple(h for h in range(1, 16) if h not in (12, 13)),
    "9": tuple(range(1, 10)),
    "scattered": (1, 3, 4, 7, 8, 10, 12, 15),
}

_DISTRIBUTIONS = (
    "nvidia-cudnn-frontend",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu13",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
)


class CudnnFrontendUnavailable(RuntimeError):
    """Raised when a real cuDNN Frontend SDPA experiment cannot be run."""


def resolve_active_heads(value: str | int) -> tuple[int, ...]:
    """Resolve a production local-head count or the scattered audit pattern."""
    key = str(value).strip().lower()
    if key not in _HEAD_PATTERNS:
        choices = ", ".join(_HEAD_PATTERNS)
        raise ValueError(f"unknown head pattern {value!r}; expected one of {choices}")
    return _HEAD_PATTERNS[key]


def regular16_plan(active_heads: tuple[int, ...] | list[int]) -> dict[str, Any]:
    """Describe the full-16-head local call with zero-dO dummy heads."""
    active = _validate_local_heads(active_heads)
    executed = tuple(range(N_Q_HEADS))
    return {
        "strategy": "cudnn_regular16",
        "executed_original_q_heads": list(executed),
        "active_original_q_heads": list(active),
        "dummy_original_q_heads": [head for head in executed if head not in active],
        "kv_map": [head // 4 for head in executed],
        "executed_q_head_count": len(executed),
        "active_q_head_count": len(active),
        "requires_zero_dout_for_dummy_heads": True,
        "requires_output_scatter": True,
        "preserves_original_head_numbers": True,
    }


def kv_split_plan(active_heads: tuple[int, ...] | list[int]) -> dict[str, Any]:
    """Describe at most four single-KV-head regular-GQA calls."""
    active = _validate_local_heads(active_heads)
    calls = []
    for kv_head in range(N_KV_HEADS):
        heads = tuple(head for head in active if head // 4 == kv_head)
        if not heads:
            continue
        calls.append(
            {
                "kv_head": kv_head,
                "original_q_heads": list(heads),
                "relative_kv_map": [0] * len(heads),
                "q_to_kv_ratio": len(heads),
                "requires_q_pack": not _is_contiguous(heads),
            }
        )
    return {
        "strategy": "cudnn_kv_split",
        "active_original_q_heads": list(active),
        "calls": calls,
        "dispatch_count": len(calls),
        "requires_output_scatter": True,
        "requires_dkv_accumulation": True,
        "preserves_original_head_numbers": True,
    }


def zero_mask_local_dout(d_out_all, active_heads, *, xp):
    """Return full-16-head dO with exact zeros for inactive/dummy local heads."""
    active = _validate_local_heads(active_heads)
    if d_out_all.ndim != 4 or int(d_out_all.shape[2]) != N_Q_HEADS:
        raise ValueError("d_out_all must have shape [B,T,16,D]")
    masked = xp.zeros_like(d_out_all)
    masked[:, :, active, :] = d_out_all[:, :, active, :]
    return masked


def logical_bhsd_metadata(array) -> dict[str, Any]:
    """Report the zero-copy logical BHSD view offered to an SDPA frontend."""
    if array.ndim != 4:
        raise ValueError("expected a BTHD tensor")
    view = array.transpose(0, 2, 1, 3)
    return {
        "physical_layout": "BTHD",
        "logical_layout": "BHSD",
        "shape": [int(value) for value in view.shape],
        "strides_bytes": [int(value) for value in view.strides],
        "d_contiguous": int(view.strides[-1]) == int(view.dtype.itemsize),
        "positive_strides": all(int(value) > 0 for value in view.strides),
        "transpose_copy_required_by_python": False,
    }


def probe_cudnn_environment() -> dict[str, Any]:
    """Probe import/package/API availability without constructing a graph."""
    distributions = {}
    for name in _DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None

    result: dict[str, Any] = {
        "python": sys.version,
        "cudnn_shared_library_from_loader": ctypes.util.find_library("cudnn"),
        "distributions": distributions,
        "cutlass_dsl_importable": importlib.util.find_spec("cutlass") is not None,
        "tvm_ffi_importable": importlib.util.find_spec("tvm_ffi") is not None,
        "frontend_importable": False,
        "pygraph_available": False,
        "sdpa_available": False,
        "sdpa_backward_available": False,
        "backend_version": None,
        "frontend_module": None,
        "frontend_version": None,
        "available": False,
        "reason": None,
    }
    try:
        cudnn = importlib.import_module("cudnn")
    except Exception as exc:  # capability reports must survive broken optional installs
        result["reason"] = (
            "cuDNN Frontend Python is not importable; real pygraph SDPA plan "
            f"construction cannot be attempted: {type(exc).__name__}: {exc}"
        )
        return result

    result["frontend_importable"] = True
    result["frontend_module"] = getattr(cudnn, "__file__", None)
    result["frontend_version"] = getattr(cudnn, "__version__", None)
    result["pygraph_available"] = hasattr(cudnn, "pygraph")
    if hasattr(cudnn, "backend_version"):
        try:
            backend_version = cudnn.backend_version()
        except TypeError:
            backend_version = cudnn.backend_version
        except Exception as exc:
            backend_version = f"unavailable: {type(exc).__name__}: {exc}"
        result["backend_version"] = str(backend_version)
    graph_type = getattr(cudnn, "pygraph", None)
    result["sdpa_available"] = bool(graph_type and hasattr(graph_type, "sdpa"))
    result["sdpa_backward_available"] = bool(
        graph_type and hasattr(graph_type, "sdpa_backward")
    )
    if not result["pygraph_available"]:
        result["reason"] = "import cudnn succeeded, but cudnn.pygraph is absent"
    elif not result["sdpa_available"] or not result["sdpa_backward_available"]:
        result["reason"] = (
            "cudnn.pygraph is present, but its type does not expose both sdpa "
            "and sdpa_backward"
        )
    else:
        result["reason"] = (
            "the frontend API is importable, but this patch was produced on a host "
            "without that API and therefore refuses to guess unvalidated build, "
            "variant-pack, stream, mask, or execution-plan signatures"
        )
    return result


def require_cudnn_frontend(capability: dict[str, Any] | None = None):
    """Return cudnn only when a validated adapter exists; otherwise fail exactly."""
    capability = probe_cudnn_environment() if capability is None else capability
    raise CudnnFrontendUnavailable(str(capability.get("reason") or "cuDNN unavailable"))


def _validate_local_heads(active_heads) -> tuple[int, ...]:
    active = tuple(int(head) for head in active_heads)
    if len(set(active)) != len(active):
        raise ValueError("active local heads must be unique")
    if any(head <= 0 or head >= N_Q_HEADS for head in active):
        raise ValueError("local heads must retain original IDs in [1,15]")
    return active


def _is_contiguous(heads: tuple[int, ...]) -> bool:
    if not heads:
        return True
    return heads == tuple(range(heads[0], heads[0] + len(heads)))
