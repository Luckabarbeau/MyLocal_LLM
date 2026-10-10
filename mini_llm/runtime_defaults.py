"""Central runtime profiles for the command-line training/inference entrypoints.

The low-level kernels intentionally continue to expose individual environment
variables for research and A/B testing.  Normal users should not have to know
all of them.  This module applies a small set of validated defaults with
``os.environ.setdefault`` so every explicit user override still wins.

Profiles
--------
``reference``
    Apply no performance defaults.  Low-level module defaults are used.
``gpu-fast``
    Enable the validated CuPy/BF16 fast paths without changing optimizer
    placement or LM-head memory policy.
``consumer-gpu``
    ``gpu-fast`` plus the memory-saving defaults used by the current wide
    ~500M local-training configuration (optimizer offload + chunked LM head).
``auto``
    NumPy -> reference.  CuPy -> gpu-fast, upgraded to consumer-gpu for large
    training configurations where local VRAM pressure is expected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


PROFILE_CHOICES = ("auto", "reference", "gpu-fast", "consumer-gpu")
_FALSE = {"0", "false", "off", "no", ""}


# These are performance choices that have been validated together on the
# project's current CuPy/BF16 path.  Strict validator flags are deliberately
# excluded: unsupported layouts should fall back instead of making the default
# profile brittle on another GPU.
_GPU_FAST_DEFAULTS: dict[str, str] = {
    "MINI_LLM_FUSED_EXPERT_GEMM": "0",
    "MINI_LLM_FUSED_LOCAL_SOFTMAX": "1",
    "MINI_LLM_BF16_LOCAL_GEMM": "1",
    "MINI_LLM_FUSED_BF16_CE": "1",
    "MINI_LLM_DIRECT_RETRIEVAL_SCATTER": "1",
    "MINI_LLM_FUSED_SWIGLU": "1",
    "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE": "1",
    "MINI_LLM_CONCURRENT_EXPERTS": "0",
    "MINI_LLM_FUSED_INDEXED_SOFTMAX": "1",
    "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE": "1",
    "MINI_LLM_DIRECT_QUERY_POOL": "1",
    "MINI_LLM_DIRECT_QUERY_POOL_BACKWARD": "1",
    "MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE": "1",
    "MINI_LLM_FUSED_BF16_RMSNORM": "1",
    "MINI_LLM_PACKED_QKV": "1",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM": "1",
    "MINI_LLM_LOCAL_CUBLAS_MODE": "grouped",
    "MINI_LLM_CUBLAS_AUTOTUNE": "0",
    "MINI_LLM_FUSED_LOCAL_MULTI_CHUNK_SOFTMAX": "1",
    "MINI_LLM_CUBLAS_GROUPED_DILATED_GEMM": "1",
    "MINI_LLM_FUSED_DILATED_MULTI_CHUNK_SOFTMAX": "1",
    "MINI_LLM_INPLACE_BF16_CE": "1",
    "MINI_LLM_COMPACT_PACKED_V": "1",
}

_CONSUMER_TRAINING_DEFAULTS: dict[str, str] = {
    "MINI_LLM_OPTIMIZER_OFFLOAD": "full",
    "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB": "16",
    "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "1024",
}

_MEMORY_MODEL_DEFAULTS: dict[str, str] = {
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS": "1",
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL": "100",
}


@dataclass(frozen=True)
class RuntimeProfileResult:
    requested: str
    resolved: str
    applied: tuple[tuple[str, str], ...]


def _normalize_profile(value: str | None) -> str:
    raw = value
    if raw is None:
        raw = os.environ.get("MINI_LLM_RUNTIME_PROFILE", "auto")
    profile = str(raw).strip().lower().replace("_", "-")
    aliases = {
        "none": "reference",
        "off": "reference",
        "debug": "reference",
        "gpu": "gpu-fast",
        "fast": "gpu-fast",
        "consumer": "consumer-gpu",
        "local": "consumer-gpu",
    }
    profile = aliases.get(profile, profile)
    if profile not in PROFILE_CHOICES:
        choices = ", ".join(PROFILE_CHOICES)
        raise ValueError(f"runtime profile must be one of: {choices}")
    return profile


def _is_large_local_model(config) -> bool:
    if config is None:
        return False
    try:
        params = int(config.estimated_parameter_count())
    except Exception:
        params = 0
    return params >= 300_000_000


def _is_memory_model(config) -> bool:
    memory = getattr(config, "memory_context", None)
    return bool(memory is not None and getattr(memory, "enabled", False))


def _is_wide_long_context(config) -> bool:
    if config is None:
        return False
    try:
        return int(config.d_model) >= 768 and int(config.context_length) >= 4096
    except Exception:
        return False


def _setdefaults(values: Mapping[str, str], applied: list[tuple[str, str]]) -> None:
    for name, value in values.items():
        if name not in os.environ:
            os.environ[name] = str(value)
            applied.append((name, str(value)))


def apply_runtime_defaults(
    requested: str | None,
    *,
    backend_name: str,
    config=None,
    training: bool,
    precision: str | None = None,
) -> RuntimeProfileResult:
    """Apply user-overridable runtime defaults and return what changed.

    The backend itself is intentionally *not* selected here because
    ``mini_llm.backend`` is imported before argument parsing.  Users select the
    array backend with ``MINI_LLM_BACKEND=cupy`` or ``numpy``.
    """

    requested_profile = _normalize_profile(requested)
    resolved = requested_profile

    if requested_profile == "auto":
        if str(backend_name).lower() != "cupy":
            resolved = "reference"
        elif training and _is_large_local_model(config):
            resolved = "consumer-gpu"
        else:
            resolved = "gpu-fast"

    applied: list[tuple[str, str]] = []
    if resolved == "reference" or str(backend_name).lower() != "cupy":
        return RuntimeProfileResult(requested_profile, resolved, tuple(applied))

    _setdefaults(_GPU_FAST_DEFAULTS, applied)

    # 1536 was the best measured local-attention query tile for the current
    # 768d / 4k+ active family.  Keep smaller models on their module defaults.
    if _is_wide_long_context(config):
        _setdefaults({"MINI_LLM_LOCAL_ATTN_QUERY_CHUNK": "1536"}, applied)

    if _is_memory_model(config):
        _setdefaults(_MEMORY_MODEL_DEFAULTS, applied)

    if resolved == "consumer-gpu" and training:
        _setdefaults(_CONSUMER_TRAINING_DEFAULTS, applied)

    return RuntimeProfileResult(requested_profile, resolved, tuple(applied))


def format_runtime_profile(result: RuntimeProfileResult) -> str:
    if result.applied:
        return (
            f"Runtime profile: {result.resolved} "
            f"({len(result.applied)} defaults applied; explicit environment overrides win)"
        )
    return f"Runtime profile: {result.resolved} (no implicit overrides applied)"
