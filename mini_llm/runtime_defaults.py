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
    # The packed ExpertFFN GEMM was slower on the 32x512 dense model.
    # Retain the established default, including for dense FFNs.
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

_GPU_FAST_TRAINING_DEFAULTS: dict[str, str] = {
    # 0089C: fusing the attention branch promotion/residual add into RMSNorm2
    # reduced complete B3/T2304 block F+B latency in two interleaved A/B runs.
    "MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM": "1",
}

_CONSUMER_TRAINING_DEFAULTS: dict[str, str] = {
    "MINI_LLM_OPTIMIZER_OFFLOAD": "full",
    "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB": "16",
    "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "1024",
}

_MEMORY_MODEL_DEFAULTS: dict[str, str] = {
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS": "1",
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL": "100",
    # Skip unsupervised trailing target rows without changing sampled tokens,
    # loss weighting, or gradients. This exact AR path has dedicated parity tests.
    "MINI_LLM_AR_COMPACT_TARGET": "1",
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


def _use_fused_historical_decode(config, *, training: bool) -> bool:
    """Enable the measured BF16 one-token kernel only for its validated family."""
    if training or config is None:
        return False
    memory = getattr(config, "memory_context", None)
    return bool(
        memory is not None
        and getattr(memory, "enabled", False)
        and getattr(memory, "integration_mode", None) == "historical_read"
        and int(getattr(config, "d_head", 0)) == 64
        and str(getattr(config, "dtype", "")).lower() in {"bfloat16", "bf16"}
        and getattr(config, "attention_layers", None) is not None
    )


def _local_query_chunk_default(config) -> str | None:
    """VRAM-conscious query tiles; explicit user overrides take priority.

    0080BO increased 512d models from 512 to 1024 and triggered a batch-3
    backward OOM with retained local probabilities on a 16 GiB RTX 5080.
    Restore the previously working 512-row default for 512d models. The
    1024d historical-hybrid models use geometry-aware automatic chunking so
    full-current and sliding-window head groups receive different tiles.
    Keep the previously tuned 1536-row setting for other 768d+ models.
    """
    if config is None:
        return None
    try:
        width = int(config.d_model)
        active = int(config.context_length)
    except (AttributeError, TypeError, ValueError):
        return None
    if active < 4096:
        return None
    integration = getattr(
        getattr(config, "memory_context", None), "integration_mode", None
    )
    if width == 1024 and integration == "historical_read":
        return "auto"
    if width >= 768:
        return "1536"
    if width >= 512:
        return "512"
    return None


def _lm_head_chunk_default(
    config, *, training: bool, precision: str | None,
) -> str | None:
    """Return only LM-head tiles measured safe and faster for this model family."""
    if config is None or not training or str(precision).lower() != "bf16-mixed":
        return None
    try:
        width = int(config.d_model)
        active = int(config.context_length)
        vocab = int(config.vocab_size)
    except (AttributeError, TypeError, ValueError):
        return None
    integration = getattr(
        getattr(config, "memory_context", None), "integration_mode", None
    )
    if (
        width == 1024
        and active == 4096
        and vocab == 32768
        and integration == "historical_read"
    ):
        return "8192"
    return None


def format_runtime_fast_paths(config=None) -> str:
    """Report effective settings, not just the count applied by a profile."""
    names = (
        ("ffn_packed_gate_up", "MINI_LLM_FUSED_EXPERT_GEMM", "0"),
        ("packed_qkv", "MINI_LLM_PACKED_QKV", "0"),
        ("fused_sparse_decode", "MINI_LLM_FUSED_SPARSE_DECODE", "0"),
        ("grouped_local_gemm", "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "0"),
        ("bf16_local_pipeline", "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE", "0"),
        ("bf16_indexed_pipeline", "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE", "0"),
        ("fused_swiglu", "MINI_LLM_FUSED_SWIGLU", "0"),
        ("bf16_ce", "MINI_LLM_FUSED_BF16_CE", "0"),
        (
            "train_residual_rmsnorm",
            "MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM",
            "0",
        ),
        ("local_chunk", "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "512"),
        ("local_prob_recompute", "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS", "0"),
        ("lm_head_chunk", "MINI_LLM_LM_HEAD_CHUNK_TOKENS", "0"),
        ("optimizer_offload", "MINI_LLM_OPTIMIZER_OFFLOAD", "none"),
    )
    return (
        "GPU effective fast paths (" + str(getattr(config, "ffn_mode", "unknown"))
        + " FFN): " + ", ".join(
            f"{label}={os.environ.get(env, default)}"
            for label, env, default in names
        )
    )


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

    # Keep packed gate/up GEMM opt-in: the 32x512 dense A/B benchmark
    # measured a regression with this path enabled.
    _setdefaults(_GPU_FAST_DEFAULTS, applied)
    if training:
        _setdefaults(_GPU_FAST_TRAINING_DEFAULTS, applied)
    if _use_fused_historical_decode(config, training=training):
        _setdefaults({"MINI_LLM_FUSED_SPARSE_DECODE": "1"}, applied)

    chunk = _local_query_chunk_default(config)
    if chunk is not None:
        _setdefaults({"MINI_LLM_LOCAL_ATTN_QUERY_CHUNK": chunk}, applied)

    if _is_memory_model(config):
        _setdefaults(_MEMORY_MODEL_DEFAULTS, applied)

    lm_head_chunk = _lm_head_chunk_default(
        config, training=training, precision=precision,
    )
    if lm_head_chunk is not None:
        _setdefaults({"MINI_LLM_LM_HEAD_CHUNK_TOKENS": lm_head_chunk}, applied)

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
