"""0105: frozen *expected* Model B training contract for regression runs.

This is a comparison profile, not a second training launcher.  The existing
train_model.py and ExtendedTrainer remain the authoritative production path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Exact configuration supplied for the active training run on 2026-11-01.
# Keep operational paths separate: never load/resume the active checkpoint in
# any regression script. Defaults are *not* assumed equivalent.
ENVIRONMENT = {
    "MINI_LLM_BACKEND": "cupy",
    "MINI_LLM_BF16_MIXED_CONTEXT": "0",
    "MINI_LLM_ACTIVATION_CHECKPOINT": "none",
    "MINI_LLM_MOE_RECOMPUTE_GATE_UP": "0",
    "MINI_LLM_OPTIMIZER_OFFLOAD": "full",
    "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB": "16",
    "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS": "0",
    "MINI_LLM_AR_COMPACT_TARGET": "0",
    "MINI_LLM_DUAL_RESOLUTION_CHECKPOINT_THRESHOLD": "0",
    "MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS": "0",
    "MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL": "triton",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL": "triton",
    "MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL": "triton",
    "MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET": "fine",
    "MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS": "1",
    "MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS": "1",
    "MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV": "1",
    "MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX": "1",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED": "1",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES": "0",
    "MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS": "0",
    "MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE": "1",
    "MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE": "1",
    "MINI_LLM_MATRIX_POOL_FUSED_FORWARD": "0",
    "MINI_LLM_MATRIX_POOL_FUSED_BACKWARD": "1",
    "MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE": "1",
    "MINI_LLM_FUSED_QKV_ROPE_FORWARD": "1",
    "MINI_LLM_ROPE_PREPARED_POSITIONS": "1",
    "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS": "1",
    "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK": "1536",
    "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "8192",
    "MINI_LLM_FUSED_EXPERT_GEMM": "0",
    "MINI_LLM_GPU_MEMORY_AUDIT": "0",
    "MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL": "0",
    "MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT": "1",
    "MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION": "1",
}

OPTIONS = {
    "model": "dual-resolution-matrix-pool-sparse-1024-20l",
    "objective": "ar",
    "precision": "bf16-mixed",
    "runtime-profile": "consumer-gpu",
    "mixed-shard-root": "./token_shards_ensemble_32k_mixed",
    "context-length": "8192",
    "batch-size": "1",
    "grad-accum-steps": "512",
    "gradient-accumulation-weighting": "supervised_tokens",
    "optimizer": "hybrid-muon",
    "muon-momentum": "0.95",
    "muon-ns-steps": "5",
    "muon-matmul-precision": "bf16",
    "total-steps": "100000",
    "warmup-steps": "750",
    "peak-lr": "3e-4",
    "grad-clip": "1.0",
    "weight-decay": "0.1",
    "val-interval": "100",
    "val-steps": "4",
    "val-per-source-steps": "1",
    "val-fixed-mixed-steps": "4",
    "val-fixed-mixed-seed": "78078",
    "save-interval": "50",
    "log-interval": "1",
    "seed": "42",
    "benchmark-suite": "./benchmarks/configs/foundation_capabilities_0097_progressive.json",
}

# Location-only arguments must NEVER be copied into regression invocation.
EXCLUDED_PATH_OPTIONS = frozenset({"checkpoint-dir", "log-file", "resume-from"})


def audit(environment: dict[str, str], options: dict[str, str]) -> dict:
    """Read-only comparison; no model import, device access, or checkpoint IO."""
    mismatches = {}
    for key, expected in ENVIRONMENT.items():
        actual = environment.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    for key, expected in OPTIONS.items():
        actual = options.get(key)
        if actual is None or (actual != expected and not _same_number(actual, expected)):
            mismatches["--" + key] = {"expected": expected, "actual": actual}
    return {"schema": "model-b-0105-profile-v1", "matches": not mismatches,
            "mismatches": mismatches, "checked_environment": len(ENVIRONMENT),
            "checked_options": len(OPTIONS)}


def _same_number(actual: str, expected: str) -> bool:
    try:
        return float(actual) == float(expected)
    except (ValueError, TypeError):
        return False


def parse_cli_options(argv: list[str]) -> dict[str, str]:
    options = {}
    i = 0
    while i < len(argv):
        part = argv[i]
        if part.startswith("--"):
            if "=" in part:
                key, value = part[2:].split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                key, value = part[2:], argv[i + 1]
                i += 1
            else:
                key, value = part[2:], "true"
            options[key] = value
        i += 1
    return options
