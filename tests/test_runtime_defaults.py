import os

import pytest

from mini_llm.config import ModelConfig
from mini_llm.runtime_defaults import (
    apply_runtime_defaults, format_runtime_fast_paths,
)


PROFILE_VARS = {
    "MINI_LLM_FUSED_EXPERT_GEMM",
    "MINI_LLM_FUSED_LOCAL_SOFTMAX",
    "MINI_LLM_BF16_LOCAL_GEMM",
    "MINI_LLM_FUSED_BF16_CE",
    "MINI_LLM_DIRECT_RETRIEVAL_SCATTER",
    "MINI_LLM_FUSED_SWIGLU",
    "MINI_LLM_SWIGLU_VEC2",
    "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE",
    "MINI_LLM_CONCURRENT_EXPERTS",
    "MINI_LLM_FUSED_INDEXED_SOFTMAX",
    "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE",
    "MINI_LLM_DIRECT_QUERY_POOL",
    "MINI_LLM_DIRECT_QUERY_POOL_BACKWARD",
    "MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE",
    "MINI_LLM_FUSED_BF16_RMSNORM",
    "MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM",
    "MINI_LLM_PACKED_QKV",
    "MINI_LLM_FUSED_QKV_ROPE_FORWARD",
    "MINI_LLM_FUSED_ROPE_BACKWARD",
    "MINI_LLM_FUSED_SPARSE_DECODE",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM",
    "MINI_LLM_LOCAL_CUBLAS_MODE",
    "MINI_LLM_CUBLAS_AUTOTUNE",
    "MINI_LLM_FUSED_LOCAL_MULTI_CHUNK_SOFTMAX",
    "MINI_LLM_CUBLAS_GROUPED_DILATED_GEMM",
    "MINI_LLM_FUSED_DILATED_MULTI_CHUNK_SOFTMAX",
    "MINI_LLM_INPLACE_BF16_CE",
    "MINI_LLM_COMPACT_PACKED_V",
    "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK",
    "MINI_LLM_OPTIMIZER_OFFLOAD",
    "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB",
    "MINI_LLM_LM_HEAD_CHUNK_TOKENS",
    "MINI_LLM_LM_HEAD_CACHE_LOGITS",
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS",
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL",
    "MINI_LLM_AR_COMPACT_TARGET",
    "MINI_LLM_ROPE_PREPARED_POSITIONS",
    "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD",
    "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS",
    "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS",
    "MINI_LLM_HISTORY_QUERY_CHUNK",
    "MINI_LLM_HISTORY_RECOMPUTE",
    "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS",
    "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD",
    "MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM",
    "MINI_LLM_BF16_MIXED_CONTEXT",
    "MINI_LLM_MOE_RECOMPUTE_GATE_UP",
    "MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS",
    "MINI_LLM_GPU_MEMORY_AUDIT",
    "MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT",
    "MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION",
    "MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL",
    "MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS",
    "MINI_LLM_DUAL_RESOLUTION_CHECKPOINT_THRESHOLD",
    "MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL",
    "MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL",
    "MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET",
    "MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS",
    "MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS",
    "MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV",
    "MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED",
    "MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES",
    "MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS",
    "MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING",
    "MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM",
    "MINI_LLM_DUAL_RESOLUTION_FUSED_GATE",
    "MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE",
    "MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE",
    "MINI_LLM_MATRIX_POOL_FUSED_FORWARD",
    "MINI_LLM_MATRIX_POOL_FUSED_BACKWARD",
    "MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE",
    "MINI_LLM_MATRIX_POOL_R4_LOGITS_BACKWARD",
}


@pytest.fixture(autouse=True)
def _restore_profile_environment():
    """Do not leak os.environ mutations from apply_runtime_defaults.

    apply_runtime_defaults writes via os.environ.setdefault, which pytest's
    monkeypatch cannot automatically undo. A later numerical test can otherwise
    accidentally enable Triton or GPU-specific kernels on the NumPy backend.
    Preserve the caller's complete MINI_LLM_* snapshot for this test only.
    """
    before = {key: value for key, value in os.environ.items()
              if key.startswith("MINI_LLM_")}
    yield
    for key in tuple(os.environ):
        if key.startswith("MINI_LLM_") and key not in before:
            del os.environ[key]
    os.environ.update(before)


def _clear(monkeypatch):
    for name in PROFILE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MINI_LLM_RUNTIME_PROFILE", raising=False)


def test_auto_numpy_keeps_reference_defaults(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        None, backend_name="numpy", config=config, training=True, precision="bf16-mixed"
    )
    assert result.resolved == "reference"
    assert result.applied == ()
    assert "MINI_LLM_OPTIMIZER_OFFLOAD" not in os.environ


def test_auto_large_cupy_training_uses_consumer_profile(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        None, backend_name="cupy", config=config, training=True, precision="bf16-mixed"
    )
    assert result.resolved == "consumer-gpu"
    assert os.environ["MINI_LLM_PACKED_QKV"] == "1"
    assert os.environ["MINI_LLM_FUSED_QKV_ROPE_FORWARD"] == "1"
    assert os.environ["MINI_LLM_FUSED_ROPE_BACKWARD"] == "1"
    assert os.environ["MINI_LLM_SWIGLU_VEC2"] == "1"
    assert os.environ["MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM"] == "1"
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "1536"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD"] == "full"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB"] == "16"
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "1024"
    assert os.environ["MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS"] == "1"
    assert os.environ["MINI_LLM_AR_COMPACT_TARGET"] == "1"


def test_dual_resolution_consumer_profile_uses_measured_lm_head_chunk(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    result = apply_runtime_defaults(
        None, backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    assert result.resolved == "consumer-gpu"
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "0"
    assert os.environ["MINI_LLM_FUSED_SWIGLU"] == "1"
    assert os.environ["MINI_LLM_SWIGLU_VEC2"] == "1"
    assert os.environ["MINI_LLM_FUSED_ROPE_BACKWARD"] == "1"
    assert "MINI_LLM_MOE_RECOMPUTE_GATE_UP" not in os.environ
    assert "MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS" not in os.environ
    assert os.environ["MINI_LLM_AR_ACCUM_LENGTH_BUCKETS"] == "0"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS"] == "0"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS"] == "0"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX"] == "1"
    assert os.environ["MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE"] == "1"
    assert os.environ["MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS"] == "1"
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "8192"
    assert os.environ["MINI_LLM_LM_HEAD_CACHE_LOGITS"] == "1"

    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "768")
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CACHE_LOGITS", "0")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "1")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS", "15")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", "0")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE", "0")
    apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "768"
    assert os.environ["MINI_LLM_LM_HEAD_CACHE_LOGITS"] == "0"
    assert os.environ["MINI_LLM_MOE_RECOMPUTE_GATE_UP"] == "1"
    assert os.environ["MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS"] == "15"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX"] == "0"
    assert os.environ["MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE"] == "0"


def test_explicit_override_wins_over_profile(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "768")
    monkeypatch.setenv("MINI_LLM_OPTIMIZER_OFFLOAD", "none")
    monkeypatch.setenv("MINI_LLM_SWIGLU_VEC2", "0")
    monkeypatch.setenv("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "0")
    config = ModelConfig.wide_500m_memory_64k()
    apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=config, training=True
    )
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "768"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD"] == "none"
    assert os.environ["MINI_LLM_SWIGLU_VEC2"] == "0"
    assert os.environ["MINI_LLM_FUSED_QKV_ROPE_FORWARD"] == "0"


def test_auto_cupy_inference_uses_fast_not_training_memory_policy(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        None, backend_name="cupy", config=config, training=False
    )
    assert result.resolved == "gpu-fast"
    assert os.environ["MINI_LLM_PACKED_QKV"] == "1"
    assert "MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM" not in os.environ
    assert "MINI_LLM_OPTIMIZER_OFFLOAD" not in os.environ
    assert "MINI_LLM_LM_HEAD_CHUNK_TOKENS" not in os.environ


def test_model_b_gpu_fast_inference_uses_forward_triton_defaults(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = dataclasses.replace(
        ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l(), dtype="bfloat16"
    )
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config, training=False,
    )
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL"] == "triton"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL"] == "triton"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL"] == "triton"
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS"] == "1"
    assert os.environ["MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE"] == "1"
    assert "MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX" not in os.environ
    assert "MINI_LLM_MATRIX_POOL_FUSED_BACKWARD" not in os.environ
    assert "MINI_LLM_LM_HEAD_CACHE_LOGITS" not in os.environ

    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config, training=False,
    )
    assert os.environ["MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL"] == "cupy"


def test_historical_bf16_inference_enables_measured_fused_decode(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = dataclasses.replace(
        ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k(),
        dtype="bfloat16",
    )
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config, training=False
    )
    assert os.environ["MINI_LLM_FUSED_SPARSE_DECODE"] == "1"

    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_FUSED_SPARSE_DECODE", "0")
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config, training=False
    )
    assert os.environ["MINI_LLM_FUSED_SPARSE_DECODE"] == "0"

    _clear(monkeypatch)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config, training=True
    )
    assert "MINI_LLM_FUSED_SPARSE_DECODE" not in os.environ


def test_reference_profile_applies_nothing(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        "reference", backend_name="cupy", config=config, training=True
    )
    assert result.resolved == "reference"
    assert result.applied == ()
    assert "MINI_LLM_PACKED_QKV" not in os.environ


def test_dense_gpu_fast_uses_safe_gemm_and_512d_tile(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.ensemble_deep_512_dense_memory_64k()
    # train_model replaces context_length with the compact active length
    # before applying the runtime profile; mirror that handoff here.
    import dataclasses
    config = dataclasses.replace(
        config, context_length=config.memory_context.active_length
    )
    result = apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    assert result.resolved == "gpu-fast"
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "0"
    assert os.environ["MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM"] == "1"
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "512"
    assert os.environ["MINI_LLM_AR_COMPACT_TARGET"] == "1"
    assert "MINI_LLM_OPTIMIZER_OFFLOAD" not in os.environ
    summary = format_runtime_fast_paths(config)
    assert "ffn_packed_gate_up=0" in summary
    assert "train_residual_rmsnorm=1" in summary
    assert "local_chunk=512" in summary


def test_dense_gpu_fast_preserves_explicit_gemm_and_chunk_overrides(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", "1")
    monkeypatch.setenv("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "1024")
    import dataclasses
    config = ModelConfig.ensemble_deep_512_dense_memory_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults("gpu-fast", backend_name="cupy", config=config, training=True)
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "1"
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "1024"


def test_moe_gpu_fast_retains_legacy_expert_setting(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    apply_runtime_defaults("gpu-fast", backend_name="cupy", config=config, training=True)
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "0"


def test_768_dense_gets_prior_measured_query_chunk(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = ModelConfig.ensemble_wide_768_dense_memory_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults("gpu-fast", backend_name="cupy", config=config, training=True)
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "0"
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "1536"


def test_1024_dense_gets_measured_query_chunk(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    assert os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] == "0"
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "auto"
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "8192"


def test_1024_dense_preserves_explicit_lm_head_chunk(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "2048")
    import dataclasses
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="bf16-mixed",
    )
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "2048"


def test_1024_dense_unmeasured_precision_has_no_lm_head_default(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=config,
        training=True, precision="float32",
    )
    assert "MINI_LLM_LM_HEAD_CHUNK_TOKENS" not in os.environ


def test_unmeasured_1024_memory_model_keeps_prior_query_chunk(monkeypatch):
    _clear(monkeypatch)
    import dataclasses
    config = ModelConfig.wide24_500m_memory_64k()
    config = dataclasses.replace(config, context_length=config.memory_context.active_length)
    apply_runtime_defaults("gpu-fast", backend_name="cupy", config=config, training=True)
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "1536"
    assert "MINI_LLM_LM_HEAD_CHUNK_TOKENS" not in os.environ


def _bf16_historical_config():
    import dataclasses
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    return dataclasses.replace(
        config, dtype="bfloat16", context_length=config.memory_context.active_length,
    )


def test_explicit_gpu_fast_enables_0089t_historical_family(monkeypatch):
    _clear(monkeypatch)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=_bf16_historical_config(),
        training=True, precision="bf16-mixed",
    )
    assert os.environ["MINI_LLM_HISTORY_QUERY_CHUNK"] == "1024"
    assert os.environ["MINI_LLM_HISTORY_RECOMPUTE"] == "1"
    assert os.environ["MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS"] == "1"
    assert os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD"] == "1"
    assert os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM"] == "1"
    assert os.environ["MINI_LLM_ROPE_PREPARED_POSITIONS"] == "1"
    assert os.environ["MINI_LLM_AR_ACCUM_LENGTH_BUCKETS"] == "1"


def test_auto_preserves_opt_in_historical_triton_boundary(monkeypatch):
    _clear(monkeypatch)
    apply_runtime_defaults(
        None, backend_name="cupy", config=_bf16_historical_config(),
        training=True, precision="bf16-mixed",
    )
    assert "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD" not in os.environ
    assert os.environ["MINI_LLM_HISTORY_QUERY_CHUNK"] == "1024"


def test_historical_fast_path_respects_overrides(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD", "0")
    monkeypatch.setenv("MINI_LLM_HISTORY_QUERY_CHUNK", "512")
    monkeypatch.setenv("MINI_LLM_HISTORY_RECOMPUTE", "0")
    apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=_bf16_historical_config(),
        training=True, precision="bf16-mixed",
    )
    assert os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD"] == "0"
    assert os.environ["MINI_LLM_HISTORY_QUERY_CHUNK"] == "512"
    assert os.environ["MINI_LLM_HISTORY_RECOMPUTE"] == "0"


def test_historical_triton_is_not_global_or_wrong_precision(monkeypatch):
    _clear(monkeypatch)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=ModelConfig.wide_500m_memory_64k(),
        training=True, precision="bf16-mixed",
    )
    assert "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD" not in os.environ
    _clear(monkeypatch)
    apply_runtime_defaults(
        "gpu-fast", backend_name="cupy", config=_bf16_historical_config(),
        training=True, precision="float32",
    )
    assert "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD" not in os.environ
    _clear(monkeypatch)
    apply_runtime_defaults(
        "gpu-fast", backend_name="numpy", config=_bf16_historical_config(),
        training=True, precision="bf16-mixed",
    )
    assert "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD" not in os.environ
