import os

from mini_llm.config import ModelConfig
from mini_llm.runtime_defaults import apply_runtime_defaults


PROFILE_VARS = {
    "MINI_LLM_FUSED_EXPERT_GEMM",
    "MINI_LLM_FUSED_LOCAL_SOFTMAX",
    "MINI_LLM_BF16_LOCAL_GEMM",
    "MINI_LLM_FUSED_BF16_CE",
    "MINI_LLM_DIRECT_RETRIEVAL_SCATTER",
    "MINI_LLM_FUSED_SWIGLU",
    "MINI_LLM_FUSED_BF16_LOCAL_PIPELINE",
    "MINI_LLM_CONCURRENT_EXPERTS",
    "MINI_LLM_FUSED_INDEXED_SOFTMAX",
    "MINI_LLM_FUSED_BF16_INDEXED_PIPELINE",
    "MINI_LLM_DIRECT_QUERY_POOL",
    "MINI_LLM_DIRECT_QUERY_POOL_BACKWARD",
    "MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE",
    "MINI_LLM_FUSED_BF16_RMSNORM",
    "MINI_LLM_PACKED_QKV",
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
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS",
    "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL",
}


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
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "1536"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD"] == "full"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB"] == "16"
    assert os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "1024"
    assert os.environ["MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS"] == "1"


def test_explicit_override_wins_over_profile(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MINI_LLM_LOCAL_ATTN_QUERY_CHUNK", "768")
    monkeypatch.setenv("MINI_LLM_OPTIMIZER_OFFLOAD", "none")
    config = ModelConfig.wide_500m_memory_64k()
    apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=config, training=True
    )
    assert os.environ["MINI_LLM_LOCAL_ATTN_QUERY_CHUNK"] == "768"
    assert os.environ["MINI_LLM_OPTIMIZER_OFFLOAD"] == "none"


def test_auto_cupy_inference_uses_fast_not_training_memory_policy(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        None, backend_name="cupy", config=config, training=False
    )
    assert result.resolved == "gpu-fast"
    assert os.environ["MINI_LLM_PACKED_QKV"] == "1"
    assert "MINI_LLM_OPTIMIZER_OFFLOAD" not in os.environ
    assert "MINI_LLM_LM_HEAD_CHUNK_TOKENS" not in os.environ


def test_reference_profile_applies_nothing(monkeypatch):
    _clear(monkeypatch)
    config = ModelConfig.wide_500m_memory_64k()
    result = apply_runtime_defaults(
        "reference", backend_name="cupy", config=config, training=True
    )
    assert result.resolved == "reference"
    assert result.applied == ()
    assert "MINI_LLM_PACKED_QKV" not in os.environ
