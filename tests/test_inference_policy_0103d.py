"""0103D policy, strictness and observed-path regression (no GPU required)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mini_llm.config import ModelConfig
from mini_llm.inference_policy import (
    configure_inference, is_model_b_graph_compatible, observe_execution,
)
from mini_llm.model_contracts import UnverifiedExecutionPath


@pytest.fixture(autouse=True)
def clean_runtime_env():
    # No irreversible os.environ mutations from runtime_defaults.
    import os
    snapshot = {k: v for k, v in os.environ.items() if k.startswith("MINI_LLM_")}
    for k in snapshot:
        os.environ.pop(k, None)
    yield
    for k in tuple(os.environ):
        if k.startswith("MINI_LLM_"):
            os.environ.pop(k, None)
    os.environ.update(snapshot)


def model_b():
    return replace(ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l(),
                   dtype="bfloat16")


def test_auto_gpu_activates_only_supported_model_b_graph():
    from os import environ
    cfg = model_b()
    result = configure_inference(cfg, backend="cupy", dtype="bfloat16")
    assert result.resolved == "fast"
    assert result.model_b_graph_eligible and result.graph_configured
    assert environ["MINI_LLM_NATIVE_CUDA_GRAPH_DECODE"] == "1"
    assert environ["MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL"] == "triton"
    assert environ["MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL"] == "triton"
    assert environ["MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL"] == "triton"
    assert "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE" in dict(result.changed)
    assert "MINI_LLM_OPTIMIZER_OFFLOAD" not in environ


def test_cpu_auto_is_reference_and_explicit_fast_is_not_silently_downgraded():
    from os import environ
    cfg = model_b()
    result = configure_inference(cfg, backend="numpy")
    assert result.resolved == "reference"
    assert result.changed == ()
    assert "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE" not in environ
    with pytest.raises(UnverifiedExecutionPath, match="CuPy"):
        configure_inference(cfg, backend="numpy", policy="fast")
    with pytest.raises(UnverifiedExecutionPath, match="CuPy"):
        configure_inference(cfg, backend="numpy", policy="strict-fast")


def test_sparse_stride_is_required_no_graph_on_model_b_without_sparse():
    from os import environ
    cfg = replace(ModelConfig.dual_resolution_matrix_pool_1024_20l(), dtype="bfloat16")
    assert not is_model_b_graph_compatible(cfg, backend="cupy")
    r = configure_inference(cfg, backend="cupy", policy="fast")
    assert not r.graph_requested
    assert "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE" not in environ
    with pytest.raises(UnverifiedExecutionPath, match="stride-16"):
        configure_inference(cfg, backend="cupy", policy="strict-fast")


def test_explicit_disabled_graph_never_counts_as_strict_fast(monkeypatch):
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0")
    with pytest.raises(UnverifiedExecutionPath, match="disabled"):
        configure_inference(model_b(), backend="cupy", policy="strict-fast")
    r = configure_inference(model_b(), backend="cupy", policy="fast")
    assert not r.graph_configured
    assert r.warnings


def test_captured_graph_must_be_executed_not_merely_enabled(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "numpy")  # observed runtime is actual imported backend
    from mini_llm.backend import BACKEND_NAME
    assert BACKEND_NAME == "numpy"
    state = SimpleNamespace(_native_cuda_graph=object(),
                            _native_cuda_graph_failure=None,
                            _cuda_graph_decode_count=0)
    engine = SimpleNamespace(dtype="bfloat16", dual_resolution_enabled=True)
    report = observe_execution(engine, state)
    assert not report.acceleration_verified
    with pytest.raises(UnverifiedExecutionPath):
        report.require_fast()
    state._cuda_graph_decode_count = 3
    report = observe_execution(engine, state)
    assert report.acceleration_verified
    assert report.active_optimizations == ("object",)
    report.require_fast()
    state._native_cuda_graph = None
    state._native_cuda_graph_failure = "CUDA OOM"
    report = observe_execution(engine, state)
    assert report.fallbacks == ("CUDA OOM",)
    with pytest.raises(UnverifiedExecutionPath):
        report.require_fast()


def test_reference_mode_refuses_inherited_graph_override(monkeypatch):
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1")
    with pytest.raises(UnverifiedExecutionPath, match="fresh process"):
        configure_inference(model_b(), backend="cupy", policy="reference")


def test_other_model_and_pretrained_never_get_model_b_graph_flag():
    from os import environ
    simple = ModelConfig(tokenizer_vocab_size=128, context_length=128,
                         n_layers=1, d_model=64, d_head=64, n_q_heads=1,
                         n_kv_heads=1, n_experts=1, top_k=1, d_ff=128,
                         ffn_mode="dense", dtype="float16")
    r = configure_inference(simple, backend="cupy", policy="fast")
    assert not r.graph_requested
    assert "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE" not in environ
    r = configure_inference(None, backend="cupy", policy="fast", model_format="gguf")
    assert not r.graph_requested
    assert "MINI_LLM_CUDA_GRAPH_DECODE" not in environ


def test_serving_model_spec_declares_policy(tmp_path):
    from mini_llm.model_sources import ModelSpec
    good = ModelSpec.from_dict({"id": "a", "format": "native", "path": "model",
                                "execution_policy": "strict-fast"}, tmp_path)
    assert good.execution_policy == "strict-fast"
    with pytest.raises(ValueError, match="execution_policy"):
        ModelSpec.from_dict({"id": "a", "format": "native", "path": "model",
                             "execution_policy": "unknown"}, tmp_path)


def test_public_model_generate_has_observed_not_inferred_execution():
    from mini_llm import Model
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    cfg = ModelConfig(tokenizer_vocab_size=128, context_length=128,
                      n_layers=1, d_model=32, d_head=8, n_q_heads=4,
                      n_kv_heads=2, n_experts=1, top_k=1,
                      d_ff=64, ffn_mode="dense", dtype="float32")
    tokenizer = SimpleBPETokenizer(vocab_size=128)
    tokenizer.train(["hello there", "small model"], vocab_size=128)
    model = Model.wrap_native(DecoderLanguageModel(cfg, rng_seed=19), tokenizer=tokenizer)
    assert not model.execution_info().acceleration_verified
    outcome = model.generate("hello", max_new_tokens=2)
    assert model.execution_info() == outcome.execution
    assert outcome.execution.backend == "numpy"
    assert not outcome.execution.acceleration_verified
    assert not outcome.execution.active_optimizations


def test_serving_adapter_reports_session_not_global_graph_state():
    from mini_llm.serving_models import ServingModel
    from mini_llm.backend import xp
    class FakeEngine:
        dtype = "float32"
        dual_resolution_enabled = False
        def prefill(self, ids, state):
            assert ids.shape == (1, 2)
            return xp.zeros((1, 4), dtype=xp.float32)
    model = ServingModel(FakeEngine(), tokenizer=None, context_length=4)
    state = SimpleNamespace()
    model.prefill([1, 2], state)
    assert state.execution_report.backend == "numpy"
    assert not state.execution_report.acceleration_verified
    strict = ServingModel(FakeEngine(), tokenizer=None, context_length=4,
                          execution_policy="strict-fast")
    # CUDA graphs may not execute until decode_one. Validate at completion,
    # not immediately after prefill, to avoid false strict-fast rejection.
    state = SimpleNamespace()
    strict.prefill([1, 2], state)
    with pytest.raises(UnverifiedExecutionPath):
        strict.execution_report(state).require_fast()
