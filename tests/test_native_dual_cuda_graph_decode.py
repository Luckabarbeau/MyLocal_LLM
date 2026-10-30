"""GPU regression for the captured Model B dual-resolution decoder."""

from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME


pytestmark = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CUDA graph GPU test")


def test_model_b_native_graph_activates_and_matches_fallback(monkeypatch):
    cp = pytest.importorskip("cupy")
    from mini_llm.config import ModelConfig
    from mini_llm.inference_model import InferenceModel
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    config = replace(base, tokenizer_vocab_size=64)
    training = DecoderLanguageModel(config, rng_seed=1217, dtype="bfloat16")
    engine = InferenceModel(config, dtype="bfloat16")
    engine.set_weights(training)
    prompt = cp.asarray([[1, 2]], dtype=cp.int32)

    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0")
    reference = engine.create_generation_state(1, 256)
    engine.prefill(prompt, reference)

    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1")
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT", "1")
    captured = engine.create_generation_state(1, 256)
    engine.prefill(prompt, captured)
    assert captured._native_cuda_graph is not None
    assert captured._native_cuda_graph_failure is None
    from mini_llm.inference_policy import observe_execution
    execution = observe_execution(engine, captured)
    assert execution.acceleration_verified
    assert "NativeDualBF16CudaGraphDecoder" in execution.active_optimizations
    execution.require_fast()

    token = cp.asarray([[3]], dtype=cp.int32)
    expected = engine.decode_one(token, reference)
    actual = engine.decode_one(token, captured)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(actual).astype(np.float32),
        cp.asnumpy(expected).astype(np.float32),
        rtol=0.02,
        atol=0.15,
    )
    counter_names = (
        "dual_local_start", "dual_local_count", "dual_sparse_count",
        "dual_summary_start", "dual_summary_count", "dual_incomplete_count",
        "dual_document_positions",
    )
    for name in counter_names:
        assert bool(cp.array_equal(getattr(captured, name), getattr(reference, name)).item())

    graph = captured._native_cuda_graph
    captured.reset()
    reference.reset()
    second_prompt = cp.asarray([[4, 5, 6]], dtype=cp.int32)
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0")
    expected_prefill = engine.prefill(second_prompt, reference)
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1")
    fallback = engine._dual_incremental_token
    monkeypatch.setattr(
        engine, "_dual_incremental_token",
        lambda *args, **kwargs: pytest.fail("reused graph prefill called fallback"),
    )
    actual_prefill = engine.prefill(second_prompt, captured)
    monkeypatch.setattr(engine, "_dual_incremental_token", fallback)
    assert captured._native_cuda_graph is graph
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(actual_prefill).astype(np.float32),
        cp.asnumpy(expected_prefill).astype(np.float32),
        rtol=0.02,
        atol=0.15,
    )

    second_token = cp.asarray([[7]], dtype=cp.int32)
    expected = engine.decode_one(second_token, reference)
    actual = engine.decode_one(second_token, captured)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(actual).astype(np.float32),
        cp.asnumpy(expected).astype(np.float32),
        rtol=0.02,
        atol=0.15,
    )
    for name in counter_names:
        assert bool(cp.array_equal(getattr(captured, name), getattr(reference, name)).item())
