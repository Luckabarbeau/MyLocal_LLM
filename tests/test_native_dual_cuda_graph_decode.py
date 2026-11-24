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


def test_model_b_native_graph_pools_the_shared_normalized_input(monkeypatch):
    """The captured graph must pool global_input, never residual + local_delta.

    Only a sequence that completes a block activates the global branch, so the
    logit-level comparison above (three tokens) cannot see this dependency at
    all.  RMSNorm output has an RMS set by the norm1 gain, while the post-local
    residual stream does not, so the pooling history's RMS separates the two
    branch topologies without depending on bf16 reduction order.  Before the
    parallel-branch fork was applied to this decoder the measured history RMS
    was ~0.19 against a gain scale of 1.0, versus ~1.00 afterwards.
    """
    cp = pytest.importorskip("cupy")
    from mini_llm.config import ModelConfig
    from mini_llm.inference_model import InferenceModel
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    config = replace(base, tokenizer_vocab_size=64)
    training = DecoderLanguageModel(config, rng_seed=1217, dtype="bfloat16")
    engine = InferenceModel(config, dtype="bfloat16")
    engine.set_weights(training)
    if not engine.blocks[0].attention.parallel_branches:
        pytest.skip("scale assertion describes the parallel default")

    length = int(config.dual_resolution.block_size) + 4
    prompt = cp.asarray([list(range(1, length + 1))], dtype=cp.int32)

    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0")
    reference = engine.create_generation_state(1, 1024)
    engine.prefill(prompt, reference)

    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1")
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT", "1")
    captured = engine.create_generation_state(1, 1024)
    engine.prefill(prompt, captured)
    assert captured._native_cuda_graph is not None
    assert captured._native_cuda_graph_failure is None
    cp.cuda.get_current_stream().synchronize()

    filled = int(cp.asnumpy(captured.dual_incomplete_count).min())
    assert filled == int(cp.asnumpy(reference.dual_incomplete_count).min()) > 0
    assert int(cp.asnumpy(captured.dual_summary_count).min()) > 0

    def row_rms(state):
        history = cp.asnumpy(state.dual_incomplete_hidden[:, 0, :filled, :]).astype(
            np.float32
        )
        return np.sqrt((history ** 2).mean(axis=-1))

    native, fallback = row_rms(captured), row_rms(reference)
    # Row-wise RMS is used deliberately: RMSNorm pins each row's magnitude, so
    # this statistic survives the bf16 reduction-order drift that dominates any
    # direction-sensitive comparison of the two paths over a long prompt, while
    # still separating a normalized input (RMS ~1) from residual + local_delta
    # (RMS ~0.19 here).  Measured native-vs-fallback drift is ~4e-4.
    np.testing.assert_allclose(native, fallback, rtol=0.02)
    # Absolute semantic pin: the pooling history must carry the norm1 gain scale
    # of the shared normalized block input rather than a residual-stream scale.
    gains = np.array([
        float(np.sqrt((cp.asnumpy(block.norm1.gamma.data).astype(np.float32) ** 2).mean()))
        for block in training.blocks
    ])
    np.testing.assert_allclose(
        native, np.broadcast_to(gains[:, None], native.shape), rtol=0.25,
    )


def test_model_b_native_graph_crosses_two_historical_block_boundaries(monkeypatch):
    """257 sequential tokens must cross two completed blocks inside one graph.

    The three-token comparison above never activates the historical-attention
    branch (global attention only starts at document position 128), so it
    cannot detect a decoder whose shared-input store, global-Q input or block
    counters desynchronize when a historical block completes.  A 4-token
    prefill plus 257 graph-replayed steps crosses the 128-row and 256-row
    block completions and keeps the global branch active at every watched
    position (127, 128, 129, 256, 257).

    Counters and document positions are discrete state: they must agree
    exactly.  The pooling history and logits are compared with the row-wise
    RMS statistics used above because bf16 reduction order is not preserved
    across paths; the measured native-vs-fallback drift over this sequence is
    ~0 (the graph replays the same kernels), so rtol=0.02 leaves the only
    realistic failure mode - a different branch topology - far outside tolerance.
    """
    cp = pytest.importorskip("cupy")
    from mini_llm.config import ModelConfig
    from mini_llm.inference_model import InferenceModel
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    monkeypatch.delenv("MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES", raising=False)
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    config = replace(base, tokenizer_vocab_size=64)
    # No skip-if-not-parallel: the canonical Model B preset IS parallel and the
    # CUDA-graph decoder must be correct under it.  A silent skip here would
    # hide the regression this test exists for.
    assert config.dual_resolution.attention_branch_mode == "parallel"

    training = DecoderLanguageModel(config, rng_seed=1217, dtype="bfloat16")
    engine = InferenceModel(config, dtype="bfloat16")
    engine.set_weights(training)
    assert engine.blocks[0].attention.parallel_branches
    assert engine.blocks[0].attention.attention_branch_mode == "parallel"
    block_size = int(config.dual_resolution.block_size)

    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0")
    reference = engine.create_generation_state(1, 1024)
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "1")
    monkeypatch.setenv("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT", "1")
    captured = engine.create_generation_state(1, 1024)

    prompt = cp.asarray([[1, 2, 3, 4]], dtype=cp.int32)
    engine.prefill(prompt, reference)
    engine.prefill(prompt, captured)
    graph = captured._native_cuda_graph
    assert graph is not None
    assert captured._native_cuda_graph_failure is None

    gains = np.array([
        float(np.sqrt((cp.asnumpy(block.norm1.gamma.data).astype(np.float32) ** 2).mean()))
        for block in training.blocks
    ])

    def row_rms(state):
        filled = int(cp.asnumpy(state.dual_incomplete_count).min())
        history = cp.asnumpy(state.dual_incomplete_hidden[:, 0, :filled, :]).astype(
            np.float32
        )
        return np.sqrt((history ** 2).mean(axis=-1))

    counter_names = (
        "dual_local_start", "dual_local_count", "dual_sparse_count",
        "dual_summary_start", "dual_summary_count", "dual_incomplete_count",
        "dual_document_positions",
    )
    completed_blocks = 0
    for step in range(1, 258):
        token = cp.asarray([[step]], dtype=cp.int32)
        expected = cp.asnumpy(engine.decode_one(token, reference)).astype(np.float32)
        actual = cp.asnumpy(engine.decode_one(token, captured)).astype(np.float32)
        cp.cuda.get_current_stream().synchronize()
        if step not in (127, 128, 129, 256, 257):
            continue
        # Discrete state must be bit-exact at the historical boundaries.
        for name in counter_names:
            assert bool(cp.array_equal(getattr(captured, name),
                                       getattr(reference, name)).item()), (
                f"{name} diverged at decode step {step}"
            )
        summaries = int(cp.asnumpy(captured.dual_summary_count).min())
        positions = int(cp.asnumpy(captured.dual_document_positions[0]).item())
        assert positions > block_size, "global branch must be active here"
        completed_blocks = max(completed_blocks, summaries // 4)
        # The history cache must carry the shared normalized input in both
        # paths, at the norm-gain scale (legacy topology would be >4x off).
        native, fallback = row_rms(captured), row_rms(reference)
        np.testing.assert_allclose(native, fallback, rtol=0.02)
        np.testing.assert_allclose(
            native, np.broadcast_to(gains[:, None], native.shape), rtol=0.25,
        )
        # Outputs: row-RMS parity catches gross topology divergence.
        np.testing.assert_allclose(
            np.sqrt((actual ** 2).mean(axis=-1)),
            np.sqrt((expected ** 2).mean(axis=-1)), rtol=0.02, atol=1e-3,
        )
    # Two complete 128-row blocks must have been pooled, and the same graph
    # must still own the state (no silent recapture under changed semantics).
    assert completed_blocks >= 2
    assert int(cp.asnumpy(captured.dual_summary_count).min()) == 8
    assert int(cp.asnumpy(captured.dual_incomplete_count).min()) == 5
    assert captured._native_cuda_graph is graph
    assert captured._native_cuda_graph_failure is None
