"""0102O Model B matrix pooling, mixed windows, causality, and integration."""

from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.ops.matrix_pool_dual_resolution_attention import (
    MatrixPoolDualResolutionAttention,
)


def _config(**updates):
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    values = dict(
        block_size=4, local_window=4, long_local_window=8,
        long_local_heads=1, summaries_per_block=4, global_heads=2,
        global_head_dim=4, max_blocks=8, global_query_chunk=4,
    )
    values.update(updates)
    return replace(base, **values)


def _attention(seed=1020, dtype="float64", **updates):
    return MatrixPoolDualResolutionAttention(
        d_model=16, n_q_heads=4, n_kv_heads=2, d_head=4,
        input_std=0.05, output_std=0.04, rng=RandomStream(seed),
        dtype=dtype, config=_config(**updates),
    )


def _tiny_model(seed=1021, dtype="float64"):
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    config = replace(
        base, context_length=12, n_layers=2, d_model=16, d_ff=32,
        n_q_heads=4, n_kv_heads=2, d_head=4, tokenizer_vocab_size=31,
        dual_resolution=_config(), dtype=dtype,
    )
    return config, DecoderLanguageModel(config, rng_seed=seed, dtype=dtype)


def test_new_preset_is_separate_and_parameter_count_matches_tiny_model():
    old = ModelConfig.dual_resolution_1024_20l()
    new = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    assert old.dual_resolution.compression_mode == "transformer"
    assert new.dual_resolution.compression_mode == "matrix_pool"
    assert old.estimated_parameter_count() == 406_199_296
    assert new.dual_resolution.summaries_per_block == 4
    assert new.dual_resolution.long_local_heads == 1
    assert new.dual_resolution.long_local_window == 4096
    assert new.dual_resolution.global_heads == 2
    config, model = _tiny_model()
    assert sum(int(p.size) for p in model.parameters()) == config.estimated_parameter_count()
    names = [p.name for p in model.parameters()]
    assert len(names) == len(set(names))
    assert any("matrix_pool.score" in name for name in names)
    assert not any("compress.layer_" in name for name in names)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy inference parity")
def test_incremental_inference_matches_full_forward_with_mixed_windows_and_summaries(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    config, training = _tiny_model(seed=1025)
    tokens = np.random.default_rng(1026).integers(
        0, config.vocab_size, size=(1, config.context_length), dtype=np.int32
    )
    reference, _ = training.forward(tokens)

    inference = InferenceModel(config, dtype="float64")
    inference.set_weights(training)
    state = inference.create_generation_state(1, config.context_length)
    for position in range(tokens.shape[1]):
        logits = inference.decode_one(tokens[:, position], state)
        np.testing.assert_allclose(
            logits, reference[:, position], rtol=3e-11, atol=3e-12
        )

    assert state.k_cache.shape[3] == config.dual_resolution.long_local_window
    assert state.dual_summary_k_cache.shape[2:] == (
        config.dual_resolution.global_heads,
        config.dual_resolution.max_blocks * config.dual_resolution.summaries_per_block,
        config.dual_resolution.global_head_dim,
    )
    assert int(state.dual_local_count[0, 0]) == config.dual_resolution.long_local_window
    assert int(state.dual_summary_count[0, 0]) == (
        (config.context_length // config.dual_resolution.block_size)
        * config.dual_resolution.summaries_per_block
    )


def test_matrix_pool_forward_backward_directional_derivatives(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("float64 directional derivative reference uses NumPy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    attention = _attention()
    rng = np.random.default_rng(1022)
    hidden = rng.normal(scale=0.2, size=(1, 8, 16))
    tokens = np.arange(8, dtype=np.int64)[None, :]
    metadata = attention._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4),
        1, 8, None,
    )
    summaries, cache = attention._pool_forward(hidden, metadata, True)
    cache["block_indices"] = metadata["block_indices"][cache["rows"], cache["slots"]]
    upstream = rng.normal(size=summaries.shape)
    attention.zero_grad()
    dx = attention._pool_backward(upstream, cache, hidden.shape, True)
    analytic_w = attention.W_pool_score.grad.copy()
    analytic_b = attention.pool_position.grad.copy()

    def objective():
        value, _ = attention._pool_forward(hidden, metadata, False)
        return float(np.sum(value * upstream))

    epsilon = 1e-5
    for array, analytic, coordinates in (
        (hidden, dx, [(0, 0, 0), (0, 5, 7)]),
        (attention.W_pool_score.data, analytic_w, [(0, 0), (7, 3)]),
        (attention.pool_position.data, analytic_b, [(0, 1), (3, 2)]),
    ):
        for coordinate in coordinates:
            old = array[coordinate]
            array[coordinate] = old + epsilon
            plus = objective()
            array[coordinate] = old - epsilon
            minus = objective()
            array[coordinate] = old
            numerical = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(analytic[coordinate], numerical,
                                       rtol=3e-4, atol=3e-5)
    probabilities = cache["probabilities"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference arrays")
def test_head_specific_local_windows_and_causality(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    attention = _attention(global_enabled=False)
    q = np.zeros((1, 8, 4, 4), dtype=np.float64)
    k = np.zeros((1, 8, 2, 4), dtype=np.float64)
    v = np.zeros_like(k)
    v[0, 0, 0, 0] = 8.0
    v[0, 7, 0, 1] = 8.0
    documents = np.zeros((1, 8), dtype=np.int64)
    output, _ = attention._local_forward(q, k, v, documents, False)
    assert output[0, 6, 0, 0] > 0  # long head includes key zero
    assert output[0, 6, 1, 0] == 0  # short head window starts at key three
    assert output[0, 6, 0, 1] == 0  # future key seven remains invisible


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference arrays")
def test_future_token_never_changes_earlier_output_and_summaries_receive_gradients(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    attention = _attention(seed=1023)
    rng = np.random.default_rng(1024)
    x = rng.normal(scale=0.1, size=(1, 8, 16))
    residual = rng.normal(scale=0.1, size=(1, 8, 16))
    tokens = np.arange(8, dtype=np.int64)[None, :]
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4)
    reference, cache = attention.forward(
        x, residual_input=residual, dual_resolution_metadata=metadata
    )
    changed_x = x.copy()
    changed_residual = residual.copy()
    changed_x[0, 3] += 2.0
    changed_residual[0, 3] -= 1.5
    changed = attention.forward(
        changed_x, residual_input=changed_residual,
        dual_resolution_metadata=metadata, return_cache=False,
    )
    np.testing.assert_array_equal(changed[:, :3], reference[:, :3])
    assert not np.allclose(changed[:, 4:], reference[:, 4:])

    attention.zero_grad()
    dy = rng.normal(size=reference.shape)
    dx_norm, dx_residual = attention.backward(dy, cache)
    assert dx_norm.shape == x.shape and dx_residual.shape == residual.shape
    assert np.all(np.isfinite(dx_norm)) and np.all(np.isfinite(dx_residual))
    assert np.all(np.linalg.norm(attention.W_pool_score.grad, axis=0) > 0)
    assert np.all(np.linalg.norm(attention.pool_position.grad, axis=0) > 0)
    for parameter in attention.parameters():
        assert np.all(np.isfinite(parameter.grad)), parameter.name


def test_eos_blocks_expand_to_four_summaries_without_cross_document_visibility(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    attention = _attention()
    tokens = np.array([[1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 10]])
    metadata = attention._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4),
        1, 12, None,
    )
    expanded = attention._summary_metadata(metadata)
    assert expanded["valid"].shape[1] == metadata["block_valid"].shape[1] * 4
    assert int(expanded["valid"].sum()) == int(metadata["block_valid"].sum()) * 4
    first_document = expanded["documents"][0, :4]
    second_document = expanded["documents"][0, 4:8]
    assert np.all(first_document == first_document[0])
    assert np.all(second_document == second_document[0])
    assert first_document[0] != second_document[0]
    allowed_at_second_document_start = (
        expanded["valid"][0]
        & (expanded["documents"][0] == metadata["document_ids"][0, 4])
        & (expanded["ends"][0] < metadata["position_ids"][0, 4])
    )
    assert not np.any(allowed_at_second_document_start)


def test_block_checkpoint_recompute_matches_retained_cache(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("deterministic checkpoint parity uses NumPy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    config, model = _tiny_model(seed=1025)
    tokens = (np.arange(12, dtype=np.int64)[None, :] % 30)
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=30, block_size=4)
    rng = np.random.default_rng(1026)
    upstream = rng.normal(size=(1, 12, 16))

    retained, cache = model.forward_body(
        tokens, activation_checkpoint=False, dual_resolution_metadata=metadata
    )
    model.zero_grad()
    model.backward_body(upstream.copy(), cache)
    retained_grads = [p.grad.copy() for p in model.parameters()]

    replayed, cache = model.forward_body(
        tokens, activation_checkpoint=True, dual_resolution_metadata=metadata
    )
    model.zero_grad()
    model.backward_body(upstream.copy(), cache)
    replayed_grads = [p.grad.copy() for p in model.parameters()]
    np.testing.assert_allclose(replayed, retained, rtol=1e-12, atol=1e-12)
    for parameter, got, expected in zip(model.parameters(), replayed_grads,
                                        retained_grads):
        np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-10,
                                   err_msg=parameter.name)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy and Triton")
def test_production_width_triton_mixed_window_and_global_parity(monkeypatch):
    import cupy as cp

    pytest.importorskip("triton")
    config = replace(
        ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution,
        max_blocks=4, global_query_chunk=128,
    )
    tokens = np.full((1, 513), 3, dtype=np.int32)
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(1027)
    x = cp.asarray(rng.normal(scale=0.03, size=(1, 513, 1024)).astype("float32")).astype("bfloat16")
    residual = cp.asarray(rng.normal(scale=0.03, size=x.shape).astype("float32"))
    dy = cp.asarray(rng.normal(scale=0.03, size=x.shape).astype("float32")).astype("bfloat16")

    def run(local_kernel, global_kernel):
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", local_kernel)
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", global_kernel)
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
        operator = MatrixPoolDualResolutionAttention(
            d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
            input_std=0.01, output_std=0.01, rng=RandomStream(1028),
            dtype="bfloat16", config=config,
        )
        output, cache = operator.forward(
            x, residual_input=residual, dual_resolution_metadata=metadata
        )
        operator.zero_grad()
        dx, dr = operator.backward(dy, cache)
        return output, dx, dr, [p.grad.copy() for p in operator.parameters()]

    reference = run("cupy", "cupy")
    triton = run("triton", "triton")
    for got, expected in zip(triton[:3], reference[:3]):
        cp.testing.assert_allclose(got.astype("float32"), expected.astype("float32"),
                                   rtol=0.12, atol=0.12)
    for got, expected in zip(triton[3], reference[3]):
        assert bool(cp.all(cp.isfinite(got)))
        cp.testing.assert_allclose(got, expected, rtol=0.22, atol=0.22)
