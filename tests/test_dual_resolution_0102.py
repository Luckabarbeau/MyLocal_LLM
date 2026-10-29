"""Focused NumPy correctness contract for PATCH 0102 dual-resolution decoding."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import RandomStream
from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention,
    build_dual_resolution_metadata,
    dual_resolution_loss_mask,
)


def _dual_config(*, block_size=4, local_window=8, compression_dim=8,
                 compression_heads=2, compression_depth=1, max_blocks=8,
                 compression_block_chunk=64,
                 recompute_attention_probs=False):
    return DualResolutionConfig(
        enabled=True,
        block_size=block_size,
        local_window=local_window,
        compression_dim=compression_dim,
        compression_heads=compression_heads,
        compression_depth=compression_depth,
        compression_block_chunk=compression_block_chunk,
        max_blocks=max_blocks,
        global_query_chunk=3,
        recompute_attention_probs=recompute_attention_probs,
    )


def test_adaptive_checkpoint_policy_uses_physical_dual_resolution_length():
    trainer = ExtendedTrainer.__new__(ExtendedTrainer)
    trainer.activation_checkpoint = "block"
    trainer.dual_resolution_config = _dual_config()
    trainer.dual_resolution_checkpoint_threshold = 8

    assert not trainer._use_block_activation_checkpoint(np.zeros((1, 8)))
    assert trainer._use_block_activation_checkpoint(np.zeros((1, 9)))

    trainer.dual_resolution_config = None
    assert trainer._use_block_activation_checkpoint(np.zeros((1, 8)))
    trainer.activation_checkpoint = "none"
    assert not trainer._use_block_activation_checkpoint(np.zeros((1, 9)))


def _attention(seed=1, *, block_size=4, local_window=8, d_model=8,
               compression_dim=8, compression_depth=1, max_blocks=8,
               recompute_attention_probs=False, compression_block_chunk=64):
    return DualResolutionAttention(
        d_model=d_model,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=d_model // 2,
        input_std=0.08,
        output_std=0.08,
        rng=RandomStream(seed),
        dtype="float64",
        config=_dual_config(
            block_size=block_size,
            local_window=local_window,
            compression_dim=compression_dim,
            compression_heads=2,
            compression_depth=compression_depth,
            compression_block_chunk=compression_block_chunk,
            max_blocks=max_blocks,
            recompute_attention_probs=recompute_attention_probs,
        ),
    )


def _tiny_model_config(*, n_layers=2, max_blocks=8):
    return ModelConfig(
        tokenizer_vocab_size=31,
        context_length=min(16, 4 * max_blocks),
        n_layers=n_layers,
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        n_experts=1,
        top_k=1,
        d_ff=12,
        ffn_mode="dense",
        dtype="float64",
        dual_resolution=_dual_config(max_blocks=max_blocks),
    )


def _compressor_parameters(attention):
    names = (".compress.",)
    return [p for p in attention.parameters() if any(part in p.name for part in names)]


def _scalar_compressor_loss(attention, x, metadata, upstream):
    summaries, _ = attention._compress_forward(x, metadata)
    return float(np.sum(np.asarray(summaries) * upstream))


def _directional_difference(evaluate, array, direction, epsilon):
    array[...] += epsilon * direction
    plus = evaluate()
    array[...] -= 2.0 * epsilon * direction
    minus = evaluate()
    array[...] += epsilon * direction
    return (plus - minus) / (2.0 * epsilon)


def test_preset_dimensions_no_convolution_no_memory_and_all_twenty_dual_layers():
    config = ModelConfig.dual_resolution_1024_20l()
    assert (config.context_length, config.n_layers, config.d_model) == (65_536, 20, 1_024)
    assert (config.n_q_heads, config.n_kv_heads, config.d_head) == (16, 4, 64)
    assert config.ffn_mode == "dense"
    assert config.attention_layers is None
    assert config.full_history_layers == ()
    assert not config.memory_context.enabled
    assert config.dual_resolution == DualResolutionConfig(enabled=True)

    # Instantiate all twenty layers at tiny width: this verifies layer wiring
    # without allocating the production-size preset merely to inspect types.
    tiny = replace(
        config,
        tokenizer_vocab_size=31,
        context_length=8,
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        d_ff=12,
        dual_resolution=_dual_config(block_size=4, local_window=8),
        dtype="float64",
    )
    model = DecoderLanguageModel(tiny, rng_seed=2, dtype="float64")
    assert len(model.blocks) == 20
    assert all(isinstance(block.attention, DualResolutionAttention) for block in model.blocks)
    assert all("conv" not in type(module).__name__.lower() for module in model.blocks)


@pytest.mark.parametrize("width", [256, 512, 1024])
def test_parameter_estimate_matches_registered_arrays_at_target_widths(width):
    config = ModelConfig(
        tokenizer_vocab_size=64,
        context_length=128,
        n_layers=1,
        d_model=width,
        n_q_heads=width // 64,
        n_kv_heads=max(1, width // 256),
        d_head=64,
        n_experts=1,
        top_k=1,
        d_ff=64,
        ffn_mode="dense",
        dtype="float32",
        dual_resolution=DualResolutionConfig(
            enabled=True, block_size=128, local_window=128
        ),
    )
    model = DecoderLanguageModel(config, rng_seed=3, dtype="float32")
    registered = sum(parameter.size for parameter in model.parameters())
    assert registered == config.estimated_parameter_count()
    assert len({parameter.name for parameter in model.parameters()}) == len(model.parameters())


@pytest.mark.parametrize("length", [1, 64, 127, 128, 129, 255, 256, 257, 512, 513])
def test_metadata_variable_lengths_emit_only_complete_blocks(length):
    metadata = build_dual_resolution_metadata(
        np.arange(length, dtype=np.int64)[None, :], eos_token_id=-1, block_size=128
    )
    complete = length // 128
    assert metadata["document_ids"].shape == (1, length)
    assert metadata["position_ids"].shape == (1, length)
    assert metadata["block_indices"].shape == (1, complete, 128)
    assert metadata["block_valid"].shape == (1, complete)
    assert metadata["block_valid"].all()
    if complete:
        expected = np.arange(complete * 128).reshape(complete, 128)
        np.testing.assert_array_equal(metadata["block_indices"][0], expected)
        np.testing.assert_array_equal(
            metadata["block_end_positions"][0], np.arange(127, complete * 128, 128)
        )


def test_metadata_eos_boundaries_and_cross_document_loss_mask():
    tokens = np.array([[10, 11, 99, 20, 21, 22, 23, 99, 30, 31, 32, 33]], dtype=np.int64)
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4)
    np.testing.assert_array_equal(metadata["document_ids"][0], [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(metadata["position_ids"][0], [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    np.testing.assert_array_equal(metadata["block_indices"][0], [[3, 4, 5, 6], [8, 9, 10, 11]])
    np.testing.assert_array_equal(metadata["block_document_ids"][0], [1, 2])
    np.testing.assert_array_equal(metadata["block_end_positions"][0], [3, 3])
    np.testing.assert_array_equal(
        dual_resolution_loss_mask(tokens, 99),
        [[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]],
    )


def test_current_summary_excluded_and_exact_next_block_visibility():
    attention = _attention()
    q = np.zeros((1, 8, 2, 4), dtype=np.float64)
    summary_k = np.zeros((1, 2, 1, 4), dtype=np.float64)
    summary_v = np.array([[[[2.0, 0, 0, 0]], [[6.0, 0, 0, 0]]]])
    document_ids = np.zeros((1, 8), dtype=np.int64)
    positions = np.arange(8, dtype=np.int64)[None, :]
    output, available, caches = attention._global_forward(
        q, summary_k, summary_v, document_ids, positions,
        np.ones((1, 2), dtype=bool), np.zeros((1, 2), dtype=np.int64),
        np.array([[3, 7]], dtype=np.int64),
    )
    assert not available[0, :4].any()
    assert available[0, 4:].all()
    # At positions 4..7 only block zero is visible; block one is still current.
    np.testing.assert_allclose(output[0, 4:, :, 0], 2.0, atol=0, rtol=0)
    for cache in caches:
        qs, qe = cache["query_start"], cache["query_stop"]
        if qe > 4:
            assert np.all(np.asarray(cache["probabilities"])[..., max(0, 4 - qs):, 1] == 0)


def test_future_token_causality_including_inside_compression_block():
    attention = _attention(seed=4)
    rng = np.random.default_rng(5)
    original = rng.normal(size=(1, 8, 8))
    altered = original.copy()
    altered[:, 2] += 7.0  # Future to positions 0 and 1, inside compressed block zero.
    first, first_cache = attention.forward(original)
    second, second_cache = attention.forward(altered)
    np.testing.assert_allclose(first[:, :2], second[:, :2], atol=1e-12, rtol=1e-12)
    assert not np.array_equal(first_cache["summaries"][:, 0], second_cache["summaries"][:, 0])
    assert not np.array_equal(first[:, 4:], second[:, 4:])


def test_local_window_is_exact_and_document_bounded():
    attention = _attention(local_window=4)
    q = np.zeros((1, 7, 2, 4), dtype=np.float64)
    k = np.zeros((1, 7, 1, 4), dtype=np.float64)
    values = np.arange(1, 8, dtype=np.float64).reshape(1, 7, 1, 1)
    v = np.broadcast_to(values, (1, 7, 1, 4)).copy()
    documents = np.array([[0, 0, 0, 0, 0, 1, 1]], dtype=np.int64)
    output, _ = attention._local_forward(q, k, v, documents)
    np.testing.assert_allclose(output[0, 4, :, 0], np.mean([2, 3, 4, 5]))
    np.testing.assert_allclose(output[0, 5, :, 0], 6.0)
    np.testing.assert_allclose(output[0, 6, :, 0], 6.5)


def test_empty_global_path_is_bit_exact_local_behavior():
    attention = _attention(seed=6)
    x = np.random.default_rng(7).normal(size=(2, 3, 8))
    _, cache = attention.forward(x)
    assert cache["summaries"].shape == (2, 0, 8)
    assert not np.asarray(cache["global_available"]).any()
    assert np.array_equal(cache["merged"], cache["local_context"].reshape(2, 3, 8))
    assert np.array_equal(cache["effective_gate"], np.zeros((2, 3, 2)))


def test_learned_gate_controls_local_global_interpolation_per_head():
    attention = _attention(seed=8)
    x = np.ones((1, 8, 8), dtype=np.float64)
    attention.W_gate.data[...] = 0.0
    _, half = attention.forward(x)
    available = np.asarray(half["global_available"])
    assert available[0, 4:].all()
    np.testing.assert_allclose(half["gate"][0, 4:], 0.5, atol=0, rtol=0)
    np.testing.assert_allclose(
        half["merged"][0, 4:].reshape(4, 2, 4),
        (half["local_context"][0, 4:] + half["global_context"][0, 4:]) / 2,
        atol=1e-14,
    )
    attention.W_gate.data[...] = 0.0
    attention.W_gate.data[:, 0] = 2.0
    _, learned = attention.forward(x)
    assert np.all(learned["gate"][0, 4:, 0] > 0.99)
    np.testing.assert_allclose(learned["gate"][0, 4:, 1], 0.5, atol=0, rtol=0)
    assert not np.array_equal(learned["merged"][0, 4:], half["merged"][0, 4:])


def test_compressor_dimensions_full_self_attention_and_learned_pooling():
    attention = _attention(seed=9, compression_dim=12, compression_depth=2)
    assert attention.W_compress_in.data.shape == (8, 12)
    assert attention.W_compress_up.data.shape == (12, 8)
    assert attention.pool_query.data.shape == (2, 6)
    assert len(attention.compression_layers) == 2
    assert all(layer["Wq"].data.shape == (12, 12) for layer in attention.compression_layers)

    x = np.random.default_rng(10).normal(size=(1, 4, 8))
    metadata = attention._regular_metadata(1, 4, None)
    summaries, cache = attention._compress_forward(x, metadata)
    assert summaries.shape == (1, 1, 8)
    assert cache["layers"][0]["probabilities"].shape == (1, 2, 4, 4)
    assert np.all(np.asarray(cache["layers"][0]["probabilities"]) > 0)
    assert cache["pool_probabilities"].shape == (1, 2, 4)
    original = summaries.copy()
    attention.pool_query.data[...] += np.random.default_rng(11).normal(size=(2, 6))
    changed, _ = attention._compress_forward(x, metadata)
    assert not np.array_equal(original, changed)


def test_all_compressor_parameter_and_earlier_input_directional_gradients():
    attention = _attention(seed=12)
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1, 8, 8))
    metadata = attention._regular_metadata(1, 8, None)
    upstream = rng.normal(size=(1, 2, 8))
    summaries, cache = attention._compress_forward(x, metadata)
    assert summaries.shape == upstream.shape
    attention.zero_grad()
    dx = attention._compress_backward(upstream, cache, accumulate_parameter_grads=True)

    # Float64 central differences at 1e-3 are in the stable plateau: the 2e-3
    # estimate must independently agree, guarding against a lucky epsilon.
    for parameter in _compressor_parameters(attention):
        direction = rng.normal(size=parameter.data.shape)
        direction /= np.linalg.norm(direction)
        analytical = float(np.sum(parameter.grad * direction))
        evaluate = lambda: _scalar_compressor_loss(attention, x, metadata, upstream)
        numerical = _directional_difference(evaluate, parameter.data, direction, 1e-3)
        coarse = _directional_difference(evaluate, parameter.data, direction, 2e-3)
        assert np.isfinite(analytical) and abs(analytical) > 1e-11, parameter.name
        np.testing.assert_allclose(numerical, coarse, rtol=2e-5, atol=2e-8, err_msg=parameter.name)
        np.testing.assert_allclose(analytical, numerical, rtol=3e-5, atol=3e-8, err_msg=parameter.name)

    input_direction = np.zeros_like(x)
    input_direction[:, :4] = rng.normal(size=(1, 4, 8))
    input_direction /= np.linalg.norm(input_direction)
    analytical_input = float(np.sum(dx * input_direction))
    evaluate_input = lambda: _scalar_compressor_loss(attention, x, metadata, upstream)
    numerical_input = _directional_difference(evaluate_input, x, input_direction, 1e-3)
    coarse_input = _directional_difference(evaluate_input, x, input_direction, 2e-3)
    assert abs(analytical_input) > 1e-11
    np.testing.assert_allclose(numerical_input, coarse_input, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(analytical_input, numerical_input, rtol=3e-5, atol=3e-8)


def test_checkpointed_and_noncheckpointed_tiny_model_gradients_match():
    config = _tiny_model_config()
    regular = DecoderLanguageModel(config, rng_seed=14, dtype="float64")
    checkpointed = DecoderLanguageModel(config, rng_seed=14, dtype="float64")
    tokens = np.random.default_rng(15).integers(0, config.vocab_size, size=(1, 8), dtype=np.int64)
    direct, direct_cache = regular.forward_body(tokens, activation_checkpoint=False)
    replay, replay_cache = checkpointed.forward_body(tokens, activation_checkpoint=True)
    np.testing.assert_array_equal(replay, direct)
    upstream = np.random.default_rng(16).normal(size=direct.shape)
    regular.zero_grad()
    checkpointed.zero_grad()
    regular.backward_body(upstream, direct_cache)
    checkpointed.backward_body(upstream, replay_cache)
    for expected, actual in zip(regular.parameters(), checkpointed.parameters()):
        assert expected.name == actual.name
        np.testing.assert_allclose(actual.grad, expected.grad, rtol=2e-11, atol=2e-12, err_msg=expected.name)


def test_checkpoint_parameter_roundtrip(tmp_path):
    model = DecoderLanguageModel(_tiny_model_config(n_layers=1), rng_seed=17, dtype="float64")
    parameters = {parameter.name: parameter.data for parameter in model.parameters()}
    save_checkpoint(tmp_path / "dual", parameters)
    restored, _, _ = load_checkpoint(tmp_path / "dual", skip_optimizer=True)
    assert set(restored) == set(parameters)
    for name, value in parameters.items():
        np.testing.assert_array_equal(restored[name], value, err_msg=name)


def test_inference_full_prefill_token_decode_and_chunked_prefill_parity():
    config = _tiny_model_config(n_layers=2)
    training = DecoderLanguageModel(config, rng_seed=18, dtype="float64")
    tokens = np.random.default_rng(19).integers(0, config.vocab_size, size=(1, 11), dtype=np.int32)
    reference, _ = training.forward(tokens)

    full = InferenceModel(config, dtype="float64")
    tokenwise = InferenceModel(config, dtype="float64")
    chunked = InferenceModel(config, dtype="float64")
    for model in (full, tokenwise, chunked):
        model.set_weights(training)

    full_state = full.create_generation_state(1, 11)
    full_logits = full.prefill(tokens, full_state)
    np.testing.assert_allclose(full_logits, reference[:, -1], rtol=2e-11, atol=2e-12)

    token_state = tokenwise.create_generation_state(1, 11)
    for position in range(tokens.shape[1]):
        logits = tokenwise.decode_one(tokens[:, position], token_state)
        np.testing.assert_allclose(logits, reference[:, position], rtol=2e-11, atol=2e-12)

    chunk_state = chunked.create_generation_state(1, 11)
    chunked.prefill(tokens[:, :3], chunk_state)
    chunked_logits = chunked.prefill(tokens[:, 3:], chunk_state)
    np.testing.assert_allclose(chunked_logits, full_logits, rtol=2e-11, atol=2e-12)
    assert full_state.length == token_state.length == chunk_state.length == 11


def test_inference_summary_growth_local_bounds_and_eos_reset():
    config = _tiny_model_config(n_layers=1, max_blocks=2)
    training = DecoderLanguageModel(config, rng_seed=20, dtype="float64")
    inference = InferenceModel(config, dtype="float64")
    inference.set_weights(training)
    state = inference.create_generation_state(1, 16, eos_token_id=30)

    inference.prefill(np.array([[1, 2, 3]], dtype=np.int32), state)
    assert int(state.dual_summary_count[0, 0]) == 0
    inference.decode_one(np.array([4], dtype=np.int32), state)
    assert int(state.dual_summary_count[0, 0]) == 1
    assert int(state.dual_incomplete_count[0, 0]) == 0
    inference.prefill(np.array([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.int32), state)
    assert int(state.dual_summary_count[0, 0]) == 2  # bounded summary ring
    assert int(state.dual_local_count[0, 0]) == config.dual_resolution.local_window

    inference.decode_one(np.array([30], dtype=np.int32), state)
    assert bool(state.dual_reset_pending[0])
    assert int(state.dual_document_positions[0]) == 13
    inference.decode_one(np.array([13], dtype=np.int32), state)
    assert not bool(state.dual_reset_pending[0])
    assert int(state.dual_document_positions[0]) == 1
    assert int(state.dual_local_count[0, 0]) == 1
    assert int(state.dual_summary_count[0, 0]) == 0
    assert int(state.dual_incomplete_count[0, 0]) == 1


def test_checkpointed_attention_probabilities_match_retained_backward_with_eos():
    """Rematerialization must match both outputs and all parameter gradients."""
    metadata = build_dual_resolution_metadata(
        np.asarray([[1, 2, 3, 4, 5, 6, 7, 99, 9, 10, 11, 12],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
        eos_token_id=99, block_size=4,
    )
    x = np.random.default_rng(222).normal(size=(2, 12, 8)).astype(np.float64)
    dy = np.random.default_rng(223).normal(size=x.shape).astype(np.float64)
    results = []
    for recompute in (False, True):
        attention = _attention(seed=224, recompute_attention_probs=recompute)
        attention.zero_grad()
        y, cache = attention.forward(x, dual_resolution_metadata=metadata)
        for entry in cache["local_cache"] + cache["global_cache"]:
            assert (entry["probabilities"] is None) == recompute
        for entry in cache["compressor_cache"]["layers"]:
            assert (entry["probabilities"] is None) == recompute
        dx = attention.backward(dy, cache)
        grads = {param.name: np.array(param.grad, copy=True)
                 for param in attention.parameters()}
        results.append((y, dx, grads))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-12, rtol=1e-12)
    assert results[0][2].keys() == results[1][2].keys()
    for name in results[0][2]:
        np.testing.assert_allclose(results[0][2][name], results[1][2][name],
                                   atol=1e-12, rtol=1e-12, err_msg=name)


def test_rematerialization_forward_without_cache_and_full_model_backward():
    config = _tiny_model_config(n_layers=2)
    config = replace(config, dual_resolution=replace(
        config.dual_resolution, recompute_attention_probs=True))
    model = DecoderLanguageModel(config, rng_seed=225, dtype="float64")
    tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=30, block_size=4)
    logits, cache = model.forward(tokens, return_cache=True,
                                  dual_resolution_metadata=metadata)
    reference = model.forward(tokens, return_cache=False,
                              dual_resolution_metadata=metadata)
    np.testing.assert_allclose(logits, reference, atol=1e-11, rtol=1e-11)
    # Exercise the public model backward API against the new operator.
    grad = np.random.default_rng(226).normal(size=logits.shape)
    model.zero_grad()
    model.backward(grad, cache)
    global_parameters = [p for p in model.parameters() if ".global." in p.name]
    assert global_parameters
    assert all(np.all(np.isfinite(np.asarray(p.grad))) for p in model.parameters())
    assert any(np.any(np.asarray(p.grad) != 0) for p in global_parameters)


@pytest.mark.parametrize("recompute", [False, True])
def test_compressor_multiple_chunks_match_unchunked_forward_and_gradients(recompute):
    """Chunking is exact, including parameter and source-token gradients."""
    # 24 tokens / block_size 4 = six compression blocks (two chunks of 3).
    metadata = build_dual_resolution_metadata(
        np.arange(24, dtype=np.int64)[None, :], eos_token_id=99, block_size=4,
    )
    x = np.random.default_rng(240).normal(size=(1, 24, 8)).astype(np.float64)
    dy = np.random.default_rng(241).normal(size=x.shape).astype(np.float64)
    outcomes = []
    for chunk in (3, 64):
        attention = _attention(
            seed=242, recompute_attention_probs=recompute,
            compression_block_chunk=chunk,
        )
        attention.zero_grad()
        output, cache = attention.forward(x, dual_resolution_metadata=metadata)
        dx = attention.backward(dy, cache)
        grads = {p.name: p.grad.copy() for p in attention.parameters()}
        outcomes.append((output, dx, grads))
    np.testing.assert_allclose(outcomes[0][0], outcomes[1][0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(outcomes[0][1], outcomes[1][1], atol=1e-12, rtol=1e-12)
    for name in outcomes[0][2]:
        np.testing.assert_allclose(outcomes[0][2][name], outcomes[1][2][name],
                                   atol=1e-12, rtol=1e-12, err_msg=name)
