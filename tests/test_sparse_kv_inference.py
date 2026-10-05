import numpy as np

from mini_llm.backend import xp, asnumpy
from mini_llm.config import (
    ModelConfig,
    AttentionLayerConfig,
    LocalAttentionConfig,
    DilatedAttentionConfig,
    GlobalSparseAttentionConfig,
    RetrievalAttentionConfig,
    ContextRouterConfig,
    MemoryContextConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel


def _tiny_sparse_config(context_length=16):
    router = ContextRouterConfig(
        history_block_size=2,
        routing_stride=2,
        query_window=3,
        router_dim=4,
        top_k_blocks=1,
        exclude_recent_tokens=2,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=1,
        router_weight_mode="logit_bias",
    )
    layer = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=4),
            DilatedAttentionConfig(window=8, dilation=2, offset=0),
            GlobalSparseAttentionConfig(stride=3, offset=0, include_current=True),
            RetrievalAttentionConfig(router, group="far"),
        )
    )
    return ModelConfig(
        tokenizer_vocab_size=64,
        context_length=context_length,
        n_layers=2,
        d_model=32,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=8,
        n_experts=2,
        top_k=1,
        d_ff=64,
        attention_layers=(layer, layer),
        dtype="float32",
    )


def _models(context_length=16):
    cfg = _tiny_sparse_config(context_length)
    training = DecoderLanguageModel(cfg, rng_seed=123, dtype="float32")
    inference = InferenceModel(cfg, dtype="float32")
    inference.set_weights(training)
    return training, inference


def test_sparse_prefill_matches_full_sequence():
    training, inference = _models()
    ids = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=xp.int32)
    reference = training.forward(ids, return_cache=False)[0, -1]
    state = inference.create_generation_state(batch_size=1, max_length=16)
    cached = inference.prefill(ids, state)[0]
    np.testing.assert_allclose(asnumpy(cached), asnumpy(reference), rtol=1e-5, atol=1e-6)


def test_sparse_decode_matches_full_sequence_with_active_retrieval():
    training, inference = _models()
    sequence = [1, 2, 3, 4]
    state = inference.create_generation_state(batch_size=1, max_length=24)
    inference.prefill(xp.asarray([sequence], dtype=xp.int32), state)

    # Retrieval becomes eligible in this tiny geometry, so this exercises the
    # route cache in addition to local/dilated/global static sparse heads.
    for token in [5, 6, 7, 8, 9, 10, 11, 12]:
        sequence.append(token)
        cached = inference.decode_one(xp.asarray([[token]], dtype=xp.int32), state)[0]
        reference = training.forward(
            xp.asarray([sequence], dtype=xp.int32), return_cache=False
        )[0, -1]
        np.testing.assert_allclose(
            asnumpy(cached), asnumpy(reference), rtol=2e-5, atol=2e-6
        )


def test_sparse_cache_can_extend_beyond_checkpoint_training_window():
    training, inference = _models(context_length=8)
    sequence = [1, 2, 3, 4]
    state = inference.create_generation_state(batch_size=1, max_length=24)
    inference.prefill(xp.asarray([sequence], dtype=xp.int32), state)
    for token in [5, 6, 7, 8, 9, 10, 11, 12]:
        sequence.append(token)
        cached = inference.decode_one(xp.asarray([[token]], dtype=xp.int32), state)[0]
    reference = training.forward(
        xp.asarray([sequence], dtype=xp.int32), return_cache=False
    )[0, -1]
    np.testing.assert_allclose(asnumpy(cached), asnumpy(reference), rtol=2e-5, atol=2e-6)
    assert state.length == len(sequence)
    assert state.max_length == 24


def test_low_precision_cached_residual_policy_matches_training_model():
    # 0059A also fixes an older inference discrepancy: training keeps the
    # low-precision residual stream in FP32.  Cached inference must do the same.
    cfg = ModelConfig.tiny_inspection()
    training = DecoderLanguageModel(cfg, rng_seed=42, dtype="float16")
    inference = InferenceModel(cfg, dtype="float16")
    inference.set_weights(training)
    prompt = xp.asarray([[1, 2, 3, 4]], dtype=xp.int32)
    state = inference.create_generation_state(batch_size=1, max_length=16)
    cached = inference.prefill(prompt, state)[0]
    reference = training.forward(prompt, return_cache=False)[0, -1]
    np.testing.assert_allclose(
        asnumpy(cached).astype(np.float32),
        asnumpy(reference).astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    cached = inference.decode_one(xp.asarray([[5]], dtype=xp.int32), state)[0]
    reference = training.forward(
        xp.asarray([[1, 2, 3, 4, 5]], dtype=xp.int32), return_cache=False
    )[0, -1]
    np.testing.assert_allclose(
        asnumpy(cached).astype(np.float32),
        asnumpy(reference).astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )


def test_terminal_landmark_checkpoint_allows_short_history_kv_fallback():
    base = _tiny_sparse_config(context_length=16)
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=32,
        recent_length=16,
        target_length=16,
        block_size=2,
        top_k_blocks=1,
        router_query_length=16,
        router_dim=4,
        integration_mode="terminal_landmark",
        memory_attention_layer=-1,
        read_heads=1,
        read_kv_heads=1,
    )
    cfg = ModelConfig(
        tokenizer_vocab_size=base.tokenizer_vocab_size,
        context_length=base.context_length,
        n_layers=base.n_layers,
        d_model=base.d_model,
        n_q_heads=base.n_q_heads,
        n_kv_heads=base.n_kv_heads,
        d_head=base.d_head,
        n_experts=base.n_experts,
        top_k=base.top_k,
        d_ff=base.d_ff,
        attention_layers=base.attention_layers,
        dtype=base.dtype,
        memory_context=memory,
    )
    training = DecoderLanguageModel(cfg, rng_seed=123, dtype="float32")
    inference = InferenceModel(cfg, dtype="float32")
    inference.set_weights(training)
    state = inference.create_generation_state(batch_size=1, max_length=16)
    assert state.max_length == 16

    long_state = inference.create_generation_state(batch_size=1, max_length=17)
    assert long_state.long_memory_enabled
    assert long_state.working_capacity == 16
    assert long_state.k_cache.shape[3] == 16


def _tiny_terminal_long_inference_config():
    router = ContextRouterConfig(
        history_block_size=2,
        routing_stride=2,
        query_window=2,
        router_dim=4,
        top_k_blocks=1,
        exclude_recent_tokens=2,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=1,
        router_weight_mode="logit_bias",
    )
    layer = AttentionLayerConfig(
        heads=(RetrievalAttentionConfig(router, group="far"),)
    )
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=16,
        recent_length=8,
        target_length=8,
        block_size=2,
        top_k_blocks=1,
        router_query_length=8,
        router_dim=4,
        query_pooling="mean",
        history_pooling="mean",
        integration_mode="terminal_landmark",
        memory_attention_layer=-1,
        read_heads=1,
        read_kv_heads=1,
    )
    return ModelConfig(
        tokenizer_vocab_size=64,
        context_length=8,
        n_layers=1,
        d_model=64,
        n_q_heads=1,
        n_kv_heads=1,
        d_head=64,
        n_experts=2,
        top_k=1,
        d_ff=128,
        attention_layers=(layer,),
        dtype="float32",
        memory_context=memory,
    )


def test_terminal_landmark_long_prefill_matches_training_terminal_prediction():
    cfg = _tiny_terminal_long_inference_config()
    training = DecoderLanguageModel(cfg, rng_seed=321, dtype="float32")
    inference = InferenceModel(cfg, dtype="float32")
    inference.set_weights(training)
    ids = xp.asarray([list(range(1, 17))], dtype=xp.int32)

    reference = training.forward(ids, return_cache=False)[0, -1]
    state = inference.create_generation_state(batch_size=1, max_length=16)
    assert state.long_memory_enabled
    assert state.k_cache.shape[3] == 8
    cached = inference.prefill(ids, state)[0]

    np.testing.assert_allclose(asnumpy(cached), asnumpy(reference), rtol=1e-5, atol=1e-6)
    assert state.terminal_memory_store.active_blocks == 4
    assert state.terminal_memory_route is not None
    assert state.working_count == 8
    assert state.working_capacity == 8


def test_terminal_landmark_long_decode_slides_bounded_working_cache():
    cfg = _tiny_terminal_long_inference_config()
    training = DecoderLanguageModel(cfg, rng_seed=321, dtype="float32")
    inference = InferenceModel(cfg, dtype="float32")
    inference.set_weights(training)
    state = inference.create_generation_state(batch_size=1, max_length=16)
    tokens = list(range(1, 17))
    inference.prefill(xp.asarray([tokens], dtype=xp.int32), state)

    # After every decoded token, compare against a fresh training forward over
    # the latest complete horizon.  This catches the subtle case where a
    # streaming implementation accidentally pins 2/128-token memory blocks to
    # global positions instead of shifting their phase with the working window.
    tokens.append(17)
    logits = inference.decode_one(xp.asarray([[17]], dtype=xp.int32), state)
    reference = training.forward(
        xp.asarray([tokens[-16:]], dtype=xp.int32), return_cache=False
    )[0, -1]
    np.testing.assert_allclose(asnumpy(logits[0]), asnumpy(reference), rtol=2e-5, atol=2e-6)
    assert logits.shape == (1, cfg.vocab_size)
    assert np.isfinite(asnumpy(logits)).all()
    assert state.k_cache.shape[3] == 8
    assert state.working_count == 8
    assert state.cache_start == 1
    assert state.working_start_abs == 9
    assert state.terminal_memory_store.token_count == 8
    assert state.terminal_memory_store.ring_start == 1
    assert state.terminal_memory_store.active_blocks == 4
    assert state.length == 17

    tokens.append(18)
    logits = inference.decode_one(xp.asarray([[18]], dtype=xp.int32), state)
    reference = training.forward(
        xp.asarray([tokens[-16:]], dtype=xp.int32), return_cache=False
    )[0, -1]
    np.testing.assert_allclose(asnumpy(logits[0]), asnumpy(reference), rtol=2e-5, atol=2e-6)
    assert np.isfinite(asnumpy(logits)).all()
    assert state.cache_start == 2
    assert state.working_start_abs == 10
    assert state.terminal_memory_store.token_count == 8
    assert state.terminal_memory_store.ring_start == 2
    assert state.terminal_memory_store.active_blocks == 4
    assert state.length == 18


def test_terminal_landmark_external_store_respects_requested_horizon():
    cfg = _tiny_terminal_long_inference_config()
    training = DecoderLanguageModel(cfg, rng_seed=321, dtype="float32")
    inference = InferenceModel(cfg, dtype="float32")
    inference.set_weights(training)

    state = inference.create_generation_state(batch_size=1, max_length=12)
    inference.prefill(xp.asarray([list(range(1, 13))], dtype=xp.int32), state)
    assert state.long_memory_enabled
    assert state.working_capacity == 8
    assert state.terminal_memory_store.external_capacity_tokens == 4
    assert state.terminal_memory_store.max_blocks == 2
    assert state.terminal_memory_store.active_blocks == 2
