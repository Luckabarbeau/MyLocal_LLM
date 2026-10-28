from dataclasses import replace

import numpy as np

from mini_llm.config import (
    AttentionLayerConfig,
    LocalAttentionConfig,
    MemoryContextConfig,
    ModelConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from train_model import model_config_from_name


def _small_config():
    layer = AttentionLayerConfig(heads=(
        LocalAttentionConfig(window=8),
        LocalAttentionConfig(window=4),
        LocalAttentionConfig(window=4),
        LocalAttentionConfig(window=4),
    ))
    return ModelConfig(
        tokenizer_vocab_size=64,
        context_length=8,
        n_layers=2,
        d_model=16,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=4,
        n_experts=1,
        top_k=1,
        d_ff=32,
        ffn_mode="dense",
        attention_layers=(layer, layer),
        memory_context=MemoryContextConfig(
            enabled=True,
            memory_length=16,
            recent_length=8,
            target_length=8,
            block_size=4,
            top_k_blocks=1,
            router_query_length=8,
            router_dim=4,
            read_heads=1,
            read_kv_heads=1,
            integration_mode="direct_full_history",
        ),
        full_history_layers=(0, 1),
    )


def test_production_preset_has_one_replacement_and_fifteen_local_heads():
    config = model_config_from_name(
        "ensemble-1024-20l-dense-direct-full-history-64k"
    )
    assert config.memory_context.integration_mode == "direct_full_history"
    assert config.full_history_layers == tuple(range(20))
    assert config.memory_context.distant_memory_length == 61_440
    for layer in config.attention_layers:
        assert [head.window for head in layer.heads] == [4096] + [512] * 15


def test_direct_full_history_replaces_regular_head_zero_and_backpropagates():
    config = _small_config()
    model = DecoderLanguageModel(config, rng_seed=7)
    assert model.memory_router is None
    assert model.historical_memory is None
    assert model.direct_full_history
    assert config.estimated_parameter_count() == sum(p.size for p in model.parameters())
    assert all(block.attention.replace_full_head_with_history for block in model.blocks)
    assert all(np.any(block.full_history_head.o.data != 0) for block in model.blocks)

    tokens = np.random.default_rng(9).integers(0, 64, (1, 16), dtype=np.int32)
    metadata = {
        "history_valid_starts": np.array([0], dtype=np.int64),
        "target_valid_lengths": np.array([8], dtype=np.int64),
    }
    output, cache = model.forward_body(
        tokens, memory_metadata=metadata, return_cache=True
    )
    assert output.shape == (1, 8, 16)
    model.backward_body(np.ones_like(output) * 0.01, cache)

    for block in model.blocks:
        assert np.linalg.norm(block.full_history_head.q.grad) > 0
        assert np.linalg.norm(block.full_history_head.o.grad) > 0
        np.testing.assert_array_equal(block.attention.Wq.grad[:, :4], 0)
        np.testing.assert_array_equal(block.attention.Wo.grad[:4], 0)


def test_direct_full_history_requires_replacement_in_every_layer():
    config = _small_config()
    try:
        replace(config, full_history_layers=(0,))
    except ValueError as exc:
        assert "every Transformer layer" in str(exc)
    else:
        raise AssertionError("partial direct_full_history configuration was accepted")
