"""0080BE: wide24 1024-dimensional 501M-parameter preset contract."""

from collections import Counter
from dataclasses import replace

import pytest

from mini_llm.config import (
    AttentionLayerConfig,
    DilatedAttentionConfig,
    GlobalSparseAttentionConfig,
    LocalAttentionConfig,
    MemoryContextConfig,
    ModelConfig,
    RetrievalAttentionConfig,
)
from train_model import model_config_from_name, parse_args


def test_wide24_preset_geometry_and_parameter_count():
    old = ModelConfig.deep_500m_memory_64k()
    cfg = ModelConfig.wide24_500m_memory_64k()
    assert cfg.estimated_parameter_count() == 501_212_208
    assert old.estimated_parameter_count() == 499_213_120
    assert (cfg.n_layers, cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads, cfg.d_head) == (24, 1024, 16, 4, 64)
    assert (cfg.n_experts, cfg.top_k, cfg.d_ff) == (2, 1, 2048)
    assert cfg.context_length == 6144
    assert cfg.memory_context == old.memory_context
    assert cfg.memory_context.memory_length == 65_536
    assert cfg.memory_context.target_length == 4096
    assert cfg.memory_context.top_k_blocks == 16
    assert cfg.memory_context.block_size == 128
    assert len(cfg.attention_layers) == 24
    assert all(layer == cfg.attention_layers[0] for layer in cfg.attention_layers)


def test_wide24_attention_modes_and_routing_unchanged():
    old_heads = ModelConfig.deep_500m_memory_64k().attention_layers[0].heads
    new_heads = ModelConfig.wide24_500m_memory_64k().attention_layers[0].heads
    counts = Counter(type(head) for head in new_heads)
    assert counts == {
        LocalAttentionConfig: 8,
        DilatedAttentionConfig: 4,
        GlobalSparseAttentionConfig: 2,
        RetrievalAttentionConfig: 2,
    }
    assert all(head == old_heads[0] for head in new_heads[:8])
    assert all(head == old_heads[6] for head in new_heads[8:12])
    assert new_heads[12:] == old_heads[8:]
    assert new_heads[14].group == new_heads[15].group == 'far'
    assert new_heads[14].context_router.num_queries == 2


def test_wide24_cli_model_dispatch_and_old_preset_untouched(monkeypatch):
    monkeypatch.setattr('sys.argv', ['train_model.py', '--model', 'wide24-500m-memory-64k'])
    assert parse_args().model == 'wide24-500m-memory-64k'
    assert model_config_from_name('wide24-500m-memory-64k') == ModelConfig.wide24_500m_memory_64k()
    assert model_config_from_name('deep-500m-memory-64k') == ModelConfig.deep_500m_memory_64k()


def test_wide24_full_depth_shape_validation_and_tiny_backward():
    # Preserve the exact 24-layer/top-1 trunk without constructing 501M arrays.
    import numpy as np
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    if BACKEND_NAME != 'numpy':
        pytest.skip('small full-depth reference test runs on NumPy')
    cfg = replace(
        ModelConfig.wide24_500m_memory_64k(),
        tokenizer_vocab_size=43,
        context_length=16,
        d_model=32,
        n_q_heads=16,
        n_kv_heads=4,
        d_head=2,
        d_ff=32,
        attention_layers=None,
        memory_context=MemoryContextConfig(enabled=False),
    )
    model = DecoderLanguageModel(cfg, rng_seed=19, dtype='float64')
    assert len(model.blocks) == 24
    tokens = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    model.zero_grad()
    hidden, cache = model.forward_body(tokens)
    assert hidden.shape == (1, 4, 32)
    model.backward_body(np.ones_like(hidden) * 0.03, cache)
    assert all(block.moe.router.k == 1 for block in model.blocks)
    for index in (0, 11, 23):
        assert any(np.any(p.grad != 0) for p in model.blocks[index].moe.router.parameters())
