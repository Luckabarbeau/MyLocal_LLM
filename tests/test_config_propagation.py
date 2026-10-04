"""Tests for model configuration propagation."""

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.ops.router import Router


class TestConfigPropagation:
    """Tests for ModelConfig propagation through the model."""

    def test_router_uses_config_n_experts(self):
        """Test that router uses n_experts from config."""
        config = ModelConfig(n_experts=3, top_k=2)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        router = Router(
            d_model=config.d_model,
            n_experts=config.n_experts,
            k=config.top_k,
            input_std=config.router_init_std,
            rng=rng,
        )

        assert router.n_experts == config.n_experts
        assert router.k == config.top_k
        assert router.W_router_param.data.shape[-1] == config.n_experts

    def test_router_uses_config_top_k(self):
        """Test that router uses top_k from config."""
        config = ModelConfig(n_experts=6, top_k=3)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        router = Router(
            d_model=config.d_model,
            n_experts=config.n_experts,
            k=config.top_k,
            input_std=config.router_init_std,
            rng=rng,
        )

        assert router.k == config.top_k

    def test_router_weights_renormalized_after_topk(self):
        """Test that selected weights sum to 1."""
        config = ModelConfig(n_experts=5, top_k=2)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        router = Router(
            d_model=config.d_model,
            n_experts=config.n_experts,
            k=config.top_k,
            input_std=config.router_init_std,
            rng=rng,
        )

        # Create test input
        batch_size, seq_len = 2, 4
        x = np.random.randn(batch_size, seq_len, config.d_model).astype(np.float32)

        output_weights, expert_indices, cache = router.forward(x)

        # Selected weights should sum to 1 (within numerical tolerance)
        weight_sums = np.sum(output_weights, axis=-1)
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-6)

    def test_transformer_block_uses_config_n_experts(self):
        """Test that transformer block uses n_experts from config."""
        config = ModelConfig(n_experts=4, top_k=2)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        block = TransformerBlock(
            d_model=config.d_model,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            top_k=config.top_k,
            input_std=config.init_std,
            output_std=config.residual_init_std,
            rope_base=config.rope_base,
            rng=rng,
        )

        assert block.moe.n_experts == config.n_experts
        assert block.moe.k == config.top_k

    def test_transformer_block_uses_config_top_k(self):
        """Test that transformer block uses top_k from config."""
        config = ModelConfig(n_experts=6, top_k=3)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        block = TransformerBlock(
            d_model=config.d_model,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            top_k=config.top_k,
            input_std=config.init_std,
            output_std=config.residual_init_std,
            rope_base=config.rope_base,
            rng=rng,
        )

        assert block.moe.k == config.top_k

    def test_model_config_used_in_decoder_lm(self):
        """Test that DecoderLanguageModel uses all config values."""
        config = ModelConfig(
            n_experts=4,
            top_k=2,
            rope_base=50_000.0,
        )

        model = DecoderLanguageModel(config, rng_seed=42)

        # Check first block uses config values
        first_block = model.blocks[0]
        assert first_block.moe.n_experts == config.n_experts
        assert first_block.moe.k == config.top_k
        assert first_block.attention.rope_base == config.rope_base


class TestRouterBackward:
    """Tests for router backward pass."""

    def test_router_backward_no_float64(self):
        """Test that router backward uses float32 gradients."""
        config = ModelConfig(n_experts=5, top_k=2)

        from mini_llm.backend import RandomStream
        rng = RandomStream(42)

        router = Router(
            d_model=config.d_model,
            n_experts=config.n_experts,
            k=config.top_k,
            input_std=config.router_init_std,
            rng=rng,
        )

        # Create test input
        batch_size, seq_len = 2, 4
        x = np.random.randn(batch_size, seq_len, config.d_model).astype(np.float32)

        output_weights, expert_indices, cache = router.forward(x)

        # Backward pass
        dweights = np.random.randn(batch_size, seq_len, config.top_k).astype(np.float32)
        dx = router.backward(dweights, cache)

        # Gradient should have same dtype as input
        assert dx.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_medium_context_4k_preset_covers_all_attention_topologies():
    from mini_llm.config import ModelConfig

    config = ModelConfig.medium_context_4k()
    assert config.tokenizer_vocab_size == 65_280
    assert config.context_length == 4_096
    assert config.n_layers == 8
    assert config.n_q_heads == 8

    expected = [
        "local", "local", "local", "local",
        "dilated", "global_sparse", "retrieval", "retrieval",
    ]
    for layer in config.attention_layers:
        assert [head.kind for head in layer.heads] == expected
        router = layer.heads[-1].context_router
        assert router.history_block_size == 128
        assert router.routing_stride == 128
        assert router.query_window == 512
        assert router.exclude_recent_tokens == 1_024
        assert router.top_k_blocks == 4
        assert router.num_queries == 2


def test_large_moe_4k_scaling_presets_preserve_active_geometry():
    from mini_llm.config import ModelConfig

    base = ModelConfig.medium_context_4k()
    cases = [
        (ModelConfig.moe_525m_context_4k(), 24, 525_578_944),
        (ModelConfig.moe_1b_context_4k(), 48, 978_662_272),
    ]
    for config, n_experts, expected_params in cases:
        assert config.context_length == base.context_length == 4_096
        assert config.n_layers == base.n_layers == 8
        assert config.d_model == base.d_model == 512
        assert config.d_ff == base.d_ff == 1_536
        assert config.top_k == base.top_k == 2
        assert config.n_experts == n_experts
        assert config.attention_layers == base.attention_layers
        assert config.estimated_parameter_count() == expected_params


def test_parameter_estimator_matches_medium_context_4k_known_count():
    from mini_llm.config import ModelConfig

    assert ModelConfig.medium_context_4k().estimated_parameter_count() == 185_766_448


def test_wide_500m_sparse_context_family_scales_width_depth_not_experts():
    from mini_llm.config import ModelConfig

    builders = [
        ModelConfig.wide_500m_context_4k,
        ModelConfig.wide_500m_context_8k,
        ModelConfig.wide_500m_context_16k,
        ModelConfig.wide_500m_context_32k,
        ModelConfig.wide_500m_context_64k,
    ]
    expected_contexts = [4_096, 8_192, 16_384, 32_768, 65_536]
    configs = [builder() for builder in builders]

    for config, context in zip(configs, expected_contexts):
        assert config.context_length == context
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_q_heads == 12
        assert config.n_kv_heads == 3
        assert config.d_head == 64
        assert config.d_ff == 2_304
        assert config.n_experts == 6
        assert config.top_k == 2
        assert config.rope_base == 1_000_000.0
        assert config.estimated_parameter_count() == 501_139_272

        kinds = [head.kind for head in config.attention_layers[0].heads]
        assert kinds == [
            "local", "local", "local", "local", "local", "local",
            "dilated", "dilated",
            "global_sparse", "global_sparse",
            "retrieval", "retrieval",
        ]

    # Every context rung must have exactly the same trainable-array shapes.
    assert len({config.estimated_parameter_count() for config in configs}) == 1


def test_wide_500m_sparse_context_budget_stays_bounded_as_context_grows():
    from mini_llm.config import ModelConfig

    # context, local_window, dilated_window, dilation, global_stride,
    # routing_stride, query_window, exclude_recent
    cases = [
        (ModelConfig.wide_500m_context_4k(), 1_024, 4_096, 4, 128, 128, 512, 1_024),
        (ModelConfig.wide_500m_context_8k(), 1_024, 8_192, 8, 128, 128, 512, 1_024),
        (ModelConfig.wide_500m_context_16k(), 512, 8_192, 16, 256, 256, 512, 1_024),
        (ModelConfig.wide_500m_context_32k(), 512, 8_192, 32, 512, 512, 1_024, 2_048),
        (ModelConfig.wide_500m_context_64k(), 256, 8_192, 32, 1_024, 1_024, 1_024, 4_096),
    ]

    for (
        config,
        local_window,
        dilated_window,
        dilation,
        global_stride,
        routing_stride,
        query_window,
        exclude_recent,
    ) in cases:
        heads = config.attention_layers[0].heads
        assert all(head.window == local_window for head in heads[:6])
        assert all(head.window == dilated_window for head in heads[6:8])
        assert all(head.dilation == dilation for head in heads[6:8])
        assert dilation <= 32
        assert heads[8].stride == heads[9].stride == global_stride
        assert heads[8].offset == 0
        assert heads[9].offset == global_stride // 2

        router = heads[-1].context_router
        assert router.history_block_size == 128
        assert router.top_k_blocks == 4
        assert router.routing_stride == routing_stride
        assert router.query_window == query_window
        assert router.query_window <= 1_024
        assert router.exclude_recent_tokens == exclude_recent
        assert router.num_queries == 2


def test_wide_500m_long_context_bounds_dilated_phase_count_and_router_routes():
    from mini_llm.config import ModelConfig

    for config in [
        ModelConfig.wide_500m_context_16k(),
        ModelConfig.wide_500m_context_32k(),
        ModelConfig.wide_500m_context_64k(),
    ]:
        heads = config.attention_layers[0].heads
        dilation = heads[6].dilation
        router = heads[-1].context_router
        # Current dilated implementation has one execution phase per residue.
        assert dilation <= 32
        # Ignore causal/exclusion trimming: this upper bound makes the intended
        # route-count scaling explicit and protects against accidental O(T)
        # route refreshes at 32k/64k.
        assert config.context_length // router.routing_stride <= 64
