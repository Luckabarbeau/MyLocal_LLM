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
