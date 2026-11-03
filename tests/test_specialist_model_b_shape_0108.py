"""Production specialists reuse the foundation's Model B block geometry."""

from dataclasses import replace

import numpy as np

from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.specialists import ResidualSpecialist, SpecialistConfig


def _config():
    return ModelConfig(
        tokenizer_vocab_size=64, context_length=32, n_layers=1,
        d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, ffn_mode="dense", dtype="float32",
        rope_base=1_000_000.0,
        dual_resolution=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8,
            compression_dim=8, compression_heads=2,
            compression_mode="matrix_pool", long_local_window=16,
            long_local_heads=1, summaries_per_block=2,
            global_heads=1, global_head_dim=8, max_blocks=16,
            global_query_chunk=4,
        ),
    )


def _block_schema(block):
    prefix = "blocks.0."
    return [
        (parameter.name.split(prefix, 1)[1], tuple(parameter.data.shape))
        for parameter in block.parameters()
    ]


def test_specialist_from_foundation_matches_model_b_block_schema_and_operator():
    config = _config()
    foundation = DecoderLanguageModel(config, rng_seed=3)
    specialist_config = SpecialistConfig.from_foundation(
        config, specialist_id="shape", n_layers=1)
    specialist = ResidualSpecialist(specialist_config, rng_seed=5)

    foundation_block = foundation.blocks[0]
    specialist_block = specialist.blocks[0]
    assert type(specialist_block.attention) is type(foundation_block.attention)
    assert specialist_block.matrix_pool_dual
    assert specialist_config.rope_base == config.rope_base
    assert specialist_config.residual_init_std == config.residual_init_std
    assert specialist_config.dual_resolution == config.dual_resolution
    assert _block_schema(specialist_block) == _block_schema(foundation_block)

    hidden = np.random.default_rng(8).normal(size=(1, 16, 16)).astype(np.float32)
    tokens = np.arange(16, dtype=np.int64)[None, :]
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=63, block_size=4)
    delta = specialist.forward_delta(
        hidden, dual_resolution_metadata=metadata, return_cache=False)
    np.testing.assert_array_equal(delta, np.zeros_like(delta))


def test_sixteen_layer_specialist_covers_every_sparse_phase_once():
    base = _config()
    config = replace(
        base, n_layers=16,
        dual_resolution=replace(
            base.dual_resolution,
            sparse_retrieval_stride=16,
            sparse_retrieval_head=1,
            sparse_retrieval_layers=tuple(range(2, 18)),
        ),
    )
    specialist_config = SpecialistConfig.from_foundation(
        config, specialist_id="phases", n_layers=16,
        recompute_attention_probs=False)
    specialist = ResidualSpecialist(specialist_config, rng_seed=5)
    phases = [block.attention.sparse_phase for block in specialist.blocks]
    assert specialist_config.dual_resolution.sparse_retrieval_layers == tuple(range(16))
    assert not specialist_config.dual_resolution.recompute_attention_probs
    assert phases == [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert sorted(phases) == list(range(16))
