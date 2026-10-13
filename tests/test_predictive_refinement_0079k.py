"""0079K parameter-scale configuration tests; CPU-safe, no 500M allocation."""

from mini_llm.config import ModelConfig
from mini_llm.predictive_refinement.scale_config import (
    WIDE_TRUNK_NAME, parameter_memory_gib, wide_static_trunk_config,
)
from examples.train_predictive_refinement_text import parse_args


def test_static_wide_preserves_backbone_exactly():
    original = ModelConfig.wide_500m_context_4k()
    static = wide_static_trunk_config(dtype='bfloat16')
    for name in ('tokenizer_vocab_size', 'n_layers', 'd_model', 'n_q_heads',
                 'n_kv_heads', 'd_head', 'n_experts', 'top_k', 'd_ff', 'rope_base'):
        assert getattr(static, name) == getattr(original, name)
    assert static.n_layers == 12 and static.top_k == 2
    assert original.estimated_parameter_count() == 501139272
    assert static.context_length == 256
    assert not static.memory_context.enabled
    assert tuple(h.kind for h in static.attention_layers[0].heads) == (
        'local', 'local', 'local', 'local', 'local', 'local',
        'dilated', 'dilated', 'global_sparse', 'global_sparse', 'local', 'local',
    )
    assert all(h.kind != 'retrieval' for a in static.attention_layers for h in a.heads)
    # The *only* missing trunk weights are the small attention routers.
    assert 500_000_000 < static.estimated_parameter_count() < original.estimated_parameter_count()


def test_memory_bounds_do_not_count_activation_buffers():
    m = parameter_memory_gib(500_000_000, 'bfloat16')
    assert abs(m['weights'] - 1e9 / (1024 ** 3)) < 1e-12
    assert abs(m['no_offload_total'] - 9e9 / (1024 ** 3)) < 1e-12
    assert m['full_offload_gpu_lower_bound'] < m['no_offload_total']
    assert parameter_memory_gib(500, 'float32')['weights'] == 2000 / (1024 ** 3)


def test_cli_exposes_opt_in_large_model_without_affecting_defaults():
    args = parse_args(['--shard-dir', '/does/not/need/to/exist', '--model-preset', WIDE_TRUNK_NAME,
                       '--no-save', '--steps', '1', '--dtype', 'bfloat16'])
    assert args.no_save and args.model_preset == WIDE_TRUNK_NAME
    assert parse_args(['--shard-dir', '/missing']).model_preset == 'custom'


def test_static_layout_runs_grouped_top_two_reference_backward():
    """Keep this a tiny CPU test; the ~500M allocation is only for the user's GPU."""
    from dataclasses import replace
    import numpy as np
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
    from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig

    large = wide_static_trunk_config(dtype='float32')
    tiny = replace(large, tokenizer_vocab_size=64, n_layers=1, d_model=24,
                   n_q_heads=12, n_kv_heads=3, d_head=2, d_ff=32, n_experts=2,
                   top_k=2, attention_layers=large.attention_layers[:1])
    model = DecoderLanguageModel(tiny, rng_seed=401)
    trainer = WindowRefinementTrainer(
        model, RefinementReferenceConfig(block_size=2, enable_shared_context_training=True)
    )
    inputs = np.asarray([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=np.int64)
    result = trainer.train_window(inputs, prefix_length=8, group_size=2)
    assert result['supervised_tokens'] == 4
    assert np.isfinite(float(result['commitment_ce']))
    assert any(np.any(np.asarray(p.grad) != 0) for p in trainer.parameters())
