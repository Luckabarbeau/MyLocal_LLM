"""Specialist composition respects Model B document metadata and freezing."""
import numpy as np
import pytest

from mini_llm.config import ModelConfig, DualResolutionConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.specialists import (FrozenFoundation, ResidualSpecialist,
                                  SpecialistConfig, FrozenResidualComposition)


def setup_model():
    model = DecoderLanguageModel(ModelConfig(
        tokenizer_vocab_size=64, context_length=8, n_layers=1,
        d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, ffn_mode='dense', dtype='float32',
        dual_resolution=DualResolutionConfig(enabled=True, block_size=2,
            local_window=4, compression_dim=8, compression_heads=2,
            compression_mode='matrix_pool', global_heads=1,
            global_head_dim=8, max_blocks=8, global_query_chunk=4)), rng_seed=25)
    foundation = FrozenFoundation(model)
    specialist = ResidualSpecialist(SpecialistConfig(
        specialist_id='general', n_layers=2, d_model=16, d_ff=32,
        n_q_heads=2, n_kv_heads=1, d_head=8, full_window=8,
        local_window=4, dtype='float32'), rng_seed=12)
    return foundation, specialist


def test_composed_initialization_matches_frozen_foundation_and_backprop():
    foundation, specialist = setup_model()
    composed = FrozenResidualComposition(foundation, specialist)
    source = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    target = (source+1) % 64
    meta = build_dual_resolution_metadata(source, eos_token_id=63, block_size=2)
    foundation_weights = {p.name: p.data.copy() for p in foundation.model.parameters()}
    baseline_loss, _ = foundation.head_loss(foundation.hidden(source)[0], target)
    actual, cache = composed.loss(source, target, dual_resolution_metadata=meta)
    assert actual == pytest.approx(baseline_loss, abs=1e-7)
    assert len(composed.parameters()) == len(specialist.parameters())
    composed.backward(cache)
    assert np.any(specialist.correction_proj.W.grad != 0)
    for p in foundation.model.parameters():
        np.testing.assert_array_equal(p.data, foundation_weights[p.name])
        if p.grad is not None:
            assert np.all(p.grad == 0)


def test_reject_cross_document_leakage_before_running_specialist():
    foundation, specialist = setup_model()
    composed = FrozenResidualComposition(foundation, specialist)
    x = np.array([[1, 2, 63, 4, 5, 6, 7, 8]], dtype=np.int32)
    meta = build_dual_resolution_metadata(x, eos_token_id=63, block_size=2)
    with pytest.raises(NotImplementedError, match='multi-document'):
        composed.loss(x, x, dual_resolution_metadata=meta)


def test_document_metadata_positions_are_passed_to_specialist():
    _, specialist = setup_model()
    hidden = np.zeros((1, 8, 16), dtype=np.float32)
    meta = build_dual_resolution_metadata(
        np.arange(8, dtype=np.int32)[None, :], eos_token_id=63, block_size=2)
    a = specialist.forward_delta(hidden, dual_resolution_metadata=meta, return_cache=False)
    b = specialist.forward_delta(hidden, position_ids=meta['position_ids'], return_cache=False)
    np.testing.assert_array_equal(a, b)
