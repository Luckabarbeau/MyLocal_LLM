"""0107A: production checkpoint and resume-architecture validation (CPU-only)."""
from dataclasses import replace

import pytest

from mini_llm.config import ModelConfig
from mini_llm.specialists import SpecialistConfig
from mini_llm.specialists.production_contract_0107 import (
    validate_model_b_specialist, validate_resume_specialist,
    validate_sparse_model_b_foundation,
)


def test_production_model_b_requires_stride16_and_exact_architecture():
    config = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    validate_sparse_model_b_foundation(config)
    with pytest.raises(ValueError, match='sparse_retrieval_stride'):
        validate_sparse_model_b_foundation(replace(config, dual_resolution=replace(
            config.dual_resolution, sparse_retrieval_stride=0,
            sparse_retrieval_layers=(),
        )))
    with pytest.raises(ValueError, match='n_layers'):
        validate_sparse_model_b_foundation(replace(config, n_layers=19))
    with pytest.raises(ValueError, match='long_local_window'):
        validate_sparse_model_b_foundation(replace(config, dual_resolution=replace(
            config.dual_resolution, long_local_window=2048,
        )))


def test_production_specialist_must_use_foundation_model_b_blocks():
    foundation = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    exact = SpecialistConfig.from_foundation(
        foundation, specialist_id='general', n_layers=12)
    validate_model_b_specialist(exact, foundation)
    with pytest.raises(ValueError, match='dual_resolution'):
        validate_model_b_specialist(
            replace(exact, dual_resolution=None), foundation)
    with pytest.raises(ValueError, match='rope_base'):
        validate_model_b_specialist(
            replace(exact, rope_base=10_000.0), foundation)


def test_attention_probability_recompute_policy_is_explicit_and_strict():
    foundation = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    retained = SpecialistConfig.from_foundation(
        foundation, specialist_id='general', n_layers=16,
        recompute_attention_probs=False)
    assert not retained.dual_resolution.recompute_attention_probs
    validate_model_b_specialist(
        retained, foundation, recompute_attention_probs=False)

    checkpoint_default = SpecialistConfig.from_foundation(
        foundation, specialist_id='general', n_layers=16)
    with pytest.raises(ValueError, match='dual_resolution'):
        validate_model_b_specialist(
            checkpoint_default, foundation, recompute_attention_probs=False)


def test_resume_rejects_silent_reinterpretation_of_checkpoint_geometry():
    spec = SpecialistConfig(specialist_id='general', n_layers=12)
    validate_resume_specialist(spec, requested_id='general', requested_layers=12)
    with pytest.raises(ValueError, match='specialist-layers'):
        validate_resume_specialist(spec, requested_id='general', requested_layers=2)
    with pytest.raises(ValueError, match='specialist-id'):
        validate_resume_specialist(spec, requested_id='python', requested_layers=12)
