"""0082G: report individual members and all ensemble baselines correctly."""
import numpy as np
import pytest

from examples.train_probability_fusion_streaming_0082f import (
    _batch_teacher_ces, _comparison_fields,
)
from examples.train_probability_fusion_0082a import _evaluate
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


@pytest.mark.parametrize('n_members', [1, 2, 4, 10])
def test_batch_reference_ces_agree_with_fixed_heldout_evaluation(n_members):
    rng = np.random.default_rng(82 + n_members)
    rows, vocab = 9, 17
    # Per-member additive logit offsets must not change CE.
    logits = rng.normal(size=(rows, n_members, vocab)).astype(np.float32)
    logits += rng.normal(size=(1, n_members, 1)).astype(np.float32) * 150.0
    targets = rng.integers(0, vocab, size=rows, dtype=np.int64)
    ids = np.arange(n_members) * 2
    geometric, individual = _batch_teacher_ces(logits, targets)
    source = (ids, [logits[:, i, :] for i in range(n_members)], targets)
    combiner = ProbabilityResidualCombiner(FusionConfig(vocab_size=vocab, max_models=24))
    _, _, expected_geometric, expected_individual = _evaluate(combiner, source, batch_rows=3)
    np.testing.assert_allclose(geometric, expected_geometric, atol=3e-5)
    np.testing.assert_allclose(individual, expected_individual, atol=3e-5)


def test_comparison_fields_use_stable_model_ids_and_positive_improvement():
    ids = [7, 2, 9]
    ces = [4.2, 3.5, 3.9]
    result = _comparison_fields(3.1, 3.3, 3.2, ids, ces)
    assert 'model_2_ce=3.50000' in result
    assert 'model_7_ce=4.20000' in result
    assert 'model_9_ce=3.90000' in result
    assert 'best_member_id=2 best_member_ce=3.50000' in result
    assert 'gain_arithmetic_best_member=+0.20000' in result
    assert 'gain_geometric_best_member=+0.30000' in result
    assert 'gain_fusion_best_member=+0.40000' in result
    assert 'gain_fusion_best_reference=+0.10000' in result
    # Relabeling input positions without changing ID/distribution pairs leaves comparison unchanged.
    assert result == _comparison_fields(3.1, 3.3, 3.2, [9, 7, 2], [3.9, 4.2, 3.5])
    assert 'gain_fusion_best_reference=-0.70000' in _comparison_fields(3.9, 3.3, 3.2, ids, ces)


def test_comparison_fields_support_single_member_and_reject_invalid():
    output = _comparison_fields(1.9, 2.0, 2.0, [4], [2.0])
    assert 'model_4_ce=2.00000' in output
    assert 'gain_fusion_best_reference=+0.10000' in output
    with pytest.raises(ValueError):
        _comparison_fields(2, 2, 2, [1, 2], [3])
