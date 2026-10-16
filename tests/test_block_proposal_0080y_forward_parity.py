"""Isolated tests for the read-only production forward-parity auditor."""
import numpy as np
import pytest

from benchmarks.audit_block_proposal_forward_parity import (
    check_next_token_alignment, compare_scores,
)


def test_next_token_alignment_with_masked_padding():
    source = np.array([[9, 8, 2, 2, 3, 4, 5], [5, 5, 7, 7, 99, 99, 99]])
    targets = np.array([[3, 4, 5, 6], [8, 99, 99, 99]])
    valid = np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool)
    assert check_next_token_alignment(source, targets, valid, 3) == [4, 1]


def test_alignment_detects_shifted_labels():
    source = np.array([[2, 3, 4, 5, 6]])
    targets = np.array([[3, 8, 6]])
    with pytest.raises(AssertionError):
        check_next_token_alignment(source, targets, np.ones((1, 3), bool), 2)


def test_comparison_rejects_target_count_mismatch():
    left = dict(counts=[10, 10], proposal=[3, 4], final=[2, 3], gain=[1, 1])
    right = dict(counts=[10, 9], proposal=[3, 4], final=[2, 3], gain=[1, 1])
    with pytest.raises(AssertionError):
        compare_scores(left, right)


def test_comparison_is_signed():
    left = dict(counts=[10], proposal=[3], final=[2], gain=[1])
    right = dict(counts=[10], proposal=[4], final=[4], gain=[0])
    result = compare_scores(left, right)
    assert result == dict(proposal=[1.0], final=[2.0], gain=[-1.0])


def test_full_production_objective_cached_vs_inference_forward_cpu():
    """Same small real routed Transformer, decoder and target window."""
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'numpy':
        pytest.skip('CPU reference numerical parity test')
    from test_block_proposal_production_benchmark import _build, _sample
    from benchmarks.audit_block_proposal_forward_parity import eval_same_batch

    trainer = _build()
    source, targets, mask, metadata = _sample()
    model = trainer.net.proposal_head.model
    original = model.forward_body
    uncached = eval_same_batch(trainer.net, source, targets, mask, metadata,
                               2, cached_forward=False)
    cached = eval_same_batch(trainer.net, source, targets, mask, metadata,
                             2, cached_forward=True)
    assert model.forward_body == original  # no persistent monkeypatch
    for key in ('proposal', 'final', 'gain'):
        np.testing.assert_allclose(uncached[key], cached[key], rtol=1e-10, atol=1e-10)
