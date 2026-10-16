"""Production training/evaluation objective parity, without optimizer mutations."""
import os

import numpy as np
import pytest

from benchmarks.audit_block_proposal_training_contract import (
    accumulate_summaries, compact_target_mode, score_same_batch,
)


def test_compact_env_restored():
    key = 'MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET'
    previous = os.environ.get(key)
    try:
        with compact_target_mode(True):
            assert os.environ[key] == '1'
        assert os.environ.get(key) == previous
        with compact_target_mode(False):
            assert os.environ[key] == '0'
        assert os.environ.get(key) == previous
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_weighted_across_batches():
    a = dict(counts=[2,2], proposal=[4.,8.], final=[3.,4.])
    b = dict(counts=[6,6], proposal=[2.,6.], final=[1.,2.])
    out = accumulate_summaries([a,b])
    assert out['counts'] == [8,8]
    np.testing.assert_allclose(out['proposal'], [2.5,6.5])
    np.testing.assert_allclose(out['final'], [1.5,2.5])
    np.testing.assert_allclose(out['gain'], [1.,4.])


def test_invalid_modes_rejected():
    with pytest.raises(ValueError, match='unknown audit mode'):
        score_same_batch(None, None, None, None, None, group_size=1, mode='invalid')


def test_real_cpu_block_objective_inference_vs_training():
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'numpy':
        pytest.skip('CPU reference-only correctness regression')
    from test_block_proposal_production_benchmark import _build, _sample
    trainer = _build()
    net = trainer.net
    source, targets, mask, metadata = _sample()
    ordinary = score_same_batch(net, source, targets, mask, metadata,
                                group_size=2, mode='eval')
    full = score_same_batch(net, source, targets, mask, metadata,
                            group_size=2, mode='train_full')
    compact = score_same_batch(net, source, targets, mask, metadata,
                               group_size=2, mode='train_compact')
    for row in (full,compact):
        assert row['counts'] == ordinary['counts']
        np.testing.assert_allclose(row['proposal'], ordinary['proposal'],
                                   atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(row['final'], ordinary['final'],
                                   atol=1e-7, rtol=1e-7)
