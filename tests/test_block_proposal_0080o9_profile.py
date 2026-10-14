"""0080O9: observational CUDA-event profiling, no objective changes."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.performance_profiler import (
    _PerformanceProfiler, PERFORMANCE_PROFILER,
    block_proposal_scope, performance_scope,
)


def test_block_detail_suppresses_coarse_nested_sync_only_within_group():
    p = _PerformanceProfiler()
    p.configure(True, reset=True)
    p.block_proposal_detail_enabled = True
    with p.block_proposal_scope('group'):
        with p.scope('attention.legacy_sync'):
            x = 2 + 2
        with p.block_proposal_detail_scope('group.inner'):
            assert x == 4
    with p.scope('trunk'):
        assert x == 4
    records = {r['name']: r for r in p.rows()}
    assert 'attention.legacy_sync' not in records
    assert records['group.host']['count'] == 1
    assert records['group.inner.host']['count'] == 1
    assert records['trunk']['count'] == 1
    p.configure(False)


def test_detail_profile_disabled_has_no_new_records():
    p = _PerformanceProfiler()
    p.configure(True, reset=True)
    p.block_proposal_detail_enabled = False
    with p.block_proposal_scope('group'):
        with p.block_proposal_detail_scope('group.inner'):
            pass
    records = {r['name']: r for r in p.rows()}
    assert set(records) == {'group'}


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU deterministic gradient regression')
def test_full_block_training_profiling_does_not_change_outputs_or_gradients():
    from test_block_proposal_0080d import _build, _sample
    net = _build(learned=False)
    source, labels, valid, metadata = _sample()
    original_enabled = PERFORMANCE_PROFILER.enabled
    original_detail = PERFORMANCE_PROFILER.block_proposal_detail_enabled
    try:
        def run(detail):
            PERFORMANCE_PROFILER.block_proposal_detail_enabled = detail
            PERFORMANCE_PROFILER.configure(detail, reset=True)
            net.zero_grad()
            result = net.train_window(source, labels, valid_target_mask=valid,
                                      memory_metadata=metadata, block_group_size=2)
            gradients = {p.name: asnumpy(p.grad).copy() for p in net.parameters()}
            return result, gradients, {r['name']: r for r in PERFORMANCE_PROFILER.rows()}

        baseline, ga, _ = run(False)
        profiled, gb, records = run(True)
        for key in ('total_loss', 'final_ce', 'proposal_ce'):
            np.testing.assert_allclose(profiled[key], baseline[key], rtol=0, atol=0)
        assert ga.keys() == gb.keys()
        for key in ga:
            np.testing.assert_array_equal(ga[key], gb[key], err_msg=key)
        for key in (
            'block_proposal.decoder_group.forward.host',
            'block_proposal.decoder_group.backward.host',
            'block_proposal.expectation.softmax.forward.host',
            'block_proposal.expectation.gemm.forward.host',
            'block_proposal.proposal_vocab.forward.host',
            'block_proposal.correction_vocab.forward.host',
            'block_proposal.proposal_ce.forward.host',
            'block_proposal.final_ce.forward.host',
            'block_proposal.expectation.softmax.backward.host',
            'block_proposal.boundary_gradient.scatter.host',
        ):
            assert key in records and records[key]['count'] > 0, key
    finally:
        PERFORMANCE_PROFILER.block_proposal_detail_enabled = original_detail
        PERFORMANCE_PROFILER.configure(original_enabled, reset=True)
