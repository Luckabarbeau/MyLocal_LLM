"""0080AA standalone train-metric parity regression."""

import numpy as np
import pytest

from mini_llm.diagnostics.block_proposal_live_audit import LiveBlockProposalAudit


def make_metrics(p=(4., 6.), f=(3., 5.), n=(10, 10)):
    counts = np.array(n, dtype=np.int64)
    proposal = np.array(p)*counts
    final = np.array(f)*counts
    return {
        'position_counts': counts,
        'proposal_position_ce_sums': proposal,
        'final_position_ce_sums': final,
        'unique_supervised_tokens': int(counts.sum()),
        'proposal_ce': float(proposal.sum()/counts.sum()),
        'final_ce': float(final.sum()/counts.sum()),
    }


def test_identical_live_paths_and_production_aggregation():
    audit=LiveBlockProposalAudit(2)
    a=make_metrics()
    b=make_metrics((5., 7.), (4., 6.), (6, 8))
    audit.add('fineweb', a, a)
    audit.add('cosmopedia_v2', b, b)
    summary={
        'position_counts': a['position_counts']+b['position_counts'],
        'proposal_position_ce_sums': a['proposal_position_ce_sums']+b['proposal_position_ce_sums'],
        'final_position_ce_sums': a['final_position_ce_sums']+b['final_position_ce_sums'],
    }
    assert audit.check_production_accumulator(summary)==0
    report=audit.format_report(5500, summary)
    assert 'h1: training' in report and 'fineweb' in report
    assert 'max |train-eval CE|=0.000000' in report


def test_nontrivial_discrepancy_and_bad_counts():
    audit=LiveBlockProposalAudit(2)
    train=make_metrics()
    eval_=make_metrics((5., 7.), (3.1, 5.2))
    audit.add('fineweb', train, eval_)
    assert audit.maximum_ce_error>0.9
    assert audit.maximum_horizon_error>0.9
    bad=make_metrics()
    bad['unique_supervised_tokens']-=1
    with pytest.raises(ValueError, match='counts'):
        audit.add('fineweb', train, bad)


def test_mismatched_ce_rejected():
    audit=LiveBlockProposalAudit(2)
    bad=make_metrics()
    bad['final_ce'] += 1.
    with pytest.raises(ValueError, match='inconsistent final_ce'):
        audit.add('fineweb', bad, make_metrics())


def test_opt_in_live_training_audit_preserves_optimizer_update(tmp_path, monkeypatch, capsys):
    # Use the existing 0080M production-trainer fixture, not a mock of
    # BlockProposalObjective: this executes real forward/backward + a real
    # AdamW step and confirms the extra inference pass is read-only.
    from mini_llm.backend import asnumpy
    from test_block_proposal_0080m_objective import _make_production, _feed

    _, baseline_obj, baseline = _make_production(tmp_path, k=8, accumulation=1)
    _feed(baseline, accumulation=1)
    monkeypatch.delenv('MINI_LLM_BLOCK_PROPOSAL_LIVE_AUDIT', raising=False)
    base_loss, _ = baseline.train_step()

    _, audit_obj, audited = _make_production(tmp_path, k=8, accumulation=1)
    _feed(audited, accumulation=1)
    monkeypatch.setenv('MINI_LLM_BLOCK_PROPOSAL_LIVE_AUDIT', '1')
    audit_loss, _ = audited.train_step()
    assert abs(base_loss-audit_loss) < 1e-6
    assert 'LIVE TRAIN-EVAL AUDIT' in capsys.readouterr().out
    for original, compared in zip(baseline_obj.parameters(), audit_obj.parameters()):
        np.testing.assert_allclose(asnumpy(original.data), asnumpy(compared.data),
                                   rtol=0, atol=1e-6)
