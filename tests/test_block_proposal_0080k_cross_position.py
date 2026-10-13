"""Frozen cross-position plan test, without changing training or the prior."""
import json

import numpy as np
import pytest

from mini_llm.backend import asnumpy
from mini_llm.block_proposal.diagnostics import SCOPES, plan_mask, score_window
from tests.test_block_proposal_0080d import _build, _sample
from tests.test_block_proposal_0080i_learning import prepared, _argv
from examples import evaluate_block_proposal_cross_position_0080k as k
from examples import train_block_proposal_cpu_learning_0080i as train


def test_horizon_specific_masks_are_precise_and_prohibit_leakage():
    assert set(SCOPES)=={'all','own_only','past_and_current','all_except_current','neutral'}
    for horizon in range(4):
        assert plan_mask('all',4)[horizon].all()
        assert not plan_mask('neutral',4)[horizon].any()
        assert np.array_equal(plan_mask('own_only',4)[horizon],
                              np.arange(4)==horizon)
        assert np.array_equal(plan_mask('past_and_current',4)[horizon],
                              np.arange(4)<=horizon)
        assert np.array_equal(plan_mask('all_except_current',4)[horizon],
                              np.arange(4)!=horizon)
    assert np.array_equal(plan_mask('past_and_current',4)[0],plan_mask('own_only',4)[0])
    assert np.array_equal(plan_mask('past_and_current',4)[-1],plan_mask('all',4)[-1])
    with pytest.raises(ValueError,match='unknown'):
        plan_mask('invalid',4)


def _check_sample(m,group):
    src,y,valid,meta=_sample()
    return score_window(m,src,y,valid,meta,group_size=group)


def test_all_reproduces_original_and_chunked_groups_match():
    m=_build(learned=True)
    # Amplify random corrections so ablations are measurable without training.
    m.net.decoder.up.W.data *= 75
    first=_check_sample(m,group=1)
    second=_check_sample(m,group=5)
    ref=m.train_window(*_sample()[:2],valid_target_mask=_sample()[2],
                       memory_metadata=_sample()[3],compute_gradients=False,
                       block_group_size=3)
    assert first['main_forward_passes']==1
    np.testing.assert_array_equal(first['counts'],asnumpy(ref['position_counts']))
    np.testing.assert_allclose(first['proposal_sums'],asnumpy(ref['proposal_position_ce_sums']),
                               atol=3e-6,rtol=3e-7)
    np.testing.assert_allclose(first['final_sums']['all'],asnumpy(ref['final_position_ce_sums']),
                               atol=3e-6,rtol=3e-7)
    for cond in SCOPES:
        np.testing.assert_allclose(first['final_sums'][cond],second['final_sums'][cond],
                                   atol=2e-7,rtol=2e-7)
    assert abs(first['final_sums']['all'].sum()-first['final_sums']['neutral'].sum()) > 1e-8
    # On horizon 1, 'own_only' and 'past_and_current' MUST agree exactly.
    np.testing.assert_allclose(first['final_sums']['own_only'][0],
                               first['final_sums']['past_and_current'][0],atol=1e-9)
    np.testing.assert_allclose(first['final_sums']['past_and_current'][-1],
                               first['final_sums']['all'][-1],atol=1e-9)


def test_forward_controls_do_not_modify_weights_or_gradients():
    m=_build(learned=True)
    before=[(asnumpy(p.data).copy(),asnumpy(p.grad).copy()) for p in m.parameters()]
    _check_sample(m,group=2)
    for p,(data,grad) in zip(m.parameters(),before):
        np.testing.assert_array_equal(asnumpy(p.data),data)
        np.testing.assert_array_equal(asnumpy(p.grad),grad)


def test_real_checkpoint_loads_and_matches_0080j_baseline(prepared,tmp_path,monkeypatch):
    output=tmp_path/'trained'
    train.main(_argv(prepared,output,'--architecture','two_stage','--batch-size','2'))
    monkeypatch.setattr(k,'_load_tokenizer',lambda path:(
        __import__('tests.test_block_proposal_0080g_cpu',fromlist=['ToyTokenizer']).ToyTokenizer(),1))
    r=k.main(['--mixed-shard-root',str(prepared),'--training-output',str(output)])
    assert r['training_step']==2
    assert r['total_supervised_tokens']>0
    assert r['main_forward_passes']==2
    assert (output/'ablations_0080k_cross_position.json').is_file()
    saved=json.loads((output/'two_stage_validation.jsonl').read_text().splitlines()[-1])
    assert r['conditions']['all']['final_ce']==pytest.approx(saved['final_ce'],abs=3e-6)
    assert r['conditions']['all']['proposal_ce']==pytest.approx(saved['proposal_ce'],abs=3e-6)
    assert r['conditions']['own_only']['horizon_ce_increase_vs_all'][0]==pytest.approx(
        r['conditions']['past_and_current']['horizon_ce_increase_vs_all'][0],abs=1e-10)
    with pytest.raises(FileExistsError,match='overwrite'):
        k.main(['--mixed-shard-root',str(prepared),'--training-output',str(output)])
