"""Bounded equal-slot policy and causal diagnostic replay, CPU only."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from mini_llm.ops.expert_heat_policy import propose_heat_moves

spec=importlib.util.spec_from_file_location('analyze_heat',
    Path(__file__).resolve().parents[1]/'scripts/analyze_0295_route_heat.py')
analysis=importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


def test_bounded_stable_proposals_do_not_mutate_inputs():
    heat=np.array([[1.,0.,10.,10.],[3.,0.,9.,8.]])
    mask=np.array([[True,True,False,False],[True,True,False,False]])
    before=mask.copy(); scores=heat.copy()
    pairs=propose_heat_moves(heat,mask,limit=2)
    assert pairs==[((0,2),(0,1)),((0,3),(1,1))]
    np.testing.assert_array_equal(mask,before); np.testing.assert_array_equal(heat,scores)
    for hot,cold in pairs: assert not mask[hot] and mask[cold]
    assert propose_heat_moves(heat,mask,limit=0)==[]


def test_hysteresis_and_no_hot_or_victim():
    heat=np.array([[4.,5.,0.]])
    mask=np.array([[True,False,False]])
    assert propose_heat_moves(heat,mask,limit=2)==[]
    assert propose_heat_moves(heat,np.ones_like(mask),limit=2)==[]
    assert propose_heat_moves(heat,np.zeros_like(mask),limit=2)==[]


@pytest.mark.parametrize('limit',[-1,257,True,1.5])
def test_movement_bound(limit):
    with pytest.raises(ValueError):
        propose_heat_moves(np.ones((1,4)),np.zeros((1,4),bool),limit=limit)


def test_reject_nonfinite_or_wrong_layout():
    with pytest.raises(ValueError):
        propose_heat_moves(np.array([[np.nan,0.]]),np.zeros((1,2),bool),limit=1)
    with pytest.raises(ValueError):
        propose_heat_moves(np.ones((1,4)),np.zeros((1,4),np.int32),limit=1)


def frame(heat,steps):
    return dict(heat=np.array([heat],np.int64),steps=steps,
                resident=np.array([[True,False,False,False]]),sizes=np.array([100]))


def test_causal_first_window_never_uses_future_routes():
    first=frame([0,0,10,0],1)
    second=frame([0,0,20,0],2)
    result=analysis.simulate([first,second],limit=1)
    assert result['windows'][0]['proposed_placement_misses']==10
    assert result['windows'][1]['proposed_placement_misses']==0
    assert result['baseline_cpu_experts']==20 and result['causal_placement_cpu_experts']==10
    # Changing future popularity cannot alter the first evaluated window.
    opposite=analysis.simulate([first,frame([0,0,10,10],2)],limit=1)
    assert result['windows'][0]==opposite['windows'][0]


def test_zero_movement_is_exact_baseline_and_resets_rejected():
    frames=[frame([0,2,3,0],1),frame([1,2,6,0],2)]
    result=analysis.simulate(frames,limit=0)
    assert result['baseline_cpu_experts']==result['causal_placement_cpu_experts']
    assert result['proposed_moves']==0
    with pytest.raises(ValueError,match='regressed'):
        analysis.simulate([frames[1],frames[0]],limit=1)
