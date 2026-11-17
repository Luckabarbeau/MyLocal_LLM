"""Sampled GGML persistent-state layout conversions, independent of checkpoint."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_0223_state_evolution import canonical, verified_events


def test_recurrent_physical_value_key_to_logical_key_value(tmp_path):
    raw=np.arange(48*128*128,dtype=np.float32).reshape(1,48,128,128)
    raw.tofile(tmp_path/'state')
    name,out=canonical(tmp_path,dict(file='state',tensor=dict(name='state_predelta-0',ne=[128,128,48,1])))
    assert name=='state' and out[0,2,3,4]==raw[0,2,4,3]


def test_oldest_first_channel_fast_history_conversion(tmp_path):
    raw=np.arange(3*10240,dtype=np.float32).reshape(10240,3)
    raw.tofile(tmp_path/'conv')
    name,out=canonical(tmp_path,dict(file='conv',tensor=dict(name='conv_state_at-0',ne=[3,10240,1,1])))
    assert name=='conv0' and out[0,1,7]==raw[7,1]


def test_only_proven_v_reshape_alias_is_removed():
    raw=dict(position=0,tensor=dict(ptr=10,name='Vcur-3',ne=[512,1,1,1],src=[]))
    shaped=dict(position=0,tensor=dict(ptr=20,name='Vcur-3',ne=[256,2,1,1],src=[10]))
    assert verified_events([raw,shaped])==[shaped]
    with pytest.raises(ValueError,match='unverified'):
        verified_events([raw])


def test_unknown_state_rejected(tmp_path):
    np.zeros(1,np.float32).tofile(tmp_path/'x')
    with pytest.raises(ValueError,match='unrecognized'):
        canonical(tmp_path,dict(file='x',tensor=dict(name='unknown',ne=[1]*4)))
