"""Layout and capture contracts for identical-reference-state block validation."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_0226_identical_block_states import Capture,recurrence_layout,history_layout


def test_reference_memory_axes_are_not_silently_reshaped():
    a=np.arange(48*128*128,dtype=np.float32)
    r=recurrence_layout(a)
    assert r.shape==(1,48,128,128)
    assert r[0,2,3,4]==a.reshape(1,48,128,128)[0,2,4,3]
    h=history_layout(np.arange(30720,dtype=np.float32))
    assert h.shape==(1,3,10240) and h[0,1,9]==28


def test_capture_count_type_and_identity_guards(tmp_path):
    a=np.arange(4,dtype=np.float32);a.tofile(tmp_path/'x.f32')
    event=dict(position=0,key='input',file='x.f32',tensor=dict(type=0,ne=[4,1,1,1]))
    (tmp_path/'events.jsonl').write_text(json.dumps(event)+'\n')
    c=Capture(tmp_path)
    np.testing.assert_array_equal(c.get(0,'input',4),a)
    with pytest.raises(ValueError,match='shape mismatch'):c.get(0,'input',5)
    (tmp_path/'events.jsonl').write_text((json.dumps(event)+'\n')*2)
    with pytest.raises(ValueError,match='duplicate'):Capture(tmp_path)


def test_nonfinite_reference_is_not_a_valid_comparison(tmp_path):
    np.array([np.nan],np.float32).tofile(tmp_path/'x')
    (tmp_path/'events.jsonl').write_text(json.dumps(dict(position=0,key='x',file='x',tensor=dict(type=0,ne=[1,1,1,1])))+'\n')
    with pytest.raises(ValueError,match='nonfinite'):Capture(tmp_path).get(0,'x')
