"""Identity, layout and metric contracts for bounded layer diagnostics."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('layer_probe', ROOT / 'scripts/probe_0220_layer_evolution.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def event(ptr, kind, layer=3):
    root = dict(ptr=ptr, name=f'hc_mixed-{layer}', op='DSV4_HC_PRE', src=[ptr+1],
                type=0, ne=[2560,1,1,1], nb=[4,10240,10240,10240], data=100)
    norm = dict(ptr=ptr+1, name=f'blk.{layer}.hc_{kind}_norm.weight', op='NONE', src=[])
    return dict(position=0, tensor=root, nodes=[root,norm])


def test_identity_not_names_storage_or_callback_order():
    a, f = event(10,'attn'), event(20,'ffn')
    assert probe.associate([f,a,a,f],complete=False) == {
        (0,20):(0,3,'hc_ffn_read'), (0,10):(0,3,'hc_attention_read')}


def test_reshaped_alias_is_not_a_mathematical_boundary():
    a = event(10,'ffn')
    a['tensor']['name'] += ' (reshaped)'
    with pytest.raises(ValueError,match='invalid layer boundary'):
        probe.classify(a)


def test_missing_or_ambiguous_ancestry_rejected():
    a = event(10,'attn')
    a['tensor']['src'].append(999)
    with pytest.raises(ValueError, match='incomplete'):
        probe.classify(a)
    with pytest.raises(ValueError, match='missing/repeated'):
        probe.associate([event(10,'attn')])


def test_stream_layout_matches_dimension_zero_fastest():
    e = event(10,'attn')
    e['tensor'].update(ne=[2560,4,1,1],nb=[4,10240,40960,40960])
    a = np.arange(10240,dtype=np.float32).reshape(4,2560)
    flat = probe.validate_layout(e,a,'post_mixer')
    assert flat[2560] == a[1,0] and np.shares_memory(a,flat)
    with pytest.raises(ValueError,match='layout mismatch'):
        probe.validate_layout(e,a.T,'post_mixer')


def test_metrics_and_nonfinite():
    r = np.array([3.,4.])
    m = probe.metrics(r,r)
    assert m['relative_l2'] == 0 and m['cosine'] == 1 and m['reference_norm'] == 5
    shifted = probe.metrics(r+1,r)
    assert shifted['rms_difference'] == 1
    assert shifted['relative_l2'] == pytest.approx(np.sqrt(2)/5)
    with pytest.raises(ValueError,match='nonfinite'):
        probe.metrics(np.array([np.nan]),np.ones(1))


def test_report_is_exclusive_and_png_is_valid(tmp_path):
    existing = tmp_path/'old.json'
    existing.write_text('preserve')
    with pytest.raises(FileExistsError):
        probe.reserve([existing])
    rows = [dict(stage=s,position=p,layer=l,relative_l2=.001*(l+1))
            for s in probe.STAGES for p in range(2) for l in range(48)]
    png = tmp_path/'evolution.png'
    probe.plot_png(rows,png)
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert existing.read_text() == 'preserve'
