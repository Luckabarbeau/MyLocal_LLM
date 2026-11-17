"""Synthetic callback identity and graph/layout contracts; no checkpoint loading."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('identity_probe', ROOT/'scripts/probe_0218_hc_identity.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def event(ptr, kind):
    root = dict(ptr=ptr, name='hc_mixed-0', op='SCALE', src=[ptr+1],
                type=0, ne=[4,1,1,1], nb=[4,16,16,16])
    weight = dict(ptr=ptr+1, name=f'blk.0.hc_{kind}_norm.weight', op='NONE', src=[])
    return dict(tensor=root,nodes=[root,weight])


def test_duplicate_identity_not_occurrence_order():
    attn, ffn = event(10,'attn'),event(20,'ffn')
    assert probe.associate([ffn,attn,attn,ffn]) == {20:'hc_ffn_read',10:'hc_attention_read'}


def test_distinct_same_name_rejected():
    with pytest.raises(ValueError,match='multiple distinct'):
        probe.associate([event(10,'attn'),event(20,'ffn'),event(30,'attn')])


def test_ambiguous_provenance_rejected():
    e = event(10,'attn')
    e['nodes'].append(dict(ptr=12,name='blk.0.hc_ffn_norm.weight',op='NONE',src=[]))
    e['tensor']['src'].append(12)
    with pytest.raises(ValueError,match='ambiguous'):
        probe.classify(e)


def test_f32_flatten_contract():
    e = event(10,'attn')
    probe.validate_layout(e,4)
    e['tensor']['ne'] = [2,2,1,1]
    with pytest.raises(ValueError):
        probe.validate_layout(e,4)


def test_no_prefetch_guard_exists():
    text = (ROOT/'scripts/probe_0218_hc_identity.py').read_text()
    assert "report['provenance'] = provenance()" in text
    assert text.index("report['provenance'] = provenance()") < text.index("report['reference_run'] = bounded")
    assert "report_path.open('x')" in text
    assert 'time.monotonic()+120' in text
