"""No-GPU regression tests for the optional activation cache probe."""
import numpy as np
import pytest
from mini_llm.activation_memory_audit import activation_storage_details, LayerMemoryAudit


def test_shared_views_are_not_counted_as_separate_storage():
    base = np.zeros((300, 8), dtype=np.float32)
    view_a = base[:100]
    view_b = base[150:]
    arr = np.ones((10, 8), dtype=np.float32)
    own, apparent, groups, paths, owners = activation_storage_details({
        'attn_cache': {'query': view_a, 'key': view_b},
        'moe_cache': {'different': arr, 'another_view': view_a},
        'module': object(),
    })
    assert len(owners) == 2
    assert own == base.nbytes + arr.nbytes
    assert groups['attn_cache'] == base.nbytes
    assert groups['moe_cache'] == base.nbytes + arr.nbytes


def test_audit_exits_at_requested_layer_and_writes_csv(tmp_path, monkeypatch):
    import mini_llm.activation_memory_audit as module
    memory = [10.0, 11.0]
    def fake_snapshot():
        return {'pool_live_mib': memory.pop(0),
                'pool_reserved_mib': 30.0,
                'device_used_gib': 3.0,
                'device_free_gib': 12.0}
    monkeypatch.setattr(module, '_gpu_snapshot', fake_snapshot)
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_FILE', str(tmp_path / 'layer.csv'))
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_AFTER_LAYER', '1')
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_USED_GIB', '12')
    probe = LayerMemoryAudit(32, np.zeros((2, 64, 768), np.float32))
    with pytest.raises(SystemExit) as exc:
        probe.after_layer(0, {'attn_cache': np.ones((8, 16), np.float32)})
    assert exc.value.code == 0
    csv = (tmp_path / 'layer.csv').read_text()
    assert 'attn_owner_mib' in csv
    assert '0,2,64,768,' in csv
