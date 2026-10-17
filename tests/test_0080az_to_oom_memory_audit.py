"""0080AZ audit: durable begin/complete/oom CSV behavior, CPU reference."""
import csv
import numpy as np
import pytest

from mini_llm.activation_memory_audit import LayerMemoryAudit


def _fake_snap():
    return {
        'pool_live_mib': 100.0, 'pool_reserved_mib': 200.0,
        'device_used_gib': 3.0, 'device_free_gib': 12.0,
        'host_rss_mib': 300.0, 'host_locked_mib': 0.0,
        'host_available_mib': 50000.0, 'pinned_pool_free_blocks': 0,
    }


def test_oom_event_is_flushed_before_unwind(tmp_path, monkeypatch):
    import mini_llm.activation_memory_audit as mod
    monkeypatch.setattr(mod, '_gpu_snapshot', _fake_snap)
    path = tmp_path / 'oom.csv'
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_FILE', str(path))
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_AFTER_LAYER', '32')
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_USED_GIB', '0')
    probe = LayerMemoryAudit(32, np.zeros((2, 4096, 768), np.float32))
    probe.before_layer(4)
    probe.forward_failure(4, MemoryError('out of memory'))
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [row['event'] for row in rows] == ['begin', 'oom']
    assert rows[-1]['layer'] == '4'
    assert rows[-1]['host_rss_mib'] == '300.0'


def test_all_32_complete_stops_before_backward(tmp_path, monkeypatch):
    import mini_llm.activation_memory_audit as mod
    monkeypatch.setattr(mod, '_gpu_snapshot', _fake_snap)
    path = tmp_path / 'full.csv'
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_FILE', str(path))
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_AFTER_LAYER', '0')
    monkeypatch.setenv('MINI_LLM_MEMORY_AUDIT_STOP_USED_GIB', '0')
    probe = LayerMemoryAudit(32, np.zeros((1, 64, 768), np.float32))
    with pytest.raises(SystemExit) as result:
        probe.after_layer(31, {'attn_cache': np.zeros((8, 8), np.float32)})
    assert result.value.code == 0
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['event'] == 'complete'


def test_pinned_slab_projection_counts_distinct_views_of_same_storage():
    from mini_llm.activation_memory_audit import projected_offload_bytes
    base = np.zeros((128,), np.float32)
    view_a = base[:80]
    view_b = base[32:]
    packed, owner, arrays, repeated_owners = projected_offload_bytes({
        'first': view_a, 'second': view_b, 'same': view_a,
    })
    assert owner == base.nbytes
    assert arrays == 2
    assert repeated_owners == 1
    # 320 bytes, round to 512, then 384 bytes = 896 bytes
    assert packed == 896
    assert packed > owner
