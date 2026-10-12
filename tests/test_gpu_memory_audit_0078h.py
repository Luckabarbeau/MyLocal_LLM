"""0078H regression coverage for optional CUDA memory measurements."""
import csv
from types import SimpleNamespace

from mini_llm import train_extended


class FakePool:
    def __init__(self):
        self.live = 100 * 1024 * 1024
        self.reserved = 300 * 1024 * 1024
        self.trim_count = 0

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.reserved

    def free_all_blocks(self):
        self.trim_count += 1
        self.reserved = self.live


def _trainer(tmp_path, enabled=True):
    instance = train_extended.ExtendedTrainer.__new__(train_extended.ExtendedTrainer)
    instance.gpu_memory_audit = enabled
    instance.gpu_memory_trim_after_validation = False
    instance.step = 48000
    instance.log_file = tmp_path / 'test_run.csv'
    return instance


def test_audit_records_live_reserved_and_driver_free(monkeypatch, tmp_path, capsys):
    pool = FakePool()
    fake = SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (600 * 1024**2, 1024**3))),
    )
    sync_count = []
    monkeypatch.setattr(train_extended, 'xp', fake)
    monkeypatch.setattr(train_extended, 'synchronize', lambda: sync_count.append(1))
    t = _trainer(tmp_path)
    before = t._gpu_memory_boundary('before_checkpoint_save')
    assert before['cupy_live_mib'] == 100
    assert before['cupy_reserved_mib'] == 300
    assert before['cupy_idle_mib'] == 200
    assert before['device_free_mib'] == 600
    assert pool.trim_count == 0
    after = t._gpu_memory_boundary('after_validation_pool_trim', trim_idle_pool=True)
    assert after['cupy_live_mib'] == 100
    assert after['cupy_reserved_mib'] == 100
    assert after['cupy_idle_mib'] == 0
    assert pool.trim_count == 1
    assert len(sync_count) == 2
    assert 'before_checkpoint_save' in capsys.readouterr().out
    with (tmp_path / 'test_run.gpu_memory.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['phase'] for row in rows] == ['before_checkpoint_save', 'after_validation_pool_trim']


def test_disabled_audit_does_not_call_cuda(monkeypatch, tmp_path):
    t = _trainer(tmp_path, enabled=False)
    monkeypatch.setattr(train_extended, 'xp', SimpleNamespace(
        get_default_memory_pool=lambda: (_ for _ in ()).throw(AssertionError('CUDA used')),
    ))
    assert t._gpu_memory_boundary('before_validation') is None
    assert not (tmp_path / 'test_run.gpu_memory.csv').exists()


def test_cpu_backend_no_op(monkeypatch, tmp_path):
    t = _trainer(tmp_path)
    monkeypatch.setattr(train_extended, 'xp', SimpleNamespace())
    assert t._gpu_memory_boundary('after_all_validation') is None
