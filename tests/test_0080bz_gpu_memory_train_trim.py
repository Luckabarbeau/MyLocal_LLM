"""0080BZ: deterministic periodic idle-pool trim at optimizer boundaries."""
from types import SimpleNamespace

from mini_llm import train_extended


class FakePool:
    def __init__(self):
        self.live = 100 * 1024**2
        self.reserved = 300 * 1024**2
        self.calls = 0

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.reserved

    def free_all_blocks(self):
        self.calls += 1
        self.reserved = self.live


def _trainer(interval):
    trainer = train_extended.ExtendedTrainer.__new__(train_extended.ExtendedTrainer)
    trainer.gpu_memory_trim_train_interval = interval
    trainer.gpu_memory_audit = False
    trainer.step = 0
    trainer.log_file = None
    return trainer


def test_no_trim_when_disabled(monkeypatch):
    trainer = _trainer(0)
    trainer.step = 50
    monkeypatch.setattr(trainer, "_gpu_memory_boundary", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected trim")))
    assert trainer._gpu_memory_maybe_trim_after_training_step() is None


def test_trim_only_at_completed_step_multiples(monkeypatch):
    trainer = _trainer(5)
    called = []
    monkeypatch.setattr(trainer, "_gpu_memory_boundary", lambda phase, **kwargs: called.append((trainer.step, phase, kwargs)))
    for step in range(0, 13):
        trainer.step = step
        trainer._gpu_memory_maybe_trim_after_training_step()
    assert called == [
        (5, "after_training_step_trimmed", {"trim_idle_pool": True}),
        (10, "after_training_step_trimmed", {"trim_idle_pool": True}),
    ]


def test_pool_trim_reclaims_only_idle_blocks_even_without_audit(monkeypatch):
    pool = FakePool()
    syncs = []
    monkeypatch.setattr(train_extended, "synchronize", lambda: syncs.append(1))
    monkeypatch.setattr(train_extended, "xp", SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (700 * 1024**2, 16 * 1024**3))),
    ))
    trainer = _trainer(5)
    trainer.step = 5
    record = trainer._gpu_memory_maybe_trim_after_training_step()
    assert pool.calls == 1
    assert syncs == [1]
    assert record["cupy_live_mib"] == 100
    assert record["cupy_reserved_mib"] == 100
    assert record["cupy_idle_mib"] == 0
    assert record["trimmed"] == 1


def test_trim_cpu_backend_is_noop(monkeypatch):
    monkeypatch.setattr(train_extended, "xp", SimpleNamespace())
    trainer = _trainer(5)
    trainer.step = 5
    assert trainer._gpu_memory_maybe_trim_after_training_step() is None
