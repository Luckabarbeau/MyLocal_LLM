"""0078J: cache inventory and isolated post-validation GC / pool-trim probe."""

import csv
from types import SimpleNamespace

from mini_llm import train_extended
from mini_llm.diagnostics.gpu_cache_audit import (
    _cuda_array_bytes, persistent_gpu_cache_inventory,
)


class FakeCudaArray:
    __cuda_array_interface__ = {"shape": (1,), "typestr": "<f4", "data": (100, False), "version": 2}

    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_gpu_array_accounting_counts_nested_arrays_once():
    a = FakeCudaArray(2048)
    b = FakeCudaArray(4096)
    seen = set()
    total = _cuda_array_bytes({"a": (a, a), "b": [b], "cpu": [1, 2]}, seen)
    assert total == 6144
    assert len(seen) == 2


def test_inventory_recognizes_routing_cache_and_packed_weights(monkeypatch):
    from mini_llm.ops import rope, attention

    fake_cos = FakeCudaArray(1024)
    fake_sin = FakeCudaArray(1024)
    monkeypatch.setattr(rope, "_ROPE_TABLE_CACHE", {(100,): (fake_cos, fake_sin)})
    monkeypatch.setattr(attention.GQAAttention, "_causal_mask_cache", {8: FakeCudaArray(64)})
    router = SimpleNamespace(_routing_metadata_cache={4: {
        "route_starts": FakeCudaArray(128), "n_blocks": 8,
    }})
    block = SimpleNamespace(attention=SimpleNamespace(_packed_qkv_weight=FakeCudaArray(8192)))
    model = SimpleNamespace(memory_router=router, blocks=[block])
    inventory = persistent_gpu_cache_inventory(model)
    assert inventory["rope_tables"] == {"entries": 1, "cuda_bytes": 2048}
    assert inventory["causal_attention_masks"] == {"entries": 1, "cuda_bytes": 64}
    assert inventory["routing_metadata"] == {"entries": 1, "cuda_bytes": 128}
    assert inventory["packed_qkv_weights"] == {"entries": 1, "cuda_bytes": 8192}


def test_inventory_does_not_inspect_transient_forward_caches():
    # A model containing a large transient GPU object must not be traversed.
    model = SimpleNamespace(memory_router=None, blocks=[], cache=FakeCudaArray(1_000_000_000))
    inventory = persistent_gpu_cache_inventory(model)
    assert inventory["packed_qkv_weights"]["cuda_bytes"] == 0


def test_gc_probe_order_and_opt_in(monkeypatch):
    order = []
    trainer = train_extended.ExtendedTrainer.__new__(train_extended.ExtendedTrainer)
    trainer.gpu_memory_audit = True
    trainer.gpu_memory_gc_probe = True
    trainer.step = 50400
    monkeypatch.setattr(train_extended, "synchronize", lambda: order.append("sync"))
    monkeypatch.setattr(train_extended.gc, "collect", lambda: order.append("gc") or 3)
    trainer._gpu_memory_boundary = lambda phase, **kw: order.append((phase, kw))
    trainer._gpu_memory_post_validation_gc_probe()
    assert order == [
        "sync", "gc", ("after_validation_python_gc", {}),
        ("after_validation_python_gc_trimmed", {"trim_idle_pool": True})
    ]
    order.clear()
    trainer.gpu_memory_gc_probe = False
    trainer._gpu_memory_post_validation_gc_probe()
    assert order == []


def test_gpu_cache_csv_uses_separate_file(tmp_path, monkeypatch, capsys):
    trainer = train_extended.ExtendedTrainer.__new__(train_extended.ExtendedTrainer)
    trainer.step = 50400
    trainer.log_file = tmp_path / "train.csv"
    monkeypatch.setattr(
        "mini_llm.diagnostics.gpu_cache_audit.persistent_gpu_cache_inventory",
        lambda model: {"rope_tables": {"entries": 2, "cuda_bytes": 2 * 1024**2}},
    )
    inventory = trainer._gpu_cache_snapshot("after_all_validation")
    assert inventory["rope_tables"]["entries"] == 2
    with (tmp_path / "train.gpu_caches.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"step": "50400", "phase": "after_all_validation", "cache": "rope_tables", "entries": "2", "cuda_mib": "2.0"}]
    assert "rope_tables=2 entries/2.00 MiB" in capsys.readouterr().out
    assert not (tmp_path / "train.gpu_memory.csv").exists()


def test_gpu_cache_audit_remains_opt_in_at_boundary(monkeypatch, tmp_path):
    class Pool:
        used_bytes = lambda self: 2**20
        total_bytes = lambda self: 2**21

    trainer = train_extended.ExtendedTrainer.__new__(train_extended.ExtendedTrainer)
    trainer.gpu_memory_audit = True
    trainer.gpu_memory_cache_audit = False
    trainer.log_file = tmp_path / "train.csv"
    trainer.step = 50400
    monkeypatch.setattr(train_extended, "synchronize", lambda: None)
    monkeypatch.setattr(train_extended, "xp", SimpleNamespace(
        get_default_memory_pool=lambda: Pool(),
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (2**20, 2**24))),
    ))
    trainer._gpu_cache_snapshot = lambda *args: (_ for _ in ()).throw(AssertionError("cache audit enabled unexpectedly"))
    trainer._gpu_memory_boundary("before_validation")
    assert (tmp_path / "train.gpu_memory.csv").exists()
    assert not (tmp_path / "train.gpu_caches.csv").exists()
