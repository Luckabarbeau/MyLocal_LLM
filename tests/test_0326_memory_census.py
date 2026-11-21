"""CPU-only census contracts; fake owners forbid every payload access."""
import builtins
from copy import deepcopy
from itertools import count
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen4exp_memory_census as census


class Owner:
    def __init__(self, path, *, address=65536, nbytes=4096):
        self.filename = path
        self.offset = 0
        self.mode = 'r'
        self.flags = SimpleNamespace(writeable=False)
        self._mmap = SimpleNamespace(closed=False)
        self.ctypes = SimpleNamespace(data=address)
        self.nbytes = nbytes

    def __getitem__(self, index):
        pytest.fail('Census must not slice checkpoint payload')

    def __array__(self, *args, **kwargs):
        pytest.fail('Census must not convert checkpoint payload')


def descriptor(name, *, shard=0, offset=0, size=90, fmt='IQ4_NL'):
    return SimpleNamespace(name=name, shard_index=shard, data_offset=offset,
                           nbytes=size, quant_type=fmt)


def forbidden(*args, **kwargs):
    pytest.fail('Unexpected payload accessor or residency query')


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'fixture.gguf'
    path.write_bytes(bytes(4096))
    name = 'per_layer_token_embd.weight'
    result = SimpleNamespace(closed=False, paths=(path,), _owners=[Owner(path)],
        tensors={name: descriptor(name)}, rows=forbidden, tensor_bytes=forbidden,
        packed_weight=forbidden)
    monkeypatch.setattr(census, 'resident_bytes', forbidden)
    monkeypatch.setattr(census, '_io_snapshot', lambda: dict(read_bytes=0, rchar=100))
    return result


def test_payload_free_grouping_formats_totals_io_and_source_unchanged(store, monkeypatch):
    path = store.paths[0]
    before_bytes, before_stat = path.read_bytes(), path.stat()
    entries = [('blk.0.ffn_gate_exps.weight', 100, 'IQ3_S'),
               ('per_layer_token_embd.weight', 90, 'IQ4_NL'),
               ('blk.0.ple_key.weight', 80, 'F32'),
               ('blk.0.ple_norm_query.weight', 70, 'F32'),
               ('token_embd.weight', 60, 'IQ4_XS'),
               ('blk.0.ffn_gate_exps.weight.extra', 50, 'F16')]
    store.tensors = {name: descriptor(name, offset=i * 128, size=size, fmt=fmt)
                     for i, (name, size, fmt) in enumerate(entries)}
    original = deepcopy(store.tensors)
    calls = []

    def resident(address, length):
        calls.append((address, length))
        return length // 2

    monkeypatch.setattr(census, 'resident_bytes', resident)
    snapshots = iter([dict(read_bytes=1, rchar=10), dict(read_bytes=2, rchar=20)])
    monkeypatch.setattr(census, '_io_snapshot', lambda: next(snapshots))
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(census.time, 'monotonic', lambda: next(ticks))
    report = census.checkpoint_census(store)
    assert calls == [(65536 + i * 128, size) for i, (_, size, _) in enumerate(entries)]
    assert [t['group'] for t in report['tensors']] == [
        'routed_experts', 'ple_table', 'ple_operators', 'ple_operators',
        'dense_other', 'dense_other']
    assert [t['format'] for t in report['tensors']] == [fmt for _, _, fmt in entries]
    assert report['groups']['ple_operators'] == dict(
        tensor_count=2, bytes=150, resident_bytes=75, nonresident_bytes=75)
    assert report['groups']['dense_other']['bytes'] == 110
    assert report['totals'] == dict(tensor_count=6, bytes=450,
                                    resident_bytes=225, nonresident_bytes=225)
    assert report['io_before_census'] == dict(read_bytes=1, rchar=10)
    assert report['io_after_census'] == dict(read_bytes=2, rchar=20)
    assert report['elapsed_seconds'] == .25
    assert not report['payload_reads'] and not report['checkpoint_writes']
    assert 'PTE' in report['scope'] and 'unique physical' in report['scope']
    assert any('race' in warning for warning in report['warnings'])
    shard = report['shards'][0]
    assert shard['size_bytes'] == before_stat.st_size
    assert shard['mtime_ns'] == before_stat.st_mtime_ns
    assert shard['uid'] == os.geteuid()
    assert shard['inode'] == before_stat.st_ino and shard['device'] == before_stat.st_dev
    assert not shard['identity_changed']
    assert store.tensors == original
    assert store._owners[0].mode == 'r' and not store._owners[0].flags.writeable
    assert path.read_bytes() == before_bytes
    after_stat = path.stat()
    assert (after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ctime_ns) == (
        before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ctime_ns)


def test_multiple_shards_empty_groups_and_numpy_integers(store, tmp_path, monkeypatch):
    second = tmp_path / 'second.gguf'
    second.write_bytes(bytes(4096))
    store.paths += (second,)
    store._owners.append(Owner(second, address=131072))
    desc = next(iter(store.tensors.values()))
    desc.shard_index, desc.data_offset, desc.nbytes = np.int64(1), np.int64(4095), np.int64(1)
    calls = []
    monkeypatch.setattr(census, 'resident_bytes', lambda a, n: (calls.append((a, n)), 1)[1])
    report = census.checkpoint_census(store)
    assert calls == [(131072 + 4095, 1)]
    assert len(report['shards']) == 2
    assert report['groups']['routed_experts'] == dict(
        tensor_count=0, bytes=0, resident_bytes=0, nonresident_bytes=0)


@pytest.mark.parametrize('field', ['shard_index', 'data_offset', 'nbytes'])
@pytest.mark.parametrize('value', [True, np.bool_(False), -1, 1.0, float('nan'),
                                   float('inf'), '1', None, sys.maxsize + 1])
def test_descriptor_integer_validation_before_query(store, field, value):
    setattr(next(iter(store.tensors.values())), field, value)
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


@pytest.mark.parametrize('changes', [dict(shard_index=1), dict(nbytes=0),
    dict(data_offset=4096), dict(data_offset=4090, nbytes=7),
    dict(data_offset=sys.maxsize, nbytes=1), dict(quant_type=''), dict(quant_type=1)])
def test_invalid_descriptor_ranges_and_formats(store, changes):
    desc = next(iter(store.tensors.values()))
    for name, value in changes.items():
        setattr(desc, name, value)
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


def test_all_descriptors_validated_before_any_query(store):
    store.tensors['bad'] = descriptor('bad', offset=4096)
    with pytest.raises(ValueError, match='beyond mapped owner'):
        census.checkpoint_census(store)


@pytest.mark.parametrize('field,value', [('nbytes', True), ('nbytes', 0),
    ('nbytes', 4097), ('nbytes', float('nan')), ('offset', 1), ('offset', False),
    ('mode', 'r+'), ('mode', 'c')])
def test_invalid_owner_metadata(store, field, value):
    setattr(store._owners[0], field, value)
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


@pytest.mark.parametrize('address', [0, -1, True, 1.0, float('inf'), sys.maxsize])
def test_invalid_owner_address(store, address):
    store._owners[0].ctypes.data = address
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


@pytest.mark.parametrize('state', ['store_closed', 'mapping_closed', 'unmapped', 'writable'])
def test_closed_or_writable_source(store, state):
    owner = store._owners[0]
    if state == 'store_closed':
        store.closed = True
    elif state == 'mapping_closed':
        owner._mmap.closed = True
    elif state == 'unmapped':
        owner._mmap = None
    else:
        owner.flags.writeable = True
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


def test_unowned_path_rejected_before_query(store, monkeypatch):
    monkeypatch.setattr(census.os, 'geteuid', lambda: store.paths[0].stat().st_uid + 1)
    with pytest.raises(ValueError, match='effective UID'):
        census.checkpoint_census(store)


def test_owner_filename_mismatch(store, tmp_path):
    other = tmp_path / 'other.gguf'
    other.write_bytes(bytes(4096))
    store._owners[0].filename = other
    with pytest.raises(ValueError, match='filename'):
        census.checkpoint_census(store)


def test_nonregular_missing_and_duplicate_paths(store, tmp_path):
    source = store.paths[0]
    store.paths = (tmp_path,)
    with pytest.raises(ValueError, match='regular file'):
        census.checkpoint_census(store)
    store.paths = (tmp_path / 'missing',)
    with pytest.raises(FileNotFoundError):
        census.checkpoint_census(store)
    store.paths = (source, source)
    store._owners *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        census.checkpoint_census(store)


@pytest.mark.parametrize('field', ['paths', '_owners', 'tensors'])
def test_empty_metadata_rejected(store, field):
    setattr(store, field, {} if field == 'tensors' else [])
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


def test_owner_count_mismatch(store):
    store._owners *= 2
    with pytest.raises(ValueError, match='one mapped owner'):
        census.checkpoint_census(store)


@pytest.mark.parametrize('field,limit', [('paths', 64), ('_owners', 64), ('tensors', 4096)])
def test_metadata_iterators_are_consumed_only_to_limit_plus_one(store, field, limit):
    seen = []

    def entries():
        for i in count():
            seen.append(i)
            yield (str(i), descriptor(str(i))) if field == 'tensors' else i

    setattr(store, field, SimpleNamespace(items=entries) if field == 'tensors' else entries())
    with pytest.raises(ValueError):
        census.checkpoint_census(store)
    assert len(seen) == limit + 1


def test_exact_tensor_and_shard_limits_are_accepted(store, tmp_path, monkeypatch):
    paths = []
    for i in range(64):
        path = tmp_path / f'shard{i}.gguf'
        path.write_bytes(bytes(4096))
        paths.append(path)
    store.paths = paths
    store._owners = [Owner(path, address=65536 * (i + 1)) for i, path in enumerate(paths)]
    store.tensors = {str(i): descriptor(str(i), shard=i % 64, size=1) for i in range(4096)}
    monkeypatch.setattr(census, 'resident_bytes', lambda a, n: 0)
    report = census.checkpoint_census(store)
    assert len(report['shards']) == 64
    assert report['totals']['tensor_count'] == report['totals']['bytes'] == 4096


@pytest.mark.parametrize('name', ['', 1])
def test_invalid_tensor_names(store, name):
    store.tensors = {name: descriptor(name)}
    with pytest.raises(ValueError, match='name'):
        census.checkpoint_census(store)


@pytest.mark.parametrize('value', [True, -1, 91, float('nan'), 1.5])
def test_invalid_residency_measurements_are_not_reported(store, monkeypatch, value):
    monkeypatch.setattr(census, 'resident_bytes', lambda a, n: value)
    with pytest.raises(ValueError):
        census.checkpoint_census(store)


def test_residency_failure_propagates(store, monkeypatch):
    def fail(*args):
        raise OSError('mincore failed')
    monkeypatch.setattr(census, 'resident_bytes', fail)
    with pytest.raises(OSError, match='mincore failed'):
        census.checkpoint_census(store)


def test_path_identity_change_is_visible_not_an_atomic_snapshot_claim(store, monkeypatch):
    identity = census._identity
    calls = []
    def changed(path):
        result = identity(path)
        if calls:
            result['mtime_ns'] += 1
        calls.append(path)
        return result
    monkeypatch.setattr(census, '_identity', changed)
    monkeypatch.setattr(census, 'resident_bytes', lambda a, n: n)
    shard = census.checkpoint_census(store)['shards'][0]
    assert shard['identity_changed']
    assert shard['stat_after']['mtime_ns'] == shard['mtime_ns'] + 1


def test_import_does_not_import_model_or_gpu(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(('mini_llm', 'cupy', 'torch'))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    namespace = runpy.run_path(census.__file__)
    assert callable(namespace['checkpoint_census'])


@pytest.mark.skipif(sys.platform != 'linux' or not Path('/proc/self/io').is_file(),
                    reason='Linux mincore and procfs required')
def test_tiny_real_readonly_memmap_without_checkpoint_or_model(tmp_path):
    import ctypes
    if not hasattr(ctypes.CDLL(None), 'mincore'):
        pytest.skip('mincore unavailable')
    path = tmp_path / 'tiny.bin'
    content = bytes(range(256)) * 32
    path.write_bytes(content)
    before = path.stat()
    owner = np.memmap(path, mode='r', dtype=np.uint8)
    store = SimpleNamespace(closed=False, paths=[path], _owners=[owner], tensors={
        'per_layer_token_embd.weight': descriptor('per_layer_token_embd.weight', offset=13, size=4097),
        'blk.0.ffn_down_exps.weight': descriptor('blk.0.ffn_down_exps.weight', offset=8191, size=1),
    }, rows=forbidden, tensor_bytes=forbidden)
    try:
        report = census.checkpoint_census(store)
        assert report['totals']['bytes'] == 4098
        assert 0 <= report['totals']['resident_bytes'] <= 4098
        for tensor in report['tensors']:
            assert tensor['resident_bytes'] + tensor['nonresident_bytes'] == tensor['bytes']
        assert not report['shards'][0]['identity_changed']
        assert isinstance(report['io_before_census']['read_bytes'], int)
        assert isinstance(report['io_after_census']['read_bytes'], int)
        assert not owner.flags.writeable and not owner._mmap.closed
    finally:
        owner._mmap.close()  # Test owns this tiny mapping; census exports no views.
    after = path.stat()
    assert path.read_bytes() == content
    assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_size, before.st_mtime_ns, before.st_ctime_ns)
