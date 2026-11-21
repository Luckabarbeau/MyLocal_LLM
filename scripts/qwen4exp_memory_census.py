"""Payload-free census of caller-owned canonical checkpoint mappings.

No model/backend is imported, checkpoint is opened, or advice is applied. The
caller must keep the store/mappings alive and serialize their retirement for the
entire call. File ownership is required to avoid potentially redacted mincore
results. This helper neither controls residency nor writes diagnostic outputs.
"""
from itertools import islice
from numbers import Integral
import os
from pathlib import Path
import stat
import sys
import time

from scripts.probe_0308_page_residency import resident_bytes


_MAX_TENSORS = 4096
_MAX_SHARDS = 64
_GROUPS = ('routed_experts', 'ple_table', 'ple_operators', 'dense_other')


def _integer(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not minimum <= value <= sys.maxsize):
        raise ValueError(f'{name} must be a non-bool integer in [{minimum}, sys.maxsize]')
    return int(value)


def _bounded(values, maximum, name):
    result = list(islice(iter(values), maximum + 1))
    if not result or len(result) > maximum:
        raise ValueError(f'{name} must contain 1..{maximum} entries')
    return result


def _identity(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Checkpoint path must be a local regular file')
    if info.st_uid != os.geteuid():
        raise ValueError('Checkpoint path must be owned by the effective UID')
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid,
                size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def _io_snapshot():
    return {key: int(value) for key, value in
            (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}


def _group(name):
    if name.endswith('_exps.weight'):
        return 'routed_experts'
    if name == 'per_layer_token_embd.weight':
        return 'ple_table'
    if 'ple' in name:
        return 'ple_operators'
    return 'dense_other'


def _totals():
    return dict(tensor_count=0, bytes=0, resident_bytes=0, nonresident_bytes=0)


def checkpoint_census(store):
    """Return a phase-independent, bounded metadata/file-page snapshot.

    Requires the canonical store contract: one live read-only mapping per path,
    mapped from file offset zero; descriptor offsets are absolute shard offsets.
    Only pointer/length metadata is inspected, never payload views or values.
    All input metadata is validated before calling ``resident_bytes``. Its
    scratch remains bounded by the existing 8192-page mincore implementation.

    Missing procfs/mincore support and syscall failures propagate to the caller;
    a failed measurement must not silently become a zero-residency observation.
    """
    started = time.monotonic()
    io_before = _io_snapshot()
    if store.closed:
        raise ValueError('Checkpoint store is closed')
    paths = _bounded(store.paths, _MAX_SHARDS, 'shard paths')
    owners = _bounded(store._owners, _MAX_SHARDS, 'shard owners')
    items = _bounded(store.tensors.items(), _MAX_TENSORS, 'tensors')
    if len(paths) != len(owners):
        raise ValueError('Require one mapped owner per shard path')

    shards, layouts, seen_paths = [], [], set()
    for source, owner in zip(paths, owners):
        path = Path(source).resolve(strict=True)
        if path in seen_paths:
            raise ValueError('Duplicate checkpoint shard path')
        seen_paths.add(path)
        identity = _identity(path)
        if (owner.mode != 'r' or owner.flags.writeable
                or owner._mmap is None or owner._mmap.closed):
            raise ValueError('Checkpoint owner must be a live read-only mapping')
        if Path(owner.filename).resolve(strict=True) != path:
            raise ValueError('Mapped owner filename does not match shard path')
        if _integer(owner.offset, 'mapping offset') != 0:
            raise ValueError('Canonical shard mapping must start at file offset zero')
        length = _integer(owner.nbytes, 'mapping bytes', 1)
        address = _integer(owner.ctypes.data, 'mapping address', 1)
        if length > identity['size_bytes'] or address + length > sys.maxsize:
            raise ValueError('Mapping exceeds file or address bounds')
        layouts.append((address, length))
        shards.append(dict(path=str(path), mapping_bytes=length, **identity))

    tensors = []
    for name, desc in items:
        if not isinstance(name, str) or not name:
            raise ValueError('Tensor name must be a nonempty string')
        shard = _integer(desc.shard_index, 'shard index')
        if shard >= len(layouts):
            raise ValueError('Tensor shard index is outside mapped owners')
        offset = _integer(desc.data_offset, 'tensor offset')
        length = _integer(desc.nbytes, 'tensor bytes', 1)
        if offset + length > layouts[shard][1]:
            raise ValueError('Tensor extends beyond mapped owner')
        fmt = desc.quant_type
        if not isinstance(fmt, str) or not fmt:
            raise ValueError('Tensor format must be a nonempty descriptor string')
        tensors.append(dict(name=name, group=_group(name), shard_index=shard,
                            format=fmt, data_offset=offset, bytes=length))

    groups = {name: _totals() for name in _GROUPS}
    totals = _totals()
    for tensor in tensors:
        address = layouts[tensor['shard_index']][0] + tensor['data_offset']
        resident = _integer(resident_bytes(address, tensor['bytes']), 'resident bytes')
        if resident > tensor['bytes']:
            raise ValueError('Resident bytes exceed tensor size')
        tensor.update(resident_bytes=resident,
                      nonresident_bytes=tensor['bytes'] - resident)
        for summary in (groups[tensor['group']], totals):
            summary['tensor_count'] += 1
            for key in ('bytes', 'resident_bytes', 'nonresident_bytes'):
                summary[key] += tensor[key]

    for shard in shards:
        after = _identity(Path(shard['path']))
        shard['identity_changed'] = any(shard[key] != value for key, value in after.items())
        shard['stat_after'] = after
    io_after = _io_snapshot()
    return dict(
        schema_version=1,
        scope='Canonical file-page residency snapshot, not cgroup charge, PTE residency or unique physical RAM',
        warnings=[
            'mincore reports file-page residency, not cgroup ownership/charges or process PTE residency.',
            'Tensor totals are byte-weighted payload sums, not unique physical page accounting; edge pages may be shared.',
            'Snapshots race with reads, reclaim and file/path changes; shard stat identities do not make the census atomic.',
            'Process I/O snapshots include concurrent work and diagnostic overhead, not per-tensor or decode-only I/O.',
        ],
        payload_reads=False, checkpoint_writes=False,
        shards=shards, tensors=tensors, groups=groups, totals=totals,
        io_before_census=io_before, io_after_census=io_after,
        elapsed_seconds=time.monotonic() - started)
