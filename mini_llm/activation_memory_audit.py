"""0080AY: opt-in, forward-only GPU activation-cache memory investigation.

This module is observational.  It never moves or retains activation arrays;
all references created by the traversal are discarded after each sampled layer.
"""

import csv
import dataclasses
import os
import sys
from collections import defaultdict


def _storage(array):
    """Underlying storage identity and owner bytes (not view nbytes)."""
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            mem = array.data.mem
            return (("cupy", int(array.device.id), int(mem.ptr)),
                    int(mem.size), int(array.nbytes))
    except ImportError:
        pass
    import numpy as np
    if isinstance(array, np.ndarray):
        owner = array
        seen = set()
        while isinstance(getattr(owner, "base", None), np.ndarray):
            if id(owner) in seen:
                break
            seen.add(id(owner))
            owner = owner.base
        return (("numpy", id(owner)), int(owner.nbytes), int(array.nbytes))
    return None


def activation_storage_details(cache, max_nodes=100000):
    """Summarize unique underlying storage referenced by a backward cache.

    Traverses containers and a short allow-list of metadata dataclasses, NOT
    arbitrary module objects (which may own model weights or permanent caches).
    Return (owned_bytes, apparent_view_bytes, groups, top_paths, storages).
    """
    storages = {}
    group_storage = defaultdict(set)
    paths = []
    visited = set()
    count = 0
    stack = [(cache, "cache", "other")]
    while stack:
        obj, path, group = stack.pop()
        count += 1
        if count > max_nodes:
            raise RuntimeError("activation audit traversal exceeded max_nodes")
        storage = _storage(obj)
        if storage is not None:
            key, size, apparent = storage
            if key not in storages:
                storages[key] = size
                paths.append((size, path, apparent))
            group_storage[group].add(key)
            continue
        ident = id(obj)
        if ident in visited:
            continue
        visited.add(ident)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'module':
                    continue  # retrieval references trainable model modules
                next_group = str(k) if path == 'cache' else group
                stack.append((v, f"{path}.{k}", next_group))
        elif isinstance(obj, (list, tuple)):
            for index, v in enumerate(obj):
                stack.append((v, f"{path}[{index}]", group))
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for field in dataclasses.fields(obj):
                stack.append((getattr(obj, field.name), f"{path}.{field.name}", group))
        elif obj.__class__.__name__ in {'RoutingPlan'} and hasattr(obj, '__dict__'):
            for key, value in vars(obj).items():
                stack.append((value, f"{path}.{key}", group))
    grouped = {name: sum(storages[k] for k in keys) for name, keys in group_storage.items()}
    return sum(storages.values()), sum(item[2] for item in paths), grouped, sorted(paths, reverse=True), storages


def projected_offload_bytes(cache):
    """Estimate the 0080AT/AU/AW pinned slab without copying any arrays.

    That offloader deduplicated by ndarray Python identity, not by underlying
    allocation identity. This counts the exact aligned packed array sizes
    for the cache shapes supported by its serializer. No tensors are retained.
    Returns (packed_bytes, unique_owner_bytes, array_count, shared_owner_views).
    """
    seen_array_ids = set()
    seen_owners = {}
    visited_containers = set()
    arrays = []
    stack = [cache]
    while stack:
        obj = stack.pop()
        owner = _storage(obj)
        if owner is not None:
            if id(obj) in seen_array_ids:
                continue
            seen_array_ids.add(id(obj))
            key, size, _view_bytes = owner
            shared = key in seen_owners
            seen_owners.setdefault(key, size)
            arrays.append((int(obj.nbytes), shared))
            continue
        ident = id(obj)
        if ident in visited_containers:
            continue
        visited_containers.add(ident)
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (tuple, list)):
            stack.extend(obj)
        elif obj.__class__.__name__ == 'RoutingPlan' and hasattr(obj, '__dict__'):
            stack.extend(vars(obj).values())
    total = 0
    for size, _shared in arrays:
        total = (total + 255) & ~255
        total += size
    return total, sum(seen_owners.values()), len(arrays), sum(shared for _, shared in arrays)


def _host_snapshot():
    """Host memory counters; read-only Linux procfs and pinned-pool inventory."""
    values = {}
    try:
        with open('/proc/self/status', encoding='utf-8') as fp:
            for line in fp:
                if line.startswith(('VmRSS:', 'VmLck:')):
                    name, amount = line.split(':', 1)
                    values[name] = float(amount.strip().split()[0]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open('/proc/meminfo', encoding='utf-8') as fp:
            for line in fp:
                if line.startswith('MemAvailable:'):
                    values['MemAvailable'] = float(line.split()[1]) / 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return {
        'host_rss_mib': round(values.get('VmRSS', 0.0), 2),
        'host_locked_mib': round(values.get('VmLck', 0.0), 2),
        'host_available_mib': round(values.get('MemAvailable', 0.0), 2),
    }


def _gpu_snapshot():
    import cupy as cp
    cp.cuda.get_current_stream().synchronize()
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    pinned = cp.get_default_pinned_memory_pool()
    return {
        **_host_snapshot(),
        'pinned_pool_free_blocks': pinned.n_free_blocks(),
        'pool_live_mib': pool.used_bytes() / (1024 ** 2),
        'pool_reserved_mib': pool.total_bytes() / (1024 ** 2),
        'device_used_gib': (total - free) / (1024 ** 3),
        'device_free_gib': free / (1024 ** 3),
    }


class LayerMemoryAudit:
    def __init__(self, n_layers, x):
        self.max_layers = int(os.environ.get('MINI_LLM_MEMORY_AUDIT_STOP_AFTER_LAYER', '8'))
        self.stop_used_gib = float(os.environ.get('MINI_LLM_MEMORY_AUDIT_STOP_USED_GIB', '12'))
        self.n_layers = n_layers
        self.file = os.environ.get('MINI_LLM_MEMORY_AUDIT_FILE',
                                   'logs/0080AY_layer_memory.csv')
        os.makedirs(os.path.dirname(self.file) or '.', exist_ok=True)
        self.fp = open(self.file, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.fp, fieldnames=[
            'layer', 'batch', 'positions', 'width', 'device_used_gib',
            'device_free_gib', 'pool_live_mib', 'pool_reserved_mib',
            'delta_live_mib', 'delta_reserved_mib', 'all_cache_owner_mib',
            'new_cache_owner_mib', 'layer_cache_owner_mib',
            'attn_owner_mib', 'moe_owner_mib', 'norm1_owner_mib',
            'norm2_owner_mib', 'residual_owner_mib', 'largest_cache_arrays',
            'event', 'note', 'host_rss_mib', 'host_locked_mib',
            'host_available_mib', 'pinned_pool_free_blocks',
            'offload_packed_mib', 'offload_owner_mib',
            'offload_duplicate_mib', 'offload_projected_total_gib',
            'offload_array_count', 'offload_shared_owner_views',
        ])
        self.writer.writeheader()
        self.fp.flush()
        self.prev = _gpu_snapshot()
        self.seen = set()
        self.seen_sizes = {}
        self.projected_offload_bytes = 0
        self.batch, self.positions, self.width = map(int, x.shape)
        print(f"[0080AY] PRE-FORWARD cache audit: B={self.batch}, T={self.positions}, D={self.width}, "
              f"depth={n_layers}, pool_live={self.prev['pool_live_mib']:.0f} MiB, "
              f"device_used={self.prev['device_used_gib']:.2f} GiB. "
              f"CSV={self.file}", file=sys.stderr, flush=True)

    def before_layer(self, layer_idx):
        # Durable breadcrumb: survives OOM inside block.forward, when after_layer
        # would otherwise never be reached.
        snap = _gpu_snapshot()
        self.writer.writerow({
            'layer': layer_idx, 'batch': self.batch,
            'positions': self.positions, 'width': self.width,
            'event': 'begin', **snap,
        })
        self.fp.flush()
        print(f"[0080AZ] BEGIN L{layer_idx:02d} "
              f"live={snap['pool_live_mib']:.0f}MiB "
              f"device_free={snap['device_free_gib']:.2f}GiB "
              f"host_RSS={snap.get('host_rss_mib', 0):.0f}MiB",
              file=sys.stderr, flush=True)

    def forward_failure(self, layer_idx, error):
        """Best-effort emergency record without allocating CuPy arrays."""
        try:
            snap = _gpu_snapshot()
        except Exception:
            snap = _host_snapshot()
        note = f'{type(error).__name__}: {error}'[:1000]
        self.writer.writerow({
            'layer': layer_idx, 'batch': self.batch,
            'positions': self.positions, 'width': self.width,
            'event': 'oom', 'note': note, **snap,
        })
        self.fp.flush()
        self.close()
        print(f'[0080AZ] OOM in layer {layer_idx}: {note}',
              file=sys.stderr, flush=True)
        print(f'[0080AZ] Last complete layer and OOM are recorded in {self.file}',
              file=sys.stderr, flush=True)

    def after_layer(self, layer_idx, cache):
        # A temporary traversal of the cache only; no CUDA data transfers.
        own, _views, grouped, paths, storages = activation_storage_details(cache)
        packed, packed_owner, packed_count, shared_owner_views = projected_offload_bytes(cache)
        self.projected_offload_bytes += packed
        new_keys = set(storages) - self.seen
        new_mib = sum(storages[k] for k in new_keys) / 1048576
        self.seen.update(storages)
        self.seen_sizes.update(storages)
        snapshot = _gpu_snapshot()
        def mib(value):
            return round(value / 1048576, 2)
        groups = {
            'attn_owner_mib': mib(grouped.get('attn_cache', 0)),
            'moe_owner_mib': mib(grouped.get('moe_cache', 0)),
            'norm1_owner_mib': mib(grouped.get('norm1_cache', 0)),
            'norm2_owner_mib': mib(grouped.get('norm2_cache', 0)),
            'residual_owner_mib': mib(grouped.get('residual1', 0) + grouped.get('residual2', 0)),
        }
        row = {
            'layer': layer_idx, 'batch': self.batch, 'positions': self.positions,
            'width': self.width, **snapshot,
            'delta_live_mib': round(snapshot['pool_live_mib'] - self.prev['pool_live_mib'], 2),
            'delta_reserved_mib': round(snapshot['pool_reserved_mib'] - self.prev['pool_reserved_mib'], 2),
            'layer_cache_owner_mib': mib(own),
            'new_cache_owner_mib': round(new_mib, 2),
            # All previous caches still live: cumulative newly observed storage.
            'all_cache_owner_mib': round(sum(self.seen_sizes.values())/1048576, 2),
            'event': 'complete',
            'offload_packed_mib': mib(packed),
            'offload_owner_mib': mib(packed_owner),
            'offload_duplicate_mib': mib(max(0, packed-packed_owner)),
            'offload_projected_total_gib': round(self.projected_offload_bytes / 2**30, 3),
            'offload_array_count': packed_count,
            'offload_shared_owner_views': shared_owner_views,
            'largest_cache_arrays': ' | '.join(
                f"{mib(sz):.0f}MiB:{path[:95]}" for sz, path, _ in paths[:4]),
            **groups,
        }
        self.writer.writerow(row)
        self.fp.flush()
        print(f"[0080AZ] END L{layer_idx:02d} live={snapshot['pool_live_mib']:.0f} MiB "
              f"(+{row['delta_live_mib']:.0f}) cache={row['layer_cache_owner_mib']:.0f} "
              f"new={new_mib:.0f} attn={groups['attn_owner_mib']:.0f} "
              f"moe={groups['moe_owner_mib']:.0f} "
              f"packed_est={mib(packed):.0f}MiB "
              f"host={snapshot.get('host_rss_mib',0):.0f}MiB "
              f"device={snapshot['device_used_gib']:.1f} GiB", file=sys.stderr, flush=True)
        self.prev = snapshot
        should_stop = (layer_idx + 1 == self.n_layers) or (
            self.max_layers > 0 and layer_idx + 1 >= self.max_layers) or (
            self.stop_used_gib > 0 and snapshot['device_used_gib'] >= self.stop_used_gib)
        if should_stop:
            self.close()
            print(f"[0080AZ] AUDIT COMPLETE: intentional stop after layer {layer_idx}. "
                  f"No backward or optimizer update was performed. CSV={self.file}",
                  file=sys.stderr, flush=True)
            raise SystemExit(0)

    def close(self):
        if not self.fp.closed:
            self.fp.close()
