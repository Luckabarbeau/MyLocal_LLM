"""Online, EOS-bounded fusion windows from disjoint teacher-validation shards.

No token-window or teacher-logit files are materialized.  The teacher's
validation shards are split by *whole files* into train/val/test for fusion;
training on the teacher's original train shards would bias fusion results.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
import numpy as np

from ..data.mixed_shards import load_weighted_shard_sources


def split_fusion_shards(sources, seed, *, min_shard_bytes=0):
    """Stable disjoint file splits; ignore sources without enough valid shards."""
    split = {key: {} for key in ('train', 'val', 'test')}
    for source in sorted(sources, key=lambda item: item.name):
        paths = list(source.val_paths)
        if min_shard_bytes:
            paths = [path for path in paths
                     if path.stat().st_size >= min_shard_bytes]
        if len(paths) < 3:
            warnings.warn(
                f'fusion: skipping source {source.name!r}: {len(paths)} validation '
                'shard(s), but independent train/val/test splits require >=3',
                RuntimeWarning, stacklevel=2)
            continue
        digest = hashlib.sha256(f'{seed}/{source.name}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        rng.shuffle(paths)
        nval = max(1, round(len(paths) * 0.15))
        ntest = max(1, round(len(paths) * 0.15))
        ntrain = len(paths) - nval - ntest
        if ntrain < 1:
            warnings.warn(
                f'fusion: skipping source {source.name!r}: empty fusion train '
                'split after reserving independent validation/test shards',
                RuntimeWarning, stacklevel=2)
            continue
        split['train'][source.name] = tuple(paths[:ntrain])
        split['val'][source.name] = tuple(paths[ntrain:ntrain+nval])
        split['test'][source.name] = tuple(paths[ntrain+nval:])
    if not split['train']:
        raise ValueError(
            'no eligible fusion sources: at least one corpus must contain '
            'three independent teacher-validation shard files')
    return split


def _outer_eos_bounds(path, dtype, eos, scan=1 << 18):
    data = np.memmap(path, mode='r', dtype=dtype)
    first = last = None
    for start in range(0, len(data), scan):
        hits = np.flatnonzero(data[start:start+scan] == eos)
        if len(hits):
            first = start + int(hits[0]); break
    for end in range(len(data), 0, -scan):
        start = max(0, end - scan)
        hits = np.flatnonzero(data[start:end] == eos)
        if len(hits):
            last = start + int(hits[-1]); break
    return data, first, last


class OnlineShardSampler:
    """Draw full-length context+1 windows without crossing an EOS in inputs."""
    def __init__(self, sources, paths, manifest, context_length, rng):
        self.sources = list(sources)
        self.names = [s.name for s in sources]
        self.weights = np.asarray([s.weight for s in sources], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.paths = {s.name: list(paths[s.name]) for s in sources}
        self.dtype = np.dtype(manifest.get('token_dtype', 'uint16'))
        self.eos = int(manifest['eos_token_id'])
        self.length = int(context_length) + 1
        self.rng = rng
        self.cache = {}
        if self.length < 3:
            raise ValueError('fusion context length must be >=2')

    def draw(self):
        for _ in range(500):
            # Shards can become exhausted at runtime (e.g. no EOS-bounded
            # window). Never keep sampling a source with no remaining files.
            eligible = [i for i, name in enumerate(self.names) if self.paths[name]]
            if not eligible:
                raise RuntimeError('all fusion shards exhausted: no EOS-bounded '
                                   'windows remain for the requested context length')
            probs = self.weights[eligible]
            probs = probs / probs.sum()
            name = self.names[int(self.rng.choice(eligible, p=probs))]
            paths = self.paths[name]
            sizes = np.array([p.stat().st_size for p in paths], dtype=np.float64)
            path = paths[int(self.rng.choice(len(paths), p=sizes / sizes.sum()))]
            if path not in self.cache:
                self.cache[path] = _outer_eos_bounds(path, self.dtype, self.eos)
            data, first, last = self.cache[path]
            if first is None or last is None or last - first < self.length:
                paths.remove(path)
                continue
            low = first + 1
            high = last - self.length + 1
            if high < low:
                paths.remove(path)
                continue
            offset = int(self.rng.integers(low, high + 1))
            tokens = np.asarray(data[offset:offset+self.length], dtype=np.int64)
            if not np.any(tokens[:-1] == self.eos):
                return tokens, name
        raise RuntimeError('unable to sample EOS-bounded online fusion window')

    def batch(self, count):
        windows, names = [], []
        for _ in range(int(count)):
            w, name = self.draw()
            windows.append(w)
            names.append(name)
        return np.stack(windows), names


def make_online_samplers(root, context_length, partition_seed, seed):
    sources, manifest = load_weighted_shard_sources(root)
    # The fusion split uses only teacher-validation files. Filter shards too
    # short for the requested context BEFORE splitting, so a tiny shard cannot
    # occupy the sole train, validation or test slot for an otherwise good source.
    dtype = np.dtype(manifest.get('token_dtype', 'uint16'))
    min_bytes = (int(context_length) + 3) * dtype.itemsize
    parts = split_fusion_shards(sources, partition_seed,
                                min_shard_bytes=min_bytes)
    # Both samplers must use the SAME eligible source set and renormalized
    # probabilities. Never fall back to pretraining shards or duplicate a file
    # across fusion train/val/test to compensate for an undersized corpus.
    sources = [source for source in sources
               if source.name in parts['train'] and float(source.weight) > 0]
    if not sources:
        raise ValueError(
            'no positive-weight fusion source has >=3 teacher-validation '
            f'shards large enough for context length {context_length}; '
            'reduce --context-length or provide additional validation shards')
    for partition in parts.values():
        for name in tuple(partition):
            if name not in {source.name for source in sources}:
                partition.pop(name)
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 10000)
    samplers = {
        'train': OnlineShardSampler(sources, parts['train'], manifest, context_length, train_rng),
        'val': OnlineShardSampler(sources, parts['val'], manifest, context_length, val_rng),
    }
    return samplers, manifest, parts
