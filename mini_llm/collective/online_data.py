"""Online, EOS-bounded fusion windows from disjoint teacher-validation shards.

No token-window or teacher-logit files are materialized.  The teacher's
validation shards are split by *whole files* into train/val/test for fusion;
training on the teacher's original train shards would bias fusion results.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from ..data.mixed_shards import load_weighted_shard_sources


def split_fusion_shards(sources, seed):
    """Stable per-source file split, compatible with the 0082B partitions."""
    split = {key: {} for key in ('train', 'val', 'test')}
    for source in sorted(sources, key=lambda item: item.name):
        paths = list(source.val_paths)
        if len(paths) < 3:
            raise ValueError(f'{source.name}: >=3 validation shards needed for fusion splits')
        digest = hashlib.sha256(f'{seed}/{source.name}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        rng.shuffle(paths)
        nval = max(1, round(len(paths) * 0.15))
        ntest = max(1, round(len(paths) * 0.15))
        ntrain = len(paths) - nval - ntest
        if ntrain < 1:
            raise ValueError(f'{source.name}: empty fusion train split')
        split['train'][source.name] = tuple(paths[:ntrain])
        split['val'][source.name] = tuple(paths[ntrain:ntrain+nval])
        split['test'][source.name] = tuple(paths[ntrain+nval:])
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
            name = self.names[int(self.rng.choice(len(self.names), p=self.weights))]
            paths = self.paths[name]
            if not paths:
                raise RuntimeError(f'no eligible fusion shard files for {name}')
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
    parts = split_fusion_shards(sources, partition_seed)
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 10000)
    samplers = {
        'train': OnlineShardSampler(sources, parts['train'], manifest, context_length, train_rng),
        'val': OnlineShardSampler(sources, parts['val'], manifest, context_length, val_rng),
    }
    return samplers, manifest, parts
