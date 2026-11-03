"""0105D: native document-first mixed-shard sampler for residual specialists.

Uses the exact existing Model B EOS document sampler and dual-resolution
metadata builders; it does not construct a full ExtendedTrainer (which would
allocate an optimizer for the frozen foundation). Supports independent stream
state and deterministic resume at an optimizer-update boundary.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from mini_llm.data.mixed_shards import (load_weighted_shard_sources,
                                       source_paths_by_split, source_weight_dict)
from mini_llm.data.token_shards import (map_token_shard, load_or_build_packed_document_index,
                                       create_dual_resolution_document_minibatch)
from mini_llm.ops.dual_resolution_attention import (
    build_dual_resolution_metadata, dual_resolution_loss_mask)


class SpecialistDocumentSampler:
    """Document-first mixed-shard stream for any positive microbatch size.

    All rows in a microbatch are sampled independently from one chosen source,
    matching the production foundation sampler. Short rows are EOS-padded and
    excluded from the loss by the per-row target mask; document metadata is
    built for every row, with invalid padded historical blocks suppressed.
    """

    def __init__(self, root, *, context_length, batch_size, eos_token_id,
                 block_size, seed=42, shape_bucket='fine', cache_size=4):
        root = Path(root).expanduser().resolve()
        sources, manifest = load_weighted_shard_sources(root)
        self.root = root
        self.manifest_sha256 = hashlib.sha256(
            (root / 'mixture_manifest.json').read_bytes()).hexdigest()
        self.source_weights = source_weight_dict(sources)
        self.names = sorted(self.source_weights)
        self.probabilities = np.asarray([self.source_weights[n] for n in self.names],
                                        dtype=np.float64)
        self.paths = {split: source_paths_by_split(sources, split)
                      for split in ('train', 'val')}
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)
        self.eos_token_id = int(eos_token_id)
        self.block_size = int(block_size)
        self.shape_bucket = str(shape_bucket)
        if self.context_length <= 0 or self.batch_size <= 0 or self.block_size <= 0:
            raise ValueError('context, batch and block size must be positive')
        if self.shape_bucket not in {'none', 'geometric', 'fine'}:
            raise ValueError('unknown shape bucket')
        if self.eos_token_id < 0 or self.eos_token_id >= int(manifest['vocab_size']):
            raise ValueError('EOS token outside tokenizer vocabulary')
        self.rng = {'train': np.random.default_rng(seed),
                    'val': np.random.default_rng(seed+1)}
        self.cursors = {split: {name: 0 for name in self.names}
                        for split in ('train', 'val')}
        self.counts = {split: {name: 0 for name in self.names}
                       for split in ('train', 'val')}
        self.cache_size = max(int(cache_size), len(self.names))
        self._cache = OrderedDict()
        self._indices = OrderedDict()

    def _get_shard(self, split, name, index, path):
        key = (split, name, index)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._indices.move_to_end(key)
            return self._cache[key], self._indices[key]
        data = map_token_shard(path, seq_length=self.context_length)
        if data.ndim != 1:
            raise ValueError('document-first specialist requires packed 1D shards')
        docidx = load_or_build_packed_document_index(path, data, self.eos_token_id)
        self._cache[key], self._indices[key] = data, docidx
        while len(self._cache) > self.cache_size:
            old, stale = self._cache.popitem(last=False)
            self._indices.pop(old)
            mmap = getattr(stale, '_mmap', None)
            if mmap is not None:
                mmap.close()
        return data, docidx

    def next_batch(self, *, split='train'):
        if split not in self.paths:
            raise ValueError('split must be train/val')
        rng = self.rng[split]
        choice = int(rng.choice(len(self.names), p=self.probabilities))
        name = self.names[choice]
        index = self.cursors[split][name]
        paths = self.paths[split][name]
        path = paths[index]
        tokens, docidx = self._get_shard(split, name, index, path)
        self.cursors[split][name] = (index + 1) % len(paths)
        self.counts[split][name] += 1
        inputs, targets, meta = create_dual_resolution_document_minibatch(
            tokens, self.batch_size, self.context_length, rng=rng,
            eos_token_id=self.eos_token_id, document_index=docidx,
            return_metadata=True, shape_bucket=self.shape_bucket)
        dual = build_dual_resolution_metadata(
            inputs, eos_token_id=self.eos_token_id, block_size=self.block_size)
        # Strictly completed and valid blocks only, including padded tails.
        if dual['block_indices'].shape[1]:
            end_indices = dual['block_indices'][:, :, -1]
            dual['block_valid'] &= end_indices < meta['target_valid_lengths'][:, None]
        # Specialist attention is allowed to ignore the artificial EOS padding
        # after each document; these positions are causally in the future of
        # all supervised tokens and never contribute to the CE.
        dual['valid_lengths'] = np.asarray(meta['target_valid_lengths'], dtype=np.int64)
        meta['dual_resolution_metadata'] = dual
        meta['target_loss_mask'] *= dual_resolution_loss_mask(
            inputs, eos_token_id=self.eos_token_id)
        meta['target_loss_mask'] *= (inputs != self.eos_token_id)
        meta['source_name'] = name
        return inputs, targets, meta

    def checkpoint_state(self):
        return {'schema': 'specialist-document-sampler-v1',
                'manifest_sha256': self.manifest_sha256,
                'context_length': self.context_length,
                'block_size': self.block_size,
                'batch_size': self.batch_size,
                'eos_token_id': self.eos_token_id,
                'shape_bucket': self.shape_bucket,
                'source_weights': dict(self.source_weights),
                'cursors': copy.deepcopy(self.cursors),
                'counts': copy.deepcopy(self.counts),
                'rng': {s: copy.deepcopy(r.bit_generator.state) for s, r in self.rng.items()}}

    def restore(self, state):
        expected = self.checkpoint_state()
        for key in ('schema', 'manifest_sha256', 'context_length', 'block_size',
                    'batch_size', 'eos_token_id', 'shape_bucket', 'source_weights'):
            if state.get(key) != expected[key]:
                raise ValueError(f'specialist sampler resume mismatch: {key}')
        for split in ('train', 'val'):
            if set(state['cursors'][split]) != set(self.cursors[split]):
                raise ValueError('specialist sampler source set changed')
            if any(not 0 <= int(state['cursors'][split][name]) < len(self.paths[split][name])
                   for name in self.names):
                raise ValueError('specialist sampler cursor outside shard list')
        for split in ('train', 'val'):
            self.rng[split].bit_generator.state = copy.deepcopy(state['rng'][split])
            self.cursors[split] = copy.deepcopy(state['cursors'][split])
            self.counts[split] = copy.deepcopy(state['counts'][split])
