"""0079I: document-bounded packed-shard sampling for refinement experiments.

Read the *existing* 65K BPE token IDs; never retokenize or bridge EOS.
Sampling mirrors 0078E's policy: choose a mixture source, choose a token-
weighted shard, choose a token-weighted document span BEFORE selecting a
window. Short documents are never rejected for failing to fill the window.
The minimum usable document is two tokens (one committed, one supervised).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from mini_llm.data.mixed_shards import load_weighted_shard_sources
from mini_llm.data.token_shards import (
    load_or_build_packed_document_index,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


class PackedRefinementDocuments:
    """Bounded LRU access to a mixture of 1D EOS-delimited token shards."""

    def __init__(self, *, mixed_root=None, shard_dir=None, split='train',
                 eos_token_id, token_dtype='uint16', cache_size=2,
                 write_sidecars=True, index_cache_size=64):
        if (mixed_root is None) == (shard_dir is None):
            raise ValueError('specify exactly one of mixed_root or shard_dir')
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        self.eos = int(eos_token_id)
        self.dtype = np.dtype(token_dtype)
        if self.dtype not in (np.dtype('uint16'), np.dtype('uint32')):
            raise ValueError('packed token dtype must be uint16 or uint32')
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError('cache_size must be positive')
        self.write_sidecars = bool(write_sidecars)
        self.cache = {}
        self.order = []
        self.index_cache = {}
        self.index_order = []
        self.index_cache_size = max(1, int(index_cache_size))
        if mixed_root is not None:
            sources, manifest = load_weighted_shard_sources(mixed_root)
            self.manifest_path = Path(mixed_root).expanduser().resolve() / 'mixture_manifest.json'
            self.manifest_hash = sha256(self.manifest_path)
            self.sources = [(s.name, float(s.weight), list(s.train_paths if split == 'train' else s.val_paths))
                            for s in sources]
            if manifest.get('token_dtype') in ('uint16', 'uint32') and manifest['token_dtype'] != self.dtype.name:
                raise ValueError('requested dtype differs from mixture manifest')
        else:
            root = Path(shard_dir).expanduser().resolve()
            self.manifest_path = None
            self.manifest_hash = None
            self.sources = [('single', 1.0, sorted(root.glob(f'{split}_shard_*.bin')))]
        if not self.sources or any(not src[2] for src in self.sources):
            raise ValueError(f'no {split} shards found')
        self.source_names = [name for name, _, _ in self.sources]
        self.source_probs = np.array([w for _, w, _ in self.sources], dtype=np.float64)
        self.source_probs /= self.source_probs.sum()
        self.shard_weights = []
        for _, _, paths in self.sources:
            sizes = np.asarray([p.stat().st_size for p in paths], dtype=np.float64)
            if np.any(sizes < 2 * self.dtype.itemsize) or np.any(sizes % self.dtype.itemsize):
                raise ValueError('packed shards must contain >=2 full token IDs and have no header')
            self.shard_weights.append(sizes / sizes.sum())

    def provenance(self):
        return {
            'sampling': '0078E_length_weighted_document_first_no_length_rejection',
            'format': 'packed_eos_1d', 'eos': self.eos, 'dtype': self.dtype.name,
            'manifest_sha256': self.manifest_hash,
            'sources': [
                {'name': name, 'weight': float(weight),
                 'shards': [[str(p.resolve()), p.stat().st_size] for p in paths]}
                for name, weight, paths in self.sources
            ],
        }

    def _get_shard(self, path):
        path = Path(path)
        if path not in self.cache:
            tokens = np.memmap(path, dtype=self.dtype, mode='r')
            if path in self.index_cache:
                index = self.index_cache[path]
                self.index_order.remove(path)
                self.index_order.append(path)
            else:
                index = load_or_build_packed_document_index(
                    path, tokens, self.eos, write_sidecar=self.write_sidecars
                )
                self.index_cache[path] = index
                self.index_order.append(path)
                if len(self.index_order) > self.index_cache_size:
                    del self.index_cache[self.index_order.pop(0)]
            self.cache[path] = (tokens, index)
            self.order.append(path)
            if len(self.order) > self.cache_size:
                oldest = self.order.pop(0)
                del self.cache[oldest]
        else:
            self.order.remove(path)
            self.order.append(path)
        return self.cache[path]

    def sample(self, rng, *, batch_size, prefix_length, block_size, window_blocks,
               return_observations=False):
        if min(batch_size, prefix_length, block_size, window_blocks) <= 0:
            raise ValueError('batch size, prefix and block dimensions must be positive')
        max_length = int(prefix_length + block_size * window_blocks)
        rows, source_counts = [], {}
        observations = []
        for _ in range(batch_size):
            source_i = int(rng.choice(len(self.sources), p=self.source_probs))
            name, _, paths = self.sources[source_i]
            shard_i = int(rng.choice(len(paths), p=self.shard_weights[source_i]))
            tokens, index = self._get_shard(paths[shard_i])
            # Choose a token-weighted document, without rejecting short
            # documents in favor of longer ones (except one-token fragments).
            # A two-token fragment always has one legitimate prediction pair.
            for _attempt in range(32):
                anchor = int(rng.integers(0, index.token_count))
                start, end = index.span(index.document_for_position(anchor))
                if end - start >= 2:
                    break
            else:
                spans = (index.span(j) for j in range(index.document_count))
                candidate = next(((a, b) for a, b in spans if b - a >= 2), None)
                if candidate is None:
                    raise ValueError('shard contains no document with a prediction pair')
                start, end = candidate
            source_counts[name] = source_counts.get(name, 0) + 1
            # Select a position only *after* selecting the document. For a
            # long document, any window with a genuine prefix may be drawn.
            allowed_prefix = min(int(prefix_length), end - start - 1)
            offset = int(rng.integers(start, end - allowed_prefix))
            row = np.asarray(tokens[offset:min(end, offset + max_length)], dtype=np.int64)
            rows.append(row)
            if return_observations:
                observations.append({
                    'source': name,
                    'sampled_document_span_tokens': int(end - start),
                    'sampled_fragment_tokens': int(len(row)),
                })
        # The current 0079H trainer uses one prefix length per batch; choose
        # the minimum actual prefix, but never pad/contextualize fake tokens.
        common_prefix = min(int(prefix_length), *(len(row) - 1 for row in rows))
        longest = min(max_length, max(len(row) for row in rows))
        padded = np.zeros((batch_size, longest), dtype=np.int64)
        mask = np.zeros((batch_size, longest), dtype=np.bool_)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = row
            mask[i, :len(row)] = True
        base = (padded, mask, common_prefix, source_counts)
        return base + (observations,) if return_observations else base
