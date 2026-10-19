"""0082B: reproducible, EOS-bounded windows from held-out packed-mixture shards.

All source files are read-only uint16/uint32 memmaps.  Only the parent LLM
validation shards are used.  Whole SHARDS are assigned to fusion train/val/test
before sampling.  Samples exclude shard-boundary document fragments and never
cross an EOS boundary (the target token itself MAY be EOS).

This is a conservative initial *window* benchmark, not a document-uniform or
full 64k routed-memory evaluation.  Its random-window sampling favors longer
eligible documents.  Train/val/test shard splitting may also separate spans
of an unusually long document when it crosses a shard boundary; boundary
fragments are excluded specifically to avoid that leakage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from mini_llm.data.mixed_shards import load_weighted_shard_sources


SPLITS = ("train", "val", "test")


def _partition_by_shard(sources, seed):
    """Assign distinct validation-shard files to three fusion splits."""
    parts = {split: {} for split in SPLITS}
    for source in sorted(sources, key=lambda s: s.name):
        paths = list(source.val_paths)
        if len(paths) < 3:
            raise ValueError(f"source {source.name} needs >=3 val shards for disjoint splits")
        # A stable seed per source, invariant to Python's randomized hash seed.
        digest = hashlib.sha256(f"{seed}/{source.name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * 0.15))
        n_test = max(1, round(len(paths) * 0.15))
        if len(paths) - n_val - n_test < 1:
            raise ValueError("source validation set is too small to partition")
        n_train = len(paths) - n_val - n_test
        parts["train"][source.name] = paths[:n_train]
        parts["val"][source.name] = paths[n_train:n_train + n_val]
        parts["test"][source.name] = paths[n_train + n_val:]
    return parts


def _outer_eos_bounds(path, dtype, eos, scan_size=1 << 18):
    """Find first/last EOS cheaply without constructing a whole-shard EOS index."""
    data = np.memmap(path, mode="r", dtype=dtype)
    first, last = None, None
    for start in range(0, len(data), scan_size):
        block = data[start:start + scan_size]
        matches = np.flatnonzero(block == eos)
        if len(matches):
            first = start + int(matches[0])
            break
    for end in range(len(data), 0, -scan_size):
        start = max(0, end - scan_size)
        matches = np.flatnonzero(data[start:end] == eos)
        if len(matches):
            last = start + int(matches[-1])
            break
    return data, first, last


class _ShardSampler:
    def __init__(self, paths, dtype, eos, window, rng):
        self.paths = list(paths)
        self.weights = np.asarray([p.stat().st_size for p in self.paths], dtype=np.float64)
        self.dtype, self.eos, self.window, self.rng = dtype, eos, window, rng
        self.index_cache = {}

    def sample(self):
        for _ in range(300):
            if not self.paths:
                raise RuntimeError("no eligible validation shards for this source")
            pick = int(self.rng.choice(len(self.paths), p=self.weights/self.weights.sum()))
            path = self.paths[pick]
            if path not in self.index_cache:
                self.index_cache[path] = _outer_eos_bounds(path, self.dtype, self.eos)
            mem, first, last = self.index_cache[path]
            # Require entire window AFTER first EOS and through last EOS.
            if first is None or last is None or (last - first) < self.window:
                # Remove files without at least one complete candidate window.
                self.paths.pop(pick)
                self.weights = np.delete(self.weights, pick)
                del self.index_cache[path]
                continue
            lo, hi = first + 1, last - self.window + 1
            start = int(self.rng.integers(lo, hi + 1))
            window = np.asarray(mem[start:start+self.window])
            # The window must not cross documents. EOS at the target is valid.
            if self.eos not in window[:-1]:
                return window.astype(np.int32), path, start
        raise RuntimeError("failed to sample 300 EOS-bounded windows; try a shorter context")


def prepare_windows(root, output, context, counts, seed):
    if context < 2 or any(int(n) < 1 for n in counts):
        raise ValueError("context length >=2 and positive split counts required")
    sources, manifest = load_weighted_shard_sources(root)
    eos = int(manifest["eos_token_id"])
    dtype = np.dtype(manifest.get("token_dtype", "uint16"))
    parts = _partition_by_shard(sources, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = np.asarray([src.weight for src in sources], dtype=np.float64)
    weights /= weights.sum()
    result = {}
    for split, count in zip(SPLITS, counts):
        rng = np.random.default_rng(seed + {"train": 0, "val": 10000, "test": 20000}[split])
        samplers = {s.name: _ShardSampler(parts[split][s.name], dtype, eos, context+1, rng)
                    for s in sources}
        out_path = output / f"fusion_{split}_windows.npy"
        rows = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int32,
                                          shape=(int(count), context+1))
        per_source = Counter()
        locations = []
        for j in range(int(count)):
            source = sources[int(rng.choice(len(sources), p=weights))]
            tokens, path, start = samplers[source.name].sample()
            rows[j] = tokens
            per_source[source.name] += 1
            locations.append([source.name, str(path.relative_to(Path(root).resolve())), start])
        rows.flush()
        del rows
        result[split] = {"windows": str(out_path), "count": int(count),
                         "sample_source_counts": dict(per_source),
                         "validation_shards": {n: [str(p.relative_to(Path(root).resolve()))
                                                  for p in paths] for n, paths in parts[split].items()},
                         "samples": locations}
        print(f"{split}: {count} windows ({context} input tokens + 1 label) -> {out_path}", flush=True)
    metadata = {"format": "fusion_eos_windows_v1", "seed": int(seed),
                "eos_token_id": eos, "context_length": int(context),
                "tokenizer_sha256": manifest.get("tokenizer_sha256"),
                "source_mixture": {s.name: s.weight for s in sources},
                "splits": result,
                "limitations": "plain-context causal windows; exporter selects prediction positions; random-window sampling is length-biased"}
    (output / "windows_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--train-windows", type=int, default=4096)
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--test-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=82)
    args = parser.parse_args(argv)
    prepare_windows(args.mixed_shard_root, args.output_dir, args.context_length,
                    (args.train_windows, args.val_windows, args.test_windows), args.seed)


if __name__ == "__main__":
    main()
