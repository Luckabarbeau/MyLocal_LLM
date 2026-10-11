"""0077A: bounded, opt-in remote document streams for the existing shard sampler.

Each HF IterableDataset is pinned to an immutable revision; one CPU thread per
source prepares at most ONE look-ahead segment. Token files are temporary packed
uint16 streams using the existing EOS-bounded training sampler. The checkpoint
cursor always describes the beginning of the active segment, so prefetched data
cannot advance the committed training cursor. No remote data enters validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class StreamSpec:
    name: str
    dataset: str
    revision: str
    weight: float
    config: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    license: str = "unspecified"

    @classmethod
    def parse(cls, row):
        if not isinstance(row, dict):
            raise ValueError("stream source must be an object")
        spec = cls(**row)
        if not spec.name or not spec.dataset or not spec.revision:
            raise ValueError("source requires name, dataset and pinned revision")
        if re.fullmatch(r"[A-Za-z0-9_-]+", spec.name) is None:
            raise ValueError("source name must contain only letters, digits, underscores, hyphens")
        if spec.revision in {"main", "master", "latest"}:
            raise ValueError("pin a commit hash or immutable dataset revision, not a branch")
        if not np.isfinite(spec.weight) or spec.weight <= 0 or spec.weight >= 1:
            raise ValueError("stream weight must be finite and between zero and one")
        if spec.split != "train":
            raise ValueError("stream sources must use the training split")
        if not spec.text_field or not spec.license:
            raise ValueError("text field and license declaration are required")
        return spec


def load_registry(path):
    document = json.loads(Path(path).read_text())
    if document.get("version") != 1:
        raise ValueError("stream registry version must be 1")
    specs = [StreamSpec.parse(row) for row in document["sources"]]
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("stream source names must be unique and nonempty")
    if sum(s.weight for s in specs) >= 1.0:
        raise ValueError("local shard mixture must retain positive sampling weight")
    return specs


def _default_factory(spec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Streaming sources need: pip install 'datasets>=3' ") from exc
    return load_dataset(
        spec.dataset, name=spec.config, split=spec.split,
        revision=spec.revision, streaming=True,
    )


class _Stream:
    def __init__(self, spec, tokenizer, cache_root, segment_tokens, reuse_batches,
                 executor, factory):
        self.spec = spec
        self.tokenizer = tokenizer
        self.cache_root = cache_root
        self.segment_tokens = int(segment_tokens)
        self.reuse_batches = int(reuse_batches)
        self.executor = executor
        self.factory = factory
        self.active = None
        self.active_uses = 0
        self.next_cursor = None
        self.pending = None
        self.generation = 0
        self.history = []
        self._resume_snapshot = None
        self._snapshot_checkpoint_dir = None
        self.eos_id = self._eos_id(tokenizer)

    @staticmethod
    def _eos_id(tokenizer):
        eos = getattr(tokenizer, "eos_token", None)
        table = getattr(tokenizer, "token_to_id", None)
        if callable(table):
            value = table(eos)
        elif isinstance(table, dict):
            value = table.get(eos)
        elif hasattr(tokenizer, "_tokenizer"):
            value = tokenizer._tokenizer.token_to_id(eos)
        else:
            value = None
        if value is None or not 0 <= int(value) < 65536:
            raise ValueError("stream tokenizer must expose uint16 EOS token")
        return int(value)

    def _produce(self, cursor, generation):
        dataset = self.factory(self.spec)
        if not hasattr(dataset, "state_dict") or not hasattr(dataset, "load_state_dict"):
            raise ValueError("stream dataset must support resumable state_dict()")
        if cursor is not None:
            dataset.load_state_dict(copy.deepcopy(cursor))
        start_cursor = copy.deepcopy(dataset.state_dict())
        documents = []
        token_count = 0
        for row in dataset:
            text = row.get(self.spec.text_field) if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            if len(text) > self.segment_tokens * 16:
                raise ValueError("document text exceeds bounded streaming text budget")
            ids = list(self.tokenizer.encode(text))
            if not ids:
                continue
            if len(ids) > self.segment_tokens:
                raise ValueError(
                    f"document exceeds stream segment budget ({len(ids)} > "
                    f"{self.segment_tokens} tokens); adjust --stream-segment-tokens"
                )
            if any(int(i) < 0 or int(i) >= 65536 for i in ids):
                raise ValueError("tokenizer emitted a token outside uint16 vocabulary")
            if int(ids[-1]) != self.eos_id:
                ids.append(self.eos_id)
            # No document truncation: a single long document may exceed budget.
            documents.append(np.asarray(ids, dtype=np.uint16))
            token_count += len(ids)
            if token_count >= self.segment_tokens:
                break
        if not documents:
            raise RuntimeError(f"stream {self.spec.name} exhausted (no more documents)")
        end_cursor = copy.deepcopy(dataset.state_dict())
        if _canonical(start_cursor) == _canonical(end_cursor):
            raise RuntimeError(
                f"stream {self.spec.name} cursor did not advance after reading documents; "
                "refusing to risk replaying training samples"
            )
        arr = np.concatenate(documents)
        file_path = self.cache_root / f"{self.spec.name}_{generation:09d}.bin"
        tmp_path = file_path.with_suffix(".tmp")
        try:
            arr.tofile(tmp_path)
            os.replace(tmp_path, file_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return {
            "path": file_path,
            "generation": generation,
            "start_cursor": start_cursor,
            "end_cursor": end_cursor,
            "documents": len(documents),
            "tokens": len(arr),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }

    def start(self):
        if self._resume_snapshot is not None:
            return  # The saved active segment is verified locally first.
        if self.pending is None:
            self.pending = self.executor.submit(
                self._produce, copy.deepcopy(self.next_cursor), self.generation,
            )

    def get(self):
        retired = []
        if self.active is not None and self.active_uses >= self.reuse_batches:
            previous = self.active
            self.history.append(self._entry(previous, self.active_uses, include_cursor=False))
            retired.append(previous["path"])
            self.next_cursor = previous["end_cursor"]
            self.generation += 1
            self.active = None
            self.active_uses = 0
        if self.active is None:
            # Until a segment is consumed, producer advances are speculative only.
            if self._resume_snapshot is not None:
                segment = self._read_snapshot()
                self._resume_snapshot = None
            elif self.pending is not None:
                segment = self.pending.result()
                self.pending = None
                # An in-flight prefetch is only valid for its original cursor.
                if segment["generation"] != self.generation:
                    segment["path"].unlink(missing_ok=True)
                    segment = self._produce(self.next_cursor, self.generation)
            else:
                segment = self._produce(self.next_cursor, self.generation)
            self.active = segment
        self.active_uses += 1
        return self.active["path"], self.active["generation"], retired

    def _read_snapshot(self):
        """Restore exact saved bytes before touching a remote data source."""
        entry = self._resume_snapshot
        relative_path = Path(entry["snapshot"])
        if (relative_path.is_absolute() or ".." in relative_path.parts or
                relative_path.parts[0] != "stream_active_segments"):
            raise ValueError("unsafe active stream snapshot path in checkpoint")
        src = self._snapshot_checkpoint_dir / relative_path
        if not src.is_file():
            raise FileNotFoundError(f"active stream snapshot missing: {src}")
        data = src.read_bytes()
        expected = entry["sha256"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"active stream snapshot corrupted: {src}")
        if len(data) != int(entry["tokens"]) * np.dtype(np.uint16).itemsize:
            raise ValueError(f"active stream snapshot size mismatch: {src}")
        dest = self.cache_root / f"{self.spec.name}_{self.generation:09d}.bin"
        tmp = dest.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return {
            "path": dest, "generation": self.generation,
            "start_cursor": copy.deepcopy(entry["start_cursor"]),
            "end_cursor": copy.deepcopy(entry["end_cursor"]),
            "documents": entry["documents"], "tokens": entry["tokens"],
            "sha256": expected,
        }

    def prefetch_next(self):
        if self.active is not None and self.pending is None:
            self.pending = self.executor.submit(
                self._produce, copy.deepcopy(self.active["end_cursor"]),
                self.generation + 1,
            )

    def _entry(self, segment, uses, *, include_cursor=True):
        entry = {
            "generation": segment["generation"],
            "sha256": segment["sha256"],
            "documents": segment["documents"], "tokens": segment["tokens"],
            "microbatches": uses,
        }
        if include_cursor:
            entry["start_cursor"] = segment["start_cursor"]
            entry["end_cursor"] = segment["end_cursor"]
        return entry

    def state_dict(self):
        return {
            "generation": self.generation,
            "next_cursor": self.next_cursor,
            "active": self._entry(self.active, self.active_uses) if self.active else None,
            "history": copy.deepcopy(self.history),
        }

    def load_state_dict(self, state, *, checkpoint_dir=None):
        self._snapshot_checkpoint_dir = (Path(checkpoint_dir) if checkpoint_dir else None)
        self._resume_snapshot = None
        self.generation = int(state["generation"])
        self.next_cursor = copy.deepcopy(state.get("next_cursor"))
        self.history = copy.deepcopy(state.get("history", []))
        active = state.get("active")
        if active:
            if int(active["microbatches"]) >= self.reuse_batches:
                # A complete segment was committed by the last optimizer step.
                self.history.append({key: active[key] for key in (
                    "generation", "sha256", "documents", "tokens", "microbatches"
                )})
                self.generation += 1
                self.next_cursor = copy.deepcopy(active["end_cursor"])
                self.active_uses = 0
                self._expected_sha256 = None
            else:
                # Replay the ACTIVE segment from its beginning: tokens and
                # document offsets must match the checkpointed fingerprint.
                self.next_cursor = copy.deepcopy(active["start_cursor"])
                self.active_uses = int(active["microbatches"])
                self._expected_sha256 = active["sha256"]
                if "snapshot" in active:
                    if self._snapshot_checkpoint_dir is None:
                        raise ValueError("checkpoint_dir required for persisted stream snapshots")
                    self._resume_snapshot = copy.deepcopy(active)
        else:
            self.active_uses = 0
            self._expected_sha256 = None


class StreamingTokenCache:
    """Bounded disk cache with per-source asynchronous preparation.

    `next_shard` is called only for selected streamed microbatches. The source
    remains on the same segment for reuse_batches calls; on rotation the trainer
    must close any mmap before deleting retired paths.
    """
    def __init__(self, specs, tokenizer, *, segment_tokens=524288,
                 reuse_batches=64, cache_dir=None, factory=None,
                 checkpoint_dir=None):
        if segment_tokens < 1024 or reuse_batches < 1:
            raise ValueError("invalid segment_tokens or reuse_batches")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.specs = {s.name: s for s in specs}
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="myllm_stream_", dir=cache_dir)
        root = Path(self._temp.name)
        self.executor = ThreadPoolExecutor(max_workers=min(len(specs), 2))
        self.sources = {
            s.name: _Stream(s, tokenizer, root, segment_tokens, reuse_batches,
                            self.executor, factory or _default_factory)
            for s in specs
        }
        self.registry_fingerprint = hashlib.sha256(
            _canonical([s.__dict__ for s in specs]).encode()
        ).hexdigest()
        self.specs_record = [s.__dict__.copy() for s in specs]
        self.segment_tokens = int(segment_tokens)
        self.reuse_batches = int(reuse_batches)

    def prefetch(self):
        for source in self.sources.values():
            source.start()

    def next_shard(self, name):
        source = self.sources[name]
        path, generation, retired = source.get()
        expected = getattr(source, "_expected_sha256", None)
        if expected is not None:
            if source.active["sha256"] != expected:
                raise ValueError(
                    f"remote training data changed: active segment sha256 mismatch "
                    f"for {name!r} generation {generation}; the checkpoint predates "
                    "persistent active segments. Do NOT skip verification; use "
                    "RESET_REMOTE_CURSORS=1 for a recorded data-phase reset."
                )
            source._expected_sha256 = None
        # Only prefetch another remote segment after the active segment has
        # passed checkpoint hash verification. This avoids runaway workers on errors.
        source.prefetch_next()
        return path, generation, retired

    def checkpoint_state(self, checkpoint_dir):
        """Persist only live segment bytes next to the checkpoint.

        These ~1 MiB files survive process restarts, unlike the ephemeral stream
        cache. Completed segments are never retained. Snapshot names are
        content-addressed and not overwritten by newer checkpoints.
        """
        state = self.state_dict()
        folder = Path(checkpoint_dir) / "stream_active_segments"
        for name, source in self.sources.items():
            segment = source.active
            if segment is None or source.active_uses >= source.reuse_batches:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            key = f"{name}_{segment['generation']:09d}_{segment['sha256'][:16]}.bin"
            dest = folder / key
            if not dest.is_file():
                tmp = folder / (key + ".tmp")
                try:
                    shutil.copyfile(segment["path"], tmp)
                    if hashlib.sha256(tmp.read_bytes()).hexdigest() != segment["sha256"]:
                        raise ValueError("active streamed segment changed during checkpoint")
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
            if hashlib.sha256(dest.read_bytes()).hexdigest() != segment["sha256"]:
                raise ValueError("checkpoint active stream snapshot hash mismatch")
            state["sources"][name]["active"]["snapshot"] = (
                "stream_active_segments/" + key
            )
        return state

    def prune_checkpoint_snapshots(self, checkpoint_dir, stream_state):
        folder = Path(checkpoint_dir) / "stream_active_segments"
        if not folder.exists():
            return
        keep = {
            Path(active["snapshot"]).name
            for src in stream_state["sources"].values()
            if (active := src.get("active")) is not None and active.get("snapshot")
        }
        for file in folder.glob("*.bin"):
            if file.name not in keep:
                file.unlink()

    def state_dict(self):
        return {
            "version": 1,
            "registry_fingerprint": self.registry_fingerprint,
            "segment_tokens": self.segment_tokens,
            "reuse_batches": self.reuse_batches,
            "sources": {name: source.state_dict() for name, source in self.sources.items()},
            "registry": self.specs_record,
        }

    def load_state_dict(self, state, *, checkpoint_dir=None):
        checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        if state["registry_fingerprint"] != self.registry_fingerprint:
            raise ValueError("stream dataset registry differs from saved checkpoint")
        if (state["segment_tokens"] != self.segment_tokens or
                state["reuse_batches"] != self.reuse_batches):
            raise ValueError("stream segment configuration differs from saved checkpoint")
        for name, source in self.sources.items():
            if checkpoint_dir is None:
                source.load_state_dict(state["sources"][name])
            else:
                source.load_state_dict(state["sources"][name], checkpoint_dir=checkpoint_dir)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._temp.cleanup()
