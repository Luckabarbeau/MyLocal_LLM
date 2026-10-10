"""Weighted collections of context-independent packed token shards.

The mixed pretraining pipeline stores each corpus in its own shard directory and
chooses the corpus at minibatch time.  This keeps dataset weights independent of
physical dataset size and avoids duplicating large corpora when experimenting
with different mixtures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class WeightedShardSource:
    """One logical pretraining source and its packed train/validation shards."""

    name: str
    weight: float
    train_paths: Tuple[Path, ...]
    val_paths: Tuple[Path, ...]
    train_tokens: int = 0
    val_tokens: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return non-negative source weights normalized to one."""

    if not weights:
        raise ValueError("at least one source weight is required")
    converted = {str(name): float(weight) for name, weight in weights.items()}
    if any(weight < 0.0 for weight in converted.values()):
        raise ValueError("source weights must be non-negative")
    total = sum(converted.values())
    if total <= 0.0:
        raise ValueError("sum of source weights must be positive")
    return {name: weight / total for name, weight in converted.items()}


def load_weighted_shard_sources(root: Path | str) -> Tuple[List[WeightedShardSource], dict]:
    """Load and validate a ``weighted_packed_mixture_v1`` dataset.

    The root manifest is intentionally the training contract.  The trainer does
    not infer weights from shard counts or byte sizes: a 15 GB code corpus can
    therefore receive a larger probability than a 30 GB web corpus without
    physically duplicating shards.
    """

    root = Path(root).expanduser().resolve()
    manifest_path = root / "mixture_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"mixed-shard manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != "weighted_packed_mixture_v1":
        raise ValueError(
            "unsupported mixed-shard format: "
            f"{manifest.get('format_version')!r}"
        )

    tokenizer_path = root / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"mixed tokenizer not found: {tokenizer_path}")
    expected_hash = manifest.get("tokenizer_sha256")
    if expected_hash and _sha256(tokenizer_path) != expected_hash:
        raise ValueError("mixed-shard tokenizer hash does not match manifest")

    if manifest.get("complete") is False:
        missing = manifest.get("missing_sources", [])
        raise ValueError(
            "mixed-shard preprocessing is incomplete; missing sources: "
            + ", ".join(str(name) for name in missing)
        )

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("mixture manifest contains no generated sources")

    normalized = normalize_source_weights(
        {name: entry.get("weight", 0.0) for name, entry in raw_sources.items()}
    )
    sources: List[WeightedShardSource] = []
    for name, entry in raw_sources.items():
        relative_dir = entry.get("relative_shard_dir", name)
        source_dir = root / relative_dir
        train_paths = tuple(sorted(source_dir.glob("train_shard_*.bin")))
        val_paths = tuple(sorted(source_dir.glob("val_shard_*.bin")))
        if not train_paths:
            raise ValueError(f"source {name!r} has no train_shard_*.bin files")
        if not val_paths:
            raise ValueError(f"source {name!r} has no val_shard_*.bin files")

        expected_train = int(entry.get("train_shards", len(train_paths)))
        expected_val = int(entry.get("val_shards", len(val_paths)))
        if len(train_paths) != expected_train or len(val_paths) != expected_val:
            raise ValueError(
                f"source {name!r} shard count does not match mixture manifest: "
                f"train {len(train_paths)}/{expected_train}, "
                f"val {len(val_paths)}/{expected_val}"
            )
        sources.append(
            WeightedShardSource(
                name=name,
                weight=normalized[name],
                train_paths=train_paths,
                val_paths=val_paths,
                train_tokens=int(entry.get("train_tokens", 0)),
                val_tokens=int(entry.get("val_tokens", 0)),
            )
        )
    return sources, manifest


def source_paths_by_split(
    sources: Sequence[WeightedShardSource], split: str
) -> Dict[str, List[str]]:
    """Convert source objects to the mapping consumed by ``ExtendedTrainer``."""

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    result: Dict[str, List[str]] = {}
    for source in sources:
        paths = source.train_paths if split == "train" else source.val_paths
        result[source.name] = [str(path) for path in paths]
    return result


def source_weight_dict(sources: Sequence[WeightedShardSource]) -> Dict[str, float]:
    return {source.name: float(source.weight) for source in sources}
