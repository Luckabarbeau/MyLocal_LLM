import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.data.mixed_shards import (
    load_weighted_shard_sources,
    normalize_source_weights,
    source_paths_by_split,
    source_weight_dict,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_source_weights():
    weights = normalize_source_weights({"code": 2.0, "text": 1.0})
    assert weights["code"] == pytest.approx(2.0 / 3.0)
    assert weights["text"] == pytest.approx(1.0 / 3.0)


def test_load_weighted_shard_sources(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"fake": true}')

    for name, token in (("github", 11), ("fineweb", 22)):
        directory = tmp_path / name
        directory.mkdir()
        np.full(128, token, dtype=np.uint16).tofile(
            directory / "train_shard_00000.bin"
        )
        np.full(64, token, dtype=np.uint16).tofile(
            directory / "val_shard_00000.bin"
        )

    manifest = {
        "format_version": "weighted_packed_mixture_v1",
        "tokenizer_sha256": _hash(tokenizer),
        "vocab_size": 128,
        "sources": {
            "github": {
                "weight": 0.7,
                "relative_shard_dir": "github",
                "train_shards": 1,
                "val_shards": 1,
                "train_tokens": 128,
                "val_tokens": 64,
            },
            "fineweb": {
                "weight": 0.3,
                "relative_shard_dir": "fineweb",
                "train_shards": 1,
                "val_shards": 1,
                "train_tokens": 128,
                "val_tokens": 64,
            },
        },
    }
    (tmp_path / "mixture_manifest.json").write_text(json.dumps(manifest))

    sources, loaded = load_weighted_shard_sources(tmp_path)
    assert loaded["vocab_size"] == 128
    assert [source.name for source in sources] == ["github", "fineweb"]
    assert source_weight_dict(sources) == pytest.approx(
        {"github": 0.7, "fineweb": 0.3}
    )
    train = source_paths_by_split(sources, "train")
    assert train["github"][0].endswith("github/train_shard_00000.bin")


def test_mixture_rejects_tokenizer_mismatch(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("tokenizer")
    (tmp_path / "mixture_manifest.json").write_text(
        json.dumps(
            {
                "format_version": "weighted_packed_mixture_v1",
                "tokenizer_sha256": "0" * 64,
                "vocab_size": 10,
                "sources": {"a": {"weight": 1.0}},
            }
        )
    )
    with pytest.raises(ValueError, match="tokenizer hash"):
        load_weighted_shard_sources(tmp_path)
