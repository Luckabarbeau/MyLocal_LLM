import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.data.mixed_shards import load_weighted_shard_sources
from recover_mixture_manifest import recover_mixture_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.uint16).tofile(path)


def test_recover_root_manifest_with_partial_source(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text('{"dummy": true}')
    tokenizer_hash = _sha(tokenizer)

    mix = {
        "name": "test_mix",
        "sources": {
            "complete": {"pretraining_weight": 0.75},
            "partial": {"pretraining_weight": 0.25},
        },
    }
    mix_path = root / "mix_config.json"
    mix_path.write_text(json.dumps(mix))

    complete_dir = root / "complete"
    _write_shard(complete_dir / "train_shard_00000.bin", range(10))
    _write_shard(complete_dir / "val_shard_00000.bin", range(4))
    (complete_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_sha256": tokenizer_hash,
                "vocab_size": 65280,
                "token_dtype": "uint16",
                "shards": {"train": 1, "val": 1},
                "tokens": {"train": 10, "val": 4},
            }
        )
    )

    partial_dir = root / "partial"
    _write_shard(partial_dir / "train_shard_00000.bin", range(7))
    _write_shard(partial_dir / "train_shard_00001.bin", range(3))
    _write_shard(partial_dir / "val_shard_00000.bin", range(2))

    manifest = recover_mixture_manifest(
        shard_root=root,
        mix_config_path=mix_path,
        partial_sources=["partial"],
    )

    assert manifest["complete"] is True
    assert manifest["recovered"] is True
    assert manifest["partial_sources"] == ["partial"]
    assert manifest["sources"]["partial"]["train_tokens"] == 10
    assert manifest["sources"]["partial"]["val_tokens"] == 2
    assert manifest["sources"]["partial"]["train_shards"] == 2
    assert manifest["sources"]["partial"]["weight"] == pytest.approx(0.25)

    sources, loaded = load_weighted_shard_sources(root)
    assert loaded["partial_sources"] == ["partial"]
    assert {source.name for source in sources} == {"complete", "partial"}


def test_recovery_rejects_odd_byte_tail_shard(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("tokenizer")
    tokenizer_hash = _sha(tokenizer)

    mix = {
        "sources": {
            "complete": {"pretraining_weight": 0.5},
            "partial": {"pretraining_weight": 0.5},
        }
    }
    mix_path = root / "mix_config.json"
    mix_path.write_text(json.dumps(mix))

    complete_dir = root / "complete"
    _write_shard(complete_dir / "train_shard_00000.bin", [1, 2])
    _write_shard(complete_dir / "val_shard_00000.bin", [3, 4])
    (complete_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_sha256": tokenizer_hash,
                "vocab_size": 65280,
                "token_dtype": "uint16",
                "shards": {"train": 1, "val": 1},
                "tokens": {"train": 2, "val": 2},
            }
        )
    )

    partial_dir = root / "partial"
    partial_dir.mkdir()
    (partial_dir / "train_shard_00000.bin").write_bytes(b"\x01\x00\x02")
    _write_shard(partial_dir / "val_shard_00000.bin", [3])

    with pytest.raises(ValueError, match="multiple of uint16"):
        recover_mixture_manifest(
            shard_root=root,
            mix_config_path=mix_path,
            partial_sources=["partial"],
        )
