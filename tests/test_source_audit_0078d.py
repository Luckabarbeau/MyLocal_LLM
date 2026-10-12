"""Read-only shard audit regression tests."""
import json
from pathlib import Path

import numpy as np

from benchmarks.audit_source_sampling_0078d import (
    run_audit, eos_from_tokenizer, simulate_sampling, spread,
)
from mini_llm.data.token_shards import build_packed_document_index


def _make_root(tmp_path):
    root = tmp_path / "mixture"
    (root / "github").mkdir(parents=True)
    (root / "cosmopedia_v2").mkdir(parents=True)
    # Small, deterministic EOS-bounded packed documents. Deliberately
    # duplicate one document in train and validation.
    doc_a = [3, 4, 5, 6, 7, 8, 1]
    doc_b = [9, 10, 11, 12, 1]
    doc_c = [3, 4, 5, 6, 7, 8, 1]
    for name in ("github", "cosmopedia_v2"):
        for split in ("train", "val"):
            words = doc_a + doc_b + doc_c
            np.asarray(words, dtype=np.uint16).tofile(
                root/name/f"{split}_shard_00000.bin")
    (root/"tokenizer.json").write_text(json.dumps({"model":{"vocab":{"<|eos|>":1}}}))
    info = {"format_version": "weighted_packed_mixture_v1", "vocab_size": 128,
            "sources": {name: {"relative_shard_dir": name,
             "train_tokens": 19, "val_tokens": 19,
             "weight": .5} for name in ("github", "cosmopedia_v2")}}
    (root/"mixture_manifest.json").write_text(json.dumps(info))
    return root


def test_sampler_detects_fallback_concentration():
    raw = np.array([3, 4, 5, 1, 6, 7, 8, 9, 1], dtype=np.uint16)
    index = build_packed_document_index(raw, 1)
    zero = simulate_sampling(index, np.random.default_rng(1), 60, 3, 0)
    excessive = simulate_sampling(index, np.random.default_rng(1), 60, 3, 100)
    assert zero["fallback_fraction"] == 0
    assert excessive["fallback_fraction"] == 1
    assert excessive["unique_docs"] == 1
    assert excessive["top_document_fraction"] == 1
    corrected = simulate_sampling(index, np.random.default_rng(1), 60, 3, 100,
                                  policy="document_first")
    assert corrected["fallback_fraction"] == 0
    assert corrected["unique_docs"] == 2
    assert corrected["top_document_fraction"] < 1


def test_detect_duplicate_docs_and_train_validation_overlap(tmp_path):
    root = _make_root(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run_audit(root, ["cosmopedia_v2", "github"], max_shards=1,
                       hash_docs=400, draws=25, target_length=4,
                       min_history=30, max_doc_hash_tokens=100,
                       check_sidecar=True)
    for name in ("cosmopedia_v2", "github"):
        s = result["sources"][name]
        assert s["file_inventory"]["train"]["manifest_matches"]
        assert s["train_validation_exact_hash_overlap"] >= 1
        assert s["splits"]["train"]["duplicate_content_hash_groups"] >= 1
        sh = s["splits"]["train"]["shards"][0]
        assert sh["sampling_retrieval_history"]["fallback_fraction"] == 1
        assert sh["sampling_document_first_history"]["fallback_fraction"] == 0
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_eos_parser_and_spread(tmp_path):
    root = _make_root(tmp_path)
    assert eos_from_tokenizer(root/"tokenizer.json") == 1
    assert spread([Path(str(i)) for i in range(10)], 3) == [Path("0"),Path("4"),Path("9")]
