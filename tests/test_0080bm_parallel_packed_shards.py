"""0080BM: parallel tokenization, input-file isolation, and mixture reweighting."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.data.mixed_shards import load_weighted_shard_sources
from mini_llm.data.text_corpus import TextCorpusSource
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
from generate_packed_cosmopedia_shards import pack_text_corpus, is_validation_document
from prepare_ensemble_mixture import pack_shards, reweight_mixture


def _setup(tmp_path):
    datasets = tmp_path / "data"
    corpus = {}
    for name in ("science", "code"):
        folder = datasets / name
        folder.mkdir(parents=True)
        corpus[name] = []
        for index in range(2):
            p = folder / f"source_{index}.jsonl"
            with p.open("w") as handle:
                for j in range(36):
                    text = f"{name} file {index} document {j} unique language sample"
                    corpus[name].append(text)
                    handle.write(json.dumps({"text": text}) + "\n")
    tokenizer = SimpleBPETokenizer(vocab_size=128)
    tokenizer.train(corpus["science"] + corpus["code"])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = tmp_path / "mix.json"
    config.write_text(json.dumps({"sources": {
        name: {"relative_path": name, "tokenizer_weight": .5,
               "pretraining_weight": .5}
        for name in corpus}}))
    return datasets, corpus, tokenizer_path, config, tokenizer


def _source_tokens(source):
    return ([[p.relative_to(source).as_posix(),
              np.fromfile(p, dtype=np.uint16).tolist()]
             for p in sorted(source.rglob("*_shard_*.bin"))])


def test_parallel_is_deterministic_and_keeps_original_file_boundaries(tmp_path):
    root, corpus, tokenizer_path, _, tokenizer = _setup(tmp_path)
    reader = TextCorpusSource("science", root / "science")
    kwargs = dict(tokenizer_path=tokenizer_path, val_ratio=0.20,
                  seed=42, batch_documents=7, shard_size_mb=0.0005)
    one = tmp_path / "single"
    many = tmp_path / "parallel"
    s = pack_text_corpus(reader, output_dir=one, workers=1, **kwargs)
    p = pack_text_corpus(reader, output_dir=many, workers=2,
                         pending_batches=2, **kwargs)
    assert s["train_documents"] == p["train_documents"]
    assert s["val_documents"] == p["val_documents"]
    assert _source_tokens(one) == _source_tokens(many)
    assert len(p["source_files"]) == 2
    assert all(entry["shard_subdir"].startswith("file_") for entry in p["source_files"])
    assert sum(x["train_documents"] + x["val_documents"]
               for x in p["source_files"]) == 72
    # Hash-based validation decisions must be independent of workers.
    assert p["val_documents"] == sum(is_validation_document(x, 0.2, 42)
                                      for x in corpus["science"])
    assert sum(int((np.fromfile(f, dtype=np.uint16) == 1).sum())
               for f in many.rglob("*_shard_*.bin")) == 72
    assert all(p["source_file"] in ("source_0.jsonl", "source_1.jsonl")
               for p in s["source_files"])


def test_independent_source_packing_and_manifest_only_reweight(tmp_path):
    root, _, tokenizer_path, config, tokenizer = _setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    args = SimpleNamespace(data_root=root, output_root=out, mix_config=config,
                           seed=42, val_ratio=0.2, batch_documents=12,
                           shard_size_mb=0.001, max_documents_per_source=None,
                           workers=1, pending_batches=2, source=["science"])
    pack_shards(args)
    manifest = json.loads((out / "mixture_manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["missing_sources"] == ["code"]
    assert (out / "science" / "file_00000").exists()
    args.source = ["code"]
    pack_shards(args)
    before, manifest = load_weighted_shard_sources(out)
    assert len(before) == 2 and manifest["complete"]
    bytes_before = {p: p.read_bytes() for s in before
                    for p in (*s.train_paths, *s.val_paths)}
    mixed = json.loads(config.read_text())
    mixed["sources"]["science"]["pretraining_weight"] = .8
    mixed["sources"]["code"]["pretraining_weight"] = .2
    config.write_text(json.dumps(mixed))
    reweight_mixture(args)
    after, _ = load_weighted_shard_sources(out)
    assert {s.name: s.weight for s in after} == {"science": .8, "code": .2}
    assert {p: p.read_bytes() for s in after
            for p in (*s.train_paths, *s.val_paths)} == bytes_before
    # Setting a source weight to zero should *not* delete its shards.
    mixed["sources"]["science"]["pretraining_weight"] = 1.0
    mixed["sources"]["code"]["pretraining_weight"] = 0.0
    config.write_text(json.dumps(mixed))
    reweight_mixture(args)
    excluded, _ = load_weighted_shard_sources(out)
    assert {s.name: s.weight for s in excluded} == {"science": 1.0, "code": 0.0}
    with pytest.raises(FileExistsError):
        pack_shards(args)


def test_zero_weight_keeps_file_shards_available(tmp_path):
    root, _, tokenizer_path, config, _ = _setup(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    args = SimpleNamespace(data_root=root, output_root=output, mix_config=config,
                           seed=42, val_ratio=0.2, batch_documents=12,
                           shard_size_mb=0.002, max_documents_per_source=None,
                           workers=1, pending_batches=2, source=None)
    pack_shards(args)
    mixed = json.loads(config.read_text())
    mixed["sources"]["science"]["pretraining_weight"] = 1.0
    mixed["sources"]["code"]["pretraining_weight"] = 0.0
    config.write_text(json.dumps(mixed))
    reweight_mixture(args)
    sources, _ = load_weighted_shard_sources(output)
    assert {s.name: s.weight for s in sources} == {"science": 1.0, "code": 0.0}
    assert next(s for s in sources if s.name == "code").train_paths


def test_source_is_not_mixed_and_partial_manifest_is_protected(tmp_path):
    root, _, tokenizer_path, config, _ = _setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    args = SimpleNamespace(data_root=root, output_root=out, mix_config=config,
                           seed=42, val_ratio=0.2, batch_documents=12,
                           shard_size_mb=0.002, max_documents_per_source=None,
                           workers=1, pending_batches=2, source=["science"])
    pack_shards(args)
    with pytest.raises(ValueError, match="incomplete"):
        load_weighted_shard_sources(out)
    args.source = ["science", "science"]
    with pytest.raises(ValueError, match="duplicate"):
        pack_shards(args)
    args.source = ["code"]
    args.seed = 43
    with pytest.raises(ValueError, match="provenance"):
        pack_shards(args)


def test_zero_weight_source_is_still_packed(tmp_path):
    root, _, tokenizer_path, config, _ = _setup(tmp_path)
    settings = json.loads(config.read_text())
    settings["sources"]["code"]["pretraining_weight"] = 0
    config.write_text(json.dumps(settings))
    out = tmp_path / "out"
    out.mkdir()
    (out / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    args = SimpleNamespace(data_root=root, output_root=out, mix_config=config,
                           seed=42, val_ratio=0.2, batch_documents=12,
                           shard_size_mb=0.002, max_documents_per_source=None,
                           workers=1, pending_batches=2, source=None)
    pack_shards(args)
    sources, manifest = load_weighted_shard_sources(out)
    assert manifest["complete"] and len(sources) == 2
    assert {s.name: s.weight for s in sources} == {"science": 1.0, "code": 0.0}
    assert (out / "code" / "file_00000").is_dir()
