"""0080BL: mixed-corpus token packing preserves the trainer contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
from mini_llm.data.mixed_shards import load_weighted_shard_sources
from prepare_ensemble_mixture import sources_from_config, pack_shards, training_texts


def test_mixed_corpus_shards_roundtrip(tmp_path):
    source_root = tmp_path / "corpora"
    (source_root / "science").mkdir(parents=True)
    (source_root / "code").mkdir(parents=True)
    source_texts = {}
    for name in ("science", "code"):
        source_texts[name] = [f"document {j} from {name} with information" for j in range(70)]
        with (source_root / name / "records.jsonl").open("w") as fp:
            for text in source_texts[name]:
                fp.write(json.dumps({"text": text}) + "\n")
    mix = {"sources": {name: {"relative_path": name,
                                 "tokenizer_weight": 0.5,
                                 "pretraining_weight": 0.5}
                       for name in source_texts}}
    mix_config = tmp_path / "mix.json"
    mix_config.write_text(json.dumps(mix))
    output = tmp_path / "out"
    output.mkdir()
    tok = SimpleBPETokenizer(vocab_size=128)
    tok.train([text for docs in source_texts.values() for text in docs[:10]])
    tok.save(str(output / "tokenizer.json"))
    args = SimpleNamespace(data_root=source_root, output_root=output,
                           mix_config=mix_config, seed=42, val_ratio=0.2,
                           batch_documents=16, shard_size_mb=0.001,
                           max_documents_per_source=None)
    pack_shards(args)
    sources, manifest = load_weighted_shard_sources(output)
    assert manifest["complete"] and len(sources) == 2
    assert sum(s.train_tokens for s in sources) > 0
    assert sum(s.val_tokens for s in sources) > 0
    assert [s.weight for s in sources] == [0.5, 0.5]
    for source in sources:
        streams = [np.fromfile(p, dtype=np.uint16) for p in source.train_paths]
        streams += [np.fromfile(p, dtype=np.uint16) for p in source.val_paths]
        assert sum(int(np.sum(v == 1)) for v in streams) == 70  # one EOS/document
        assert all(int(v.max()) < len(tok) for v in streams)
    with pytest.raises(FileExistsError):
        pack_shards(args)


def test_tokenizer_samples_exclude_validation_and_respect_source_weights(tmp_path):
    root = tmp_path / 'in'
    root.mkdir()
    cfg = {"sources": {}}
    for name, weight in (("first", 0.75), ("second", 0.25)):
        path = root / name
        path.mkdir()
        (path / 'text.txt').write_text('')
        with (path / 'rows.jsonl').open('w') as fp:
            for j in range(60):
                fp.write(json.dumps({'text': f'{name}-doc-{j}'}) + '\n')
        cfg['sources'][name] = {'relative_path': name,
                                'tokenizer_weight': weight,
                                'pretraining_weight': weight}
    config = tmp_path / 'mix.json'
    config.write_text(json.dumps(cfg))
    sources = sources_from_config(config, root, 'tokenizer_weight')
    sample = list(training_texts(sources, 20, 0.2, 42))
    assert len(sample) == 20
    assert sum(s.startswith('first') for s in sample) == 15
    assert sum(s.startswith('second') for s in sample) == 5
    from generate_packed_cosmopedia_shards import is_validation_document
    assert all(not is_validation_document(s, 0.2, 42) for s in sample)


def test_compressed_json_corpus(tmp_path):
    import gzip
    from mini_llm.data.text_corpus import TextCorpusSource
    root = tmp_path / "github"
    root.mkdir()
    with gzip.open(root / "gharchive.json.gz", "wt") as handle:
        handle.write(json.dumps({"text": "def f(): return 2"}) + "\n")
        handle.write(json.dumps({"text": "print(f())"}) + "\n")
    with gzip.open(root / "rows.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"text": "import math"}) + "\n")
    assert list(TextCorpusSource("github", root).iter_texts(shuffle_files=False)) == [
        "def f(): return 2", "print(f())", "import math",
    ]
