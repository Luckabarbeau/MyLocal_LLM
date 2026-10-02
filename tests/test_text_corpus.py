import json
from pathlib import Path

import pytest

from mini_llm.data.text_corpus import (
    CorpusSampleStats,
    CorpusSourceSpec,
    TextCorpusSource,
    normalized_weights,
    weighted_text_sample,
)


def test_text_source_streams_text_and_jsonl(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.txt").write_text("plain text")
    with (root / "b.jsonl").open("w") as handle:
        handle.write(json.dumps({"content": "code-ish content"}) + "\n")
        handle.write(json.dumps({"content": "second record"}) + "\n")

    source = TextCorpusSource("mixed", root)
    texts = list(source.iter_texts(seed=1, shuffle_files=False))
    assert texts == ["plain text", "code-ish content", "second record"]



def test_text_source_streams_json_suffix_jsonlines(tmp_path):
    root = tmp_path / "github"
    root.mkdir()
    path = root / "gharchive-dolma-0004.json"
    with path.open("w") as handle:
        handle.write(json.dumps({"id": "a", "text": "def alpha():\n    return 1"}) + "\n")
        handle.write(json.dumps({"id": "b", "text": "class Beta:\n    pass"}) + "\n")

    source = TextCorpusSource("github", root)
    assert source.discover_files() == [path]
    assert list(source.iter_texts(seed=1, shuffle_files=False)) == [
        "def alpha():\n    return 1",
        "class Beta:\n    pass",
    ]
    inspected = source.inspect()
    assert inspected["suffix_counts"][".json"] == 1
    assert inspected["json_fields"] == ["id", "text"]
    assert inspected["selected_text_column"] == "text"


def test_text_source_streams_top_level_json_array(tmp_path):
    root = tmp_path / "array"
    root.mkdir()
    path = root / "records.json"
    with path.open("w") as handle:
        json.dump(
            [
                {"content": "first document"},
                {"content": "second document"},
            ],
            handle,
            indent=2,
        )

    source = TextCorpusSource("array", root)
    assert list(source.iter_texts(seed=1, shuffle_files=False)) == [
        "first document",
        "second document",
    ]


def test_choose_text_column_prefers_text_then_code_candidates():
    assert TextCorpusSource.choose_text_column(["id", "text", "code"]) == "text"
    assert TextCorpusSource.choose_text_column(["id", "code"]) == "code"
    assert TextCorpusSource.choose_text_column(["id", "payload"], "payload") == "payload"
    with pytest.raises(ValueError):
        TextCorpusSource.choose_text_column(["id", "payload"])


def test_normalized_weights():
    specs = [
        CorpusSourceSpec("a", Path("a"), 2.0),
        CorpusSourceSpec("b", Path("b"), 1.0),
    ]
    weights = normalized_weights(specs)
    assert weights["a"] == pytest.approx(2.0 / 3.0)
    assert weights["b"] == pytest.approx(1.0 / 3.0)


def test_weighted_sample_tracks_requested_byte_mix(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    # Many equal-sized documents make the byte mixture exact enough for a
    # deterministic test while exercising interleaving and quota accounting.
    with (a / "docs.jsonl").open("w") as handle:
        for i in range(100):
            handle.write(json.dumps({"text": "A" * 100}) + "\n")
    with (b / "docs.jsonl").open("w") as handle:
        for i in range(100):
            handle.write(json.dumps({"text": "B" * 100}) + "\n")

    specs = [
        CorpusSourceSpec("a", a, 0.75),
        CorpusSourceSpec("b", b, 0.25),
    ]
    stats = {"a": CorpusSampleStats(), "b": CorpusSampleStats()}
    values = list(weighted_text_sample(specs, total_bytes=4000, seed=3, stats=stats))

    assert values
    assert stats["a"].bytes == 3000
    assert stats["b"].bytes == 1000
    assert stats["a"].documents == 30
    assert stats["b"].documents == 10


def test_coding_mix_profiles_are_normalized():
    config_path = Path(__file__).parents[1] / "configs" / "pretraining_coding_mix.json"
    with config_path.open() as handle:
        config = json.load(handle)
    tokenizer_total = sum(
        source["tokenizer_weight"] for source in config["sources"].values()
    )
    pretraining_total = sum(
        source["pretraining_weight"] for source in config["sources"].values()
    )
    assert tokenizer_total == pytest.approx(1.0)
    assert pretraining_total == pytest.approx(1.0)
    assert config["sources"]["github"]["tokenizer_weight"] > config["sources"]["github"]["pretraining_weight"]
