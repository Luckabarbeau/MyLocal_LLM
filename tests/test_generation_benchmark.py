"""CPU-only tests for patches 0078A/0078B; no checkpoint/GPU required."""

import json

import pytest

from benchmarks.evaluate_generation import main, parse_seeds, parse_strategies, select_prompts
from benchmarks.generation_metrics import (
    first_repeated_ngram_end, grouped_summary, longest_repeated_span,
    repetition_metrics,
)


def test_repetition_metrics_distinct_no_repeat():
    m = repetition_metrics(list(range(12)))
    assert m["distinct_4"] == 1.0
    assert m["repeated_8gram_rate"] == 0
    assert m["first_repeated_8gram_end"] is None
    assert m["longest_repeated_span_tokens"] == 0


def test_repetition_metrics_exact_loop():
    tokens = list(range(8)) * 3
    m = repetition_metrics(tokens)
    assert m["first_repeated_8gram_end"] == 16
    assert m["repeated_8gram_rate"] > 0
    assert m["distinct_4"] < 1
    assert m["longest_repeated_span_tokens"] >= 16


def test_short_eos_and_overlapping_spans():
    assert repetition_metrics([])["distinct_4"] is None
    assert first_repeated_ngram_end([1, 2, 1, 2, 1], 2) == 4
    assert longest_repeated_span([3, 3, 3, 3]) == 3


def test_seeds_and_prompt_selection():
    assert parse_seeds("42, 123,2026") == [42, 123, 2026]
    assert parse_strategies("top-k,top-p") == ["top-k", "top-p"]
    with pytest.raises(Exception):
        parse_strategies("top-k,top-k")
    with pytest.raises(Exception):
        parse_strategies("unsupported")
    with pytest.raises(Exception):
        parse_seeds("42,42")
    suite = {"prompts": [
        {"id": f"p{i}", "domain": "coding", "topic": "test",
         "length": length, "format": "code", "prompt": "hi"}
        for i, length in enumerate(("short", "medium", "long"))
    ]}
    assert [p["id"] for p in select_prompts(suite, "p0,p2", "coding", "short,long")] == ["p0", "p2"]
    with pytest.raises(ValueError):
        select_prompts(suite, "missing", None, None)


def test_summary_is_per_sample_not_token_micro_average():
    records = []
    for ids, length in ((list(range(12)), "short"), ([1] * 12, "long")):
        records.append({"prompt": {"domain": "story", "length": length},
                        "metrics": repetition_metrics(ids)})
    summary = grouped_summary(records)
    assert summary["overall"]["count"] == 2
    assert summary["overall"]["repeat_8gram_onset_fraction"] == 0.5
    assert summary["by_domain"]["story"]["count"] == 2
    assert summary["by_length"]["long"]["mean_repeated_5gram_rate"] > 0


def test_smoke_and_resume(tmp_path, monkeypatch):
    """Simulate benchmark outputs; verify no rerun and provenance enforcement."""
    suite_path = tmp_path / "prompts.json"
    suite_path.write_text(json.dumps({"recommended_new_tokens": 16, "recommended_seeds": [42, 123],
                                      "recommended_control": {"strategy": "top-k", "top_k": 40, "temperature": 0.8},
                                      "prompts": [{"id": "test", "domain": "story", "length": "short",
                                                   "topic": "sample", "format": "story", "prompt": "Once upon"}]}))
    out = tmp_path / "out"
    from benchmarks import evaluate_generation as runner

    class FakeGenerator:
        calls = []
        def generate(self, **kwargs):
            self.calls.append((kwargs["seed"], kwargs["strategy"], kwargs["top_k"], kwargs["top_p"]))
            return "generated", {"generated_token_ids": [1, 2, 3, 1, 2, 3, 99],
                                 "eos_token_id": 99, "tokens_per_sec": 42,
                                 "total_time_s": 0.1, "prefill_time_s": 0.01}

    fake = FakeGenerator()
    meta = {"step": 45600, "context": 65536, "model_dtype": "bfloat16", "runtime_env": {}}
    monkeypatch.setattr(runner, "load_generator", lambda *a, **k: (fake, meta))
    common = ["--checkpoint", str(tmp_path / "ckpt"), "--output-dir", str(out),
              "--prompt-suite", str(suite_path)]
    assert main(common + ["--dry-run"]) == 0
    assert not out.exists()
    assert main(common) == 0
    rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert [r["decoding"]["strategy"] for r in rows] == ["top-k", "top-p"] * 2
    assert all(r["generated_token_ids"] == [1, 2, 3, 1, 2, 3] for r in rows)
    assert fake.calls == [
        (42, "top-k", 40, None), (42, "top-p", None, 1.0),
        (123, "top-k", 40, None), (123, "top-p", None, 1.0),
    ]
    assert main(common + ["--resume"]) == 0
    assert len(fake.calls) == 4
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"]
    assert summary["expected"] == 4
    assert summary["by_strategy"]["top-k"]["count"] == 2
    assert summary["by_strategy"]["top-p"]["count"] == 2
    assert summary["by_strategy_and_length"]["top-p/short"]["count"] == 2
    assert summary["by_strategy_and_domain"]["top-k/story"]["count"] == 2
    assert "by_strategy" not in str(summary.get("metric_definitions", {}))
    csv_summary = (out / "summary.csv").read_text()
    assert "by_strategy,top-p,2" in csv_summary
    with pytest.raises(ValueError, match="configuration differs"):
        main(common + ["--resume", "--max-new-tokens", "32"])
    with pytest.raises(ValueError, match="configuration differs"):
        main(common + ["--resume", "--strategy", "top-k"])
    with pytest.raises(ValueError, match="configuration differs"):
        main(common + ["--resume", "--top-p", "0.9"])


def test_single_strategy_mode(tmp_path, monkeypatch):
    """Legacy --strategy mode remains available for cheap smoke tests."""
    from benchmarks import evaluate_generation as runner
    class FakeGenerator:
        def generate(self, **kwargs):
            assert kwargs["strategy"] == "top-p"
            assert kwargs["top_p"] == 0.9
            assert kwargs["top_k"] is None
            return "sample", {"generated_token_ids": [3, 4, 5],
                              "eos_token_id": 99}
    monkeypatch.setattr(runner, "load_generator", lambda *a, **k:
                        (FakeGenerator(), {"step": 1, "context": 65536,
                                           "model_dtype": "bfloat16", "runtime_env": {}}))
    suite = tmp_path / "prompts.json"
    suite.write_text(json.dumps({"recommended_new_tokens": 8, "recommended_seeds": [42],
        "prompts": [{"id": "only", "domain": "story", "length": "short",
                     "topic": "test", "format": "story", "prompt": "Once upon"}]}))
    out = tmp_path / "single"
    assert main(["--checkpoint", str(tmp_path / "ckpt"), "--output-dir", str(out),
                 "--prompt-suite", str(suite), "--strategy", "top-p", "--top-p", "0.9"]) == 0
    rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["decoding"] == {"strategy": "top-p", "temperature": 0.8,
                                   "top_k": None, "top_p": 0.9}
