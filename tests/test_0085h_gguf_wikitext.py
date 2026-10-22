"""CPU-only, real native-window scoring tests for the GGUF WikiText adapter."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.evaluate_external_wikitext import (
    _ExternalWindowModel,
    _GGUFTokenIDs,
    calculate_summary,
)
from benchmarks.evaluate_wikitext import (
    run_windowed_evaluation,
    tokenize_eval_streams,
)


class FakeTokenizer:
    spec = SimpleNamespace(add_bos=False, bos_id=None)

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[ord(ch) % 8 for ch in text])


def test_embedding_tokenizer_adapter_is_no_specials():
    adapter = _GGUFTokenIDs(FakeTokenizer())
    assert adapter.encode("abc") == [1, 2, 3]


def test_explicit_bos_is_preserved_when_requested_by_gguf():
    tokenizer = FakeTokenizer()
    tokenizer.spec = SimpleNamespace(add_bos=True, bos_id=7)
    assert _GGUFTokenIDs(tokenizer).encode("a") == [7, 1]
    tokenizer.spec = SimpleNamespace(add_bos=True, bos_id=None)
    with pytest.raises(ValueError, match="no BOS token ID"):
        _GGUFTokenIDs(tokenizer).encode("a")


class ConstantLogitsModel:
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        # Scores are uniformly logit-zero regardless of history, so all
        # windows and document boundaries must give CE = log(vocab_size).
        return np.zeros((x.shape[0], x.shape[1], self.vocab_size), dtype=np.float32)


def test_real_native_sliding_window_scoring_reused_without_overlap_double_count():
    raw = "abcdefghijklmno"
    streams, tokens, n_docs = tokenize_eval_streams(
        raw, _GGUFTokenIDs(FakeTokenizer()), document_mode="continuous"
    )
    assert tokens == len(raw) and n_docs == 1
    for context, stride in ((4, 4), (4, 2), (5, 1)):
        model = ConstantLogitsModel()
        result = run_windowed_evaluation(
            _ExternalWindowModel(model), streams,
            context=context, stride=stride,
            routed_eval=False, memory_cfg=None, progress_every=0,
        )
        assert result["total_scored"] == len(raw) - 1
        assert result["total_nll"] == pytest.approx((len(raw) - 1) * math.log(8), rel=1e-6)
        assert model.calls == result["steps"]
        metrics = calculate_summary(result, raw_bytes=len(raw),
                                    tokenizer_tokens=tokens, truncated=False)
        assert metrics["perplexity"] == pytest.approx(8.0, rel=1e-6)
        assert metrics["bits_per_byte"] == pytest.approx((len(raw) - 1) * 3 / len(raw), rel=1e-6)


def test_article_mode_resets_first_token_and_truncated_disables_bpb():
    raw = "= A =\nabc\n= B =\nxyz\n"
    streams, tok_count, docs = tokenize_eval_streams(
        raw, _GGUFTokenIDs(FakeTokenizer()), document_mode="article"
    )
    assert docs == 2 and len(streams) == 2
    model = ConstantLogitsModel()
    results = run_windowed_evaluation(
        _ExternalWindowModel(model), streams,
        context=4, stride=2, routed_eval=False, memory_cfg=None,
        progress_every=0,
    )
    assert results["total_scored"] == tok_count - 2
    metrics = calculate_summary(results, raw_bytes=len(raw),
                                tokenizer_tokens=tok_count, truncated=True)
    assert metrics["bits_per_byte"] is None
    assert metrics["tokens_per_byte"] is None


def test_zero_scored_tokens_rejected():
    with pytest.raises(ValueError, match="No next-token"):
        calculate_summary({"total_scored": 0}, raw_bytes=5,
                          tokenizer_tokens=1, truncated=True)
