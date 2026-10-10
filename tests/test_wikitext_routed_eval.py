from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from types import SimpleNamespace

import numpy as np

from evaluate_wikitext import build_routed_eval_window, resolve_memory_mode


def _memory_cfg():
    return SimpleNamespace(
        distant_memory_length=8,
        target_length=4,
        source_input_length=12,
        router_temperature_min=0.25,
    )


def test_routed_window_right_aligns_partial_history():
    tokens = list(range(100, 120))
    source, targets, metadata = build_routed_eval_window(
        tokens,
        begin=3,
        target_end=8,
        memory_cfg=_memory_cfg(),
    )

    assert source.shape == (1, 12)
    # Three real historical tokens are right-aligned in the eight-token store.
    np.testing.assert_array_equal(source[0, 5:8], np.array([100, 101, 102]))
    np.testing.assert_array_equal(source[0, 8:12], np.array([103, 104, 105, 106]))
    np.testing.assert_array_equal(targets[0], np.array([104, 105, 106, 107]))
    np.testing.assert_array_equal(metadata["history_valid_starts"], np.array([5]))
    np.testing.assert_array_equal(metadata["target_valid_lengths"], np.array([4]))
    assert metadata["use_retrieval"] is True
    assert metadata["router_stochastic"] is False
    assert metadata["router_temperature"] == 0.25


def test_routed_window_keeps_only_bounded_recent_history():
    tokens = list(range(100, 140))
    source, targets, metadata = build_routed_eval_window(
        tokens,
        begin=12,
        target_end=17,
        memory_cfg=_memory_cfg(),
    )

    # History capacity is eight, so tokens 104..111 are retained.
    np.testing.assert_array_equal(source[0, :8], np.arange(104, 112))
    np.testing.assert_array_equal(source[0, 8:12], np.arange(112, 116))
    np.testing.assert_array_equal(targets[0], np.arange(113, 117))
    np.testing.assert_array_equal(metadata["history_valid_starts"], np.array([0]))


def test_explicit_routed_mode_overrides_legacy_disable_flag():
    mode, overridden = resolve_memory_mode("routed", True)
    assert mode == "routed"
    assert overridden is True


def test_legacy_disable_flag_still_selects_plain_mode():
    mode, overridden = resolve_memory_mode("auto", True)
    assert mode == "plain"
    assert overridden is False

from evaluate_wikitext import (
    routed_inference_mode_from_state,
    single_token_nll,
    split_wikitext_articles,
    tokenize_eval_streams,
)


def test_article_split_keeps_sections_inside_article_and_preserves_text():
    raw = (
        "\n = Article One = \n\n"
        "first paragraph\n"
        "\n = = Section = = \n"
        "section text\n"
        "\n = Article Two = \n\n"
        "second article\n"
    )
    articles = split_wikitext_articles(raw)
    assert len(articles) == 2
    assert "= = Section = =" in articles[0]
    assert "Article Two" not in articles[0]
    assert "Article Two" in articles[1]
    assert "".join(articles) == raw


class _ToyTokenizer:
    def encode(self, text):
        # Deliberately make tokenization local to each supplied document so the
        # test can verify that article mode invokes encode independently.
        return [len(part) for part in text.split()]


def test_article_tokenization_uses_independent_streams_and_global_limit():
    raw = " = A = \nalpha beta\n = B = \ngamma delta epsilon\n"
    streams, total, detected = tokenize_eval_streams(
        raw, _ToyTokenizer(), document_mode="article", max_tokens=6
    )
    assert detected == 2
    assert len(streams) == 2
    assert total == 6
    assert sum(map(len, streams)) == 6


def test_routed_inference_state_mode_classification():
    none_state = SimpleNamespace(routed_prefix_route=None)
    assert routed_inference_mode_from_state(none_state) == "none"

    direct_route = SimpleNamespace(selected_blocks=np.empty((1, 0), dtype=np.int64))
    direct_state = SimpleNamespace(routed_prefix_route=direct_route)
    assert routed_inference_mode_from_state(direct_state) == "direct_history"

    routed_route = SimpleNamespace(selected_blocks=np.asarray([[1, 3]], dtype=np.int64))
    routed_state = SimpleNamespace(routed_prefix_route=routed_route)
    assert routed_inference_mode_from_state(routed_state) == "routed_blocks"


def test_single_token_nll_matches_known_softmax():
    logits = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    got = single_token_nll(logits, 2)
    expected = np.log(np.exp(0.0) + np.exp(1.0) + np.exp(2.0)) - 2.0
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from evaluate_wikitext import run_autoregressive_evaluation


def test_autoregressive_eval_runs_real_routed_inference_path():
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=4,
        block_size=2,
        top_k_blocks=1,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        integration_mode="routed_prefix",
        memory_training="joint",
        read_heads=1,
        read_kv_heads=1,
        inference_route_refresh_tokens=2,
    )
    config = ModelConfig(
        tokenizer_vocab_size=32,
        context_length=memory.active_length,
        n_layers=1,
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        n_experts=2,
        top_k=1,
        d_ff=16,
        dtype="float64",
        memory_context=memory,
    )
    model = DecoderLanguageModel(config, rng_seed=3, dtype=config.dtype)
    result = run_autoregressive_evaluation(
        model,
        config,
        [list(range(1, 13))],
        memory_cfg=memory,
        progress_every=0,
    )
    assert result["total_scored"] == 11
    assert np.isfinite(result["total_nll"])
    assert result["routed_modes"]["unknown"] == 0
    assert result["routed_modes"]["none"] > 0
    assert result["routed_modes"]["direct_history"] > 0
    assert result["routed_modes"]["routed_blocks"] > 0
    assert result["route_refreshes"] > 0
    assert result["refresh_tokens"] == 2
