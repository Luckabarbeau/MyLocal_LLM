"""0103E: one token lifecycle behind CLI, benchmarks and HTTP."""
from __future__ import annotations

import numpy as np
import pytest

from mini_llm.generation import GenerationSession, sample_token
from mini_llm.serving_api import token_events


class Fake:
    context_length = 16
    stop_ids = {3}

    def __init__(self):
        self.states = []
        self.processed = []

    def encode(self, prompt):
        return [1, 2]

    def decode(self, ids):
        return "".join({4: "a", 5: "b", 6: "c", 3: "<eos>"}[i] for i in ids)

    def create_state(self, size):
        state = {"capacity": size}
        self.states.append(state)
        return state

    def prefill(self, ids, state):
        self.processed.append(("prefill", list(ids), state))
        return np.asarray([[0, 1, 2, -5, 3, 2, 1]], dtype=np.float32)

    def decode_one(self, token, state):
        self.processed.append(("decode", token, state))
        return self.prefill([], state)


def test_common_sampler_greedy_matches_http_and_does_not_modify_logits():
    row = np.asarray([[0., 10., -2., 5.]], dtype=np.float32)
    from mini_llm.serving_api import sample_token as service_sample
    assert sample_token(row, temperature=0, repetition_penalty=3, recent_counts={1: 5}) == 3
    assert service_sample(row, temperature=0, repetition_penalty=3, recent_counts={1: 5}) == 3
    np.testing.assert_array_equal(row, [[0, 10, -2, 5]])


def test_session_and_http_use_same_tokens_streaming_and_eos():
    model = Fake()
    outputs = iter([4, 5, 3])
    session = GenerationSession(model, sampler=lambda *a, **kw: next(outputs))
    events = list(session.stream("hello", max_tokens=6, stop=["zz"]))
    assert [e.get("text") for e in events if e["kind"] == "content"] == ["a"] + ["b"]
    assert events[-1] == {"kind": "done", "reason": "stop", "count": 2}
    assert session.generated_ids == [4, 5]
    assert len(model.states) == 1
    assert model.processed[0][0] == "prefill"
    assert len([x for x in model.processed if x[0] == "decode"]) == 2


def test_sessions_own_request_state_even_with_shared_model():
    model = Fake()
    s1 = GenerationSession(model, sampler=lambda *_a, **_kw: 4)
    s2 = GenerationSession(model, sampler=lambda *_a, **_kw: 5)
    a = s1.generate("a", max_tokens=1)
    b = s2.generate("b", max_tokens=1)
    assert (a.text, a.token_ids) == ("a", (4,))
    assert (b.text, b.token_ids) == ("b", (5,))
    assert s1.state is not s2.state
    c = s1.generate("a", max_tokens=1)
    assert s1.state is not model.states[0]
    assert c.text == a.text


def test_eos_compatibility_cli_includes_eos_http_does_not():
    model = Fake()
    # Native CLI has historically returned the EOS ID in generation metrics.
    cli = GenerationSession(model, sampler=lambda *a, **k: 3).generate(
        "hello", max_tokens=3, include_stop_token=True)
    assert cli.token_ids == (3,) and cli.text == "<eos>"
    http = GenerationSession(model, sampler=lambda *a, **k: 3).generate(
        "hello", max_tokens=3)
    assert http.token_ids == () and http.text == ""
    assert http.reason == "stop"


def test_stop_prefix_is_not_leaked_by_stream():
    model = Fake()
    token_ids = iter([4, 5, 6])
    session = GenerationSession(model, sampler=lambda *a, **k: next(token_ids))
    events = list(session.stream("a", max_tokens=3, stop=["bc"]))
    assert "".join(e["text"] for e in events if e["kind"] == "content") == "a"
    assert events[-1]["reason"] == "stop"


def test_session_context_validation_and_zero_tokens():
    model = Fake()
    model.context_length = 2
    with pytest.raises(ValueError, match="exceeds context"):
        list(GenerationSession(model).stream("x", max_tokens=2))
    model.context_length = 16
    with pytest.raises(ValueError, match="positive"):
        list(GenerationSession(model).stream("x", max_tokens=0))


def test_model_facade_exposes_request_sessions_without_training_state():
    from mini_llm import Model
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel

    class Tokenizer:
        eos_token = "<eos>"

        def encode(self, text):
            return [2] if text == "<eos>" else [4, 5]

        def decode(self, ids):
            return "".join("x" for _ in ids)

    config = ModelConfig(tokenizer_vocab_size=128, context_length=32,
                         n_layers=1, d_model=32, n_q_heads=4, n_kv_heads=2,
                         d_head=8, n_experts=1, top_k=1, d_ff=64,
                         ffn_mode="dense", dtype="float32")
    model = Model.wrap_native(DecoderLanguageModel(config), tokenizer=Tokenizer())
    assert model.capabilities.streaming
    assert not model.capabilities.persistent_sessions  # no append-only multi-turn KV yet
    session = model.session()
    outcome = session.generate("abc", max_tokens=2, temperature=0)
    assert len(outcome.token_ids) <= 2
    assert session.state is not None
    assert model.session().state is None  # independent new request state


def test_batch_output_decodes_once_not_quadratically():
    class CountDecode(Fake):
        def __init__(self):
            super().__init__()
            self.decode_calls = 0

        def decode(self, ids):
            self.decode_calls += 1
            return super().decode(ids)

    model = CountDecode()
    result = GenerationSession(model, sampler=lambda *a, **k: 4).generate(
        "hello", max_tokens=10)
    assert result.text == "a" * 10
    assert model.decode_calls == 1


def test_sampling_strategies_use_shared_sampler(monkeypatch):
    from mini_llm.legacy_generation import greedy_decode, top_k_sample, top_p_sample
    row = np.array([[1, 8, 2]], dtype=np.float32)
    assert greedy_decode(row) == sample_token(row, temperature=0)
    assert top_k_sample(row, 1, temperature=0.8) == 1
    assert top_p_sample(row, p=0.01, temperature=0.8) == 1


def test_nonstreamed_http_response_decodes_once():
    class CountDecode(Fake):
        def __init__(self):
            super().__init__()
            self.decode_calls = 0

        def decode(self, ids):
            self.decode_calls += 1
            return super().decode(ids)

    model = CountDecode()
    events = list(token_events(
        model, "hello", max_tokens=6, temperature=0,
        top_p=1.0, top_k=None, stop=[], text_streaming=False))
    assert model.decode_calls == 1
    assert events[-1]["kind"] == "done"
    assert "".join(e["text"] for e in events if e["kind"] == "content") == "a" * 6
