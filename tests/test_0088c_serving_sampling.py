"""0088C: per-model sampling defaults and bounded repetition penalties (CPU)."""
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from mini_llm.serving_models import ModelSpec, SamplingConfig, load_specs
from mini_llm.serving_api import ModelManager, create_server, sample_token, token_events


class TinyModel:
    context_length = 128
    stop_ids = {3}

    def encode(self, text):
        return [1, 2, 2, 2]

    def decode(self, ids):
        return "x" * len(ids)

    def create_state(self, size):
        return {}

    def prefill(self, ids, state):
        return np.array([[0, 0, 0, 10, 6, 5, 4, 3, 2, 1]], dtype=np.float32)

    def decode_one(self, token, state):
        return self.prefill([], state)


def post(url, data):
    req = Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return urlopen(req, timeout=10)


def test_per_model_registry_sampling(tmp_path):
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": [
        {"id": "native", "path": "native", "sampling": {
            "temperature": 0.8, "top_p": 0.9, "top_k": 40,
            "repetition_penalty": 1.1, "repetition_window": 256}},
        {"id": "external", "path": "model.gguf", "sampling": {"temperature": 0.6}}
    ]}))
    specs = load_specs(manifest)
    assert specs["native"].sampling.temperature == 0.8
    assert specs["native"].sampling.repetition_penalty == 1.1
    assert specs["external"].sampling.temperature == 0.6
    assert specs["external"].sampling.repetition_penalty == 1.0
    assert specs["native"].sampling is not specs["external"].sampling


@pytest.mark.parametrize("sampling", [
    {"temperature": -1}, {"temperature": True}, {"temperature": "0.8"},
    {"top_p": 0}, {"top_k": 0}, {"top_k": True},
    {"repetition_penalty": 0}, {"repetition_window": -1},
    {"repetition_window": 1.5}, {"frequency_penalty": 4},
    {"presence_penalty": float("nan")}, {"unknown_parameter": 1},
    [0.8],
])
def test_reject_invalid_registry_sampling(sampling):
    with pytest.raises(ValueError):
        ModelSpec.from_dict({"id": "native", "path": "native", "sampling": sampling}, Path.cwd())


def test_sampling_preserves_old_defaults_when_missing():
    spec = ModelSpec.from_dict({"id": "test", "path": "model.gguf"}, Path.cwd())
    assert spec.sampling == SamplingConfig()
    assert spec.sampling.temperature == 0.0
    assert spec.sampling.repetition_penalty == 1.0


def test_sign_aware_greedy_repetition_penalty():
    row = np.array([[0.0, 10.0, -2.0, 5.0]], dtype=np.float32)
    assert sample_token(row, temperature=0) == 1
    assert sample_token(row, temperature=0, repetition_penalty=3, recent_counts={1: 5}) == 3
    assert np.array_equal(row, [[0, 10, -2, 5]])  # original logits must not be mutated


def test_frequency_and_presence_penalty():
    row = np.array([[0.0, 10.0, 8.0, 0.0]], dtype=np.float32)
    assert sample_token(row, temperature=0, frequency_penalty=1.5,
                        recent_counts={1: 2}) == 2
    assert sample_token(row, temperature=0, presence_penalty=3,
                        recent_counts={1: 1}) == 2


def test_bounded_history_updates_including_prompt(monkeypatch):
    import mini_llm.serving_api as api
    seen = []
    outputs = iter([4, 5, 3])

    def fake_sampler(logits, **options):
        seen.append(dict(options["recent_counts"]))
        return next(outputs)

    monkeypatch.setattr(api, "sample_token", fake_sampler)
    events = list(token_events(TinyModel(), "hello", max_tokens=3,
                               temperature=0.8, top_p=0.9, top_k=40,
                               repetition_penalty=1.1, repetition_window=2, stop=[]))
    assert seen == [{2: 2}, {2: 1, 4: 1}, {4: 1, 5: 1}]
    assert events[-1]["kind"] == "done"


def test_http_model_defaults_and_request_override(monkeypatch):
    import mini_llm.serving_api as api
    captures = []

    def fake_sampler(logits, **options):
        captures.append(options)
        return 3  # EOS

    monkeypatch.setattr(api, "sample_token", fake_sampler)
    specs = {
        "native": ModelSpec(id="native", path=Path("native"), sampling=SamplingConfig(
            temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.1)),
        "pretrained": ModelSpec(id="pretrained", path=Path("other"), sampling=SamplingConfig(
            temperature=0.6, top_p=0.95, top_k=50)),
    }
    manager = ModelManager(specs, loader=lambda spec: TinyModel())
    server = create_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        base = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}
        with post(url, {**base, "model": "native"}) as response:
            assert response.status == 200
        assert captures[-1]["temperature"] == 0.8
        assert captures[-1]["repetition_penalty"] == 1.1
        with post(url, {**base, "model": "native", "temperature": 0.4, "top_k": 10}) as response:
            assert response.status == 200
        assert captures[-1]["temperature"] == 0.4
        assert captures[-1]["top_p"] == 0.9
        assert captures[-1]["top_k"] == 10
        assert captures[-1]["repetition_penalty"] == 1.1
        with post(url, {**base, "model": "pretrained"}) as response:
            assert response.status == 200
        assert captures[-1]["temperature"] == 0.6
        assert captures[-1]["repetition_penalty"] == 1.0
        with pytest.raises(HTTPError) as exc:
            post(url, {**base, "model": "native", "repetition_penalty": -1})
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
