"""CPU-only protocol regressions; no checkpoint or GPU required."""
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest

from mini_llm.serving_api import (
    ModelManager, APIError, create_server, parse_tool_envelope, render_messages, token_events,
)
from mini_llm.serving_models import ModelSpec, load_specs


class FakeModel:
    """Deterministic causal model with real HTTP/SSE, no inference kernels."""
    context_length = 1000
    stop_ids = {3}

    def __init__(self, answer="Hello"):
        self.answer = answer

    def encode(self, text):
        return [1, 2]

    def decode(self, tokens):
        return "".join(self.answer for _ in tokens)

    def create_state(self, total):
        return {"calls": 0}

    def prefill(self, ids, state):
        return np.asarray([[0., 2., 0., -1.]])

    def decode_one(self, token, state):
        return np.asarray([[0., 0., 0., 2.]])


def post(url, data):
    payload = json.dumps(data).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    return urlopen(req, timeout=10)


@pytest.fixture
def server():
    spec = ModelSpec(id="test", path=Path("unused"), tool_format="json_envelope")
    current = {"model": FakeModel()}
    manager = ModelManager({"test": spec}, loader=lambda _: current["model"])
    httpd = create_server(manager, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield "http://127.0.0.1:" + str(httpd.server_port), current
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=3)


def test_registry(tmp_path):
    file = tmp_path / "registry.json"
    file.write_text(json.dumps({"models": [{"id": "smol", "path": "model.gguf"}]}))
    models = load_specs(file)
    assert models["smol"].resolved_format() == "gguf"
    assert models["smol"].path == tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="unique IDs"):
        file.write_text(json.dumps({"models": [{"id": "x", "path": "a.gguf"},
                                                 {"id": "x", "path": "b.gguf"}]}))
        load_specs(file)


def test_prompt_and_tool_parser():
    x = render_messages([{"role": "system", "content": "Use tools"},
                         {"role": "user", "content": "Write code"}], template="chatml")
    assert x.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>user\nWrite code<|im_end|>" in x
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    parsed = parse_tool_envelope('{"tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]}', tools)
    assert json.loads(parsed[0]["function"]["arguments"]) == {"path": "x"}
    assert parse_tool_envelope('{"tool_calls":[{"name":"delete_all","arguments":{}}]}', tools) is None


def test_chat_nonstream_and_models(server):
    url, current = server
    with urlopen(url + "/v1/models") as response:
        assert json.load(response)["data"][0]["id"] == "test"
    with post(url + "/v1/chat/completions", {
        "model": "test", "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10}) as response:
        row = json.load(response)
    assert row["choices"][0]["message"]["content"] == "Hello"
    assert row["choices"][0]["finish_reason"] == "stop"
    assert row["usage"]["prompt_tokens"] == 2
    assert row["usage"]["completion_tokens"] == 1


def test_chat_stream(server):
    url, current = server
    with post(url + "/v1/chat/completions", {
        "model": "test", "messages": [{"role": "user", "content": "hello"}],
        "stream": True, "max_tokens": 10}) as response:
        lines = response.read().decode("utf-8").splitlines()
    rows = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert any(x["choices"][0]["delta"].get("content") == "Hello" for x in rows)
    assert rows[-1]["choices"][0]["finish_reason"] == "stop"
    assert "data: [DONE]" in lines


def test_tool_stream_json_envelope(server):
    url, current = server
    current["model"] = FakeModel('{"tool_calls":[{"name":"read_file","arguments":{"path":"main.py"}}]}')
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    with post(url + "/v1/chat/completions", {
        "model": "test", "messages": [{"role": "user", "content": "open file"}],
        "tools": tools, "stream": True}) as response:
        lines = response.read().decode().splitlines()
    rows = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    calls = [x["choices"][0]["delta"].get("tool_calls") for x in rows]
    calls = [c for c in calls if c]
    assert len(calls) == 1
    assert calls[0][0]["function"]["name"] == "read_file"
    assert rows[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_bad_requests(server):
    url, _ = server
    with pytest.raises(HTTPError) as exc:
        post(url + "/v1/chat/completions", {"model": "unknown", "messages": []})
    assert exc.value.code == 404
    with pytest.raises(HTTPError) as exc:
        post(url + "/v1/chat/completions", {"model": "test", "messages": [],
                                            "temperature": -2})
    assert exc.value.code == 400
    with pytest.raises(ValueError, match="--api-key"):
        create_server(ModelManager({}), host="0.0.0.0", port=0)


def test_text_completions_stream(server):
    url, _ = server
    with post(url + "/v1/completions", {
        "model": "test", "prompt": "Hello", "stream": True, "max_tokens": 4
    }) as response:
        lines = response.read().decode().splitlines()
    rows = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert rows[0]["object"] == "text_completion"
    assert any(row["choices"][0]["text"] == "Hello" for row in rows)
    assert all("delta" not in row["choices"][0] for row in rows)


def test_reject_unsupported_json_schema(server):
    url, _ = server
    with pytest.raises(HTTPError) as exc:
        post(url + "/v1/chat/completions", {
            "model": "test", "messages": [{"role": "user", "content": "json"}],
            "response_format": {"type": "json_schema"}
        })
    assert exc.value.code == 400


def test_tiny_native_checkpoint_serves_real_inference(tmp_path):
    """Exercise production checkpoint -> InferenceModel -> HTTP, not FakeModel."""
    from dataclasses import asdict
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.serving_models import load_model

    cfg = ModelConfig(tokenizer_vocab_size=128, context_length=128,
                      n_layers=1, d_model=32, n_q_heads=4, n_kv_heads=2,
                      d_head=8, n_experts=1, top_k=1, d_ff=64,
                      ffn_mode="dense", dtype="float32")
    model = DecoderLanguageModel(cfg, rng_seed=5, dtype=cfg.dtype)
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    tokenizer = SimpleBPETokenizer(vocab_size=128)
    tokenizer.train(["hello world!", "the sky is blue.", "blue blue !"], vocab_size=128)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    save_checkpoint(tmp_path, {q.name: q.data for q in model.parameters()})
    spec = ModelSpec(id="native-tiny", path=tmp_path, format="native",
                     dtype="float32", chat_template="plain", context_length=40)
    loaded = load_model(spec)
    assert loaded.context_length == 40
    assert loaded.encode("hello")
    manager = ModelManager({spec.id: spec}, loader=load_model)
    server = create_server(manager, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with post(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", {
            "model": spec.id, "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3}) as response:
            row = json.load(response)
        assert row["choices"][0]["message"]["role"] == "assistant"
        assert row["usage"]["completion_tokens"] <= 3
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=3)
