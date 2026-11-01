"""0103F: end-to-end remote-model HTTP and SSE adapter tests (CPU only)."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mini_llm.model_sources import ModelSpec
from mini_llm.serving_api import ModelManager, create_server
from mini_llm.serving_models import load_model


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = None
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(content_length))
        self.requests.append((self.path, payload, self.headers.get("Authorization")))
        if payload.get("trigger_error"):
            self.send_response(429)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if payload.get("trigger_bad_json"):
            body = b"<html>bad</html>"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if payload.get("stream"):
            parts = [
                {"object": "chat.completion.chunk", "model": payload["model"],
                 "choices": [{"delta": {"role": "assistant"}, "index": 0}]},
                {"object": "chat.completion.chunk", "model": payload["model"],
                 "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "tool_1", "type": "function",
                                                           "function": {"name": "read_file", "arguments": "{}"}}]},
                              "index": 0, "finish_reason": "tool_calls"}]},
            ]
            raw = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in parts) + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        response = {
            "id": "upstream-id", "object": "chat.completion" if "chat" in self.path else "text_completion",
            "model": payload["model"], "choices": [
                {"index": 0, "finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "tool_1", "type": "function", "function": {
                                 "name": "read_file", "arguments": '{"path":"main.py"}'}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        }
        raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        pass


@contextmanager
def live_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def post(url, payload, *, auth=None):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    request = Request(url, json.dumps(payload).encode(), headers)
    return urlopen(request, timeout=10)


@pytest.fixture
def servers(monkeypatch):
    Upstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    with live_server(upstream) as upstream_url:
        monkeypatch.setenv("TEST_MODEL_KEY", "upstream-secret")
        spec = ModelSpec.from_dict({
            "id": "local-alias", "format": "remote", "api_url": upstream_url + "/v1",
            "remote_model": "provider-model", "api_key_env": "TEST_MODEL_KEY",
            "tool_format": "native", "context_length": 32768,
        }, Path.cwd())
        registry = {"local-alias": spec}
        manager = ModelManager(registry, loader=load_model)
        local = create_server(manager, port=0, api_key="local-secret")
        with live_server(local) as local_url:
            yield local_url, upstream_url, manager, monkeypatch


def test_nonstream_tools_and_alias_mapping(servers):
    local, _upstream, _manager, _ = servers
    request = {
        "model": "local-alias", "messages": [{"role": "user", "content": "Run my tool"}],
        "max_tokens": 55, "temperature": 0.35,
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}}}}],
        "tool_choice": "required", "response_format": {"type": "text"},
    }
    with post(local + "/v1/chat/completions", request, auth="local-secret") as reply:
        data = json.load(reply)
    assert data["model"] == "local-alias"
    assert data["usage"]["total_tokens"] == 12
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    path, forwarded, auth = Upstream.requests[-1]
    assert path == "/v1/chat/completions"
    assert forwarded["model"] == "provider-model"
    assert forwarded["messages"] == request["messages"]
    assert forwarded["tools"] == request["tools"]
    assert forwarded["tool_choice"] == "required"
    assert forwarded["temperature"] == 0.35
    assert auth == "Bearer upstream-secret"  # never relay local API key


def test_streamed_tool_call_deltas_are_preserved(servers):
    local, *_ = servers
    with post(local + "/v1/chat/completions", {
        "model": "local-alias", "messages": [{"role": "user", "content": "hello"}],
        "stream": True, "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }, auth="local-secret") as response:
        assert response.headers["Content-Type"].startswith("text/event-stream")
        lines = response.read().decode().splitlines()
    chunks = [line[6:].strip() for line in lines if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    data = [json.loads(chunk) for chunk in chunks[:-1]]
    assert all(row["model"] == "local-alias" for row in data)
    assert data[-1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_text_completions_forwarding(servers):
    local, *_ = servers
    with post(local + "/v1/completions", {"model": "local-alias", "prompt": "hello", "max_tokens": 3},
              auth="local-secret") as response:
        result = json.load(response)
    assert result["model"] == "local-alias"
    assert Upstream.requests[-1][0] == "/v1/completions"


def test_upstream_http_errors_do_not_reveal_credentials(servers):
    local, *_ = servers
    with pytest.raises(HTTPError) as err:
        post(local + "/v1/chat/completions", {
            "model": "local-alias", "messages": [], "trigger_error": True,
        }, auth="local-secret")
    assert err.value.code == 429
    assert b"upstream-secret" not in err.value.read()


def test_upstream_invalid_json_is_rejected(servers):
    local, *_ = servers
    with pytest.raises(HTTPError) as err:
        post(local + "/v1/completions", {"model": "local-alias", "trigger_bad_json": True},
             auth="local-secret")
    assert err.value.code == 502


def test_upstream_key_missing_is_reported_without_gpu_allocation(servers):
    local, _upstream, manager, monkeypatch = servers
    monkeypatch.delenv("TEST_MODEL_KEY")
    with pytest.raises(HTTPError) as err:
        post(local + "/v1/completions", {"model": "local-alias"}, auth="local-secret")
    assert err.value.code == 503
    assert manager.current is None


def test_remote_does_not_evict_loaded_local_model(servers):
    local, _, manager, _ = servers
    # Simulate an existing native model resident in GPU memory.
    sentinel = object()
    with manager.lock:
        manager.current_id = "native-active"
        manager.current = sentinel
    with post(local + "/v1/completions", {"model": "local-alias"}, auth="local-secret") as response:
        json.load(response)
    assert manager.current is sentinel and manager.current_id == "native-active"
    assert manager.acquire("local-alias") is manager.acquire("local-alias")


def test_registry_and_authentication(servers):
    local, *_ = servers
    with pytest.raises(HTTPError) as err:
        post(local + "/v1/completions", {"model": "local-alias"})
    assert err.value.code == 401
    with urlopen(Request(local + "/v1/models", headers={"Authorization": "Bearer local-secret"}), timeout=5) as response:
        rows = json.load(response)["data"]
    assert rows[0]["id"] == "local-alias"
    assert rows[0]["capabilities"]["tools"] is True


def test_remote_registry_rejects_redirect_or_url_credentials(tmp_path):
    base = {"id": "x", "format": "remote", "remote_model": "m"}
    for url in ("http://user:pw@localhost/v1", "file:///tmp/no", "http://localhost/v1?x=abc"):
        with pytest.raises(ValueError):
            ModelSpec.from_dict({**base, "api_url": url}, tmp_path)
