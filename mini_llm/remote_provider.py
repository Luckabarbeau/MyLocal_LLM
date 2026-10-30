"""OpenAI-compatible completion transport for externally executed models.

A remote engine operates at the HTTP completion boundary: it cannot provide
local logits, KV caches, training, or token-level generation sessions. This
module deliberately has no NumPy/CuPy imports and never allocates GPU memory.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class RemoteProviderError(Exception):
    """Upstream failure with an HTTP status safe to expose through our API."""

    def __init__(self, message: str, *, status: int = 502, code: str = "upstream_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class _NoRedirects(HTTPRedirectHandler):
    """Do not forward configured provider credentials to redirected hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteCompletionEngine:
    """Single configured remote model; make each call independent and stateless.

    Providers receive the client's OpenAI-compatible payload, with only the
    public model alias replaced by their actual model ID. Upstream manages
    tokenization, sampling, tool calling, context and response formatting.
    """

    MAX_RESPONSE_BYTES = 32 * 1024 * 1024
    TIMEOUT_SECONDS = 120

    def __init__(self, spec, *, opener=None):
        if spec.resolved_format() != "remote":
            raise ValueError("RemoteCompletionEngine requires format='remote'")
        self.spec = spec
        self.opener = opener if opener is not None else build_opener(_NoRedirects)
        self.is_remote = True
        self.context_length = spec.context_length

    def _endpoint(self, *, chat: bool) -> str:
        base = self.spec.api_url.rstrip("/")
        return base + ("/chat/completions" if chat else "/completions")

    def open(self, payload: dict, *, chat: bool):
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        body = dict(payload, model=self.spec.remote_model)
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.spec.api_key_env:
            key = os.environ.get(self.spec.api_key_env)
            if not key:
                raise RemoteProviderError(
                    f"Remote provider credential environment variable {self.spec.api_key_env} is not set",
                    status=503, code="provider_not_configured")
            headers["Authorization"] = "Bearer " + key
        request = Request(self._endpoint(chat=chat),
                          data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            return self.opener.open(request, timeout=self.TIMEOUT_SECONDS)
        except HTTPError as exc:
            # Preserve status, but not untrusted upstream details that might
            # contain credentials or arbitrary HTML from a proxy.
            status = int(exc.code)
            exc.close()
            raise RemoteProviderError(
                f"Upstream model returned HTTP {status}",
                status=status if 400 <= status < 600 else 502,
                code="upstream_http_error") from None
        except (TimeoutError, URLError, OSError) as exc:
            raise RemoteProviderError("Could not connect to the configured model provider") from exc

    def complete(self, payload: dict, *, chat: bool, public_id: str) -> dict:
        """Forward one non-streamed call and return OpenAI-shaped JSON."""
        with self.open(payload, chat=chat) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                raise RemoteProviderError("Provider streamed a non-streaming request")
            raw = response.read(self.MAX_RESPONSE_BYTES + 1)
            if len(raw) > self.MAX_RESPONSE_BYTES:
                raise RemoteProviderError("Upstream response exceeds size limit")
        try:
            obj = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RemoteProviderError("Upstream returned invalid JSON") from exc
        if not isinstance(obj, dict):
            raise RemoteProviderError("Upstream completion response must be a JSON object")
        obj["model"] = public_id
        return obj

    def stream(self, payload: dict, *, chat: bool, public_id: str):
        """Yield serialized SSE lines, preserving upstream deltas/tool calls.

        Never synthesize tool calls or reformat model output. In particular,
        remote tool calls arrive as native OpenAI tool_calls deltas.
        """
        with self.open(payload, chat=chat) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RemoteProviderError("Provider did not return SSE for a streaming request")
            done = False
            total = 0
            for line in response:
                total += len(line)
                if total > self.MAX_RESPONSE_BYTES:
                    raise RemoteProviderError("Upstream stream exceeds size limit")
                # A line is either metadata/comment, an empty delimiter, or data.
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                    elif data:
                        try:
                            event = json.loads(data)
                            if isinstance(event, dict) and "model" in event:
                                event["model"] = public_id
                                line = b"data: " + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                        except (UnicodeDecodeError, ValueError):
                            # Preserve non-JSON provider extensions.
                            pass
                yield line
            if not done:
                raise RemoteProviderError("Provider stream ended without [DONE]")
