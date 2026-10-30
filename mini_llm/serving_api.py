"""Small OpenAI Chat Completions / Completions HTTP surface for native inference.

One GPU-resident model and one inference request at a time. The HTTP layer
never reimplements attention, projections, checkpoint loading or tokenization.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
import threading
import time
import uuid

from mini_llm.serving_models import SamplingConfig, load_model


class APIError(Exception):
    def __init__(self, message, status=400, code="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def render_messages(messages, *, template="chatml", tools=None):
    """Explicit, predictable templates; never silently interpret foreign Jinja."""
    if not isinstance(messages, list) or not messages:
        raise APIError("messages must be a nonempty array")
    if template not in ("chatml", "plain"):
        raise APIError(f"Unsupported chat template {template!r}")
    records = []
    for row in messages:
        if not isinstance(row, dict) or row.get("role") not in {
            "system", "developer", "user", "assistant", "tool"
        }:
            raise APIError("Each message requires a valid role")
        role = row["role"]
        content = row.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise APIError("Only text content parts are currently supported")
                parts.append(part.get("text", ""))
            content = "\n".join(parts)
        if not isinstance(content, str):
            raise APIError("Message content must be a string or text-part array")
        if row.get("tool_calls"):
            calls = []
            for item in row["tool_calls"]:
                fn = item.get("function", {})
                calls.append({"name": fn.get("name"), "arguments": fn.get("arguments", "{}")})
            content += "\n" + json.dumps({"tool_calls": calls}, ensure_ascii=False)
        if role == "tool":
            content = f"[{row.get('tool_call_id', '')}] {content}"
        records.append(("system" if role == "developer" else role, content))
    if tools:
        instruction = (
            "You may call external tools supplied by the client. The client executes them; "
            "never claim you ran them yourself. For a tool call, output ONLY one JSON "
            'object: {"tool_calls":[{"name":"tool_name","arguments":{...}}]}. '
            "For a normal reply, output regular text, not JSON. Available tools:\n"
            + json.dumps(tools, ensure_ascii=False)
        )
        records.insert(0, ("system", instruction))
    if template == "chatml":
        return "".join(f"<|im_start|>{role}\n{text}<|im_end|>\n"
                       for role, text in records) + "<|im_start|>assistant\n"
    return "".join(f"{role.upper()}:\n{text}\n\n" for role, text in records) + "ASSISTANT:\n"


def parse_tool_envelope(text, tools):
    """Only recognize exact JSON envelopes referring to supplied tool names."""
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"tool_calls"}:
        return None
    calls = obj["tool_calls"]
    if not isinstance(calls, list) or not calls:
        return None
    names = {t["function"]["name"] for t in tools}
    result = []
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in names:
            return None
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if not isinstance(args, dict):
            return None
        result.append({"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                       "function": {"name": call["name"],
                                    "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    return result


# Public compatibility hook: tests and existing integrations may monkeypatch this
# name. Actual sampling now lives in mini_llm.generation.
from .generation import sample_token


class ModelManager:
    def __init__(self, registry, *, loader=load_model):
        self.registry = registry
        self.loader = loader
        self.lock = threading.RLock()
        self.current_id = None
        self.current = None
        self.remote_engines = {}

    def acquire(self, model_id):
        if model_id not in self.registry:
            raise APIError(f"Unknown model {model_id!r}", 404, "model_not_found")
        spec = self.registry[model_id]
        if spec.format == "remote":
            # Completion-level remote models neither consume GPU memory nor
            # mutate local numerical runtime flags. Preserve the loaded local
            # engine and allow concurrent remote requests.
            with self.lock:
                if model_id not in self.remote_engines:
                    self.remote_engines[model_id] = self.loader(spec)
                return self.remote_engines[model_id]
        if self.current_id == model_id:
            return self.current
        # Do not keep multiple VRAM-sized models resident on small GPUs.
        self.current_id = None
        self.current = None
        import gc
        gc.collect()
        try:
            from mini_llm.backend import BACKEND_NAME, xp
            if BACKEND_NAME == "cupy":
                xp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass
        self.current = self.loader(self.registry[model_id])
        self.current_id = model_id
        return self.current


def token_events(model, prompt, *, max_tokens, temperature, top_p, top_k, stop,
                 repetition_penalty=1.0, repetition_window=256,
                 frequency_penalty=0.0, presence_penalty=0.0,
                 text_streaming=True):
    """Backward-compatible HTTP events from the shared generation session.

    Non-streamed responses decode once, avoiding quadratic BPE decoding;
    streamed responses or explicit stop strings retain incremental decoding.
    """
    from .generation import GenerationSession
    session = GenerationSession(model, sampler=lambda logits, **kw: sample_token(logits, **kw))
    try:
        if not text_streaming and stop:
            raise ValueError("Stop strings require incremental decoding")
        for event in session.stream(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=stop,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            emit_text=text_streaming):
            if event["kind"] == "done" and getattr(model, "execution_policy", None) == "strict-fast":
                report = session.execution_info()
                if report is None:
                    raise RuntimeError("strict-fast: no execution report after decoding")
                report.require_fast()
            if event["kind"] == "done" and not text_streaming:
                full_text = session.model.decode(session.generated_ids)
                if full_text:
                    yield {"kind": "content", "text": full_text}
            yield event
    except ValueError as exc:
        raise APIError(str(exc)) from exc


def _normalize_request(request, *, chat):
    if not isinstance(request, dict):
        raise APIError("Request body must be a JSON object")
    model = request.get("model")
    if not isinstance(model, str) or not model:
        raise APIError("model must be a nonempty string")
    messages = request.get("messages") if chat else None
    prompt = request.get("prompt") if not chat else None
    if chat and not isinstance(messages, list):
        raise APIError("messages must be an array")
    if not chat and not isinstance(prompt, str):
        raise APIError("prompt must be a string")
    value = request.get("max_completion_tokens", request.get("max_tokens", 256))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise APIError("max_tokens / max_completion_tokens must be a positive integer")
    # Model-specific defaults are resolved after looking up the model ID.
    # Explicit per-request sampling keys override only the named parameters.
    sampling_overrides = {key: request[key]
                          for key in SamplingConfig.__dataclass_fields__
                          if key in request and request[key] is not None}
    stop = request.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(x, str) for x in stop):
        raise APIError("stop must be a string or array of strings")
    if request.get("n", 1) != 1:
        raise APIError("Only n=1 completions are supported")
    response_format = request.get("response_format") or {"type": "text"}
    if not isinstance(response_format, dict) or response_format.get("type", "text") != "text":
        raise APIError("Constrained response_format is not implemented")
    tools = request.get("tools") or []
    if not isinstance(tools, list):
        raise APIError("tools must be an array")
    for tool in tools:
        if (not isinstance(tool, dict) or tool.get("type") != "function"
            or not isinstance(tool.get("function"), dict)
            or not isinstance(tool["function"].get("name"), str)):
            raise APIError("Only named function tools are supported")
    return {"model": model, "messages": messages, "prompt": prompt,
            "max_tokens": value, "sampling_overrides": sampling_overrides, "stop": stop,
            "tools": tools, "stream": request.get("stream", False) is True,
            "tool_choice": request.get("tool_choice", "auto")}


def create_server(manager, *, host="127.0.0.1", port=8000, api_key=None):
    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("Binding beyond localhost requires --api-key")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, error):
            self._json({"error": {"message": str(error), "type": error.code}}, error.status)

        def _authenticated(self):
            if api_key is None:
                return True
            val = self.headers.get("Authorization", "")
            return hmac.compare_digest(val, f"Bearer {api_key}")

        def do_GET(self):
            if not self._authenticated():
                return self._error(APIError("Unauthorized", 401, "authentication_error"))
            if self.path in ("/health", "/healthz"):
                return self._json({"status": "ok", "loaded_model": manager.current_id})
            if self.path in ("/v1/models", "/models"):
                return self._json({"object": "list", "data": [
                    {"id": mid, "object": "model", "created": 0, "owned_by": "my_local_llm",
                     "context_length": spec.context_length,
                     "capabilities": {"tools": spec.tool_format != "none", "images": False}}
                    for mid, spec in manager.registry.items()]})
            self._error(APIError("Not found", 404, "not_found"))

        def do_POST(self):
            self._sse_started = False
            if not self._authenticated():
                return self._error(APIError("Unauthorized", 401, "authentication_error"))
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                return self._error(APIError("Not found", 404, "not_found"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise APIError("Request body must be within 16 MiB", 413)
                request = json.loads(self.rfile.read(length))
                chat = self.path.endswith("chat/completions")
                # A remote engine owns the upstream prompt template, sampling
                # and tool protocol. Forward the client's OpenAI payload; do
                # not tokenize or wrap it in our in-process ChatML template.
                if isinstance(request, dict) and isinstance(request.get("model"), str):
                    spec = manager.registry.get(request["model"])
                    if spec is not None and spec.format == "remote":
                        self._remote_completion(request, spec, chat=chat)
                        return
                args = _normalize_request(request, chat=chat)
                spec = manager.registry.get(args["model"])
                if spec is None:
                    raise APIError(f"Unknown model {args['model']}", 404, "model_not_found")
                tools = args["tools"] if args["tool_choice"] != "none" else []
                if tools and spec.tool_format == "none":
                    raise APIError("Model is not configured for tool calling; use a tool-capable model")
                if tools and not chat:
                    raise APIError("Tool calling requires chat/completions")
                if tools and args["tool_choice"] not in ("auto", "required", None):
                    raise APIError("Forced named tool_choice is not implemented")
                prompt = (render_messages(args["messages"], template=spec.chat_template, tools=tools)
                          if chat else args["prompt"])
                try:
                    sampling = spec.sampling.with_overrides(args["sampling_overrides"])
                except ValueError as exc:
                    raise APIError(f"Invalid sampling parameters: {exc}") from exc
                with manager.lock:
                    model = manager.acquire(args["model"])
                    events = token_events(model, prompt, max_tokens=args["max_tokens"],
                                          **sampling.__dict__, stop=args["stop"],
                                          text_streaming=bool(args["stop"]) or
                                          (args["stream"] and not tools))
                    first = next(events)  # Prefill-token count, before sending headers
                    completion_id = "chatcmpl-" + uuid.uuid4().hex
                    created = int(time.time())
                    if args["stream"]:
                        self._stream_completion(events, first, args, completion_id, created,
                                                tools=tools, chat=chat)
                    else:
                        self._complete(events, first, args, completion_id, created,
                                       tools=tools, chat=chat)
            except (APIError, ValueError, TypeError, json.JSONDecodeError) as exc:
                err = (exc if isinstance(exc, APIError) else
                       APIError(f"Invalid request or model configuration: {exc}"))
                if self._sse_started:
                    self._sse_error(err)
                else:
                    self._error(err)
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as exc:
                self.log_error("Generation error: %s: %s", type(exc).__name__, exc)
                try:
                    err = APIError(f"Inference failed: {type(exc).__name__}: {exc}",
                                   500, "server_error")
                    if self._sse_started:
                        self._sse_error(err)
                    else:
                        self._error(err)
                except (OSError, ValueError):
                    pass

        def _remote_completion(self, request, spec, *, chat):
            from .remote_provider import RemoteProviderError
            try:
                engine = manager.acquire(spec.id)
                if request.get("stream") is True:
                    # Fetch the upstream response before committing local SSE
                    # headers, so connection and provider errors remain JSON.
                    stream = engine.stream(request, chat=chat, public_id=spec.id)
                    try:
                        first = next(stream)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self._sse_started = True
                        self.close_connection = True
                        self.wfile.write(first)
                        for line in stream:
                            self.wfile.write(line)
                            # Flush completed SSE events immediately. Delaying
                            # until exhaustion would destroy token streaming.
                            if line in (b"\n", b"\r\n"):
                                self.wfile.flush()
                        self.wfile.flush()
                    finally:
                        stream.close()
                else:
                    result = engine.complete(request, chat=chat, public_id=spec.id)
                    self._json(result)
            except RemoteProviderError as exc:
                error = APIError(str(exc), exc.status, exc.code)
                if self._sse_started:
                    self._sse_error(error)
                else:
                    self._error(error)

        def _collect(self, events):
            text_parts = []
            reason, completion_tokens = "length", 0
            for event in events:
                if event["kind"] == "content":
                    text_parts.append(event["text"])
                elif event["kind"] == "done":
                    reason, completion_tokens = event["reason"], event["count"]
            return "".join(text_parts), reason, completion_tokens

        def _choice(self, text, reason, tools, *, chat):
            if tools:
                parsed = parse_tool_envelope(text, tools)
                if parsed:
                    return {"index": 0, "finish_reason": "tool_calls",
                            "message": {"role": "assistant", "content": None,
                                        "tool_calls": parsed}}
            if chat:
                return {"index": 0, "finish_reason": reason,
                        "message": {"role": "assistant", "content": text}}
            return {"index": 0, "finish_reason": reason, "text": text}

        def _complete(self, events, first, args, identifier, created, *, tools, chat):
            text, reason, count = self._collect(events)
            if tools and args["tool_choice"] == "required" and not parse_tool_envelope(text, tools):
                raise APIError("Model did not produce a valid required tool call", 422)
            self._json({"id": identifier, "object": "chat.completion" if chat else "text_completion",
                        "created": created, "model": args["model"],
                        "choices": [self._choice(text, reason, tools, chat=chat)],
                        "usage": {"prompt_tokens": first["count"], "completion_tokens": count,
                                  "total_tokens": first["count"] + count}})

        def _sse(self, row):
            self.wfile.write(b"data: " + json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        def _sse_error(self, error):
            # After 200 SSE headers we cannot send a second HTTP status code.
            self._sse({"error": {"message": str(error), "type": error.code}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _stream_completion(self, events, first, args, identifier, created, *, tools, chat):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._sse_started = True
            self.close_connection = True
            obj_type = "chat.completion.chunk" if chat else "text_completion"

            def chunk(delta, reason=None):
                choice = ({"index": 0, "delta": delta, "finish_reason": reason}
                          if chat else
                          {"index": 0, "text": delta.get("text", ""),
                           "logprobs": None, "finish_reason": reason})
                return {"id": identifier, "object": obj_type, "created": created,
                        "model": args["model"], "choices": [choice]}
            if chat:
                self._sse(chunk({"role": "assistant", "content": ""}))
            # With tools we buffer the entire answer: tool arguments must be
            # validated before advertising an OpenAI tool_calls finish_reason.
            if tools:
                output, reason, completion_tokens = self._collect(events)
                parsed = parse_tool_envelope(output, tools)
                if not parsed and args["tool_choice"] == "required":
                    raise APIError("Model did not produce a valid required tool call", 422)
                if parsed:
                    self._sse(chunk({"tool_calls": [{"index": i, **call}
                                                  for i, call in enumerate(parsed)]}))
                    reason = "tool_calls"
                else:
                    self._sse(chunk({"content": output}))
            else:
                reason, completion_tokens = "length", 0
                for event in events:
                    if event["kind"] == "content":
                        self._sse(chunk({"content": event["text"]} if chat else {"text": event["text"]}))
                    elif event["kind"] == "done":
                        reason, completion_tokens = event["reason"], event["count"]
            self._sse(chunk({}, reason))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, fmt, *args):
            super().log_message(fmt, *args)

    return ThreadingHTTPServer((host, port), Handler)
