"""Small OpenAI Chat Completions / Completions HTTP surface for native inference.

One GPU-resident model and one inference request at a time. The HTTP layer
never reimplements attention, projections, checkpoint loading or tokenization.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter, deque
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


def sample_token(logits, temperature=0.0, top_p=1.0, top_k=None, *,
                 repetition_penalty=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, recent_counts=None):
    """Sample on the active backend, applying bounded recent-token penalties."""
    from mini_llm.backend import xp
    penalize = bool(recent_counts) and (repetition_penalty != 1.0 or
                                         frequency_penalty != 0.0 or presence_penalty != 0.0)
    # Never modify the logits buffer returned by the inference backend.
    row = logits[0].reshape(-1).astype(xp.float32, copy=penalize)
    if penalize:
        # Keep the history/counts on CPU and send only the distinct token IDs
        # in the bounded window; do not transfer a full vocabulary to CPU.
        ids = xp.asarray(list(recent_counts), dtype=xp.int32)
        counts = xp.asarray(list(recent_counts.values()), dtype=xp.float32)
        values = row[ids]
        if repetition_penalty != 1.0:
            values = xp.where(values > 0, values / xp.float32(repetition_penalty),
                              values * xp.float32(repetition_penalty))
        if frequency_penalty != 0:
            values = values - xp.float32(frequency_penalty) * counts
        if presence_penalty != 0:
            values = values - xp.float32(presence_penalty)
        row[ids] = values
    if temperature == 0:
        return int(xp.argmax(row).item())
    if not 0 < top_p <= 1:
        raise APIError("top_p must be within (0,1]")
    vocab = int(row.shape[0])
    k = min(vocab, int(top_k) if top_k is not None else vocab)
    if k <= 0:
        raise APIError("top_k must be >= 1")
    indices = (xp.arange(vocab) if k == vocab else
               xp.argpartition(row, vocab - k)[vocab - k:])
    values = row[indices] / xp.float32(temperature)
    if top_p < 1:
        order = xp.argsort(values)[::-1]
        indices, values = indices[order], values[order]
    weights = xp.exp(values - xp.max(values))
    if top_p < 1:
        cum = xp.cumsum(weights)
        keep = (cum - weights) < xp.float32(top_p) * cum[-1]
        weights = xp.where(keep, weights, 0)
    cdf = xp.cumsum(weights)
    point = xp.random.random() * cdf[-1]
    slot = xp.searchsorted(cdf, point, side="right")
    slot = xp.minimum(slot, len(indices) - 1)
    return int(indices[slot].item())


class ModelManager:
    def __init__(self, registry, *, loader=load_model):
        self.registry = registry
        self.loader = loader
        self.lock = threading.RLock()
        self.current_id = None
        self.current = None

    def acquire(self, model_id):
        if model_id not in self.registry:
            raise APIError(f"Unknown model {model_id!r}", 404, "model_not_found")
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
                 frequency_penalty=0.0, presence_penalty=0.0):
    """Yield decoded deltas while preserving partial UTF-8 and stop prefixes."""
    ids = model.encode(prompt)
    if not ids:
        raise APIError("Prompt tokenized to an empty sequence")
    remaining = model.context_length - len(ids)
    if remaining <= 0:
        raise APIError(f"Prompt ({len(ids)} tokens) exceeds context ({model.context_length})")
    max_tokens = min(max_tokens, remaining)
    if max_tokens < 1:
        raise APIError("max_tokens must be positive")
    yield {"kind": "prefill", "count": len(ids)}
    state = model.create_state(len(ids) + max_tokens)
    logits = model.prefill(ids, state)
    generated = []
    # A bounded prompt+generation window: no O(64K) scan or large GPU mask.
    recent = deque(ids[-repetition_window:] if repetition_window else [])
    counts = Counter(recent)
    emitted = ""
    decoded = ""
    stops = [s for s in stop if s]
    stop_buffer = max((len(s) - 1 for s in stops), default=0)
    reason = "length"
    for iteration in range(max_tokens):
        token_id = sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k,
                                repetition_penalty=repetition_penalty,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty, recent_counts=counts)
        if token_id in model.stop_ids:
            reason = "stop"
            break
        generated.append(token_id)
        if repetition_window:
            if len(recent) >= repetition_window:
                old = recent.popleft()
                counts[old] -= 1
                if not counts[old]:
                    del counts[old]
            recent.append(token_id)
            counts[token_id] += 1
        decoded = model.decode(generated)
        pos = min((decoded.index(s) for s in stops if s in decoded), default=-1)
        if pos >= 0:
            decoded = decoded[:pos]
            reason = "stop"
        # A byte-level BPE decoder can temporarily produce a replacement glyph
        # for an incomplete UTF-8 sequence. Do not stream that unstable tail.
        visible = decoded.rstrip("\ufffd")
        if reason != "stop" and stop_buffer:
            visible = visible[:-stop_buffer] if len(visible) > stop_buffer else ""
        if not visible.startswith(emitted):
            raise RuntimeError("Tokenizer decoded output is not prefix-stable")
        delta = visible[len(emitted):]
        if delta:
            emitted = visible
            yield {"kind": "content", "text": delta}
        if reason == "stop":
            break
        if iteration + 1 < max_tokens:
            logits = model.decode_one(token_id, state)
    # Release stop-prefix buffer if generation simply hit its length cap.
    if reason == "length":
        final = decoded.rstrip("\ufffd")
        if final.startswith(emitted) and final != emitted:
            yield {"kind": "content", "text": final[len(emitted):]}
    yield {"kind": "done", "reason": reason, "count": len(generated)}


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
                                          **sampling.__dict__, stop=args["stop"])
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
