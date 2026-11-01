"""0103E: framework-neutral request-scoped generation and sampling.

The adapter supplies encode/decode/create_state/prefill/decode_one; this module
owns token selection, stopping, bounded penalties and safe text streaming.
Numerical kernels and backend selection remain in the existing engines.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time
from typing import Callable

from .backend import xp, synchronize


def sample_token(logits, temperature=0.0, top_p=1.0, top_k=None, *,
                 repetition_penalty=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, recent_counts=None):
    """One backend-local sampler shared by CLI, benchmarks and HTTP serving.

    In particular, sign-aware repetition penalties never modify engine logits.
    The sampler draws a single scalar on the active device and transfers only
    the resulting token ID to the host.
    """
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be within (0,1]")
    penalize = bool(recent_counts) and (repetition_penalty != 1.0 or
                                         frequency_penalty != 0.0 or presence_penalty != 0.0)
    row = logits.reshape(-1).astype(xp.float32, copy=penalize)
    if penalize:
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
    vocab = int(row.shape[0])
    k = min(vocab, int(top_k) if top_k is not None else vocab)
    if k <= 0:
        raise ValueError("top_k must be >= 1")
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
    slot = xp.searchsorted(cdf, xp.random.random() * cdf[-1], side="right")
    slot = xp.minimum(slot, len(indices) - 1)
    return int(indices[slot].item())


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: tuple[int, ...]
    reason: str
    prompt_tokens: int
    completion_tokens: int


class GenerationSession:
    """Explicit request lifecycle; weights shared, token caches request-local.

    Reusing the session object starts a new request, with a fresh or reset
    model state. It is not yet an append-only multi-turn KV continuation API.
    Consumers should stream the iterator fully before starting another request.
    """

    def __init__(self, model, *, sampler: Callable = sample_token):
        self.model = model
        self.sampler = sampler
        self.state = None
        self.generated_ids = []
        self.prompt_tokens = 0
        self.reason = None
        self.prefill_time_s = 0.0
        self.forward_times = []
        self.sample_times = []

    def stream(self, prompt, *, max_tokens, temperature=0.0, top_p=1.0,
               top_k=None, stop=(), repetition_penalty=1.0,
               repetition_window=256, frequency_penalty=0.0,
               presence_penalty=0.0, seed=None, log_speed=False,
               include_stop_token=False, emit_text=True):
        """Yield prefill/content/done events for a single autoregressive request."""
        stop = tuple(stop or ())
        if not emit_text and any(stop):
            raise ValueError("Stop strings require streaming text decoding")
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
            xp.random.seed(seed)
        ids = list(self.model.encode(prompt))
        if not ids:
            raise ValueError("Prompt tokenized to an empty sequence")
        remaining = int(self.model.context_length) - len(ids)
        if remaining <= 0:
            raise ValueError(f"Prompt ({len(ids)} tokens) exceeds context ({self.model.context_length})")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        max_tokens = min(max_tokens, remaining)
        self.generated_ids = []
        self.prompt_tokens = len(ids)
        self.reason = "length"
        self.forward_times = []
        self.sample_times = []
        self.prefill_time_s = 0.0
        yield {"kind": "prefill", "count": len(ids)}
        self.state = self.model.create_state(len(ids) + max_tokens)
        if log_speed:
            synchronize()
            start = time.perf_counter()
        logits = self.model.prefill(ids, self.state)
        if log_speed:
            synchronize()
            self.prefill_time_s = time.perf_counter() - start
            self.forward_times.append(self.prefill_time_s)
        recent = deque(ids[-repetition_window:] if repetition_window else [])
        counts = Counter(recent)
        emitted = ""
        decoded = ""
        stops = [s for s in stop if s]
        stop_buffer = max((len(s) - 1 for s in stops), default=0)
        ended_on_eos = False
        for iteration in range(max_tokens):
            if log_speed:
                sample_start = time.perf_counter()
            token_id = self.sampler(
                logits, temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty, recent_counts=counts)
            if log_speed:
                self.sample_times.append(time.perf_counter() - sample_start)
            if token_id in self.model.stop_ids:
                if include_stop_token:
                    self.generated_ids.append(token_id)
                self.reason = "stop"
                ended_on_eos = True
                break
            self.generated_ids.append(token_id)
            if repetition_window:
                if len(recent) >= repetition_window:
                    old = recent.popleft()
                    counts[old] -= 1
                    if not counts[old]:
                        del counts[old]
                recent.append(token_id)
                counts[token_id] += 1
            if emit_text:
                decoded = self.model.decode(self.generated_ids)
                pos = min((decoded.index(s) for s in stops if s in decoded), default=-1)
                if pos >= 0:
                    decoded = decoded[:pos]
                    self.reason = "stop"
                visible = decoded.rstrip("\ufffd")
                if self.reason != "stop" and stop_buffer:
                    visible = visible[:-stop_buffer] if len(visible) > stop_buffer else ""
                if not visible.startswith(emitted):
                    raise RuntimeError("Tokenizer decoded output is not prefix-stable")
                delta = visible[len(emitted):]
                if delta:
                    emitted = visible
                    yield {"kind": "content", "text": delta}
            if self.reason == "stop":
                break
            if iteration + 1 < max_tokens:
                if log_speed:
                    synchronize()
                    forward_start = time.perf_counter()
                logits = self.model.decode_one(token_id, self.state)
                if log_speed:
                    synchronize()
                    self.forward_times.append(time.perf_counter() - forward_start)
        if emit_text and (self.reason == "length" or ended_on_eos):
            final = decoded.rstrip("\ufffd")
            if final.startswith(emitted) and final != emitted:
                yield {"kind": "content", "text": final[len(emitted):]}
        yield {"kind": "done", "reason": self.reason,
               "count": len(self.generated_ids)}

    def execution_info(self):
        """Observed execution for this request, if the adapter exposes it."""
        if self.state is None:
            return None
        reporter = getattr(self.model, "execution_report", None)
        return reporter(self.state) if reporter is not None else getattr(self.state, "execution_report", None)

    def generate(self, prompt, **kwargs):
        # One-shot generation avoids O(tokens²) repeated BPE decoding unless
        # stop strings require inspecting partially decoded text each step.
        kwargs.setdefault("emit_text", not not kwargs.get("stop"))
        parts = []
        for event in self.stream(prompt, **kwargs):
            if event["kind"] == "content":
                parts.append(event["text"])
        include_stop = kwargs.get("include_stop_token", False)
        text = ("".join(parts) if kwargs["emit_text"] and not include_stop
                else self.model.decode(self.generated_ids))
        return GenerationOutput(text=text, token_ids=tuple(self.generated_ids),
                                reason=self.reason, prompt_tokens=self.prompt_tokens,
                                completion_tokens=len(self.generated_ids))


class NativeGeneratorAdapter:
    """Adapt one legacy TextGenerator to the shared session protocol.

    Retains its InferenceModel and captured graph across consecutive requests.
    """

    def __init__(self, generator):
        self.generator = generator
        self.context_length = generator.max_context
        self.stop_ids = {generator._get_eos_id()}

    def encode(self, prompt):
        return list(self.generator.tokenizer.encode(prompt))

    def decode(self, ids):
        return self.generator.tokenizer.decode(ids)

    def create_state(self, total_tokens):
        self.generator._ensure_inference_model(reset=True)
        return self.generator._state

    def prefill(self, ids, state):
        return self.generator._inference_model.prefill(
            xp.asarray([ids], dtype=xp.int32), state)

    def decode_one(self, token_id, state):
        return self.generator._inference_model.decode_one(
            xp.asarray([[token_id]], dtype=xp.int32), state)

    def execution_report(self, state):
        from .inference_policy import observe_execution
        return observe_execution(self.generator._inference_model, state)
