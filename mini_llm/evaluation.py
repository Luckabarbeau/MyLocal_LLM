"""0103G: portable evaluation boundary over the 0103E generation session.

Model execution stays in serving_models/generation/remote_provider; evaluation
owns neither kernels nor sampling. Do not synthesize token usage for remotes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any


@dataclass(frozen=True)
class EvaluationSample:
    text: str
    token_ids: tuple[int, ...] | None
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str
    finish_reason: str | None
    wall_s: float
    prefill_s: float | None
    decode_s: float | None
    execution: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def _report_dict(report: Any) -> dict | None:
    if report is None:
        return None
    if hasattr(report, '__dataclass_fields__'):
        return asdict(report)
    if isinstance(report, dict):
        return dict(report)
    raise TypeError('Unexpected inference execution report type')


class EvaluationEngine:
    """Small adapter for local token sessions or remote completion providers.

    For local models, time excludes model loading, includes tokenization and
    sampling, and synchronizes the GPU before/after the request. Prefill time
    is an independently synchronized measurement from GenerationSession.
    For remote models only end-to-end HTTP latency is observable; any provider
    token usage is labeled as provider-reported, not verified locally.
    """

    def __init__(self, model: Any, *, model_id: str, strict_fast: bool = False,
                 remote_mode: str = "completion"):
        self.model = model
        self.model_id = model_id
        self.is_remote = bool(getattr(model, 'is_remote', False))
        self.strict_fast = strict_fast
        if remote_mode not in {"chat", "completion"}:
            raise ValueError("remote_mode must be chat or completion")
        self.remote_mode = remote_mode
        if self.is_remote and strict_fast:
            raise ValueError('strict-fast requires locally observed CUDA execution; remote paths are unverified')

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int | None = None, seed: int | None = None) -> EvaluationSample:
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        if self.is_remote:
            if top_k is not None:
                raise ValueError('Remote OpenAI completion API does not guarantee top_k support')
            request = {
                'model': self.model_id,
                'max_tokens': max_new_tokens, 'temperature': temperature,
                'top_p': top_p, 'stream': False,
            }
            if self.remote_mode == 'chat':
                request['messages'] = [{'role': 'user', 'content': prompt}]
            else:
                request['prompt'] = prompt
            if seed is not None:
                request['seed'] = seed  # upstream may ignore; not claimed deterministic
            start = time.perf_counter()
            response = self.model.complete(request, chat=self.remote_mode == "chat",
                                           public_id=self.model_id)
            duration = time.perf_counter() - start
            choices = response.get('choices') or []
            if not choices:
                raise ValueError('Upstream model returned no completion choices')
            first = choices[0]
            if self.remote_mode == 'chat':
                message = first.get('message') or {}
                if message.get('tool_calls'):
                    raise ValueError('Benchmark expected plain text, but remote model requested tools')
                content = message.get('content')
            else:
                content = first.get('text')
            if not isinstance(content, str):
                raise ValueError('Upstream completion did not return text')
            usage = response.get('usage') or {}
            prompt_n = usage.get('prompt_tokens')
            completion_n = usage.get('completion_tokens')
            for value in (prompt_n, completion_n):
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError('Invalid upstream token count')
            return EvaluationSample(content, None, prompt_n, completion_n,
                                    'provider_reported' if completion_n is not None else 'unknown',
                                    first.get('finish_reason'), duration, None, None,
                                    {'backend': 'remote', 'verified_fast': False,
                                     'latency_includes_transport': True})

        from .generation import GenerationSession
        from .backend import synchronize
        session = GenerationSession(self.model)
        synchronize()
        start = time.perf_counter()
        outcome = session.generate(prompt, max_tokens=max_new_tokens, temperature=temperature,
                                   top_p=top_p, top_k=top_k, seed=seed, log_speed=True)
        synchronize()
        elapsed = time.perf_counter() - start
        report = session.execution_info()
        if self.strict_fast:
            if report is None:
                raise RuntimeError('strict-fast: inference did not expose an execution report')
            report.require_fast()
        prefill = session.prefill_time_s
        return EvaluationSample(outcome.text, tuple(outcome.token_ids),
                                outcome.prompt_tokens, outcome.completion_tokens,
                                'local_tokenizer', outcome.reason, elapsed, prefill,
                                max(elapsed - prefill, 0.0), _report_dict(report))
