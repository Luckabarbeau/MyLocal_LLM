"""Text-only terminal-session helpers; no HTTP framework or model mathematics.

Render the actual GGUF Jinja template. Prefer the environment's Jinja2; Ubuntu's
system Python is an isolated offline fallback when the inference venv lacks it.
Never add system site-packages to the inference interpreter's search path.
"""
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def _render(template, values):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    environment.globals['raise_exception'] = fail
    environment.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    return environment.from_string(template).render(**values)


def render_chat(tokenizer, messages, *, thinking=True, reasoning_effort=None):
    if not tokenizer.chat_template:
        raise ValueError('Checkpoint has no chat template; refusing to guess framing')
    if any(m.get('role') not in ('system', 'user', 'assistant') or
           not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Terminal chat supports text system/user/assistant messages only')
    values = dict(messages=messages, tools=None, add_generation_prompt=True,
                  enable_thinking=thinking, preserve_thinking=True, add_vision_id=False)
    if reasoning_effort is not None:
        values['reasoning_effort'] = reasoning_effort
    if importlib.util.find_spec('jinja2') is not None:
        return _render(tokenizer.chat_template, values)
    executable = Path('/usr/bin/python3')
    if not executable.exists():
        raise RuntimeError('Chat needs Jinja2: install the external optional dependencies')
    # Only this small text renderer runs under system Python: no model, CUDA or
    # checkpoint payload is loaded in the helper process.
    result = subprocess.run([str(executable), str(Path(__file__).resolve())],
        input=json.dumps(dict(template=tokenizer.chat_template, values=values)),
        capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('Chat-template rendering failed; install Jinja2 in the venv if needed:\n'
                           + result.stderr[-1800:])
    return result.stdout


class ChatSession:
    """One serial conversation; keep the model and expert cache warm across turns."""
    def __init__(self, model, tokenizer, context, *, system='', thinking=True, reasoning_effort=None):
        self.model, self.tokenizer, self.context = model, tokenizer, context
        self.system, self.thinking, self.reasoning_effort = system, thinking, reasoning_effort
        self.reset()

    def reset(self):
        self.state = None
        gc.collect()  # Release the old aggregate state reservation before replacing it.
        self.consumed = []
        self.messages = [dict(role='system', content=self.system)] if self.system else []
        self.last_stats = {}

    def generate(self, text, max_new_tokens, *, sampler=None):
        """Yield streamed text pieces. ``sampler(logits_row)->int`` overrides greedy argmax.

        Passing None (the terminal default) keeps the argmax path bit-identical.
        """
        from mini_llm.backend import xp, asnumpy, synchronize
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Please enter a nonempty text prompt')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        history = self.messages + [dict(role='user', content=text)]
        prompt = render_chat(self.tokenizer, history, thinking=self.thinking,
                             reasoning_effort=self.reasoning_effort)
        ids = self.tokenizer.encode(prompt).ids
        if len(ids) + max_new_tokens - 1 > self.context:
            raise ValueError(f'Context needs up to {len(ids)+max_new_tokens-1} tokens, capacity '
                             f'{self.context}; use /reset or reduce --max-new-tokens')
        reused = len(self.consumed)
        rebuilt = self.state is not None and ids[:reused] != self.consumed
        if rebuilt:
            # Templates may trim whitespace or canonicalize thinking on old turns.
            # Recompute rather than reuse state from a different token prefix.
            self.state = None
            self.consumed = []
            gc.collect()
            reused = 0
        if self.state is None:
            self.state = self.model.create_generation_state(1, self.context)
        tail = ids[reused:]
        if not tail:
            raise ValueError('Chat template did not append a new prompt')
        start = time.perf_counter()
        logits = self.model.prefill(tail, self.state)
        synchronize()
        prefill_seconds = time.perf_counter()-start
        self.consumed.extend(tail)
        generated, emitted, forwards = [], '', 0
        start = time.perf_counter()
        stop = False
        for index in range(max_new_tokens):
            token = int(sampler(logits[0])) if sampler is not None \
                else int(asnumpy(xp.argmax(logits[0])))
            if token in self.model.stop_token_ids:
                stop = True
                break
            generated.append(token)
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
            stable = decoded.rstrip('\ufffd')
            if not stable.startswith(emitted):
                raise RuntimeError('Tokenizer revised already streamed text')
            piece = stable[len(emitted):]
            emitted = stable
            if piece:
                yield piece
            if index + 1 < max_new_tokens:
                logits = self.model.step(token, self.state)
                self.consumed.append(token)
                forwards += 1
        final = self.tokenizer.decode(generated, skip_special_tokens=False)
        if not final.startswith(emitted):
            raise RuntimeError('Tokenizer revised already streamed text')
        if final[len(emitted):]:
            yield final[len(emitted):]
        synchronize()
        decode_seconds = time.perf_counter()-start
        reasoning, answer = '', final
        if '</think>' in final:
            reasoning, answer = final.split('</think>', 1)
            reasoning = reasoning.removeprefix('<think>').strip()
        elif self.thinking:
            reasoning, answer = final.strip(), ''  # Truncated reasoning is not a final answer.
        self.messages = history + [dict(role='assistant', content=answer.strip(), reasoning_content=reasoning)]
        self.last_stats = dict(prompt_tokens=len(ids), new_prompt_tokens=len(tail), reused_tokens=reused,
            rebuilt_prefix=rebuilt, state_tokens=len(self.consumed), generated_tokens=len(generated),
            stopped=stop, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
            decode_forward_tokens=forwards,
            decode_forwards_per_second=forwards/max(decode_seconds, 1e-30))


if __name__ == '__main__':
    payload = json.load(sys.stdin)
    sys.stdout.write(_render(payload['template'], payload['values']))
