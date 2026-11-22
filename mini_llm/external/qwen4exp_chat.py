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
           not isinstance(m.get('content'), str) or
                      ('reasoning_content' in m and not isinstance(m['reasoning_content'], str))
                      for m in messages):
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
    def __init__(self, model, tokenizer, context, *, system='', thinking=True, reasoning_effort=None,
                     prefill_chunk_size=0):
        self.model, self.tokenizer, self.context = model, tokenizer, context
        self.system, self.thinking, self.reasoning_effort = system, thinking, reasoning_effort
        if isinstance(prefill_chunk_size, bool) or not isinstance(prefill_chunk_size, int) or prefill_chunk_size < 0:
            raise ValueError('prefill_chunk_size must be a nonnegative integer')
        self.prefill_chunk_size = prefill_chunk_size
        self.reset(reason='initial_state')

    def reset(self, *, reason='explicit_reset'):
        self._reset_reason = reason
        self._prompt_checkpoint = None
        self.state = None
        gc.collect()  # Release the old aggregate state reservation before replacing it.
        self.consumed = []
        self.messages = [dict(role='system', content=self.system)] if self.system else []
        self.last_stats = {}
        self._returned_messages = [dict(m) for m in self.messages]
        self._canonical_messages = [dict(m) for m in self.messages]

    def prepare(self, messages):
        """Install request history; only generate's exact token check authorizes reuse."""
        if (not messages or messages[-1].get('role') != 'user' or
                any(m.get('role') not in ('system', 'user', 'assistant') or
                    not isinstance(m.get('content'), str) or
                    ('reasoning_content' in m and not isinstance(m['reasoning_content'], str))
                    for m in messages)):
            raise ValueError('Expected text messages ending with a user message')
        self.system = next((m['content'] for m in messages if m['role'] == 'system'), '')
        incoming = [dict(m) for m in messages[:-1]]
        # Only reinterpret an exact message we returned, using the parsed form
        # retained from that generation. Unknown raw text is not guessed to be reasoning.
        self.messages = [dict(self._canonical_messages[i])
                         if i < len(self._returned_messages) and m == self._returned_messages[i]
                         else dict(m) for i, m in enumerate(incoming)]
        self._returned_messages = incoming

    def generate(self, text, max_new_tokens, *, sampler=None, raw_response=False):
        """Yield streamed text pieces. ``sampler(logits_row)->int`` overrides greedy argmax.

        Passing None (the terminal default) keeps the argmax path bit-identical.
        """
        from mini_llm.backend import xp, asnumpy, synchronize
        from .qwen4exp_prompt_checkpoint import PromptCheckpoint
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Please enter a nonempty text prompt')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        generation_start = time.perf_counter()
        first_piece_seconds = None
        history = self.messages + [dict(role='user', content=text)]
        prompt = render_chat(self.tokenizer, history, thinking=self.thinking,
                             reasoning_effort=self.reasoning_effort)
        ids = self.tokenizer.encode(prompt).ids
        returned_history = self._returned_messages + [dict(role='user', content=text)]
        if (self.state is not None and ids[:len(self.consumed)] != self.consumed
                and returned_history != history):
            # A template may already accept raw history without loss (e.g. whitespace).
            # Prefer it over a parsed representation only after exact token verification.
            returned_prompt = render_chat(self.tokenizer, returned_history, thinking=self.thinking,
                                          reasoning_effort=self.reasoning_effort)
            returned_ids = self.tokenizer.encode(returned_prompt).ids
            if returned_ids[:len(self.consumed)] == self.consumed:
                history, ids = returned_history, returned_ids
        if len(ids) + max_new_tokens - 1 > self.context:
            raise ValueError(f'Context needs up to {len(ids)+max_new_tokens-1} tokens, capacity '
                             f'{self.context}; use /reset or reduce --max-new-tokens')
        reused = len(self.consumed)
        prefix_match = self.state is not None and ids[:reused] == self.consumed
        checkpoint_restore = False
        restored_logits = None
        restore_start = time.perf_counter()
        if (self._prompt_checkpoint is not None and
                (not prefix_match or len(ids) == reused)):
            restored_logits = self._prompt_checkpoint.restore(self.state, ids, self.consumed)
            if restored_logits is not None:
                self.consumed = list(self._prompt_checkpoint.tokens)
                reused = len(self.consumed)
                prefix_match = True
                checkpoint_restore = True
        checkpoint_restore_seconds = time.perf_counter() - restore_start if checkpoint_restore else 0.0
        rebuilt = self.state is not None and not prefix_match
        reset_reason = ('prompt_checkpoint_restore' if checkpoint_restore else
                        'token_prefix_mismatch' if rebuilt else
                        self._reset_reason if self.state is None else None)
        prefix_mismatch = None
        if rebuilt:
            index = next((i for i, (old, new) in enumerate(zip(self.consumed, ids))
                          if old != new), min(len(self.consumed), len(ids)))
            prefix_mismatch = dict(index=index, consumed_tokens=len(self.consumed),
                                   prompt_tokens=len(ids),
                                   consumed_token_id=self.consumed[index]
                                   if index < len(self.consumed) else None,
                                   prompt_token_id=ids[index] if index < len(ids) else None)
            # Templates may trim whitespace or canonicalize thinking on old turns.
            # Recompute rather than reuse state from a different token prefix.
            self._prompt_checkpoint = None
            self.state = None
            self.consumed = []
            gc.collect()
            reused = 0
        if self.state is None:
            self.state = self.model.create_generation_state(1, self.context)
        tail = ids[reused:]
        if not tail and restored_logits is None:
            raise ValueError('Chat template did not append a new prompt')
        start = time.perf_counter()
        if tail:
            if self.prefill_chunk_size:
                logits = self.model.prefill(tail, self.state, chunk_size=self.prefill_chunk_size)
            else:
                logits = self.model.prefill(tail, self.state)
            synchronize()
            self.consumed.extend(tail)
            # Retain only one bounded rollback point, never a QSA history copy.
            self._prompt_checkpoint = None
            self._prompt_checkpoint = PromptCheckpoint.capture(self.state, self.consumed, logits)
        else:
            logits = restored_logits
        prefill_seconds = time.perf_counter()-start
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
                if first_piece_seconds is None:
                    first_piece_seconds = time.perf_counter() - generation_start
                yield piece
            if index + 1 < max_new_tokens:
                logits = self.model.step(token, self.state)
                self.consumed.append(token)
                forwards += 1
        final = self.tokenizer.decode(generated, skip_special_tokens=False)
        if not final.startswith(emitted):
            raise RuntimeError('Tokenizer revised already streamed text')
        if final[len(emitted):]:
            if first_piece_seconds is None:
                first_piece_seconds = time.perf_counter() - generation_start
            yield final[len(emitted):]
        synchronize()
        decode_seconds = time.perf_counter()-start
        reasoning, answer = '', final
        if '</think>' in final:
            reasoning, answer = final.split('</think>', 1)
            reasoning = reasoning.removeprefix('<think>').strip()
        elif self.thinking:
            reasoning, answer = final.strip(), ''  # Truncated reasoning is not a final answer.
        canonical = dict(role='assistant', content=answer.strip(), reasoning_content=reasoning)
        assistant = dict(role='assistant', content=final) if raw_response else canonical
        self._canonical_messages = history + [canonical]
        self._returned_messages = self._returned_messages + [dict(role='user', content=text), assistant]
        self.messages = history + [assistant]
        self.last_stats = dict(prompt_tokens=len(ids), new_prompt_tokens=len(tail), reused_tokens=reused,
            rebuilt_prefix=rebuilt, prefix_match=prefix_match, prefix_mismatch=prefix_mismatch,
            state_reset=rebuilt or (reset_reason is not None and not checkpoint_restore),
            reset_reason=reset_reason, checkpoint_restore=checkpoint_restore,
            checkpoint_restore_seconds=checkpoint_restore_seconds,
            checkpoint_host_array_bytes=(self._prompt_checkpoint.host_array_bytes
                                         if self._prompt_checkpoint is not None else 0),
                        state_tokens=len(self.consumed), generated_tokens=len(generated),
            stopped=stop, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
            first_piece_seconds=first_piece_seconds,
            decode_forward_tokens=forwards,
            decode_forwards_per_second=forwards/max(decode_seconds, 1e-30))


if __name__ == '__main__':
    payload = json.load(sys.stdin)
    sys.stdout.write(_render(payload['template'], payload['values']))
