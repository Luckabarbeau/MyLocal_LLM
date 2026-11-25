"""Opt-in WP8 text adapter. No server registration, model math or Python sampling.

One session exclusively borrows its decoder and sampler for its entire lifetime.
Callers must close abandoned iterators (e.g. contextlib.closing on disconnect).
"""
import json
import threading
import time

from mini_llm.external.gguf_tokenizer import GGUFTokenizer, load_gguf_tokenizer
from mini_llm.external.qwen4exp_chat import render_chat
from . import NativeError, _integer
from .generation import NativeGeneration


class IncrementalDetokenizer:
    """Use the tokenizer's streaming decoder, with a once-only exact final flush."""
    def __init__(self, tokenizer):
        from tokenizers.decoders import DecodeStream
        if not isinstance(tokenizer, GGUFTokenizer):
            raise TypeError('native chat requires the existing GGUFTokenizer')
        # The helper builds ByteLevel only; reject later decoder replacement rather
        # than assume its output has an append-only streaming representation.
        decoder = json.loads(tokenizer._tokenizer.to_str()).get('decoder')
        if not decoder or decoder.get('type') != 'ByteLevel':
            raise ValueError('unsupported streaming tokenizer decoder')
        self.tokenizer = tokenizer
        self.stream = DecodeStream(skip_special_tokens=False)
        self.tokens, self.pieces = [], []
        self.finished = False

    def push(self, token):
        if self.finished:
            raise RuntimeError('detokenizer already finished')
        self.tokens.append(token)
        piece = self.stream.step(self.tokenizer._tokenizer, token) or ''
        self.pieces.append(piece)
        return piece

    def finish(self):
        if self.finished:
            return ''
        self.finished = True
        # DecodeStream has no EOF flush. Decode once, not once per emitted token,
        # preserving incomplete UTF-8 replacements and literal U+FFFD at EOF.
        final = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        emitted = ''.join(self.pieces)
        if not final.startswith(emitted):
            raise RuntimeError('Tokenizer revised already streamed text')
        return final[len(emitted):]


class NativeChatSession:
    """Serial warm native owner; reuse is authorized only by actual token equality.

    committed_tokens excludes pending_token (the final sampled, unconsumed ID,
    including EOS). One native prompt-boundary checkpoint caches the exact last
    prompt. Other shorter/mismatched prompts reset/replay; continuations reuse
    only actual committed tokens.
    Cancellation/errors mark require_reset: unread/discarded IDs are never guessed.
    Ownership of native allocations remains with the caller, not this adapter.
    """
    def __init__(self, decoder, sampler, tokenizer, *, eos_ids=None, queue_capacity=1):
        _integer(queue_capacity, 'queue_capacity', 4096)
        if not queue_capacity:
            raise ValueError('queue_capacity must be positive')
        IncrementalDetokenizer(tokenizer)  # Fail before any native mutation.
        self.decoder, self.sampler, self.tokenizer = decoder, sampler, tokenizer
        if len(tokenizer.spec.tokens) != decoder.vocab or sampler.vocab != decoder.vocab:
            raise ValueError('tokenizer/sampler/decoder vocabulary mismatch')
        if eos_ids is None:
            eos_ids = (() if tokenizer.spec.eos_id is None else (tokenizer.spec.eos_id,))
        self.eos_ids = tuple(eos_ids)
        for token in self.eos_ids:
            _integer(token, 'eos_id', decoder.vocab-1)
        self.queue_capacity = queue_capacity
        self._lock = threading.Lock()
        self._committed = []
        self._prompt_ids = None
        self._checkpoint = None
        self.pending_token = None
        self.require_reset = True
        self.last_stats = {}

    @classmethod
    def from_gguf(cls, decoder, sampler, reader, **kwargs):
        """Reuse existing offline GGUF metadata/tokenizer construction; no weights."""
        return cls(decoder, sampler, load_gguf_tokenizer(reader, vocab_size=decoder.vocab), **kwargs)

    @property
    def committed_tokens(self):
        return tuple(self._committed)

    def reset(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('native chat session already active')
        try:
            self._reset()
        finally:
            self._lock.release()

    def _reset(self):
        self.require_reset = True
        if self._checkpoint is not None:
            self._checkpoint.close()
            self._checkpoint = None
        self._prompt_ids = None
        self.decoder.reset()
        self.sampler.reset()
        self._committed = []
        self.pending_token = None
        self.require_reset = False

    def stream(self, messages, max_new_tokens, *, thinking=True, reasoning_effort=None,
               temperature=0., top_p=1., top_k=None):
        """Render the existing text-only GGUF chat template and stream raw text.

        History is explicit per request; no reasoning parsing or message rewriting.
        EOS IDs must be supplied by the caller when more than spec.eos_id is needed.
        """
        prompt = render_chat(self.tokenizer, messages, thinking=thinking,
                             reasoning_effort=reasoning_effort)
        return self.stream_tokens(self.tokenizer.encode(prompt).ids, max_new_tokens,
                                  temperature=temperature, top_p=top_p, top_k=top_k)

    def stream_tokens(self, prompt_tokens, max_new_tokens, *, temperature=0., top_p=1., top_k=None):
        """Yield text from a bounded native queue; always close on disconnect.

        max_new_tokens includes sampled EOS, but EOS is not emitted as text.
        Sampling arguments go directly to the native worker (no Python sampler).
        """
        started = time.perf_counter()
        ids = list(prompt_tokens)
        _integer(max_new_tokens, 'max_new_tokens', 2**64-1)
        if not ids or not max_new_tokens:
            raise ValueError('nonempty prompt and positive max_new_tokens required')
        for token in ids:
            _integer(token, 'prompt token', self.decoder.vocab-1)
        if len(ids)+max_new_tokens-1 > self.decoder.capacity:
            raise ValueError('request exceeds native context capacity')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('native chat session already active')
        worker = None
        complete = False
        cleaned = False
        stats = dict(prompt_tokens=len(ids), reused_tokens=0, new_prompt_tokens=len(ids),
                     prefill_seconds=0., decode_seconds=0., output_queue_wait_seconds=0.,
                     ttft_seconds=None, emitted_pieces=0, sampled_tokens=0,
                     termination=None, reset_reason=None, checkpoint_restore=False,
                     checkpoint_restore_seconds=0., checkpoint_host_bytes=0)
        self.last_stats = stats
        try:
            position, failed = self.decoder.state
            prefix = ids[:len(self._committed)] == self._committed
            reason = ('require_reset' if self.require_reset else
                      'owner_state_changed' if failed or position != len(self._committed) else
                      'token_prefix_mismatch' if not prefix else
                      'no_prompt_tail' if len(ids) == len(self._committed) else None)
            cached = (not self.require_reset and not failed and
                      position == len(self._committed) and self._checkpoint is not None and
                      tuple(ids) == self._prompt_ids)
            if cached:
                start = time.perf_counter()
                restored = self._checkpoint.restore(self.decoder)
                stats['checkpoint_restore_seconds'] = time.perf_counter()-start
                if restored != len(ids)-1:
                    raise RuntimeError('native checkpoint boundary mismatch')
                self._committed = ids[:-1]
                self.pending_token = None
                reason = 'prompt_checkpoint_restore'
                stats['checkpoint_restore'] = True
            elif reason:
                self._reset()
            stats['reset_reason'] = reason
            reused = len(self._committed)
            stats.update(reused_tokens=reused, new_prompt_tokens=len(ids)-reused)
            tail = ids[reused:]
            detokenizer = IncrementalDetokenizer(self.tokenizer)
            if not cached:
                start = time.perf_counter()
                position = self.decoder.consume(tail[:-1])
                stats['prefill_seconds'] = time.perf_counter()-start
                if position != len(ids)-1:
                    raise RuntimeError('native prefill committed length mismatch')
                if self._checkpoint is not None:
                    self._checkpoint.close()
                    self._checkpoint = None
                self._prompt_ids = None
                self._checkpoint = self.decoder.checkpoint()
                self._prompt_ids = tuple(ids)
            stats['checkpoint_host_bytes'] = self._checkpoint.host_bytes
            # The worker consumes the final prompt token before sampling token 1.
            # Never consume the complete prompt here and send its last ID again.
            worker = NativeGeneration.start(self.decoder, self.sampler, tail[-1], max_new_tokens,
                eos_ids=self.eos_ids, temperature=temperature, top_p=top_p, top_k=top_k,
                queue_capacity=self.queue_capacity)
            sampled = []
            while True:
                start = time.perf_counter()
                event = worker.poll(100)
                stats['output_queue_wait_seconds'] += time.perf_counter()-start
                if event is None:
                    continue
                stats['decode_seconds'] = event.decode_seconds
                if event.kind == 2:
                    stats['termination'] = event.termination
                    if event.termination == 'error':
                        raise NativeError(event.status, event.error.decode('utf-8', 'replace'))
                    if event.termination not in ('max_tokens', 'eos', 'capacity') or event.failed:
                        raise RuntimeError('native generation interrupted: '+event.termination)
                    if (not sampled or event.sampled != len(sampled) or not event.pending or
                            event.pending_token != sampled[-1] or
                            event.position != len(ids)+len(sampled)-1):
                        raise RuntimeError('native generation committed/pending metadata mismatch')
                    piece = detokenizer.finish()
                    if piece:
                        stats['emitted_pieces'] += 1
                        if stats['ttft_seconds'] is None:
                            stats['ttft_seconds'] = time.perf_counter()-started
                        yield piece
                    self._committed = ids + sampled[:-1]
                    self.pending_token = sampled[-1]
                    complete = True
                    break
                if event.kind != 1:
                    raise RuntimeError('unknown native generation event')
                sampled.append(event.sample.token)
                stats['sampled_tokens'] = len(sampled)
                if event.sample.token in self.eos_ids:
                    continue
                piece = detokenizer.push(event.sample.token)
                if piece:
                    stats['emitted_pieces'] += 1
                    if stats['ttft_seconds'] is None:
                        stats['ttft_seconds'] = time.perf_counter()-started
                    yield piece
        finally:
            try:
                if worker is not None:
                    # close joins/drains in-flight work and discards unread queue
                    # events. Unknown discarded IDs invalidate token-prefix reuse.
                    try:
                        worker.cancel()
                    finally:
                        final = worker.close()
                        stats['decode_seconds'] = final.decode_seconds
                        stats['termination'] = final.termination
                cleaned = True
            finally:
                if not complete or not cleaned:
                    self.require_reset = True
                    self.pending_token = None
                stats['require_reset'] = self.require_reset
                stats['request_seconds'] = time.perf_counter()-started
                self._lock.release()
