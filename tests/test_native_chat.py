"""CPU-only tokenizer and session-contract tests; no model/backend import."""
from types import SimpleNamespace

import pytest

from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
from mini_llm.native.chat import IncrementalDetokenizer, NativeChatSession
from mini_llm.native import NativeError


def tokenizer():
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['ab', '<eos>', '<custom>']
    types = [1]*257 + [3, 4]
    return load_gguf_tokenizer({
        'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
        'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.merges': ['a b'],
        'tokenizer.ggml.token_type': types, 'tokenizer.ggml.eos_token_id': 257,
        'tokenizer.chat_template': "{% for m in messages %}{{ m.content }}{% endfor %}a",
    })


@pytest.mark.parametrize('text', ['ab é🙂中', 'a\ufffdb', '<custom><eos>', ' \n\r\t'])
def test_incremental_exact(text):
    t = tokenizer()
    # Force byte boundaries, including UTF-8 characters split into several IDs.
    ids = t.encode(text).ids
    stream = IncrementalDetokenizer(t)
    assert ''.join(stream.push(i) for i in ids)+stream.finish() == t.decode(ids)
    assert stream.finish() == ''
    with pytest.raises(RuntimeError):
        stream.push(ids[0])


def test_incomplete_utf8_and_one_final_decode(monkeypatch):
    t = tokenizer()
    ids = t.encode('🙂').ids
    assert len(ids) == 4
    calls = []
    decode = t.decode
    monkeypatch.setattr(t, 'decode', lambda ids, **kw: (calls.append(tuple(ids)), decode(ids, **kw))[1])
    for suffix in (ids, ids[:1], ids[:2], ids[:3]):
        stream = IncrementalDetokenizer(t)
        pieces = [stream.push(i) for i in suffix]
        assert ''.join(pieces)+stream.finish() == decode(suffix)
    assert len(calls) == 4  # One batch decode per completed output, not per token.


class FakeDecoder:
    capacity = 64
    def __init__(self, vocab):
        self.vocab, self.tokens, self.resets, self.consume_calls = vocab, [], 0, []
    @property
    def state(self):
        return len(self.tokens), False
    def reset(self):
        self.tokens.clear()
        self.resets += 1
    def checkpoint(self):
        decoder = self
        class Checkpoint:
            host_bytes = 64
            def __init__(self):
                self.tokens = list(decoder.tokens)
                self.closed = False
            def close(self):
                self.closed = True
            def restore(self, owner):
                assert owner is decoder and not self.closed
                decoder.tokens[:] = self.tokens
                return len(self.tokens)
        return Checkpoint()
    def consume(self, ids):
        self.consume_calls.append(list(ids))
        self.tokens.extend(ids)
        return len(self.tokens)


class FakeSampler:
    def __init__(self, vocab):
        self.vocab = vocab
    def reset(self):
        pass


class FakeWorker:
    def __init__(self, decoder, input_token, maximum, output, eos_ids, fail=False):
        self.decoder, self.input, self.maximum = decoder, input_token, maximum
        self.output, self.eos_ids, self.fail = output, eos_ids, fail
        self.sampled, self.pending, self.cancelled, self.closed = 0, None, False, False
    def event(self, kind, reason=None):
        return SimpleNamespace(kind=kind, termination=reason, decode_seconds=.01*self.sampled,
            position=len(self.decoder.tokens), sampled=self.sampled, pending=bool(self.sampled),
            pending_token=self.pending, failed=self.fail, status=1, error=b'fake failure',
            sample=SimpleNamespace(token=self.pending))
    def poll(self, timeout):
        if self.fail:
            return self.event(2, 'error')
        if self.sampled == self.maximum or self.pending in self.eos_ids:
            return self.event(2, 'eos' if self.pending in self.eos_ids else 'max_tokens')
        self.decoder.tokens.append(self.input)
        self.pending = self.output[self.sampled % len(self.output)]
        self.input = self.pending
        self.sampled += 1
        return self.event(1)
    def cancel(self):
        self.cancelled = True
    def close(self):
        self.closed = True
        return self.event(2, 'cancelled' if self.sampled < self.maximum else 'max_tokens')


@pytest.fixture
def fake(monkeypatch):
    t = tokenizer()
    d, s = FakeDecoder(len(t.spec.tokens)), FakeSampler(len(t.spec.tokens))
    output, workers, kwargs_seen = [t.token_to_id('a'), t.token_to_id('b')], [], []
    def start(decoder, sampler, token, maximum, **kwargs):
        kwargs_seen.append(kwargs)
        w = FakeWorker(decoder, token, maximum, output, kwargs['eos_ids'])
        workers.append(w)
        return w
    monkeypatch.setattr('mini_llm.native.chat.NativeGeneration.start', start)
    return NativeChatSession(d, s, t), d, output, workers, kwargs_seen


def test_repeated_continuation_prefix_and_sampling(fake):
    chat, d, output, workers, seen = fake
    prompt = [1, 2, 3]
    assert ''.join(chat.stream_tokens(prompt, 2, temperature=.7, top_p=.8, top_k=4)) == 'ab'
    assert chat.committed_tokens == tuple(prompt+[output[0]])
    assert chat.pending_token == output[1]
    assert d.tokens == list(chat.committed_tokens)
    assert d.consume_calls == [[1, 2]]
    assert seen[-1]['temperature'] == .7 and seen[-1]['top_k'] == 4
    continuation = list(chat.committed_tokens)+[chat.pending_token]
    assert ''.join(chat.stream_tokens(continuation, 1)) == 'a'
    assert chat.last_stats['reused_tokens'] == 4
    assert d.tokens == continuation  # Pending consumed once, not twice.
    assert d.resets == 1
    list(chat.stream_tokens(prompt, 1))
    assert d.resets == 2 and chat.last_stats['reset_reason'] == 'token_prefix_mismatch'
    calls = len(d.consume_calls)
    list(chat.stream_tokens(prompt, 1))  # Restore the boundary before the last input.
    assert d.resets == 2 and chat.last_stats['reset_reason'] == 'prompt_checkpoint_restore'
    assert len(d.consume_calls) == calls and chat.last_stats['checkpoint_restore']
    assert chat.last_stats['prefill_seconds'] == 0
    assert all(w.cancelled and w.closed for w in workers)
    assert chat.last_stats['decode_seconds'] == .01
    assert chat.last_stats['ttft_seconds'] is not None


@pytest.mark.parametrize('prompt', [[1], [1,2,3,4]])
def test_repeated_request_uses_only_cached_boundary(fake, monkeypatch, prompt):
    chat, d, _, _, _ = fake
    expected = ''.join(chat.stream_tokens(prompt, 3))
    committed, pending = chat.committed_tokens, chat.pending_token
    def forbidden(*args, **kwargs):
        raise AssertionError('cache hit must not reset or prefill')
    monkeypatch.setattr(d, 'consume', forbidden)
    monkeypatch.setattr(d, 'reset', forbidden)
    for _ in range(2):
        assert ''.join(chat.stream_tokens(prompt, 3)) == expected
        assert chat.committed_tokens == committed and chat.pending_token == pending
        assert chat.last_stats['checkpoint_restore']
        assert chat.last_stats['reused_tokens'] == len(prompt)-1


def test_cancel_disconnect_serialization_and_reset(fake):
    chat, d, output, workers, _ = fake
    stream = chat.stream_tokens([1, 2], 3)
    assert next(stream) == 'a'
    with pytest.raises(RuntimeError, match='active'):
        next(chat.stream_tokens([1], 1))
    with pytest.raises(RuntimeError, match='active'):
        chat.reset()
    stream.close()
    assert chat.require_reset and chat.pending_token is None
    assert workers[-1].cancelled and workers[-1].closed
    list(chat.stream_tokens([1, 2], 1))
    assert d.resets == 2 and not chat.require_reset
    chat.reset()
    assert not chat.committed_tokens and not chat.require_reset


def test_eos_error_and_external_state(fake, monkeypatch):
    chat, d, output, workers, _ = fake
    output[:] = [chat.eos_ids[0]]
    assert list(chat.stream_tokens([1], 3)) == []
    assert chat.committed_tokens == (1,) and chat.pending_token == output[0]
    assert chat.last_stats['ttft_seconds'] is None
    d.tokens.append(5)
    list(chat.stream_tokens([1, 2, 3], 1))
    assert chat.last_stats['reset_reason'] == 'owner_state_changed'
    start = FakeWorker.poll
    monkeypatch.setattr(FakeWorker, 'poll', lambda self, timeout: (setattr(self, 'fail', True), start(self, timeout))[1])
    with pytest.raises(NativeError, match='fake failure'):
        list(chat.stream_tokens([1], 1))
    assert chat.require_reset and workers[-1].closed


def test_cleanup_error_invalidates_successful_terminal(fake, monkeypatch):
    chat, _, _, workers, _ = fake
    def fail_close(self):
        self.closed = True
        raise RuntimeError('close failed')
    monkeypatch.setattr(FakeWorker, 'close', fail_close)
    with pytest.raises(RuntimeError, match='close failed'):
        list(chat.stream_tokens([1], 1))
    assert chat.require_reset and chat.pending_token is None
    assert workers[-1].cancelled and workers[-1].closed
    # Lock must be released even if native cleanup raises.
    chat.reset()


def test_render_validation_and_unsupported(fake):
    chat, d, *_ = fake
    assert ''.join(chat.stream([{'role':'user', 'content':'ab'}], 1)) == 'a'
    with pytest.raises(ValueError, match='text'):
        chat.stream([{'role':'user', 'content':[]}], 1)
    chat.tokenizer.chat_template = None
    with pytest.raises(ValueError, match='template'):
        chat.stream([{'role':'user', 'content':'a'}], 1)
    for ids, maximum in (([],1), ([1],0), ([999],1), ([1]*64,2)):
        with pytest.raises(ValueError):
            list(chat.stream_tokens(ids, maximum))
    from tokenizers.decoders import WordPiece
    chat.tokenizer._tokenizer.decoder = WordPiece()
    with pytest.raises(ValueError, match='unsupported'):
        IncrementalDetokenizer(chat.tokenizer)
    with pytest.raises(TypeError, match='GGUFTokenizer'):
        IncrementalDetokenizer(object())
