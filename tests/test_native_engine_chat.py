"""GPU1 tiny-fixture WP8 integration; NOT whole-checkpoint trajectory parity."""
from contextlib import ExitStack
import ctypes as C

import pytest

from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
from mini_llm.native import NativeError
from mini_llm.native.chat import NativeChatSession
from test_native_engine_decoder import small_setup, forbidden
from test_native_engine_moe import libraries, selected_library

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def retain_ggml_library(selected_library):
    library = C.CDLL(selected_library)
    yield library


def tokenizer():
    return load_gguf_tokenizer({
        'tokenizer.ggml.model':'gpt2', 'tokenizer.ggml.pre':'qwen35',
        'tokenizer.ggml.tokens':['a','b','ab','c','d','e','<eos>'],
        'tokenizer.ggml.merges':['a b'],
        'tokenizer.ggml.token_type':[1]*6+[3],
        'tokenizer.ggml.eos_token_id':6,
        'tokenizer.chat_template': '{{ messages[0].content }}',
    })


def test_native_consume_replay_and_atomic_validation(tmp_path, libraries, monkeypatch):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack, tmp_path, libraries)
        sampler = stack.enter_context(e.sampler(7, seed=1))
        for token in (1,2,3):
            d.step_device(token)
        expected = sampler.sample_decoder(d)
        d.reset(); sampler.reset()
        with monkeypatch.context() as patch:
            patch.setattr(d, 'step', forbidden)
            patch.setattr(d, 'step_device', forbidden)
            assert d.consume([1,2,3]) == 3
            assert d.consume([]) == 3
        assert sampler.sample_decoder(d) == expected
        assert d.state == (3, False)
        with pytest.raises(NativeError, match='capacity'):
            d.consume([1,2])
        assert d.state == (3, False)
        # Exercise native validation independently of Python's integer guard.
        committed = C.c_uint64()
        status = d._runtime._lib.ml_decoder_consume(d._handle, (C.c_int32*2)(1,7), 2, C.byref(committed))
        assert status == 2 and committed.value == 3  # Capacity prevalidation wins.
        d.reset()
        status = d._runtime._lib.ml_decoder_consume(d._handle, (C.c_int32*2)(1,7), 2, C.byref(committed))
        assert status == 1 and committed.value == 0 and d.state == (0, False)
        assert d.consume([1]) == 1


def test_native_chat_queue_pending_repeated_and_continuation(tmp_path, libraries, monkeypatch):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack, tmp_path, libraries)
        sampler = stack.enter_context(e.sampler(7, seed=5))
        expected = []
        token = 1
        for _ in range(2):
            d.step_device(token)
            token = sampler.sample_decoder(d)
            expected.append(token)
        expected_next = d.step(token).tobytes()
        d.reset(); sampler.reset()
        chat = NativeChatSession(d, sampler, tokenizer(), eos_ids=(), queue_capacity=1)
        with monkeypatch.context() as patch:
            patch.setattr(d, 'step', forbidden)
            patch.setattr(d, 'step_device', forbidden)
            patch.setattr(sampler, 'sample_decoder', forbidden)
            text = ''.join(chat.stream_tokens([1], 2))
        assert text == chat.tokenizer.decode(expected)
        assert chat.committed_tokens == (1, expected[0])
        assert chat.pending_token == expected[-1] and d.state == (2, False)
        assert d.step(chat.pending_token).tobytes() == expected_next
        # Deliberate external owner mutation must not silently reuse token history.
        list(chat.stream_tokens([1], 2))
        assert chat.last_stats['reset_reason'] == 'owner_state_changed'
        prefix = list(chat.committed_tokens)+[chat.pending_token]
        list(chat.stream_tokens(prefix, 2))
        assert chat.last_stats['reused_tokens'] == 2 and d.state == (4, False)
        list(chat.stream_tokens([1], 2))
        assert chat.last_stats['reset_reason'] == 'token_prefix_mismatch'
        assert chat.committed_tokens == (1, expected[0])
        assert chat.last_stats['decode_seconds'] > 0
        assert chat.last_stats['output_queue_wait_seconds'] >= 0
        assert e.stats.device_bytes < 512*2**20


def test_native_chat_disconnect_and_eos(tmp_path, libraries):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack, tmp_path, libraries)
        sampler = stack.enter_context(e.sampler(7, seed=2))
        chat = NativeChatSession(d, sampler, tokenizer(), eos_ids=(), queue_capacity=1)
        stream = chat.stream_tokens([1], 4)
        assert next(stream)
        stream.close()  # May discard a queued/in-flight token: never resume unknown IDs.
        assert chat.require_reset and chat.pending_token is None
        list(chat.stream_tokens([1], 1))
        assert chat.last_stats['reset_reason'] == 'require_reset'
        assert d.state == (1, False) and not chat.require_reset
        chat.reset()
        assert d.state == (0, False) and chat.committed_tokens == ()
        eos_chat = NativeChatSession(d, sampler, tokenizer(), eos_ids=range(7))
        assert list(eos_chat.stream_tokens([1,2], 3)) == []
        assert eos_chat.committed_tokens == (1,2) and eos_chat.pending_token in range(7)
        assert eos_chat.last_stats['termination'] == 'eos'


def test_native_prefill_and_worker_error_reset(tmp_path, libraries):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack, tmp_path, libraries, bad_last=True)
        sampler = stack.enter_context(e.sampler(7, seed=2))
        chat = NativeChatSession(d, sampler, tokenizer())
        # One-token prompt fails inside worker; two-token prompt fails in consume.
        for prompt in ([1], [1,2]):
            with pytest.raises(NativeError, match='MoE'):
                list(chat.stream_tokens(prompt, 1))
            assert chat.require_reset and d.state == (0, True)
            chat.reset()
            assert d.state == (0, False)
