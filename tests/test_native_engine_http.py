"""Finite GPU1 tiny native HTTP lifecycle; no actual checkpoint or parity claim."""
from contextlib import ExitStack
import ctypes as C
import json

import pytest

from mini_llm.native.chat import NativeChatSession
from scripts import serve_qwen4exp as http
from test_native_engine_decoder import small_setup
from test_native_engine_moe import libraries, selected_library
from test_native_engine_chat import tokenizer
from test_native_http import args, body, events, native_service, request, running

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def retain_ggml_library(selected_library):
    library = C.CDLL(selected_library)
    yield library


def test_tiny_native_http_eos_disconnect_and_recovery(tmp_path, libraries):
    with ExitStack() as stack:
        decoder, (_, engine, _), _ = small_setup(stack, tmp_path, libraries)
        sampler = stack.enter_context(engine.sampler(7, seed=2))
        chat = NativeChatSession(decoder, sampler, tokenizer(), eos_ids=range(7))
        service = native_service(chat)
        handler = http.make_handler(service, args('--context', '4'))
        write = handler._write
        disconnect = [False]

        def broken_write(self, data):
            if disconnect[0] and b'"delta": {"content":' in data:
                disconnect[0] = False
                return False
            return write(self, data)

        handler._write = broken_write
        payload = body()
        payload['messages'][0]['content'] = 'a'
        with running(service, handler) as port:
            for stream in (False, True):
                status, text = request(port, dict(payload, stream=stream))
                assert status == 200
                if stream:
                    chunks = events(text)
                    assert chunks[-2]['choices'][0]['finish_reason'] == 'stop'
                    usage = chunks[-1]['usage']
                    assert text.endswith('data: [DONE]\n\n')
                else:
                    result = json.loads(text)
                    assert result['choices'][0]['finish_reason'] == 'stop'
                    assert result['choices'][0]['message']['content'] == ''
                    usage = result['usage']
                assert usage['completion_tokens'] == 1
                assert chat.last_stats['termination'] == 'eos'
                assert not chat.require_reset

            chat.eos_ids = ()
            disconnect[0] = True
            status, text = request(port, dict(payload, stream=True, max_tokens=4))
            assert status == 200 and '[DONE]' not in text
            # The shared lock makes this an observation after cancellation/drain.
            metrics = json.loads(request(port, path='/metrics')[1])
            assert metrics['mylocal_stats']['require_reset']
            assert chat.require_reset and chat.pending_token is None
            status, text = request(port, payload)
            assert status == 200
            assert json.loads(text)['usage']['completion_tokens'] == 2
            assert chat.last_stats['reset_reason'] == 'require_reset'
            assert not chat.require_reset
            assert engine.stats.device_bytes <= 512 * 2**20
