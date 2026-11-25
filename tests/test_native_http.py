"""Finite CPU-only HTTP tests: real adapter/tokenizer, mocked native worker."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from scripts import serve_qwen4exp as http
from test_native_chat import fake, FakeWorker  # Reuse the adapter's CPU-only fixture.
from test_qwen4exp_serving_state import service as python_service


def args(*extra):
    return http.argument_parser().parse_args(['--port', '0', '--no-thinking', *extra])


def native_service(chat):
    return http.Service(chat, dict(backend='native', experimental=True, thinking=False,
        arithmetic_policy='reference_fp32 / ggml_q8',
        diagnostic_trajectory_differences=http.TRAJECTORY_DIAGNOSTICS))


@contextmanager
def running(service, handler=None):
    server = http.QuietServer(('127.0.0.1', 0), handler or http.make_handler(service, args()))
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()
        assert not any(t.is_alive() for t in server._threads)


def request(port, payload=None, path='/v1/chat/completions'):
    connection = HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request('GET' if payload is None else 'POST', path,
                           None if payload is None else json.dumps(payload),
                           {'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def test_native_repeated_http_prompt_checkpoint(fake):
    chat, decoder, output, workers, seen = fake
    service = native_service(chat)
    with running(service) as port:
        status, first = request(port, body())
        assert status == 200
        calls = len(decoder.consume_calls)
        resets = decoder.resets
        status, second = request(port, body())
        assert status == 200
        assert json.loads(first)['choices'][0]['message'] == json.loads(second)['choices'][0]['message']
        _, metrics = request(port, path='/metrics')
        stats = json.loads(metrics)['mylocal_stats']
        assert stats['checkpoint_restore'] is True
        assert stats['reset_reason'] == 'prompt_checkpoint_restore'
        assert stats['reused_tokens'] == stats['prompt_tokens'] - 1
        assert stats['new_prompt_tokens'] == 1 and stats['prefill_seconds'] == 0
        assert len(decoder.consume_calls) == calls and decoder.resets == resets
        assert stats['checkpoint_host_bytes'] > 0 and not stats['require_reset']


def test_native_banner_reports_native_policy(monkeypatch, capsys):
    from unittest.mock import Mock
    server = Mock()
    server.serve_forever.side_effect = KeyboardInterrupt
    monkeypatch.setattr(http, 'QuietServer', Mock(return_value=server))
    config = args('--backend', 'native', '--native-threads', '4',
                  '--native-budget-gib', '10', '--native-dense-budget-gib', '7')
    http.serve_http(Mock(native=True), config)
    banner = capsys.readouterr().err
    assert 'all routed experts CPU mmap, 4 CPU threads' in banner
    assert 'device budget 10.0GiB, dense 7.0GiB' in banner
    assert 'cache 6GiB' not in banner and '19 CPU workers' not in banner
    server.server_close.assert_called_once()
    assert not server.daemon_threads


def body(**kwargs):
    return dict(messages=[dict(role='user', content='Hi')], max_tokens=2, **kwargs)


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]']


@pytest.mark.parametrize('stream', [False, True])
def test_native_http_roundtrip(fake, stream):
    chat, decoder, output, workers, seen = fake
    service = native_service(chat)
    with running(service) as port:
        status, text = request(port, body(stream=stream, temperature=.7, top_p=.8, top_k=4))
        assert status == 200
        if stream:
            chunks = events(text)
            assert text.endswith('data: [DONE]\n\n')
            assert ''.join(c['choices'][0]['delta'].get('content', '')
                           for c in chunks if c['choices']) == 'ab'
            assert chunks[-2]['choices'][0]['finish_reason'] == 'length'
            usage = chunks[-1]['usage']
        else:
            result = json.loads(text)
            assert result['object'] == 'chat.completion'
            assert result['choices'][0]['message'] == dict(role='assistant', content='ab')
            assert result['choices'][0]['finish_reason'] == 'length'
            usage = result['usage']
        assert usage['completion_tokens'] == 2
        assert usage['total_tokens'] == usage['prompt_tokens'] + 2
        health = json.loads(request(port, path='/health')[1])
        metrics = json.loads(request(port, path='/metrics')[1])
        assert health['config']['backend'] == 'native'
        assert health['config']['experimental']
        assert 'strict' in health['config']['diagnostic_trajectory_differences']
        assert metrics['experts'] is None
        assert metrics['mylocal_stats']['arithmetic_policy'] == service.config['arithmetic_policy']
    assert seen[-1]['temperature'] == .7 and seen[-1]['top_p'] == .8 and seen[-1]['top_k'] == 4
    assert all(w.closed and w.cancelled for w in workers)


@pytest.mark.parametrize('stream', [False, True])
def test_eos_http(fake, monkeypatch, stream):
    chat, _, output, workers, seen = fake
    output[:] = [chat.eos_ids[0]]

    class TerminalWorker(FakeWorker):
        terminal = None

        def poll(self, timeout):
            event = super().poll(timeout)
            if event.kind == 2:
                self.terminal = event
            return event

        def close(self):
            final = super().close()
            # Native close joins and returns the immutable terminal snapshot;
            # cancellation after EOS does not replace its termination reason.
            return self.terminal if self.terminal is not None else final

    def start(decoder, sampler, token, maximum, **kwargs):
        seen.append(kwargs)
        worker = TerminalWorker(decoder, token, maximum, output, kwargs['eos_ids'])
        workers.append(worker)
        return worker

    monkeypatch.setattr('mini_llm.native.chat.NativeGeneration.start', start)
    with running(native_service(chat)) as port:
        status, text = request(port, body(stream=stream))
        assert status == 200
        if stream:
            chunks = events(text)
            assert chunks[-2]['choices'][0]['finish_reason'] == 'stop'
            assert not any(c['choices'][0]['delta'].get('content')
                           for c in chunks if c['choices'])
            assert text.endswith('data: [DONE]\n\n')
            usage = chunks[-1]['usage']
        else:
            result = json.loads(text)
            assert result['choices'][0]['finish_reason'] == 'stop'
            assert result['choices'][0]['message']['content'] == ''
            usage = result['usage']
    assert usage['completion_tokens'] == 1  # Native counts sampled EOS.
    assert chat.last_stats['termination'] == workers[-1].close().termination == 'eos'
    assert workers[-1].closed and workers[-1].cancelled
    assert not chat.require_reset


@pytest.mark.parametrize('failure', ['return_false', 'reset'])
def test_disconnect_closes_worker_before_next_request(fake, failure):
    chat, decoder, _, workers, _ = fake
    service = native_service(chat)
    handler = http.make_handler(service, args())
    write = handler._write
    disconnected = threading.Event()

    def disconnect(self, data):
        if b'"content": "a"' in data and not disconnected.is_set():
            disconnected.set()
            if failure == 'reset':
                raise ConnectionResetError('injected disconnect')
            return False
        return write(self, data)

    handler._write = disconnect
    with running(service, handler) as port:
        status, text = request(port, body(stream=True))
        assert status == 200 and '[DONE]' not in text
        # Metrics takes the same lock, so observes completed cancel/drain.
        request(port, path='/metrics')
        assert chat.require_reset and chat.pending_token is None
        assert workers[-1].closed and workers[-1].cancelled
        assert workers[-1].sampled == 1  # Never finish the old output loop.
        resets = decoder.resets
        assert json.loads(request(port, body())[1])['choices'][0]['message']['content'] == 'ab'
        assert decoder.resets == resets + 1
        assert not chat.require_reset


def test_requests_are_serialized(fake, monkeypatch):
    chat, _, _, workers, _ = fake
    entered, release = threading.Event(), threading.Event()
    poll = FakeWorker.poll

    def blocked(self, timeout):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return poll(self, timeout)

    monkeypatch.setattr(FakeWorker, 'poll', blocked)
    service = native_service(chat)
    handler = http.make_handler(service, args())
    second_entered = threading.Event()
    original = handler._do_POST

    def received(self):
        if entered.is_set():
            second_entered.set()
        return original(self)

    handler._do_POST = received
    with running(service, handler) as port, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, port, body())
        try:
            assert entered.wait(3)
            second = pool.submit(request, port, body())
            assert second_entered.wait(3)
            assert len(workers) == 1
        finally:
            release.set()
        assert first.result(5)[0] == second.result(5)[0] == 200
    assert len(workers) == 2 and all(w.closed for w in workers)


@pytest.mark.parametrize('stream', [False, True])
def test_native_errors_are_reported_and_recover(fake, monkeypatch, stream):
    chat, _, _, workers, _ = fake
    poll = FakeWorker.poll
    def fail(self, timeout):
        self.fail = True
        return poll(self, timeout)
    monkeypatch.setattr(FakeWorker, 'poll', fail)
    with running(native_service(chat)) as port:
        status, text = request(port, body(stream=stream))
        error = events(text)[-1]['error'] if stream else json.loads(text)['error']
        assert status == (200 if stream else 500)
        assert error['type'] == 'server_error' and 'fake failure' in error['message']
        assert chat.require_reset and workers[-1].closed
        monkeypatch.setattr(FakeWorker, 'poll', poll)
        assert request(port, body())[0] == 200
        assert not chat.require_reset


@pytest.mark.parametrize('payload', [[], body(temperature=-1), {'messages': []},
                                     {'messages': [{'role': 'user', 'content': 3}]}])
def test_invalid_requests_do_not_start_native_worker(fake, payload):
    chat, _, _, workers, _ = fake
    with running(native_service(chat)) as port:
        status, text = request(port, payload)
    assert status == 400 and json.loads(text)['error']['type'] == 'invalid_request_error'
    assert not workers


@pytest.mark.parametrize('stream', [False, True])
def test_python_http_regression(stream):
    service = python_service(thinking=True, reply='why</think>OK')
    with running(service) as port:
        status, text = request(port, dict(messages=[dict(role='user', content='Hi')],
                                        max_tokens=128, stream=stream))
    assert status == 200
    if stream:
        chunks = events(text)
        assert ''.join(c['choices'][0]['delta'].get('content', '')
                       for c in chunks if c['choices']) == 'why</think>OK'
        assert '[DONE]' in text
    else:
        message = json.loads(text)['choices'][0]['message']
        assert message == dict(role='assistant', content='OK', reasoning_content='why')


def test_cli_default_and_explicit_unavailable(monkeypatch):
    assert args().backend == 'python'
    assert args().cache_gib == '6' and args().context == 16384
    assert args('--backend', 'native').backend == 'native'
    monkeypatch.setattr(http, 'serve_http', lambda *a: pytest.fail('must fail before listen'))
    monkeypatch.setattr(sys, 'argv', ['serve_qwen4exp', '--backend', 'native'])
    before = {key: http.os.environ.get(key) for key in http.BEST_ENV}
    with pytest.raises(SystemExit) as error:
        http.main()
    assert error.value.code == 2
    assert before == {key: http.os.environ.get(key) for key in http.BEST_ENV}
    result = subprocess.run([sys.executable, str(Path(http.__file__)), '--help'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and '--backend {python,native}' in result.stdout
    assert '--native-budget-gib' in result.stdout


@pytest.mark.parametrize('serve_error', [False, True])
def test_native_setup_retains_same_engine_and_closes_owners(tmp_path, monkeypatch, serve_error):
    from types import SimpleNamespace
    from mini_llm.native.decoder import NativeDecoder
    from mini_llm.external.gguf_store import MultiShardTensorStore

    closed, engines = [], []

    class Owner:
        def __init__(self, name, **values):
            self.name, self.active = name, False
            self.__dict__.update(values)

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *exc):
            self.active = False
            closed.append(self.name)

        def sampler(self, vocab, *, seed):
            assert self.active and vocab == 7 and seed == 0
            return Owner('sampler', engine=self)

    class Runtime:
        def __init__(self, library):
            assert library.is_file()

        def engine(self, **options):
            assert options == dict(budget_bytes=http.GiB, max_sessions=1)
            engine = Owner('engine')
            engines.append(engine)
            return engine

    def open_decoder(runtime, path, **options):
        assert options['capacity'] == 4 and options['budget_bytes'] == http.GiB
        # Reproduce NativeDecoder.open's inner ExitStack engine scope.
        with runtime.engine(budget_bytes=options['budget_bytes'], max_sessions=1) as engine:
            decoder = Owner('decoder', engine=engine, vocab=7)
        assert engine.active  # Serving outer stack, not factory, retains this owner.
        return decoder

    def session(decoder, sampler, tokenizer, **kwargs):
        assert decoder.engine is sampler.engine
        assert decoder.active and sampler.active and decoder.engine.active
        assert kwargs == dict(eos_ids=(6, 5, 4), queue_capacity=1)
        assert tokenizer == 'existing-tokenizer'
        return SimpleNamespace(decoder=decoder, sampler=sampler)

    def serve(service, options, profile):
        assert len(engines) == 1 and engines[0].active
        assert service.session.decoder.active and service.session.sampler.active
        assert service.config['backend'] == 'native'
        assert service.config['python_tuning_options_applied'] is False
        if serve_error:
            raise RuntimeError('injected transport setup failure')

    library = tmp_path / 'library.so'
    library.touch()
    cli = ['--backend', 'native', '--context', '4', '--native-physical-pages', '1']
    for option in ('library', 'bridge-library', 'ggml-library'):
        cli += ['--native-' + option, str(library)]
    for option in ('budget', 'dense-budget', 'mapped-budget', 'host-budget'):
        cli += ['--native-' + option + '-gib', '1']
    metadata = dict(zip(('tokenizer.ggml.eos_token_id', 'tokenizer.ggml.eot_token_id',
                         'tokenizer.ggml.eom_token_id'), (6, 5, 4)))
    monkeypatch.setattr('mini_llm.native.Runtime', Runtime)
    monkeypatch.setattr(NativeDecoder, 'open', open_decoder)
    monkeypatch.setattr(MultiShardTensorStore, 'open', lambda path: Owner('store', metadata=metadata))
    monkeypatch.setattr('mini_llm.external.gguf_tokenizer.load_gguf_tokenizer',
                        lambda metadata, **kw: 'existing-tokenizer')
    monkeypatch.setattr('mini_llm.native.chat.NativeChatSession', session)
    monkeypatch.setattr(http, 'serve_http', serve)
    before = {key: http.os.environ.get(key) for key in http.BEST_ENV}
    if serve_error:
        with pytest.raises(RuntimeError, match='injected transport'):
            http.run_native(args(*cli))
    else:
        http.run_native(args(*cli))
    assert closed == ['store', 'sampler', 'decoder', 'engine']
    assert before == {key: http.os.environ.get(key) for key in http.BEST_ENV}
