"""CPU-only exact-launch contracts: no installed CLI, model, native engine or GPU."""
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts import probe_0324_strata_exact_launch as probe


class Response(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream; charset=utf-8'}


def wire(content='answer', reasoning='thinking', usage=True, done=True):
    events = [dict(choices=[dict(index=0, delta=dict(role='assistant', reasoning_content=reasoning))]),
              dict(choices=[dict(index=0, delta=dict(content=content))]),
              dict(choices=[dict(index=0, delta={}, finish_reason='stop')])]
    if usage:
        events.append(dict(choices=[], usage=dict(prompt_tokens=10, completion_tokens=7, total_tokens=17),
                           timings=dict(predicted_n=7, predicted_ms=100, predicted_per_second=70)))
    result = ''.join('data: ' + json.dumps(e) + '\r\n\r\n' for e in events)
    return (result + ('data: [DONE]\r\n\r\n' if done else '')).encode()


class StreamHTTP:
    response = None

    def __init__(self, data=None):
        self.data = data if data is not None else wire()
        self.calls = []

    def open(self, path, payload, timeout):
        self.calls.append((path, copy.deepcopy(payload), timeout))
        return Response(self.data)

    def cancel(self):
        pass


class Clock:
    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def snapshot(t=0, ticks=10):
    return dict(monotonic=t, clk_tck=100, processes=[dict(pid=9001, starttime=123, ppid=1,
        executable='/actual/native-name', cmdline=['/actual/native-name', '--native-flag', 'actual'],
        utime=ticks, stime=5, major_faults=int(t), io=dict(read_bytes=int(t) * 1024),
        threads=[dict(tid=9001, starttime=123, utime=ticks, stime=5, affinity=[1, 3])],
        log_files=[], memory_bytes=dict(RssAnon=100, RssFile=200, VmLck=50))],
        meminfo_bytes=dict(MemAvailable=20 * probe.GIB),
        cgroup={'memory.peak': '1024', 'memory.high': str(40 * probe.GIB),
                'memory.swap.current': '0'})


class FakeOwned:
    def __init__(self):
        self.directory = Path('/fake/cgroup')
        self.cgroup = Path('/user.slice/app.slice/mylocal-bench-fake.service')
        self.proc = Path('/fake/proc')
        self.known = {(9001, 123)}
        self.signals = []
        self.alive = True
        self.n = 0

    def snapshot(self):
        self.n += 1
        return snapshot(self.n, 10 + self.n * 10)

    def scan(self):
        return []

    def refresh(self):
        return [dict(pid=9001, starttime=123)] if self.alive else []

    def signal(self, sig):
        self.signals.append(sig)
        self.alive = False


class FakeMonitor:
    def __init__(self, owned, output, started, deadline):
        self.output, self.deadline = output, deadline
        self.phase = 'preflight'
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def check(self):
        pass


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    owned = FakeOwned()
    http = StreamHTTP()
    calls = []
    process = SimpleNamespace(pid=9000, poll=lambda: 0, wait=lambda timeout: 0)
    monkeypatch.setattr(probe.isolation, 'require_isolation', lambda: calls.append('gate'))
    monkeypatch.setattr(probe.isolation, 'snapshot', lambda directory: dict(
        memory_current=0, memory_peak=0, memory_events={}, host_available=60 * probe.GIB,
        ancestor_headroom=None, psi_some_avg10=0, psi_full_avg10=0))
    monkeypatch.setattr(probe, 'port_unused', lambda: calls.append('port'))
    monkeypatch.setattr(probe, 'OwnedProcesses', lambda: owned)
    monkeypatch.setattr(probe, 'no_concurrent_model', lambda owned: calls.append('concurrent'))
    monkeypatch.setattr(probe, 'Monitor', FakeMonitor)
    monkeypatch.setattr(probe, 'HTTP', lambda: http)
    monkeypatch.setattr(probe, 'provenance', lambda: dict(configs=[], captured=True))
    monkeypatch.setattr(probe, 'owns_listener', lambda owned: True)
    monkeypatch.setattr(probe, 'startup', lambda *a: calls.append('startup'))
    monkeypatch.setattr(probe, 'capture_endpoints', lambda *a: {
        'v1/models': dict(status=200, json=dict(data=[dict(id='actual-installed-model')]))})
    def spawn(command, **kwargs):
        calls.append(('spawn', command, kwargs))
        assert (tmp_path / 'capture/provenance_before.json').exists()
        assert calls[0] == 'gate'
        return process
    monkeypatch.setattr(probe.subprocess, 'Popen', spawn)
    def run(command, **kwargs):
        calls.append(('run', command, kwargs))
        if '--help' in command:
            return SimpleNamespace(returncode=0, stdout='--stop --port --yes', stderr='')
        return SimpleNamespace(returncode=0, stdout='stopped', stderr='')
    monkeypatch.setattr(probe.subprocess, 'run', run)
    return SimpleNamespace(output=tmp_path / 'capture', owned=owned, http=http, calls=calls, process=process)


def test_exact_command_and_environment_and_conversation(fake_run, monkeypatch):
    monkeypatch.setenv('STRATA_KEEP_EXISTING', '1')
    before = dict(os.environ)
    assert probe.run(fake_run.output) == 0
    assert probe.COMMAND == (
        '/home/lucka/.local/bin/strata', 'launch', '--model', 'iq3_s', '--context', '262144',
        '--parallel', '1', '--vision', 'off', '--tune', 'skip', '--access', 'local',
        '--no-api-key', '--port', '8095', '--yes')
    spawn = next(c for c in fake_run.calls if isinstance(c, tuple) and c[0] == 'spawn')
    assert spawn[1] == probe.COMMAND
    assert spawn[2]['env'] == dict(before, PYTHONUNBUFFERED='1')
    assert spawn[2]['stdout'] is spawn[2]['stderr']
    assert spawn[2]['stdin'] is subprocess.DEVNULL
    assert spawn[2]['start_new_session'] is True
    assert fake_run.output.joinpath('launch.log').stat().st_ino == fake_run.output.joinpath('startup.log').stat().st_ino
    requests = [c[1] for c in fake_run.http.calls]
    assert [r['max_tokens'] for r in requests] == [128, 512, 512, 512]
    assert [len(r['messages']) for r in requests] == [1, 1, 3, 5]
    assert all(r['stream'] and r['stream_options'] == {'include_usage': True} for r in requests)
    assert all(r['model'] == 'actual-installed-model' for r in requests)
    assert requests[2]['messages'][1] == dict(role='assistant', content='answer', reasoning_content='thinking')
    assert requests[3]['messages'][1] == requests[2]['messages'][1]
    assert 'collision' in requests[2]['messages'][-1]['content']
    assert 'Python' in requests[3]['messages'][-1]['content']
    assert requests[1]['seed'] == requests[2]['seed'] == requests[3]['seed'] == 1
    result = json.loads((fake_run.output / 'result.json').read_text())
    assert result['ok'] and result['resource_limit_bound']
    assert result['launcher_returncode_at_ready'] == 0
    assert 'NOT-standard-v1-full-suite' in result['suite']
    assert 'unverified' in result['reported_75_tokens_per_second']
    for name in ('warmup', 'conversation-1', 'conversation-2', 'conversation-3'):
        assert all((fake_run.output / name / file).exists() for file in (
            'request.json', 'response.json', 'stream.jsonl', 'stats.json', 'process_before.json', 'process_after.json'))
    assert fake_run.owned.signals == [signal.SIGTERM, signal.SIGKILL]
    assert any(c[1] == probe.STOP for c in fake_run.calls if isinstance(c, tuple) and c[0] == 'run')


def test_gate_precedes_all_inspection_and_subprocess(monkeypatch, tmp_path):
    def refuse():
        raise RuntimeError('not isolated')
    monkeypatch.setattr(probe.isolation, 'require_isolation', refuse)
    monkeypatch.setattr(probe, 'provenance', lambda: pytest.fail('inspection before gate'))
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a, **k: pytest.fail('launch before gate'))
    with pytest.raises(RuntimeError, match='not isolated'):
        probe.run(tmp_path / 'capture')
    assert not (tmp_path / 'capture').exists()


def test_exclusive_output_preserved(fake_run):
    fake_run.output.mkdir()
    (fake_run.output / 'existing').write_text('keep')
    with pytest.raises(FileExistsError):
        probe.run(fake_run.output)
    assert (fake_run.output / 'existing').read_text() == 'keep'
    assert fake_run.calls == ['gate']


def test_active_port_refused_without_launch_or_stop(fake_run, monkeypatch):
    def active():
        raise RuntimeError('Port 8095 already active')
    monkeypatch.setattr(probe, 'port_unused', active)
    assert probe.run(fake_run.output) == 1
    assert not any(isinstance(c, tuple) for c in fake_run.calls)
    assert not fake_run.owned.signals
    assert 'already active' in json.loads((fake_run.output / 'result.json').read_text())['error']


@pytest.mark.parametrize('failure', ['startup', 'provenance', 'stream', 'spawn'])
def test_exception_keeps_artifacts_and_cleans_owned(fake_run, monkeypatch, failure):
    def fail(*a, **k):
        raise RuntimeError('specific blocker')
    if failure == 'provenance':
        monkeypatch.setattr(probe, 'provenance', fail)
    elif failure == 'startup':
        monkeypatch.setattr(probe, 'startup', fail)
    elif failure == 'spawn':
        monkeypatch.setattr(probe.subprocess, 'Popen', fail)
    else:
        fake_run.http.data = wire(done=False)
    assert probe.run(fake_run.output) == 1
    result = json.loads((fake_run.output / 'result.json').read_text())
    assert not result['ok']
    assert (fake_run.output / 'cleanup.json').exists()
    assert fake_run.owned.signals == [signal.SIGTERM, signal.SIGKILL]
    if failure == 'stream':
        response = json.loads((fake_run.output / 'warmup/response.json').read_text())
        assert response['message']['content'] == 'answer'
        assert 'without [DONE]' in response['error']
        assert (fake_run.output / 'warmup/process_after.json').exists()


def test_interruption_outcome_precedes_slow_cleanup(fake_run, monkeypatch):
    def interrupted_startup(*args):
        signal.raise_signal(signal.SIGTERM)

    def blocked_cleanup(*args):
        value = json.loads((fake_run.output / 'outcome_before_cleanup.json').read_text())
        assert not value['ok'] and not value['finalized']
        assert value['phase'] == 'startup' and value['cleanup_status'] == 'pending'
        assert 'InterruptedError: Interrupted by signal' in value['error']
        assert not (fake_run.output / 'result.json').exists()
        assert not (fake_run.output / 'cleanup.json').exists()
        raise RuntimeError('simulated cleanup interruption')

    monkeypatch.setattr(probe, 'startup', interrupted_startup)
    monkeypatch.setattr(probe, 'cleanup', blocked_cleanup)
    old_handler = signal.getsignal(signal.SIGTERM)
    assert probe.run(fake_run.output) == 1
    assert signal.getsignal(signal.SIGTERM) == old_handler
    result = json.loads((fake_run.output / 'result.json').read_text())
    assert result['cleanup_error'] == 'simulated cleanup interruption'


def test_no_stop_when_listener_not_owned_or_help_unsupported(tmp_path, monkeypatch):
    for index, supported in enumerate((False, True)):
        output = tmp_path / str(index)
        output.mkdir()
        owned = FakeOwned()
        if supported:
            def foreign(*a):
                raise RuntimeError('foreign listener')
            monkeypatch.setattr(probe, 'owns_listener', foreign)
        else:
            monkeypatch.setattr(probe, 'owns_listener', lambda owned: True)
        monkeypatch.setattr(probe.subprocess, 'run', lambda *a, **k: pytest.fail('unsafe stop'))
        report = probe.cleanup(None, owned, output, supported, StreamHTTP())
        assert not report['stop_attempted']
        assert owned.signals == [signal.SIGTERM, signal.SIGKILL]


def test_stop_timeout_still_terminates_daemonized_children(tmp_path, monkeypatch):
    owned = FakeOwned()
    monkeypatch.setattr(probe, 'owns_listener', lambda owned: True)
    def timeout(command, **kwargs):
        assert command == probe.STOP and kwargs['timeout'] == 10
        raise subprocess.TimeoutExpired(command, 10)
    monkeypatch.setattr(probe.subprocess, 'run', timeout)
    report = probe.cleanup(SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0), owned, tmp_path, True, StreamHTTP())
    assert report['stop_attempted'] and 'stop_error' in report
    assert not report['remaining_owned']
    assert owned.signals == [signal.SIGTERM, signal.SIGKILL]


def test_sse_byte_splits_comments_multiline_unicode_and_eof():
    parser = probe.SSE()
    data = ': ping\r\nevent: message\r\ndata: {"x":\r\ndata: "ž"}\r\n\r\ndata: [DONE]'.encode()
    result = []
    for byte in data:
        result += parser.feed(bytes([byte]))
    result += parser.feed(b'', eof=True)
    assert json.loads(result[0]) == {'x': 'ž'}
    assert result[1] == '[DONE]'
    with pytest.raises(RuntimeError, match='Oversized'):
        probe.SSE().feed(b'x' * (4 * 1024**2 + 1))


def test_stream_uses_usage_not_chunk_count_and_records_metadata(tmp_path):
    http = StreamHTTP()
    ticks = iter(range(20))
    result = probe.consume_stream(http, {'stream': True}, tmp_path, clock=lambda: next(ticks) / 10)
    assert result['done'] and result['usage']['completion_tokens'] == 7
    assert result['message'] == dict(role='assistant', content='answer', reasoning_content='thinking')
    assert result['first_reasoning_seconds'] < result['first_content_seconds']
    assert result['timings']['predicted_per_second'] == 70
    assert len(result['events']) == 4
    assert 'token intervals' in result['timing_note']
    assert 'token_interval' not in result
    assert json.loads((tmp_path / 'response.json').read_text())['usage']['completion_tokens'] == 7


@pytest.mark.parametrize('usage,done,error', [(False, True, 'Missing final usage'), (True, False, 'without [DONE]')])
def test_incomplete_stream_is_failure_not_fallback(tmp_path, usage, done, error):
    http = StreamHTTP(wire(usage=usage, done=done))
    with pytest.raises(RuntimeError, match=re_escape(error)):
        probe.consume_stream(http, {}, tmp_path)
    assert len(http.calls) == 1
    assert 'error' in json.loads((tmp_path / 'response.json').read_text())


def re_escape(text):
    import re
    return re.escape(text)


def test_nonstream_response_explicitly_fails(tmp_path):
    http = StreamHTTP()
    response = Response(b'{"error":"stream unsupported"}')
    response.headers = {'Content-Type': 'application/json'}
    http.open = lambda *a, **k: response
    with pytest.raises(RuntimeError, match='no nonstream fallback'):
        probe.consume_stream(http, {}, tmp_path)
    assert 'unsupported' in json.loads((tmp_path / 'response.json').read_text())['unsupported_response']


def test_fake_http_proxy_bypass_errors_and_stream(tmp_path, monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            status = 503 if self.path == '/status' else 200
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(dict(status='ok', api_key='do-not-retain')).encode())

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if self.path == '/reject':
                self.send_error(400, 'stream unsupported')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            data = wire()
            for start in range(0, len(data), 3):
                self.wfile.write(data[start:start + 3])
                self.wfile.flush()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('no_proxy', '')
    try:
        http = probe.HTTP(f'http://127.0.0.1:{server.server_port}')
        row = http.get('/status')
        assert row['status'] == 503 and row['json']['api_key'] == '[REDACTED]'
        assert 'do-not-retain' not in json.dumps(row)
        result = probe.consume_stream(http, dict(stream=True), tmp_path)
        assert result['done'] and len(requests) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_startup_120s_deadline_no_http_during_download(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(probe.time, 'monotonic', clock)
    monkeypatch.setattr(probe.time, 'sleep', clock.sleep)
    monkeypatch.setattr(probe, 'owns_listener', lambda owned: False)
    monitor = FakeMonitor(None, tmp_path, 0, 280)
    http = SimpleNamespace(get=lambda *a, **k: pytest.fail('HTTP while downloading/not listening'))
    with pytest.raises(TimeoutError, match='120s startup deadline'):
        probe.startup(http, FakeOwned(), monitor, 120)
    assert 120 <= clock() < 120.2


def test_startup_waits_for_health_not_cli_exit(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(probe.time, 'monotonic', clock)
    monkeypatch.setattr(probe.time, 'sleep', clock.sleep)
    monkeypatch.setattr(probe, 'owns_listener', lambda owned: True)
    rows = iter([dict(status=503, json=dict(status='loading')), dict(status=200, json=dict(status='ok'))])
    http = SimpleNamespace(get=lambda *a, **k: next(rows))
    bounded = probe.bounded
    monkeypatch.setattr(probe, 'bounded', lambda call, deadline, check: bounded(call, deadline, check, clock=clock))
    probe.startup(http, FakeOwned(), FakeMonitor(None, tmp_path, 0, 280), 120)
    assert clock() == pytest.approx(.1)
    assert len((tmp_path / 'health_poll.jsonl').read_text().splitlines()) == 2


def test_progress_independent_timer_every_five_seconds(tmp_path, monkeypatch, capsys):
    clock = Clock()
    monitor = probe.Monitor(FakeOwned(), tmp_path, 0, 280, clock)
    monitor.phase = 'startup'
    monitor.last_line = 'fetching optional drafter'
    monitor.hwm = 40 * probe.GIB
    class Stop:
        def is_set(self):
            return clock() > 15
        def wait(self, seconds):
            clock.sleep(seconds)
    monitor.stopped = Stop()
    monkeypatch.setattr(monitor, 'sample', lambda: None)
    monitor.run()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    for seconds, line in zip((0, 5, 10, 15), lines):
        assert f'elapsed={seconds:.1f}s' in line and 'cgroup_hwm=40.00GiB' in line
        assert 'fetching optional drafter' in line


def test_bounded_wall_clock_cancellation_not_socket_inactivity():
    blocker = threading.Event()
    cancelled = []
    def cancel():
        cancelled.append(True)
        blocker.set()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='Wall-clock'):
        probe.bounded(lambda: blocker.wait(2), started + .05, cancel=cancel)
    assert cancelled and time.monotonic() - started < .5


def test_proc_stat_cpu_memory_accounting_and_affinities():
    fields = ['S', '123'] + ['0'] * 20
    fields[7], fields[9], fields[11], fields[12], fields[19] = '11', '3', '400', '100', '987'
    row = probe.parse_stat('77 (name with ) tricky) ' + ' '.join(fields))
    assert row == dict(ppid=123, minor_faults=11, major_faults=3, utime=400, stime=100, starttime=987)
    before, after = snapshot(10, 100), snapshot(12, 400)
    delta = probe.cpu_delta(before, after)
    assert delta['wall_seconds'] == 2
    assert delta['processes'][0]['active_core_equivalents'] == 1.5
    assert delta['processes'][0]['read_bytes'] == 2048
    assert delta['processes'][0]['major_faults'] == 2
    assert delta['processes'][0]['threads'][0]['affinity'] == [1, 3]
    after['processes'][0]['starttime'] += 1
    assert not probe.cpu_delta(before, after)['processes']
    values = probe.kv_numbers('RssAnon: 10 kB\nRssFile: 20 kB\nVmLck: 3 kB\nread_bytes: 99\n')
    assert values == dict(RssAnon=10240, RssFile=20480, VmLck=3072, read_bytes=99)


def test_owned_cgroup_and_actual_descendants_exclude_baseline_and_pid_reuse(monkeypatch):
    owned = object.__new__(probe.OwnedProcesses)
    owned.cgroup = Path('/app.slice/mylocal-bench-test.service')
    owned.baseline = {(1, 1), (2, 2)}
    owned.known = set()
    owned.lock = threading.Lock()
    def row(pid, birth, cg, parent=1):
        return dict(pid=pid, starttime=birth, ppid=parent, cgroup='0::' + cg)
    rows = [row(1, 1, str(owned.cgroup)), row(2, 2, '/desktop'),
            row(3, 3, str(owned.cgroup)), row(4, 4, '/escaped', 3), row(5, 5, '/other')]
    owned.scan = lambda: rows
    assert {r['pid'] for r in owned.refresh()} == {3, 4}
    rows[:] = [row(4, 4, '/escaped'), row(3, 999, '/unrelated')]
    assert {r['pid'] for r in owned.refresh()} == {4}


def test_provenance_redacts_secrets_and_preserves_source_files(tmp_path):
    config = tmp_path / 'config'
    install = tmp_path / 'install'
    config.mkdir()
    install.mkdir()
    launcher = tmp_path / 'executable'
    launcher.write_bytes(b'not actually executable')
    settings = config / 'settings.json'
    settings.write_text(json.dumps(dict(api_key='private-A', args=['--api-key', 'private-B', '--native', '/weights'],
                                       nested=dict(password='private-C'), max_tokens=512)))
    run = install / 'run-iq3_s.sh'
    run.write_text('API_KEY="private-D"\nstrata --api-key=private-E\n')
    before = {p: p.read_bytes() for p in (launcher, settings, run)}
    value = probe.provenance(config, install, launcher)
    rendered = json.dumps(value)
    assert all(secret not in rendered for secret in ('private-A', 'private-B', 'private-C', 'private-D', 'private-E'))
    assert '/weights' in rendered and 'max_tokens' in rendered
    assert value['launcher']['sha256'] == probe.hashlib.sha256(before[launcher]).hexdigest()
    assert {p: p.read_bytes() for p in before} == before
    settings.write_text('{"max_tokens":256}')
    after = probe.provenance(config, install, launcher)
    assert after['configs'] != value['configs']
    assert 'NOT full configuration' in after['limitations']


def test_native_logs_can_be_snapshotted_before_and_after(tmp_path):
    log = tmp_path / 'engine.log'
    log.write_text('INFO native flags: --actual-option 1\n')
    output = tmp_path / 'out'
    output.mkdir()
    def snap():
        value = snapshot()
        value['processes'][0]['log_files'] = [str(log)]
        return value
    owned = SimpleNamespace(snapshot=snap)
    probe.capture_native_logs(owned, output)
    log.write_text(log.read_text() + 'INFO after requests\n')
    probe.capture_native_logs(owned, output)
    captures = [json.loads(p.read_text()) for p in output.glob('native_log_*.json')]
    assert any('INFO after requests' in c['head'] + c['tail'] for c in captures)


def test_real_retained_sources_without_tokenizer_or_model():
    imported_before = set(sys.modules)
    requests, source = probe.load_workload()
    retained, _ = probe.load_workload(probe.RETAINED)
    assert requests == retained
    assert requests[0]['max_tokens'] == 128 and requests[1]['max_tokens'] == 512
    assert 'hash table' in requests[1]['messages'][0]['content']
    assert source['sha256']
    # Other suites may already have imported a backend; this loader must not.
    newly_imported = set(sys.modules) - imported_before
    assert not any(k in newly_imported for k in ('cupy', 'torch', 'mini_llm.models.qwen4'))


@pytest.mark.parametrize('enabled', [False, True])
def test_decode_wrapper_route_profile_is_explicit_and_no_movement(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    from scripts import probe_0302_decode_configuration as wrapper
    calls = []
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=lambda: calls.append('gate')))
    output = tmp_path / 'capture'
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(output)] +
                        (['--route-profile'] if enabled else []))
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_ROUTE_PROFILE', str(int(not enabled)))

    def run(command, **kwargs):
        assert calls == ['gate']
        calls.append('spawn')
        assert kwargs['env']['MINI_LLM_NATIVE_MOE_ROUTE_PROFILE'] == str(int(enabled))
        assert kwargs['env']['MINI_LLM_NATIVE_MOE_STAGE_LIMIT'] == '0'
        assert kwargs['env']['MINI_LLM_SIZED_EXPERT_SLOTS'] == '1'
        assert command[command.index('--promotion-limit') + 1] == '0'
        assert command[command.index('--resident-host-gib') + 1] == '0'
        output.mkdir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, 'run', run)
    with pytest.raises(SystemExit) as exc:
        wrapper.main()
    assert exc.value.code == 0
    value = json.loads((output / 'reproduce.json').read_text())
    assert value['route_profile_diagnostic'] is enabled
    assert value['environment']['MINI_LLM_NATIVE_MOE_ROUTE_PROFILE'] == str(int(enabled))
    assert os.environ['MINI_LLM_NATIVE_MOE_ROUTE_PROFILE'] == str(int(not enabled))


@pytest.mark.parametrize('trace', [False, True])
def test_decode_wrapper_forwards_setup_profile_and_explicit_tokens(tmp_path, monkeypatch, trace):
    from scripts import probe_0302_decode_configuration as wrapper
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark', SimpleNamespace(require_isolation=lambda: None))
    output = tmp_path/'capture'
    profile = tmp_path/'prior.json'
    trajectory = tmp_path/'tokens.json'
    monkeypatch.setattr(sys, 'argv', ['probe','--output',str(output),'--tokens-from',str(trajectory),
        '--expert-profile-from',str(profile),'--profile-case-index','1'] +
        (['--trace','--trace-response','2','--native-timeline'] if trace else []))

    def run(command, **kwargs):
        assert command[command.index('--force-tokens-from')+1] == str(trajectory)
        assert command[command.index('--expert-profile-from')+1] == str(profile)
        assert command[command.index('--profile-case-index')+1] == '1'
        assert '--warmup-frequency-placement' not in command
        assert kwargs['env']['MINI_LLM_NATIVE_MOE_TIMELINE'] == str(int(trace))
        if trace:
            assert command[0] == '/usr/local/cuda/bin/nsys'
            assert command[command.index('--capture-response')+1] == '2'
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, 'run', run)
    with pytest.raises(SystemExit) as exc:
        wrapper.main()
    assert exc.value.code == 0


@pytest.mark.parametrize('arguments', [
    ['--natural','--tokens-from','prior.json'],
    ['--parity','--expert-profile-from','prior.json'],
    ['--trace','--warmup-frequency-placement'],
    ['--expert-profile-from','prior.json','--warmup-frequency-placement'],
    ['--profile-case-index','-1'],
    ['--trace-response','2'],
])
def test_decode_wrapper_rejects_incompatible_profile_flags(tmp_path, monkeypatch, arguments):
    from scripts import probe_0302_decode_configuration as wrapper
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark', SimpleNamespace(require_isolation=lambda: None))
    monkeypatch.setattr(sys, 'argv', ['probe','--output',str(tmp_path/'capture'),*arguments])
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda *a,**kw: pytest.fail('invalid profile started process'))
    with pytest.raises(SystemExit) as exc:
        wrapper.main()
    assert exc.value.code == 2


def test_decode_wrapper_isolation_precedes_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    from scripts import probe_0302_decode_configuration as wrapper

    def refuse():
        raise RuntimeError('not isolated')

    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark', SimpleNamespace(require_isolation=refuse))
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(tmp_path / 'capture'), '--route-profile'])
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda *a, **k: pytest.fail('spawn before isolation'))
    with pytest.raises(RuntimeError, match='not isolated'):
        wrapper.main()
    assert not (tmp_path / 'capture').exists()


def test_bad_retained_sampling_refused(tmp_path):
    rows = json.loads(probe.RETAINED.read_text())
    rows['cases'][0]['captured_request']['temperature'] = 0
    path = tmp_path / 'retained.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='sampling mismatch'):
        probe.load_workload(path)


def test_optional_endpoint_errors_structured_required_props_fatal(tmp_path):
    monitor = FakeMonitor(None, tmp_path, 0, time.monotonic() + 20)
    def get(path):
        return dict(status=404 if path not in ('/health', '/props') else 200, json={})
    result = probe.capture_endpoints(SimpleNamespace(get=get), tmp_path, 'before', monitor)
    assert result['slots']['status'] == 404
    assert (tmp_path / 'v1_models_before.json').exists()
    def missing(path):
        return dict(status=404 if path == '/props' else 200, json={})
    with pytest.raises(RuntimeError, match='Required /props'):
        probe.capture_endpoints(SimpleNamespace(get=missing), tmp_path, 'after', monitor)
