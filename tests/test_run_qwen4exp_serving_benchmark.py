"""CPU-only owned-runner safety and forwarding regressions; no subprocess/GPU work."""
import io
import json
import signal
import socket
import subprocess
from unittest.mock import Mock

import pytest

from scripts import run_qwen4exp_serving_benchmark as runner
from scripts import benchmark_qwen4exp_serving as benchmark

GPU = 'GPU-e961516f-d77d-813c-98e9-2f5403f71631'


def setup(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    probe = socket.socket()
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
    probe.close()
    server = Mock(pid=987654, returncode=None)
    server.poll.return_value = None
    def wait(**kwargs):
        server.returncode = 0
    server.wait.side_effect = wait
    spawn = Mock(return_value=server)
    run = Mock(return_value=Mock(returncode=0))
    kill = Mock()
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    monkeypatch.setattr(runner.os, 'killpg', kill)
    opener = Mock()
    opener.open.side_effect = lambda *a, **k: io.StringIO(json.dumps(
        dict(loaded=True, max_context=256, model='fixture')))
    monkeypatch.setattr(runner.urllib.request, 'build_opener', Mock(return_value=opener))
    return ['--output', str(tmp_path / 'new'), '--port', str(port), '--context', '256'], spawn, run, kill


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_owned_commands_env_forwarding(monkeypatch, tmp_path, backend):
    args, spawn, run, kill = setup(monkeypatch, tmp_path)
    extra = ['--backend', backend, '--gpu-uuid', GPU, '--model', '/fixture/model.gguf',
             '--max-requests', '2', '--repeat', '2', '--turns', '1',
             '--startup-timeout', '90', '--timeout', '90', '--experiment-timeout', '120']
    native = {'library': '/fixture/native.so', 'bridge-library': '/fixture/bridge.so',
              'ggml-library': '/fixture/ggml.so', 'budget-gib': '10.0',
              'dense-budget-gib': '7.0', 'mapped-budget-gib': '96.0',
              'host-budget-gib': '1.0', 'threads': '4', 'physical-pages': '2',
              'page-size': '16', 'seed': '0', 'queue-capacity': '1'}
    if backend == 'native':
        for key, value in native.items():
            extra += ['--native-' + key, value]
    assert runner.main(args + extra) == 0
    command = spawn.call_args.args[0]
    client = run.call_args.args[0]
    assert command[command.index('--backend') + 1] == backend
    assert command[command.index('--model') + 1] == '/fixture/model.gguf'
    assert client[client.index('--gguf') + 1] == '/fixture/model.gguf'
    assert client[client.index('--max-requests') + 1] == '2'
    assert client[client.index('--resource-gpu-uuid') + 1] == GPU
    for call in (spawn.call_args, run.call_args):
        assert call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == GPU
    assert spawn.call_args.kwargs['start_new_session'] is True
    assert 0 < run.call_args.kwargs['timeout'] <= 120
    assert kill.call_args.args == (987654, signal.SIGINT)
    if backend == 'native':
        for key, value in native.items():
            assert command[command.index('--native-' + key) + 1] == value


@pytest.mark.parametrize('mask', ['', '0', '1', GPU + ',GPU-other'])
def test_no_implicit_gpu_index(monkeypatch, tmp_path, mask):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    spawn = Mock()
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    with pytest.raises(SystemExit):
        runner.main(['--output', str(tmp_path / 'not-created')])
    spawn.assert_not_called()
    assert not (tmp_path / 'not-created').exists()


def test_inherited_uuid_and_python_default(monkeypatch, tmp_path):
    args, spawn, run, _ = setup(monkeypatch, tmp_path)
    assert runner.main(args) == 0
    command = spawn.call_args.args[0]
    assert command[command.index('--backend') + 1] == 'python'
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == GPU


def test_existing_listener_never_spawned_or_killed(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    spawn, kill = Mock(), Mock()
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    monkeypatch.setattr(runner.os, 'killpg', kill)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(OSError):
            runner.main(['--output', str(tmp_path / 'not-created'), '--port', str(listener.getsockname()[1])])
    spawn.assert_not_called()
    kill.assert_not_called()


def test_client_timeout_still_drains_owned_server(monkeypatch, tmp_path):
    args, _, run, kill = setup(monkeypatch, tmp_path)
    run.side_effect = subprocess.TimeoutExpired('client', 120)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.main(args)
    kill.assert_called_once_with(987654, signal.SIGINT)
    assert json.loads((tmp_path / 'new' / 'error.json').read_text())['type'] == 'TimeoutExpired'
    assert (tmp_path / 'new' / 'exit.json').exists()


def test_live_request_cap(monkeypatch, tmp_path):
    args = benchmark.parser().parse_args(['--model', 'fixture', '--output', str(tmp_path / 'http'),
        '--sizes', '128,160', '--context', '256', '--max-output', '8', '--repeat', '3',
        '--turns', '1', '--max-requests', '2'])
    monkeypatch.setattr(benchmark.HTTP, 'snapshot', lambda *a: {})
    monkeypatch.setattr(benchmark, 'verify_context', lambda *a: {})
    monkeypatch.setattr(benchmark, 'Sizer', lambda *a, **k: Mock(source='fixture'))
    monkeypatch.setattr(benchmark, 'sized_source', lambda *a: ([dict(content='code')], 128))
    request = Mock(return_value=dict(ok=True, seconds=1., ttft_seconds=.5, tokens=8))
    monkeypatch.setattr(benchmark, 'benchmark_request', request)
    assert benchmark.run(args) == 0
    assert request.call_count == 2
