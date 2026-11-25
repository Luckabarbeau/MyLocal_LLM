"""CPU-only production handoff/control checks, never throughput benchmarks."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

import pytest

from mini_llm.native import strata_production as production


@pytest.fixture
def profile(tmp_path):
    assets = {}
    for flag in ('--pack', '--native', '--ple-gguf', '--expert-profile'):
        path = tmp_path / flag[2:]
        path.touch()
        assets[flag] = str(path)
    return dict(exe='component:strata-engine', args=[
        *[value for pair in assets.items() for value in pair],
        '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '4',
        '--max-context', '262144', '--kv', 'int8', '--pool-workers', '15'],
        tokenizer=str(tmp_path), model_name='test', cwd=str(tmp_path),
        mtp=False, gpu=[1, 0], gpu_roles={'0': 'main', '1': 'helper'})


def config(profile, tmp_path):
    return production.production_config(profile, host='127.0.0.1', port=8091,
                                        log=tmp_path / 'engine.log')


def test_inherit_no_mutation(profile, tmp_path):
    before = copy.deepcopy(profile)
    cfg = config(profile, tmp_path)
    assert profile == before
    assert cfg['args'] == before['args'] + ['--ram-keep-gib', '4']
    assert cfg['mtp'] is False
    assert production.option(cfg['args'], '--spec') == '4'
    assert cfg['gpu'] == [0] and cfg['gpu_roles'] == {'0': 'main'}
    assert cfg['exe'] == str(production.ENGINE)
    assert cfg['cwd'] == before['cwd'] and cfg['tokenizer'] == before['tokenizer']


@pytest.mark.parametrize('bad', ['--mmap-experts', '--resident-experts', '--gpu=1',
                                 '--peer-device', '--layer-split', '--slots'])
def test_refuse_alternate_modes(profile, tmp_path, bad):
    profile['args'].append(bad)
    with pytest.raises(ValueError):
        config(profile, tmp_path)


@pytest.mark.parametrize('key', ['before_load', 'env', 'user_args', 'replicas',
                                 'split_fallback', 'drafting', 'lazy_load'])
def test_refuse_config_launch_overrides(profile, tmp_path, key):
    profile[key] = {}
    with pytest.raises(ValueError):
        config(profile, tmp_path)


def test_memory_gate_no_swap_credit(monkeypatch):
    monkeypatch.setattr(production, 'available_memory', lambda: 48 * production.GIB)
    with pytest.raises(RuntimeError, match='51.84 GiB required'):
        production.require_headroom()
    monkeypatch.setattr(production, 'available_memory', lambda: 52 * production.GIB)
    production.require_headroom()


def test_sealed_json(profile, tmp_path):
    cfg = config(profile, tmp_path)
    fd = production.sealed_config(cfg)
    try:
        assert json.loads(Path(f'/proc/self/fd/{fd}').read_text()) == cfg
        with pytest.raises(OSError):
            os.write(fd, b'x')
        assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
    finally:
        os.close(fd)


def test_environment_no_inherited_experiments(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setenv('STRATA_MMAP_EXPERTS', '1')
    monkeypatch.setenv('MINI_LLM_BACKEND', 'cupy')
    monkeypatch.setenv('PYTHONPATH', str(production.ROOT))
    env = production.child_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == production.GPU
    assert 'STRATA_MMAP_EXPERTS' not in env and 'MINI_LLM_BACKEND' not in env
    assert str(production.ROOT) not in env['PYTHONPATH'].split(os.pathsep)


def test_guards_keep_upstream_ownership(profile, tmp_path, monkeypatch):
    calls = []
    server = SimpleNamespace(child_env=lambda cfg, **kw: {'CUDA_VISIBLE_DEVICES': '0'},
                             popen=lambda *a, **kw: calls.append((a, kw)),
                             runconfig=SimpleNamespace(save=lambda *a: None))
    production.install_launch_guards(server)
    cfg = config(profile, tmp_path)
    assert server.child_env(cfg)['CUDA_VISIBLE_DEVICES'] == production.GPU
    with pytest.raises(ValueError):
        server.child_env({**cfg, 'gpu': [1]})
    with pytest.raises(ValueError, match='immutable'):
        server.runconfig.save('anything', {})
    assert server.apply_saved_tuning(cfg) is cfg
    monkeypatch.setattr(production, 'available_memory', lambda: 48 * production.GIB)
    with pytest.raises(RuntimeError, match='BLOCKED'):
        server.popen('the Strata engine', [str(production.ENGINE), '--serve', *cfg['args']],
                     env={'CUDA_VISIBLE_DEVICES': production.GPU})
    assert not calls
    with pytest.raises(ValueError, match='non-built'):
        server.popen('the Strata engine', ['/installed/strata', '--serve'], env={})


def test_default_backends_unchanged():
    spec = importlib.util.spec_from_file_location('qwen_serve_flags', production.ROOT / 'scripts/serve_qwen4exp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.argument_parser()
    assert parser.parse_args([]).backend == 'python'
    assert parser.parse_args(['--backend', 'native']).backend == 'native'
    assert parser.parse_args(['--backend', 'strata-native']).backend == 'strata-native'


def test_isolated_upstream_help_and_import_ownership(tmp_path):
    command = production.child_command('/unused', tmp_path / 'ready.json', 999)
    # Help exits before config, owner-fd or model access. Same bootstrap as production.
    result = subprocess.run(command[:6] + ['--help'], cwd=production.ROOT,
                            env=production.child_environment(), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--engine' in result.stdout and '--offline' in result.stdout


def test_upstream_synthetic_ready_gen_done(tmp_path):
    fake = tmp_path / 'fake-engine'
    fake.write_text('#!' + sys.executable + '\n'
                    'import sys\n'
                    'print("INFO kv=int8", flush=True)\n'
                    'print("READY 4096 stop", flush=True)\n'
                    'for line in sys.stdin:\n'
                    ' if line.startswith("GEN "):\n'
                    '  print("PP 2 2 1", flush=True)\n'
                    '  print("T 42", flush=True)\n'
                    '  print("DONE 1 2 1 1 stop 0 0 0", flush=True)\n'
                    ' elif line.startswith("QUIT"): break\n')
    fake.chmod(0o700)
    code = f'''import sys, threading
sys.path[:0] = [{str(production.SOURCE)!r}, {str(production.SOURCE / 'tools')!r}]
from serve import server
assert server.__file__ == {str(production.SOURCE / 'serve/server.py')!r}
server.end_leftover_engines = lambda exe: None
engine = server.StrataEngine({str(fake)!r}, [], env={{'CUDA_VISIBLE_DEVICES': ''}})
try:
    assert engine.max_context == 4096 and engine.can_stop
    assert list(engine.generate([10, 11], 1, {{}}, threading.Event())) == [None, 42]
finally:
    engine.close()
print('upstream READY/GEN/T/DONE synthetic control passed')
'''
    result = subprocess.run([str(production.PYTHON), '-I', '-B', '-c', code],
                            cwd=production.ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_upstream_http_owner_ready_and_eof_shutdown(tmp_path):
    fd = production.sealed_config({})
    owner_read, owner_write = os.pipe()
    ready = tmp_path / 'instance.json'
    env = production.child_environment()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['STRATA_OWNER_ATTEST_KEY'] = 'ab' * 32
    command = production.child_command(f'/proc/self/fd/{fd}', ready, owner_read)
    index = command.index('--engine')
    command[index + 1] = 'mock'
    command += ['--port', '0', '--script', 'Synthetic control response.']
    with (tmp_path / 'server.log').open('w+') as log:
        child = subprocess.Popen(command, cwd=production.ROOT, env=env, start_new_session=True,
                                 pass_fds=(fd, owner_read), stdout=log, stderr=subprocess.STDOUT)
        os.close(owner_read)
        try:
            deadline = time.monotonic() + 8
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            log.flush()
            assert ready.exists(), (tmp_path / 'server.log').read_text()
            instance = json.loads(ready.read_text())
            assert instance['pid'] == child.pid
            port = instance['port']
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                assert response.status == 200
            payload = json.dumps(dict(model='strata', messages=[dict(role='user', content='hello')],
                                      max_tokens=64, stream=True)).encode()
            request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=3) as response:
                assert 'data: [DONE]' in response.read().decode()
            os.close(owner_write)
            owner_write = None
            assert child.wait(timeout=6) == 0
        finally:
            if owner_write is not None:
                os.close(owner_write)
            os.close(fd)
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
