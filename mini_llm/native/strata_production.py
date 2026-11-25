"""Launch the pinned, source-built Strata HTTP pipeline; no token execution here."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'mini_llm/native_cpp/strata_production'
SOURCE = ROOT / 'mini_llm/native_cpp/third_party/strata-712fec6b4297'
ENGINE = WRAPPER / 'build/rtx5080-release/upstream/strata'
INSTALL = Path('/home/lucka/.local/share/strata')
CONFIG = INSTALL / 'install/strata-iq3_s.json'
PYTHON = INSTALL / 'terminal/0.1.59.1/python/bin/python3'
GPU = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'
PIN = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
GIB = 1024**3
REQUIRED_GIB = 46.84 + 4 + 1
REQUIRED_OPTIONS = {'--expert-cache': 'auto', '--prefill': 'auto',
                    '--max-context': '262144', '--kv': 'int8', '--pool-workers': '15'}


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable; refusing model load')


def require_headroom():
    available = available_memory()
    if available < REQUIRED_GIB * GIB:
        raise RuntimeError(f'Model load BLOCKED: {available / GIB:.2f} GiB MemAvailable; '
                           f'{REQUIRED_GIB:.2f} GiB required (46.84 arena + 4 reserve + 1 startup). '
                           'No oversubscription or existing-job control permitted.')


def option(args, key):
    positions = [i for i, value in enumerate(args) if value == key]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise ValueError(f'Expected exactly one {key} with a value')
    return args[positions[0] + 1]


def validate_args(args):
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError('Production args must be a string list')
    for key, value in REQUIRED_OPTIONS.items():
        if option(args, key) != value:
            raise ValueError(f'Production requires {key} {value}')
    for value in args:
        key = value.split('=', 1)[0]
        if key in ('--mmap-experts', '--resident-experts', '--gpu', '--layer-split',
                   '--batch', '--slots') or key.startswith(('--peer-', '--remote-', '--split-')):
            raise ValueError(f'Not permitted in single-GPU resident production: {value}')
    if '--ram-keep-gib' in args:
        reserve = float(option(args, '--ram-keep-gib'))
        if not math.isfinite(reserve) or reserve < 4:
            raise ValueError('Host reserve must be finite and at least 4 GiB')


def production_config(original, *, host, port, log, api_key=None):
    """Copy installed assets/drafting declaration, never write the installed config."""
    cfg = copy.deepcopy(original)
    forbidden = ('before_load', 'mcp_servers', 'replicas', 'split_serve', 'vision', 'images',
                 'split_fallback', 'user_args', 'env', 'parallel', 'drafting', 'tuning',
                 'lazy_load', 'open_browser')
    if any(key in cfg for key in forbidden):
        raise ValueError('Auxiliary services, tuning or alternate launch modes require explicit review')
    if cfg.get('backend') not in (None, 'cuda') or cfg.get('arch') not in (None, 'qwen4exp'):
        raise ValueError('Only the contracted CUDA Qwen production profile is supported')
    validate_args(cfg['args'])
    for key in ('--pack', '--native', '--ple-gguf', '--expert-profile'):
        path = Path(option(cfg['args'], key))
        if not path.is_absolute() or not path.exists():
            raise ValueError(f'{key} must name an existing absolute production asset')
    if not Path(cfg['tokenizer']).is_absolute() or not Path(cfg['tokenizer']).is_dir():
        raise ValueError('Expected existing absolute production tokenizer directory')
    if '--ram-keep-gib' not in cfg['args']:
        cfg['args'] += ['--ram-keep-gib', '4']
    cfg.update(exe=str(ENGINE), gpu=[0], gpu_roles={'0': 'main'}, gpus_asked=True,
               gpu_order='as_given', layer_split='auto', host=host, port=port, log=str(log))
    if api_key is not None:
        cfg['api_key'] = api_key
    return cfg


def identity():
    """Check the build receipt, not an installed executable with a built label."""
    receipt = json.loads((WRAPPER / 'build_validation.json').read_text())
    if receipt['source_revision'] != PIN:
        raise ValueError('Unexpected Strata build source revision')
    artifacts = {}
    for name in ('strata', 'libstrata_kq.so'):
        path = ENGINE.parent / name
        expected = receipt['artifacts']['build/rtx5080-release/upstream/' + name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing regular source-built artifact: {path}')
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError(f'Source-built artifact differs from build receipt: {path}')
        artifacts[name] = digest
    return dict(source_revision=PIN, engine=str(ENGINE), artifacts=artifacts,
                http_server=str(SOURCE / 'serve/server.py'))


def verify_gpu():
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                    '--format=csv,noheader'], text=True, timeout=5)
    if not any(line.startswith('0, ' + GPU + ',') and '5080' in line for line in info.splitlines()):
        raise RuntimeError('Physical GPU 0 is not the contracted RTX 5080 UUID')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                                         '--format=csv,noheader'], text=True, timeout=5)
    busy = [line for line in processes.splitlines() if GPU in line and 'zed' not in line.lower()]
    if busy:
        raise RuntimeError('GPU 0 busy; no existing job will be stopped: ' + repr(busy))


def child_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('STRATA_', 'MINI_LLM_', 'PYTHON', 'CUDA_VISIBLE_',
                                'HIP_VISIBLE_', 'ROCR_VISIBLE_'))}
    env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONPATH=os.pathsep.join((str(SOURCE), str(SOURCE / 'tools'))),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', STRATA_RAM_KEEP_GIB='4')
    return env


def child_command(config_path, ready_file, owner_fd):
    # -I excludes cwd/PYTHONPATH/site-user collisions; explicitly insert only pinned source.
    bootstrap = ('import runpy,sys; '
                 f'sys.path[:0]=[{str(SOURCE)!r},{str(SOURCE / "tools")!r}]; '
                 f'm=runpy.run_path({str(Path(__file__).resolve())!r}); '
                 'raise SystemExit(m["child_main"](sys.argv[1:]))')
    return [str(PYTHON), '-I', '-B', '-u', '-c', bootstrap, '--engine', 'strata',
            '--config', config_path, '--host', '127.0.0.1', '--ready-file', str(ready_file),
            '--owner-fd', str(owner_fd), '--shutdown-timeout', '2', '--offline']


def install_launch_guards(server):
    """Wrapper-only process/config safety; retain upstream Engine and all generation logic."""
    original_env, original_popen = server.child_env, server.popen

    def pinned_env(cfg, **kwargs):
        if cfg.get('gpu') != [0] or cfg.get('gpu_roles') != {'0': 'main'}:
            raise ValueError('Only physical GPU 0 is permitted')
        env = original_env(cfg, **kwargs)
        env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID')
        return env

    def guarded_popen(label, command, **kwargs):
        if label == 'the Strata engine':
            if command[0] != str(ENGINE) or command[1] != '--serve':
                raise ValueError('Refusing non-built engine or alternate entrypoint')
            validate_args(command[2:])
            if kwargs['env'].get('CUDA_VISIBLE_DEVICES') != GPU:
                raise ValueError('Engine UUID visibility mismatch')
            require_headroom()
            print('[strata-native] effective upstream engine argv: ' + json.dumps(command), flush=True)
        return original_popen(label, command, **kwargs)

    def immutable(*args, **kwargs):
        raise ValueError('strata-native launch configuration is immutable; restart with a reviewed profile')

    server.child_env = pinned_env
    server.popen = guarded_popen
    server.apply_saved_tuning = lambda cfg, **kwargs: cfg
    # Never signal an orphan or process recorded by another launcher.
    server.end_leftover_engines = lambda exe: None
    server.runconfig.save = immutable


def child_main(argv):
    from serve import server
    if Path(server.__file__).resolve() != SOURCE / 'serve/server.py':
        raise RuntimeError('Upstream serve.server ownership mismatch')
    install_launch_guards(server)
    return server.main(argv)


def sealed_config(cfg):
    """Linux sealed anonymous JSON: readable by upstream, impossible to rewrite/rename."""
    fd = os.memfd_create('strata-production-config.json', os.MFD_ALLOW_SEALING)
    try:
        os.write(fd, json.dumps(cfg).encode('utf-8'))
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return fd
    except BaseException:
        os.close(fd)
        raise


def run(args):
    if args.host != '127.0.0.1':
        raise ValueError('strata-native currently binds only 127.0.0.1')
    build = identity()
    original_raw = args.strata_config.read_bytes()
    original = json.loads(original_raw)
    with tempfile.TemporaryDirectory(prefix='mini-llm-strata-') as directory:
        work = Path(directory)
        cfg = production_config(original, host=args.host, port=args.port,
                                log=args.strata_log.resolve(), api_key=args.api_key)
        report = dict(backend='strata-native', **build, config_sha256=hashlib.sha256(original_raw).hexdigest(),
                      config=copy.deepcopy(cfg), physical_gpu=0, cuda_visible_devices=GPU,
                      mem_available_gib=available_memory() / GIB, required_gib=REQUIRED_GIB,
                      launch_blocked=available_memory() < REQUIRED_GIB * GIB,
                      performance_acceptance='pending; no model-loaded validation')
        if 'api_key' in report['config']:
            report['config']['api_key'] = '<redacted>'
        if args.strata_dry_run:
            print(json.dumps(report, indent=2))
            return
        require_headroom()
        verify_gpu()
        config_fd = sealed_config(cfg)
        owner_read, owner_write = os.pipe()
        env = child_environment()
        env['STRATA_OWNER_ATTEST_KEY'] = secrets.token_hex(32)
        command = child_command(f'/proc/self/fd/{config_fd}', work / 'instance.json', owner_read)
        command += ['--port', str(args.port)]
        process = None
        previous = {}
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, stop)
            print('[strata-native] launching pinned upstream HTTP service; acceptance pending', flush=True)
            process = subprocess.Popen(command, cwd=SOURCE, env=env, start_new_session=True,
                                       pass_fds=(config_fd, owner_read))
            os.close(owner_read)
            owner_read = None
            while process.poll() is None and not stopped:
                if available_memory() < 4 * GIB:
                    stop(signal.SIGTERM, None)
                    raise RuntimeError('Host reserve fell below 4 GiB; stopping only owned Strata service')
                time.sleep(0.5)
            if not stopped and process.returncode:
                raise RuntimeError(f'Upstream Strata service exited {process.returncode}; see {cfg["log"]}')
        finally:
            os.close(owner_write)
            if owner_read is not None:
                os.close(owner_read)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
            os.close(config_fd)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
