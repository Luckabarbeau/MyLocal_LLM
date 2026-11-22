#!/usr/bin/env python3
"""Bounded owned-server HTTP experiment; never stops an existing listener/process."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--chunk-size', type=int, default=0)
    p.add_argument('--prefill-experts', choices=('decode', 'grouped', 'stream', 'strata'), default='decode')
    p.add_argument('--strata-prefill-library')
    p.add_argument('--sizes', default='128')
    p.add_argument('--max-output', type=int, default=128)
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--turns', type=int, default=2)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--experiment-timeout', type=float, default=1800)
    p.add_argument('--replay-from')
    p.add_argument('--max-requests', type=int)
    p.add_argument('--source-file', action='append', default=[])
    p.add_argument('--no-thinking', action='store_true')
    p.add_argument('--nonstream', action='store_true')
    p.add_argument('--profile', action='store_true', help='Capture production handler cProfile (diagnostic only)')
    p.add_argument('--cache-gib', default='6')
    p.add_argument('--port', type=int, default=8080)
    args = p.parse_args()
    # Bind before spawning; refusing is safer than treating an unrelated service as ours.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    url = f'http://127.0.0.1:{args.port}'
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'
    command = [sys.executable, str(ROOT / 'scripts/serve_qwen4exp.py'),
               '--context', str(args.context), '--qsa-state-storage', 'host',
               '--prefill-chunk-size', str(args.chunk_size), '--prefill-experts', args.prefill_experts,
               '--cache-gib', args.cache_gib,
               '--port', str(args.port)]
    if args.strata_prefill_library:
        command += ['--strata-prefill-library', str(Path(args.strata_prefill_library).resolve())]
    if args.profile:
        command += ['--profile-directory', str(out / 'profiles')]
    if args.no_thinking:
        command.append('--no-thinking')
    (out / 'launch.json').write_text(json.dumps(dict(command=command,
        gpu_uuid=env['CUDA_VISIBLE_DEVICES']), indent=2) + '\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (out / 'server.log').open('w') as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'owned_pid.txt').write_text(str(server.pid) + '\n')
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Owned server exited {server.returncode}; see {out / "server.log"}')
                try:
                    with opener.open(url + '/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('loaded') and health.get('max_context') == args.context:
                        (out / 'ready.json').write_text(json.dumps(health, indent=2) + '\n')
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Owned server readiness deadline exceeded')
            client = [sys.executable, str(ROOT / 'scripts/benchmark_qwen4exp_serving.py'),
                      '--url', url, '--model', health['model'], '--context', str(args.context),
                      '--output', str(out / 'http'), '--sizes', args.sizes,
                      '--max-output', str(args.max_output), '--repeat', str(args.repeat),
                      '--turns', str(args.turns), '--timeout', str(args.timeout),
                      '--workload-id', 'production-recovery-coding-v1',
                      '--resource-pid', str(server.pid), '--resource-gpu-uuid', env['CUDA_VISIBLE_DEVICES']]
            if args.max_requests:
                client += ['--max-requests', str(args.max_requests)]
            for source in args.source_file:
                client += ['--source-file', str(Path(source).resolve())]
            if args.replay_from:
                client += ['--replay-from', str(Path(args.replay_from).resolve())]
            else:
                # This is the same canonical checkpoint as the production entry point.
                from serve_qwen4exp import MODEL
                client += ['--gguf', MODEL]
            if args.no_thinking:
                client.append('--no-thinking')
            if args.nonstream:
                client.append('--nonstream')
            client_env = dict(env, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            with (out / 'client.log').open('w') as client_log:
                result = subprocess.run(client, cwd=ROOT, env=client_env, stdout=client_log,
                                        stderr=subprocess.STDOUT, timeout=args.experiment_timeout)
            return result.returncode
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=10)
            (out / 'exit.json').write_text(json.dumps(dict(server_returncode=server.returncode)) + '\n')


if __name__ == '__main__':
    sys.exit(main())
