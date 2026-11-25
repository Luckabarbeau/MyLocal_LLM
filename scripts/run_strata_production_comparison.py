#!/usr/bin/env python3
"""Finite, owned installed-Strata HTTP reference. No restart, config mutation or GPU microbench.

Run with the installed Strata Python so preparation uses its real tokenizer/template.
Every invocation has a hard deadline (including teardown); only its own process group is signalled.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import benchmark_qwen4exp_serving as bench

ROOT = Path(__file__).resolve().parents[1]
INSTALL = Path('/home/lucka/.local/share/strata')
TERMINAL = INSTALL / 'terminal/0.1.59.1'
ENGINE = INSTALL / 'components/engine/0.1.59-linux-x86_64-cuda-cuda13/strata'
GPU = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'
GIB = 1024**3


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def memory():
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapFree')}


def guarded_config(original, engine, output, port):
    cfg = copy.deepcopy(original)
    if any(key in cfg for key in ('before_load', 'mcp_servers', 'replicas', 'split_serve', 'vision')):
        raise ValueError('Auxiliary services/hooks require explicit review; not launched')
    if '--mmap-experts' in cfg['args'] or '--resident-experts' in cfg['args']:
        raise ValueError('Resident arena is required, not mmap/resident-subset mode')
    cfg.update(exe=str(engine), gpu=[0], gpu_roles={'0': 'main'}, gpus_asked=True,
               gpu_order='as_given', layer_split='auto', port=port,
               log=str(output / 'engine.log'), host='127.0.0.1')
    # Guard the default arena; do not request a pinned 50-GB allocation.
    if '--ram-keep-gib' not in cfg['args']:
        cfg['args'] += ['--ram-keep-gib', '4']
    elif float(cfg['args'][cfg['args'].index('--ram-keep-gib') + 1]) < 4:
        raise ValueError('Host reserve must be at least 4 GiB')
    return cfg


def prepare(cfg, source_files, workload_id, max_output):
    """Same standard workload generator, but installed server tokenizer/template (CPU only)."""
    sys.path.insert(0, str(TERMINAL))
    sys.path.insert(0, str(TERMINAL / 'tools'))
    from serve.frontend import ChatTemplate
    import strata_tokenizer as st
    directory = Path(cfg['tokenizer'])
    vocab = json.loads((directory / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tcfg = json.loads((directory / 'tokenizer.json').read_text())
    tok = st.Tokenizer(tokens, (directory / 'merges.txt').read_text().split('\n'),
                       json.loads((directory / 'token_type.json').read_text()),
                       tcfg.get('pre') or 'qwen35', tcfg.get('special_ids') or None)
    template = ChatTemplate(directory / 'chat_template.jinja')

    class Sizer:
        def count(self, messages):
            return len(tok.encode(template.render(messages, enable_thinking=True), parse_special=True))

    source, files = bench.load_source_files(source_files) if source_files else (None, [])
    sampling = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, **cfg.get('sampling', {})}
    rows = []
    for index, (label, target) in enumerate((('128', 128), ('1K', 1024), ('4K', 4096))):
        messages, count = bench.sized_source(Sizer(), target, f'{workload_id}-{index}', source)
        payload = dict(model=cfg['model_name'], messages=messages, max_tokens=max_output,
                       stream=True, stream_options={'include_usage': True},
                       chat_template_kwargs={'enable_thinking': True}, **sampling)
        raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        rows.append(dict(size=label, target_prompt_tokens=target, sizing_prompt_tokens=count,
                         request_utf8=raw.decode('utf-8'), request_sha256=digest(raw)))
    identities = {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}
    manifest = dict(schema='production-http-v1', context=262144, max_output=max_output,
                    workload_id=workload_id, source_files=files, tokenizer_files=identities,
                    sampler=sampling, user_agent='qwen4exp-production-http/1', requests=rows)
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return dict(**manifest, workload_sha256=digest(canonical))


def acceptance(rows, expected=3):
    """Conservative evidence gate, not an HTTP-success count or token-parity claim."""
    pending = []
    if len(rows) != expected:
        pending.append('incomplete fresh matrix')
    for row in rows:
        r = row['response']
        name = row['size']
        rates = r.get('production_rates', {})
        tokens = r.get('tokens', {})
        if not r.get('ok') or not r.get('done') or not r.get('usage_available'):
            pending.append(name + ': incomplete HTTP/SSE/usage')
        if (tokens.get('completion_tokens') or 0) < 128:
            pending.append(name + ': natural response too short for sustained decode acceptance')
        if tokens.get('reused_prompt_tokens') is None:
            pending.append(name + ': cache reuse unverified')
        elif tokens['reused_prompt_tokens'] != 0:
            pending.append(name + ': fresh prompt has cache reuse; classify separately')
        if tokens.get('prompt_tokens') != row.get('sizing_prompt_tokens'):
            pending.append(name + ': template/tokenizer agreement unverified')
        prefill = rates.get('uncached_prompt_tokens_per_prefill_second')
        decode = rates.get('emitted_tokens_per_active_decode_second')
        if prefill is None or prefill < 2500:
            pending.append(name + ': prefill target not established')
        if decode is None or decode < 75:
            pending.append(name + ': emitted decode target not established')
    return dict(status='pending' if pending else 'rate_targets_observed_reference_only',
                reasons=pending, matched_candidate_pipeline_verified=False,
                production_comparison_accepted=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(INSTALL / 'install/strata-iq3_s.json'))
    p.add_argument('--engine', default=str(ENGINE))
    p.add_argument('--deadline', type=float, default=170)
    p.add_argument('--startup-timeout', type=float, default=75)
    p.add_argument('--max-output', type=int, default=256)
    p.add_argument('--max-requests', type=int, default=3)
    p.add_argument('--source-file', action='append', default=[])
    p.add_argument('--workload-id', default='production-coding-http-v1')
    p.add_argument('--manifest', help='Reuse exact frozen payload bytes; model ID must match')
    args = p.parse_args(argv)
    if not math.isfinite(args.deadline) or not 30 <= args.deadline <= 175:
        p.error('--deadline must be 30..175 seconds (reserve bounded teardown)')
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        p.error('Positive finite --startup-timeout required')
    if args.max_output < 128 or not 1 <= args.max_requests <= 3:
        p.error('At least 128 natural output budget and 1..3 requests required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    server = None
    owner_write = None
    rows = []
    summary = dict(status='pending', production_comparison_accepted=False)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError('Finite production run deadline exceeded')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.deadline)
    try:
        original_raw = Path(args.config).read_bytes()
        original = json.loads(original_raw)
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                                           text=True, timeout=5)
        if not any(line.startswith('0, ' + GPU + ',') and '5080' in line for line in gpu_info.splitlines()):
            raise ValueError('Physical GPU 0 is not the contracted RTX 5080 UUID')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                                             '--format=csv,noheader'], text=True, timeout=5)
        competitors = [line for line in processes.splitlines() if GPU in line and 'zed' not in line.lower()]
        if competitors:
            raise ValueError('Target GPU busy; no existing job will be stopped: ' + repr(competitors))
        before = memory()
        bench.save_json(output / 'preflight.json', dict(memory=before, gpu=gpu_info, processes=processes,
                        expected_arena_gib=46.84, reserve_gib=4,
                        installed_config_sha256=digest(original_raw)))
        bench.save_json(output / 'installed_config_snapshot.json', original)
        bench.save_json(output / 'engine_identity.json', dict(
            path=str(Path(args.engine).resolve()), sha256=digest(Path(args.engine).read_bytes()),
            server_sha256=digest((TERMINAL / 'serve/server.py').read_bytes())))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cfg = guarded_config(original, Path(args.engine).resolve(), output, port)
        bench.save_json(output / 'runconfig.json', cfg)
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else prepare(
            cfg, args.source_file, args.workload_id, args.max_output)
        canonical = dict(manifest)
        claimed = canonical.pop('workload_sha256')
        if digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode('utf-8')) != claimed:
            raise ValueError('Workload manifest hash mismatch')
        if manifest['context'] != 262144:
            raise ValueError('Only full-context 262144 is permitted')
        bench.save_json(output / 'workload.json', manifest)
        current_memory = memory()
        bench.save_json(output / 'launch_headroom.json', current_memory)
        if current_memory['MemAvailable'] < (46.84 + 4 + 1) * GIB:
            raise ValueError('Insufficient MemAvailable for arena + 4-GiB reserve + 1-GiB startup allowance')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONPATH=str(TERMINAL), PYTHONUNBUFFERED='1', STRATA_RAM_KEEP_GIB='4')
        # Remove inherited experiment knobs; preserve ordinary loader paths.
        removed = {}
        for key in list(env):
            if key.startswith('STRATA_') and key != 'STRATA_RAM_KEEP_GIB':
                removed[key] = env.pop(key)
        command = [str(TERMINAL / 'python/bin/python3'), '-m', 'serve.server', '--engine', 'strata',
                   '--config', str(output / 'runconfig.json'), '--host', '127.0.0.1', '--port', str(port),
                   '--ready-file', str(output / 'instance.json'), '--shutdown-timeout', '2', '--offline']
        bench.save_json(output / 'launch.json', dict(command=command, cwd=str(TERMINAL),
                        env_overrides={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                            'PYTHONPATH', 'STRATA_RAM_KEEP_GIB')}, removed_experiment_environment=removed))
        owner_read, owner_write = os.pipe()
        command += ['--owner-fd', str(owner_read)]
        try:
            with (output / 'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=TERMINAL, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(owner_read,))
        finally:
            os.close(owner_read)
        bench.save_json(output / 'owned.json', dict(pid=server.pid, pgid=server.pid))
        http = bench.HTTP(f'http://127.0.0.1:{port}', timeout=2)
        ready_deadline = min(started + args.deadline - 10, time.monotonic() + args.startup_timeout)
        while time.monotonic() < ready_deadline:
            if server.poll() is not None:
                raise RuntimeError(f'Owned server exited {server.returncode}; see server.log/engine.log')
            health = http.probe('/health')
            if health['status'] == 200 and (health['json'] or {}).get('loaded'):
                break
            if memory()['MemAvailable'] < 4 * GIB:
                raise MemoryError('Host reserve fell below 4 GiB during startup')
            time.sleep(0.5)
        else:
            raise TimeoutError('Production readiness deadline exceeded')
        snapshot = http.snapshot()
        bench.save_json(output / 'ready.json', dict(snapshot=snapshot, memory=memory(),
                        startup_seconds=time.monotonic() - started))
        bench.verify_context(snapshot, cfg['model_name'], 262144)
        for index, request in enumerate(manifest['requests'][:args.max_requests]):
            if memory()['MemAvailable'] < 4 * GIB:
                raise MemoryError('Host reserve below 4 GiB; refusing next request')
            raw = request['request_utf8'].encode('utf-8')
            payload = json.loads(raw)
            if digest(raw) != request['request_sha256'] or payload['model'] != cfg['model_name']:
                raise ValueError('Request hash/model mismatch; no silent model rewrite')
            if not payload.get('stream') or payload.get('max_tokens', 0) < 128:
                raise ValueError('Production streaming/natural budget contract violated')
            forbidden = ('ignore_eos', 'min_tokens', 'logit_bias', 'allowed_token_ids')
            if any(k in payload for k in forbidden):
                raise ValueError('Forced continuation/masking is not a production workload')
            http.timeout = min(120, started + args.deadline - time.monotonic() - 8)
            if http.timeout < 3:
                raise TimeoutError('No deadline remaining for next production request')
            metadata = dict(context=262144, workload='fresh_first_submission', **request)
            metadata.pop('request_utf8')
            response = bench.benchmark_request(http, payload, output / f'{index:04d}-{request["size"]}',
                metadata, resource_pid=server.pid, resource_gpu_uuid=GPU, raw_body=raw)
            row = dict(size=request['size'], sizing_prompt_tokens=request['sizing_prompt_tokens'], response=response)
            rows.append(row)
            bench.journal(output / 'results.jsonl', row)
            print(json.dumps(dict(size=request['size'], ok=response['ok'], tokens=response['tokens'],
                                  ttft=response['ttft_seconds'], rates=response['production_rates'])), flush=True)
            if not response['ok']:
                break
        summary.update(acceptance(rows))
        summary['workload_sha256'] = manifest['workload_sha256']
        summary['pipeline_evidence'] = 'Raw ready/before/after metrics plus engine.log; manual matching required'
    except Exception as exc:
        summary.update(status='pending', error=f'{type(exc).__name__}: {exc}')
        print(summary['error'], file=sys.stderr, flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if owner_write is not None:
            os.close(owner_write)
        # The group is unique to this Popen. Signal it even when the parent exited first.
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait(timeout=2)
        summary.update(requests_completed=len(rows), elapsed_seconds=time.monotonic() - started,
                       final_memory=memory(), server_returncode=server.returncode if server else None)
        bench.save_json(output / 'summary.json', summary)
    return 1 if summary.get('error') or any(not r['response']['ok'] for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
