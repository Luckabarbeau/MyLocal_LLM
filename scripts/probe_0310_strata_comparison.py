#!/usr/bin/env python3
"""Bounded installed Strata stdin comparison; CLI requires the 0290 isolation gate.

Controlled continuations are a timing diagnostic, NOT sampled-output or full
configuration parity. Importing this module loads no model/CUDA dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path('/home/lucka/.local/share/strata/components/engine/0.1.59-linux-x86_64-cuda-cuda13/strata')
ENGINE_SHA256 = '0ba914b28df1c8e09c0b5209c2ca6040c75fb650a7408ff3d9ef3ad159a13eb9'
SOURCE_COMMIT = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
CONFIG = Path('/home/lucka/.local/share/strata/install/strata-iq3_s.json')
RETAINED = ROOT / 'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json'
HELP = ROOT / 'docs/qwen4exp/diagnostics_0310_preflight_v1/engine_help.txt'
SOURCE = HELP.parent / 'reference/src_program_generate.cpp'
COUNTERS = ('drafts_accepted', 'drafts_offered', 'reused', 'hits', 'lookups',
            'ram_blobs', 'file_blobs', 'file_mb', 'prompt_tokens_read', 'offloaded')
CLEANUP_RESERVE = 10.0
MAX_STDOUT = 8 * 1024**2


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_info(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def token_ids(value):
    if not isinstance(value, list) or not value or any(
            type(t) is not int or not 0 <= t <= 2147483647 for t in value):
        raise ValueError('Expected a nonempty list of nonnegative int32 token IDs')
    return list(value)


def load_requests(path=RETAINED, repeats=2):
    if type(repeats) is not int or not 2 <= repeats <= 4:
        raise ValueError('Require 2..4 response repeats for sustained-rate measurement')
    retained = json.loads(Path(path).read_text())
    cases = retained['cases']
    requests = []
    for name, prompt_n, generated_n in [('warmup', 75, 128), ('performance-short-r1', 77, 359)]:
        matches = [row for row in cases if row['request_id'] == name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one retained {name}')
        row = matches[0]
        prompt = token_ids(row['prompt']['token_ids'])
        follow = token_ids(row['output_token_ids'])
        if (len(prompt) != prompt_n or len(follow) != generated_n
                or row['prompt']['strata_token_ids'] != prompt
                or row['prompt']['expected_prompt_tokens'] != prompt_n
                or row['captured_usage']['prompt_tokens'] != prompt_n):
            raise ValueError(f'Retained prompt/continuation count or tokenizer mismatch: {name}')
        sampling = row['captured_request']
        expected = dict(temperature=1.0, top_p=.95, top_k=20, seed=1,
                        min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
        if any(sampling.get(k) != v for k, v in expected.items()) or sampling.get('frequency_penalty', 0) != 0:
            raise ValueError(f'Retained sampling mismatch: {name}')
        for i in range(1 if name == 'warmup' else repeats):
            requests.append(dict(request_id='warmup' if name == 'warmup' else f'response-r{i+1}',
                                 retained_request_id=name, prompt=prompt, follow=follow))
    return requests


def engine_command(config, engine=ENGINE):
    # Whitelist only model asset arguments, never replay a launcher configuration.
    args = config['args']
    assets = []
    for key in ('--pack', '--native', '--ple-gguf', '--expert-profile'):
        if args.count(key) != 1:
            raise ValueError(f'Expected one installed {key}')
        index = args.index(key)
        if index + 1 >= len(args) or not Path(args[index + 1]).is_absolute():
            raise ValueError(f'Expected absolute installed asset path: {key}')
        assets += [key, args[index + 1]]
    return [str(engine), *assets, '--gpu', '0', '--mmap-experts',
            '--ram-keep-gib', '12', '--memory-limit-mib', '36864',
            '--expert-cache', '2918', '--pool-workers', '19', '--max-context', '1024',
            '--kv', 'int8', '--prefill', '128', '--spec', '2', '--suffix-draft', '0',
            '--lookup-max-t', '0', '--adapt-every', '0', '--adapt-swaps', '0',
            '--pcie-frac', '0', '--prompt-cache', '0', '--conversation-cache-mib', '0', '--serve']


def engine_environment(base, follow_path, profile=False):
    # Start from an environment without inherited Strata experimental switches.
    env = {key: value for key, value in base.items() if not key.startswith('STRATA_')}
    env.update(STRATA_ROUTE_TAIL_SKIP='0', STRATA_FOLLOW=str(follow_path))
    if profile:
        env.update(STRATA_VERIFY_PROFILE='1', STRATA_DECODE_TIMING='1')
    return env


def request_line(request):
    prompt, follow = token_ids(request['prompt']), token_ids(request['follow'])
    if len(prompt) + len(follow) > 1024:
        raise ValueError('Request exceeds the fixed 1024-token context')
    return (f'GEN {len(follow)} temperature=1 top_p=.95 top_k=20 seed=1 ckpt=0 '
            + ','.join(map(str, prompt)) + '\n').encode('ascii')


def request_progress(line, prompt_tokens, state):
    """Validate the pinned serial serve prefill protocol, without resetting deadlines."""
    fields = line.split()
    kind = fields[0] if fields else ''
    progress = dict(kind=kind, raw=line)
    if kind in ('RESUME', 'REUSED'):
        if len(fields) != 2:
            raise ValueError(f'Malformed {kind}: {line}')
        reused = int(fields[1])
        if reused != 0:
            raise ValueError(f'{kind} must be zero with checkpoint caches disabled: {line}')
        if kind == 'RESUME':
            if state['resume'] is not None:
                raise ValueError('Duplicate RESUME')
            state['resume'] = reused
        else:
            if state['resume'] is None or state['reused'] is not None:
                raise ValueError('REUSED must follow RESUME exactly once')
            state['reused'] = reused
        progress['reused'] = reused
    elif kind == 'PP':
        if len(fields) != 5:
            raise ValueError(f'Malformed PP: {line}')
        position, total = int(fields[1]), int(fields[2])
        elapsed_ms, tokens_per_second = float(fields[3]), float(fields[4])
        if state['resume'] is None or state['reused'] is not None:
            raise ValueError('PP must occur between RESUME and REUSED')
        if total != prompt_tokens or not state['position'] <= position <= total:
            raise ValueError(f'PP prompt count, bounds or monotonic position mismatch: {line}')
        if (any(not math.isfinite(v) or v < 0 for v in (elapsed_ms, tokens_per_second))
                or elapsed_ms < state['elapsed_ms']):
            raise ValueError(f'Invalid PP timing/rate: {line}')
        state.update(position=position, elapsed_ms=elapsed_ms)
        progress.update(position=position, prompt_tokens=total, elapsed_ms=elapsed_ms,
                        tokens_per_second=tokens_per_second)
    else:
        raise ValueError(f'Unexpected request progress: {line!r}')
    return progress


def parse_done(line):
    fields = line.split()
    if len(fields) < 6 or fields[0] != 'DONE':
        raise ValueError(f'Malformed DONE: {line}')
    result = dict(generated=int(fields[1]), prompt_tokens=int(fields[2]),
                  prompt_ms=float(fields[3]), decode_ms=float(fields[4]), finish=fields[5])
    if (result['generated'] < 0 or result['prompt_tokens'] < 1
            or result['finish'] not in ('stop', 'length', 'cancel', 'nonfinite')
            or any(not math.isfinite(result[k]) or result[k] < 0 for k in ('prompt_ms', 'decode_ms'))):
        raise ValueError(f'Invalid DONE count/timing: {line}')
    positional, keyed = [], {}
    for value in fields[6:]:
        if '=' in value:
            key, raw = value.split('=', 1)
            if key in keyed:
                raise ValueError(f'Duplicate DONE counter: {key}')
            keyed[key] = raw
        else:
            positional.append(value)
    counters = {}
    for key, raw in zip(COUNTERS, positional):
        value = float(raw) if key == 'file_mb' else int(raw)
        if value < 0 or not math.isfinite(value):
            raise ValueError(f'Invalid DONE counter: {key}')
        counters[key] = value
    for key, raw in keyed.items():
        counters[key] = int(raw) if key in ('chunk', 'experts_streamed') else raw
    result.update(counters=counters, extra_positional=positional[len(COUNTERS):], raw=line)
    return result


def timing_rates(n, decode_ms, first, last, done):
    last_span, done_span = last - first, done - first
    return dict(
        engine_N_per_decode_second=n / (decode_ms / 1000) if decode_ms > 0 else None,
        engine_decode_seconds=decode_ms / 1000,
        observed_firstT_to_lastT_seconds=last_span,
        observed_N_minus_1_per_firstT_to_lastT_second=(n - 1) / last_span if last_span > 0 else None,
        observed_firstT_to_DONE_seconds=done_span,
        observed_N_minus_1_per_firstT_to_DONE_second=(n - 1) / done_span if done_span > 0 else None)


class BinaryProtocol:
    """One binary reader owns stdout; complete buffered lines precede select()."""

    def __init__(self, process, stdout):
        self.process, self.stdout = process, stdout
        self.buffer = bytearray()
        self.observed = 0.0
        self.total = 0
        self.selector = selectors.DefaultSelector()
        self.selector.register(process.stdout, selectors.EVENT_READ)
        os.set_blocking(process.stdin.fileno(), False)

    def send(self, data, deadline):
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not writable.select(remaining):
                    raise TimeoutError('stdin request deadline exceeded')
                try:
                    written = os.write(self.process.stdin.fileno(), data)
                except BlockingIOError:
                    continue
                data = data[written:]

    def line(self, deadline):
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('stdout protocol deadline exceeded')
            if b'\n' in self.buffer:
                raw, _, rest = self.buffer.partition(b'\n')
                self.buffer = bytearray(rest)
                return raw.decode('ascii').rstrip('\r'), self.observed
            if len(self.buffer) > 65536:
                raise ValueError('Oversized stdout protocol line')
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError('stdout protocol deadline exceeded')
            block = os.read(self.process.stdout.fileno(), 65536)
            self.observed = time.monotonic()
            if not block:
                raise RuntimeError('Engine stdout EOF before protocol completion')
            self.stdout.write(block)
            self.stdout.flush()
            self.total += len(block)
            if self.total > MAX_STDOUT:
                raise ValueError('stdout capture limit exceeded')
            self.buffer.extend(block)

    def close(self):
        self.selector.close()


def process_snapshot(pid):
    # This existing helper is standard-library-only at import time.
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    from probe_0254_strata_workload import memory_snapshot
    return memory_snapshot(pid, include_io=True)


def cleanup(process):
    actions = []
    if process.poll() is None:
        try:
            os.write(process.stdin.fileno(), b'QUIT\n')
            actions.append('QUIT')
        except (OSError, ValueError):
            actions.append('QUIT unavailable')
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            actions.append('terminate')
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                actions.append('kill')
                process.wait(timeout=2)
    for stream in (process.stdin, process.stdout):
        stream.close()
    return dict(actions=actions, returncode=process.returncode)


def run_harness(command, requests, output, *, env=None, cwd=None, timeout=230,
                startup_timeout=75, request_timeout=65, profile=False,
                provenance=None, snapshot=None, started=None):
    """Protocol runner; injection points let CPU tests use a tiny Python peer only."""
    if not CLEANUP_RESERVE < timeout <= 230 or not all(
            math.isfinite(v) and 0 < v <= timeout for v in (startup_timeout, request_timeout)):
        raise ValueError('Require total <=230s, positive stage deadlines, and cleanup reserve')
    started = time.monotonic() if started is None else started
    deadline = started + timeout - CLEANUP_RESERVE
    snapshot = process_snapshot if snapshot is None else snapshot
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(ok=False, requests=[], clock='monotonic seconds',
                  timing_note='Pipe-read observation times may coalesce tokens; include proc snapshot overhead. '
                              'Engine N/decode_ms and observer (N-1) rates have different denominators. '
                              'Serial serve decode_ms includes the first emitted token; prompt_ms ends before REUSED.',
                  sustained_response_rates=None)
    process = reader = None
    try:
        for request in requests:
            request_line(request)
        follow = output / 'follow.csv'
        with follow.open('x') as stream:
            for request in requests:
                stream.write(','.join(map(str, request['follow'])) + '\n')
        environment = engine_environment(os.environ if env is None else env, follow, profile)
        write_json(output / 'provenance.json', dict(
            command=command, cwd=str(cwd) if cwd else None, profile=profile,
            environment={k: v for k, v in environment.items() if k.startswith('STRATA_')},
            requests=requests, total_timeout_seconds=timeout, startup_timeout_seconds=startup_timeout,
            request_timeout_seconds=request_timeout, cleanup_reserve_seconds=CLEANUP_RESERVE,
            source_commit=SOURCE_COMMIT, full_configuration_equivalence=False,
            expert_cache_units='2918 largest-blob units (~7.24 GiB), not bytes or actual slot count',
            retained_provenance=provenance))
        with (output / 'stdout.bin').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr, \
                (output / 'events.jsonl').open('x', buffering=1) as events:
            def record(event, **values):
                events.write(json.dumps(dict(event=event, **values), allow_nan=False) + '\n')

            def observe(event, when=None):
                row = dict(monotonic=time.monotonic() if when is None else when)
                try:
                    row['process'] = snapshot(process.pid)
                except (OSError, ValueError) as exc:
                    row['snapshot_error'] = str(exc)
                return row

            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Total deadline exhausted before engine startup')
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=stderr, cwd=cwd, env=environment, bufsize=0)
                reader = BinaryProtocol(process, stdout)
                ready_deadline = min(deadline, time.monotonic() + startup_timeout)
                while True:
                    line, when = reader.line(ready_deadline)
                    if line.startswith('INFO '):
                        record('info', line=line, monotonic=when)
                    elif line.startswith('READY '):
                        break
                    else:
                        raise ValueError(f'Expected READY, got {line!r}')
                record('ready', line=line, **observe('ready', when))
                for request in requests:
                    stage_deadline = min(deadline, time.monotonic() + request_timeout)
                    row = dict(request_id=request['request_id'], token_ids=[], snapshots={}, progress=[])
                    progress_state = dict(resume=None, reused=None, position=0, elapsed_ms=0.0)
                    result['requests'].append(row)
                    row['snapshots']['before'] = observe('before')
                    record('before', request_id=request['request_id'], **row['snapshots']['before'])
                    reader.send(request_line(request), stage_deadline)
                    last = None
                    while True:
                        line, when = reader.line(stage_deadline)
                        if line.partition(' ')[0] in ('RESUME', 'PP', 'REUSED'):
                            progress = request_progress(line, len(request['prompt']), progress_state)
                            progress['monotonic'] = when
                            row['progress'].append(progress)
                            record('progress', request_id=request['request_id'], **progress)
                        elif line.startswith('T '):
                            if progress_state['reused'] is None:
                                raise ValueError('Token received before REUSED')
                            fields = line.split()
                            if len(fields) != 2:
                                raise ValueError(f'Malformed token line: {line}')
                            token = int(fields[1])
                            index = len(row['token_ids'])
                            row['token_ids'].append(token)
                            if index >= len(request['follow']) or token != request['follow'][index]:
                                raise ValueError(f'Token mismatch {request["request_id"]} at index {index}: {token}')
                            last = observe('token', when)
                            if index == 0:
                                row['snapshots']['first_T'] = last
                                record('first_T', request_id=request['request_id'], **last)
                        elif line.startswith('DONE '):
                            row['snapshots']['DONE'] = observe('DONE', when)
                            if last is not None:
                                row['snapshots']['last_T'] = last
                                record('last_T', request_id=request['request_id'], **last)
                            record('DONE', request_id=request['request_id'], line=line, **row['snapshots']['DONE'])
                            done = row['engine'] = parse_done(line)
                            n = len(request['follow'])
                            if row['token_ids'] != request['follow'] or done['generated'] != n:
                                raise ValueError(f'Generated count mismatch: expected {n}, T={len(row["token_ids"])}, DONE={done["generated"]}')
                            if done['prompt_tokens'] != len(request['prompt']):
                                raise ValueError('DONE prompt count mismatch')
                            if done['finish'] not in ('length', 'stop'):
                                raise ValueError(f'Unexpected finish: {done["finish"]}')
                            counters = done['counters']
                            if any(counters.get(k, 0) != 0 for k in ('drafts_accepted', 'drafts_offered', 'reused', 'offloaded')):
                                raise ValueError('Unexpected drafting, checkpoint reuse or PCIe offload')
                            if counters.get('prompt_tokens_read', len(request['prompt'])) != len(request['prompt']):
                                raise ValueError('DONE fresh prompt count mismatch')
                            first = row['snapshots']['first_T']['monotonic']
                            row['rates'] = timing_rates(n, done['decode_ms'], first, last['monotonic'], when)
                            row['ttft_observed_seconds'] = first - row['snapshots']['before']['monotonic']
                            row['validated'] = True
                            break
                        else:
                            raise ValueError(f'Unexpected protocol line: {line!r}')
                responses = [r for r in result['requests'] if r['request_id'] != 'warmup']
                if responses:
                    result['sustained_response_rates'] = dict(repeats=len(responses), **timing_rates(
                        sum(r['engine']['generated'] for r in responses),
                        sum(r['engine']['decode_ms'] for r in responses), 0,
                        sum(r['rates']['observed_firstT_to_lastT_seconds'] for r in responses),
                        sum(r['rates']['observed_firstT_to_DONE_seconds'] for r in responses)))
                    # Each repeat has its own prefill-produced first token.
                    numerator = sum(len(r['token_ids']) - 1 for r in responses)
                    rates = result['sustained_response_rates']
                    rates['observed_decode_forwards'] = numerator
                    for end in ('lastT', 'DONE'):
                        span = rates[f'observed_firstT_to_{end}_seconds']
                        rates[f'observed_N_minus_1_per_firstT_to_{end}_second'] = numerator / span if span > 0 else None
                result['ok'] = True
            finally:
                if process is not None:
                    result['cleanup'] = cleanup(process)
                    record('cleanup', **result['cleanup'])
                    if result['cleanup']['returncode'] != 0 and result['ok']:
                        raise RuntimeError('Engine did not exit cleanly after QUIT')
                if reader is not None:
                    reader.close()
    except BaseException as exc:
        result.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['total_wall_seconds'] = time.monotonic() - started
        write_json(output / 'result.json', result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--response-repeats', type=int, choices=range(2, 5), default=2)
    parser.add_argument('--timeout', type=float, default=230)
    parser.add_argument('--startup-timeout', type=float, default=75)
    parser.add_argument('--request-timeout', type=float, default=65)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args(argv)
    if not CLEANUP_RESERVE < args.timeout <= 230 or not all(
            math.isfinite(v) and 0 < v <= args.timeout for v in (args.startup_timeout, args.request_timeout)):
        parser.error('Require finite total <=230s with cleanup reserve and positive stage deadlines')
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    started = time.monotonic()
    engine = source_info(ENGINE)
    if engine['sha256'] != ENGINE_SHA256:
        raise RuntimeError('Installed engine SHA256 differs from the Patch0310 pin')
    config = json.loads(CONFIG.read_text())
    command = engine_command(config)
    for key in ('--pack', '--native', '--ple-gguf', '--expert-profile'):
        if not Path(command[command.index(key) + 1]).exists():
            raise FileNotFoundError(f'Missing installed asset: {key}')
    provenance = dict(engine=engine, config=source_info(CONFIG), retained=source_info(RETAINED),
                      engine_help=source_info(HELP), source_reference=source_info(SOURCE),
                      assets={key: command[command.index(key) + 1] for key in
                              ('--pack', '--native', '--ple-gguf', '--expert-profile')})
    return run_harness(command, load_requests(repeats=args.response_repeats), args.output,
                       cwd=config['cwd'], timeout=args.timeout, startup_timeout=args.startup_timeout,
                       request_timeout=args.request_timeout, profile=args.profile,
                       provenance=provenance, started=started)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Patch0310 refused/aborted: {error}', file=sys.stderr)
        sys.exit(125)
