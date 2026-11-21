#!/usr/bin/env python3
"""Capture the exact installed launch, not standard-v1 or configuration parity.

Run ONLY through run_0290_isolated_benchmark.py (reserve >=12 GiB, swap=0;
38 GiB high / 40 GiB hard for the retained exact launch, outer timeout 300s). The resource
limits can change engine decisions or reject its default pinned arena. This
probe never acquires assets, edits settings/weights, or substitutes launch flags.
Imports and unit tests use only the Python standard library, never model code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_0290_isolated_benchmark as isolation
from scripts.probe_0310_strata_comparison import source_info

LAUNCHER = '/home/lucka/.local/bin/strata'
COMMAND = (LAUNCHER, 'launch', '--model', 'iq3_s', '--context', '262144',
           '--parallel', '1', '--vision', 'off', '--tune', 'skip', '--access',
           'local', '--no-api-key', '--port', '8095', '--yes')
STOP = (LAUNCHER, 'launch', '--stop', '--port', '8095', '--yes')
BASE_URL = 'http://127.0.0.1:8095'
RETAINED = ROOT / 'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json'
OUTPUTS = ROOT / 'docs/qwen4exp/diagnostics_0243_off_v1/benchmark/outputs.jsonl'
INSTALL = Path('/home/lucka/.local/share/strata/install')
CONFIG_ROOT = Path('/home/lucka/.config/strata')
GIB = 1024**3
SECRET = re.compile(r'(?i)(api[-_]?key|authorization|password|secret|access[-_]?token|credential)')


def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if SECRET.search(k) and k != '--no-api-key' else redact(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        result, hide = [], False
        for item in value:
            if hide:
                result.append('[REDACTED]')
                hide = False
            elif isinstance(item, str) and item.startswith('--') and SECRET.search(item) and item != '--no-api-key':
                result.append(item.split('=', 1)[0] + ('=[REDACTED]' if '=' in item else ''))
                hide = '=' not in item
            else:
                result.append(redact(item))
        return result
    if isinstance(value, str):
        value = re.sub(r'(?i)(bearer\s+)\S+', r'\1[REDACTED]', value)
        value = re.sub(r'(?i)((?:api[-_]?key|password|secret|access[-_]?token|authorization)'
                       r'[\s\"\x27:=]+)([^\s,;\"\x27]+)', r'\1[REDACTED]', value)
        value = re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1[REDACTED]@', value)
    return value


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(redact(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def journal(path, value):
    with Path(path).open('a', buffering=1) as stream:
        stream.write(json.dumps(redact(value), allow_nan=False) + '\n')


def optional_text(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return ''


def kv_numbers(text):
    values = {}
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields = value.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * (1024 if fields[1:] == ['kB'] else 1)
    return values


def parse_stat(text):
    # comm may contain spaces and parentheses; fields after its LAST ')' are fixed.
    fields = text[text.rindex(')') + 2:].split()
    return dict(ppid=int(fields[1]), minor_faults=int(fields[7]), major_faults=int(fields[9]),
                utime=int(fields[11]), stime=int(fields[12]), starttime=int(fields[19]))


def process_info(pid, proc=Path('/proc')):
    directory = proc / str(pid)
    row = dict(pid=int(pid), **parse_stat((directory / 'stat').read_text()))
    row['cmdline'] = redact((directory / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'))
    row['cgroup'] = optional_text(directory / 'cgroup')
    row['executable'] = os.readlink(directory / 'exe')
    return row


def detailed_process(row, proc=Path('/proc')):
    row = dict(row)
    directory = proc / str(row['pid'])
    status = kv_numbers(optional_text(directory / 'status'))
    row['memory_bytes'] = {key: status.get(key) for key in (
        'VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmLck', 'VmSwap')}
    row['smaps_rollup_bytes'] = kv_numbers(optional_text(directory / 'smaps_rollup'))
    row['io'] = kv_numbers(optional_text(directory / 'io'))
    threads = []
    for task in sorted((directory / 'task').glob('[0-9]*')):
        try:
            stat = parse_stat((task / 'stat').read_text())
            threads.append(dict(tid=int(task.name), utime=stat['utime'], stime=stat['stime'],
                                starttime=stat['starttime'], affinity=sorted(os.sched_getaffinity(int(task.name)))))
        except (OSError, ValueError, ProcessLookupError):
            continue
    row['threads'] = threads
    logs = []
    for fd in (directory / 'fd').glob('*'):
        try:
            target = os.readlink(fd)
            if target.endswith('.log') and Path(target).is_file():
                logs.append(target)
        except OSError:
            pass
    row['log_files'] = sorted(set(logs))
    return row


class OwnedProcesses:
    """PID+birth identities prevent PID reuse and baseline/unrelated kills."""
    def __init__(self, proc=Path('/proc')):
        self.proc = proc
        self.cgroup = isolation.membership((proc / 'self/cgroup').read_text())
        self.directory = isolation.CGROUP_ROOT.joinpath(*self.cgroup.parts[1:])
        self.baseline = {(r['pid'], r['starttime']) for r in self.scan()}
        self.known = set()
        self.lock = threading.Lock()

    def scan(self):
        rows = []
        for entry in self.proc.glob('[0-9]*'):
            try:
                rows.append(process_info(int(entry.name), self.proc))
            except (OSError, ValueError):
                continue
        return rows

    def refresh(self):
        with self.lock:
            rows = self.scan()
            selected = set()
            for row in rows:
                identity = (row['pid'], row['starttime'])
                cg = next((line[3:] for line in row['cgroup'].splitlines() if line.startswith('0::')), '')
                if identity not in self.baseline and (cg == str(self.cgroup) or cg.startswith(str(self.cgroup) + '/')
                                                     or identity in self.known):
                    selected.add(row['pid'])
            # Include actual descendants, even if they entered a nested cgroup.
            while True:
                more = {r['pid'] for r in rows if r['ppid'] in selected
                        and (r['pid'], r['starttime']) not in self.baseline}
                if more <= selected:
                    break
                selected |= more
            result = [r for r in rows if r['pid'] in selected]
            self.known.update((r['pid'], r['starttime']) for r in result)
            return result

    def snapshot(self):
        rows = []
        for row in self.refresh():
            try:
                rows.append(detailed_process(row, self.proc))
            except OSError:
                continue
        mem = kv_numbers(optional_text(self.proc / 'meminfo'))
        return dict(monotonic=time.monotonic(), clk_tck=os.sysconf('SC_CLK_TCK'), processes=rows,
                    meminfo_bytes=mem, dram=dict(
                        used_bytes=mem.get('MemTotal', 0) - mem.get('MemFree', 0) - mem.get('Buffers', 0)
                                   - mem.get('Cached', 0) - mem.get('SReclaimable', 0) + mem.get('Shmem', 0),
                        used_definition='total-free-buffers-cached-SReclaimable+Shmem; not process anonymous RAM',
                        available_bytes=mem.get('MemAvailable'), buffers_bytes=mem.get('Buffers'),
                        cached_bytes=mem.get('Cached'), sreclaimable_bytes=mem.get('SReclaimable')),
                    cgroup={name: optional_text(self.directory / name) for name in (
                        'memory.current', 'memory.peak', 'memory.stat', 'memory.events', 'memory.high',
                        'memory.max', 'memory.swap.current', 'memory.swap.max', 'memory.pressure')})

    def signal(self, sig):
        for row in self.refresh():
            try:
                # Recheck birth immediately before signalling; never signal a recycled PID.
                if parse_stat((self.proc / str(row['pid']) / 'stat').read_text())['starttime'] == row['starttime']:
                    os.kill(row['pid'], sig)
            except (OSError, ValueError):
                pass


def cpu_delta(before, after):
    seconds = after['monotonic'] - before['monotonic']
    old = {(r['pid'], r['starttime']): r for r in before['processes']}
    rows = []
    for row in after['processes']:
        previous = old.get((row['pid'], row['starttime']))
        if previous is None:
            continue
        ticks = row['utime'] + row['stime'] - previous['utime'] - previous['stime']
        old_threads = {(t['tid'], t['starttime']): t for t in previous['threads']}
        thread_deltas = []
        for thread in row['threads']:
            prior = old_threads.get((thread['tid'], thread['starttime']))
            if prior:
                dt = thread['utime'] + thread['stime'] - prior['utime'] - prior['stime']
                thread_deltas.append(dict(tid=thread['tid'], cpu_ticks=dt, affinity=thread['affinity'],
                    active_core_equivalents=dt / after['clk_tck'] / seconds if seconds > 0 else None))
        rows.append(dict(pid=row['pid'], cpu_ticks=ticks,
                         active_core_equivalents=ticks / after['clk_tck'] / seconds if seconds > 0 else None,
                         read_bytes=row['io'].get('read_bytes', 0) - previous['io'].get('read_bytes', 0),
                         major_faults=row['major_faults'] - previous['major_faults'], threads=thread_deltas))
    return dict(wall_seconds=seconds, clk_tck=after['clk_tck'], processes=rows,
                note='Process ticks include exited threads; matched thread deltas need not sum to process ticks. '
                     'New/exited processes are not assigned invented CPU deltas. Chunk counts are not token counts.')


def port_inodes(port=8095, proc=Path('/proc')):
    result = set()
    for name in ('tcp', 'tcp6'):
        for line in optional_text(proc / 'net' / name).splitlines()[1:]:
            fields = line.split()
            if len(fields) > 9 and fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port:
                result.add(fields[9])
    return result


def port_unused():
    if port_inodes():
        raise RuntimeError('Port 8095 already active; refusing without stopping its owner')
    # Also refuse non-listening/reserved sockets; no HTTP request to an existing server.
    with socket.socket() as sock:
        try:
            sock.bind(('0.0.0.0', 8095))
        except OSError as exc:
            raise RuntimeError('Port 8095 unavailable; refusing startup') from exc


def owns_listener(owned):
    inodes = port_inodes(proc=owned.proc)
    if not inodes:
        return False
    found = set()
    for row in owned.refresh():
        for fd in (owned.proc / str(row['pid']) / 'fd').glob('*'):
            try:
                link = os.readlink(fd)
                if link.startswith('socket:['):
                    found.add(link[8:-1])
            except OSError:
                pass
    if not inodes <= found:
        raise RuntimeError('Port 8095 listener is not wholly owned by this run; refusing HTTP/stop command')
    return True


def no_concurrent_model(owned):
    for row in owned.scan():
        if row['pid'] == os.getpid():
            continue
        args = row['cmdline']
        exe = Path(row['executable']).name.lower()
        script = next((Path(a).name for a in args[1:] if isinstance(a, str) and a.endswith('.py')), '')
        model = (exe in ('strata', 'llama-server', 'llama-cli')
                 or script in ('inference.py', 'serve.py', 'train_model.py', 'train_heterogeneous.py')
                 or (script.startswith('probe_') and script != Path(__file__).name))
        if model:
            raise RuntimeError(f'Concurrent model/probe process {row["pid"]} ({exe or script}); refusing startup')


def load_workload(path=None):
    path = Path(path) if path else (OUTPUTS if OUTPUTS.exists() else RETAINED)
    data = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == '.jsonl' else json.loads(path.read_text())['cases'])
    result = []
    for name in ('warmup', 'performance-short-r1'):
        matches = [r for r in data if r['request_id'] == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one retained {name}')
        row = matches[0]
        request = copy.deepcopy(row.get('request', row.get('captured_request')))
        expected = dict(temperature=1.0, top_p=.95, top_k=20, seed=1, min_p=0.0,
                        presence_penalty=0.0, repetition_penalty=1.0,
                        max_tokens=128 if name == 'warmup' else 512,
                        reasoning_effort='xhigh', reasoning_budget_tokens=0,
                        chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=True))
        if not isinstance(request, dict) or any(request.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Retained sampling mismatch: {name}')
        if request.get('frequency_penalty', 0) != 0 or row.get('tools') is not None:
            raise ValueError('Unexpected retained tools/penalties')
        if not row.get('messages') or any(m.get('role') not in ('system', 'user', 'assistant') for m in row['messages']):
            raise ValueError('Missing/invalid retained messages')
        if set(request) - set(expected) - {'stream', 'stream_options', 'frequency_penalty'}:
            raise ValueError('Unexpected retained request fields; refusing to force output or guess')
        request.update(messages=copy.deepcopy(row['messages']), stream=True, stream_options=dict(include_usage=True))
        result.append(request)
    return result, source_info(path)


def provenance(config_root=CONFIG_ROOT, install=INSTALL, launcher=Path(LAUNCHER)):
    result = dict(configs=[], launcher=None,
                  limitations='Resource-limit-bound exact CLI capture, NOT full configuration equivalence. '
                  'Launcher may update its settings; probe does not edit them. No checkpoint hashes/reads.')
    try:
        result['launcher'] = source_info(launcher)
    except OSError as exc:
        result['launcher_error'] = str(exc)
    paths = set(install.glob('*.json')) | set(install.glob('run-*.sh'))
    paths |= set(config_root.rglob('*.json'))
    for path in sorted(paths):
        try:
            if path.stat().st_size > 2 * 1024**2:
                result['configs'].append(dict(path=str(path), error='config exceeds 2MiB capture limit'))
                continue
            text = path.read_text()
            try:
                content = json.loads(text)
            except ValueError:
                content = text
            result['configs'].append(dict(**source_info(path), content=redact(content)))
        except (OSError, UnicodeError) as exc:
            result['configs'].append(dict(path=str(path), error=str(exc)))
    return result


class HTTP:
    def __init__(self, base=BASE_URL):
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.response = None

    def open(self, path, payload=None, timeout=1):
        request = urllib.request.Request(self.base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream' if payload else 'application/json'})
        return self.opener.open(request, timeout=timeout)

    def get(self, path, timeout=1):
        started = time.monotonic()
        try:
            with self.open(path, timeout=timeout) as response:
                body = response.read(2 * 1024**2 + 1)
                return http_record(path, response.status, body, started)
        except urllib.error.HTTPError as exc:
            with exc:
                return http_record(path, exc.code, exc.read(2 * 1024**2 + 1), started)
        except (OSError, urllib.error.URLError) as exc:
            return dict(path=path, status=None, error=redact(str(exc)), seconds=time.monotonic() - started)

    def cancel(self):
        # shutdown wakes an in-flight socket read; close alone can wait for its lock.
        response = self.response
        if response is not None:
            try:
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass


def http_record(path, status, body, started):
    text = body[:2 * 1024**2].decode(errors='replace')
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    return dict(path=path, status=status, json=redact(parsed), raw=redact(text),
                truncated=len(body) > 2 * 1024**2, seconds=time.monotonic() - started)


class SSE:
    """Incremental SSE framing, including CRLF, comments, multiline data and UTF-8 splits."""
    def __init__(self):
        self.buffer = b''
        self.data = []

    def feed(self, block, eof=False):
        self.buffer += block
        events = []
        while b'\n' in self.buffer or (eof and self.buffer):
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
            else:
                line, self.buffer = self.buffer, b''
            line = line.rstrip(b'\r')
            if not line:
                if self.data:
                    events.append(b'\n'.join(self.data).decode('utf-8'))
                    self.data.clear()
            elif line.startswith(b'data:'):
                self.data.append(line[5:].removeprefix(b' '))
        if eof and self.data:
            events.append(b'\n'.join(self.data).decode('utf-8'))
            self.data.clear()
        if len(self.buffer) + sum(map(len, self.data)) > 4 * 1024**2:
            raise RuntimeError('Oversized SSE event')
        return events


def consume_stream(http, payload, output, clock=time.monotonic):
    started = clock()
    result = dict(message=dict(role='assistant', content=''), usage=None, timings=None,
                  first_content_seconds=None, first_reasoning_seconds=None, done=False, events=[],
                  timing_note='Client event timestamps are chunk arrivals, NOT token intervals or token counts.')
    try:
        try:
            response = http.open('/v1/chat/completions', payload, timeout=60)
        except urllib.error.HTTPError as exc:
            with exc:
                result['http_error'] = http_record('/v1/chat/completions', exc.code, exc.read(2 * 1024**2), started)
            raise RuntimeError('Streaming HTTP request rejected; no nonstream fallback') from exc
        http.response = response
        with response:
            result['http_status'] = response.status
            if 'text/event-stream' not in response.headers.get('Content-Type', '').lower():
                result['unsupported_response'] = response.read(2 * 1024**2).decode(errors='replace')
                raise RuntimeError('API did not return SSE; no nonstream fallback')
            parser = SSE()
            total = 0
            while not result['done']:
                block = response.read1(65536)
                total += len(block)
                if total > 16 * 1024**2:
                    raise RuntimeError('Streaming response exceeds 16MiB')
                for event in parser.feed(block, eof=not block):
                    elapsed = clock() - started
                    journal(output / 'stream.jsonl', dict(seconds=elapsed, data=event))
                    if event.strip() == '[DONE]':
                        result['done'] = True
                        result['done_seconds'] = elapsed
                        break
                    data = json.loads(event)
                    result['events'].append(dict(seconds=elapsed, data=data))
                    if data.get('error'):
                        raise RuntimeError(f'API stream error: {redact(data["error"])}')
                    if data.get('usage') is not None:
                        result['usage'] = data['usage']
                    if data.get('timings') is not None:
                        result['timings'] = data['timings']
                    for choice in data.get('choices', []):
                        if choice.get('index', 0) != 0:
                            raise RuntimeError('Unexpected multiple stream choices')
                        delta = choice.get('delta') or {}
                        for key, value in delta.items():
                            if key in ('content', 'reasoning_content', 'reasoning') and value is not None:
                                if not isinstance(value, str):
                                    raise RuntimeError('Unsupported non-text assistant delta')
                                result['message'][key] = result['message'].get(key, '') + value
                                first = 'first_content_seconds' if key == 'content' else 'first_reasoning_seconds'
                                if value and result[first] is None:
                                    result[first] = elapsed
                            elif key == 'role':
                                result['message']['role'] = value
                            elif value is not None:
                                raise RuntimeError(f'Unsupported assistant delta {key}; refusing lossy followup')
                        if choice.get('finish_reason') is not None:
                            result['finish_reason'] = choice['finish_reason']
                if not block:
                    break
            if not result['done']:
                raise RuntimeError('SSE ended without [DONE]')
            if not isinstance(result['usage'], dict) or not isinstance(result['usage'].get('completion_tokens'), int):
                raise RuntimeError('Missing final usage; chunk counts cannot replace token counts')
            if result.get('finish_reason') not in ('stop', 'length'):
                raise RuntimeError('Missing/unsupported finish reason')
            result['client_completion_tokens_per_second'] = result['usage']['completion_tokens'] / max(clock() - started, 1e-9)
            return result
    except BaseException as exc:
        result['error'] = redact(str(exc))
        raise
    finally:
        http.response = None
        result['seconds'] = clock() - started
        write_json(output / 'response.json', result)


def bounded(call, deadline, check=lambda: None, cancel=lambda: None, clock=time.monotonic):
    results = queue.Queue()
    def work():
        try:
            results.put((True, call()))
        except BaseException as exc:
            results.put((False, exc))
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            check()
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError('Wall-clock operation deadline reached')
            try:
                ok, value = results.get(timeout=min(.1, remaining))
                if ok:
                    return value
                raise value
            except queue.Empty:
                pass
    except BaseException:
        cancel()
        thread.join(timeout=.2)
        raise


class Monitor:
    """Parent progress is independent of CLI exit, blocked HTTP and GPU sampling."""
    def __init__(self, owned, output, started, deadline, clock=time.monotonic):
        self.owned, self.output, self.started, self.deadline, self.clock = owned, output, started, deadline, clock
        self.stopped = threading.Event()
        self.failure = None
        self.phase = 'preflight'
        self.last_line = '(no newline yet)'
        self.pending = b''
        self.offset = 0
        self.hwm = 0
        self.binary_cache = {}
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.gpu_thread = threading.Thread(target=self.gpu, daemon=True)
        self.gpu_binary = shutil.which('nvidia-smi')

    def start(self):
        self.thread.start()
        self.gpu_thread.start()

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.clock() >= self.deadline:
            raise TimeoutError('280s total workload deadline reached')

    def progress(self):
        print(f'[0324] {self.phase} elapsed={self.clock()-self.started:.1f}s '
              f'cgroup_hwm={self.hwm/GIB:.2f}GiB last_newline={redact(self.last_line)[-360:]}', flush=True)

    def sample(self):
        snapshot = self.owned.snapshot()
        journal(self.output / 'processes.jsonl', snapshot)
        for row in snapshot['processes']:
            exe = row['executable']
            if exe not in self.binary_cache:
                # /proc/exe is an actual executable, never hash a checkpoint argument.
                try:
                    info = source_info(self.owned.proc / str(row['pid']) / 'exe')
                    self.binary_cache[exe] = info
                except OSError as exc:
                    self.binary_cache[exe] = dict(error=str(exc))
                journal(self.output / 'executables.jsonl', dict(executable=exe, command=row['cmdline'],
                                                               identity=self.binary_cache[exe]))
        self.hwm = max(self.hwm, int(snapshot['cgroup'].get('memory.peak') or 0))
        guard = isolation.snapshot(self.owned.directory)
        reason = isolation.stop_reason(guard, 12 * GIB)
        journal(self.output / 'guard.jsonl', guard)
        if int(snapshot['cgroup'].get('memory.swap.current') or 0):
            reason = reason or 'Nonzero cgroup swap usage'
        if guard['memory_current'] >= int(snapshot['cgroup']['memory.high']):
            reason = reason or 'MemoryHigh reached; default arena may not fit resource limits'
        if reason:
            raise RuntimeError(reason)
        path = self.output / 'launch.log'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(self.offset)
                block = stream.read(256 * 1024)
                self.offset += len(block)
            lines = (self.pending + block).split(b'\n')
            self.pending = lines.pop()[-8192:]
            if lines:
                self.last_line = lines[-1].decode(errors='replace').strip()

    def run(self):
        next_progress = self.clock()
        try:
            while not self.stopped.is_set():
                self.sample()
                if self.clock() >= next_progress:
                    self.progress()
                    next_progress = self.clock() + 5
                self.stopped.wait(.5)
        except BaseException as exc:
            self.failure = str(exc)
            self.progress()

    def gpu(self):
        if not self.gpu_binary:
            journal(self.output / 'gpu.jsonl', dict(available=False))
            return
        while not self.stopped.is_set():
            started = self.clock()
            try:
                process = subprocess.run([self.gpu_binary,
                    '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=.5)
                row = dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
            except (OSError, subprocess.SubprocessError) as exc:
                row = dict(error=str(exc))
            journal(self.output / 'gpu.jsonl', dict(monotonic=started, **row))
            self.stopped.wait(max(0, 1 - (self.clock() - started)))

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=1)
        self.gpu_thread.join(timeout=1)


def capture_endpoints(http, output, phase, monitor):
    rows = {}
    for endpoint in ('health', 'props', 'metrics', 'status', 'slots', 'v1/models'):
        rows[endpoint] = bounded(lambda e=endpoint: http.get('/' + e),
                                 min(monitor.deadline, time.monotonic() + 2), monitor.check)
        write_json(output / f'{endpoint.replace("/", "_")}_{phase}.json', rows[endpoint])
    for endpoint in ('health', 'props'):
        if rows[endpoint]['status'] != 200 or not isinstance(rows[endpoint].get('json'), dict):
            raise RuntimeError(f'Required /{endpoint} unavailable ({phase})')
    return rows


def startup(http, owned, monitor, deadline):
    while time.monotonic() < deadline:
        monitor.check()
        if owns_listener(owned):
            row = bounded(lambda: http.get('/health', timeout=.5), min(deadline, time.monotonic() + 1), monitor.check)
            journal(monitor.output / 'health_poll.jsonl', row)
            if row['status'] == 200 and isinstance(row.get('json'), dict) and row['json'].get('status') not in ('loading', 'error'):
                return
        time.sleep(.1)
    raise TimeoutError('120s startup deadline: endpoint never ready; inspect launch.log/startup.log and process snapshots. '
                       'Optional default drafting acquisition may be a blocker; no assets acquired separately.')


def capture_native_logs(owned, output):
    paths = set()
    for row in owned.snapshot()['processes']:
        paths.update(row['log_files'])
    for index, path in enumerate(sorted(paths)):
        try:
            if Path(path).resolve() == (output / 'launch.log').resolve():
                continue
            with open(path, 'rb') as stream:
                size = os.fstat(stream.fileno()).st_size
                head = stream.read(256 * 1024)
                stream.seek(max(len(head), size - 256 * 1024))
                tail = stream.read(256 * 1024)
            write_json(output / f'native_log_{time.monotonic_ns()}_{index}.json', dict(path=path, size_bytes=size,
                       head=redact(head.decode(errors='replace')), tail=redact(tail.decode(errors='replace')),
                       truncated=size > 512 * 1024))
        except OSError as exc:
            journal(output / 'events.jsonl', dict(event='native_log_error', path=path, error=str(exc)))


def cleanup(process, owned, output, stop_supported, http, clock=time.monotonic):
    report = dict(stop_attempted=False)
    try:
        if owns_listener(owned) and stop_supported:
            report['stop_attempted'] = True
            try:
                response = subprocess.run(STOP, capture_output=True, text=True, timeout=10,
                                          env=dict(os.environ, PYTHONUNBUFFERED='1'))
                report.update(stop_returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
            except (OSError, subprocess.SubprocessError) as exc:
                report['stop_error'] = str(exc)
    except BaseException as exc:
        report['ownership_error'] = str(exc)
    finally:
        http.cancel()
        # Known cgroup children include daemonized CLI descendants. Never signal baseline PIDs.
        owned.signal(signal.SIGTERM)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        end = clock() + 2
        while owned.refresh() and clock() < end:
            time.sleep(.05)
        owned.signal(signal.SIGKILL)
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                report['launcher_wait_timeout'] = True
        report['remaining_owned'] = owned.refresh()
        report['containment'] = '0290 parent/systemd KillMode=control-group is the final cgroup-wide backstop'
        write_json(output / 'cleanup.json', report)
    return report


def run(output, retained=None):
    # This must precede all installed CLI/provenance inspection and any subprocess.
    isolation.require_isolation()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 280
    process = None
    owned = None
    monitor = None
    http = HTTP()
    stop_supported = False
    result = dict(ok=False, suite='0324-natural-conversation-NOT-standard-v1-full-suite', command=COMMAND,
                  resource_limit_bound=True, reported_75_tokens_per_second='unverified; never inferred from chunks',
                  environment_override={'PYTHONUNBUFFERED': '1'})
    old_handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'Interrupted by signal {signum}')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        port_unused()
        owned = OwnedProcesses()
        no_concurrent_model(owned)
        safety = isolation.snapshot(owned.directory)
        reason = isolation.stop_reason(safety, 12 * GIB)
        if reason:
            raise RuntimeError(reason)
        write_json(output / 'limits_before.json', dict(**safety, cgroup=str(owned.cgroup),
                   memory_high=optional_text(owned.directory / 'memory.high'),
                   memory_max=optional_text(owned.directory / 'memory.max'),
                   swap_max=optional_text(owned.directory / 'memory.swap.max')))
        requests, retained_info = load_workload(retained)
        write_json(output / 'workload_source.json', retained_info)
        monitor = Monitor(owned, output, started, deadline)
        monitor.start()
        write_json(output / 'provenance_before.json', provenance())
        # Help is bounded and starts no engine; only used to verify the exact stop flags.
        try:
            help_result = subprocess.run([LAUNCHER, 'launch', '--help'], timeout=2,
                                         capture_output=True, text=True,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1'))
            text = help_result.stdout + help_result.stderr
            write_json(output / 'launch_help.json', dict(returncode=help_result.returncode, text=text))
            stop_supported = help_result.returncode == 0 and all(
                re.search(r'(?<![\w-])' + flag + r'(?![\w-])', text) for flag in ('--stop', '--port', '--yes'))
        except (OSError, subprocess.SubprocessError) as exc:
            journal(output / 'events.jsonl', dict(event='help_unavailable', error=str(exc)))
        monitor.check()
        port_unused()
        with (output / 'launch.log').open('xb', buffering=0) as launch_log:
            os.link(output / 'launch.log', output / 'startup.log')
            monitor.phase = 'startup'
            launch_started = time.monotonic()
            process = subprocess.Popen(COMMAND, stdout=launch_log, stderr=launch_log,
                                       stdin=subprocess.DEVNULL, start_new_session=True,
                                       env=dict(os.environ, PYTHONUNBUFFERED='1'))
            startup(http, owned, monitor, min(deadline, launch_started + 120))
            result['startup_seconds'] = time.monotonic() - launch_started
            # CLI may have returned while the cgroup-owned engine binds/serves.
            result['launcher_returncode_at_ready'] = process.poll()
            monitor.phase = 'capture-before'
            endpoints = capture_endpoints(http, output, 'before', monitor)
            models = endpoints['v1/models'].get('json')
            model_id = 'strata'
            if isinstance(models, dict) and models.get('data'):
                model_id = models['data'][0].get('id') or model_id
            capture_native_logs(owned, output)
            messages = copy.deepcopy(requests[1]['messages'])
            followups = (
                'Give a concrete collision example, showing every lookup step and the equality checks.',
                'Implement that collision strategy in compact Python and add tests for collisions and a missing key.')
            result['requests'] = []
            for index in range(4):
                monitor.check()
                if time.monotonic() + 1 >= deadline:
                    raise TimeoutError('Insufficient total budget for next request')
                if not owns_listener(owned):
                    raise RuntimeError('Owned server disappeared before request')
                name = 'warmup' if index == 0 else f'conversation-{index}'
                monitor.phase = name
                request_dir = output / name
                request_dir.mkdir()
                payload = copy.deepcopy(requests[0 if index == 0 else 1])
                if index:
                    if index > 1:
                        messages.append(dict(role='user', content=followups[index - 2]))
                    payload['messages'] = copy.deepcopy(messages)
                payload['model'] = model_id
                write_json(request_dir / 'request.json', payload)
                before = owned.snapshot()
                write_json(request_dir / 'process_before.json', before)
                request_started = time.monotonic()
                request_result = None
                try:
                    request_result = bounded(lambda: consume_stream(http, payload, request_dir),
                        min(deadline, request_started + 60), monitor.check, http.cancel)
                finally:
                    after = owned.snapshot()
                    write_json(request_dir / 'process_after.json', after)
                    stats = dict(request_id=name, monotonic_start=request_started,
                                 seconds=time.monotonic() - request_started, cpu=cpu_delta(before, after))
                    write_json(request_dir / 'stats.json', stats)
                    journal(output / 'requests.jsonl', stats)
                result['requests'].append(dict(request_id=name, usage=request_result['usage'],
                    timings=request_result['timings'], seconds=request_result['seconds']))
                if index:
                    messages.append(copy.deepcopy(request_result['message']))
            monitor.phase = 'capture-after'
            capture_endpoints(http, output, 'after', monitor)
            result['ok'] = True
    except BaseException as exc:
        result['error'] = redact(f'{type(exc).__name__}: {exc}')
        print(f'[0324] blocked/failed: {result["error"]}', flush=True)
    finally:
        # Ignore repeated interrupts only during bounded owned-resource retirement.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        # The parent may kill the group before slow launcher cleanup returns.
        # Persist the execution outcome first; this is NOT proof of retirement.
        write_json(output / 'outcome_before_cleanup.json', dict(
            **result, finalized=False, cleanup_status='pending',
            phase=monitor.phase if monitor else 'preflight',
            seconds_before_cleanup=time.monotonic() - started))
        if monitor:
            monitor.phase = 'cleanup'
        try:
            if owned is not None:
                try:
                    write_json(output / 'process_final.json', owned.snapshot())
                    capture_native_logs(owned, output)
                except BaseException as exc:
                    result['final_snapshot_error'] = str(exc)
                if process is not None or owned.known:
                    report = cleanup(process, owned, output, stop_supported, http)
                    if report['remaining_owned']:
                        result.update(ok=False, cleanup_error='Owned processes remain; parent cgroup stop required')
        except BaseException as exc:
            result.update(ok=False, cleanup_error=str(exc))
        finally:
            if monitor:
                monitor.stop()
            try:
                write_json(output / 'provenance_after.json', provenance())
            except BaseException as exc:
                result['provenance_after_error'] = str(exc)
            result['seconds'] = time.monotonic() - started
            write_json(output / 'result.json', result)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print(f'[0324] {"complete" if result["ok"] else "FAILED"} '
              f'elapsed={result["seconds"]:.1f}s artifacts={output.resolve()}', flush=True)
    return 0 if result['ok'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New exclusive capture directory (not the parent guard directory)')
    parser.add_argument('--retained', type=Path, help='Retained outputs.jsonl or 0284 result.json; never new prompts/assets')
    args = parser.parse_args(argv)
    return run(args.output, args.retained)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f'[0324] refused: {error}', file=sys.stderr, flush=True)
        sys.exit(125)
