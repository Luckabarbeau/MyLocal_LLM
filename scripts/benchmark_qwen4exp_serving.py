#!/usr/bin/env python3
"""HTTP-only Strata/MyLocal benchmark; never loads model weights or resets a server.

Default sizing uses llama.cpp-compatible /apply-template then /tokenize. If these
routes are unavailable, use --gguf for the repository's CPU tokenizer/template.
Only backend usage is reported as actual served token counts. Sizing counts are
separate and template/tokenizer agreement with the server is not assumed.
Artifacts contain source prompts, responses and raw telemetry: treat as private.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


DEFAULT_SIZES = '128,1K,4K,16K,32K,64K,128K,near262K'
TELEMETRY_PATHS = ('/health', '/v1/models', '/props', '/metrics', '/slots')


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def journal(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class HTTP:
    """Proxy-free urllib transport, adapted from probe_0324_strata_exact_launch."""
    def __init__(self, base_url, timeout=600, api_key=None):
        self.base_url = base_url.rstrip('/').removesuffix('/v1')
        self.timeout = timeout
        self.api_key = api_key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, path, body=None, stream=False):
        headers = {'Content-Type': 'application/json',
                   'Accept': 'text/event-stream' if stream else 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers)
        return self.opener.open(request, timeout=self.timeout)

    def probe(self, path):
        start = time.monotonic()
        result = dict(path=path, status=None, raw=None, json=None, error=None)
        try:
            try:
                response = self.open(path)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                result['status'] = response.code
                result['headers'] = dict(response.headers)
                raw = response.read(8 * 1024**2 + 1)
                result['truncated'] = len(raw) > 8 * 1024**2
                result['raw'] = raw[:8 * 1024**2].decode('utf-8', errors='replace')
                try:
                    result['json'] = json.loads(result['raw'])
                except ValueError:
                    pass
        except (OSError, urllib.error.URLError) as exc:
            result['error'] = str(exc)
        result['seconds'] = time.monotonic() - start
        return result

    def post_json(self, path, payload):
        with self.open(path, json.dumps(payload).encode('utf-8')) as response:
            data = json.load(response)
        if isinstance(data, dict) and data.get('error'):
            raise ValueError(f'{path}: {data["error"]}')
        return data

    def snapshot(self, paths=TELEMETRY_PATHS):
        return {path: self.probe(path) for path in paths}


class SSE:
    """Byte framing reused from the existing probe; CRLF/multiline/UTF-8 safe."""
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
            raise ValueError('Incomplete SSE frame at EOF')
        if len(self.buffer) + sum(map(len, self.data)) > 4 * 1024**2:
            raise ValueError('Oversized SSE event')
        return events


def token_telemetry(events, usage):
    """Explicit named fields only. Missing cache telemetry is NOT zero reuse."""
    result = dict(prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
                  processed_prompt_tokens=None, reused_prompt_tokens=None,
                  decode_forward_tokens=None)
    if isinstance(usage, dict):
        for key in ('prompt_tokens', 'completion_tokens'):
            result[key] = usage.get(key)
        result['reasoning_tokens'] = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        result['reused_prompt_tokens'] = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    for event in events:
        for obj in (event, event.get('timings') or {}, event.get('metrics') or {},
                    event.get('stats') or {}, event.get('mylocal_stats') or {}):
            for dest, keys in (
                ('processed_prompt_tokens', ('new_prompt_tokens', 'processed_prompt_tokens', 'prompt_n')),
                ('reused_prompt_tokens', ('reused_tokens', 'reused_prompt_tokens', 'cache_n')),
                ('decode_forward_tokens', ('decode_forward_tokens',)),
            ):
                for key in keys:
                    if obj.get(key) is not None:
                        result[dest] = obj[key]
        stats = event.get('mylocal_stats') or {}
        for dest, key in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'generated_tokens')):
            if result[dest] is None and stats.get(key) is not None:
                result[dest] = stats[key]
    return result


def server_telemetry(events, usage, before=None, after=None):
    """Normalize explicit server measurements, never SSE event counts or global deltas.

    Response telemetry is request-local. A /metrics request record is usable only
    with an exact ID match; snapshots of hardware/expert caches are global samples.
    """
    def metrics(snapshot):
        record = (snapshot or {}).get('/metrics') or {}
        data = record.get('json')
        return data if record.get('status') == 200 and isinstance(data, dict) else {}

    before_metrics, after_metrics = metrics(before), metrics(after)
    ids = set()
    for event in events:
        for obj in (event, event.get('mylocal_stats') or {}):
            for key in ('id', 'request_id'):
                if isinstance(obj.get(key), str):
                    ids.add(obj[key])
    supplemental = []
    matched = None
    stats = after_metrics.get('mylocal_stats') or {}
    if stats.get('request_id') in ids:
        supplemental.append({'mylocal_stats': stats})
    for row in after_metrics.get('requests') or []:
        if row.get('request_id', row.get('id')) in ids:
            matched = row
            supplemental.append(dict(
                prompt_tokens=row.get('prompt_tokens'),
                new_prompt_tokens=row.get('prompt_read'), reused_tokens=row.get('reused'),
                timings=dict(prompt_ms=row.get('prompt_ms'), predicted_ms=row.get('decode_ms'),
                             predicted_per_second=row.get('decode_tok_s'))))
            break
    # Direct response measurements take precedence over matching metrics snapshots.
    records = supplemental + list(events)
    tokens = token_telemetry(records, usage)
    if matched:
        for dest, key in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'output_tokens')):
            if tokens[dest] is None:
                tokens[dest] = matched.get(key)
    phases = dict(prefill_seconds=None, decode_seconds=None, total_seconds=None,
                  first_piece_seconds=None, first_engine_token_seconds=None,
                  prefill_tokens_per_second=None, decode_tokens_per_second=None,
                  decode_forwards_per_second=None)
    mylocal_stats = None
    for event in records:
        timings = event.get('timings') or {}
        for dest, key, factor in (
            ('prefill_seconds', 'prompt_ms', 0.001),
            ('decode_seconds', 'predicted_ms', 0.001),
            ('prefill_tokens_per_second', 'prompt_per_second', 1),
            ('decode_tokens_per_second', 'predicted_per_second', 1),
        ):
            if timings.get(key) is not None:
                phases[dest] = timings[key] * factor
        stats = event.get('mylocal_stats')
        if isinstance(stats, dict):
            mylocal_stats = {**(mylocal_stats or {}), **stats}
            for key in ('prefill_seconds', 'decode_seconds', 'total_seconds',
                        'first_piece_seconds', 'decode_forwards_per_second'):
                if stats.get(key) is not None:
                    phases[key] = stats[key]
    if matched:
        phases['total_seconds'] = matched.get('duration_s')
        phases['first_engine_token_seconds'] = matched.get('first_engine_token_s')
    def sample(data, keys):
        values = {key: data[key] for key in keys if data.get(key) is not None}
        return values or None
    hardware = {name: sample(data, ('hardware', 'hardware_static')) for name, data in (
        ('before', before_metrics), ('after', after_metrics))}
    cache_snapshots = {name: sample(data, ('conversation_cache', 'experts')) for name, data in (
        ('before', before_metrics), ('after', after_metrics))}
    return dict(tokens=tokens, phases=phases, mylocal_stats=mylocal_stats,
                cache=(mylocal_stats or {}).get('cache'),
                metrics_request=matched,
                hardware=hardware if any(hardware.values()) else None,
                cache_snapshots=cache_snapshots if any(cache_snapshots.values()) else None)


class ResourceSampler:
    """Opt-in Linux sampler. Worker alone owns JSONL writes; stop joins it fully."""
    def __init__(self, directory, pid, gpu_uuid=None, proc_root='/proc', interval=1.0):
        self.path = Path(directory) / 'resources.jsonl'
        self.pid, self.gpu_uuid = pid, gpu_uuid
        self.proc_root, self.interval = Path(proc_root), interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = time.monotonic()
        self.previous = None
        self.process_start_ticks = None
        self.ticks_per_second = os.sysconf('SC_CLK_TCK')
        self.page_bytes = os.sysconf('SC_PAGE_SIZE')
        self.summary = dict(sample_count=0, metrics={}, error=None, pid=pid, gpu_uuid=gpu_uuid,
                            interval_seconds=interval, labels={
                                'process': 'Selected PID only; CPU percent uses one logical core=100%, may exceed 100%.',
                                'global_cpu': 'Whole host, all processes; interval busy percent.',
                                'global_memory': 'Whole host; used_bytes = total_bytes - available_bytes.',
                                'gpu': 'Whole selected GPU, all processes; not PID-attributed.'})

    def _nvidia(self, fields, selector, raw):
        command = ['nvidia-smi', '--id=' + selector, '--query-gpu=' + fields,
                   '--format=csv,noheader,nounits']
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2,
                                       check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raw.append(dict(command=command, error=str(exc)))
            raise ValueError(f'nvidia-smi: {exc}') from exc
        raw.append(dict(command=command, stdout=completed.stdout, stderr=completed.stderr,
                        returncode=completed.returncode))
        if completed.returncode:
            raise ValueError(f'nvidia-smi exited {completed.returncode}')
        return completed.stdout.strip()

    def collect(self):
        now = time.monotonic()
        sample = dict(seconds=now - self.started, timestamp=time.time(), pid=self.pid,
                      process=dict(cpu_percent_one_core=None, rss_bytes=None, virtual_bytes=None),
                      global_cpu=dict(busy_percent=None),
                      global_memory=dict(total_bytes=None, available_bytes=None, used_bytes=None),
                      gpu=dict(uuid=self.gpu_uuid, utilization_percent=None,
                               memory_utilization_percent=None, memory_used_bytes=None,
                               memory_total_bytes=None, power_watts=None, temperature_c=None),
                      raw={}, errors=[])
        raw = sample['raw']
        process_ticks = cpu_total = cpu_idle = None
        try:
            raw['process_stat'] = (self.proc_root / str(self.pid) / 'stat').read_text()
            fields = raw['process_stat'].rsplit(')', 1)[1].split()
            start_ticks = int(fields[19])
            if self.process_start_ticks is None:
                self.process_start_ticks = start_ticks
            if self.process_start_ticks != start_ticks:
                raise ValueError('PID reused: process start time changed; refusing attribution')
            process_ticks = int(fields[11]) + int(fields[12])
            sample['process'].update(rss_bytes=int(fields[21]) * self.page_bytes,
                                     virtual_bytes=int(fields[20]))
        except (OSError, ValueError, IndexError) as exc:
            sample['errors'].append(f'process: {exc}')
        try:
            raw['system_stat'] = (self.proc_root / 'stat').read_text()
            values = [int(value) for value in raw['system_stat'].splitlines()[0].split()[1:9]]
            cpu_total, cpu_idle = sum(values), values[3] + values[4]
        except (OSError, ValueError, IndexError) as exc:
            sample['errors'].append(f'global_cpu: {exc}')
        try:
            raw['meminfo'] = (self.proc_root / 'meminfo').read_text()
            mem = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
                   for line in raw['meminfo'].splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
            total, available = mem.get('MemTotal'), mem.get('MemAvailable')
            sample['global_memory'].update(total_bytes=total, available_bytes=available,
                used_bytes=total - available if total is not None and available is not None else None)
        except (OSError, ValueError, IndexError) as exc:
            sample['errors'].append(f'global_memory: {exc}')
        if self.previous is not None:
            previous_time, previous_process, previous_total, previous_idle = self.previous
            elapsed = now - previous_time
            if (process_ticks is not None and previous_process is not None and elapsed > 0
                    and process_ticks >= previous_process):
                sample['process']['cpu_percent_one_core'] = (
                    100 * (process_ticks - previous_process) / self.ticks_per_second / elapsed)
            if cpu_total is not None and previous_total is not None and cpu_total > previous_total:
                sample['global_cpu']['busy_percent'] = 100 * (
                    1 - (cpu_idle - previous_idle) / (cpu_total - previous_total))
        self.previous = now, process_ticks, cpu_total, cpu_idle
        raw['nvidia_smi'] = []
        try:
            if self.gpu_uuid is None:
                candidate = self._nvidia('uuid', '0', raw['nvidia_smi'])
                if not candidate.startswith('GPU-') or any(char.isspace() for char in candidate) or ',' in candidate:
                    raise ValueError('GPU 0 did not resolve to one explicit GPU UUID')
                self.gpu_uuid = candidate
            sample['gpu']['uuid'] = self.gpu_uuid
            values = self._nvidia('uuid,utilization.gpu,utilization.memory,memory.used,memory.total,'
                                  'power.draw,temperature.gpu', self.gpu_uuid, raw['nvidia_smi']).split(',')
            if len(values) != 7 or values[0].strip() != self.gpu_uuid:
                raise ValueError('GPU sample UUID/field mismatch')
            for key, value, scale in zip(
                ('utilization_percent', 'memory_utilization_percent', 'memory_used_bytes',
                 'memory_total_bytes', 'power_watts', 'temperature_c'), values[1:],
                (1, 1, 1024**2, 1024**2, 1, 1)):
                try:
                    number = float(value.strip())
                    sample['gpu'][key] = number * scale if math.isfinite(number) else None
                except ValueError:
                    pass  # Unsupported driver measurements remain null; raw output is retained.
        except ValueError as exc:
            sample['errors'].append(str(exc))
        return sample

    def _run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                sample = self.collect()
                journal(self.path, sample)
                self.summary['sample_count'] += 1
                self.summary['gpu_uuid'] = self.gpu_uuid
                for group in ('process', 'global_cpu', 'global_memory', 'gpu'):
                    for key, value in sample[group].items():
                        if key == 'uuid':
                            continue
                        bounds = self.summary['metrics'].setdefault(group + '.' + key, dict(min=None, max=None))
                        if isinstance(value, (int, float)):
                            bounds['min'] = value if bounds['min'] is None else min(bounds['min'], value)
                            bounds['max'] = value if bounds['max'] is None else max(bounds['max'], value)
                self.stop_event.wait(max(0, self.interval - (time.monotonic() - started)))
        except Exception as exc:
            self.summary['error'] = str(exc)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        # No file is shared with the caller. Joining prevents late writes after return.
        self.thread.join()
        return self.summary


def benchmark_request(http, payload, directory, metadata=None, telemetry_paths=TELEMETRY_PATHS,
                      resource_pid=None, resource_gpu_uuid=None):
    """One request. Failures return records too; raw chunks/events survive interruption."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    # These bytes, not a re-serialization, are sent on the wire.
    with (directory / 'request.json').open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    save_json(directory / 'metadata.json', metadata or {})
    before = http.snapshot(telemetry_paths)
    save_json(directory / 'before.json', before)
    result = dict(ok=False, error=None, http_status=None, usage=None, timings=None,
                  message={'role': 'assistant', 'content': ''}, finish_reason=None,
                  done=False, first_content_seconds=None, first_reasoning_seconds=None,
                  ttft_seconds=None, seconds=None, response_headers=None,
                  timing_note='Client arrivals are durations, not tokens per SSE event. '
                              'Nonstream TTFT is unavailable; response arrival is not first-token time.',
                  cache=None, hardware=None)
    events = []
    sampler = (ResourceSampler(directory, resource_pid, resource_gpu_uuid)
               if resource_pid is not None else None)
    start = time.monotonic()
    def accept(data, elapsed, streaming):
        if not isinstance(data, dict):
            raise ValueError('Expected JSON object')
        events.append(data)
        if data.get('error') is not None:
            raise ValueError(f'Backend error: {data["error"]}')
        if data.get('usage') is not None:
            # Some servers publish usage in more than one chunk.
            result['usage'] = {**(result['usage'] or {}), **data['usage']}
        if data.get('timings') is not None:
            result['timings'] = data['timings']
        for choice in data.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Multiple choices not supported')
            if choice.get('finish_reason') is not None:
                result['finish_reason'] = choice['finish_reason']
            message = choice.get('delta' if streaming else 'message') or {}
            for key, value in message.items():
                if key in ('content', 'reasoning_content', 'reasoning'):
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        raise ValueError('Non-text assistant response')
                    result['message'][key] = result['message'].get(key, '') + value
                    if streaming and value:
                        field = 'first_content_seconds' if key == 'content' else 'first_reasoning_seconds'
                        if result[field] is None:
                            result[field] = elapsed
                elif key == 'role':
                    if value != 'assistant':
                        raise ValueError('Unexpected response role')
                elif value is not None:
                    raise ValueError(f'Unsupported assistant field {key}; refusing lossy followup')
    try:
        if sampler is not None:
            sampler.start()
        try:
            response = http.open('/v1/chat/completions', body, payload.get('stream', False))
        except urllib.error.HTTPError as exc:
            with exc:
                result['http_status'] = exc.code
                raw = exc.read()
                (directory / 'response.body').write_bytes(raw)
            raise ValueError(f'HTTP {exc.code}: {raw.decode(errors="replace")}') from exc
        with response, (directory / 'response.body').open('xb') as raw_stream:
            result['http_status'] = response.status
            result['response_headers'] = dict(response.headers)
            streaming = payload.get('stream', False)
            if streaming and 'text/event-stream' not in response.headers.get('Content-Type', '').lower():
                raise ValueError('Expected text/event-stream; no silent fallback')
            parser = SSE()
            total = 0
            while True:
                # Socket timeout bounds idle reads; deadline also bounds busy streams.
                remaining = http.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Request deadline exceeded')
                try:
                    response.fp.raw._sock.settimeout(remaining)
                except AttributeError:
                    pass
                block = response.read1(65536)
                elapsed = time.monotonic() - start
                raw_stream.write(block)
                raw_stream.flush()
                os.fsync(raw_stream.fileno())
                total += len(block)
                if total > 64 * 1024**2:
                    raise ValueError('Response exceeds 64MiB')
                if streaming:
                    for text in parser.feed(block, eof=not block):
                        journal(directory / 'events.jsonl', dict(seconds=elapsed, data=text))
                        if text.strip() == '[DONE]':
                            result['done'] = True
                            continue
                        if result['done']:
                            raise ValueError('Data after [DONE]')
                        accept(json.loads(text), elapsed, True)
                    if result['done']:
                        break
                if not block:
                    break
            if not streaming:
                raw_stream.flush()
                data = json.loads((directory / 'response.body').read_bytes())
                journal(directory / 'events.jsonl', dict(seconds=elapsed, data=data))
                accept(data, elapsed, False)
            elif not result['done']:
                raise ValueError('SSE ended without [DONE]')
        if result['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Missing/unsupported finish reason')
        result['ok'] = True
    except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['seconds'] = time.monotonic() - start
        result['resources'] = sampler.stop() if sampler is not None else None
        firsts = [result[key] for key in ('first_content_seconds', 'first_reasoning_seconds')
                  if result[key] is not None]
        result['ttft_seconds'] = min(firsts) if firsts else None
        result.update(server_telemetry(events, result['usage'], before=before))
        completion = result['tokens']['completion_tokens']
        result['completion_tokens_per_second_end_to_end'] = (
            completion / result['seconds'] if isinstance(completion, int) else None)
        result['usage_available'] = result['usage'] is not None
        result['raw_server_telemetry'] = [
            {key: event[key] for key in ('timings', 'metrics', 'stats', 'mylocal_stats', 'usage') if key in event}
            for event in events if any(key in event for key in ('timings', 'metrics', 'stats', 'mylocal_stats', 'usage'))]
        # Keep a durable response checkpoint before optional probes can time out.
        save_json(directory / 'response.partial.json', result)
        after = http.snapshot(telemetry_paths)
        save_json(directory / 'after.json', after)
        result.update(server_telemetry(events, result['usage'], before, after))
        completion = result['tokens']['completion_tokens']
        result['completion_tokens_per_second_end_to_end'] = (
            completion / result['seconds'] if isinstance(completion, int) else None)
        save_json(directory / 'response.json', result)
    return result


def verify_context(snapshot, model, expected):
    models = snapshot.get('/v1/models', {})
    health = snapshot.get('/health', {})
    evidence = []
    if models.get('status') == 200:
        matches = [row for row in (models.get('json') or {}).get('data', []) if row.get('id') == model]
        if not matches:
            raise ValueError(f'Model {model!r} not present in /v1/models')
    else:
        matches = []
    def collect(value, source):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('n_ctx', 'max_context', 'max_model_len', 'max_context_length') and isinstance(item, int):
                    evidence.append(dict(source=source + '.' + key, value=item))
                elif isinstance(item, dict):
                    collect(item, source + '.' + key)
    for row in matches:
        collect(row, '/v1/models:' + model)
    if health.get('status') == 200:
        data = health.get('json') or {}
        if data.get('model') in (None, model):
            collect(data, '/health')
    if not evidence or any(row['value'] != expected for row in evidence):
        raise ValueError(f'Configured context {expected} not verified via health/models: {evidence}')
    return evidence


class Sizer:
    def __init__(self, http, model, gguf=None, template_path='/apply-template',
                 tokenize_path='/tokenize', thinking=True):
        self.http, self.model = http, model
        self.template_path, self.tokenize_path = template_path, tokenize_path
        self.thinking = thinking
        self.tokenizer = None
        self.source = 'backend_template_and_tokenize'
        if gguf:
            # Lazy imports: default client and --help require only stdlib.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from mini_llm.external.gguf_store import MultiShardTensorStore
            from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
            from mini_llm.external.qwen4exp_chat import render_chat
            with MultiShardTensorStore.open(gguf) as store:
                self.tokenizer = load_gguf_tokenizer(store.metadata)
            self.render_chat = render_chat
            self.source = 'local_gguf_template_and_tokenizer:' + str(gguf)

    def count(self, messages):
        if self.tokenizer is not None:
            rendered = self.render_chat(self.tokenizer, messages, thinking=self.thinking)
            return len(self.tokenizer.encode(rendered).ids)
        rendered = self.http.post_json(self.template_path, dict(
            model=self.model, messages=messages, add_generation_prompt=True,
            chat_template_kwargs={'enable_thinking': self.thinking}))
        prompt = rendered.get('prompt')
        if not isinstance(prompt, str):
            raise ValueError('Template route must return a prompt string; use --gguf if unavailable')
        data = self.http.post_json(self.tokenize_path, dict(content=prompt, add_special=False))
        if not isinstance(data.get('tokens'), list):
            raise ValueError('Tokenize route must return tokens list; no character-count fallback')
        return len(data['tokens'])


def parse_sizes(text, context, max_output, headroom):
    sizes = []
    for part in text.split(','):
        label = part.strip()
        if label.lower() == 'near262k':
            if context != 262144:
                raise ValueError('near262K requires configured context 262144')
            count = context - max_output - headroom
        else:
            count = int(label[:-1]) * 1024 if label.lower().endswith('k') else int(label)
        if count < 1 or count + max_output > context:
            raise ValueError(f'Size {label} violates context/output reserve')
        sizes.append((label, count))
    return sizes


def source_messages(code, nonce):
    return [dict(role='user', content=f'Workload {nonce}\nReview this Python source. '
                 'Identify one correctness issue and suggest a minimal fix.\n```python\n' + code + '\n```')]


def load_source_files(paths):
    """Read-only concatenation in CLI order; banners are part of the sized prompt."""
    chunks, sources = [], []
    for name in paths:
        path = Path(name)
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        chunks.append(f'# ===== FILE: {name} =====\n{text}\n# ===== END FILE: {name} =====\n')
        sources.append(dict(path=str(path), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return '\n'.join(chunks), sources


def sized_source(sizer, target, nonce, source_code=None):
    """Search source character length using actual tokenizer calls, not char/token ratios."""
    code = ''
    block = ('def compute_{i}(values):\n'
             '    """Return a stable average of the supplied samples."""\n'
             '    total = sum(value for value in values if value is not None)\n'
             '    return total / len(values)\n\n')
    index = 0
    while True:
        messages = source_messages(code, nonce)
        count = sizer.count(messages)
        if count >= target:
            break
        if len(code) > 32 * 1024**2:
            raise ValueError('Sizing exceeded 32MiB source bound')
        # Exponential growth limits expensive tokenizer HTTP round trips.
        number = max(1, index)
        if source_code is not None:
            code += source_code * number
        else:
            code += ''.join(block.format(i=i) for i in range(index, index + number))
        index += number
    low, high = 0, len(code)
    best, best_count = None, None
    while low <= high:
        mid = (low + high) // 2
        messages = source_messages(code[:mid], nonce)
        count = sizer.count(messages)
        if count <= target:
            best, best_count = messages, count
            low = mid + 1
        else:
            high = mid - 1
    if best is None:
        raise ValueError(f'Target {target} smaller than prompt wrapper')
    return best, best_count


def run_replay(args, http, output, paths):
    """Fixed trajectories: prior assistant history is never replaced by new output."""
    source = Path(args.replay_from)
    requests = sorted(source.glob('*/request.json')) if source.is_dir() else []
    if not requests:
        raise ValueError(f'No saved request payloads in replay directory: {source}')
    if args.max_requests is not None:
        requests = requests[:args.max_requests]
    successes = failures = 0
    for index, request in enumerate(requests):
        raw = request.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or 'model' not in payload:
            raise ValueError(f'Invalid replay payload: {request}')
        original_model = payload['model']
        payload['model'] = args.model
        metadata = dict(
            workload='fixed_replay', trajectory='fixed_replay',
            replay_source=str(request.resolve()), replay_index=index,
            replay_source_sha256=hashlib.sha256(raw).hexdigest(),
            original_model=original_model, sizing_prompt_tokens=None,
            sizing_source=None, token_counts_source='backend_usage',
            server_reset=False, context=args.context,
            output_reserve=payload.get('max_tokens'),
            weights_label=('unknown_initial_state' if successes == 0 else
                           'after_successful_request_residency_unverified'))
        metadata['request_sha256'] = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False).encode('utf-8')).hexdigest()
        directory = output / f'{index:04d}-fixed-replay'
        result = benchmark_request(http, payload, directory, metadata, paths,
                                   args.resource_pid, args.resource_gpu_uuid)
        successes += int(result['ok'])
        failures += int(not result['ok'])
        journal(output / 'results.jsonl', dict(request_directory=directory.name,
                                              metadata=metadata, **result))
        print(f'{directory.name}: ok={result["ok"]} seconds={result["seconds"]:.3f} '
              f'ttft={result["ttft_seconds"]} tokens={result["tokens"]}', flush=True)
    save_json(output / 'summary.json', dict(successes=successes, failures=failures,
                                          trajectory='fixed_replay'))
    return 1 if failures else 0


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    http = HTTP(args.url, args.timeout, os.environ.get(args.api_key_env))
    paths = tuple(dict.fromkeys(TELEMETRY_PATHS + tuple(args.telemetry_path)))
    config = vars(args).copy()
    config['api_key_present'] = bool(http.api_key)
    config['normalized_url'] = http.base_url
    config['workload_id'] = args.workload_id if args.replay_from else (
        args.workload_id if args.workload_id is not None else uuid.uuid4().hex)
    config['trajectory'] = 'fixed_replay' if args.replay_from else 'live_followups'
    source_code, source_files = (load_source_files(args.source_file)
                                 if args.source_file and not args.replay_from else (None, []))
    config['source_files'] = source_files
    config['source_content'] = 'source_files' if source_code is not None else 'generated_synthetic_code'
    config['notes'] = [
        'No server reset, eviction or weight warmup is performed.',
        'cold_prompt means first submission here, not verified empty KV cache.',
        'Weight residency and cache reuse are unknown unless backend telemetry reports them.',
        'Global snapshots can include other clients; no per-request attribution is inferred.',
        'Sizing is independent; backend usage is authoritative for the served request.',
        'Near-context headroom reserves followups in addition to max_tokens.',
        'Replay preserves every saved payload field except model, including prior assistant history; '
        'generation/sizing flags and workload-id do not modify replay payloads.']
    save_json(output / 'config.json', config)
    snapshot = http.snapshot(paths)
    save_json(output / 'initial.json', snapshot)
    evidence = verify_context(snapshot, args.model, args.context)
    save_json(output / 'context_verification.json', evidence)
    if args.replay_from:
        return run_replay(args, http, output, paths)
    sizes = parse_sizes(args.sizes, args.context, args.max_output, args.headroom)
    sizer = Sizer(http, args.model, args.gguf, args.template_path, args.tokenize_path,
                  thinking=not args.no_thinking)
    successes = 0
    failures = 0
    request_index = 0
    nonce = config['workload_id']
    for size_index, (label, target) in enumerate(sizes):
        base, measured = sized_source(sizer, target, f'{nonce}-{size_index}', source_code)
        source_characters = (len(base[0]['content']) - len(source_messages('', f'{nonce}-{size_index}')[0]['content']))
        repeated_source = source_code is not None and source_characters > len(source_code)
        previous = None
        messages = base
        for case in range(args.repeat + args.turns - 1):
            repeated = case < args.repeat
            kind = 'cold_prompt_first_submission' if case == 0 else (
                'identical_prefix_repeat' if repeated else 'multi_turn')
            if repeated:
                messages = copy.deepcopy(base)
                count = measured
            else:
                if previous is None or not previous['ok']:
                    break
                messages = messages + [previous['message'], dict(
                    role='user', content='Now propose a regression test for the issue you identified.')]
                count = sizer.count(messages)
            metadata = dict(size=label, target_prompt_tokens=target,
                            sizing_prompt_tokens=count, sizing_source=sizer.source,
                            workload=kind, repeat_index=case if repeated else None,
                            source_content=config['source_content'],
                            source_files=source_files, repeated_source_content=repeated_source,
                            weights_label=('unknown_initial_state' if successes == 0 else
                                           'after_successful_request_residency_unverified'),
                            server_reset=False, context=args.context,
                            output_reserve=args.max_output)
            if count + args.max_output > args.context:
                journal(output / 'results.jsonl', dict(metadata=metadata, skipped='context/output reserve'))
                break
            payload = dict(model=args.model, messages=messages, max_tokens=args.max_output,
                           temperature=0, stream=args.stream,
                           chat_template_kwargs={'enable_thinking': not args.no_thinking})
            if args.stream:
                payload['stream_options'] = {'include_usage': True}
            directory = output / f'{request_index:04d}-{label}-{kind}'
            metadata['request_sha256'] = hashlib.sha256(json.dumps(
                payload, ensure_ascii=False).encode('utf-8')).hexdigest()
            previous = benchmark_request(http, payload, directory, metadata, paths,
                                         args.resource_pid, args.resource_gpu_uuid)
            request_index += 1
            successes += int(previous['ok'])
            failures += int(not previous['ok'])
            journal(output / 'results.jsonl', dict(request_directory=directory.name,
                                                  metadata=metadata, **previous))
            print(f'{directory.name}: ok={previous["ok"]} seconds={previous["seconds"]:.3f} '
                  f'ttft={previous["ttft_seconds"]} tokens={previous["tokens"]}', flush=True)
    save_json(output / 'summary.json', dict(successes=successes, failures=failures))
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='http://127.0.0.1:8080', help='Server root or /v1 URL; trailing slash accepted')
    p.add_argument('--model', required=True, help='Exact served /v1/models ID')
    p.add_argument('--output', default='benchmarks/qwen4exp-serving-' + uuid.uuid4().hex[:12],
                   help='New artifact directory; existing paths are never overwritten')
    p.add_argument('--workload-id', help='Deterministic source nonce, recorded in config; random if omitted')
    p.add_argument('--replay-from', help='Prior artifact directory: sorted saved payloads, replace ONLY model; '
                   'fixed trajectories, no sizing or new followup construction')
    p.add_argument('--max-requests', type=int, help='Replay only: send at most this many sorted saved requests')
    p.add_argument('--source-file', action='append', default=[],
                   help='Read-only UTF-8 source path; repeat for ordered file banners/concatenation. '
                   'Token-sized truncation; whole source repeats for large targets (labeled).')
    p.add_argument('--resource-pid', type=int, help='Opt-in local server PID resource sampling at 1s; '
                   'enables /proc and bounded nvidia-smi calls, request-local resources.jsonl')
    p.add_argument('--resource-gpu-uuid', help='Explicit GPU- UUID; otherwise resolve GPU 0 once per request')
    p.add_argument('--sizes', default=DEFAULT_SIZES, help='Comma-separated prompt token targets; K=1024')
    p.add_argument('--context', type=int, default=262144, help='Must be verified via health/models')
    p.add_argument('--max-output', type=int, default=256, help='max_tokens including reasoning, backend dependent')
    p.add_argument('--headroom', type=int, default=1024, help='Additional near262K followup/template reserve')
    p.add_argument('--repeat', type=int, default=2, help='Identical base requests, including first submission')
    p.add_argument('--turns', type=int, default=2, help='Conversation turns including the base response')
    p.add_argument('--timeout', type=float, default=600, help='Per generation deadline / probe timeout seconds')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--stream', dest='stream', action='store_true', default=True)
    mode.add_argument('--nonstream', dest='stream', action='store_false')
    p.add_argument('--gguf', help='Local GGUF shard/directory; optional repository CPU tokenizer dependencies')
    p.add_argument('--template-path', default='/apply-template')
    p.add_argument('--tokenize-path', default='/tokenize')
    p.add_argument('--no-thinking', action='store_true')
    p.add_argument('--telemetry-path', action='append', default=[], help='Additional raw GET snapshot route; repeatable')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY', help='Environment variable; key is not saved')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (min(args.context, args.max_output, args.repeat, args.turns) < 1 or args.headroom < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error('Positive context/output/repeat/turns/timeout and nonnegative headroom required')
    if args.max_requests is not None and (args.max_requests < 1 or not args.replay_from):
        p.error('--max-requests must be positive and requires --replay-from')
    if args.resource_pid is not None and args.resource_pid < 1:
        p.error('--resource-pid must be positive')
    if args.resource_gpu_uuid is not None and (
            args.resource_pid is None or not args.resource_gpu_uuid.startswith('GPU-')
            or any(char.isspace() for char in args.resource_gpu_uuid) or ',' in args.resource_gpu_uuid):
        p.error('--resource-gpu-uuid requires --resource-pid and one explicit GPU- UUID')
    try:
        if not args.replay_from:
            parse_sizes(args.sizes, args.context, args.max_output, args.headroom)
        return run(args)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f'Benchmark stopped: {exc}. Existing artifacts are preserved.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
