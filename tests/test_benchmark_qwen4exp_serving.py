"""CPU-only HTTP contract tests; run directly with unittest (no model imports)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_qwen4exp_serving.py'
spec = importlib.util.spec_from_file_location('serving_benchmark', SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


MYLOCAL_STATS = {
    'prompt_tokens': 101, 'new_prompt_tokens': 81, 'reused_tokens': 20,
    'generated_tokens': 7, 'decode_forward_tokens': 6,
    'prefill_seconds': 1.25, 'decode_seconds': 0.75,
    'first_piece_seconds': 1.4, 'total_seconds': 2.1,
    'decode_forwards_per_second': 8.0, 'request_id': 'mock-request',
    'rebuilt_prefix': False, 'prefix_match': True, 'state_reset': False,
    'reset_reason': None, 'state_tokens': 107,
    'cache': {'total_tokens': 101, 'new_tokens': 81, 'reused_tokens': 20,
              'prefix_match': True, 'state_reset': False, 'reset_reason': None,
              'state_tokens': 107},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, status=200, content_type='application/json'):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        data = {
            '/health': {'model': 'mock', 'max_context': 262144},
            '/v1/models': {'data': [{'id': 'mock', 'meta': {'n_ctx': 262144}}]},
            '/metrics': b'cache_hits 12\nhardware_gpu_memory_bytes 1234\n',
        }
        if self.server.mode.startswith('mylocal'):
            data['/metrics'] = {'mylocal_stats': MYLOCAL_STATS,
                                'experts': {'device': {'hit_rate': 0.6, 'hits': 123}}}
        self.send(data.get(self.path, {'error': 'unavailable'}),
                  200 if self.path in data else 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        payload = json.loads(raw)
        if self.path == '/apply-template':
            self.send({'prompt': json.dumps(payload['messages'], ensure_ascii=False)})
            return
        if self.path == '/tokenize':
            # Deliberately non-character sizing: UTF-8 byte tokenizer fixture.
            self.send({'tokens': list(payload['content'].encode('utf-8'))})
            return
        self.server.received.append((self.path, raw, payload))
        if self.server.mode == 'http_error':
            self.send({'error': {'message': 'bad request'}}, 400)
            return
        usage = {'prompt_tokens': 101, 'completion_tokens': 7,
                 'completion_tokens_details': {'reasoning_tokens': 4}}
        if self.server.mode == 'no_usage':
            usage = None
        extra = {}
        if self.server.mode.startswith('mylocal'):
            extra['id'] = 'mock-request'
            if self.server.mode != 'mylocal_metrics_only':
                extra['mylocal_stats'] = MYLOCAL_STATS
        if not payload['stream']:
            self.send({**extra, 'choices': [{'message': {'role': 'assistant', 'content': 'answer',
                                               'reasoning_content': 'analysis'},
                                    'finish_reason': 'stop'}], 'usage': usage})
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        events = [
            {'choices': [{'delta': {'role': 'assistant', 'content': ''}}]},
            {'choices': [{'delta': {'reasoning_content': 'réason'}}]},
            {'choices': [{'delta': {'content': 'answer with several tokens'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': usage,
             'timings': {'prompt_n': 81, 'cache_n': 20, 'predicted_ms': 123},
             'metrics': {'hardware': {'gpu_busy': 0.5}}},
        ]
        if self.server.mode.startswith('mylocal'):
            # MyLocal final usage chunk has no llama.cpp timings block.
            events[-1] = {'choices': [], 'usage': usage, **extra}
        if self.server.mode == 'error':
            events = events[:2] + [{'error': {'message': 'generation failed'}}]
        if self.server.mode == 'missing_finish':
            events.pop(3)
        for event in events:
            encoded = json.dumps(event, ensure_ascii=False).encode()
            # Force separate network arrivals, including empty role delta.
            self.wfile.write(b': heartbeat\r\ndata: ' + encoded + b'\r\n\r\n')
            self.wfile.flush()
            time.sleep(0.003)
        if self.server.mode not in ('incomplete', 'error'):
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()


class ServingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.mode = 'ok'
        cls.server.received = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server.mode = 'ok'
        self.server.received.clear()
        self.http = bench.HTTP(self.url, timeout=3)

    def request(self, stream=True):
        self.payload = {'model': 'mock', 'messages': [{'role': 'user', 'content': 'code α'}],
                        'stream': stream, 'max_tokens': 16}
        return bench.benchmark_request(self.http, self.payload, self.root / 'request')

    def test_stream_reasoning_ttft_usage_and_artifacts(self):
        result = self.request()
        self.assertTrue(result['ok'], result)
        self.assertTrue(result['done'])
        self.assertEqual(result['tokens']['completion_tokens'], 7)
        self.assertEqual(result['tokens']['reasoning_tokens'], 4)
        self.assertEqual(result['tokens']['processed_prompt_tokens'], 81)
        self.assertEqual(result['tokens']['reused_prompt_tokens'], 20)
        self.assertEqual(result['ttft_seconds'], result['first_reasoning_seconds'])
        # TCP may coalesce distinct SSE events into one client arrival.
        self.assertLessEqual(result['first_reasoning_seconds'], result['first_content_seconds'])
        self.assertEqual(result['message']['reasoning_content'], 'réason')
        self.assertEqual(result['timings']['predicted_ms'], 123)
        self.assertAlmostEqual(result['phases']['decode_seconds'], 0.123)
        self.assertIsNone(result['tokens']['decode_forward_tokens'])
        self.assertEqual(result['completion_tokens_per_second_end_to_end'], 7 / result['seconds'])
        directory = self.root / 'request'
        self.assertEqual((directory / 'request.json').read_bytes(), self.server.received[0][1])
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['data'], '[DONE]')
        for name in ('before.json', 'after.json'):
            snapshot = json.loads((directory / name).read_text())
            self.assertIn('hardware_gpu_memory_bytes', snapshot['/metrics']['raw'])
            self.assertEqual(snapshot['/slots']['status'], 404)
        self.assertIsNone(result['hardware'])  # Unknown schema remains raw, not fabricated.

    def test_mylocal_stats_stream_nonstream_and_metrics_fallback(self):
        for mode, streaming in (('mylocal', True), ('mylocal', False),
                                ('mylocal_metrics_only', True)):
            with self.subTest(mode=mode, streaming=streaming):
                self.server.mode = mode
                directory = self.root / f'{mode}-{streaming}'
                result = bench.benchmark_request(self.http, {
                    'model': 'mock', 'messages': [{'role': 'user', 'content': 'source'}],
                    'stream': streaming, 'max_tokens': 16,
                }, directory)
                self.assertTrue(result['ok'], result)
                self.assertEqual(result['tokens']['prompt_tokens'], 101)
                self.assertEqual(result['tokens']['processed_prompt_tokens'], 81)
                self.assertEqual(result['tokens']['reused_prompt_tokens'], 20)
                self.assertEqual(result['tokens']['decode_forward_tokens'], 6)
                self.assertEqual(result['tokens']['completion_tokens'], 7)
                self.assertEqual(result['phases']['prefill_seconds'], 1.25)
                self.assertEqual(result['phases']['decode_seconds'], 0.75)
                self.assertEqual(result['phases']['total_seconds'], 2.1)
                self.assertEqual(result['phases']['first_piece_seconds'], 1.4)
                self.assertEqual(result['phases']['decode_forwards_per_second'], 8)
                self.assertIsNone(result['phases']['decode_tokens_per_second'])
                self.assertEqual(result['cache'], MYLOCAL_STATS['cache'])
                self.assertEqual(result['mylocal_stats'], MYLOCAL_STATS)
                self.assertEqual(result['cache_snapshots']['after']['experts']['device']['hits'], 123)
                self.assertEqual(json.loads((directory / 'response.json').read_text()), result)
                if mode == 'mylocal':
                    self.assertEqual(result['raw_server_telemetry'][-1]['mylocal_stats'], MYLOCAL_STATS)
                if not streaming:
                    self.assertIsNone(result['ttft_seconds'])

    def test_reported_zero_and_missing_mylocal_stats_are_distinct(self):
        stats = {'prompt_tokens': 127, 'new_prompt_tokens': 127, 'reused_tokens': 0,
                 'decode_forward_tokens': 127, 'generated_tokens': 128,
                 'prefill_seconds': 15.146162407000702, 'decode_seconds': 15.583035726000162}
        result = bench.server_telemetry([{'mylocal_stats': stats}], None)
        self.assertEqual(result['tokens']['prompt_tokens'], 127)
        self.assertEqual(result['tokens']['completion_tokens'], 128)
        self.assertEqual(result['tokens']['processed_prompt_tokens'], 127)
        self.assertEqual(result['tokens']['reused_prompt_tokens'], 0)
        self.assertEqual(result['tokens']['decode_forward_tokens'], 127)
        self.assertEqual(result['phases']['decode_seconds'], stats['decode_seconds'])
        missing = bench.server_telemetry([{'choices': []}] * 50, None)
        self.assertTrue(all(value is None for value in missing['tokens'].values()))
        self.assertTrue(all(value is None for value in missing['phases'].values()))
        self.assertIsNone(missing['cache'])

    def test_strata_metrics_request_mapping_and_hardware_samples(self):
        row = {'id': 'strata-request', 'prompt_tokens': 127, 'prompt_read': 5,
               'reused': 122, 'output_tokens': 128, 'prompt_ms': 50.4,
               'decode_ms': 1419.9, 'decode_tok_s': 90.1,
               'duration_s': 1.5, 'first_engine_token_s': 0.063}
        snapshot = {'/metrics': {'status': 200, 'json': {
            'requests': [row], 'hardware': {'gpu_util': 50},
            'hardware_static': {'gpu_name': 'mock GPU'},
            'conversation_cache': {'reused_tokens': 99999}}}}
        result = bench.server_telemetry([{'id': 'strata-request'}], None, snapshot, snapshot)
        self.assertEqual(result['tokens']['processed_prompt_tokens'], 5)
        self.assertEqual(result['tokens']['reused_prompt_tokens'], 122)
        self.assertEqual(result['tokens']['completion_tokens'], 128)
        self.assertIsNone(result['tokens']['decode_forward_tokens'])
        self.assertAlmostEqual(result['phases']['prefill_seconds'], 0.0504)
        self.assertAlmostEqual(result['phases']['decode_seconds'], 1.4199)
        self.assertEqual(result['phases']['decode_tokens_per_second'], 90.1)
        self.assertEqual(result['phases']['first_engine_token_seconds'], 0.063)
        self.assertEqual(result['metrics_request'], row)
        self.assertEqual(result['hardware']['after']['hardware']['gpu_util'], 50)
        stale = bench.server_telemetry([{'id': 'other-request'}], None, snapshot, snapshot)
        self.assertIsNone(stale['metrics_request'])
        self.assertIsNone(stale['tokens']['reused_prompt_tokens'])
        self.assertIsNone(stale['phases']['decode_seconds'])
        # A stale MyLocal last-request sample also cannot populate request-local fields.
        snapshot['/metrics']['json'] = {'mylocal_stats': MYLOCAL_STATS}
        stale = bench.server_telemetry([{'id': 'other-request'}], None, after=snapshot)
        self.assertIsNone(stale['mylocal_stats'])
        self.assertIsNone(stale['tokens']['processed_prompt_tokens'])

    def test_nonstream_no_fake_ttft_or_reuse(self):
        result = self.request(stream=False)
        self.assertTrue(result['ok'], result)
        self.assertIsNone(result['ttft_seconds'])
        self.assertIsNone(result['first_content_seconds'])
        self.assertIsNone(result['tokens']['reused_prompt_tokens'])
        self.assertIsNone(result['tokens']['processed_prompt_tokens'])
        self.assertEqual(result['message']['content'], 'answer')
        self.assertEqual(result['message']['reasoning_content'], 'analysis')
        self.assertGreater(result['seconds'], 0)

    def test_incomplete_stream_preserves_partial_output(self):
        self.server.mode = 'incomplete'
        result = self.request()
        self.assertFalse(result['ok'])
        self.assertIn('without [DONE]', result['error'])
        self.assertEqual(result['message']['content'], 'answer with several tokens')
        self.assertTrue((self.root / 'request' / 'response.json').exists())
        self.assertTrue((self.root / 'request' / 'events.jsonl').exists())

    def test_error_stream(self):
        self.server.mode = 'error'
        result = self.request()
        self.assertFalse(result['ok'])
        self.assertIn('generation failed', result['error'])
        self.assertEqual(result['message']['reasoning_content'], 'réason')

    def test_done_does_not_replace_finish_reason(self):
        self.server.mode = 'missing_finish'
        result = self.request()
        self.assertFalse(result['ok'])
        self.assertIn('finish reason', result['error'])

    def test_http_error_saved(self):
        self.server.mode = 'http_error'
        result = self.request()
        self.assertFalse(result['ok'])
        self.assertEqual(result['http_status'], 400)
        self.assertIn('bad request', (self.root / 'request' / 'response.body').read_text())

    def test_missing_usage_is_null_not_event_token_count(self):
        self.server.mode = 'no_usage'
        result = self.request()
        self.assertTrue(result['ok'], result)
        self.assertIsNone(result['tokens']['completion_tokens'])
        self.assertIsNone(result['completion_tokens_per_second_end_to_end'])
        self.assertFalse(result['usage_available'])

    def test_sse_split_utf8_multiline_and_incomplete_frame(self):
        parser = bench.SSE()
        body = ': comment\r\ndata: {"text":\r\ndata: "é"}\r\n\r\n'.encode()
        events = []
        for byte in body:
            events.extend(parser.feed(bytes([byte])))
        self.assertEqual(json.loads(events[0]), {'text': 'é'})
        parser.feed(b'data: unfinished')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            parser.feed(b'', eof=True)

    def test_context_verification_requires_configured_not_training_context(self):
        snapshot = self.http.snapshot()
        self.assertTrue(bench.verify_context(snapshot, 'mock', 262144))
        with self.assertRaises(ValueError):
            bench.verify_context(snapshot, 'wrong-model', 262144)
        snapshot['/health']['json'] = {'n_ctx_train': 262144}
        snapshot['/v1/models']['json']['data'][0]['meta'] = {'n_ctx_train': 262144}
        with self.assertRaisesRegex(ValueError, 'not verified'):
            bench.verify_context(snapshot, 'mock', 262144)

    def test_sizes_and_independent_backend_sizing(self):
        sizes = bench.parse_sizes(bench.DEFAULT_SIZES, 262144, 256, 1024)
        self.assertEqual([n for _, n in sizes[:-1]], [128, 1024, 4096, 16384, 32768, 65536, 131072])
        self.assertEqual(sizes[-1][1], 262144 - 256 - 1024)
        sizer = bench.Sizer(self.http, 'mock')
        messages, count = bench.sized_source(sizer, 1024, 'unique')
        self.assertEqual(count, sizer.count(messages))
        self.assertLessEqual(count, 1024)
        self.assertGreater(count, 1000)
        self.assertIn('def compute_', messages[0]['content'])
        self.assertGreater(sizer.count([{'role': 'user', 'content': 'é'}]),
                           sizer.count([{'role': 'user', 'content': 'e'}]))
        with self.assertRaises(ValueError):
            bench.parse_sizes('262144', 262144, 256, 1024)

    def test_run_repeats_and_lossless_multiturn_incremental_results(self):
        output = self.root / 'run'
        status = bench.main(['--url', self.url, '--model', 'mock', '--output', str(output),
                             '--sizes', '1K', '--repeat', '2', '--turns', '2', '--max-output', '16'])
        self.assertEqual(status, 0)
        received = self.server.received
        self.assertEqual(len(received), 3)
        self.assertEqual(received[0][1], received[1][1])
        messages = received[2][2]['messages']
        self.assertEqual(messages[0], received[0][2]['messages'][0])
        self.assertEqual(messages[1]['reasoning_content'], 'réason')
        self.assertEqual(messages[1]['content'], 'answer with several tokens')
        rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]['metadata']['workload'], 'identical_prefix_repeat')
        self.assertEqual(rows[2]['metadata']['workload'], 'multi_turn')
        self.assertFalse(rows[0]['metadata']['server_reset'])
        self.assertEqual(bench.main(['--url', self.url, '--model', 'mock', '--output', str(output)]), 1)
        self.assertEqual(len(self.server.received), 3)  # Existing artifacts never overwritten.

    def test_root_and_v1_url_normalization(self):
        for suffix in ('', '/', '/v1', '/v1/'):
            with self.subTest(suffix=suffix):
                http = bench.HTTP(self.url + suffix, timeout=3)
                self.assertEqual(http.base_url, self.url)
                self.assertTrue(bench.verify_context(http.snapshot(), 'mock', 262144))
                result = bench.benchmark_request(http, {
                    'model': 'mock', 'messages': [{'role': 'user', 'content': 'source'}],
                    'stream': False, 'max_tokens': 8,
                }, self.root / ('url-' + str(len(self.server.received))))
                self.assertTrue(result['ok'], result)
                self.assertEqual(self.server.received[-1][0], '/v1/chat/completions')

    def test_workload_id_deterministic_and_recorded(self):
        for name in ('first', 'second'):
            output = self.root / name
            status = bench.main([
                '--url', self.url, '--model', 'mock', '--output', str(output),
                '--workload-id', 'shared-code-study', '--sizes', '1K',
                '--repeat', '1', '--turns', '1', '--nonstream',
            ])
            self.assertEqual(status, 0)
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['workload_id'], 'shared-code-study')
            self.assertEqual(config['trajectory'], 'live_followups')
        self.assertEqual(self.server.received[0][1], self.server.received[1][1])
        self.assertIn('shared-code-study-0', self.server.received[0][2]['messages'][0]['content'])

    def test_fixed_replay_only_replaces_model_sorted_and_without_sizing(self):
        source = self.root / 'prior'
        source.mkdir()
        base = {'model': 'prior-backend', 'messages': [{'role': 'user', 'content': 'source α'}],
                'stream': False, 'max_tokens': 7, 'temperature': 0.75,
                'vendor_extension': {'keep': ['exact', 123]}}
        followup = {**base, 'messages': base['messages'] + [
            {'role': 'assistant', 'content': 'prior answer', 'reasoning_content': 'prior reasoning'},
            {'role': 'user', 'content': 'test it'}], 'stream': True,
            'stream_options': {'include_usage': True}}
        # Creation order differs from saved request order.
        for name, payload in (('0001-followup', followup), ('0000-base', base)):
            directory = source / name
            directory.mkdir()
            bench.save_json(directory / 'request.json', payload)
        original_bytes = {path: path.read_bytes() for path in source.glob('*/request.json')}
        output = self.root / 'replay'
        with patch.object(bench, 'Sizer', side_effect=AssertionError('Replay must not size')):
            status = bench.main([
                '--url', self.url + '/v1/', '--model', 'mock', '--output', str(output),
                '--replay-from', str(source), '--sizes', 'ignored-invalid-size',
                '--gguf', '/unused.gguf', '--max-output', '999', '--repeat', '9',
                '--turns', '9', '--workload-id', 'must-not-change-prompts', '--nonstream',
            ])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.server.received), 2)
        for received, expected in zip(self.server.received, (base, followup)):
            self.assertEqual(received[2], {**expected, 'model': 'mock'})
        rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
        for index, row in enumerate(rows):
            self.assertEqual(row['metadata']['trajectory'], 'fixed_replay')
            self.assertIsNone(row['metadata']['sizing_prompt_tokens'])
            self.assertEqual(row['tokens']['prompt_tokens'], 101)
            self.assertEqual(row['metadata']['replay_index'], index)
            saved = output / row['request_directory'] / 'request.json'
            self.assertEqual(saved.read_bytes(), self.server.received[index][1])
        self.assertEqual(json.loads((output / 'config.json').read_text())['trajectory'], 'fixed_replay')
        self.assertEqual(json.loads((output / 'summary.json').read_text())['trajectory'], 'fixed_replay')
        for path, raw in original_bytes.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_replay_requires_saved_payloads(self):
        source = self.root / 'empty'
        source.mkdir()
        status = bench.main(['--url', self.url, '--model', 'mock',
                             '--output', str(self.root / 'replay'), '--replay-from', str(source)])
        self.assertEqual(status, 1)
        self.assertEqual(self.server.received, [])

    def test_replay_max_requests_bound(self):
        source = self.root / 'prior-bounded'
        source.mkdir()
        for index in (2, 0, 1):
            directory = source / f'{index:04d}-saved'
            directory.mkdir()
            bench.save_json(directory / 'request.json', dict(
                model='original', messages=[{'role': 'user', 'content': str(index)}],
                stream=False, max_tokens=5))
        output = self.root / 'bounded'
        self.assertEqual(bench.main([
            '--url', self.url, '--model', 'mock', '--replay-from', str(source),
            '--output', str(output), '--max-requests', '2',
        ]), 0)
        self.assertEqual([row[2]['messages'][0]['content'] for row in self.server.received], ['0', '1'])
        self.assertEqual(json.loads((output / 'summary.json').read_text())['successes'], 2)

    def test_source_files_banners_token_sizing_and_repetition_labels(self):
        first, second = self.root / 'a.py', self.root / 'b.py'
        first.write_text('def real_code():\n    return "α"\n', encoding='utf-8')
        second.write_text('assert real_code() == "α"\n', encoding='utf-8')
        code, identities = bench.load_source_files([str(first), str(second)])
        self.assertLess(code.index('FILE: ' + str(first)), code.index('FILE: ' + str(second)))
        self.assertIn(first.read_text(), code)
        self.assertIn(second.read_text(), code)
        self.assertEqual(len(identities), 2)
        sizer = bench.Sizer(self.http, 'mock')
        small, count = bench.sized_source(sizer, 400, 'fixed', code)
        self.assertEqual(sizer.count(small), count)
        self.assertLessEqual(count, 400)
        self.assertIn('def real_code', small[0]['content'])
        self.assertNotIn('def compute_', small[0]['content'])
        output = self.root / 'file-run'
        self.assertEqual(bench.main([
            '--url', self.url, '--model', 'mock', '--output', str(output),
            '--source-file', str(first), '--source-file', str(second),
            '--workload-id', 'real-files', '--sizes', '4K', '--repeat', '1',
            '--turns', '1', '--nonstream',
        ]), 0)
        row = json.loads((output / 'results.jsonl').read_text())
        self.assertTrue(row['metadata']['repeated_source_content'])
        self.assertEqual(row['metadata']['source_content'], 'source_files')
        self.assertEqual(row['metadata']['source_files'], identities)
        self.assertGreater(self.server.received[0][2]['messages'][0]['content'].count('def real_code'), 1)
        self.assertEqual(first.read_text(), 'def real_code():\n    return "α"\n')

    def test_resource_proc_parsing_cpu_deltas_and_uuid_selected_gpu(self):
        proc = self.root / 'proc'
        (proc / '123').mkdir(parents=True)
        fields = ['0'] * 22
        fields[0], fields[11], fields[12] = 'S', '100', '50'
        fields[19], fields[20], fields[21] = '999', '1000000', '10'
        (proc / '123' / 'stat').write_text('123 (server name) with spaces) ' + ' '.join(fields))
        (proc / 'stat').write_text('cpu 100 0 50 850 0 0 0 0 0 0\n')
        (proc / 'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 400 kB\n')
        sampler = bench.ResourceSampler(self.root, 123, proc_root=proc)
        sampler.ticks_per_second, sampler.page_bytes = 100, 4096
        def nvidia(command, **kwargs):
            text = ('GPU-test\n' if '--query-gpu=uuid' in command else
                    'GPU-test, 50, 25, 100, 1000, 120.5, 60\n')
            return subprocess.CompletedProcess(command, 0, text, '')
        with patch.object(bench.subprocess, 'run', side_effect=nvidia) as run:
            with patch.object(bench.time, 'monotonic', return_value=1):
                first = sampler.collect()
            fields[11] = '150'
            (proc / '123' / 'stat').write_text('123 (server name) ' + ' '.join(fields))
            (proc / 'stat').write_text('cpu 200 0 50 950 0 0 0 0 0 0\n')
            with patch.object(bench.time, 'monotonic', return_value=2):
                second = sampler.collect()
        self.assertIsNone(first['process']['cpu_percent_one_core'])
        self.assertEqual(second['process']['cpu_percent_one_core'], 50)
        self.assertEqual(second['global_cpu']['busy_percent'], 50)
        self.assertEqual(second['process']['rss_bytes'], 40960)
        self.assertEqual(second['global_memory']['used_bytes'], 600 * 1024)
        self.assertEqual(second['global_memory']['available_bytes'], 400 * 1024)
        self.assertEqual(second['gpu']['memory_used_bytes'], 100 * 1024**2)
        self.assertEqual(second['gpu']['uuid'], 'GPU-test')
        self.assertIn('process_stat', second['raw'])
        self.assertEqual(run.call_count, 3)
        self.assertIn('--id=0', run.call_args_list[0].args[0])
        for call in run.call_args_list[1:]:
            self.assertIn('--id=GPU-test', call.args[0])
            self.assertEqual(call.kwargs['timeout'], 2)

    def test_resources_missing_proc_gpu_timeout_and_pid_reuse(self):
        sampler = bench.ResourceSampler(self.root, 456, gpu_uuid='GPU-test',
                                        proc_root=self.root / 'missing')
        with patch.object(bench.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 2)):
            sample = sampler.collect()
        self.assertIsNone(sample['process']['rss_bytes'])
        self.assertIsNone(sample['gpu']['utilization_percent'])
        self.assertEqual(len(sample['errors']), 4)
        self.assertIn('error', sample['raw']['nvidia_smi'][0])

    def test_resource_thread_jsonl_minmax_and_join_on_request_error(self):
        sample = dict(process={'rss_bytes': 10, 'cpu_percent_one_core': None},
                      global_cpu={'busy_percent': 20},
                      global_memory={'used_bytes': 30}, gpu={'uuid': 'GPU-test', 'power_watts': 40},
                      raw={'mock': True}, errors=[])
        sampler = bench.ResourceSampler(self.root, 123, 'GPU-test', interval=0.001)
        count = []
        def collect():
            count.append(1)
            if len(count) == 2:
                sampler.stop_event.set()
            return {**sample, 'process': {**sample['process'], 'rss_bytes': len(count) * 10}}
        with patch.object(sampler, 'collect', side_effect=collect):
            sampler.start()
            sampler.thread.join(timeout=3)
            summary = sampler.stop()
        self.assertFalse(sampler.thread.is_alive())
        self.assertTrue(sampler.thread.daemon)
        self.assertEqual(summary['sample_count'], 2)
        self.assertEqual(summary['metrics']['process.rss_bytes'], {'min': 10, 'max': 20})
        self.assertEqual(summary['metrics']['process.cpu_percent_one_core'], {'min': None, 'max': None})
        self.assertEqual(len((self.root / 'resources.jsonl').read_text().splitlines()), 2)
        # HTTP failure still stops and joins the sampler before returning its summary.
        self.server.mode = 'http_error'
        request_sampler = bench.ResourceSampler(self.root / 'failed', 123, 'GPU-test')
        with patch.object(bench, 'ResourceSampler', return_value=request_sampler), \
                patch.object(request_sampler, 'collect', return_value=sample):
            result = bench.benchmark_request(self.http, dict(model='mock', messages=[], stream=True),
                                             self.root / 'failed', resource_pid=123)
        self.assertFalse(result['ok'])
        self.assertFalse(request_sampler.thread.is_alive())
        self.assertEqual(result['resources']['sample_count'], 1)
        self.assertIn('Whole selected GPU', result['resources']['labels']['gpu'])
        raw = (self.root / 'failed' / 'resources.jsonl').read_bytes()
        time.sleep(0.005)
        self.assertEqual((self.root / 'failed' / 'resources.jsonl').read_bytes(), raw)

    def test_no_resource_sampling_by_default(self):
        with patch.object(bench, 'ResourceSampler', side_effect=AssertionError('Not opted in')):
            result = self.request(stream=False)
        self.assertTrue(result['ok'])
        self.assertIsNone(result['resources'])
        self.assertFalse((self.root / 'request' / 'resources.jsonl').exists())

    def test_help_stdlib(self):
        completed = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--nonstream', completed.stdout)
        self.assertIn('--gguf', completed.stdout)


if __name__ == '__main__':
    unittest.main()
