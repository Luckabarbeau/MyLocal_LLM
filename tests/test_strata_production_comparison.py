"""CPU-only contract gates. These mocks are never performance evidence."""
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('production', SCRIPTS / 'run_strata_production_comparison.py')
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


class ProductionContractTests(unittest.TestCase):
    def test_config_is_copied_and_single_gpu(self):
        original = dict(exe='component:strata-engine', args=['--spec', '4'], gpu=[1, 0],
                        gpu_roles={'0': 'main', '1': 'helper'}, mtp=False)
        cfg = production.guarded_config(original, Path('/explicit/strata'), Path('/new/artifacts'), 1234)
        self.assertEqual(original['gpu'], [1, 0])
        self.assertEqual(original['args'], ['--spec', '4'])
        self.assertEqual(cfg['gpu'], [0])
        self.assertEqual(cfg['gpu_roles'], {'0': 'main'})
        self.assertFalse(cfg['mtp'])
        self.assertIn('--ram-keep-gib', cfg['args'])
        self.assertNotIn('--mmap-experts', cfg['args'])

    def test_mmap_and_hooks_rejected(self):
        for cfg in (dict(args=['--mmap-experts']), dict(args=[], before_load='stop-other')):
            with self.assertRaises(ValueError):
                production.guarded_config(cfg, Path('/exe'), Path('/out'), 1)

    def test_low_headroom_never_spawns_and_records_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'serve').mkdir()
            (root / 'serve/server.py').write_text('mock server identity')
            engine = root / 'engine'
            engine.write_bytes(b'mock engine identity')
            config = root / 'config.json'
            original = dict(args=['--spec', '4'], model_name='mock', mtp=False, gpu=[1, 0])
            config.write_text(json.dumps(original))
            manifest = dict(context=262144, requests=[])
            manifest['workload_sha256'] = production.digest(json.dumps(
                manifest, ensure_ascii=False, sort_keys=True).encode('utf-8'))
            with patch.object(production, 'TERMINAL', root), \
                 patch.object(production, 'prepare', return_value=manifest), \
                 patch.object(production, 'memory', return_value={'MemAvailable': 47 * production.GIB}), \
                 patch.object(production.subprocess, 'check_output', side_effect=[
                     '0, ' + production.GPU + ', NVIDIA GeForce RTX 5080\n', '']), \
                 patch.object(production.subprocess, 'Popen') as spawn:
                code = production.main(['--output', str(root / 'out'), '--config', str(config),
                                        '--engine', str(engine)])
            self.assertEqual(code, 1)
            spawn.assert_not_called()
            self.assertEqual(json.loads(config.read_text()), original)
            saved = json.loads((root / 'out/runconfig.json').read_text())
            self.assertEqual(saved['gpu'], [0])
            summary = json.loads((root / 'out/summary.json').read_text())
            self.assertIn('Insufficient MemAvailable', summary['error'])
            self.assertEqual(summary['requests_completed'], 0)
            self.assertFalse(summary['production_comparison_accepted'])

    def test_two_matching_tokens_cannot_accept(self):
        response = dict(ok=True, done=True, usage_available=True,
                        tokens=dict(completion_tokens=2, prompt_tokens=128, reused_prompt_tokens=0),
                        production_rates=dict(uncached_prompt_tokens_per_prefill_second=3000,
                                              emitted_tokens_per_active_decode_second=100))
        result = production.acceptance([dict(size='128', sizing_prompt_tokens=128, response=response)])
        self.assertEqual(result['status'], 'pending')
        self.assertFalse(result['production_comparison_accepted'])
        self.assertTrue(any('too short' in x for x in result['reasons']))

    def test_exact_wire_bytes_and_emitted_rate(self):
        class Response:
            status = 200
            headers = {'Content-Type': 'text/event-stream'}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read1(self, count):
                body, self.body = self.body, b''
                return body

        class HTTP:
            timeout = 2
            def snapshot(self, paths):
                return {}
            def open(self, path, body, stream):
                self.sent = body
                response = Response()
                event = dict(choices=[dict(delta={'content': 'natural answer'}, finish_reason='stop')],
                    usage=dict(prompt_tokens=128, completion_tokens=150),
                    timings=dict(prompt_n=128, cache_n=0, prompt_ms=50, predicted_ms=2000,
                                 predicted_per_second=999))
                response.body = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
                return response

        raw = b'{ "stream": true, "model": "mock", "max_tokens": 256 }\n'
        http = HTTP()
        with tempfile.TemporaryDirectory() as directory:
            result = production.bench.benchmark_request(http, json.loads(raw), Path(directory) / 'request',
                                                        {'context': 262144}, raw_body=raw)
            self.assertTrue(result['ok'])
            self.assertEqual(http.sent, raw)
            self.assertEqual((Path(directory) / 'request/request.json').read_bytes(), raw)
            self.assertEqual(result['production_rates']['emitted_tokens_per_active_decode_second'], 75)
            self.assertEqual(result['production_rates']['uncached_prompt_tokens_per_prefill_second'], 2560)
            self.assertEqual(result['production_rates']['context_occupancy_tokens'], 278)

    def test_unknown_cache_is_not_fresh(self):
        response = dict(ok=True, done=True, usage_available=True,
                        tokens=dict(completion_tokens=256, prompt_tokens=128, reused_prompt_tokens=None),
                        production_rates=dict(uncached_prompt_tokens_per_prefill_second=3000,
                                              emitted_tokens_per_active_decode_second=100))
        result = production.acceptance([dict(size='128', sizing_prompt_tokens=128, response=response)], expected=1)
        self.assertEqual(result['status'], 'pending')
        self.assertTrue(any('reuse unverified' in x for x in result['reasons']))


if __name__ == '__main__':
    unittest.main()
