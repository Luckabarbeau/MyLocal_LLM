"""Standard-library offline analyzer tests; CLI runs in-process, never subprocess."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_0324_strata_capture as analyzer


CURRENT = '2026-11-20 21:22:52'
HISTORICAL = '2026-11-20 20:40:37'
LATER = '2026-11-20 21:30:00'


def marker(timestamp):
    return f'[strata] {timestamp} engine started: /native/strata --serve --spec 4 --pool-workers 15'


def generation(tokens=12261, elapsed=163053, rate='75.2'):
    return (f'strata serve: prompt 92 tokens = 0 reused + 92 read in 581 ms (158.3 tok/s), '
            f'{tokens} generated in {elapsed} ms ({rate} tok/s), drafts accepted 134 of 362, 1 checkpoints')


def capture(text, split=None):
    return dict(path='/append-only/engine.log', size_bytes=len(text.encode('utf-8')),
                head=text if split is None else text[:split],
                tail='' if split is None else text[split:], truncated=False)


def synthetic_log():
    return '\n'.join([
        generation(rate='999.9'),
        'strata serve: route tail skip: 999 missed experts skipped (1000 entries) since the start',
        marker(HISTORICAL),
        'strata generate: expert arena: cudaHostRegister PORTABLE ok',
        'strata generate: loaded 46.84 GiB at 4.56 GiB/s',
        'strata generate: expert cache 2792 slots, 5.35 GiB of VRAM; policy is',
        'strata generate: CPU pool tasks/phase: 48 (automatic), participating threads: 16',
        'strata generate: native pack: /pack/iq3_s experts',
        'strata generate: PCIe probe: 55.6 GB/s host->device',
        'strata serve: STRATA_ROUTE_TAIL_SKIP=7 is ON (default)',
        generation(),
        'strata serve: route tail skip: 776114 missed experts skipped (777951 entries) since the start',
        'strata serve: adaptive overlap: off',
        'strata serve: asynchronous adaptive tier: 2999 rounds, 235515 experts swapped in',
        'strata serve: suffix drafts: 311 windows, 134 of 362 drafts accepted',
        'ordinary irrelevant progress must not be dumped',
        marker(CURRENT),
        'strata generate: memory guard: limit 38.0 GiB (source: cgroup)',
        'strata generate: not enough RAM for the expert arena: limit 40.00 GiB',
        'strata generate: expert cache 2871 slots, 5.50 GiB of VRAM; policy is',
        'FileExpertSource: allocating 32.82 GiB page-locked cache complement',
    ]) + '\n'


class CaptureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / 'capture.json'
        self.output = self.directory / 'report.json'
        self.write_capture(capture(synthetic_log()))

    def write_capture(self, value):
        self.source.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def analyze(self):
        return analyzer.analyze_capture(self.source, CURRENT)

    def cli(self, extra=None):
        args = ['--current-start', CURRENT, '--engine-capture', str(self.source),
                '--output', str(self.output)]
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = analyzer.main(args if extra is None else extra)
        return result, stdout.getvalue()

    def assert_cli_error(self, args=None):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as caught:
                self.cli(args)
        self.assertEqual(caught.exception.code, 2)
        return stderr.getvalue()

    def test_historical_fast_current_allocating_has_no_fresh_throughput(self):
        report = self.analyze()
        old, current = report['sessions']
        self.assertEqual([old['scope'], current['scope']], ['historical', 'current'])
        stats = old['generation_stats'][0]
        self.assertEqual(stats, dict(
            line=11, scope='request', prompt_tokens=92, reused_tokens=0, read_tokens=92,
            prompt_elapsed_ms=581, prompt_reported_tok_s=158.3, generated_tokens=12261,
            generation_elapsed_ms=163053, generation_reported_tok_s=75.2,
            drafts_accepted=134, drafts_proposed=362, checkpoints=1))
        self.assertEqual(current['generation_stats'], [])
        self.assertFalse(report['current_summary']['fresh_throughput_available'])
        self.assertEqual(report['current_summary']['reported_generation_rates_tok_s'], [])
        self.assertIn('no fresh throughput', report['current_summary']['conclusion'])
        self.assertIn('allocating', current['evidence']['arena_residency'][-1]['text'])
        self.assertNotIn('ordinary irrelevant progress', json.dumps(report))

    def test_exact_marker_and_prefix_exclusion(self):
        report = self.analyze()
        self.assertEqual(report['excluded_unscoped_prefix'],
                         dict(line_count=2, line_span=dict(start=1, end=2)))
        self.assertNotIn('999.9', json.dumps(report))
        self.assertEqual(len(report['sessions'][0]['cumulative_tail_skip']), 1)
        text = ' ' + marker(CURRENT) + '\nnoise ' + marker(CURRENT) + '\n' + marker(CURRENT)
        self.write_capture(capture(text))
        self.assertEqual(len(self.analyze()['sessions']), 1)
        self.assertEqual(self.analyze()['excluded_unscoped_prefix']['line_count'], 2)

    def test_all_sessions_and_later_unexpected_do_not_supply_current_rate(self):
        self.write_capture(capture(synthetic_log() + marker(LATER) + '\n' + generation(rate='800.0')))
        report = self.analyze()
        self.assertEqual([row['scope'] for row in report['sessions']],
                         ['historical', 'current', 'unexpected'])
        self.assertEqual(report['sessions'][2]['generation_stats'][0]['generation_reported_tok_s'], 800.0)
        self.assertFalse(report['current_summary']['fresh_throughput_available'])

    def test_classification_uses_marker_order_not_timestamp_sort(self):
        self.write_capture(capture(marker(LATER) + '\n' + marker(CURRENT) + '\n' + marker(HISTORICAL)))
        self.assertEqual([row['scope'] for row in self.analyze()['sessions']],
                         ['historical', 'current', 'unexpected'])

    def test_current_rates_are_preserved_not_converted_or_maximized(self):
        self.write_capture(capture(synthetic_log() + generation(tokens=1, elapsed=1, rate='12.3') + '\n'
                                   + generation(tokens=2, elapsed=999, rate='4.5')))
        summary = self.analyze()['current_summary']
        self.assertTrue(summary['fresh_throughput_available'])
        self.assertEqual(summary['completed_generation_count'], 2)
        self.assertEqual(summary['reported_generation_rates_tok_s'], [12.3, 4.5])

    def test_scoped_evidence_and_cumulative_tail_are_distinct_from_requests(self):
        old, current = self.analyze()['sessions']
        for category in ['native_argv', 'arena_residency', 'cache', 'pool', 'native',
                         'pcie', 'tail_skip', 'adaptive', 'draft']:
            self.assertTrue(old['evidence'][category], category)
        self.assertEqual(old['cumulative_tail_skip'], [dict(
            line=12, scope='session_cumulative_since_start', missed_experts_skipped=776114, entries=777951)])
        self.assertNotIn('missed_experts_skipped', old['generation_stats'][0])
        self.assertNotIn('fraction', json.dumps(old))
        self.assertNotIn('adaptive', current['evidence'])
        self.assertTrue(current['evidence']['memory_guard'])

    def test_source_sha_and_inclusive_session_spans(self):
        raw = self.source.read_bytes()
        report = self.analyze()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(report['source']['sha256'], digest)
        self.assertEqual(report['source']['capture_bytes'], len(raw))
        self.assertEqual(report['source']['line_count'], 21)
        self.assertEqual([row['line_span'] for row in report['sessions']],
                         [dict(start=3, end=16), dict(start=17, end=21)])
        for session in report['sessions']:
            self.assertEqual(session['source_sha256'], digest)
            for category in session['evidence'].values():
                for row in category:
                    self.assertGreaterEqual(row['line'], session['line_span']['start'])
                    self.assertLessEqual(row['line'], session['line_span']['end'])

    def test_duplicate_and_missing_current_markers(self):
        for text, count in [(marker(CURRENT) + '\n' + marker(CURRENT), 2),
                            (marker(HISTORICAL), 0), ('no markers', 0)]:
            with self.subTest(count=count, text=text):
                self.write_capture(capture(text))
                with self.assertRaisesRegex(ValueError, f'found {count}'):
                    self.analyze()
                self.assertIn(f'found {count}', self.assert_cli_error())
                self.assertFalse(self.output.exists())

    def test_truncated_or_unknown_coverage_rejected_without_output(self):
        for flag in [True, None, 'false', 0]:
            with self.subTest(flag=flag):
                value = capture(synthetic_log(), split=40)
                value['truncated'] = flag
                self.write_capture(value)
                self.assertIn('refusing to stitch gaps', self.assert_cli_error())
                self.assertFalse(self.output.exists())

    def test_unflagged_gap_rejected(self):
        value = capture(synthetic_log(), split=40)
        value['size_bytes'] += 100
        self.write_capture(value)
        with self.assertRaisesRegex(ValueError, 'byte coverage'):
            self.analyze()

    def test_contiguous_head_tail_even_mid_marker_and_unicode(self):
        text = 'prefix café\n' + marker(CURRENT) + '\n'
        self.write_capture(capture(text, split=25))
        report = self.analyze()
        self.assertEqual(report['source']['log_size_bytes'], len(text.encode('utf-8')))
        self.assertEqual(report['sessions'][0]['line_span'], dict(start=2, end=2))

    def test_input_bound_including_exact_limit(self):
        raw = self.source.read_bytes()
        self.source.write_bytes(raw + b' ' * (analyzer.MAX_INPUT_BYTES - len(raw)))
        self.assertEqual(self.analyze()['source']['capture_bytes'], analyzer.MAX_INPUT_BYTES)
        with self.source.open('ab') as stream:
            stream.write(b' ')
        self.assertIn('2 MiB', self.assert_cli_error())
        self.assertFalse(self.output.exists())

    def test_invalid_capture_shapes_and_json(self):
        for value in [[], {}, dict(capture(marker(CURRENT)), head=10),
                      dict(capture(marker(CURRENT)), size_bytes=True)]:
            with self.subTest(value=value):
                self.write_capture(value)
                self.assert_cli_error()
                self.assertFalse(self.output.exists())
        self.source.write_bytes(b'{invalid JSON')
        self.assert_cli_error()
        self.assertFalse(self.output.exists())

    def test_unparsed_generation_not_silently_promoted(self):
        self.write_capture(capture(marker(CURRENT) + '\nstrata serve: prompt 92 tokens, generated in unknown'))
        report = self.analyze()
        self.assertEqual(len(report['sessions'][0]['unparsed_generation_lines']), 1)
        self.assertFalse(report['current_summary']['fresh_throughput_available'])

    def test_large_evidence_line_is_bounded(self):
        self.write_capture(capture(marker(CURRENT) + 'x' * 5000))
        item = self.analyze()['sessions'][0]['evidence']['native_argv'][0]
        self.assertEqual(len(item['text']), 4096)
        self.assertTrue(item['text_truncated'])

    def test_cli_required_arguments_and_timestamp_validation(self):
        full = ['--current-start', CURRENT, '--engine-capture', str(self.source), '--output', str(self.output)]
        for start in [0, 2, 4]:
            with self.subTest(missing=full[start]):
                self.assert_cli_error(full[:start] + full[start + 2:])
        for timestamp in ['2026-11-20', '2026-11-20 1:22:52', '2026-99-20 21:22:52']:
            with self.subTest(timestamp=timestamp):
                self.assert_cli_error(['--current-start', timestamp] + full[2:])
        self.assertFalse(self.output.exists())

    def test_cli_success_source_unchanged_and_deterministic_output(self):
        before = self.source.read_bytes()
        mtime = self.source.stat().st_mtime_ns
        first_analysis = self.analyze()
        self.assertEqual(first_analysis, self.analyze())
        result, stdout = self.cli()
        self.assertEqual(result, 0)
        self.assertIn('fresh throughput available: False', stdout)
        first = self.output.read_bytes()
        self.assertEqual(json.loads(first), first_analysis)
        self.output = self.directory / 'second.json'
        self.cli()
        self.assertEqual(first, self.output.read_bytes())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.source.stat().st_mtime_ns, mtime)

    def test_exclusive_output_preserves_existing_and_source_alias(self):
        self.output.write_bytes(b'keep this existing result')
        self.assert_cli_error()
        self.assertEqual(self.output.read_bytes(), b'keep this existing result')
        original = self.source.read_bytes()
        self.output = self.source
        self.assert_cli_error()
        self.assertEqual(self.source.read_bytes(), original)

    def test_script_entrypoint_without_subprocess(self):
        script = Path(analyzer.__file__)
        with patch.object(sys, 'argv', [str(script), '--current-start', CURRENT,
                                      '--engine-capture', str(self.source), '--output', str(self.output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    runpy.run_path(str(script), run_name='__main__')
        self.assertEqual(caught.exception.code, 0)
        self.assertFalse(json.loads(self.output.read_text())['current_summary']['fresh_throughput_available'])


if __name__ == '__main__':
    unittest.main()
