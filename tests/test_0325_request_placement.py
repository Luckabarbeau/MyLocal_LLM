"""Synthetic standard-library-only tests: no model, backend, GPU or benchmark."""
import builtins
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_0325_request_placement as analysis


EXPERTS = 20


def heat(old=0, new=10):
    return [old] * 10 + [new] * 10


def snapshot(counts, *, calls, sizes=(257, 513), residents=None, formats=None):
    if residents is None:
        residents = [list(range(10)) for _ in sizes]
    classes = {}
    for size, ids in zip(sizes, residents):
        cls = str((size + 255) // 256 * 256)
        entry = classes.setdefault(cls, dict(capacity=0, free=0))
        entry['capacity'] += len(ids)
    rows = []
    for layer, (values, size, ids) in enumerate(zip(counts, sizes, residents)):
        row = dict(layer=layer, calls=calls, route_heat=list(values),
                   profile_resident_ids=list(ids), expert_packed_bytes=size)
        if formats is not None and formats[layer] is not None:
            row['formats'] = list(formats[layer])
        rows.append(row)
    return dict(native_handoffs=rows,
                device=dict(promotions=20, evictions=0),
                slots=dict(size_classes=classes, capacity=sum(v['capacity'] for v in classes.values()),
                           device_reserved_bytes=sum(int(s) * v['capacity'] for s, v in classes.items())))


def fixture(deltas=None, *, sizes=(257, 513), formats=None):
    if deltas is None:
        deltas = [[heat() for _ in sizes], [heat(3, 7) for _ in sizes]]
    cumulative = [[0] * EXPERTS for _ in sizes]
    before = snapshot(cumulative, calls=0, sizes=sizes, formats=formats)
    calls = 0
    cases = []
    for counts in deltas:
        if len({sum(v) for v in counts}) != 1 or sum(counts[0]) % 10:
            raise AssertionError('Invalid synthetic fixture')
        calls += sum(counts[0]) // 10
        cumulative = [[a + b for a, b in zip(old, new)] for old, new in zip(cumulative, counts)]
        after = snapshot(cumulative, calls=calls, sizes=sizes, formats=formats)
        cases.append(dict(request_id='response-repeat', cache_before=deepcopy(before), cache_after=after))
        before = after
    return dict(cases=cases)


def analyze(data, **options):
    sizes = len(data['cases'][-1]['cache_after']['native_handoffs'])
    return analysis.analyze_result(data, layers=sizes, experts=EXPERTS, **options)


class RequestPlacementTests(unittest.TestCase):
    def test_metrics_causal_caps_and_no_input_mutation(self):
        data = fixture()
        original = deepcopy(data)
        report = analyze(data)
        self.assertEqual(data, original)
        self.assertFalse(report['model_movement'])
        boundary = report['boundaries'][0]
        self.assertEqual(boundary['swap_count'], 8)
        self.assertEqual(boundary['incoming_transfer_bytes'], 8 * 257)
        self.assertEqual(boundary['swaps'][0]['incoming'], [0, 10])
        self.assertEqual(boundary['swaps'][0]['victim'], [0, 0])
        metrics = boundary['evaluation']['total']
        self.assertEqual(metrics, dict(routes=200, baseline_misses=140,
                                      proposed_misses=108, saved_misses=56,
                                      displaced_hits=24, net_saved_misses=32))
        self.assertEqual([g['slot_class_bytes'] for g in boundary['evaluation']['by_class']], [512, 768])
        self.assertEqual(boundary['evaluation']['by_class'][0]['net_saved_misses'], 32)
        self.assertEqual(boundary['evaluation']['by_format'], [])
        self.assertFalse(boundary['evaluation']['format_attribution']['complete'])
        self.assertNotIn('throughput', boundary)

    def test_future_counts_never_change_earlier_proposal(self):
        a = analyze(fixture())['boundaries'][0]
        b = analyze(fixture(deltas=[[heat(), heat()], [heat(10, 0), heat(10, 0)]]))['boundaries'][0]
        self.assertEqual(a['swaps'], b['swaps'])
        self.assertEqual(b['evaluation']['total']['saved_misses'], 0)
        self.assertEqual(b['evaluation']['total']['displaced_hits'], 80)
        self.assertEqual(b['evaluation']['total']['net_saved_misses'], -80)

    def test_only_previous_delta_not_entire_history_or_rolling_placement(self):
        counts = [[heat(), heat()], [heat(10, 0), heat(10, 0)], [heat(), heat()]]
        report = analyze(fixture(counts))
        self.assertEqual(len(report['boundaries']), 2)
        self.assertEqual(report['boundaries'][0]['swap_count'], 8)
        self.assertEqual(report['boundaries'][1]['swaps'], [])
        shorter = analyze(fixture(counts[:2]))
        self.assertEqual(report['boundaries'][0], shorter['boundaries'][0])

    def test_zero_swap_or_byte_cap_preserves_all_counts(self):
        for options in (dict(max_swaps=0), dict(max_transfer_bytes=0)):
            with self.subTest(options=options):
                boundary = analyze(fixture(), **options)['boundaries'][0]
                self.assertEqual(boundary['swaps'], [])
                self.assertEqual(boundary['incoming_transfer_bytes'], 0)
                for values in [boundary['evaluation']['total'], *boundary['evaluation']['by_class']]:
                    self.assertEqual(values['baseline_misses'], values['proposed_misses'])
                    self.assertEqual(values['net_saved_misses'], 0)
                    self.assertEqual(values['displaced_hits'], 0)

    def test_32mib_cap_counts_complete_raw_incoming_bundles(self):
        data = fixture(sizes=(8 * 1024**2,))
        boundary = analyze(data)['boundaries'][0]
        self.assertEqual(boundary['swap_count'], 4)
        self.assertEqual(boundary['incoming_transfer_bytes'], 32 * 1024**2)
        small = analyze(fixture(), max_transfer_bytes=2 * 257)['boundaries'][0]
        self.assertEqual(small['swap_count'], 2)
        self.assertEqual(small['incoming_transfer_bytes'], 514)

    def test_oversized_candidate_skipped_without_partial_bundle(self):
        data = fixture(sizes=(33 * 1024**2, 513))
        boundary = analyze(data)['boundaries'][0]
        self.assertEqual(boundary['swap_count'], 8)
        self.assertTrue(all(s['incoming'][0] == 1 for s in boundary['swaps']))

    def test_swaps_never_cross_unequal_alignment_classes(self):
        data = fixture(deltas=[[heat(10, 0), heat()], [heat(), heat()]], sizes=(257, 513))
        boundary = analyze(data)['boundaries'][0]
        self.assertTrue(boundary['swaps'])
        self.assertTrue(all(s['incoming'][0] == s['victim'][0] == 1 for s in boundary['swaps']))
        self.assertTrue(all(s['slot_class_bytes'] == 768 for s in boundary['swaps']))

    def test_cross_layer_equal_class_and_deterministic_ties(self):
        data = fixture(deltas=[[heat(10, 0), heat()], [heat(), heat()]], sizes=(257, 511))
        original = analyze(data)
        swaps = original['boundaries'][0]['swaps']
        self.assertEqual(swaps[0]['incoming'], [1, 10])
        self.assertEqual(swaps[0]['victim'], [1, 0])
        # Equal frequencies break ties by incoming layer/expert then victim.
        tied = fixture(sizes=(257, 511))
        want = analyze(tied)
        for case in tied['cases']:
            for cache in ('cache_before', 'cache_after'):
                case[cache]['native_handoffs'].reverse()
                case[cache]['slots']['size_classes'] = dict(reversed(list(case[cache]['slots']['size_classes'].items())))
                for row in case[cache]['native_handoffs']:
                    row['profile_resident_ids'].reverse()
        self.assertEqual(analyze(tied), want)
        self.assertEqual(want['boundaries'][0]['swaps'][0]['incoming'], [0, 10])

    def test_actual_smallest_fitting_class_not_unrecorded_rounding(self):
        data = fixture(sizes=(257, 513))
        # This recorded layout has only 768-byte slots. Both bundle sizes fit.
        for case in data['cases']:
            for key in ('cache_before', 'cache_after'):
                case[key]['slots'] = dict(size_classes={'768': dict(capacity=20, free=0)},
                                          capacity=20, device_reserved_bytes=20 * 768)
        report = analyze(data)
        self.assertEqual(len(report['boundaries'][0]['evaluation']['by_class']), 1)
        self.assertEqual(report['boundaries'][0]['swaps'][0]['slot_class_bytes'], 768)

    def test_no_positive_training_gain_means_no_churn(self):
        data = fixture(deltas=[[heat(5, 5), heat(5, 5)], [heat(), heat()]])
        self.assertEqual(analyze(data)['boundaries'][0]['swaps'], [])

    def test_format_attribution_only_when_explicit(self):
        fmt = ('IQ2_S', 'IQ2_S', 'Q2_0')
        report = analyze(fixture(formats=(fmt, None)))
        evaluation = report['boundaries'][0]['evaluation']
        self.assertEqual(evaluation['format_attribution']['missing_layers'], [1])
        self.assertEqual(evaluation['by_format'][0]['formats'], list(fmt))
        self.assertEqual(evaluation['by_format'][0]['routes'], 100)
        full = analyze(fixture(formats=(fmt, fmt)))['boundaries'][0]['evaluation']
        self.assertTrue(full['format_attribution']['complete'])
        self.assertEqual(full['by_format'][0]['routes'], 200)

    def test_initial_empty_handoffs_and_unprofiled_warmup(self):
        data = fixture()
        data['cases'][0]['cache_before']['native_handoffs'] = []
        data['cases'].insert(0, dict(request_id='warmup',
                                   cache_before=dict(native_handoffs=[]), cache_after=dict(native_handoffs=[])))
        report = analyze(data)
        self.assertEqual(report['inferred_initial_zero_cases'], [1])
        self.assertEqual(report['skipped_cases'][0]['case_index'], 0)
        self.assertEqual(report['boundaries'][0]['training_case_index'], 1)

    def test_initial_empty_requires_unchanged_placement_evidence(self):
        for change in ('missing', 'changed'):
            data = fixture()
            data['cases'][0]['cache_before']['native_handoffs'] = []
            if change == 'missing':
                del data['cases'][0]['cache_before']['device']['promotions']
            else:
                data['cases'][0]['cache_after']['device']['evictions'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(data)

    def test_later_empty_or_partial_profile_rejected(self):
        for rows in ([], [dict(layer=0)]):
            data = fixture()
            data['cases'][1]['cache_before']['native_handoffs'] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analyze(data)

    def test_missing_metadata_invalid_integer_and_geometry(self):
        changes = [
            lambda c: c.pop('slots'),
            lambda c: c['slots'].pop('size_classes'),
            lambda c: c['native_handoffs'][0].pop('expert_packed_bytes'),
            lambda c: c['native_handoffs'][0].pop('profile_resident_ids'),
            lambda c: c['native_handoffs'][0].pop('route_heat'),
            lambda c: c['native_handoffs'][0].update(route_heat=None),
            lambda c: c['native_handoffs'][0].update(calls=10.0),
            lambda c: c['native_handoffs'][0].update(layer=True),
            lambda c: c['native_handoffs'][0].update(expert_packed_bytes=True),
            lambda c: c['native_handoffs'][0].update(profile_resident_ids=[0, 0]),
            lambda c: c['native_handoffs'][0].update(profile_resident_ids=[True]),
            lambda c: c['native_handoffs'][0].update(profile_resident_ids=[20]),
            lambda c: c['native_handoffs'][0]['route_heat'].__setitem__(0, 0.0),
            lambda c: c['native_handoffs'][0]['route_heat'].__setitem__(0, True),
            lambda c: c['native_handoffs'][0]['route_heat'].__setitem__(0, -1),
            lambda c: c['native_handoffs'][0]['route_heat'].__setitem__(0, 2**64),
            lambda c: c['native_handoffs'][0]['route_heat'].pop(),
            lambda c: c['native_handoffs'].pop(),
            lambda c: c['native_handoffs'][1].update(layer=0),
            lambda c: c['native_handoffs'][1].update(layer=2),
            lambda c: c['slots']['size_classes']['512'].update(capacity=10.0),
            lambda c: c['slots'].update(device_reserved_bytes=1),
            lambda c: c['native_handoffs'][0].update(formats=['IQ2_S']),
        ]
        for index, change in enumerate(changes):
            data = fixture()
            change(data['cases'][0]['cache_after'])
            with self.subTest(change=index), self.assertRaises(ValueError):
                analyze(data)

    def test_heat_sum_and_call_delta_contracts(self):
        data = fixture()
        data['cases'][0]['cache_after']['native_handoffs'][0]['route_heat'][0] = 1
        with self.assertRaisesRegex(ValueError, 'heat'):
            analyze(data)
        # Every individual cumulative snapshot is plausible, but a request's
        # delta cannot select one expert more than once per native call.
        data = fixture()
        for row in data['cases'][1]['cache_after']['native_handoffs']:
            row['route_heat'] = [20] + [0] * 9 + [18] * 10
        with self.assertRaisesRegex(ValueError, 'delta'):
            analyze(data)

    def test_counter_reset_and_unattributed_gaps(self):
        data = fixture()
        data['cases'][1]['cache_before'] = deepcopy(data['cases'][0]['cache_before'])
        with self.assertRaisesRegex(ValueError, 'reset|regressed'):
            analyze(data)
        data = fixture()
        for case in data['cases'][1:]:
            for key in ('cache_before', 'cache_after'):
                for row in case[key]['native_handoffs']:
                    row['calls'] += 1
                    row['route_heat'] = [v + int(e < 10) for e, v in enumerate(row['route_heat'])]
        with self.assertRaisesRegex(ValueError, 'Unattributed'):
            analyze(data)

    def test_changed_residents_formats_size_or_layout_rejected(self):
        for change in ('residents', 'formats', 'size', 'layout'):
            data = fixture()
            cache = data['cases'][1]['cache_after']
            if change == 'residents':
                cache['native_handoffs'][0]['profile_resident_ids'][-1] = 19
            elif change == 'formats':
                cache['native_handoffs'][0]['formats'] = ['A', 'A', 'B']
            elif change == 'size':
                cache['native_handoffs'][0]['expert_packed_bytes'] = 258
            else:
                cache['slots']['size_classes']['512']['free'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(data)

    def test_invalid_caps_and_missing_cases_fail_closed(self):
        for value in (-1, 9, True, 1.0):
            with self.subTest(swaps=value), self.assertRaises(ValueError):
                analyze(fixture(), max_swaps=value)
        for value in (-1, 32 * 1024**2 + 1, True, 1.0):
            with self.subTest(bytes=value), self.assertRaises(ValueError):
                analyze(fixture(), max_transfer_bytes=value)
        for data in ({}, dict(cases=[]), dict(cases=fixture()['cases'][:1])):
            with self.subTest(data=data), self.assertRaises(ValueError):
                analysis.analyze_result(data, layers=2, experts=EXPERTS)
        with self.assertRaises(ValueError):
            analysis.analyze_result(fixture(), layers=3, experts=20)


class FileInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / 'result.json'
        self.output = self.directory / 'analysis.json'
        self.source.write_text(json.dumps(fixture()), encoding='utf-8')

    def arguments(self):
        return ['--input', str(self.source), '--output', str(self.output), '--layers', '2', '--experts', '20']

    def test_cli_exclusive_json_hash_and_source_unchanged(self):
        raw = self.source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(analysis.main(self.arguments()), 0)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['source']['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report['source']['size_bytes'], len(raw))
        self.assertEqual(self.source.read_bytes(), raw)
        self.assertIn('count-only', output.getvalue())
        saved = self.output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            analysis.main(self.arguments())
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.output.read_bytes(), saved)

    def test_invalid_source_never_creates_output(self):
        self.source.write_text('{"cases": []}', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analysis.main(self.arguments())
        self.assertFalse(self.output.exists())

    def test_bounded_source_and_duplicate_json_keys(self):
        with self.source.open('wb') as stream:
            stream.truncate(analysis.MAX_SOURCE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, '64MiB'):
            analysis.analyze_file(self.source, layers=2, experts=20)
        self.source.write_text('{"cases": [], "cases": []}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            analysis.analyze_file(self.source)

    def test_cli_zero_cap_and_out_of_range_cap(self):
        with contextlib.redirect_stdout(io.StringIO()):
            analysis.main(self.arguments() + ['--max-swaps', '0'])
        self.assertEqual(json.loads(self.output.read_text())['boundaries'][0]['swaps'], [])
        self.output.unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analysis.main(self.arguments() + ['--max-transfer-mib', '33'])
        self.assertFalse(self.output.exists())

    def test_import_has_no_backend_numpy_model_or_execution_dependency(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mini_llm', 'numpy', 'cupy', 'torch', 'subprocess'):
                raise AssertionError(f'Forbidden diagnostic dependency: {name}')
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location('offline_0325', analysis.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.__import__', side_effect=guarded):
            spec.loader.exec_module(module)
            report = module.analyze_result(fixture(), layers=2, experts=20)
        self.assertFalse(report['model_movement'])


if __name__ == '__main__':
    unittest.main()
