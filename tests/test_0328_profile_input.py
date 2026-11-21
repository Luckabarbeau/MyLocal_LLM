"""Synthetic CPU tests; importing the input helper must remain stdlib-only."""
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import qwen4exp_profile_input as profile

# Reuse the stdlib fixture factory, not a probe or model implementation.
_spec = importlib.util.spec_from_file_location(
    'profile_input_fixture_0325', Path(__file__).with_name('test_0325_request_placement.py'))
_factory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_factory)


class ProfileInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mini_llm.backend import BACKEND_NAME
        from mini_llm.ops.expert_residency import ExpertKey
        if BACKEND_NAME != 'numpy':
            raise unittest.SkipTest('Run these tests with MINI_LLM_BACKEND=numpy')
        cls.key = ExpertKey

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / 'result.json'
        # Deliberately absent checkpoint files: no weight reads are necessary.
        self.sources = [dict(path=str(self.directory / 'shard.gguf'),
                             size_bytes=12345, mtime_ns=123456789)]
        self.data = _factory.fixture()
        self.data['provenance'] = dict(checkpoint_sources=deepcopy(self.sources))
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, allow_nan=False), encoding='utf-8')

    def load(self, **options):
        arguments = dict(case_index=0, checkpoint_sources=self.sources,
                         model_id='profile-0328', layers=2, experts=20)
        arguments.update(options)
        return profile.load_profile_counts(self.path, **arguments)

    def test_selected_delta_and_complete_provenance(self):
        counts, provenance = self.load(case_index=1)
        self.assertEqual(counts, {self.key('profile-0328', layer, expert): (3 if expert < 10 else 7)
                                  for layer in range(2) for expert in range(20)})
        self.assertTrue(all(type(value) is int and value >= 0 for value in counts.values()))
        self.assertEqual(provenance, dict(
            source=dict(path=str(self.path.resolve()), size_bytes=self.path.stat().st_size,
                        sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()),
            case_index=1, request_id='response-repeat', calls=10, layers=2, experts=20,
            top_k=10, totals=dict(per_layer=[100, 100], all_layers=200),
            inferred_initial_zero=False, checkpoint_sources=self.sources,
            checkpoint_verification='stat_identity_only_not_cryptographic'))
        json.dumps(provenance, allow_nan=False)

    def test_single_completed_case_is_sufficient_and_zeros_are_retained(self):
        self.data['cases'] = self.data['cases'][:1]
        self.save()
        counts, provenance = self.load()
        self.assertEqual(len(counts), 40)
        self.assertEqual(sum(value == 0 for value in counts.values()), 20)
        self.assertEqual(provenance['calls'], 10)

    def test_default_complete_48_by_512_geometry(self):
        sizes = (257,) * 48
        before = _factory.snapshot([[0] * 512 for _ in sizes], calls=0, sizes=sizes)
        after = _factory.snapshot([[7] * 10 + [0] * 502 for _ in sizes], calls=7, sizes=sizes)
        self.data['cases'] = [dict(request_id='full', cache_before=before, cache_after=after)]
        self.save()
        counts, provenance = profile.load_profile_counts(
            self.path, case_index=0, checkpoint_sources=self.sources, model_id='full')
        self.assertEqual(len(counts), 48 * 512)
        self.assertEqual(counts[self.key('full', 47, 511)], 0)
        self.assertEqual(provenance['totals'], dict(per_layer=[70] * 48, all_layers=3360))

    def test_future_content_is_not_validated_or_used(self):
        counts, provenance = self.load()
        for future in ([None, 42, 'not a case'],
                       [dict(request_id=None, cache_after={'native_handoffs': []})],
                       [dict(route_heat=[-1, True, 'invalid'])], []):
            self.data['cases'][1:] = future
            self.save()
            with self.subTest(future=future):
                actual, evidence = self.load()
                self.assertEqual(actual, counts)
                self.assertEqual({k: v for k, v in evidence.items() if k != 'source'},
                                 {k: v for k, v in provenance.items() if k != 'source'})
                self.assertEqual(evidence['source']['sha256'],
                                 hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_future_profile_changes_only_source_hash(self):
        counts, evidence = self.load()
        self.data['cases'][1]['cache_after']['native_handoffs'][0]['route_heat'] = [999] * 20
        self.save()
        actual, changed = self.load()
        self.assertEqual(actual, counts)
        self.assertNotEqual(evidence['source']['sha256'], changed['source']['sha256'])

    def test_initial_empty_requires_unchanged_placement_and_allows_prior_warmup(self):
        self.data['cases'][0]['cache_before']['native_handoffs'] = []
        self.data['cases'].insert(0, dict(request_id='warmup',
            cache_before=dict(native_handoffs=[]), cache_after=dict(native_handoffs=[])))
        self.save()
        counts, evidence = self.load(case_index=1)
        self.assertTrue(evidence['inferred_initial_zero'])
        self.assertEqual(sum(counts.values()), 200)
        with self.assertRaises(ValueError):
            self.load(case_index=0)

    def test_initial_empty_rejects_changed_or_missing_placement_evidence(self):
        original = deepcopy(self.data)
        for field in ('promotions', 'evictions', 'slots', 'missing_promotions', 'bool_evictions'):
            self.data = deepcopy(original)
            before = self.data['cases'][0]['cache_before']
            before['native_handoffs'] = []
            if field == 'slots':
                before['slots']['device_reserved_bytes'] += 256
            elif field == 'missing_promotions':
                before['device'].pop('promotions')
            elif field == 'bool_evictions':
                before['device']['evictions'] = False
            else:
                before['device'][field] += 1
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()

    def test_invalid_index_geometry_and_model_id_before_source_read(self):
        invalid = [dict(case_index=value) for value in (-1, True, 1.0, '0', None, 2**64)]
        invalid += [dict(layers=value) for value in (0, 65, True, 2.0)]
        invalid += [dict(experts=value) for value in (9, 513, False, 20.0)]
        invalid += [dict(model_id=value) for value in ('', ' ', None, 1)]
        for options in invalid:
            with self.subTest(options=options), patch.object(Path, 'open',
                    side_effect=AssertionError('Unexpected source read')), self.assertRaises(ValueError):
                self.load(**options)
        for index in (2, 100):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.load(case_index=index)
        with self.assertRaises(TypeError):
            profile.load_profile_counts(self.path, checkpoint_sources=self.sources, model_id='x')
        for options in (dict(layers=3), dict(experts=21)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(**options)

    def test_missing_partial_unprofiled_and_incomplete_cases_rejected(self):
        cases = deepcopy(self.data['cases'])
        for replacement in (None, [], {}, [None], [dict(request_id='incomplete')],
                            [dict(request_id='warmup', cache_before=dict(native_handoffs=[]),
                                  cache_after=dict(native_handoffs=[]))]):
            self.data['cases'] = replacement
            self.save()
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.load()
        self.data['cases'] = cases
        for field in ('cache_before', 'cache_after'):
            for rows in (None, [], [dict(layer=0)]):
                self.data['cases'] = deepcopy(cases)
                self.data['cases'][1][field]['native_handoffs'] = rows
                self.save()
                with self.subTest(field=field, rows=rows), self.assertRaises(ValueError):
                    self.load(case_index=1)

    def test_strict_snapshot_counters_geometry_and_sized_not_cg_slots(self):
        changes = [
            lambda c: c.pop('slots'),
            lambda c: c.update(slots=None),
            lambda c: c.update(slots=dict(capacity=20, stride_bytes=768)),
            lambda c: c['slots'].pop('size_classes'),
            lambda c: c['slots'].update(device_reserved_bytes=1),
            lambda c: c['slots']['size_classes']['512'].update(free=1),
            lambda c: c['slots']['size_classes']['512'].update(capacity=True),
            lambda c: c['native_handoffs'].pop(),
            lambda c: c['native_handoffs'][1].update(layer=0),
            lambda c: c['native_handoffs'][0].update(calls=10.0),
            lambda c: c['native_handoffs'][0].update(layer=True),
            lambda c: c['native_handoffs'][0].pop('expert_packed_bytes'),
            lambda c: c['native_handoffs'][0].pop('profile_resident_ids'),
            lambda c: c['native_handoffs'][0].pop('route_heat'),
            lambda c: c['native_handoffs'][0]['route_heat'].pop(),
            lambda c: c['native_handoffs'][0].update(profile_resident_ids=[0, 0]),
            lambda c: c['native_handoffs'][0].update(formats=['IQ2_S']),
        ]
        for value in (True, 0.0, -1, 2**64, '0', None):
            changes.append(lambda c, v=value: c['native_handoffs'][0]['route_heat'].__setitem__(0, v))
        original = deepcopy(self.data)
        for field in ('cache_before', 'cache_after'):
            for index, change in enumerate(changes):
                self.data = deepcopy(original)
                change(self.data['cases'][0][field])
                self.save()
                with self.subTest(field=field, change=index), self.assertRaises(ValueError):
                    self.load()

    def test_delta_rejects_resets_gaps_zero_calls_and_unique_route_violation(self):
        original = deepcopy(self.data)
        for problem in ('reset', 'gap', 'zero_calls', 'unique_routes', 'heat_sum'):
            self.data = deepcopy(original)
            case = self.data['cases'][1]
            if problem == 'reset':
                case['cache_before'] = deepcopy(original['cases'][0]['cache_before'])
            elif problem == 'gap':
                for field in ('cache_before', 'cache_after'):
                    for row in case[field]['native_handoffs']:
                        row['calls'] += 1
                        row['route_heat'] = [v + int(i < 10) for i, v in enumerate(row['route_heat'])]
            elif problem == 'zero_calls':
                case['cache_after'] = deepcopy(case['cache_before'])
            elif problem == 'unique_routes':
                for row in case['cache_after']['native_handoffs']:
                    row['route_heat'] = [20] + [0] * 9 + [18] * 10
            else:
                case['cache_after']['native_handoffs'][0]['route_heat'][0] += 1
            self.save()
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                self.load(case_index=1)
        # A complete but smaller after snapshot also fails within the selected case.
        self.data = deepcopy(original)
        self.data['cases'] = [dict(request_id='reset',
            cache_before=original['cases'][1]['cache_after'],
            cache_after=original['cases'][0]['cache_after'])]
        self.save()
        with self.assertRaisesRegex(ValueError, 'reset|regressed'):
            self.load()

    def test_delta_rejects_resident_format_size_and_layout_changes(self):
        original = deepcopy(self.data)
        for change in ('residents', 'formats', 'size', 'layout'):
            self.data = deepcopy(original)
            cache = self.data['cases'][0]['cache_after']
            if change == 'residents':
                cache['native_handoffs'][0]['profile_resident_ids'][-1] = 19
            elif change == 'formats':
                cache['native_handoffs'][0]['formats'] = ['A', 'A', 'B']
            elif change == 'size':
                cache['native_handoffs'][0]['expert_packed_bytes'] = 258
            else:
                cache['slots']['size_classes']['512']['capacity'] += 1
                cache['slots']['size_classes']['512']['free'] += 1
                cache['slots']['capacity'] += 1
                cache['slots']['device_reserved_bytes'] += 512
            self.save()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_earlier_invalid_profile_not_hidden_by_valid_selected_case(self):
        self.data['cases'][0]['cache_after']['native_handoffs'][0]['route_heat'][0] = True
        self.save()
        with self.assertRaises(ValueError):
            self.load(case_index=1)

    def test_provenance_matching_canonical_paths_order_and_no_input_mutation(self):
        second = dict(path=str(self.directory / 'second.gguf'), size_bytes=99, mtime_ns=77)
        self.sources.append(second)
        self.data['provenance']['checkpoint_sources'].append(deepcopy(second))
        alias = self.directory / 'alias.gguf'
        alias.symlink_to(self.directory / 'shard.gguf')
        self.data['provenance']['checkpoint_sources'][0]['path'] = str(alias)
        self.data['provenance']['checkpoint_sources'].reverse()
        self.save()
        saved_sources, saved_data = deepcopy(self.sources), deepcopy(self.data)
        _, evidence = self.load(checkpoint_sources=tuple(self.sources))
        self.assertEqual(evidence['checkpoint_sources'], sorted(self.sources, key=lambda r: r['path']))
        self.assertEqual(self.sources, saved_sources)
        self.assertEqual(self.data, saved_data)
        evidence['checkpoint_sources'][0]['size_bytes'] = 1
        self.assertEqual(self.sources, saved_sources)

    def test_checkpoint_identity_mismatch_rejected(self):
        for field, value in (('path', str(self.directory / 'other.gguf')),
                             ('size_bytes', 12346), ('mtime_ns', 123456790)):
            current = deepcopy(self.sources)
            current[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identities'):
                self.load(checkpoint_sources=current)
        with self.assertRaises(ValueError):
            self.load(checkpoint_sources=self.sources + [dict(
                path=str(self.directory / 'extra.gguf'), size_bytes=1, mtime_ns=1)])

    def test_strict_recorded_and_current_checkpoint_provenance(self):
        invalid = [None, [], {}, [None], [dict(path='x')], self.sources * 2]
        for field in ('size_bytes', 'mtime_ns'):
            for value in (True, 1.0, '1', -1, None, 2**64):
                record = deepcopy(self.sources[0])
                record[field] = value
                invalid.append([record])
        for value in ('', ' ', None, 1, 'bad\x00path'):
            invalid.append([dict(self.sources[0], path=value)])
        invalid.extend(([dict(self.sources[0], size_bytes=0)],
                        [dict(self.sources[0], sha256='not a checkpoint hash')]))
        for records in invalid:
            for side in ('recorded', 'current'):
                self.data['provenance']['checkpoint_sources'] = (
                    deepcopy(records) if side == 'recorded' else deepcopy(self.sources))
                self.save()
                with self.subTest(side=side, records=records), self.assertRaises(ValueError):
                    self.load(checkpoint_sources=records if side == 'current' else self.sources)
        for provenance in (None, {}, dict(checkpoint_sources=[])):
            self.data['provenance'] = provenance
            self.save()
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                self.load()

    def test_duplicate_canonical_source_alias_rejected(self):
        alias = self.directory / 'alias.gguf'
        alias.symlink_to(self.directory / 'shard.gguf')
        duplicate = dict(self.sources[0], path=str(alias))
        with self.assertRaisesRegex(ValueError, 'Duplicate canonical'):
            self.load(checkpoint_sources=self.sources + [duplicate])

    def test_source_opened_read_once_and_existing_artifacts_unchanged(self):
        artifact = self.directory / 'placement_profile.json'
        artifact.write_bytes(b'existing artifact\n')
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        open_ = Path.open
        reads = []
        streams = []

        def tracked_open(path, mode='r', *args, **kwargs):
            self.assertEqual(path, self.path.resolve())
            self.assertEqual(mode, 'rb')
            stream = open_(path, mode, *args, **kwargs)
            spy = Mock(wraps=stream)
            streams.append(spy)
            class Reader:
                def __enter__(self):
                    return spy
                def __exit__(self, *error):
                    stream.close()
            reads.append(path)
            return Reader()

        with patch.object(Path, 'open', tracked_open):
            self.load()
        self.assertEqual(reads, [self.path.resolve()])
        streams[0].read.assert_called_once_with(profile.MAX_SOURCE_BYTES + 1)
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, before)
        self.data['cases'][0]['cache_after']['native_handoffs'] = []
        self.save()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, before)

    def test_source_bound_precheck_growth_and_exact_limit(self):
        with self.path.open('wb') as stream:
            stream.truncate(profile.MAX_SOURCE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, '64MiB'):
            self.load()
        self.save()
        size = self.path.stat().st_size
        with patch.object(profile, 'MAX_SOURCE_BYTES', size):
            self.load()
        with patch.object(profile, 'MAX_SOURCE_BYTES', size - 1), patch.object(
                profile.os, 'fstat', return_value=SimpleNamespace(st_size=0)):
            with self.assertRaisesRegex(ValueError, '64MiB'):
                self.load()

    def test_invalid_json_including_future_content_rejected(self):
        for raw in ('{"cases": [], "cases": []}', '{', '[]',
                    '{"cases": [null, {"future": NaN}]}',
                    '{"cases": [null, {"future": Infinity}]}',
                    '{"cases": [null, {"future": 1, "future": 2}]}'):
            self.path.write_text(raw, encoding='utf-8')
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load()

    def test_import_is_stdlib_only_and_invalid_input_never_imports_model(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mini_llm', 'numpy', 'cupy', 'torch', 'subprocess', 'scripts'):
                raise AssertionError(f'Forbidden import dependency: {name}')
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location('stdlib_profile_input', profile.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.__import__', side_effect=guarded):
            spec.loader.exec_module(module)
        def no_model(name, *args, **kwargs):
            if name.split('.')[0] in ('mini_llm', 'numpy', 'cupy', 'torch', 'subprocess'):
                raise AssertionError(f'Invalid input imported model dependency: {name}')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_model), self.assertRaises(ValueError):
            module.load_profile_counts(self.path, case_index=True,
                                       checkpoint_sources=self.sources, model_id='x')


if __name__ == '__main__':
    unittest.main()
