"""Stdlib-only hot-trace contracts; no real runner, model, backend or GPU imports."""
import builtins
from contextlib import ExitStack, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    spec = importlib.util.spec_from_file_location('hot_trace_probe',
        ROOT / 'scripts/probe_0266_critical_path.py')
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', list(sys.path)):
        spec.loader.exec_module(module)
    return module


def fake_module(name, **attributes):
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


class Reader:
    def read(self):
        return 'read'


class Harness:
    def __init__(self, test, *, fail_position=None, last_position=80, shared_readers=False):
        self.test = test
        self.fail_position = fail_position
        self.last_position = last_position
        self.events = []
        self.ranges = []
        self.stack = []
        self.active = False
        self.phase = None
        self.args = None
        self.returns = []
        self.original_sample = lambda *a, **k: 7
        self.generation = fake_module('mini_llm.generation', sample_token=self.original_sample)
        self.xp = SimpleNamespace(cuda=SimpleNamespace(
            profiler=SimpleNamespace(start=self.start, stop=self.stop),
            nvtx=SimpleNamespace(RangePush=self.push, RangePop=self.pop),
            get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize)))
        noop = lambda *a, **k: None
        pool = SimpleNamespace(submit=noop, _submit_registered=noop, _finish=noop)
        executor = SimpleNamespace(try_forward_device_plan=noop,
            try_forward_decode_batch=noop, _prepare=noop, _ggml_row_pool=pool)
        attention = Reader()
        ffn = attention if shared_readers else Reader()
        block = SimpleNamespace(attention_hc=attention, ffn_hc=ffn,
            mixer=SimpleNamespace(step=noop), moe=SimpleNamespace(router=SimpleNamespace(forward=noop)))
        self.model = SimpleNamespace(expert_executor=executor, blocks=[block], _step=self.step)
        self.owners = [self.model, executor, pool, attention, ffn, block.mixer, block.moe.router]
        self.snapshots = [(owner, vars(owner).copy()) for owner in self.owners]
        self.runner = fake_module('probe_0254_strata_workload',
            replay_case=self.replay, child=self.child)
        self.original_replay = self.runner.replay_case
        self.modules = {
            'run_0290_isolated_benchmark': fake_module('run_0290_isolated_benchmark',
                require_isolation=lambda: self.events.append('isolation')),
            'mini_llm': fake_module('mini_llm'),
            'mini_llm.backend': fake_module('mini_llm.backend', xp=self.xp),
            'mini_llm.generation': self.generation,
            'probe_0254_strata_workload': self.runner,
        }

    def start(self):
        self.test.assertFalse(self.active)
        self.active = True
        self.events.append(('start', self.phase))

    def stop(self):
        self.test.assertTrue(self.active)
        self.test.assertFalse(self.stack)
        self.active = False
        self.events.append(('stop', self.phase))

    def synchronize(self):
        self.events.append(('sync', self.phase))

    def push(self, name):
        self.test.assertTrue(self.active)
        self.stack.append(name)
        self.ranges.append((self.phase, name, len(self.stack)))

    def pop(self):
        self.stack.pop()

    def assert_restored(self):
        for owner, snapshot in self.snapshots:
            self.test.assertEqual(vars(owner), snapshot)
        self.test.assertIs(self.generation.sample_token, self.original_sample)

    def step(self, token, state, **kwargs):
        self.test.assertEqual(token, 7)
        self.test.assertEqual(kwargs, {'sentinel': 'kept'})
        executor = self.model.expert_executor
        executor.try_forward_device_plan()
        executor.try_forward_decode_batch()
        executor._prepare()
        pool = executor._ggml_row_pool
        pool.submit()
        pool._submit_registered()
        pool._finish()
        block = self.model.blocks[0]
        block.attention_hc.read()
        block.ffn_hc.read()
        block.mixer.step()
        block.moe.router.forward()
        if self.phase == self.args.response_repeats and state.length - 3 == self.fail_position:
            raise RuntimeError('decode failure')
        return 'logits'

    def replay(self, model, tokenizer, row, prompt, xp, sample, guard, journal):
        self.test.assertIs(model, self.model)
        self.test.assertEqual(tokenizer, 'tokenizer')
        self.test.assertEqual(prompt, {'token_ids': [10, 11, 12]})
        self.test.assertIs(xp, self.xp)
        self.test.assertEqual((guard, journal), ('guard', 'journal'))
        selected = self.phase == self.args.response_repeats
        if selected:
            for owner, snapshot in self.snapshots:
                for name, original in snapshot.items():
                    if callable(original):
                        self.test.assertIsNot(getattr(owner, name), original)
            self.test.assertIsNot(sample, self.original_sample)
        else:
            self.assert_restored()
            self.test.assertIs(sample, self.original_sample)
        self.test.assertIs(self.generation.sample_token, self.original_sample)
        self.events.append(('replay', self.phase, selected))
        # Prefill and positions before 64 must not start the profiler.
        for position in [-3, *range(self.last_position + 1)]:
            logits = model._step(7, SimpleNamespace(length=position + 3), sentinel='kept')
            self.test.assertEqual(logits, 'logits')
            self.test.assertEqual(sample(logits, temperature=0), 7)
        result = object()
        self.returns.append(result)
        return result

    def child(self, args):
        self.test.assertEqual(self.events, ['isolation'])
        self.args = args
        for phase in ['warmup', *range(1, args.response_repeats + 1)]:
            self.phase = phase
            request_id = 'warmup' if phase == 'warmup' else 'performance-short-r1'
            result = self.runner.replay_case(self.model, 'tokenizer', {'request_id': request_id},
                {'token_ids': [10, 11, 12]}, self.xp, self.original_sample, 'guard', 'journal')
            self.test.assertIs(result, self.returns[-1])
            self.assert_restored()

    def run(self, output, *options):
        self.events.clear()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, self.modules))
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch.object(sys, 'argv', ['probe', '--output', str(output), *options]))
            load_probe().main()


class HotTraceTests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('mini_llm', 'cupy', 'numpy', 'probe_0254', 'run_0290')):
                self.fail(f'unexpected import: {name}')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            load_probe()

    def test_cli_defaults_and_profile_passthrough_types(self):
        probe = load_probe()
        args = probe.parse_args(['--output', 'new'])
        self.assertEqual((args.capture_response, args.profile_case_index), (1, 1))
        self.assertIsNone(args.expert_profile_from)
        self.assertIsNone(args.force_tokens_from)
        args = probe.parse_args(['--output', 'new', '--capture-response', '2',
            '--expert-profile-from', 'profile.json', '--force-tokens-from', 'tokens.json',
            '--profile-case-index', '0'])
        self.assertEqual(args.expert_profile_from, Path('profile.json'))
        self.assertEqual(args.force_tokens_from, Path('tokens.json'))
        self.assertEqual(args.profile_case_index, 0)

    def test_invalid_cli_fails_before_isolation_or_backend(self):
        for options in (['--capture-response', '0'], ['--capture-response', '3'],
                        ['--capture-response', 'x'], ['--profile-case-index', '-1']):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    load_probe().parse_args(['--output', 'new', *options])
                self.assertEqual(error.exception.code, 2)

    def test_isolation_precedes_output_backend_and_runner_work(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('mini_llm', 'cupy', 'probe_0254')):
                self.fail(f'backend/model work before isolation: {name}')
            return original(name, *args, **kwargs)
        def refuse():
            raise RuntimeError('not isolated')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            guard = fake_module('run_0290_isolated_benchmark', require_isolation=refuse)
            with patch.dict(sys.modules, {'run_0290_isolated_benchmark': guard}), \
                    patch.object(sys, 'argv', ['probe', '--output', str(output)]), \
                    patch.dict(os.environ), patch('builtins.__import__', guarded):
                before = dict(os.environ)
                with self.assertRaisesRegex(RuntimeError, 'not isolated'):
                    load_probe().main()
                self.assertEqual(dict(os.environ), before)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        harness = Harness(self)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'retained').write_text('keep')
            with self.assertRaises(FileExistsError):
                harness.run(output)
            self.assertIsNone(harness.args)
            self.assertEqual((output / 'retained').read_text(), 'keep')

    def check_capture(self, capture_response, *, profile=False, shared_readers=False):
        harness = Harness(self, shared_readers=shared_readers)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'trace'
            options = [] if capture_response == 1 else ['--capture-response', '2']
            source = Path(directory) / 'earlier-result.json'
            if profile:
                # The probe must forward paths, not load/install placement itself.
                options += ['--expert-profile-from', str(source), '--profile-case-index', '0',
                    '--force-tokens-from', str(source)]
            harness.run(output, *options)
            args = harness.args
            self.assertEqual(args.response_repeats, capture_response)
            self.assertEqual(args.cases, ['warmup', 'performance-short-r1'])
            self.assertEqual(args.promotion_limit, 0)
            self.assertEqual((args.cpu_workers, args.expert_cache_gib, args.host_warm_gib), (15, 6, 0))
            self.assertTrue(args.overlap_shared)
            self.assertTrue(args.gdn_graph_decode)
            self.assertEqual(args.expert_profile_from, source if profile else None)
            self.assertEqual(args.profile_case_index, 0 if profile else 1)
            self.assertEqual(args.force_tokens_from, source if profile else None)
            self.assertEqual([e for e in harness.events if isinstance(e, tuple) and e[0] == 'replay'],
                [('replay', 'warmup', False)] +
                [('replay', i, i == capture_response) for i in range(1, capture_response + 1)])
            self.assertEqual([e for e in harness.events if isinstance(e, tuple) and e[0] in ('start', 'stop', 'sync')],
                [('start', capture_response), ('stop', capture_response)])
            anchors = json.loads((output / 'clock_anchors.json').read_text())
            self.assertEqual([a['name'] for a in anchors], [f'token/{i}' for i in range(64, 80)])
            self.assertTrue(all(a['before_ns'] <= a['after_ns'] for a in anchors))
            boundaries = json.loads((output / 'capture_window.json').read_text())
            self.assertEqual([(b['event'], b['position']) for b in boundaries], [('start', 64), ('stop', 80)])
            self.assertTrue(all(isinstance(b['seconds'], float) for b in boundaries))
            self.assertEqual({phase for phase, _, _ in harness.ranges}, {capture_response})
            self.assertEqual(sum(name == 'sampler_and_output_wait' for _, name, _ in harness.ranges), 16)
            self.assertEqual(max(depth for _, _, depth in harness.ranges), 2)
            metadata = json.loads((output / 'capture_metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['captured_token_count'], 16)
            self.assertEqual(metadata['observed_response_count'], capture_response)
            self.assertEqual(metadata['capture_response'], capture_response)
            self.assertEqual(metadata['captured_response_index'], capture_response)
            self.assertEqual(metadata['captured_case_index'], capture_response)
            self.assertEqual(metadata['expert_profile_from'], str(source.resolve()) if profile else None)
            self.assertEqual(metadata['force_tokens_from'], str(source.resolve()) if profile else None)
            self.assertEqual(metadata['placement_profile_source'], 'placement_profile_source.json' if profile else None)
            self.assertFalse(metadata['rate_claim'])
        harness.assert_restored()
        self.assertIs(harness.runner.replay_case, harness.original_replay)

    def test_default_captures_first_response(self):
        self.check_capture(1)

    def test_second_response_only_with_static_profile_and_forced_trajectory(self):
        self.check_capture(2, profile=True)

    def test_profile_is_optional_for_second_response(self):
        self.check_capture(2)

    def test_shared_methods_are_not_double_wrapped(self):
        self.check_capture(2, shared_readers=True)

    def test_exception_stops_profiler_and_restores_retained_model_and_runner(self):
        harness = Harness(self, fail_position=65)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'trace'
            with self.assertRaisesRegex(RuntimeError, 'decode failure'):
                harness.run(output, '--capture-response', '2')
            metadata = json.loads((output / 'capture_metadata.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertEqual(metadata['captured_token_count'], 2)
            self.assertEqual(harness.events[-2:], [('sync', 2), ('stop', 2)])
            self.assertFalse(harness.active)
            self.assertFalse(harness.stack)
        harness.assert_restored()
        self.assertIs(harness.runner.replay_case, harness.original_replay)

    def test_short_capture_is_explicitly_incomplete(self):
        for last_position, count in ((63, 0), (70, 7)):
            with self.subTest(last_position=last_position), tempfile.TemporaryDirectory() as directory:
                harness = Harness(self, last_position=last_position)
                output = Path(directory) / 'trace'
                harness.run(output)
                metadata = json.loads((output / 'capture_metadata.json').read_text())
                self.assertFalse(metadata['complete'])
                self.assertEqual(metadata['captured_token_count'], count)
                harness.assert_restored()
                self.assertFalse(harness.active)

    def test_repeated_main_calls_do_not_accumulate_wrappers(self):
        harness = Harness(self)
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2):
                harness.run(Path(directory) / str(index), '--capture-response', '2')
                harness.assert_restored()
                self.assertIs(harness.runner.replay_case, harness.original_replay)


if __name__ == '__main__':
    unittest.main()
