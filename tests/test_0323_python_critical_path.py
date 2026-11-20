"""CPU-only profiling brackets and wrapper contracts; never open a model/CUDA."""
import builtins
import json
import marshal
from pathlib import Path
import pstats
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import probe_0323_python_critical_path as probe


class FakeProfile:
    def __init__(self, events):
        self.events = events
        self.enabled = False
        self.stats = {}

    def enable(self):
        assert not self.enabled
        self.enabled = True
        self.events.append('enable')

    def disable(self):
        assert self.enabled
        self.enabled = False
        self.events.append('disable')

    def dump_stats(self, filename):
        assert not self.enabled
        self.events.append('dump')
        with open(filename, 'wb') as stream:
            marshal.dump(self.stats, stream)


def row(tokens, request_id='performance-short-r1'):
    return dict(request_id=request_id, request=dict(max_tokens=512),
                _forced_output_token_ids=tokens)


def invoke(wrapped, tokens, sample, *, model=None):
    return wrapped(model or SimpleNamespace(stop_token_ids={999}), 'tokenizer',
                   row(tokens), {'token_ids': [1]}, 'xp', sample, 'guard', 'journal')


@pytest.mark.parametrize('length', [1, 2, 359])
def test_exact_sample_range_and_unchanged_replay_arguments(length, tmp_path):
    events, profiles, records = [], [], []
    model = SimpleNamespace(stop_token_ids={999})
    tokens = list(range(length))

    def factory():
        profile = FakeProfile(events)
        profiles.append(profile)
        return profile

    timer = probe.DecodeProfiler(tmp_path, profile_factory=factory,
        clock=lambda: float(len(events)), emit=lambda output, profile, record: records.append(record))
    result = object()
    proposed = []

    def sample(logits, *, temperature):
        events.append(('sample_return', logits, temperature))
        proposed.append(logits)
        return 999  # Discarded proposal must NOT cause an early stop boundary.

    def replay(m, tokenizer, r, prompt, xp, sample_token, guard, journal):
        assert (m, tokenizer, prompt, xp, guard, journal) == (
            model, 'tokenizer', {'token_ids': [1]}, 'xp', 'guard', 'journal')
        assert r['_forced_output_token_ids'] is tokens
        events.append('prefill')
        for index in range(length):
            assert profiles[0].enabled == (index > 0)
            events.append(('step', index))
            assert sample_token(index, temperature=.5) == 999
            assert profiles[0].enabled == (index < length-1 and length > 1)
            events.append(('text', index))
        events.extend(['journal', 'sync', 'cleanup'])
        assert not profiles[0].enabled
        return result

    assert invoke(timer.replay(replay), tokens, sample, model=model) is result
    assert proposed == list(range(length))
    if length > 1:
        assert events[events.index('enable')-1] == ('sample_return', 0, .5)
        assert events[events.index('disable')-1] == ('sample_return', length-1, .5)
        assert events.count('enable') == events.count('disable') == 1
    else:
        assert 'enable' not in events and 'disable' not in events
    assert len(records) == 1 and records[0]['status'] == 'complete'
    assert records[0]['sample_calls'] == length
    assert records[0]['profile_started'] == (length > 1)
    assert records[0]['profiling_overhead_included'] and not records[0]['rate_claim']
    assert not records[0]['optimization_claim'] and not timer.active


def test_warmup_does_not_create_profiler_or_wrap_sampler(tmp_path):
    original_sample = lambda: 3

    def replay(model, tokenizer, r, prompt, xp, sample_token, guard, journal):
        assert r['request_id'] == 'warmup' and sample_token is original_sample
        return 'warm'

    timer = probe.DecodeProfiler(tmp_path,
        profile_factory=lambda: pytest.fail('warmup profiler'),
        emit=lambda *args: pytest.fail('warmup artifact'))
    assert timer.replay(replay)(None, None, row(None, 'warmup'), None, None,
                              original_sample, None, None) == 'warm'
    assert timer.repeat_index == 0


@pytest.mark.parametrize('failure', ['prefill', 'step', 'sampler', 'cleanup'])
def test_exception_cleanup_writes_incomplete_artifacts_and_preserves_exception(failure, tmp_path):
    events = []
    profile = FakeProfile(events)
    timer = probe.DecodeProfiler(tmp_path, profile_factory=lambda: profile)
    sample_calls = 0

    def fail():
        raise RuntimeError('stop failure: ' + failure)

    def sample():
        nonlocal sample_calls
        sample_calls += 1
        if failure == 'sampler' and sample_calls == 2:
            fail()
        return 17

    def replay(model, tokenizer, r, prompt, xp, sample_token, guard, journal):
        if failure == 'prefill':
            fail()
        sample_token()
        if failure == 'step':
            fail()
        sample_token()
        sample_token()
        assert not profile.enabled
        fail()

    with pytest.raises(RuntimeError, match='stop failure: ' + failure):
        invoke(timer.replay(replay), [1, 2, 3], sample)
    assert not timer.active and not profile.enabled
    assert events[-1] == 'dump'
    assert events.count('disable') == (0 if failure == 'prefill' else 1)
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert metadata['status'] == 'incomplete'
    assert metadata['error'] == 'RuntimeError: stop failure: ' + failure
    assert metadata['sample_calls'] == dict(prefill=0, step=1, sampler=1, cleanup=3)[failure]
    assert len(list(tmp_path.iterdir())) == 3


def test_forced_eos_boundary_is_disabled_before_cleanup_and_tagged_incomplete(tmp_path):
    events, records = [], []
    profile = FakeProfile(events)
    timer = probe.DecodeProfiler(tmp_path, profile_factory=lambda: profile,
        emit=lambda output, p, r: records.append(r))

    def replay(model, tokenizer, r, prompt, xp, sample_token, guard, journal):
        sample_token()
        assert profile.enabled
        sample_token()
        assert not profile.enabled
        return 'stopped'

    assert invoke(timer.replay(replay), [1, 999, 3], lambda: 4) == 'stopped'
    assert records[0]['status'] == 'incomplete'
    assert records[0]['last_sample_boundary'] == records[0]['sample_calls'] == 2


@pytest.mark.parametrize('tokens', [None, [], [1]*360, [-1], [True], [1.5]])
def test_invalid_trajectory_rejected_before_replay_or_profile(tmp_path, tokens):
    timer = probe.DecodeProfiler(tmp_path, profile_factory=lambda: pytest.fail('profile'))
    with pytest.raises(ValueError, match='1..359'):
        invoke(timer.replay(lambda *args: pytest.fail('replay')), tokens, lambda: 1)
    assert not list(tmp_path.iterdir())


def test_full_records_graph_costs_top_text_and_exclusive_files(tmp_path, capsys):
    events = []
    profile = FakeProfile(events)
    profile.stats = {
        ('/repo/mini_llm/external/qwen4exp_token_graph.py', 170, '_validate_live'): (2, 3, .1, .6, {}),
        ('/repo/mini_llm/external/qwen4exp_token_graph.py', 71, '_configuration'): (4, 4, .2, .4, {}),
        ('~', 0, '<cupy.cuda.runtime.memcpy>'): (1, 1, .8, .8, {}),
        ('/other.py', 1, '_validate_live'): (9, 9, .3, .3, {}),
    }
    metadata = dict(instance_index=1, repeat_index=1, status='complete', error=None)
    paths = probe.write_profile(tmp_path, profile, metadata)
    data = json.loads(paths['json'].read_text())
    assert len(data['functions']) == 4
    graph = data['costs']['token_graph_methods']
    assert set(graph) == set(probe.TOKEN_GRAPH_METHODS)
    assert graph['_validate_live'][0]['primitive_calls'] == 2
    assert graph['_validate_live'][0]['total_calls'] == 3
    assert graph['_validate_live'][0]['self_seconds'] == .1
    assert graph['_validate_live'][0]['cumulative_seconds'] == .6
    assert len(graph['_validate_live']) == 1 and graph['_validate_states'] == []
    assert len(data['costs']['cuda_copy_or_wait_candidates']) == 1
    assert 'NOT CPU utilization' in data['timing']
    assert 'overlap' in data['timing'] and 'heuristic' in data['costs']['interpretation']
    assert pstats.Stats(str(paths['pstats'])).stats == profile.stats
    assert 'NOT a rate claim' in paths['top'].read_text()
    before = {path: path.read_bytes() for path in paths.values()}
    with pytest.raises(FileExistsError):
        probe.write_profile(tmp_path, profile, metadata)
    assert {path: path.read_bytes() for path in paths.values()} == before
    assert not capsys.readouterr().out


def test_repeats_and_instances_have_distinct_filenames_and_require_existing_output(tmp_path):
    def replay(model, tokenizer, r, prompt, xp, sample_token, guard, journal):
        sample_token()
        sample_token()
        return 7

    first = probe.DecodeProfiler(tmp_path)
    second = probe.DecodeProfiler(tmp_path)
    for timer in (first, first, second):
        assert invoke(timer.replay(replay), [1, 2], lambda: 17) == 7
    records = [json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
    assert len(records) == 3 and len(list(tmp_path.iterdir())) == 9
    assert sorted(r['repeat_index'] for r in records) == [1, 1, 2]
    assert len({r['instance_index'] for r in records}) == 2
    absent = tmp_path / 'not_parent_created'
    with pytest.raises(FileNotFoundError):
        invoke(probe.DecodeProfiler(absent).replay(replay), [1, 2], lambda: 17)
    assert not absent.exists()


def test_real_cprofile_excludes_prefill_first_sampler_and_last_cleanup(tmp_path):
    def prefill_only():
        return 1

    def decode_only():
        return 2

    def cleanup_only():
        return 3

    def original_sample():
        return 17

    def replay(model, tokenizer, r, prompt, xp, sample_token, guard, journal):
        prefill_only()
        sample_token()
        decode_only()
        sample_token()
        cleanup_only()

    invoke(probe.DecodeProfiler(tmp_path).replay(replay), [1, 2], original_sample)
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    functions = {r['function']: r for r in metadata['functions']}
    assert functions['decode_only']['total_calls'] == 1
    assert functions['original_sample']['total_calls'] == 1  # Not sample 1.
    assert 'prefill_only' not in functions and 'cleanup_only' not in functions
    assert metadata['status'] == 'complete'


def test_import_does_not_import_backend_model_workload_or_cuda(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(('mini_llm', 'cupy', 'probe_0254'))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    runpy.run_path(probe.__file__, run_name='import_only')


@pytest.mark.parametrize('arguments', [[], ['--bootstrap'], ['--child']])
def test_isolation_is_required_before_any_launch_or_patch(monkeypatch, arguments):
    original_run = subprocess.run

    def refuse():
        assert subprocess.run is original_run
        raise RuntimeError('strict isolation required')

    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=refuse))
    monkeypatch.setattr(probe, 'launch', lambda args: pytest.fail('launch'))
    monkeypatch.setattr(probe, 'bootstrap', lambda args: pytest.fail('bootstrap'))
    with pytest.raises(RuntimeError, match='strict isolation required'):
        probe.main(arguments)


@pytest.mark.parametrize('flags', [['--natural'], ['--trace'], ['--telemetry'], ['--parity'],
                                  ['--response-repeats', '2'], ['--response-repeats=3']])
def test_rejects_multiple_responses_or_other_instrumentation_before_launch(flags):
    with pytest.raises(ValueError):
        probe.launch(['--output', 'unused', *flags])


def test_real_0302_forwards_configuration_without_loading_multiple_models(monkeypatch, tmp_path):
    calls, checks = [], []
    previous_argv = sys.argv

    def fake_run(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=lambda: checks.append('isolated')))
    output = tmp_path / 'created_only_by_0254_parent'
    with pytest.raises(SystemExit) as exited:
        probe.main(['--output', str(output), '--token-graph', '--kernel-publish',
                    '--qsa-warp-dots', '--shared-grid', '--expert-cache-gib', '7',
                    '--response-repeats', '1'])
    assert exited.value.code == 0 and len(calls) == 1
    command, args, kwargs = calls[0]
    assert command[:3] == [sys.executable, str(Path(probe.__file__).resolve()), '--bootstrap']
    assert '--force-tokens-from' in command
    assert command[command.index('--expert-cache-gib')+1] == '7.0'
    assert command[command.index('--response-repeats')+1] == '1'
    assert '--cases' not in command  # 0254 default is warmup + r1, one model.
    for key in ('TOKEN_GRAPH', 'NATIVE_MOE_KERNEL_PUBLISH', 'QSA_WARP_DOTS', 'Q8_SHARED_GRID'):
        assert kwargs['env']['MINI_LLM_' + key] == '1'
    assert kwargs['env']['MINI_LLM_NATIVE_MOE_TIMELINE'] == '0'
    assert kwargs['env']['MINI_LLM_NATIVE_MOE_ROUTE_PROFILE'] == '0'
    assert kwargs['timeout'] == 185 and kwargs['cwd'] == probe.ROOT
    assert checks == ['isolated', 'isolated']
    assert sys.argv is previous_argv and subprocess.run is fake_run
    assert not args and not output.exists()


@pytest.mark.parametrize('child', [False, True])
def test_bootstrap_preserves_0254_parent_and_child_hop_and_restores_on_failure(monkeypatch, child):
    original_replay = lambda *args: None
    workload = SimpleNamespace(__file__=str(probe.WORKLOAD), replay_case=original_replay)
    previous_argv = sys.argv
    arguments = ['--output=out', '--force-tokens-from', 'trajectory.json', '--response-repeats', '1']
    if child:
        arguments.insert(0, '--child')

    def workload_main():
        assert workload.__file__ == str(Path(probe.__file__).resolve())
        assert workload.replay_case is not original_replay
        assert sys.argv[1:] == arguments
        assert not Path('out').exists()
        raise SystemExit(7)

    workload.main = workload_main
    monkeypatch.setitem(sys.modules, 'probe_0254_strata_workload', workload)
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=lambda: None))
    with pytest.raises(SystemExit) as exited:
        probe.main(arguments if child else ['--bootstrap', *arguments])
    assert exited.value.code == 7
    assert workload.__file__ == str(probe.WORKLOAD) and workload.replay_case is original_replay
    assert sys.argv is previous_argv


def test_intercept_is_exact_and_restores_subprocess_and_argv_on_exception(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, 'run', fake_run)
    original_argv = sys.argv
    unrelated = [['nvidia-smi'], [sys.executable, '/other/probe_0254_strata_workload.py'],
                 [sys.executable, str(probe.CONFIGURATION)], 'echo unrelated']

    def fake_runpy(path, *, run_name):
        assert path == str(probe.CONFIGURATION) and run_name == '__main__'
        for command in unrelated:
            subprocess.run(command, timeout=3)
        subprocess.run([sys.executable, str(probe.WORKLOAD), '--output', 'unused'], timeout=185)
        raise RuntimeError('launcher failed')

    monkeypatch.setattr(probe.runpy, 'run_path', fake_runpy)
    with pytest.raises(RuntimeError, match='launcher failed'):
        probe.launch(['--output', 'unused'])
    assert [command for command, _ in calls[:-1]] == unrelated
    assert calls[-1] == ([sys.executable, str(Path(probe.__file__).resolve()), '--bootstrap',
                         '--output', 'unused'], dict(timeout=185))
    assert subprocess.run is fake_run and sys.argv is original_argv


@pytest.mark.parametrize('arguments', [
    ['--output', 'out'],
    ['--output', 'out', '--force-tokens-from', 'fixed', '--cases', 'warmup', 'performance-short-r2'],
    ['--output', 'out', '--force-tokens-from', 'fixed', '--cases=performance-short-r1'],
])
def test_bootstrap_rejects_uncontrolled_or_multi_case_workload(arguments):
    with pytest.raises(ValueError):
        probe.validate_arguments(arguments, bootstrap=True)
