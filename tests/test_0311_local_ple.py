"""CPU-only PLE aggregation/launcher tests; no model imports or execution."""
import builtins

import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts import probe_0311_local_ple as probe


class Meter:
    def __init__(self):
        self.seconds = 0.0
        self.major = 0
        self.fault_reads = 0
        self.snapshots = []

    def clock(self):
        return self.seconds

    def faults(self):
        self.fault_reads += 1
        return self.major

    def advance(self, seconds, faults=0):
        self.seconds += seconds
        self.major += faults

    def snapshot(self):
        self.snapshots.append(self.seconds)
        return dict(major_faults=self.major, minor_faults=2*self.major,
                    io=dict(read_bytes=100*self.major), VmRSS=123)


def fake_types(meter):
    class Store:
        def rows(self, name, first, last, *, outer=()):
            # A nested call must execute but must not double-count rows time.
            meter.advance(1)
            if not outer:
                return self.rows(name, first, last, outer=(1,))
            return (name, first, last)

    class Packed:
        def dequantize_reference(self):
            meter.advance(3, 2)
            return 'decoded'

    class PLE:
        def embedding(self, token, predecessors):
            assert predecessors == (-1, -1)
            raw = Store().rows('table', token, token+1)
            assert raw == ('table', token, token+1)
            assert Packed().dequantize_reference() == 'decoded'
            meter.advance(1, 1)
            return token + 100

    return PLE, Packed, Store


def test_phase_aggregation_nested_rows_scope_snapshots_and_append(tmp_path):
    meter = Meter()
    timer = probe.PLETimer(clock=meter.clock, faults=meter.faults)
    ple, packed, store = fake_types(meter)
    original_methods = ple.embedding, packed.dequantize_reference, store.rows
    output = tmp_path / 'ple_timing.json'
    sampled = []

    def sample(logits, *, temperature):
        sampled.append((logits, temperature))
        meter.advance(2)
        return 17

    result = object()

    def replay(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
        # Unrelated primitives inside a request are still excluded.
        store().rows('other', 0, 1)
        packed().dequantize_reference()
        assert ple().embedding(1, (-1, -1)) == 101
        assert sample_token('prefill', temperature=.5) == 17
        assert ple().embedding(2, (-1, -1)) == 102
        assert ple().embedding(3, (-1, -1)) == 103
        assert sample_token('decode', temperature=.5) == 17
        meter.advance(10, 4)  # Simulate untimed end cleanup.
        return result

    wrapped = timer.replay(replay, meter.snapshot,
        lambda record: probe.append_diagnostic(output, record),
        hooks=lambda: timer.instrument(ple, packed, store))
    for name in ('warmup', 'response'):
        assert wrapped(None, None, dict(request_id=name, _forced_output_token_ids=[8, 9]),
                       {}, None, sample, None, None) is result
        assert not timer.local.active
        assert timer.local.request is None
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['request_id'] for row in records] == ['warmup', 'response']
    assert len(meter.snapshots) == 4  # First sample and replay exit only.
    for row in records:
        assert row['phases']['prefill'] == dict(embedding_calls=1, embedding_host_seconds=6,
            embedding_major_faults=3, rows_calls=1, rows_host_seconds=2,
            decoder_calls=1, decoder_host_seconds=3)
        assert row['phases']['decode'] == dict(embedding_calls=2, embedding_host_seconds=12,
            embedding_major_faults=6, rows_calls=2, rows_host_seconds=4,
            decoder_calls=2, decoder_host_seconds=6)
        assert row['total']['embedding_calls'] == 3
        assert row['total']['embedding_host_seconds'] == 18
        assert row['total']['embedding_major_faults'] == 9
        bracket = row['decode_bracket']
        assert bracket['host_seconds'] == 24
        assert bracket['process_io_fault_delta'] == dict(major_faults=10, minor_faults=20,
                                                       io=dict(read_bytes=1000))
        assert bracket['end_label'] == 'after_replay_exit_including_untimed_cleanup'
        assert row['diagnostic_only'] and row['control_overhead_included']
        assert not row['rate_claim'] and row['forced_trajectory']
        assert row['sample_calls'] == 2 and row['error'] is None
    assert meter.fault_reads == 12  # Exactly twice per embedding, nowhere else.
    assert (ple.embedding, packed.dequantize_reference, store.rows) == original_methods
    assert sampled == [('prefill', .5), ('decode', .5)] * 2


def test_thread_local_active_flag_and_no_request_work_excluded():
    meter = Meter()
    timer = probe.PLETimer(clock=meter.clock, faults=meter.faults)
    counters = probe.empty_counters()
    timer.local.request = dict(phase='prefill', phases=dict(prefill=counters))
    side_calls = []

    def rows():
        side_calls.append(threading.get_ident())
        meter.advance(2)
        return 'rows'

    component = timer.component(rows, 'rows')

    @timer.embedding
    def embedding(nested):
        if nested:
            return embedding(False)
        assert component() == 'rows'
        worker = threading.Thread(target=component)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        return 'embedding'

    assert embedding(True) == 'embedding'
    assert counters['embedding_calls'] == 1
    assert counters['rows_calls'] == 1
    assert counters['embedding_host_seconds'] == 4
    assert counters['rows_host_seconds'] == 2
    assert len(side_calls) == 2 and side_calls[0] != side_calls[1]
    assert meter.fault_reads == 2
    before = counters.copy()
    timer.local.request = None
    assert embedding(True) == 'embedding'
    assert component() == 'rows'
    assert counters == before and meter.fault_reads == 2


def test_embedding_exception_restores_scope_and_original_methods():
    meter = Meter()
    timer = probe.PLETimer(clock=meter.clock, faults=meter.faults)
    ple, packed, store = fake_types(meter)

    def fail(self, token, predecessors):
        store().rows('table', 0, 1)
        meter.advance(1, 2)
        raise ValueError('fake embedding failure')

    ple.embedding = fail
    records = []

    def replay(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
        ple().embedding(1, (-1, -1))

    wrapped = timer.replay(replay, meter.snapshot, records.append,
                           hooks=lambda: timer.instrument(ple, packed, store))
    with pytest.raises(ValueError, match='fake embedding failure'):
        wrapped(None, None, dict(request_id='failed'), {}, None, lambda: 0, None, None)
    assert ple.embedding is fail
    assert timer.local.request is None and not timer.local.active
    assert records[0]['total']['embedding_calls'] == 1
    assert records[0]['total']['embedding_major_faults'] == 2
    assert records[0]['error'] == 'ValueError: fake embedding failure'
    assert records[0]['decode_bracket']['process_io_fault_delta'] is None
    assert not meter.snapshots


@pytest.mark.parametrize('samples', [1, 4])
def test_natural_replay_keyword_sampler_and_constant_snapshot_count(samples):
    meter = Meter()
    timer = probe.PLETimer(clock=meter.clock, faults=meter.faults)
    records = []

    def replay(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
        for _ in range(samples):
            assert sample_token() == 4
        return 'unchanged'

    wrapped = timer.replay(replay, meter.snapshot, records.append)
    assert wrapped(None, None, dict(request_id='natural'), {}, None,
                   sample_token=lambda: 4, guard=None, journal=None) == 'unchanged'
    assert len(meter.snapshots) == 2
    assert records[0]['sample_calls'] == samples
    assert not records[0]['forced_trajectory']


def test_import_is_model_and_cuda_free(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(('mini_llm', 'cupy', 'probe_0254'))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    runpy.run_path(probe.__file__, run_name='import_only')


@pytest.mark.parametrize('arguments', [[], ['--bootstrap', '--output', 'unused'],
                                      ['--child', '--output', 'unused']])
def test_isolation_failure_precedes_patch_runpy_and_bootstrap(monkeypatch, arguments):
    original_run = subprocess.run

    def refuse():
        assert subprocess.run is original_run
        raise RuntimeError('isolation required')

    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=refuse))
    monkeypatch.setattr(probe, 'launch', lambda args: pytest.fail('launch before isolation'))
    monkeypatch.setattr(probe, 'bootstrap', lambda args: pytest.fail('bootstrap before isolation'))
    with pytest.raises(RuntimeError, match='isolation required'):
        probe.main(arguments)
    assert subprocess.run is original_run


def test_real_0302_runpy_reuses_settings_and_forwards_flags(monkeypatch, tmp_path):
    calls, checks = [], []
    previous_argv = sys.argv

    def fake_run(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=lambda: checks.append('isolated')))
    output = tmp_path / 'not_created_by_fake_subprocess'
    with pytest.raises(SystemExit) as exited:
        probe.main(['--output', str(output), '--shared-grid', '--natural',
                    '--workers', '7', '--task-factor', '4', '--fused-shared'])
    assert exited.value.code == 0
    assert checks == ['isolated', 'isolated']
    assert subprocess.run is fake_run and sys.argv is previous_argv
    assert len(calls) == 1
    command, args, kwargs = calls[0]
    assert command[:3] == [sys.executable, str(Path(probe.__file__).resolve()), '--bootstrap']
    assert command[3:5] == ['--output', str(output)]
    assert '--force-tokens-from' not in command
    assert command[command.index('--cpu-workers')+1] == '7'
    assert kwargs['env']['MINI_LLM_Q8_SHARED_GRID'] == '1'
    assert kwargs['env']['MINI_LLM_GGML_POOL_TASK_FACTOR'] == '4'
    assert kwargs['env']['MINI_LLM_NATIVE_MOE_FUSED_SHARED'] == '1'
    assert kwargs['timeout'] == 185 and kwargs['cwd'] == probe.ROOT
    assert not args and not output.exists()


def test_only_exact_workload_run_is_rewritten_and_restored_on_error(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, 'run', fake_run)
    original_argv = sys.argv
    unrelated = [['nvidia-smi'], [sys.executable, '/elsewhere/probe_0254_strata_workload.py'],
                 [sys.executable, str(probe.CONFIGURATION)], 'echo probe_0254_strata_workload.py']

    def fake_runpy(path, *, run_name):
        assert path == str(probe.CONFIGURATION) and run_name == '__main__'
        assert sys.argv[1:] == ['--output', 'out', '--shared-grid', '--natural']
        for command in unrelated:
            subprocess.run(command, timeout=3)
        subprocess.run([sys.executable, str(probe.WORKLOAD), '--output', 'out'], timeout=185)
        raise RuntimeError('fake launcher failure')

    monkeypatch.setattr(probe.runpy, 'run_path', fake_runpy)
    with pytest.raises(RuntimeError, match='fake launcher failure'):
        probe.launch(['--output', 'out', '--shared-grid', '--natural'])
    assert [command for command, _ in calls[:-1]] == unrelated
    assert calls[-1] == ([sys.executable, str(Path(probe.__file__).resolve()), '--bootstrap',
                          '--output', 'out'], dict(timeout=185))
    assert sys.argv is original_argv and subprocess.run is fake_run


@pytest.mark.parametrize('arguments', [['--bootstrap', '--output', 'out'],
                                      ['--child', '--output=out']])
def test_bootstrap_preserves_supervisor_hop_and_restores_module(monkeypatch, arguments):
    events = []
    original_replay = lambda *args: None
    workload = SimpleNamespace(__file__=str(probe.WORKLOAD), replay_case=original_replay,
                               memory_snapshot=lambda: {})
    old_argv = sys.argv

    def workload_main():
        events.append('main')
        assert workload.__file__ == str(Path(probe.__file__).resolve())
        assert workload.replay_case is not original_replay
        assert sys.argv[1:] == (arguments[1:] if arguments[0] == '--bootstrap' else arguments)
        # This is precisely the script path the 0254 parent uses for --child.
        assert Path(workload.__file__).resolve() == Path(probe.__file__).resolve()
        raise SystemExit(7)

    workload.main = workload_main
    monkeypatch.setitem(sys.modules, 'probe_0254_strata_workload', workload)
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark',
                        SimpleNamespace(require_isolation=lambda: events.append('isolation')))
    with pytest.raises(SystemExit) as exited:
        probe.main(arguments)
    assert exited.value.code == 7
    assert events == ['isolation', 'main']
    assert workload.__file__ == str(probe.WORKLOAD) and workload.replay_case is original_replay
    assert sys.argv is old_argv


def test_thread_fault_reader_uses_rusage_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(probe.resource, 'getrusage',
        lambda who: (seen.append(who), SimpleNamespace(ru_majflt=19))[1])
    assert probe.thread_major_faults() == 19
    assert seen == [probe.resource.RUSAGE_THREAD]
