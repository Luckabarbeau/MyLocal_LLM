"""Workload contracts and CPU-only exact tokenizer preflight; never run a model."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('probe_0254', ROOT/'scripts/probe_0254_strata_workload.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def rows():
    return probe.select_records(probe.BENCHMARK/'outputs.jsonl', probe.CASES)


def test_exact_fixture_selection_and_settings(rows):
    assert [r['request_id'] for r in rows] == list(probe.CASES)
    assert [r['request']['seed'] for r in rows] == [1, 1, 2, 3]
    assert [r['request']['max_tokens'] for r in rows] == [128, 512, 512, 512]
    assert [r['usage']['prompt_tokens'] for r in rows] == [75, 77, 77, 77]
    for row in rows:
        assert row['messages'][0]['content'].startswith(f'[case {row["request_id"]}]\n')
        assert probe.sampling_settings(row) == dict(temperature=1., top_p=.95,
            top_k=20, repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)
    selected = probe.select_records(probe.BENCHMARK/'outputs.jsonl', probe.CASES[1:])
    assert selected == rows[1:]


def test_selection_rejects_missing_duplicate_unknown(tmp_path, rows):
    path = tmp_path/'outputs.jsonl'
    path.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError, match='Missing'):
        probe.select_records(path, probe.CASES)
    path.write_text((json.dumps(rows[0])+'\n')*2)
    with pytest.raises(ValueError, match='Duplicate'):
        probe.select_records(path, ['warmup'])
    for cases in ([], ['warmup', 'warmup'], ['conversation']):
        with pytest.raises(ValueError, match='unique'):
            probe.select_records(path, cases)


@pytest.mark.parametrize('key,value', [('min_p', .1), ('temperature', 0),
    ('max_tokens', 16), ('seed', 9), ('frequency_penalty', .5),
    ('reasoning_effort', 'low'), ('chat_template_kwargs', {'enable_thinking': False})])
def test_no_silent_settings_changes(rows, key, value):
    row = copy.deepcopy(rows[1])
    row['request'][key] = value
    with pytest.raises(ValueError):
        probe.sampling_settings(row)


def test_timing_formulas_and_single_token():
    timing = probe.timing_summary(512, 2., 2.5, 10., 12.5, 1.)
    assert timing['decode_forwards'] == 511
    assert timing['decode_forward_rate_N_minus_1_per_decode_second'] == 51.1
    assert timing['emitted_N_per_decode_second'] == 51.2
    assert timing['emitted_N_per_total_second'] == 40.96
    assert timing['first_token_from_prefill'] == 1
    single = probe.timing_summary(1, 1, 1, 0, 1, 0)
    assert single['decode_forwards'] == 0
    assert single['decode_forward_rate_N_minus_1_per_decode_second'] is None
    assert single['emitted_N_per_total_second'] == 1


def test_no_silent_prompt_count_or_id_mismatch(rows):
    template = '{% for m in messages %}{{ m.content }}{% endfor %}'
    class Tokens:
        def __init__(self, ids):
            self.ids = ids
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return SimpleNamespace(ids=self.ids)
    for a, b in (([1]*74, [1]*75), ([1]*75, [1]*74), ([1]*75, [2]*75)):
        with pytest.raises(ValueError, match='STOP prompt mismatch'):
            probe.checked_prompt(rows[0], Tokens(a), Tokens(b), template)
    row = copy.deepcopy(rows[0])
    row['usage']['prompt_tokens'] = 74
    with pytest.raises(ValueError, match='STOP prompt mismatch'):
        probe.checked_prompt(row, Tokens([1]*75), Tokens([1]*75), template)


def test_actual_installed_tokenizer_ids_and_captured_template(rows):
    # Read-only GGUF headers/mappings and CPU tokenizer; no Qwen model or GPU.
    tokenizer, prompts, provenance = probe.preflight(rows)
    assert provenance['exact_prompt_ids_verified'] is True
    assert len(provenance['checkpoint_sources']) == 2
    assert all(s['size_bytes'] > 0 and s['mtime_ns'] > 0 for s in provenance['checkpoint_sources'])
    for prompt, expected in zip(prompts, (75, 77, 77, 77)):
        assert len(prompt['token_ids']) == expected
        assert prompt['token_ids'] == prompt['strata_token_ids']
        assert tokenizer.decode(prompt['token_ids']) == prompt['text']
        assert 'Reasoning effort is set to xhigh.' in prompt['text']
        assert prompt['text'].endswith('<|im_start|>assistant\n<think>\n')


def test_supervisor_snapshot_does_not_open_ptrace_io(monkeypatch):
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if str(path).startswith('/proc/') and path.name == 'io':
            raise PermissionError('simulated exiting-child ptrace denial')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    memory = probe.memory_snapshot(include_io=False)
    assert memory['RssAnon'] > 0 and memory['MemAvailable'] > 0
    assert 'io' not in memory
    with pytest.raises(PermissionError, match='ptrace denial'):
        probe.memory_snapshot()


def test_zombie_status_is_exit_not_zero_memory(monkeypatch):
    original = Path.read_text
    def zombie(path, *args, **kwargs):
        if str(path) == '/proc/123/status':
            return 'Name:\tpython\nState:\tZ (zombie)\nThreads:\t1\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', zombie)
    with pytest.raises(ProcessLookupError, match='zombie status'):
        probe.memory_snapshot(123, include_io=False)


def test_parent_uses_memory_only_snapshot(monkeypatch, tmp_path):
    class Process:
        pid, returncode = 123, 0
        def __init__(self):
            self.polls = 0
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0
        def wait(self, timeout):
            return 0
    process = Process()
    calls = []
    def snapshot(pid, *, include_io):
        assert pid == process.pid and include_io is False
        calls.append(pid)
        return dict(RssAnon=probe.GiB, MemAvailable=10*probe.GiB)
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a, **kw: process)
    monkeypatch.setattr(probe, 'memory_snapshot', snapshot)
    monkeypatch.setattr(probe.time, 'sleep', lambda seconds: None)
    args = SimpleNamespace(output=tmp_path/'run', cases=['warmup'], host_warm_gib=0,
        mem_available_floor_gib=4, rss_anon_limit_gib=8, victim_policy='heap',
        cpu_backend='ggml_pool', cpu_workers=4, group_mixed_decode=False)
    probe.parent(args)
    assert calls == [process.pid]
    result = json.loads((args.output/'parent.json').read_text())
    assert result['returncode'] == 0 and result['stop_reason'] is None


def test_memory_guards_use_anon_not_mmap_rss():
    memory = dict(RssAnon=probe.GiB, RssFile=46*probe.GiB, VmRSS=47*probe.GiB,
                  MemAvailable=10*probe.GiB)
    probe.enforce_limits(memory, 8*probe.GiB, 8*probe.GiB, float('inf'))
    with pytest.raises(RuntimeError, match='RssAnon'):
        probe.enforce_limits(dict(memory, RssAnon=9*probe.GiB), 8*probe.GiB, 8*probe.GiB, float('inf'))
    with pytest.raises(RuntimeError, match='MemAvailable'):
        probe.enforce_limits(dict(memory, MemAvailable=7*probe.GiB), 8*probe.GiB, 8*probe.GiB, float('inf'))
    with pytest.raises(RuntimeError, match='time budget'):
        probe.enforce_limits(memory, 0, 0, 0)


def test_capacity_fast_options_and_caveats():
    assert probe.MODEL_OPTIONS['max_length'] == 1024
    assert probe.MODEL_OPTIONS['cache_bytes'] == 6*probe.GiB
    assert probe.MODEL_OPTIONS['hc_graph_decode'] is False  # Native read is a separate mode.
    for flag in ('hc_native_read', 'elementwise_fast_decode',
                 'expert_cache_slots', 'group_after_admission', 'qsa_fast_decode',
                 'hc_fast_decode', 'router_fast_decode', 'gdn_fast_decode'):
        assert probe.MODEL_OPTIONS[flag] is True
    assert probe.COMPARISON['full_configuration_equivalence'] is False


@pytest.mark.parametrize('forced',[False,True])
def test_replay_eos_fresh_state_windows_and_only_final_sync(rows,forced):
    calls, journal, seeds = [], [], []
    class Event:
        def record(self):
            pass
    xp = SimpleNamespace(random=SimpleNamespace(seed=seeds.append),
        cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 100.,
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: calls.append('sync'))))
    class FakeModel:
        config=SimpleNamespace(vocab_size=100)
        stop_token_ids = {99}
        expert_executor = SimpleNamespace(statistics=lambda: {'gpu_experts': 10})
        def create_generation_state(self, batch, capacity):
            assert (batch, capacity) == (1, 1024)
            calls.append('new_state')
            return SimpleNamespace()
        def prefill(self, ids, state):
            assert ids == [1]*77
            calls.append('prefill')
            return None
        def step(self, token, state):
            calls.append('step')
            return None
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: ','.join(map(str, ids)))
    model = FakeModel()
    for n in (1, 66):
        sequence = iter([2]*(n-1)+[99])
        def sample(logits, **settings):
            assert settings == probe.sampling_settings(rows[1])
            return 0 if forced else next(sequence)
        row=dict(rows[1])
        if forced: row['_forced_output_token_ids']=[2]*(n-1)+[99]
        result = probe.replay_case(model, tokenizer, row, {'token_ids': [1]*77},
            xp, sample, lambda: {}, journal.append)
        assert result['output_token_ids'] == [2]*(n-1)+[99]
        assert result['stop_reason'] == 'eos'
        assert result['forced_trajectory']==forced
        assert result['timing']['decode_forwards'] == n-1
        assert result['timing']['emitted_tokens_including_eos'] == n
        assert sum(len(w['token_ids']) for w in result['windows_64_tokens']) == n
        assert all(len(w['token_ids']) <= 64 for w in result['windows_64_tokens'])
    assert calls.count('new_state') == calls.count('prefill') == calls.count('sync') == 2
    assert calls.count('step') == 65
    assert seeds == [1, 1]
    assert [r['emitted_tokens'] for r in journal] == [1, 32, 64, 66]
