"""CPU-only Patch0310 tests: the only subprocess peer is Python, never Strata."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import probe_0310_strata_comparison as probe


PEER = r'''
import os, signal, sys, time
mode = sys.argv[1]
if mode == 'startup_timeout':
    time.sleep(60)
    sys.exit(0)
if mode == 'kill_timeout':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode in ('info', 'info_timeout', 'unknown_startup'):
    print('INFO logit_bias=1 context=1024 kv=int8 spec=2', flush=True)
    print('INFO expert_slots=2918 pool_workers=19', flush=True)
    if mode == 'info_timeout':
        time.sleep(60)
    if mode == 'unknown_startup':
        print('UNKNOWN fake startup', flush=True)
print('READY 1024 stop', flush=True)
follow = [list(map(int, line.strip().split(','))) for line in open(os.environ['STRATA_FOLLOW'])]
index = 0
for line in sys.stdin:
    if line == 'QUIT\n':
        if mode == 'slow_quit':
            time.sleep(1)
        sys.exit(7 if mode == 'nonzero_quit' else 0)
    expected = 'GEN %d temperature=1 top_p=.95 top_k=20 seed=1 ckpt=0 ' % len(follow[index])
    assert line.startswith(expected), repr(line)
    prompt_n = len(line.split()[-1].split(','))
    if mode in ('request_timeout', 'kill_timeout'):
        time.sleep(60)
    if mode == 'eof':
        sys.exit(0)
    if mode == 'error':
        print('ERR fake failure', flush=True)
        continue
    progress = 'RESUME 0\nPP %d %d 1 1000.0\nREUSED 0\n' % (prompt_n - 1, prompt_n)
    if mode == 'progress':
        progress = 'RESUME 0\nPP 0 2 0 0.0\nPP 1 2 1 1000.0\nPP 1 2 1 1000.0\nREUSED 0\n'
    if mode == 'resume_nonzero':
        progress = 'RESUME 1\n'
    if mode == 'reused_nonzero':
        progress = 'RESUME 0\nREUSED 1\n'
    if mode == 'unknown_progress':
        progress = 'RESUME 0\nPROGRESS 1\n'
    if mode == 'request_info':
        progress = 'RESUME 0\nINFO context=1024\n'
    print(progress, end='', flush=True)
    if mode == 'progress_timeout':
        time.sleep(60)
    # Larger than a pipe buffer: the harness must use a stderr file.
    sys.stderr.write('fake diagnostic\n' * 6000)
    sys.stderr.flush()
    ids = follow[index]
    sent = ids[:-1] if mode == 'missing' else ids
    if mode == 'wrong_token':
        sent = [ids[0] + 1, *ids[1:]]
    token_lines = ''.join('T %d\n' % t for t in sent).encode('ascii')
    if mode == 'late_progress':
        token_lines += b'PP 2 2 2 1000.0\n'
    # Exercise partial lines and several T/DONE lines in one os.read chunk.
    os.write(1, token_lines[:1])
    time.sleep(.01)
    count = len(ids) + (1 if mode == 'wrong_count' else 0)
    prompt_n += 1 if mode == 'wrong_prompt' else 0
    reuse = 1 if mode == 'reused' else 0
    finish = mode.removeprefix('finish_') if mode.startswith('finish_') else 'length'
    done = ('DONE %d %d 2.5 40.0 %s 0 0 %d 5 8 0 3 1.5 %d 0 chunk=128 experts_streamed=24\n'
            % (count, prompt_n, finish, reuse, prompt_n))
    os.write(1, token_lines[1:] + done.encode('ascii'))
    index += 1
'''


def requests():
    return [dict(request_id=name, prompt=[1, 2], follow=[10, 11, 12])
            for name in ('warmup', 'response-r1', 'response-r2')]


def run_peer(tmp_path, mode='ok', **kwargs):
    return probe.run_harness(
        [sys.executable, '-u', '-c', PEER, mode], requests(), tmp_path / 'capture',
        startup_timeout=2, request_timeout=2, timeout=12,
        snapshot=lambda pid: dict(pid=pid, VmRSS=123, io={'read_bytes': 5},
                                  minor_faults=1, major_faults=0), **kwargs)


def test_binary_protocol_repeats_snapshots_logs_and_rates(tmp_path):
    result = run_peer(tmp_path)
    output = tmp_path / 'capture'
    assert result['ok']
    assert result['cleanup'] == dict(actions=['QUIT'], returncode=0)
    assert len(result['requests']) == 3
    for row in result['requests']:
        assert row['validated'] and row['token_ids'] == [10, 11, 12]
        assert set(row['snapshots']) == {'before', 'first_T', 'last_T', 'DONE'}
        for snapshot in row['snapshots'].values():
            assert snapshot['process']['VmRSS'] == 123
            assert 'io' in snapshot['process'] and 'major_faults' in snapshot['process']
        assert row['engine']['counters']['file_mb'] == 1.5
        assert row['engine']['counters']['experts_streamed'] == 24
        assert row['rates']['engine_N_per_decode_second'] == 75
    sustained = result['sustained_response_rates']
    assert sustained['repeats'] == 2
    assert sustained['observed_decode_forwards'] == 4
    assert sustained['engine_N_per_decode_second'] == 75
    assert (output / 'follow.csv').read_text().splitlines() == ['10,11,12'] * 3
    assert (output / 'stdout.bin').read_bytes().count(b'DONE ') == 3
    assert (output / 'stderr.log').stat().st_size > 65536
    assert json.loads((output / 'result.json').read_text())['ok']
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['source_commit'] == probe.SOURCE_COMMIT
    assert provenance['cleanup_reserve_seconds'] == probe.CLEANUP_RESERVE == 10.0
    assert provenance['environment'] == {
        'STRATA_ROUTE_TAIL_SKIP': '0', 'STRATA_FOLLOW': str(output / 'follow.csv')}
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['ready', *([
        'before', 'progress', 'progress', 'progress', 'first_T', 'last_T', 'DONE'] * 3), 'cleanup']
    for row in result['requests']:
        assert [p['kind'] for p in row['progress']] == ['RESUME', 'PP', 'REUSED']
        assert row['progress'][1]['position'] == 1  # Last prompt token goes through the first verify window.
        logged = [e for e in events if e['event'] == 'progress' and e['request_id'] == row['request_id']]
        assert [e['raw'] for e in logged] == [p['raw'] for p in row['progress']]


def test_slow_quit_completes_without_termination(tmp_path):
    result = run_peer(tmp_path, 'slow_quit')
    assert result['ok'] and all(row['validated'] for row in result['requests'])
    assert result['cleanup'] == dict(actions=['QUIT'], returncode=0)
    assert result['total_wall_seconds'] >= 1


def test_nonzero_quit_still_fails_after_validated_requests(tmp_path):
    with pytest.raises(RuntimeError, match='Engine did not exit cleanly after QUIT'):
        run_peer(tmp_path, 'nonzero_quit')
    result = json.loads((tmp_path / 'capture/result.json').read_text())
    assert not result['ok'] and len(result['requests']) == 3
    assert all(row['validated'] for row in result['requests'])
    assert result['cleanup'] == dict(actions=['QUIT'], returncode=7)
    assert 'Engine did not exit cleanly after QUIT' in result['error']
    assert (tmp_path / 'capture/stdout.bin').read_bytes().count(b'DONE ') == 3
    assert (tmp_path / 'capture/stderr.log').stat().st_size > 0


def test_startup_info_recorded_before_ready(tmp_path):
    result = run_peer(tmp_path, 'info')
    assert result['ok']
    output = tmp_path / 'capture'
    expected = ['INFO logit_bias=1 context=1024 kv=int8 spec=2',
                'INFO expert_slots=2918 pool_workers=19']
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events[:4]] == ['info', 'info', 'ready', 'before']
    assert [event['line'] for event in events[:2]] == expected
    assert all(event['monotonic'] <= events[2]['monotonic'] for event in events[:2])
    assert (output / 'stdout.bin').read_text().splitlines()[:3] == [*expected, 'READY 1024 stop']


@pytest.mark.parametrize('mode,exception,message', [
    ('resume_nonzero', ValueError, 'RESUME must be zero'),
    ('reused_nonzero', ValueError, 'REUSED must be zero'),
    ('unknown_progress', ValueError, 'Unexpected protocol'),
    ('request_info', ValueError, 'Unexpected protocol'),
    ('late_progress', ValueError, 'PP must occur between RESUME and REUSED'),
    ('finish_cancel', ValueError, 'Unexpected finish'),
    ('finish_nonfinite', ValueError, 'Unexpected finish'),
    ('finish_eos', ValueError, 'Invalid DONE'),
    ('unknown_startup', ValueError, 'Expected READY'),
    ('missing', ValueError, 'Generated count mismatch'),
    ('wrong_count', ValueError, 'Generated count mismatch'),
    ('wrong_token', ValueError, 'Token mismatch'),
    ('wrong_prompt', ValueError, 'prompt count mismatch'),
    ('reused', ValueError, 'checkpoint reuse'),
    ('error', ValueError, 'Unexpected protocol'),
    ('eof', RuntimeError, 'EOF'),
])
def test_protocol_failure_preserved_and_process_retired(tmp_path, mode, exception, message):
    with pytest.raises(exception, match=message):
        run_peer(tmp_path, mode)
    result = json.loads((tmp_path / 'capture/result.json').read_text())
    assert not result['ok'] and message in result['error']
    assert result['cleanup']['returncode'] is not None


@pytest.mark.parametrize('mode', ['startup_timeout', 'info_timeout', 'request_timeout', 'progress_timeout', 'kill_timeout'])
def test_deadline_cleanup_quit_terminate_kill(tmp_path, mode):
    start = time.monotonic()
    with pytest.raises(TimeoutError, match='deadline'):
        probe.run_harness([sys.executable, '-u', '-c', PEER, mode], requests(),
                          tmp_path / 'capture', timeout=13, startup_timeout=.15,
                          request_timeout=.15, snapshot=lambda pid: dict(pid=pid))
    assert time.monotonic() - start < 7
    result = json.loads((tmp_path / 'capture/result.json').read_text())
    assert not result['ok']
    actions = result['cleanup']['actions']
    assert actions[:2] == ['QUIT', 'terminate']
    if mode == 'kill_timeout':
        assert actions[-1] == 'kill'
        assert result['cleanup']['returncode'] == -9
    for row in result['requests']:
        pid = row['snapshots']['before']['process']['pid']
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_exclusive_output_never_relaunches(tmp_path, monkeypatch):
    output = tmp_path / 'capture'
    output.mkdir()
    retained = output / 'result.json'
    retained.write_text('preserve me')
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
    with pytest.raises(FileExistsError):
        run_peer(tmp_path)
    assert retained.read_text() == 'preserve me'


def test_spawn_failure_still_records_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe.run_harness([str(tmp_path / 'nonexistent')], requests(), tmp_path / 'capture')
    assert not json.loads((tmp_path / 'capture/result.json').read_text())['ok']


def test_multiple_prefill_updates_preserve_request_and_token_counts(tmp_path):
    result = run_peer(tmp_path, 'progress')
    assert result['ok']
    for row in result['requests']:
        assert row['token_ids'] == [10, 11, 12]
        assert [p['position'] for p in row['progress'] if p['kind'] == 'PP'] == [0, 1, 1]
        assert row['engine']['counters']['reused'] == 0
        assert row['engine']['counters']['prompt_tokens_read'] == 2


def test_captured_startup_and_resume_prefix(tmp_path):
    capture = probe.ROOT / 'docs/qwen4exp/diagnostics_0310_strata_v1/stdout.bin'
    lines = capture.read_text().splitlines()
    assert lines[0].startswith('INFO ')
    assert lines[1:] == ['READY 1024 stop', 'RESUME 0']
    # Replay the retained prefix with a fake Python peer, then finish without a model.
    peer = PEER.replace("print('READY 1024 stop', flush=True)",
                        f"print({lines[0]!r}, flush=True)\nprint('READY 1024 stop', flush=True)")
    result = probe.run_harness([sys.executable, '-u', '-c', peer, 'ok'], requests(),
                              tmp_path / 'capture', snapshot=lambda pid: dict(pid=pid))
    assert result['ok']
    assert result['requests'][0]['progress'][0]['raw'] == lines[2]


@pytest.mark.parametrize('line', [
    'RESUME', 'RESUME 0 extra', 'RESUME 1', 'RESUME -1', 'REUSED 1', 'REUSED 0 extra',
    'PP 1 2 1', 'PP 1 3 1 1000', 'PP -1 2 1 1000', 'PP 3 2 1 1000',
    'PP 1 2 nan 1000', 'PP 1 2 inf 1000', 'PP 1 2 -1 1000',
    'PP 1 2 1 nan', 'PP 1 2 1 inf', 'PP 1 2 1 -1', 'INFO context=1024', 'UNKNOWN 0',
])
def test_invalid_request_progress(line):
    state = dict(resume=None if line.startswith('RESUME') else 0,
                 reused=None, position=0, elapsed_ms=0.0)
    with pytest.raises(ValueError):
        probe.request_progress(line, 2, state)


@pytest.mark.parametrize('lines', [
    ['REUSED 0'], ['PP 1 2 1 1000'], ['RESUME 0', 'RESUME 0'],
    ['RESUME 0', 'REUSED 0', 'REUSED 0'], ['RESUME 0', 'REUSED 0', 'PP 1 2 1 1000'],
    ['RESUME 0', 'PP 2 2 2 1000', 'PP 1 2 3 333.3'],
    ['RESUME 0', 'PP 1 2 2 500', 'PP 2 2 1 2000'],
])
def test_request_progress_order_and_monotonicity(lines):
    state = dict(resume=None, reused=None, position=0, elapsed_ms=0.0)
    with pytest.raises(ValueError):
        for line in lines:
            probe.request_progress(line, 2, state)


def test_done_pinned_source_positional_semantics():
    # Pinned printf order: drafts, reuse, hits/lookups, tier reads, decimal MB, fresh reads, offload.
    line = 'DONE 359 77 123.4 5678.9 stop 1 2 3 4 5 6 7 8.5 74 10 chunk=128 experts_streamed=11'
    result = probe.parse_done(line)
    assert result['prompt_ms'] == 123.4 and result['decode_ms'] == 5678.9
    assert list(result['counters'].items()) == [
        ('drafts_accepted', 1), ('drafts_offered', 2), ('reused', 3), ('hits', 4),
        ('lookups', 5), ('ram_blobs', 6), ('file_blobs', 7), ('file_mb', 8.5),
        ('prompt_tokens_read', 74), ('offloaded', 10), ('chunk', 128), ('experts_streamed', 11)]
    assert result['counters']['reused'] + result['counters']['prompt_tokens_read'] == result['prompt_tokens']


def test_done_named_and_future_counters():
    parsed = probe.parse_done('DONE 359 77 12.3 456.7 stop 0 0 0 12 34 0 4 1.2 77 0 chunk=128 experts_streamed=42 future=x')
    assert parsed['generated'] == 359 and parsed['finish'] == 'stop'
    assert parsed['counters'] == dict(drafts_accepted=0, drafts_offered=0, reused=0,
        hits=12, lookups=34, ram_blobs=0, file_blobs=4, file_mb=1.2,
        prompt_tokens_read=77, offloaded=0, chunk=128, experts_streamed=42, future='x')
    assert probe.parse_done('DONE 3 2 1 2 length')['counters'] == {}


@pytest.mark.parametrize('line', ['DONE', 'DONE -1 2 1 1 stop', 'DONE 1 0 1 1 stop',
    'DONE 1 2 nan 1 stop', 'DONE 1 2 1 inf stop', 'DONE 1 2 1 -1 stop',
    'DONE 1 2 1 1 stop -2', 'DONE 1 2 1 1 stop chunk=1 chunk=2'])
def test_invalid_done(line):
    with pytest.raises(ValueError):
        probe.parse_done(line)


def test_explicit_rate_denominators_and_coalesced_tokens():
    rates = probe.timing_rates(359, 2000, 10, 12, 14)
    assert rates['engine_N_per_decode_second'] == 179.5
    assert rates['observed_N_minus_1_per_firstT_to_lastT_second'] == 179
    assert rates['observed_N_minus_1_per_firstT_to_DONE_second'] == 89.5
    zero = probe.timing_rates(1, 0, 10, 10, 10)
    assert zero['engine_N_per_decode_second'] is None
    assert zero['observed_N_minus_1_per_firstT_to_lastT_second'] is None


def test_installed_asset_whitelist_and_fixed_bounds():
    config = dict(args=['--pack', '/pack', '--native', '/native', '--ple-gguf', '/ple',
                        '--expert-profile', '/profile', '--mtp', '/must-not-load',
                        '--expert-cache', 'auto', '--prefill', 'auto', '--max-context', '262144'])
    command = probe.engine_command(config)
    assert command[0] == str(probe.ENGINE)
    for key, expected in {'--pack': '/pack', '--native': '/native', '--ple-gguf': '/ple',
            '--expert-profile': '/profile', '--expert-cache': '2918', '--prefill': '128',
            '--spec': '2', '--pool-workers': '19', '--max-context': '1024', '--kv': 'int8',
            '--ram-keep-gib': '12', '--memory-limit-mib': '36864', '--suffix-draft': '0',
            '--lookup-max-t': '0', '--adapt-every': '0', '--adapt-swaps': '0',
            '--pcie-frac': '0', '--prompt-cache': '0', '--conversation-cache-mib': '0'}.items():
        assert command.count(key) == 1 and command[command.index(key)+1] == expected
    assert '--mmap-experts' in command and command[-1] == '--serve'
    assert '--mtp' not in command and '--spec-follow' not in command
    assert 'auto' not in command and 'generate' not in command
    with pytest.raises(ValueError, match='Expected one installed'):
        probe.engine_command(dict(args=[]))


def test_environment_presence_switches_are_unset_unless_profile():
    base = dict(PATH='/bin', STRATA_VERIFY_PROFILE='0', STRATA_DECODE_TIMING='0',
                STRATA_FOLLOW='/stale', STRATA_ROUTE_TAIL_SKIP='1', STRATA_RESIDENT_HEADROOM_GIB='0')
    plain = probe.engine_environment(base, Path('/new'))
    assert plain == dict(PATH='/bin', STRATA_FOLLOW='/new', STRATA_ROUTE_TAIL_SKIP='0')
    profiled = probe.engine_environment(base, Path('/new'), profile=True)
    assert profiled['STRATA_VERIFY_PROFILE'] == profiled['STRATA_DECODE_TIMING'] == '1'
    assert base['STRATA_FOLLOW'] == '/stale'


def test_retained_exact_workload_without_tokenization():
    rows = probe.load_requests()
    assert [len(row['follow']) for row in rows] == [128, 359, 359]
    assert [len(row['prompt']) for row in rows] == [75, 77, 77]
    assert rows[1]['prompt'] == rows[2]['prompt']
    assert rows[1]['follow'] == rows[2]['follow']
    assert rows[-1]['follow'][-1] == 248046
    assert probe.request_line(rows[1]).startswith(b'GEN 359 temperature=1 top_p=.95 top_k=20 seed=1 ckpt=0 ')


def test_retained_mismatch_refused(tmp_path):
    data = json.loads(probe.RETAINED.read_text())
    data['cases'][0]['prompt']['token_ids'][0] += 1
    path = tmp_path / 'retained.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='tokenizer mismatch'):
        probe.load_requests(path)


@pytest.mark.parametrize('ids', [[], [True], [-1], [2147483648], ['1']])
def test_invalid_token_ids(ids):
    with pytest.raises(ValueError):
        probe.token_ids(ids)


@pytest.mark.parametrize('value', ['231', 'nan', 'inf', '6', '10'])
def test_invalid_total_budget(value):
    with pytest.raises(SystemExit):
        probe.parse_args(['--output', '/unused', '--timeout', value])


def test_isolation_required_before_engine_inspection(monkeypatch, tmp_path):
    from scripts import run_0290_isolated_benchmark as isolation
    # main imports the helper by its historical scripts-directory module name.
    monkeypatch.setitem(sys.modules, 'run_0290_isolated_benchmark', isolation)
    def refuse():
        raise RuntimeError('not isolated')
    monkeypatch.setattr(isolation, 'require_isolation', refuse)
    monkeypatch.setattr(probe, 'source_info', lambda *a: pytest.fail('engine inspected before gate'))
    with pytest.raises(RuntimeError, match='not isolated'):
        probe.main(['--output', str(tmp_path / 'capture')])
    assert not (tmp_path / 'capture').exists()


def test_import_and_reused_memory_snapshot_never_import_cuda():
    code = '''
import sys
from scripts import probe_0310_strata_comparison as p
row = p.process_snapshot('self')
assert all(k in row for k in ('VmRSS', 'RssAnon', 'io', 'minor_faults', 'major_faults'))
assert not any(k.split('.')[0] in ('cupy', 'torch', 'mini_llm') for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], cwd=probe.ROOT, check=True, timeout=10)
