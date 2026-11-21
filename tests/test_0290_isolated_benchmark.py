"""CPU-only fail-closed guard tests; no model, CUDA or systemd service required."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('isolated_benchmark',
    Path(__file__).resolve().parents[1] / 'scripts/run_0290_isolated_benchmark.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def fixture_limits(tmp_path):
    unit = 'mylocal-bench-test.service'
    relative = Path('/user.slice/user-1000.slice/user@1000.service/app.slice') / unit
    directory = tmp_path.joinpath(*relative.parts[1:])
    directory.mkdir(parents=True)
    for key, value in [('memory.high', 128), ('memory.max', 256),
                       ('memory.swap.max', 0), ('memory.oom.group', 1)]:
        (directory / key).write_text(str(value))
    return unit, relative, directory


def test_membership_requires_unified_absolute_non_escaping_path():
    assert guard.membership('0::/user.slice/test.service\n') == Path('/user.slice/test.service')
    for text in ('1:memory:/old', '0::relative', '0::/../escape', '0::/a\n0::/b'):
        with pytest.raises(RuntimeError):
            guard.membership(text)


def test_verified_before_exec_limits(tmp_path):
    unit, relative, directory = fixture_limits(tmp_path)
    guard.verify_limits(directory, relative, unit, 128, 256)
    (directory / 'memory.max').write_text('max')
    with pytest.raises(RuntimeError, match='memory.max'):
        guard.verify_limits(directory, relative, unit, 128, 256)


@pytest.mark.parametrize('name,value', [('memory.high', 'max'), ('memory.swap.max', 'max'),
                                        ('memory.oom.group', '0')])
def test_reject_missing_safety_properties(tmp_path, name, value):
    unit, relative, directory = fixture_limits(tmp_path)
    (directory / name).write_text(value)
    with pytest.raises(RuntimeError, match=name):
        guard.verify_limits(directory, relative, unit, 128, 256)


def test_reject_desktop_membership(tmp_path):
    unit, relative, directory = fixture_limits(tmp_path)
    desktop = Path('/user.slice/user-1000.slice/user@1000.service/session.slice/org.gnome.Shell@x11.service')
    with pytest.raises(RuntimeError, match='outside the desktop'):
        guard.verify_limits(directory, desktop, unit, 128, 256)


def test_pressure_and_reserve_abort():
    row = dict(host_available=20*guard.GIB, ancestor_headroom=None,
               psi_some_avg10=0, psi_full_avg10=0, memory_events={})
    assert guard.stop_reason(row, 12*guard.GIB) is None
    for key, value, expected in [('host_available', 11*guard.GIB, 'host RAM'),
                                  ('ancestor_headroom', 11*guard.GIB, 'ancestor'),
                                  ('psi_full_avg10', 10, 'pressure'),
                                  ('psi_some_avg10', 30, 'pressure'),
                                  ('memory_events', {'oom_kill': 1}, 'OOM')]:
        assert expected in guard.stop_reason(dict(row, **{key: value}), 12*guard.GIB)


def test_invalid_psi_not_silently_zero(tmp_path):
    path = tmp_path / 'psi'
    path.write_text('some avg10=0.12 total=10\nfull avg10=0.03 total=1\n')
    assert guard.pressure(path) == {'some': .12, 'full': .03}
    path.write_text('some avg10=0.12\n')
    with pytest.raises(KeyError):
        guard.pressure(path)


def test_launcher_is_service_not_scope_and_propagates_thread_choices(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('PRIVATE_API_KEY', 'do-not-export')
    args = guard.parse_args(['--output', str(tmp_path / 'new'), '--', '/usr/bin/true'])
    command = guard.command_line(args, 'mylocal-bench-test.service')
    assert '--slice=app.slice' in command
    assert '--scope' not in command
    assert '--property=OOMPolicy=kill' in command
    assert '--property=KillMode=control-group' in command
    assert '--property=MemorySwapMax=0' in command
    assert '--setenv=OMP_NUM_THREADS=1' in command
    assert not any('PRIVATE_API_KEY' in arg for arg in command)
    assert '--gate-unit' in command


@pytest.mark.parametrize('options', [
    ['--reserve-gib', '0'], ['--reserve-gib', 'nan'], ['--reserve-gib', 'inf'],
    ['--memory-high-mib', '1024'], ['--timeout', '0'], ['--timeout', '1801']])
def test_unsafe_cli_rejected(tmp_path, options):
    with pytest.raises(SystemExit):
        guard.parse_args(['--output', str(tmp_path / 'new'), *options, '--', '/usr/bin/true'])


def test_no_payload_if_gate_not_isolated(tmp_path, monkeypatch):
    args = guard.parse_args(['--output', str(tmp_path / 'new'), '--gate-unit',
                             'mylocal-bench-test.service', '--', '/usr/bin/true'])
    monkeypatch.setattr(guard, 'membership', lambda text: Path('/session.slice/desktop.service'))
    monkeypatch.setattr(guard.subprocess, 'Popen', lambda *a, **k: pytest.fail('Payload must not execute'))
    with pytest.raises(RuntimeError, match='outside the desktop'):
        guard.gate(args)
    assert not args.output.exists()


def test_probe_precondition_rejects_desktop(monkeypatch):
    monkeypatch.setattr(guard, 'membership', lambda text: Path('/session.slice/desktop.service'))
    with pytest.raises(RuntimeError, match='direct model probes are disabled'):
        guard.require_isolation()


def test_probe_precondition_accepts_only_verified_limits(tmp_path, monkeypatch):
    unit, relative, directory = fixture_limits(tmp_path)
    monkeypatch.setattr(guard, 'CGROUP_ROOT', tmp_path)
    monkeypatch.setattr(guard, 'membership', lambda text: relative)
    guard.require_isolation()
    (directory / 'memory.swap.max').write_text('1')
    with pytest.raises(RuntimeError, match='memory.swap.max'):
        guard.require_isolation()


def test_gate_records_child_failure_without_inference(tmp_path, monkeypatch):
    unit, relative, directory = fixture_limits(tmp_path)
    monkeypatch.setattr(guard, 'CGROUP_ROOT', tmp_path)
    monkeypatch.setattr(guard, 'membership', lambda text: relative)
    args = guard.parse_args(['--output', str(tmp_path / 'new'), '--gate-unit', unit,
                             '--', '/usr/bin/false'])
    monkeypatch.setattr(guard, 'verify_limits', lambda *a: None)
    row = dict(memory_current=0, memory_peak=0, memory_events={},
               host_available=100*guard.GIB, ancestor_headroom=None,
               psi_some_avg10=0, psi_full_avg10=0)
    monkeypatch.setattr(guard, 'snapshot', lambda *a: row)
    assert guard.gate(args) == 1
    assert '"returncode": 1' in (args.output / 'guard.jsonl').read_text()


def test_snapshot_accounts_ancestor_limits_and_pressure(tmp_path, monkeypatch):
    root = tmp_path / 'cg'
    directory = root / 'parent' / 'child'
    directory.mkdir(parents=True)
    for node in (root, root / 'parent', directory):
        (node / 'memory.pressure').write_text('some avg10=0\nfull avg10=0\n')
    # Root has no memory.max/high: Linux does not expose these on the root.
    (root / 'parent' / 'memory.high').write_text('500')
    (root / 'parent' / 'memory.max').write_text('1000')
    (root / 'parent' / 'memory.current').write_text('200')
    for name, value in [('memory.current', '100'), ('memory.peak', '150'),
                        ('memory.events', 'oom 0\noom_kill 0\n')]:
        (directory / name).write_text(value)
    monkeypatch.setattr(guard, 'memory_available', lambda: 2000)
    row = guard.snapshot(directory, root=root)
    assert row['ancestor_headroom'] == 300
    assert row['memory_current'] == 100 and row['memory_peak'] == 150


def test_workload_cli_checks_isolation_before_parent(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('workload_guard_test',
        Path(__file__).resolve().parents[1] / 'scripts/probe_0254_strata_workload.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    import run_0290_isolated_benchmark as imported_guard
    def refuse():
        raise RuntimeError('test unisolated')
    monkeypatch.setattr(imported_guard, 'require_isolation', refuse)
    monkeypatch.setattr(probe, 'parent', lambda *a: pytest.fail('No parent before guard'))
    monkeypatch.setattr(probe.sys, 'argv', ['probe', '--output', str(tmp_path / 'new')])
    with pytest.raises(RuntimeError, match='test unisolated'):
        probe.main()


def test_preserve_existing_results(tmp_path):
    with pytest.raises(SystemExit):
        guard.parse_args(['--output', str(tmp_path), '--', '/usr/bin/true'])


@pytest.fixture
def snapshot_cgroup(tmp_path, monkeypatch):
    root = tmp_path / 'cg'
    directory = root / 'parent' / 'child'
    directory.mkdir(parents=True)
    for node in (root, directory.parent, directory):
        (node / 'memory.pressure').write_text('some avg10=0\nfull avg10=0\n')
    for name, value in [('memory.high', '500'), ('memory.max', '1000'),
                        ('memory.current', '200')]:
        (directory.parent / name).write_text(value)
    for name, value in [('memory.current', '100'), ('memory.peak', '150'),
                        ('memory.events', 'oom 0\noom_kill 0\n'),
                        ('memory.stat', 'anon 4096\n'), ('memory.swap.current', '0\n')]:
        (directory / name).write_text(value)
    read_pressure = guard.pressure
    monkeypatch.setattr(guard, 'pressure', lambda path:
        {'some': 0, 'full': 0} if path == Path('/proc/pressure/memory') else read_pressure(path))
    monkeypatch.setattr(guard, 'memory_available', lambda: 2000)
    return root, directory


def test_snapshot_records_raw_memory_stat_bytes_and_counters(snapshot_cgroup):
    root, directory = snapshot_cgroup
    values = dict(anon=4096, file=8192, kernel=1024, shmem=512, file_mapped=2048,
                  file_dirty=0, file_writeback=0, inactive_file=4096, active_file=2048,
                  pgfault=123, pgmajfault=4, workingset_refault_file=7,
                  future_kernel_counter_v42=19)
    (directory / 'memory.stat').write_text(''.join(f'{key} {value}\n' for key, value in values.items()))
    row = guard.snapshot(directory, root=root)
    assert row['memory_stat'] == values
    assert 'memory_stat_error' not in row
    assert row['memory_swap_current'] == 0
    assert 'memory_swap_current_error' not in row
    assert row['memory_current'] == 100 and row['memory_peak'] == 150
    assert row['memory_events'] == {'oom': 0, 'oom_kill': 0}
    assert row['ancestor_headroom'] == 300 and row['host_available'] == 2000
    assert row['psi_some_avg10'] == row['psi_full_avg10'] == 0


@pytest.mark.parametrize('filename,field', [('memory.stat', 'memory_stat'),
                                           ('memory.swap.current', 'memory_swap_current')])
def test_snapshot_missing_optional_telemetry(snapshot_cgroup, filename, field):
    root, directory = snapshot_cgroup
    (directory / filename).unlink()
    row = guard.snapshot(directory, root=root)
    assert row[field] is None
    assert 'FileNotFoundError' in row[f'{field}_error']
    assert guard.stop_reason(row, 100) is None


@pytest.mark.parametrize('text', ['', '\n', 'anon', 'anon 1 extra\n', 'anon -1\n',
                                  'anon +1\n', 'anon 1.5\n', 'anon 1_000\n',
                                  'anon nope\n', 'anon 1\nanon 2\n',
                                  'anon 1\nfile -2\n', 'bad-key 1\n', 'anon ١\n'])
def test_snapshot_rejects_malformed_memory_stat(snapshot_cgroup, text):
    root, directory = snapshot_cgroup
    (directory / 'memory.stat').write_text(text)
    row = guard.snapshot(directory, root=root)
    assert row['memory_stat'] is None
    assert row['memory_stat_error']
    assert row['memory_swap_current'] == 0
    assert guard.stop_reason(row, 100) is None


@pytest.mark.parametrize('size', [64 * 1024, 64 * 1024 + 1])
@pytest.mark.parametrize('filename,field', [('memory.stat', 'memory_stat'),
                                           ('memory.swap.current', 'memory_swap_current')])
def test_snapshot_diagnostic_reads_are_bounded(snapshot_cgroup, monkeypatch, size, filename, field):
    import io

    root, directory = snapshot_cgroup
    reads = []

    class BoundedStream(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 64 * 1024
            data = super().read(size)
            reads.append(len(data))
            return data

    original_open = Path.open
    def open_fixture(path, *args, **kwargs):
        if path == directory / filename:
            return BoundedStream(b'anon 1\n'.ljust(size, b' '))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_fixture)
    row = guard.snapshot(directory, root=root)
    assert reads == [64 * 1024]
    assert row[field] is None
    assert '64 KiB' in row[f'{field}_error']


@pytest.mark.parametrize('filename,field', [('memory.stat', 'memory_stat'),
                                           ('memory.swap.current', 'memory_swap_current')])
def test_snapshot_unreadable_optional_telemetry(snapshot_cgroup, monkeypatch, filename, field):
    root, directory = snapshot_cgroup
    original_open = Path.open
    def open_fixture(path, *args, **kwargs):
        if path == directory / filename:
            raise PermissionError('fixture telemetry denied')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_fixture)
    row = guard.snapshot(directory, root=root)
    assert row[field] is None
    assert 'PermissionError' in row[f'{field}_error']
    assert guard.stop_reason(row, 100) is None


@pytest.mark.parametrize('value', [0, 4096])
def test_snapshot_swap_current_bytes(snapshot_cgroup, value):
    root, directory = snapshot_cgroup
    (directory / 'memory.swap.current').write_text(f'{value}\n')
    row = guard.snapshot(directory, root=root)
    assert row['memory_swap_current'] == value
    assert 'memory_swap_current_error' not in row


@pytest.mark.parametrize('text', ['', '-1', '+1', '1.5', '1_000', 'max', '1\n2', '١'])
def test_snapshot_rejects_malformed_swap_current(snapshot_cgroup, text):
    root, directory = snapshot_cgroup
    (directory / 'memory.swap.current').write_text(text)
    row = guard.snapshot(directory, root=root)
    assert row['memory_swap_current'] is None
    assert row['memory_swap_current_error']
    assert row['memory_stat'] == {'anon': 4096}


@pytest.mark.parametrize('case,expected', [
    ('safe', None), ('host', 'host RAM reserve reached'),
    ('ancestor', 'ancestor cgroup reserve reached'),
    ('some', 'memory pressure threshold reached'), ('full', 'memory pressure threshold reached'),
    ('oom', 'cgroup OOM event'), ('oom_kill', 'cgroup OOM event')])
def test_snapshot_optional_failures_preserve_stop_behavior(snapshot_cgroup, monkeypatch, case, expected):
    root, directory = snapshot_cgroup
    if case == 'host':
        monkeypatch.setattr(guard, 'memory_available', lambda: 99)
    elif case == 'ancestor':
        (directory.parent / 'memory.high').write_text('299')
    elif case in ('some', 'full'):
        (directory / 'memory.pressure').write_text(
            f'some avg10={30 if case == "some" else 0}\nfull avg10={10 if case == "full" else 0}\n')
    elif case in ('oom', 'oom_kill'):
        (directory / 'memory.events').write_text(f'{case} 1\n')
    before = guard.snapshot(directory, root=root)
    assert guard.stop_reason(before, 100) == expected
    (directory / 'memory.stat').write_text('anon -1\n')
    (directory / 'memory.swap.current').unlink()
    after = guard.snapshot(directory, root=root)
    for key in ('memory_current', 'memory_peak', 'memory_events', 'host_available',
                'ancestor_headroom', 'psi_some_avg10', 'psi_full_avg10'):
        assert after[key] == before[key]
    assert guard.stop_reason(after, 100) == expected


@pytest.mark.parametrize('filename,text,error', [
    ('memory.current', 'bad', ValueError), ('memory.peak', 'bad', ValueError),
    ('memory.events', 'oom bad\n', ValueError), ('memory.pressure', 'some avg10=0\n', KeyError),
    ('../memory.high', 'bad', ValueError), ('memory.current', None, FileNotFoundError)])
def test_snapshot_optional_errors_do_not_mask_safety_errors(snapshot_cgroup, filename, text, error):
    root, directory = snapshot_cgroup
    (directory / 'memory.stat').write_text('anon -1\n')
    (directory / 'memory.swap.current').unlink()
    if text is None:
        (directory / filename).unlink()
    else:
        (directory / filename).write_text(text)
    with pytest.raises(error):
        guard.snapshot(directory, root=root)
