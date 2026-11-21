#!/usr/bin/env python3
"""Run bounded diagnostics outside the desktop cgroup; never import CUDA/model code.

Kernel limits are authoritative. The polling guard is an additional early stop,
not a guarantee against global OOM, driver failures, or competing applications.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

GIB = 1024**3
CGROUP_ROOT = Path('/sys/fs/cgroup')


def memory_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) * 1024


def membership(text):
    paths = [line[3:] for line in text.splitlines() if line.startswith('0::')]
    if len(paths) != 1:
        raise RuntimeError('Verified cgroup-v2 membership is required')
    path = Path(paths[0])
    if not path.is_absolute() or '..' in path.parts:
        raise RuntimeError('Invalid cgroup namespace path')
    return path


def verify_limits(directory, relative, unit, high, maximum):
    if relative.name != unit or relative.parent.name != 'app.slice' or 'session.slice' in relative.parts:
        raise RuntimeError('Workload must be a distinct app.slice service, outside the desktop cgroup')
    for name, expected in [('memory.high', high), ('memory.max', maximum),
                           ('memory.swap.max', 0), ('memory.oom.group', 1)]:
        actual = (directory / name).read_text().strip()
        if actual != str(expected):
            raise RuntimeError(f'Unverified {name}: expected {expected}, got {actual}')


def require_isolation():
    """CLI precondition for model probes, including descendants of the gate."""
    relative = membership(Path('/proc/self/cgroup').read_text())
    directory = CGROUP_ROOT.joinpath(*relative.parts[1:])
    if not relative.name.startswith('mylocal-bench-'):
        raise RuntimeError('Use scripts/run_0290_isolated_benchmark.py; direct model probes are disabled')
    try:
        high = int((directory / 'memory.high').read_text())
        maximum = int((directory / 'memory.max').read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('Finite service memory limits are required') from exc
    if not 0 < high < maximum:
        raise RuntimeError('Require finite MemoryHigh below MemoryMax')
    verify_limits(directory, relative, relative.name, high, maximum)


def pressure(path):
    rows = {}
    for line in path.read_text().splitlines():
        name, *fields = line.split()
        rows[name] = dict(field.split('=', 1) for field in fields)
    # Missing/malformed PSI is a safety-check failure, not zero pressure.
    return {name: float(rows[name]['avg10']) for name in ('some', 'full')}


def _memory_diagnostic(path, *, key_values=False):
    """Optional telemetry only; never use this reader for safety checks."""
    try:
        with path.open('rb') as stream:
            data = stream.read(64 * 1024)
        # Reject a full buffer rather than read beyond the cap to check EOF.
        if len(data) == 64 * 1024:
            raise ValueError('Telemetry reaches the 64 KiB read limit')
        text = data.decode('ascii')

        def count(value):
            if not value or not value.isascii() or not value.isdecimal():
                raise ValueError('Expected a nonnegative decimal count')
            return int(value)

        if not key_values:
            return count(text.strip()), None
        values = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError('Expected one key and count per line')
            key, value = fields
            if not key.isascii() or not key.isidentifier():
                raise ValueError('Invalid memory.stat key')
            if key in values:
                raise ValueError(f'Duplicate memory.stat key: {key}')
            values[key] = count(value)
        if not values:
            raise ValueError('Empty memory.stat')
        return values, None
    except (OSError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def snapshot(directory, *, root=CGROUP_ROOT):
    readings = []
    current = directory
    headroom = None
    while True:
        readings.append(pressure(current / 'memory.pressure'))
        if current != directory and current != root:
            for limit_name in ('memory.high', 'memory.max'):
                limit = (current / limit_name).read_text().strip()
                if limit != 'max':
                    free = max(0, int(limit) - int((current / 'memory.current').read_text()))
                    headroom = free if headroom is None else min(headroom, free)
        if current == root:
            break
        current = current.parent
    readings.append(pressure(Path('/proc/pressure/memory')))
    row = dict(memory_current=int((directory / 'memory.current').read_text()),
                memory_peak=int((directory / 'memory.peak').read_text()),
                memory_events=dict((key, int(value)) for key, value in
                                   (line.split() for line in (directory / 'memory.events').read_text().splitlines())),
                host_available=memory_available(), ancestor_headroom=headroom,
                psi_some_avg10=max(row['some'] for row in readings),
                psi_full_avg10=max(row['full'] for row in readings))
    # Raw memory.stat keys mix byte quantities and event/page counters, not all bytes.
    for name, filename, key_values in [('memory_stat', 'memory.stat', True),
                                      ('memory_swap_current', 'memory.swap.current', False)]:
        row[name], error = _memory_diagnostic(directory / filename, key_values=key_values)
        if error is not None:
            row[f'{name}_error'] = error
    return row


def stop_reason(row, reserve):
    if row['host_available'] < reserve:
        return 'host RAM reserve reached'
    if row['ancestor_headroom'] is not None and row['ancestor_headroom'] < reserve:
        return 'ancestor cgroup reserve reached'
    if row['psi_full_avg10'] >= 10 or row['psi_some_avg10'] >= 30:
        return 'memory pressure threshold reached'
    if row['memory_events'].get('oom', 0) or row['memory_events'].get('oom_kill', 0):
        return 'cgroup OOM event'
    return None


def command_line(args, unit):
    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect',
               '--service-type=exec', '--slice=app.slice', f'--unit={unit}',
               f'--working-directory={Path.cwd()}',
               f'--property=MemoryHigh={args.high}', f'--property=MemoryMax={args.maximum}',
               '--property=MemorySwapMax=0', '--property=MemoryAccounting=yes',
               '--property=OOMPolicy=kill', '--property=KillMode=control-group',
               '--property=TimeoutStopSec=5s', f'--property=RuntimeMaxSec={args.timeout + 10}s']
    # User services do not inherit the caller's environment. Preserve explicit
    # inference/thread choices, not secrets or arbitrary systemd manager state.
    for key, value in sorted(os.environ.items()):
        if key.startswith(('MINI_LLM_', 'GGML_')) or key in (
                'CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'PYTHONPATH'):
            command.append(f'--setenv={key}={value}')
    command += ['--', sys.executable, str(Path(__file__).resolve()),
                '--gate-unit', unit, '--memory-high-mib', str(args.high // 1024**2),
                '--memory-max-mib', str(args.maximum // 1024**2),
                '--reserve-gib', str(args.reserve / GIB), '--timeout', str(args.timeout),
                '--output', str(args.output.resolve()), '--', *args.command]
    return command


def gate(args):
    relative = membership(Path('/proc/self/cgroup').read_text())
    directory = CGROUP_ROOT.joinpath(*relative.parts[1:])
    verify_limits(directory, relative, args.gate_unit, args.high, args.maximum)
    row = snapshot(directory)
    reason = stop_reason(row, args.reserve)
    free = min(row['host_available'], row['ancestor_headroom'] or row['host_available'])
    if reason or free < args.maximum + args.reserve:
        raise RuntimeError(reason or 'Insufficient RAM for full hard limit plus reserve')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'guard.jsonl').open('x', buffering=1) as log:
        def record(value):
            log.write(json.dumps(value, allow_nan=False) + '\n')
        record(dict(event='verified_before_exec', cgroup=str(relative), high=args.high,
                    maximum=args.maximum, reserve=args.reserve, command=args.command, **row))
        process = None
        started = time.monotonic()
        reason = None
        try:
            process = subprocess.Popen(args.command, start_new_session=True)
            while True:
                code = process.poll()
                row = snapshot(directory)
                elapsed = time.monotonic() - started
                record(dict(event='sample', seconds=elapsed, **row))
                reason = stop_reason(row, args.reserve)
                if row['memory_current'] >= args.high:
                    reason = reason or 'MemoryHigh reached; stopping before sustained reclaim'
                if elapsed >= args.timeout:
                    reason = reason or 'workload timeout'
                if reason:
                    raise RuntimeError(reason)
                if code is not None:
                    record(dict(event='exit', returncode=code, seconds=elapsed))
                    return code if code >= 0 else 128 - code
                time.sleep(.5)
        except BaseException as exc:
            record(dict(event='aborted', reason=str(exc)))
            raise
        finally:
            # systemd KillMode also retires descendants outside this process group.
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-high-mib', type=int, default=768)
    parser.add_argument('--memory-max-mib', type=int, default=1024)
    parser.add_argument('--reserve-gib', type=float, default=12)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--output', type=Path, required=True, help='New, exclusive artifact directory')
    parser.add_argument('--gate-unit', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    if (not args.command or not 64 <= args.memory_high_mib < args.memory_max_mib
            or not math.isfinite(args.reserve_gib) or args.reserve_gib < 12
            or not 1 <= args.timeout <= 1800):
        parser.error('Require command, 64MiB <= high < max, reserve >=12GiB, timeout 1..1800s')
    args.high = args.memory_high_mib * 1024**2
    args.maximum = args.memory_max_mib * 1024**2
    args.reserve = int(args.reserve_gib * GIB)
    if args.output.exists():
        parser.error('Artifact directory already exists; previous results must be preserved')
    return args


def main():
    args = parse_args()
    if args.gate_unit:
        return gate(args)
    if memory_available() < args.maximum + args.reserve:
        raise RuntimeError('Insufficient available host RAM for hard limit plus reserve')
    unit = f'mylocal-bench-{uuid.uuid4().hex}.service'
    # Only the small launcher remains in the desktop cgroup. There is no direct
    # fallback when systemd, delegation, limits or the gate are unavailable.
    try:
        return subprocess.run(command_line(args, unit), timeout=args.timeout + 30).returncode
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f'Isolated benchmark refused/aborted: {error}', file=sys.stderr)
        sys.exit(125)
