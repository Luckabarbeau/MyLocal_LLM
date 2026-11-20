#!/usr/bin/env python3
"""Profile one forced RESPONSE through 0302/0254 under the 0290 strict caps.

All configuration flags are forwarded to 0302. Only its default warmup + one
response workload is supported; forced trajectories are capped at 359 tokens.
Artifacts are exclusive .pstats, bounded top-function .txt, and full-record .json
files in the directory created by the unchanged 0254 parent. No profile is
printed to the console. cProfile overhead is included: NOT a throughput or
optimization claim. The supplied 42.2s token-graph / 43.7s default-pipeline
observations are context only, not measurements made by this probe.
"""
from __future__ import annotations

from contextlib import ExitStack
import cProfile
from functools import wraps
import importlib
import itertools
import json
import os
from pathlib import Path
import pstats
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_0311_local_ple import argv_for
WORKLOAD = ROOT / 'scripts/probe_0254_strata_workload.py'
CONFIGURATION = ROOT / 'scripts/probe_0302_decode_configuration.py'
MAX_FORCED_TOKENS = 359
INSTANCE_INDICES = itertools.count(1)
TOKEN_GRAPH_METHODS = ('_validate_live', '_configuration', '_validate_states',
                       'is_eligible', '_source_identity')
TIMING_NOTE = ('cProfile submitting-thread host wall time, including Python and native '
               'calls and GPU-blocking API waits; NOT CPU utilization or GPU kernel time. '
               'Self time excludes callees; cumulative times overlap and are not additive. '
               'Other/native worker threads are not profiled. No added CUDA sync/events.')


def function_records(stats):
    """Keep every pstats function, including builtins; do not strip source paths."""
    return sorted((dict(filename=filename, line=line, function=name,
                        primitive_calls=primitive, total_calls=total,
                        self_seconds=self_seconds, cumulative_seconds=cumulative)
                   for (filename, line, name),
                   (primitive, total, self_seconds, cumulative, _callers) in stats.stats.items()),
                  key=lambda row: (-row['cumulative_seconds'], row['filename'],
                                   row['line'], row['function']))


def cost_breakdown(records):
    """Expose exact graph methods and explicitly heuristic copy/wait candidates."""
    graph = {name: [row for row in records
                    if row['filename'].replace('\\', '/').endswith('/qwen4exp_token_graph.py')
                    and row['function'] == name]
             for name in TOKEN_GRAPH_METHODS}
    candidates = [row for row in records
                  if any(word in (row['filename'] + ' ' + row['function']).lower()
                         for word in ('cuda', 'cupy'))
                  and any(word in row['function'].lower()
                          for word in ('memcpy', 'copy', 'synchronize', 'wait', 'get'))]
    return dict(token_graph_methods=graph, cuda_copy_or_wait_candidates=candidates,
                interpretation='Graph metadata validation is host work; its elapsed costs '
                'are not a CPU-utilization measurement. Copy/wait matches are heuristic, '
                'may overlap, and include queued GPU completion, not just transfer time. '
                'Empty lists mean no matching recorded calls, not proof of zero cost.')


def write_profile(output, profiler, metadata):
    """0254 owns directory creation. Reserve every artifact exclusively first."""
    stem = (f'python_critical_path_pid{os.getpid()}_instance{metadata["instance_index"]:03d}'
            f'_repeat{metadata["repeat_index"]:03d}')
    paths = {kind: output / (stem + suffix) for kind, suffix in
             (('pstats', '.pstats'), ('top', '.txt'), ('json', '.json'))}
    with ExitStack() as stack:
        # dump_stats opens its own filename. Reserve ownership before allowing it
        # to write; never overwrite a profile from an earlier replay/instance.
        stack.enter_context(paths['pstats'].open('xb')).close()
        text = stack.enter_context(paths['top'].open('x'))
        data = stack.enter_context(paths['json'].open('x'))
        profiler.dump_stats(str(paths['pstats']))
        # A failure before sample 1 (or a one-token trajectory) has no calls.
        stats = (pstats.Stats(str(paths['pstats']), stream=text) if profiler.stats
                 else pstats.Stats(stream=text))
        records = function_records(stats)
        text.write(TIMING_NOTE + '\nProfiling overhead included; NOT a rate claim.\n')
        text.write(f'Status: {metadata["status"]}; error: {metadata["error"]}\n')
        stats.sort_stats('cumulative').print_stats(60)
        stats.sort_stats('tottime').print_stats(60)
        json.dump(dict(**metadata, timing=TIMING_NOTE, functions=records,
                       costs=cost_breakdown(records),
                       artifacts={key: str(path) for key, path in paths.items()}),
                  data, indent=2, allow_nan=False)
        data.write('\n')
    return paths


class DecodeProfiler:
    """One submitting-thread profile per non-warmup replay; no model hooks."""

    def __init__(self, output, *, profile_factory=cProfile.Profile,
                 clock=time.perf_counter, emit=write_profile):
        self.output = Path(output)
        self.profile_factory, self.clock, self.emit = profile_factory, clock, emit
        self.instance_index = next(INSTANCE_INDICES)
        self.repeat_index = 0
        self.active = False

    def replay(self, function):
        @wraps(function)
        def wrapped(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
            if row['request_id'] == 'warmup':
                return function(model, tokenizer, row, prompt, xp, sample_token, guard, journal)
            if self.active:
                raise RuntimeError('Nested replay profiling is unsupported')
            forced = row.get('_forced_output_token_ids')
            if (not isinstance(forced, list) or not 1 <= len(forced) <= MAX_FORCED_TOKENS
                    or len(forced) > row['request']['max_tokens']
                    or any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                           for token in forced)):
                raise ValueError('Profile requires a forced trajectory of 1..359 token IDs')
            # Determine the stop boundary from the actual forced trajectory,
            # never from the sampler's discarded proposed token.
            last_sample = next((i for i, token in enumerate(forced, 1)
                                if token in model.stop_token_ids), len(forced))
            profiler = self.profile_factory()
            self.repeat_index += 1
            self.active = True
            samples = 0
            started = ended = None
            enabled = False
            error = None
            returned = False

            @wraps(sample_token)
            def sample(*args, **kwargs):
                nonlocal samples, started, ended, enabled
                token = sample_token(*args, **kwargs)
                samples += 1
                if samples == last_sample:
                    # This must precede text decode, journal, sync and cleanup.
                    if enabled:
                        profiler.disable()
                        enabled = False
                    ended = self.clock()
                elif samples == 1:
                    started = self.clock()
                    profiler.enable()
                    enabled = True
                return token

            try:
                result = function(model, tokenizer, row, prompt, xp, sample, guard, journal)
                returned = True
                return result
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                try:
                    if enabled:
                        profiler.disable()
                        ended = self.clock()
                finally:
                    self.active = False
                    complete = returned and samples == len(forced)
                    self.emit(self.output, profiler, dict(
                        request_id=row['request_id'], instance_index=self.instance_index,
                        repeat_index=self.repeat_index, status='complete' if complete else 'incomplete',
                        error=error, sample_calls=samples, expected_sample_calls=len(forced),
                        last_sample_boundary=last_sample, forced_trajectory=True,
                        diagnostic_only=True, rate_claim=False, optimization_claim=False,
                        profiling_overhead_included=True, profiling_overhead_seconds=None,
                        start_label='after_first_original_sampler_return',
                        end_label=('after_last_original_sampler_return' if samples == last_sample
                                   else 'exception_or_replay_exit_before_last_sample'),
                        profile_started=started is not None,
                        profiled_decode_host_seconds=(None if started is None or ended is None
                                                      else ended-started),
                        context_only_seconds=dict(token_graph=42.2, default_pipeline=43.7)))
        return wrapped


def argument_value(arguments, flag, default=None):
    value = default
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 == len(arguments):
                raise ValueError(f'{flag} requires a value')
            value = arguments[index + 1]
        elif argument.startswith(flag + '='):
            value = argument.split('=', 1)[1]
    return value


def validate_arguments(arguments, *, bootstrap=False):
    if any(flag in arguments for flag in ('--trace', '--telemetry', '--parity', '--natural')):
        raise ValueError('Only one forced warmup + RESPONSE is supported; no trace/telemetry/parity/natural')
    if argument_value(arguments, '--response-repeats', '1') != '1':
        raise ValueError('Use --response-repeats 1: profile one RESPONSE only')
    if bootstrap:
        # 0302 emits no --cases; pin its established default rather than permit
        # direct bootstrap callers to select another workload or multiple cases.
        for index, argument in enumerate(arguments):
            if argument == '--cases':
                end = next((i for i in range(index + 1, len(arguments))
                            if arguments[i].startswith('--')), len(arguments))
                if arguments[index + 1:end] != ['warmup', 'performance-short-r1']:
                    raise ValueError('Require warmup + performance-short-r1 only')
            elif argument.startswith('--cases='):
                raise ValueError('Require warmup + performance-short-r1 only')
        if argument_value(arguments, '--force-tokens-from') is None:
            raise ValueError('Bootstrap requires --force-tokens-from')
    output = argument_value(arguments, '--output')
    if output is None:
        raise ValueError('Requires --output')
    return Path(output)


def bootstrap(arguments):
    output = validate_arguments(arguments, bootstrap=True)
    workload = importlib.import_module('probe_0254_strata_workload')
    original_file, original_replay = workload.__file__, workload.replay_case
    timer = DecodeProfiler(output)
    try:
        # Preserve the 0254 parent, Popen, caps and guard. Its child hop comes
        # back here so only replay_case is wrapped, after backend selection.
        workload.__file__ = str(Path(__file__).resolve())
        workload.replay_case = timer.replay(original_replay)
        with argv_for(WORKLOAD, arguments):
            return workload.main()
    finally:
        workload.__file__, workload.replay_case = original_file, original_replay


def launch(arguments):
    validate_arguments(arguments)
    original_run = subprocess.run

    @wraps(original_run)
    def intercepted(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 2
                and str(command[0]) == sys.executable and Path(command[1]).resolve() == WORKLOAD):
            command = [sys.executable, str(Path(__file__).resolve()), '--bootstrap', *command[2:]]
        return original_run(command, *args, **kwargs)

    try:
        subprocess.run = intercepted
        with argv_for(CONFIGURATION, arguments):
            return runpy.run_path(str(CONFIGURATION), run_name='__main__')
    finally:
        subprocess.run = original_run


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Before patches, runpy, workload or backend imports.
    if arguments[:1] == ['--bootstrap']:
        return bootstrap(arguments[1:])
    if '--child' in arguments:
        return bootstrap(arguments)
    return launch(arguments)


if __name__ == '__main__':
    main()
