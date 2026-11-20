#!/usr/bin/env python3
"""Wrap 0302 with local PLE host diagnostics (instrumented, not a rate claim).

Use the 0290 isolation launcher. All public arguments belong to 0302 and are
forwarded unchanged. ple_timing.json is append-only JSON Lines, one per replay.
No extra CUDA events/synchronizations are introduced. End snapshots include
0254's untimed request cleanup; they are not exact decode-completion brackets.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
import importlib
import json
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / 'scripts/probe_0254_strata_workload.py'
CONFIGURATION = ROOT / 'scripts/probe_0302_decode_configuration.py'


def thread_major_faults():
    return resource.getrusage(resource.RUSAGE_THREAD).ru_majflt


def empty_counters():
    return dict(embedding_calls=0, embedding_host_seconds=0.0,
                embedding_major_faults=0, rows_calls=0, rows_host_seconds=0.0,
                decoder_calls=0, decoder_host_seconds=0.0)


def snapshot_delta(start, end):
    """Process counters, unlike the embedding-only submitting-thread faults."""
    return dict(
        io={key: end['io'][key] - value for key, value in start.get('io', {}).items()
            if key in end.get('io', {})},
        **{key: end[key] - start[key] for key in ('minor_faults', 'major_faults')
           if key in start and key in end})


class PLETimer:
    """Thread-local request/embedding scope; all wrapped math remains untouched."""

    def __init__(self, *, clock=time.perf_counter, faults=thread_major_faults):
        self.clock, self.faults = clock, faults
        self.local = threading.local()

    def embedding(self, function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            request = getattr(self.local, 'request', None)
            if request is None or getattr(self.local, 'active', False):
                return function(*args, **kwargs)
            counters = request['phases'][request['phase']]
            before_faults = self.faults()
            started = self.clock()
            self.local.active = True
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = self.clock() - started
                after_faults = self.faults()
                self.local.active = False
                counters['embedding_calls'] += 1
                counters['embedding_host_seconds'] += elapsed
                counters['embedding_major_faults'] += after_faults - before_faults
        return wrapped

    def component(self, function, name):
        @wraps(function)
        def wrapped(*args, **kwargs):
            # Neither unrelated row access/dequantization nor another thread's
            # work belongs to the PLE embedding, even during the same replay.
            flag = name + '_active'
            if not getattr(self.local, 'active', False) or getattr(self.local, flag, False):
                return function(*args, **kwargs)
            counters = self.local.request['phases'][self.local.request['phase']]
            started = self.clock()
            setattr(self.local, flag, True)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = self.clock() - started
                setattr(self.local, flag, False)
                counters[name + '_calls'] += 1
                counters[name + '_host_seconds'] += elapsed
        return wrapped

    @contextmanager
    def instrument(self, ple_type, packed_type, store_type):
        originals = [(ple_type, 'embedding', ple_type.embedding),
                     (packed_type, 'dequantize_reference', packed_type.dequantize_reference),
                     (store_type, 'rows', store_type.rows)]
        try:
            ple_type.embedding = self.embedding(originals[0][2])
            packed_type.dequantize_reference = self.component(originals[1][2], 'decoder')
            store_type.rows = self.component(originals[2][2], 'rows')
            yield
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)

    @contextmanager
    def model_hooks(self):
        # 0254 selects the CuPy backend in child(), not at module import. Wait
        # until replay so importing the PLE cannot accidentally select NumPy.
        from mini_llm.external.qwen4exp_ple import Qwen4ExpPLE
        from mini_llm.ops.quantized_weight import PackedGGUFWeight
        from mini_llm.external.gguf_store import MultiShardTensorStore
        with self.instrument(Qwen4ExpPLE, PackedGGUFWeight, MultiShardTensorStore):
            yield

    def replay(self, function, snapshot, emit, *, hooks=nullcontext):
        @wraps(function)
        def wrapped(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
            previous = getattr(self.local, 'request', None)
            if previous is not None:
                raise RuntimeError('Nested replay instrumentation is unsupported')
            request = dict(phase='prefill', phases={phase: empty_counters()
                           for phase in ('prefill', 'decode')})
            self.local.request = request
            samples = 0
            start_snapshot = None
            first_sample_time = None
            started = self.clock()
            error = None

            @wraps(sample_token)
            def sample(*args, **kwargs):
                nonlocal samples, start_snapshot, first_sample_time
                token = sample_token(*args, **kwargs)
                samples += 1
                if samples == 1:
                    # Sampling (including host-ID completion) finishes before
                    # phase transition. Never replace the sampled/forced IDs.
                    first_sample_time = self.clock()
                    request['phase'] = 'decode'
                    start_snapshot = snapshot()
                return token

            try:
                with hooks():
                    return function(model, tokenizer, row, prompt, xp, sample, guard, journal)
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                ended = self.clock()
                self.local.request = previous
                end_snapshot = snapshot() if start_snapshot is not None else None
                totals = {key: sum(phase[key] for phase in request['phases'].values())
                          for key in empty_counters()}
                emit(dict(request_id=row['request_id'], diagnostic_only=True,
                    rate_claim=False, control_overhead_included=True,
                    timing='perf_counter host elapsed; no added CUDA sync/events; inclusive, not additive',
                    fault_scope='embedding only, resource.RUSAGE_THREAD',
                    forced_trajectory=row.get('_forced_output_token_ids') is not None,
                    sample_calls=samples, replay_host_seconds_including_untimed_cleanup=ended-started,
                    phases=request['phases'], total=totals, error=error,
                    decode_bracket=dict(
                        start_label='after_first_sample_completion',
                        end_label='after_replay_exit_including_untimed_cleanup',
                        host_seconds=None if first_sample_time is None else ended-first_sample_time,
                        decode_start=start_snapshot, decode_end=end_snapshot,
                        process_io_fault_delta=None if end_snapshot is None else
                            snapshot_delta(start_snapshot, end_snapshot))))
        return wrapped


def append_diagnostic(path, record):
    # The unmodified 0254 parent must already have created its exclusive output.
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


@contextmanager
def argv_for(script, arguments):
    previous = sys.argv
    sys.argv = [str(script), *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def bootstrap(arguments):
    workload = importlib.import_module('probe_0254_strata_workload')
    # Its unchanged parent builds a --child command from __file__. Route that
    # hop back here too, without replacing its supervisor, limits, or Popen.
    original_file, original_replay = workload.__file__, workload.replay_case
    parser_output = next((value.split('=', 1)[1] for value in arguments
                          if value.startswith('--output=')), None)
    if parser_output is None and '--output' in arguments:
        parser_output = arguments[arguments.index('--output') + 1]
    if parser_output is None:
        raise ValueError('Bootstrap requires --output')
    timer = PLETimer()
    try:
        workload.__file__ = str(Path(__file__).resolve())
        workload.replay_case = timer.replay(original_replay, workload.memory_snapshot,
            lambda record: append_diagnostic(Path(parser_output) / 'ple_timing.json', record),
            hooks=timer.model_hooks)
        with argv_for(WORKLOAD, arguments):
            return workload.main()
    finally:
        workload.__file__, workload.replay_case = original_file, original_replay


def launch(arguments):
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
    require_isolation()  # Before any monkeypatch, runpy, or model imports.
    if arguments[:1] == ['--bootstrap']:
        return bootstrap(arguments[1:])
    if '--child' in arguments:
        return bootstrap(arguments)
    return launch(arguments)


if __name__ == '__main__':
    main()
