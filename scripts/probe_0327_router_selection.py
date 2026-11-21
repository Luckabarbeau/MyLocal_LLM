#!/usr/bin/env python3
"""Isolated synthetic router selection A/B; no model or checkpoint imports.

Import/help use only the standard library. GPU execution requires the 0290 gate.
Each graph contains 48 identical-input, preallocated router calls. CUDA events
measure warmed, alternating old/warp graph batches, not Python dispatch loops.
Compilation, allocation, capture and parity readbacks are outside timing.
--warp-reductions selects the full exact-tree variant; default remains 0327.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCH_CALLS = 48


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; parent must exist')
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--floor', type=float, default=2**-14)
    parser.add_argument('--seed', type=int, default=327)
    parser.add_argument('--warmups', type=int, default=10, help='Graph replays per variant before timing')
    parser.add_argument('--replays', type=int, default=20, help='48-call graph replays per timed sample')
    parser.add_argument('--trials', type=int, default=9)
    parser.add_argument('--warp-reductions', action='store_true',
                        help='Compare original against warp selection plus exact-tree max/sum reductions')
    args = parser.parse_args(argv)
    if not 1 <= args.n <= 4096 or not 1 <= args.k <= min(32, args.n):
        parser.error('require n=1..4096 and k=1..min(32,n)')
    if not math.isfinite(args.floor) or not 1.1754943508222875e-38 <= args.floor <= 3.4028234663852886e38:
        parser.error('floor must be finite and within the supported FP32 range')
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be 0..2**32-1')
    if not (1 <= args.warmups <= 100 and 1 <= args.replays <= 1000 and 3 <= args.trials <= 51):
        parser.error('require warmups=1..100, replays=1..1000, trials=3..51')
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error('output parent must exist and output must be a new file')
    return args


def candidate_source(router, *, warp_reductions=False):
    source = router._REDUCE_SOURCE if warp_reductions else router._WARP_SOURCE
    restored = source
    if warp_reductions:
        restored = restored.replace(router._WARP_MAX_REDUCTION, router._MAX_REDUCTION, 1).replace(
            router._WARP_SUM_REDUCTION, router._SUM_REDUCTION, 1)
    restored = restored.replace(router._WARP_SELECTION, router._BLOCK_SELECTION, 1)
    return source, restored == router._SOURCE


def provenance(router, *, warp_reductions=False):
    paths = [Path(__file__), Path(router.__file__), ROOT / 'mini_llm/backend.py',
             ROOT / 'tests/test_0327_router_warp_selection.py',
             ROOT / 'tests/test_0329_router_reductions.py',
             ROOT / 'scripts/run_0290_isolated_benchmark.py']
    files = {}
    for path in paths:
        data = path.read_bytes()
        files[str(path.resolve().relative_to(ROOT))] = dict(sha256=digest(data), bytes=len(data))

    def git(*args):
        return subprocess.check_output(['git', '--no-pager', '--no-optional-locks', *args],
                                       cwd=ROOT, text=True).strip()

    selected, remainder_identical = candidate_source(router, warp_reductions=warp_reductions)
    return dict(files=files, git_head=git('rev-parse', 'HEAD'),
                git_status=git('status', '--short', '--', *files),
                python=sys.version, executable=sys.executable, argv=sys.argv,
                isolation_cgroup=Path('/proc/self/cgroup').read_text(),
                environment={key: value for key, value in sorted(os.environ.items())
                             if key.startswith('MINI_LLM_') or key in (
                                 'CUDA_VISIBLE_DEVICES', 'CUPY_CACHE_DIR', 'CUPY_ACCELERATORS')},
                compiler_options=list(router._OPTIONS),
                selected_candidate=dict(name='full' if warp_reductions else 'warp',
                                        sha256=digest(selected.encode()), source=selected,
                                        remainder_byte_identical=remainder_identical),
                timed_kernel_sources=dict(old='old', warp='full' if warp_reductions else 'warp'),
                kernels={name: dict(sha256=digest(source.encode()), source=source)
                         for name, source in (('old', router._SOURCE), ('warp', router._WARP_SOURCE),
                                              ('full', router._REDUCE_SOURCE))})


def run(args):
    from scripts.run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Fail closed before any optional GPU/backend import.
    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    import numpy as np
    import cupy as cp
    from mini_llm.ops import router_decode as router
    if router.BACKEND_NAME != 'cupy':
        raise RuntimeError('Router backend was imported before CuPy selection; use a fresh process')

    _, selected_remainder_identical = candidate_source(router, warp_reductions=args.warp_reductions)
    device = cp.cuda.runtime.getDevice()
    name = cp.cuda.runtime.getDeviceProperties(device)['name']
    if isinstance(name, bytes):
        name = name.decode()
    result = dict(scope='synthetic router only; no model/checkpoint/native CPU service',
                  provenance=provenance(router, warp_reductions=args.warp_reductions),
                  warp_reductions=args.warp_reductions, device=device, device_name=name,
                  cupy=cp.__version__, numpy=np.__version__,
                  cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                  cuda_driver=cp.cuda.runtime.driverGetVersion(),
                  n=args.n, k=args.k, floor=float(np.float32(args.floor)), seed=args.seed,
                  batch_calls=BATCH_CALLS, warmups=args.warmups,
                  replays_per_sample=args.replays, trials=args.trials,
                  precision=('Original max/sum FP32 trees with warp communication; exp/division/mass/floor unchanged'
                             if args.warp_reductions else
                             'Original max/softmax/division/mass/floor source; only selection reduction differs'),
                  timing='CUDA events around repeated warmed 48-call graph replays; includes graph execution '
                         'overhead and inter-replay host submission gaps, not model throughput or profiler kernel time',
                  overrides=dict(old=False, warp=True),
                  reduction_overrides=dict(old=False, warp=args.warp_reductions),
                  selected_remainder_byte_identical=selected_remainder_identical,
                  remainder_byte_identical=router._WARP_SOURCE.replace(
                      router._WARP_SELECTION, router._BLOCK_SELECTION, 1) == router._SOURCE)
    host = np.random.default_rng(args.seed).normal(0, 5, args.n).astype(np.float32)
    kernels = {key: router._get_kernel(device, warp_select=mode,
                                        warp_reduce=bool(mode and args.warp_reductions))
               for key, mode in (('old', False), ('warp', True))}
    samples = {key: [] for key in kernels}
    orders, graphs = [], {}
    with cp.cuda.Stream(non_blocking=True) as stream:
        result['stream_ptr'] = int(stream.ptr)
        logits = cp.asarray(host)
        expected_weights, expected_ids = router.try_normalized_topk(
            logits, args.k, args.floor, warp_select=False, warp_reduce=False)
        candidate_weights, candidate_ids = router.try_normalized_topk(
            logits, args.k, args.floor, warp_select=True, warp_reduce=args.warp_reductions)
        expected_weights, expected_ids = cp.asnumpy(expected_weights), cp.asnumpy(expected_ids)
        result['api_bitwise_equal'] = bool(
            np.array_equal(expected_weights.view(np.uint32), cp.asnumpy(candidate_weights).view(np.uint32))
            and np.array_equal(expected_ids, cp.asnumpy(candidate_ids)))
        outputs = {key: (cp.empty((BATCH_CALLS, args.k), cp.float32),
                         cp.empty((BATCH_CALLS, args.k), cp.int64)) for key in kernels}
        # Keep all views alive through graph retirement; capture allocates nothing.
        arguments = {key: [(logits, weights[i], ids[i], np.int32(args.n), np.int32(args.k),
                            np.float32(args.floor)) for i in range(BATCH_CALLS)]
                     for key, (weights, ids) in outputs.items()}
        for key, kernel in kernels.items():
            for call_args in arguments[key]:
                kernel((1,), (128,), call_args)  # Compile/warm before capture.
        stream.synchronize()

        def output_checks():
            checks = {}
            for key, (weights, ids) in outputs.items():
                actual_weights, actual_ids = cp.asnumpy(weights), cp.asnumpy(ids)
                checks[key] = dict(
                    finite=bool(np.isfinite(actual_weights).all()),
                    weights_bitwise_equal=bool(np.array_equal(actual_weights.view(np.uint32),
                        np.broadcast_to(expected_weights.view(np.uint32), actual_weights.shape))),
                    ids_exact=bool(np.array_equal(actual_ids,
                        np.broadcast_to(expected_ids, actual_ids.shape))))
            return checks

        result['precheck'] = output_checks()
        accepted = (result['remainder_byte_identical'] and result['selected_remainder_byte_identical']
                    and result['api_bitwise_equal']
                    and all(all(row.values()) for row in result['precheck'].values()))
        if accepted:
            for key, kernel in kernels.items():
                stream.begin_capture()
                for call_args in arguments[key]:
                    kernel((1,), (128,), call_args)
                graphs[key] = stream.end_capture()
            for warmup in range(args.warmups):
                for key in (('old', 'warp') if warmup % 2 == 0 else ('warp', 'old')):
                    graphs[key].launch(stream)
            stream.synchronize()
            events = {key: (cp.cuda.Event(), cp.cuda.Event()) for key in kernels}
            for trial in range(args.trials):
                order = ('old', 'warp') if trial % 2 == 0 else ('warp', 'old')
                orders.append(list(order))
                for key in order:
                    start, end = events[key]
                    start.record(stream)
                    for _ in range(args.replays):
                        graphs[key].launch(stream)
                    end.record(stream)
                    end.synchronize()
                    samples[key].append(float(cp.cuda.get_elapsed_time(start, end)))
        result['postcheck'] = output_checks()
        result['inputs_bitwise_unchanged'] = bool(np.array_equal(
            host.view(np.uint32), cp.asnumpy(logits).view(np.uint32)))
        result['current_stream_preserved'] = cp.cuda.get_current_stream().ptr == stream.ptr
        result['input_sha256'] = digest(host.tobytes())
        result['graph_kernel_nodes'] = {key: BATCH_CALLS for key in graphs}
        stream.synchronize()
        graphs.clear()  # Fence/retire before releasing any captured input/output view.

    result['accepted'] = bool(accepted and all(all(row.values()) for row in result['postcheck'].values())
                              and result['inputs_bitwise_unchanged'] and result['current_stream_preserved'])
    result['trial_orders'] = orders
    result['samples_total_ms'] = samples
    result['samples_batch_ms'] = {key: [v / args.replays for v in rows] for key, rows in samples.items()}
    result['samples_call_us'] = {key: [v * 1000 / (args.replays * BATCH_CALLS) for v in rows]
                                 for key, rows in samples.items()}
    medians = {key: statistics.median(rows) if rows else None
               for key, rows in result['samples_call_us'].items()}
    result['median_call_us'] = medians
    result['paired_old_over_warp'] = [a / b for a, b in zip(samples['old'], samples['warp'])] if (
        result['accepted'] and all(v > 0 for rows in samples.values() for v in rows)) else []
    result['median_old_over_warp'] = (medians['old'] / medians['warp']
                                    if result['accepted'] and medians['warp'] else None)
    return result


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    result = run(args)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(f"0327 accepted={result['accepted']} old/warp us={result['median_call_us']} "
          f"ratio={result['median_old_over_warp']} report={args.output}", flush=True)
    if not result['accepted']:
        raise RuntimeError('Router exactness/input-preservation failed; timing is not acceptance')


if __name__ == '__main__':
    main()
