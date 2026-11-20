#!/usr/bin/env python3
"""Isolated synthetic serial/warp-dot QSA A/B; never model performance.

Run only via run_0290_isolated_benchmark.py. Kernel-path events wrap one graph
of preallocated raw calls (100 by default), avoiding inter-call Python gaps.
API events wrap allocating calls and include host validation/launch gaps.
Compilation, capture, setup, copies and parity checks are outside timing.
Full provenance goes only to a new JSON file; stdout is a compact summary.
"""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def provenance(serial, warp, helpers, value_tile, finalizer_source):
    def git(*args):
        return subprocess.check_output(['git', '--no-pager', '--no-optional-locks', *args],
                                       cwd=ROOT, text=True).strip()

    files = [Path(__file__), Path(serial.__file__), Path(warp.__file__), Path(helpers.__file__),
             ROOT / 'scripts/run_0290_isolated_benchmark.py', ROOT / 'tests/test_0320_qsa_warp_dots.py']
    paths = [str(p.resolve().relative_to(ROOT)) for p in files]
    return dict(git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                git_diff=git('diff', '--', *paths), python=sys.version,
                executable=sys.executable, argv=sys.argv,
                isolation_cgroup=Path('/proc/self/cgroup').read_text(),
                environment={key: value for key, value in sorted(os.environ.items())
                             if key.startswith('MINI_LLM_') or key in (
                                 'CUDA_VISIBLE_DEVICES', 'CUPY_CACHE_DIR', 'CUPY_ACCELERATORS')},
                files={name: dict(sha256=digest(p.read_text()), source=p.read_text())
                       for name, p in zip(paths, files)},
                value_tile=value_tile,
                kernels={name: dict(sha256=digest(source), source=source) for name, source in (
                    ('serial', serial._SOURCE), ('warp_dots', warp._DOTS_SOURCE),
                    ('warp_finalize_default', warp._SOURCE),
                    ('warp_finalize', finalizer_source))}, compiler_options=list(warp._OPTIONS))


def exact_check(np, cp, operators, inputs, n, scale):
    outputs = {name: cp.asnumpy(op(*inputs, n, scale)) for name, op in operators.items()}
    a, b = outputs['warp'], outputs['serial']
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    equal = bool(np.array_equal(a, b))
    bits = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    return dict(finite=finite, array_equal=equal, bitwise_equal=bits,
                differing_fp32_bits=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
                max_abs=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if finite else None,
                accepted=finite and equal and bits), outputs


def print_summary(result, output):
    print(f"0320 value_tile={result['value_tile']} accepted={result['accepted']} report={output}", flush=True)
    for row in result['cases']:
        timings = []
        for mode in ('kernel', 'api'):
            medians = row['median_ms'][mode]
            a, b = medians['serial'], medians['warp']
            timings.append(f'{mode}=unavailable' if a is None or b is None else
                           f'{mode}(serial/warp)={a:.6f}/{b:.6f}ms ratio={a/b:.3f}x')
        print(f"n={row['n']} exact={row['accepted']} " + ' '.join(timings), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; parent must exist')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--value-tile', type=int, choices=(0, 32, 64, 128), default=0)
    args = parser.parse_args()
    if not (1 <= args.warmups <= 100 and 1 <= args.repetitions <= 1000 and 1 <= args.trials <= 25):
        parser.error('require warmups 1..100, repetitions 1..1000, trials 1..25')
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error('output parent must exist and output must be a new file')
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # No fallback, before any CUDA import or initialization.

    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    import numpy as np
    import cupy as cp
    from mini_llm.ops import qsa_decode_gpu as serial
    from mini_llm.ops import qsa_warp_dots_gpu as warp
    import probe_0319_qsa_staged_operator as helpers

    # Original-source launch avoids any environment-enabled model dispatch.
    old = cp.RawKernel(serial._SOURCE, 'qsa_decode', options=warp._OPTIONS)
    dots = cp.RawKernel(warp._DOTS_SOURCE, 'qsa_warp_dots', options=warp._OPTIONS)
    finalizer_source = warp.finalizer_source(serial._SOURCE, value_tile=args.value_tile)
    finalize = cp.RawKernel(finalizer_source, 'qsa_decode', options=warp._OPTIONS)

    def serial_call(q, qg, k, v, keys, values, n, scale):
        output = cp.empty(q.size, dtype=cp.float32)
        strides = tuple(np.int64(s // a.itemsize)
                        for a in (q, qg, k, v, keys, values) for s in a.strides)
        old((q.shape[0],), (128,),
            (q, qg, k, v, keys, values, output, np.int32(n), np.int32(256),
             np.int32(q.shape[0] // k.shape[0]), np.float32(scale), *strides))
        return output

    def kernel_calls(inputs, n):
        q, qg, k, v, keys, values = inputs
        outputs = {name: cp.empty(q.size, cp.float32) for name in ('serial', 'warp')}
        logits = cp.empty(q.shape[0] * n, cp.float32)
        strides = tuple(np.int64(s // a.itemsize) for a in inputs for s in a.strides)
        dot_strides = tuple(np.int64(s // a.itemsize) for a in (q, k, keys) for s in a.strides)
        common = (np.int32(n), np.int32(256), np.int32(q.shape[0] // k.shape[0]), np.float32(.0625))
        dot_args = (q, k, keys, logits, common[0], common[2], common[3], np.int32(n), *dot_strides)
        serial_args = (*inputs, outputs['serial'], *common, *strides)
        final_args = (*inputs, outputs['warp'], *common, logits, np.int32(n), *strides)

        def serial_kernel():
            old((q.shape[0],), (128,), serial_args)

        def warp_kernels():
            dots((q.shape[0], (n+3)//4), (128,), dot_args)
            grid = (q.shape[0],) if args.value_tile == 0 else (q.shape[0], (256+args.value_tile-1)//args.value_tile)
            finalize(grid, (128,), final_args)

        return dict(serial=serial_kernel, warp=warp_kernels), outputs, logits

    operators = dict(serial=serial_call, warp=partial(warp.qsa_warp_dots_gpu, value_tile=args.value_tile))
    restored = finalizer_source
    if args.value_tile:
        restored = restored.replace(warp._TILED_VALUE_LOOP.format(tile=args.value_tile), warp._VALUE_LOOP)
    restored = restored.replace(warp._LOGITS_SIGNATURE, warp._SIGNATURE, 1).replace(
        warp._LOGITS_DOT, warp._SERIAL_DOT, 1)
    remainder_unchanged = restored == serial._SOURCE
    device = cp.cuda.runtime.getDevice()
    name = cp.cuda.runtime.getDeviceProperties(device)['name']
    if isinstance(name, bytes):
        name = name.decode()
    result = dict(scope='synthetic standalone QSA operator only; no model/weights/quantization',
                  precision='FP32 products and d=0..255 left-to-right additions; --fmad=false; uint32 parity',
                  provenance=provenance(serial, warp, helpers, args.value_tile, finalizer_source),
                  remainder_byte_identical=remainder_unchanged, value_tile=args.value_tile,
                  finalizer_grid=[24] if args.value_tile == 0 else [24, (256+args.value_tile-1)//args.value_tile],
                  device=device, device_name=name, cupy=cp.__version__, numpy=np.__version__,
                  cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
                  heads=24, kv_heads=2, dim=256, scale=.0625, seed=320,
                  warmups=args.warmups, repetitions=args.repetitions, trials=args.trials,
                  timing=dict(kernel='Events around one graph replay of preallocated raw operator calls; '
                                     'includes graph execution overhead, excludes Python gaps within graph; '
                                     'not individual-kernel profiler duration',
                              api='Events around allocating API loops; includes host validation/launch gaps',
                              wall='Host perf_counter including end-event wait, divided by repetitions',
                              order='Alternating serial/warp and kernel/API order per trial; no speedup assertion'),
                  view_checks=[], cases=[])
    rng = np.random.default_rng(320)
    with cp.cuda.Stream(non_blocking=True) as stream:
        result['stream_ptr'] = int(stream.ptr)
        for n in (141, 435, 1024):
            for layout in ('plain', 'strided', 'reverse', 'broadcast'):
                inputs, bases = helpers.make_inputs(np, cp, rng, n, layout)
                before = helpers.snapshots(np, cp, bases)
                check, _ = exact_check(np, cp, operators, inputs, n, .0625)
                intact = helpers.unchanged(np, cp, bases, before)
                check.update(n=n, layout=layout, inputs_bitwise_unchanged=intact,
                             strides_bytes=[list(a.strides) for a in inputs])
                check['accepted'] = check['accepted'] and intact
                result['view_checks'].append(check)
            inputs, bases = helpers.make_inputs(np, cp, rng, n, current_k_case=True)
            before = helpers.snapshots(np, cp, bases)
            check, outputs = exact_check(np, cp, operators, inputs, n, 1.)
            current_k_used = bool(np.array_equal(outputs['warp'], np.full(24*256, 3.5, np.float32)))
            intact = helpers.unchanged(np, cp, bases, before)
            check.update(n=n, layout='current-k-dominates-nan-tail', current_k_used=current_k_used,
                         inputs_bitwise_unchanged=intact)
            check['accepted'] = check['accepted'] and current_k_used and intact
            result['view_checks'].append(check)

            inputs, bases = helpers.make_inputs(np, cp, rng, n)
            before = helpers.snapshots(np, cp, bases)
            precheck, _ = exact_check(np, cp, operators, inputs, n, .0625)
            calls, kernel_outputs, scratch = kernel_calls(inputs, n)
            for _ in range(args.warmups):
                for key in operators:
                    operators[key](*inputs, n, .0625)
                    calls[key]()  # Compile/warm raw kernels before capture.
            stream.synchronize()
            kernel_equal = bool(np.array_equal(cp.asnumpy(kernel_outputs['serial']).view(np.uint32),
                                               cp.asnumpy(kernel_outputs['warp']).view(np.uint32)))
            accepted = bool(remainder_unchanged and precheck['accepted'] and kernel_equal
                            and all(row['accepted'] for row in result['view_checks']))
            samples = {mode: {key: [] for key in operators} for mode in ('kernel', 'api')}
            wall_samples = {mode: {key: [] for key in operators} for mode in samples}
            orders, graphs = [], {}
            if accepted:
                for key, call in calls.items():
                    stream.begin_capture()
                    for _ in range(args.repetitions):
                        call()
                    graphs[key] = stream.end_capture()
                for trial in range(args.trials):
                    order = ('serial', 'warp') if trial % 2 == 0 else ('warp', 'serial')
                    modes = ('kernel', 'api') if trial % 2 == 0 else ('api', 'kernel')
                    orders.append(dict(operators=list(order), modes=list(modes)))
                    for mode in modes:
                        for key in order:
                            start, end = cp.cuda.Event(), cp.cuda.Event()
                            wall_start = time.perf_counter()
                            start.record(stream)
                            if mode == 'kernel':
                                graphs[key].launch(stream)
                            else:
                                for _ in range(args.repetitions):
                                    output = operators[key](*inputs, n, .0625)
                            end.record(stream)
                            end.synchronize()
                            wall_samples[mode][key].append((time.perf_counter() - wall_start)*1000 / args.repetitions)
                            samples[mode][key].append(float(cp.cuda.get_elapsed_time(start, end) / args.repetitions))
            postcheck, _ = exact_check(np, cp, operators, inputs, n, .0625)
            kernel_postcheck = bool(np.array_equal(cp.asnumpy(kernel_outputs['serial']).view(np.uint32),
                                                   cp.asnumpy(kernel_outputs['warp']).view(np.uint32)))
            intact = helpers.unchanged(np, cp, bases, before)
            stream_preserved = cp.cuda.get_current_stream().ptr == stream.ptr
            accepted = bool(accepted and postcheck['accepted'] and kernel_postcheck and intact and stream_preserved)
            medians = {mode: {key: float(np.median(values)) if values else None
                              for key, values in rows.items()} for mode, rows in samples.items()}
            ratios = {mode: rows['serial'] / rows['warp'] if accepted and rows['warp'] else None
                      for mode, rows in medians.items()}
            result['cases'].append(dict(n=n, accepted=accepted, precheck=precheck, postcheck=postcheck,
                                        kernel_precheck_bitwise_equal=kernel_equal,
                                        kernel_postcheck_bitwise_equal=kernel_postcheck,
                                        inputs_bitwise_unchanged=intact, current_stream_preserved=stream_preserved,
                                        scratch_bytes=int(scratch.nbytes), output_bytes=int(kernel_outputs['warp'].nbytes),
                                        graph_kernel_nodes={key: args.repetitions * count
                                                            for key, count in [('serial', 1), ('warp', 2)]},
                                        trial_orders=orders, samples_ms=samples, wall_samples_ms=wall_samples,
                                        median_ms=medians, measured_serial_over_warp=ratios))
            # Retire graphs before releasing the buffers to which their nodes point.
            del graphs, calls, kernel_outputs, scratch
    result['accepted'] = bool(remainder_unchanged and all(row['accepted'] for row in result['view_checks'])
                              and all(row['accepted'] for row in result['cases']))
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print_summary(result, args.output)
    if not result['accepted']:
        raise RuntimeError('Strict operator parity/input-preservation failed; timings are not acceptance')


if __name__ == '__main__':
    main()
