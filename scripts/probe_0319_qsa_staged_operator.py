#!/usr/bin/env python3
"""Synthetic serial/staged QSA operator A/B, never model latency or throughput.

Requires scripts/run_0290_isolated_benchmark.py before any CUDA import. Event
samples cover repeated allocating operator calls, including host launch gaps;
compilation, setup, copies and exactness checks are outside timed regions.
No speedup is asserted: raw samples and measured ratios are emitted only here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def provenance(serial, staged):
    def git(*args):
        return subprocess.check_output(['git', '--no-pager', '--no-optional-locks', *args],
                                       cwd=ROOT, text=True).strip()

    files = [Path(__file__), Path(serial.__file__), Path(staged.__file__)]
    return dict(git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                python=sys.version, argv=sys.argv, isolation_cgroup=Path('/proc/self/cgroup').read_text(),
                files={str(p.resolve().relative_to(ROOT)): dict(sha256=digest(p.read_text())) for p in files},
                kernels=dict(serial=dict(sha256=digest(serial._SOURCE), source=serial._SOURCE),
                             staged=dict(sha256=digest(staged._SOURCE), source=staged._SOURCE)),
                compiler_options=list(staged._OPTIONS))


def make_inputs(np, cp, rng, n, layout='plain', current_k_case=False):
    bases = []

    def array(shape):
        step = 2 if layout in ('strided', 'reverse') else 1
        backing = cp.asarray(rng.normal(size=tuple(step*s for s in shape)).astype(np.float32))
        bases.append(backing)
        view = backing[tuple(slice(None, None, step) for _ in shape)]
        return view[tuple(slice(None, None, -1) for _ in shape)] if layout == 'reverse' else view

    inputs = [array(s) for s in ((24, 256), (24, 512), (2, 256), (2, 256),
                                  (n+2, 2, 256), (n+2, 2, 256))]
    q, qg, k, v, keys, values = inputs
    qg[:, :256] = cp.nan
    keys[n-1:] = cp.nan
    values[n-1:] = cp.nan
    if layout == 'broadcast':
        for i in (0, 2, 3):
            inputs[i] = cp.broadcast_to(inputs[i][:1], inputs[i].shape)
    if current_k_case:
        q[:] = 0
        q[:, 0] = 1
        qg[:, 256:] = 0
        keys[:n-1] = -100
        values[:n-1] = -17
        k[:] = 100
        v[:] = 7
    return inputs, bases


def snapshots(np, cp, bases):
    return [cp.asnumpy(a).view(np.uint32).copy() for a in bases]


def unchanged(np, cp, bases, before):
    return all(np.array_equal(cp.asnumpy(a).view(np.uint32), b) for a, b in zip(bases, before))


def exact_check(np, cp, operators, inputs, n, scale):
    outputs = {key: cp.asnumpy(op(*inputs, n, scale)) for key, op in operators.items()}
    a, b = outputs['staged'], outputs['serial']
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    equal = bool(np.array_equal(a, b))
    bitwise = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    return dict(finite=finite, array_equal=equal, bitwise_equal=bitwise,
                differing_fp32_bits=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
                max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if finite else None,
                accepted=finite and equal and bitwise), outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; parent must exist')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--trials', type=int, default=7)
    args = parser.parse_args()
    if not (1 <= args.warmups <= 100 and 1 <= args.repetitions <= 1000 and 1 <= args.trials <= 25):
        parser.error('require warmups 1..100, repetitions 1..1000, trials 1..25')
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error('output parent must exist and output must be a new file')
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # No direct-execution fallback; precedes CUDA imports.

    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    import numpy as np
    import cupy as cp
    from mini_llm.ops import qsa_decode_gpu as serial
    from mini_llm.ops import qsa_staged_gpu as staged

    # Direct original-source launch cannot accidentally dispatch an env-enabled
    # alternative operator. Allocation, geometry and arguments match the old API.
    old_kernel = cp.RawKernel(serial._SOURCE, 'qsa_decode', options=staged._OPTIONS)

    def serial_call(q, qg, k, v, keys, values, n, scale):
        output = cp.empty(q.size, dtype=cp.float32)
        strides = tuple(np.int64(s // a.itemsize)
                        for a in (q, qg, k, v, keys, values) for s in a.strides)
        old_kernel((q.shape[0],), (128,),
                   (q, qg, k, v, keys, values, output, np.int32(n), np.int32(256),
                    np.int32(q.shape[0] // k.shape[0]), np.float32(scale), *strides))
        return output

    operators = dict(serial=serial_call, staged=staged.qsa_staged_gpu)
    device = cp.cuda.runtime.getDevice()
    name = cp.cuda.runtime.getDeviceProperties(device)['name']
    if isinstance(name, bytes): name = name.decode()
    generated = staged._SOURCE
    remainder_unchanged = generated.replace(staged._STAGED_SHARED, staged._SHARED, 1).replace(
        staged._STAGED_DOT, staged._SERIAL_DOT, 1) == serial._SOURCE
    result = dict(scope='synthetic standalone operator only; no model, weights, IQ data, or quantization',
                  precision='FP32 serial d=0..255 accumulation; --fmad=false; exact uint32 equality required',
                  provenance=provenance(serial, staged), remainder_byte_identical=remainder_unchanged,
                  device=device, device_name=name, cupy=cp.__version__, numpy=np.__version__,
                  cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
                  heads=24, kv_heads=2, dim=256, scale=.0625, seed=319,
                  warmups=args.warmups, repetitions=args.repetitions, trials=args.trials,
                  timing='CuPy events on nonblocking current stream; alternating A/B trial order; '
                         'allocating API loops include host validation/launch gaps, not kernel-only latency',
                  view_checks=[], cases=[])
    rng = np.random.default_rng(319)
    with cp.cuda.Stream(non_blocking=True) as stream:
        result['stream_ptr'] = int(stream.ptr)
        for n in (141, 435, 1024):
            for layout in ('plain', 'strided', 'reverse', 'broadcast'):
                inputs, bases = make_inputs(np, cp, rng, n, layout)
                before = snapshots(np, cp, bases)
                check, _ = exact_check(np, cp, operators, inputs, n, .0625)
                intact = unchanged(np, cp, bases, before)
                check.update(n=n, layout=layout, inputs_bitwise_unchanged=intact,
                             strides_bytes=[list(a.strides) for a in inputs])
                check['accepted'] = check['accepted'] and intact
                result['view_checks'].append(check)
            inputs, bases = make_inputs(np, cp, rng, n, current_k_case=True)
            before = snapshots(np, cp, bases)
            check, outputs = exact_check(np, cp, operators, inputs, n, 1.)
            current_k_used = bool(np.array_equal(outputs['staged'], np.full(24*256, 3.5, np.float32)))
            intact = unchanged(np, cp, bases, before)
            check.update(n=n, layout='current-k-dominates-nan-cache-tail', current_k_used=current_k_used,
                         inputs_bitwise_unchanged=intact)
            check['accepted'] = check['accepted'] and current_k_used and intact
            result['view_checks'].append(check)

            inputs, bases = make_inputs(np, cp, rng, n)
            before = snapshots(np, cp, bases)
            check, _ = exact_check(np, cp, operators, inputs, n, .0625)
            accepted = check['accepted'] and remainder_unchanged and all(
                row['accepted'] for row in result['view_checks'])
            samples = {key: [] for key in operators}
            orders = []
            if accepted:
                for _ in range(args.warmups):
                    for op in operators.values(): op(*inputs, n, .0625)
                stream.synchronize()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                for trial in range(args.trials):
                    order = ('serial', 'staged') if trial % 2 == 0 else ('staged', 'serial')
                    orders.append(list(order))
                    for key in order:
                        start.record(stream)
                        for _ in range(args.repetitions):
                            output = operators[key](*inputs, n, .0625)
                        end.record(stream)
                        end.synchronize()
                        samples[key].append(float(cp.cuda.get_elapsed_time(start, end) / args.repetitions))
            postcheck, _ = exact_check(np, cp, operators, inputs, n, .0625)
            intact = unchanged(np, cp, bases, before)
            stream_preserved = cp.cuda.get_current_stream().ptr == stream.ptr
            accepted = bool(accepted and postcheck['accepted'] and intact and stream_preserved)
            medians = {key: float(np.median(value)) if value else None for key, value in samples.items()}
            ratio = medians['serial'] / medians['staged'] if accepted and medians['staged'] else None
            result['cases'].append(dict(n=n, accepted=accepted, precheck=check, postcheck=postcheck,
                                        inputs_bitwise_unchanged=intact, current_stream_preserved=stream_preserved,
                                        trial_orders=orders, samples_ms=samples, median_ms=medians,
                                        measured_serial_over_staged=ratio))
    result['accepted'] = bool(remainder_unchanged and all(row['accepted'] for row in result['view_checks'])
                              and all(row['accepted'] for row in result['cases']))
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result['accepted']:
        raise RuntimeError('Strict operator parity failed; JSON timings are not acceptance')


if __name__ == '__main__':
    main()
