"""0106Y bounded grouped-expert submission vs warmed kernel-graph benchmark.

Synthetic packed weights at configurable dimensions, not model throughput.
The existing pytest packed-byte fixtures keep mixed quantization well conditioned.
Graph replay excludes pointer upload, route permutation, validation and allocations;
its event interval still includes replay scheduling/host gaps, not pure kernel time.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d-model', type=int, default=2048)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--experts', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--formats', nargs=3, default=['Q4_K', 'Q4_K', 'Q6_K'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 1 <= args.experts <= 32 or not 1 <= args.iterations <= 200
            or not 256 <= args.d_model <= 4096 or not 256 <= args.d_ff <= 2048
            or args.d_model % 256 or args.d_ff % 256):
        parser.error('Requires 1..32 experts, 1..200 iterations and bounded 256-aligned widths')
    if args.output.exists():
        parser.error('Choose a new output path; existing measurements are preserved')
    from mini_llm.backend import BACKEND_NAME, xp as cp, asnumpy
    from mini_llm.ops import grouped_quantized_experts as grouped
    from test_0106t_grouped_quantized_experts import _hosts, _upload
    if BACKEND_NAME != 'cupy':
        parser.error('Requires MINI_LLM_BACKEND=cupy')
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS
    if any(fmt not in BLOCK_FORMATS for fmt in args.formats):
        parser.error('Unsupported quantization format')
    host_bytes = args.experts * sum(
        rows * cols // BLOCK_FORMATS[fmt][0] * BLOCK_FORMATS[fmt][1]
        for fmt, rows, cols in zip(args.formats, (args.d_ff, args.d_ff, args.d_model),
                                  (args.d_model, args.d_model, args.d_ff)))
    free, _ = cp.cuda.runtime.memGetInfo()
    if host_bytes + 2 * 1024**3 > free:
        parser.error('Packed weights would violate the 2 GiB free-VRAM reserve')
    stream = cp.cuda.Stream(non_blocking=True)
    report = dict(patch='0106Y', scope='Synthetic grouped-expert microbenchmark, not tokens/s',
        configuration=vars(args) | {'output': str(args.output)}, packed_weight_bytes=host_bytes,
        iterations=args.iterations, modes={}, passed=False,
        timing_semantics={
            'enqueue_ms': 'Host submission loop / iterations, no per-call sync',
            'completion_ms': 'Wall time through completion / iterations',
            'event_ms': 'CUDA events around full loop, includes host submission gaps',
            'kernel_graph_event_ms': 'Replay of exact warmed 3-kernel sequence, no metadata upload or allocation; includes replay scheduling'})
    with stream:
        handles = _upload(_hosts(tuple(args.formats), d=args.d_model, f=args.d_ff, count=args.experts))
        x = cp.asarray(np.random.default_rng(10625).normal(0, .01, (1, args.d_model)).astype(np.float32))
        routes = cp.asarray(np.linspace(.01, .1, args.experts, dtype=np.float32))
        slots = tuple(reversed(range(args.experts)))
        def materialized():
            ordered = cp.concatenate(tuple(routes[s:s + 1] for s in slots))
            return grouped.grouped_quantized_experts_forward(x, handles, ordered)
        def indexed():
            return grouped.grouped_quantized_experts_forward(x, handles, routes, routing_slots=slots)
        expected = materialized()
        actual = indexed()
        stream.synchronize()
        reference = asnumpy(expected)
        np.testing.assert_array_equal(asnumpy(actual), reference)
        report['max_abs_error'] = float(np.max(np.abs(asnumpy(actual) - reference)))
        for name, fn in [('materialized', materialized), ('indexed', indexed)]:
            for _ in range(3):
                warm = fn()
            stream.synchronize()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            begin = time.perf_counter()
            for _ in range(args.iterations):
                result = fn()
            enqueue = time.perf_counter() - begin
            end.record(stream)
            end.synchronize()
            completion = time.perf_counter() - begin
            np.testing.assert_array_equal(asnumpy(result), reference)
            row = dict(enqueue_ms=enqueue * 1000 / args.iterations,
                completion_ms=completion * 1000 / args.iterations,
                event_ms=float(cp.cuda.get_elapsed_time(start, end)) / args.iterations)
            # Retain exact pointer arguments, including their allocations, for
            # capture/replay. No weight arithmetic or launch geometry changes.
            launches = []
            launch = grouped._launch
            def record(kernel, grid, *values, **options):
                launches.append((kernel, grid, values, options))
                return launch(kernel, grid, *values, **options)
            grouped._launch = record
            try:
                captured_output = fn()
            finally:
                grouped._launch = launch
            assert len(launches) == 3
            stream.synchronize()
            stream.begin_capture()
            try:
                for kernel, grid, values, options in launches:
                    launch(kernel, grid, *values, **options)
                graph = stream.end_capture()
            except Exception:
                if stream.is_capturing():
                    stream.end_capture()
                raise
            graph.upload(stream=stream)
            graph.launch(stream=stream)
            stream.synchronize()
            start.record(stream)
            for _ in range(args.iterations):
                graph.launch(stream=stream)
            end.record(stream)
            end.synchronize()
            np.testing.assert_array_equal(asnumpy(captured_output), reference)
            row['kernel_graph_event_ms'] = float(cp.cuda.get_elapsed_time(start, end)) / args.iterations
            report['modes'][name] = row
            print(json.dumps(dict(mode=name, **row)))
            # Complete replay before releasing graph-owned pointer arguments.
            del graph, launches
        report['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        report['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    main()
