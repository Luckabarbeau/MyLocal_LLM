"""0080CB: CPU-only attention GEMM descriptor profiling; GEMM is a no-op.

This measures descriptor construction separately from the grouped cuBLAS
host submission mock. It does NOT measure RTX 5080 training performance.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import statistics
import time
from unittest.mock import patch

from benchmarks.benchmark_grouped_dispatch_host_0080ca import mock_cuda_dispatch, sample_groups
from mini_llm.ops import cublas_grouped as bridge


def _time(fn, iterations: int, repeats: int):
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter() - begin) * 1000 / iterations)
    return statistics.median(samples)


def benchmark_descriptors(*, chunks=8, batch=3, heads=4, iterations=1000, repeats=5):
    result = {}
    with mock_cuda_dispatch(capture=False), patch.dict(os.environ, {
        'MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD': '1',
    }):
        for name, klass in (
            ('frozen_dataclass', bridge.RowMajorGemmProblem),
            ('compact_namedtuple', bridge.CompactRowMajorGemmProblem),
        ):
            with patch.object(bridge, 'RowMajorGemmProblem', klass):
                def build():
                    return sample_groups(chunks=chunks, batch=batch, heads=heads)
                groups = build()
                def dispatch_only():
                    if not bridge.grouped_bf16_gemm(groups):
                        raise RuntimeError('Mock dispatch failed')
                def build_and_dispatch():
                    if not bridge.grouped_bf16_gemm(build()):
                        raise RuntimeError('Mock dispatch failed')
                for _ in range(20):
                    build_and_dispatch()
                result[name] = {
                    'create_ms': _time(build, iterations, repeats),
                    'dispatch_ms': _time(dispatch_only, iterations, repeats),
                    'total_ms': _time(build_and_dispatch, iterations, repeats),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=int, default=8)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--profile', action='store_true')
    a = parser.parse_args()
    params = dict(chunks=a.chunks, batch=a.batch, heads=a.heads,
                  iterations=a.iterations, repeats=a.repeats)
    results = benchmark_descriptors(**params)
    print('CPU-only grouped-GEMM host microbenchmark: all GPU work mocked')
    print(f"problems/dispatch={a.chunks*a.batch*a.heads}, chunks={a.chunks}")
    for name, item in results.items():
        print(f"{name:19s} build={item['create_ms']:.4f} ms "
              f"dispatch={item['dispatch_ms']:.4f} ms total={item['total_ms']:.4f} ms")
    old = results['frozen_dataclass']['total_ms']
    new = results['compact_namedtuple']['total_ms']
    print(f'CPU-only combined reduction: {(1 - new / old)*100:.1f}%')
    print('CAUTION: This does not predict end-to-end GPU training speed.')
    if a.profile:
        prof = cProfile.Profile()
        with patch.object(bridge, 'RowMajorGemmProblem', bridge.RowMajorGemmProblem):
            prof.enable()
            for _ in range(a.iterations):
                sample_groups(chunks=a.chunks, batch=a.batch, heads=a.heads)
            prof.disable()
        out = io.StringIO()
        pstats.Stats(prof, stream=out).strip_dirs().sort_stats('tottime').print_stats(12)
        print(out.getvalue())


if __name__ == '__main__':
    main()
