"""CPU-only proxy benchmark for the host side of BF16 grouped cuBLAS dispatch.

CUDA/cuBLAS are mocked; this measures Python descriptor validation, NumPy host
packing, and the number of device-array submissions. Descriptor *creation* is
benchmarked separately rather than being included in the dispatch timing. It does NOT estimate GPU
execution time, PCIe transfer latency, or end-to-end training throughput.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import statistics
import sys
import time
import types
from unittest.mock import patch

import numpy as np

from mini_llm.ops import cublas_grouped as bridge


class _DeviceArray:
    def __init__(self, values):
        self.array = np.array(values, copy=True)
        self.data = types.SimpleNamespace(ptr=self.array.ctypes.data)


class _MockXp:
    def __init__(self):
        self.submissions = 0
        self.cuda = types.SimpleNamespace(get_current_stream=lambda: types.SimpleNamespace(ptr=123))

    def asarray(self, values):
        self.submissions += 1
        return _DeviceArray(values)


@contextlib.contextmanager
def mock_cuda_dispatch(*, capture=False):
    xp = _MockXp()
    mock_cublas = types.ModuleType('cupy_backends.cuda.libs.cublas')
    mock_cublas.CUBLAS_COMPUTE_32F = 68
    mock_cublas.setStream = lambda *args: None
    mock_cublas.CUBLAS_GEMM_DEFAULT = -1
    mock_device = types.ModuleType('cupy.cuda.device')
    mock_device.get_cublas_handle = lambda: 987
    modules = {
        'cupy': types.ModuleType('cupy'),
        'cupy.cuda': types.ModuleType('cupy.cuda'),
        'cupy.cuda.device': mock_device,
        'cupy_backends': types.ModuleType('cupy_backends'),
        'cupy_backends.cuda': types.ModuleType('cupy_backends.cuda'),
        'cupy_backends.cuda.libs': types.ModuleType('cupy_backends.cuda.libs'),
        'cupy_backends.cuda.libs.cublas': mock_cublas,
        'cupy_backends.cuda.api': types.ModuleType('cupy_backends.cuda.api'),
    }
    modules['cupy'].cuda = modules['cupy.cuda']
    modules['cupy.cuda'].device = mock_device
    # Store the arrays to verify correctness inside a fake API call.
    calls = []
    def submit(*args):
        if not capture:
            return 0
        n = int(args[17])
        sizes = np.ctypeslib.as_array(args[18], shape=(n,))
        count = int(sizes.sum())
        arrays = []
        for index in (7, 10, 14):
            ptr = int(args[index].value)
            raw = (ctypes.c_uint64 * count).from_address(ptr)
            arrays.append(tuple(raw))
        calls.append(tuple(arrays))
        return 0
    with patch.dict(sys.modules, modules), \
         patch.object(bridge, 'xp', xp), \
         patch.object(bridge, '_enabled', lambda: True), \
         patch.object(bridge, '_load_grouped_symbol', lambda: submit), \
         patch.object(bridge, 'local_cublas_mode', lambda: 'grouped'):
        yield xp, calls


def sample_groups(*, chunks=8, batch=3, heads=4):
    groups = []
    for chunk in range(chunks):
        problems = []
        for b in range(batch):
            for h in range(heads):
                i = chunk * batch * heads + b * heads + h
                problems.append(bridge.RowMajorGemmProblem(
                    a_ptr=0x1000 + i * 512,
                    b_ptr=0x200000 + i * 768,
                    c_ptr=0x400000 + i * 1024,
                    m=128, n=384, k=64, lda=256, ldb=256, ldc=384,
                    trans_a=False, trans_b=True,
                ))
        groups.append(bridge.RowMajorGemmGroup(tuple(problems)))
    return tuple(groups)


def run_benchmark(repeats=5, iterations=800, chunks=8):
    groups = sample_groups(chunks=chunks)
    times = []
    with mock_cuda_dispatch(capture=False) as (xp, calls):
        # Avoid retaining repeated fake call payloads in the timed loop.
        fn = bridge.grouped_bf16_gemm
        for _ in range(100):
            assert fn(groups)
        calls.clear()
        for _ in range(repeats):
            begin = time.perf_counter()
            for _ in range(iterations):
                assert fn(groups)
            times.append((time.perf_counter() - begin) * 1000 / iterations)
            calls.clear()
        per_call = xp.submissions / (100 + repeats * iterations)
    return statistics.median(times), per_call, times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations', type=int, default=800)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--chunks', type=int, default=8)
    args = p.parse_args()
    print(f'Grouped cuBLAS host proxy: chunks={args.chunks}, batch=3, heads=4')
    build_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            sample_groups(chunks=args.chunks)
        build_times.append((time.perf_counter() - t0) * 1000 / args.iterations)
    print(f'Python descriptor construction only: {statistics.median(build_times):.4f} ms/build')
    with patch.dict(__import__('os').environ, {'MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD': '0'}):
        baseline, baseline_submits, t0 = run_benchmark(args.repeats, args.iterations, args.chunks)
    with patch.dict(__import__('os').environ, {'MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD': '1'}):
        packed, packed_submits, t1 = run_benchmark(args.repeats, args.iterations, args.chunks)
    print(f'Original  CPU-only dispatch: {baseline:.4f} ms/call; submissions={baseline_submits:.1f}')
    print(f'Packed    CPU-only dispatch: {packed:.4f} ms/call; submissions={packed_submits:.1f}')
    print(f'Host-only relative time: {(packed/baseline-1)*100:+.1f}%')
    print('Original repeats (ms):', ', '.join(f'{x:.4f}' for x in t0))
    print('Packed repeats (ms):', ', '.join(f'{x:.4f}' for x in t1))
    print('NOTE: No GPU was used. This is NOT a model training speed measurement.')


if __name__ == '__main__':
    main()
