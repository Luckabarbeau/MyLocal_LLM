#!/usr/bin/env python3
"""Plain dense CuPy/HIP GEMM: FP32/FP16/BF16 inputs, FP32 compute/output.

No model, autograd, Triton, transfers, allocation, or input conversion in timing.
FMA counts as two FLOPs; dense FLOPs = 2*N**3. Not an instruction-level peak.
"""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or min(args.warmup, args.repeats, args.iterations) < 1:
        parser.error('sizes and iteration counts must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('This benchmark is restricted to AMD HIP')
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    info = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in props.items()
            if k in ('name', 'gcnArchName', 'multiProcessorCount', 'clockRate',
                     'memoryClockRate', 'totalGlobalMem', 'memoryBusWidth')}
    print('Device:', json.dumps(info), flush=True)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    old_pointer_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    alpha, beta = np.array(1., np.float32), np.array(0., np.float32)
    rows = []
    start_time = time.time()
    try:
        cp.random.seed(113)
        for n in args.sizes:
            for dtype, datatype in [('float32', cp.cuda.runtime.CUDA_R_32F),
                                    ('float16', cp.cuda.runtime.CUDA_R_16F),
                                    ('bfloat16', cp.cuda.runtime.CUDA_R_16BF)]:
                # BF16/f16 input rounding happens before, never inside timing.
                a = (cp.random.standard_normal((n, n), dtype=cp.float32)*.1).astype(dtype)
                b = (cp.random.standard_normal((n, n), dtype=cp.float32)*.1).astype(dtype)
                c = cp.empty((n, n), dtype=cp.float32)
                def gemm():
                    # Row-major C = A @ B is column-major C.T = B.T @ A.T.
                    # Legacy HIP gemmEx uses CUDA_R_32F (0) for FP32 compute.
                    cublas.gemmEx(
                        handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
                        n, n, n, alpha.ctypes.data,
                        b.data.ptr, datatype, n, a.data.ptr, datatype, n,
                        beta.ctypes.data, c.data.ptr, cp.cuda.runtime.CUDA_R_32F, n,
                        cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT)
                for _ in range(args.warmup):
                    gemm()
                cp.cuda.get_current_stream().synchronize()
                # Sample an 8x8 output grid against float64 CPU dot products
                # of the actual quantized inputs; validation is outside timing.
                indices = np.unique(np.linspace(0, n-1, min(n,8), dtype=np.int64))
                ai = cp.asnumpy(a[indices].astype(cp.float32)).astype(np.float64)
                bi = cp.asnumpy(b[:, indices].astype(cp.float32)).astype(np.float64)
                reference = ai @ bi
                got = cp.asnumpy(c[indices[:, None],indices[None,:]])
                abs_error = float(np.max(np.abs(got-reference)))
                rel_error = float(np.linalg.norm(got-reference)/max(np.linalg.norm(reference),1e-30))
                np.testing.assert_allclose(got, reference, rtol=2e-4, atol=2e-5)
                times = []
                for _ in range(args.repeats):
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    for _ in range(args.iterations):
                        gemm()
                    end.record()
                    end.synchronize()
                    times.append(float(cp.cuda.get_elapsed_time(start, end))/args.iterations)
                ms = median(times)
                row = dict(n=n,input_dtype=dtype,accumulation_dtype='float32',output_dtype='float32',
                           layout='row-major NN',median_ms=ms,min_ms=min(times),samples_ms=times,
                           dense_tflops=2*n**3/(ms*1e9),
                           best_sample_tflops=2*n**3/(min(times)*1e9),
                           sample_max_abs_error=abs_error,sample_relative_l2=rel_error,
                           sample_validation='passed')
                rows.append(row)
                print(f'{dtype:9s} N={n:5d}: {ms:9.4f} ms, {row["dense_tflops"]:8.2f} dense TFLOP/s; sample check passed',flush=True)
                del a,b,c,ai,bi,reference,got
                cp.get_default_memory_pool().free_all_blocks()
    finally:
        cublas.setPointerMode(handle,old_pointer_mode)
    report = dict(device=info,cupy_version=cp.__version__,hip_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                  operation='C=A@B, alpha=1 beta=0; FP32 compute/output',
                  flop_convention='2*N^3 dense FLOPs; one FMA=2; no sparsity multiplier',
                  timing='GPU events around repeated GEMMs; median per-call time across samples',
                  warmup=args.warmup,repeats=args.repeats,iterations=args.iterations,
                  elapsed_seconds=time.time()-start_time,results=rows)
    with args.output.open('x') as out:
        json.dump(report,out,indent=2)
    print('Saved',args.output,flush=True)


if __name__ == '__main__':
    main()
