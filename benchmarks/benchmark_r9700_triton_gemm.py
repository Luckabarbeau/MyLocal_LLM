#!/usr/bin/env python3
"""Standalone HIP Triton vs CuPy dense GEMM, with genuine IEEE FP32 inputs.

All cases compute row-major C=A@B with FP32 accumulation/output. No sparsity,
model, allocator work, conversions, or host transfers inside timed regions.
"""
import argparse
import json
from pathlib import Path
from statistics import median

import cupy as cp
import numpy as np
import triton
import triton.language as tl
from cupy_backends.cuda.libs import cublas


class Pointer:
    def __init__(self, array):
        self.array = array
        self.dtype = {'float32': tl.float32, 'float16': tl.float16,
                      'bfloat16': tl.bfloat16}[str(array.dtype)]

    def data_ptr(self):
        return int(self.array.data.ptr)


@triton.jit
def matmul(A, B, C, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(N, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    group = pid // per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % per_group) % group_m
    pn = (pid % per_group) // group_m
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(N, BK)):
        kk = start * BK + k
        a = tl.load(A + m[:, None] * N + kk[None, :],
                    mask=(m[:, None] < N) & (kk[None, :] < N), other=0.)
        b = tl.load(B + kk[:, None] * N + n[None, :],
                    mask=(kk[:, None] < N) & (n[None, :] < N), other=0.)
        # On FP32 this must not become a reduced-input-precision matrix dot.
        # FP16/BF16 inputs use native matrix instructions and FP32 accumulators.
        acc = tl.dot(a, b, acc, input_precision='ieee')
    tl.store(C + m[:, None] * N + n[None, :], acc,
             mask=(m[:, None] < N) & (n[None, :] < N))


def measure(fn, warmup, repeats, iterations):
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    times = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end))/iterations)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes + [args.warmup, args.repeats, args.iterations]) < 1:
        parser.error('sizes and iteration counts must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('AMD HIP required')
    device = cp.cuda.runtime.getDevice()
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip':
        raise RuntimeError(f'Expected HIP Triton target, got {target}')
    stream = int(cp.cuda.get_current_stream().ptr)
    if int(triton.runtime.driver.active.get_current_stream(device)) != stream:
        raise RuntimeError('CuPy and Triton streams must match for event timing')
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, stream)
    old_mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    alpha, beta = np.array(1., np.float32), np.array(0., np.float32)
    rows = []
    cp.random.seed(113)
    try:
        # Full small reference includes partial tiles; large cases check 8x8
        # selected results using float64 CPU accumulation of quantized inputs.
        for n in [129, *args.sizes]:
            for dtype, datatype in [('float32', cp.cuda.runtime.CUDA_R_32F),
                                    ('float16', cp.cuda.runtime.CUDA_R_16F),
                                    ('bfloat16', cp.cuda.runtime.CUDA_R_16BF)]:
                a = (cp.random.standard_normal((n,n), dtype=cp.float32)*.1).astype(dtype)
                b = (cp.random.standard_normal((n,n), dtype=cp.float32)*.1).astype(dtype)
                c = cp.empty((n,n), dtype=cp.float32)
                ap, bp, cpp = Pointer(a), Pointer(b), Pointer(c)
                indices = np.arange(n) if n == 129 else np.unique(np.linspace(0,n-1,8,dtype=np.int64))
                reference = (cp.asnumpy(a[indices].astype(cp.float32)).astype(np.float64)
                             @ cp.asnumpy(b[:,indices].astype(cp.float32)).astype(np.float64))
                def validate():
                    cp.cuda.get_current_stream().synchronize()
                    got = cp.asnumpy(c[indices[:,None],indices[None,:]])
                    np.testing.assert_allclose(got, reference, rtol=2e-4, atol=2e-5)
                    return dict(max_abs_error=float(np.max(np.abs(got-reference))),
                                relative_l2=float(np.linalg.norm(got-reference)/max(np.linalg.norm(reference),1e-30)))
                def blas():
                    cublas.gemmEx(handle,cublas.CUBLAS_OP_N,cublas.CUBLAS_OP_N,
                                  n,n,n,alpha.ctypes.data,b.data.ptr,datatype,n,
                                  a.data.ptr,datatype,n,beta.ctypes.data,c.data.ptr,
                                  cp.cuda.runtime.CUDA_R_32F,n,cp.cuda.runtime.CUDA_R_32F,
                                  cublas.CUBLAS_GEMM_DEFAULT)
                blas()
                errors = validate()
                if n != 129:
                    times = measure(blas,args.warmup,args.repeats,args.iterations)
                    ms = median(times)
                    row = dict(backend='cupy-gemmEx',n=n,input_dtype=dtype,median_ms=ms,
                               dense_tflops=2*n**3/(ms*1e9),samples_ms=times,**errors)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                configs = ([(32,64,32,4),(64,64,32,4),(64,64,64,4)] if dtype == 'float32'
                           else [(64,64,64,4),(128,128,64,4),(128,256,64,8)])
                for bm,bn,bk,warps in configs:
                    def launch():
                        return matmul[(triton.cdiv(n,bm)*triton.cdiv(n,bn),)](
                            ap,bp,cpp,N=n,BM=bm,BN=bn,BK=bk,GROUP_M=8,
                            num_warps=warps,num_stages=2)
                    kernel = launch()
                    errors = validate()
                    if n == 129:
                        continue
                    times = measure(launch,args.warmup,args.repeats,args.iterations)
                    ms = median(times)
                    row = dict(backend='triton',n=n,input_dtype=dtype,median_ms=ms,
                               dense_tflops=2*n**3/(ms*1e9),samples_ms=times,
                               tile=[bm,bn,bk],num_warps=warps,num_stages=2,
                               n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
                               shared_bytes=kernel.metadata.shared,**errors)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                del a,b,c,ap,bp,cpp,reference
                cp.get_default_memory_pool().free_all_blocks()
    finally:
        cublas.setPointerMode(handle,old_mode)
    report = dict(target=str(target),cupy_version=cp.__version__,triton_version=triton.__version__,
                  fp32_precision='ieee; no TF32 or BF16 input truncation',
                  compute_and_output='float32',flop_convention='2*N^3 dense FLOPs',
                  warmup=args.warmup,repeats=args.repeats,iterations=args.iterations,
                  validation='full 129x129 with tails plus sampled 8x8 large outputs vs float64 CPU',
                  all_checks_passed=True,results=rows)
    with args.output.open('x') as out:
        json.dump(report,out,indent=2)
    print('PASS; saved',args.output,flush=True)


if __name__ == '__main__':
    main()
