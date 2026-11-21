"""Opt-in, same-input packed Q8 projection pair; no model dispatch changes.

Weights remain packed and read-only. Only activation quantization and launch
submission are shared: every output row uses mixed_q8_gpu's unchanged row_dot.
"""
from functools import lru_cache
import operator
import os

import numpy as np

from . import mixed_q8_gpu as mixed

q8 = mixed.q8
iq = mixed.iq
cp = mixed.cp

_SOURCE = r'''
template<int F0,int F1> __global__ void q8_linear_pair(
        const unsigned char* w0,const unsigned char* w1,const unsigned char* q,
        float* out0,float* out1,int n0,int n1,int k,int rb0,int rb1,int blocks0) {
#ifdef SHARED_IQ_GRID
    constexpr int size0=F0==0?8192:F0==1?2048:F0==2?1024:1;
    constexpr int size1=F1==0?8192:F1==1?2048:F1==2?1024:1;
    __shared__ __align__(16) unsigned char grid[size0>size1?size0:size1];
#else
    unsigned char* grid=nullptr;
#endif
    int lane=threadIdx.x%32;
    const unsigned char* x=q+(long long)blockIdx.y*(k/32)*36;
    // The branch is CTA-uniform. Each CTA initializes only its own table,
    // before any tail warp returns; different CTAs never share this storage.
    if(blockIdx.x<(unsigned int)blocks0) {
        init_grid<F0>(grid);
        int row=blockIdx.x*4+threadIdx.x/32;
        if(row>=n0) return;
        float sum=row_dot<F0>(w0+(long long)row*rb0,x,k,lane,grid);
        if(!lane) out0[(long long)blockIdx.y*n0+row]=sum;
    } else {
        init_grid<F1>(grid);
        int row=(blockIdx.x-blocks0)*4+threadIdx.x/32;
        if(row>=n1) return;
        float sum=row_dot<F1>(w1+(long long)row*rb1,x,k,lane,grid);
        if(!lane) out1[(long long)blockIdx.y*n1+row]=sum;
    }
}
'''


@lru_cache(maxsize=128)
def _module(device, formats, shared_grid):
    f0, f1 = (mixed.FORMAT_ID[fmt] for fmt in formats)
    name = f'q8_linear_pair<{f0},{f1}>'
    with cp.cuda.Device(device):
        prefix = '#define SHARED_IQ_GRID\n' if shared_grid else ''
        base = iq._cuda_source('global').replace('INPUT_T', 'float')
        if shared_grid:
            base = base.replace('__device__ const', '__device__ __align__(16) const')
        return cp.RawModule(code=prefix + base + mixed._SOURCE + _SOURCE,
                            options=('--std=c++17', '--fmad=false'),
                            name_expressions=(name,))


def _validate_weight(x, weight, device):
    """Metadata-only counterpart of mixed.linear_packed's weight checks."""
    if not all(hasattr(weight, field) for field in ('data', 'shape', 'quant_type')):
        raise TypeError('Expected packed device weight with data/shape/quant_type')
    if weight.quant_type not in mixed.FORMATS:
        raise NotImplementedError(f'No Q8 GPU dot for {weight.quant_type}')
    shape = tuple(operator.index(v) for v in weight.shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('packed weight shape must be positive [N, K]')
    n, k = shape
    if k != x.shape[-1]:
        raise ValueError('Input width mismatch')
    if weight.quant_type in iq._FORMAT_ID:
        iq._validate_weight(x, weight)
    else:
        data = weight.data
        qk, qb = iq.BLOCK_FORMATS[weight.quant_type]
        if (not isinstance(data, cp.ndarray) or data.dtype != cp.uint8 or
                not data.flags.c_contiguous or data.device.id != device):
            raise ValueError('Expected contiguous packed bytes on the current device')
        if (max(n, k) > 2147483647 or k % qk or
                data.size != n * (k // qk) * qb):
            raise ValueError('Invalid packed shape/alignment/byte length')
    return n, weight.data.nbytes // n


def _overlaps(a, b):
    return (a.nbytes and b.nbytes and
            a.data.ptr < b.data.ptr + b.nbytes and
            b.data.ptr < a.data.ptr + a.nbytes)


def _output(out, shape, x):
    if out is None:
        return cp.empty(shape, dtype=np.float32)
    if not isinstance(out, cp.ndarray) or out.dtype != np.float32:
        raise TypeError('out must be a CuPy float32 array')
    if out.shape != shape:
        raise ValueError(f'out must have exact shape {shape}')
    if not out.flags.c_contiguous:
        raise ValueError('out must be contiguous')
    # Flatten only a verified contiguous view; the codec checks device,
    # alignment and input overlap without launching a kernel or allocating.
    q8._output(out.reshape(-1, shape[-1]), (x.shape[0], shape[-1]), np.float32, x)
    return out


def linear_pair(x, w0, w1, *, out=None):
    """Return two FP32 projections of contiguous FP32 [M,K] or [B,T,K].

    Both weights follow mixed.linear_packed's supported packed-device layout.
    They are never modified. ``out`` may be a two-tuple of output arrays (or
    None entries for allocation), each with shape x.shape[:-1] + (N_i,).
    Outputs must be disjoint from x, both weight buffers, and each other.

    Nonempty execution enqueues one existing native Q8_1 quantizer and one
    paired projection kernel, asynchronously on the current CUDA stream. The
    caller owns stream ordering and must not modify weights during execution.
    Shared-grid selection follows MINI_LLM_Q8_SHARED_GRID. No address-based
    activation cache is retained; graph replays quantize their current input.
    """
    if cp is None or not isinstance(x, cp.ndarray) or x.dtype != np.float32:
        raise TypeError('x must be a CuPy float32 array')
    if x.ndim not in (2, 3):
        raise ValueError('x must be [M,K] or [B,T,K]')
    if not x.flags.c_contiguous:
        raise ValueError('x must be contiguous')
    k = x.shape[-1]
    if k == 0 or k % 32 or x.size > 2147483647:
        raise ValueError('K must be positive, divisible by 32; total size <= INT32_MAX')
    flat = x.reshape(-1, k)
    device = q8._validate(flat, np.float32, 'x')
    m = flat.shape[0]
    if m > 65535:
        raise ValueError('Too many Q8 rows')
    n0, rb0 = _validate_weight(flat, w0, device)
    n1, rb1 = _validate_weight(flat, w1, device)
    if out is None:
        out = (None, None)
    if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError('out must be a two-tuple')
    results = tuple(_output(value, x.shape[:-1] + (n,), flat)
                    for value, n in zip(out, (n0, n1)))
    for result in results:
        if any(_overlaps(result, weight.data) for weight in (w0, w1)):
            raise ValueError('out must not overlap weights')
    if _overlaps(*results):
        raise ValueError('outputs must not overlap each other')
    if m:
        formats = (w0.quant_type, w1.quant_type)
        f0, f1 = (mixed.FORMAT_ID[fmt] for fmt in formats)
        shared_grid = os.environ.get('MINI_LLM_Q8_SHARED_GRID', '0') == '1'
        kernel = _module(device, formats, shared_grid).get_function(f'q8_linear_pair<{f0},{f1}>')
        packed = q8.quantize_q8_1(flat)
        blocks0 = (n0 + 3) // 4
        kernel((blocks0 + (n1 + 3) // 4, m), (128,),
               (w0.data, w1.data, packed, *results,
                *(np.int32(v) for v in (n0, n1, k, rb0, rb1, blocks0))))
    return results
