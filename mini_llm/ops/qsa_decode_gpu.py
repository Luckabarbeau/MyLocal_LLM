"""FP32 all-causal-cell QSA decode, without weights or cache mutation.

The caller proves contiguous sequential text and n <= selection width, and
retains projection/array validation. One CTA per query head uses fixed shared
storage and the single output allocation as its running numerator. No selected
KV copies, host transfers, default-stream override, or context-sized scratch.
"""

import os
import numpy as np

from ..backend import xp


_KERNELS = {}
_SOURCE = r'''
#define INFINITY (__int_as_float(0x7f800000))
extern "C" __global__ void qsa_decode(
    const float* q, const float* qg, const float* k, const float* v,
    const float* keys, const float* values, float* out,
    int n, int dim, int groups, float scale,
    long long qs0, long long qs1, long long gs0, long long gs1,
    long long k0, long long k1, long long v0, long long v1,
    long long ks0, long long ks1, long long ks2,
    long long vs0, long long vs1, long long vs2) {
    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;
    __shared__ float reduction[128], weights[128];
    __shared__ float maximum, denominator, alpha;
    if (lane == 0) { maximum = -INFINITY; denominator = 0.0f; }
    for (int d = lane; d < dim; d += 128) out[h * dim + d] = 0.0f;
    __syncthreads();
    for (int base = 0; base < n; base += 128) {
        const int cell = base + lane, count = min(128, n - base);
        float logit = -INFINITY;
        if (cell < n) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float key = cell == n - 1 ? k[kh * k0 + d * k1]
                    : keys[cell * ks0 + kh * ks1 + d * ks2];
                dot += q[h * qs0 + d * qs1] * key;
            }
            logit = dot * scale;
        }
        reduction[lane] = logit;
        __syncthreads();
        for (int s = 64; s; s >>= 1) {
            if (lane < s) reduction[lane] = fmaxf(reduction[lane], reduction[lane + s]);
            __syncthreads();
        }
        if (lane == 0) {
            float next = fmaxf(maximum, reduction[0]);
            alpha = expf(maximum - next);
            maximum = next;
        }
        __syncthreads();
        float weight = cell < n ? expf(logit - maximum) : 0.0f;
        weights[lane] = weight;
        reduction[lane] = weight;
        __syncthreads();
        for (int s = 64; s; s >>= 1) {
            if (lane < s) reduction[lane] += reduction[lane + s];
            __syncthreads();
        }
        if (lane == 0) denominator = denominator * alpha + reduction[0];
        for (int d = lane; d < dim; d += 128) {
            float sum = 0.0f;
            for (int j = 0; j < count; ++j) {
                int token = base + j;
                float value = token == n - 1 ? v[kh * v0 + d * v1]
                    : values[token * vs0 + kh * vs1 + d * vs2];
                sum += weights[j] * value;
            }
            out[h * dim + d] = out[h * dim + d] * alpha + sum;
        }
        // All readers must finish before the next tile overwrites shared data.
        __syncthreads();
    }
    for (int d = lane; d < dim; d += 128) {
        float gate = qg[h * gs0 + (dim + d) * gs1];
        float z = expf(-fabsf(gate));
        float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
        out[h * dim + d] = (out[h * dim + d] / denominator) * sigmoid;
    }
}
'''


def qsa_decode_gpu(q, qg, k, v, keys, values, n, scale):
    """Return flattened gated context on the current CuPy stream.

    Inputs are validated FP32 arrays from Qwen4ExpAttention.step. Strides are
    honored, including callback views and caller-owned strided cache storage.
    Only keys/values[:n-1] are read; the current k/v are not yet committed.
    """
    if (os.environ.get('MINI_LLM_QSA_WARP_DOTS', '0') == '1' and
            q.shape[1] == 256 and q.shape[0] <= 24 and n <= 8192):
        from .qsa_warp_dots_gpu import qsa_warp_dots_gpu
        return qsa_warp_dots_gpu(q, qg, k, v, keys, values, n, scale, value_tile=32)
    if os.environ.get('MINI_LLM_QSA_COOPERATIVE', '0') == '1' and n <= 8192:
        from .qsa_cooperative_gpu import qsa_cooperative_gpu
        return qsa_cooperative_gpu(q, qg, k, v, keys, values, n, scale)
    device = q.device.id
    if device not in _KERNELS:
        _KERNELS[device] = xp.RawKernel(
            _SOURCE, "qsa_decode", options=("--std=c++14", "--fmad=false"))
    heads, dim = q.shape
    output = xp.empty(heads * dim, dtype=xp.float32)
    strides = tuple(np.int64(s // a.itemsize)
                    for a in (q, qg, k, v, keys, values) for s in a.strides)
    _KERNELS[device]((heads,), (128,),
                     (q, qg, k, v, keys, values, output,
                      np.int32(n), np.int32(dim), np.int32(heads // k.shape[0]),
                      np.float32(scale), *strides))
    return output
