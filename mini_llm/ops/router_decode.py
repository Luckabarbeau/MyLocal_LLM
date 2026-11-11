"""Exact-selection FP32 single-row softmax/top-k/renormalization for inference.

No training changes or approximate routing. Full softmax is retained before
selected-mass normalization, including the caller's normalization floor.
"""

import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


_KERNELS = {}
_SOURCE = r"""
#define INFINITY __int_as_float(0x7f800000)
#define NAN __int_as_float(0x7fc00000)
__device__ bool better(float a, int ia, float b, int ib, int n) {
    if (ia == n) return false;
    if (ib == n) return true;
    // Stable descending value order, NaNs last, lower IDs first on ties.
    if (isnan(a)) return isnan(b) && ia < ib;
    if (isnan(b)) return true;
    return a > b || (a == b && ia < ib);
}
extern "C" __global__ void normalized_topk(
    const float* logits, float* weights, long long* ids,
    int n, int k, float floor_value) {
    int tid = threadIdx.x;
    __shared__ float values[4096], partial[128], best_values[128], selected[32];
    __shared__ int best_ids[128];
    __shared__ unsigned char used[4096];
    float maximum = -INFINITY;
    for (int i = tid; i < n; i += 128) {
        float v = logits[i];
        values[i] = v; used[i] = 0;
        maximum = isnan(v) || isnan(maximum) ? NAN : fmaxf(maximum, v);
    }
    partial[tid] = maximum;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (tid < stride) {
            float a = partial[tid], b = partial[tid + stride];
            partial[tid] = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    maximum = partial[0];
    float denominator = 0.0f;
    for (int i = tid; i < n; i += 128)
        denominator += expf(values[i] - maximum);
    __syncthreads();
    partial[tid] = denominator;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    denominator = partial[0];
    for (int slot = 0; slot < k; ++slot) {
        float best = -INFINITY;
        int index = n;
        for (int i = tid; i < n; i += 128) {
            if (!used[i] && better(values[i], i, best, index, n)) {
                best = values[i]; index = i;
            }
        }
        best_values[tid] = best; best_ids[tid] = index;
        __syncthreads();
        for (int stride = 64; stride; stride >>= 1) {
            if (tid < stride && better(best_values[tid + stride], best_ids[tid + stride],
                                       best_values[tid], best_ids[tid], n)) {
                best_values[tid] = best_values[tid + stride];
                best_ids[tid] = best_ids[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int chosen = best_ids[0];
            ids[slot] = chosen;
            used[chosen] = 1;
            selected[slot] = __fdiv_rn(expf(values[chosen] - maximum), denominator);
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mass = 0.0f;
        for (int slot = 0; slot < k; ++slot) mass += selected[slot];
        // maximum(NaN, floor) propagates NaN in the portable graph.
        float normalizer = isnan(mass) ? mass : fmaxf(mass, floor_value);
        for (int slot = 0; slot < k; ++slot)
            weights[slot] = __fdiv_rn(selected[slot], normalizer);
    }
}
"""


def decode_enabled():
    return BACKEND_NAME == 'cupy' and os.environ.get(
        'MINI_LLM_ROUTER_DECODE', '1').lower() not in {'0', 'false', 'off', 'no'}


def try_normalized_topk(logits, k, normalization_floor):
    """Return weights/IDs or None before allocation for unsupported metadata."""
    if (BACKEND_NAME != 'cupy' or not isinstance(logits, xp.ndarray)
            or logits.dtype != xp.float32 or not logits.flags.c_contiguous
            or logits.ndim < 1 or logits.size != logits.shape[-1]
            or not 1 <= logits.shape[-1] <= 4096
            or not 1 <= k <= min(32, logits.shape[-1])
            or not np.finfo(np.float32).tiny <= normalization_floor <= np.finfo(np.float32).max):
        return None
    device = xp.cuda.runtime.getDevice()
    if logits.device.id != device:
        return None
    if device not in _KERNELS:
        _KERNELS[device] = xp.RawKernel(_SOURCE, 'normalized_topk',
                                      options=('--std=c++14', '--fmad=false'))
    shape = logits.shape[:-1] + (k,)
    weights = xp.empty(shape, dtype=xp.float32)
    ids = xp.empty(shape, dtype=xp.int64)
    _KERNELS[device]((1,), (128,), (logits, weights, ids, np.int32(logits.shape[-1]),
                                 np.int32(k), np.float32(normalization_floor)))
    return weights, ids
