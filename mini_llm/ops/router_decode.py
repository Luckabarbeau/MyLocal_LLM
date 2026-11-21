"""Exact-selection FP32 single-row softmax/top-k/renormalization for inference.

No training changes or approximate routing. Full softmax is retained before
selected-mass normalization, including the caller's normalization floor.
"""

import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


_KERNELS = {}
_OPTIONS = ('--std=c++14', '--fmad=false')
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


# Retain the original kernel verbatim as the independently selectable oracle.
_BLOCK_SELECTION = """        best_values[tid] = best; best_ids[tid] = index;
        __syncthreads();
        for (int stride = 64; stride; stride >>= 1) {
            if (tid < stride && better(best_values[tid + stride], best_ids[tid + stride],
                                       best_values[tid], best_ids[tid], n)) {
                best_values[tid] = best_values[tid + stride];
                best_ids[tid] = best_ids[tid + stride];
            }
            __syncthreads();
        }
"""
_WARP_SELECTION = """        int lane = tid & 31, warp = tid >> 5;
        // Every lane participates, including threads with no remaining expert.
        for (int stride = 16; stride; stride >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, best, stride);
            int other_id = __shfl_down_sync(0xffffffffu, index, stride);
            if (lane + stride < 32 && better(other, other_id, best, index, n)) {
                best = other; index = other_id;
            }
        }
        if (lane == 0) {
            best_values[warp] = best; best_ids[warp] = index;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < 4 ? best_values[lane] : -INFINITY;
            index = lane < 4 ? best_ids[lane] : n;
            for (int stride = 2; stride; stride >>= 1) {
                float other = __shfl_down_sync(0xffffffffu, best, stride);
                int other_id = __shfl_down_sync(0xffffffffu, index, stride);
                if (lane + stride < 4 && better(other, other_id, best, index, n)) {
                    best = other; index = other_id;
                }
            }
            // Thread zero alone consumes this winner before the existing barrier.
            if (lane == 0) best_ids[0] = index;
        }
"""


def _warp_selection_source(source):
    if not isinstance(source, str):
        raise TypeError('Expected a CUDA source string')
    if source.count(_BLOCK_SELECTION) != 1 or _WARP_SELECTION in source:
        raise ValueError('Router selection source contract changed')
    return source.replace(_BLOCK_SELECTION, _WARP_SELECTION, 1)


_WARP_SOURCE = _warp_selection_source(_SOURCE)


_MAX_REDUCTION = """    for (int stride = 64; stride; stride >>= 1) {
        if (tid < stride) {
            float a = partial[tid], b = partial[tid + stride];
            partial[tid] = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
"""
_SUM_REDUCTION = """    for (int stride = 64; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
"""
_WARP_MAX_REDUCTION = """    for (int stride = 64; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            float a = partial[tid], b = partial[tid + stride];
            partial[tid] = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    if (tid < 32) {
        int lane = tid;
        float a = partial[lane];
        for (int stride = 16; stride; stride >>= 1) {
            float b = __shfl_down_sync(0xffffffffu, a, stride);
            if (lane < stride) a = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
        }
        if (lane == 0) partial[0] = a;
    }
    __syncthreads();
"""
_WARP_SUM_REDUCTION = """    for (int stride = 64; stride >= 32; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        int lane = tid;
        float sum = partial[lane];
        for (int stride = 16; stride; stride >>= 1) {
            float b = __shfl_down_sync(0xffffffffu, sum, stride);
            if (lane < stride) sum += b;
        }
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
"""


def _warp_reduction_source(source):
    """Change only the two trees' communication, not their FP32 operations."""
    if not isinstance(source, str):
        raise TypeError('Expected a CUDA source string')
    if (source.count(_WARP_SELECTION) != 1 or source.count(_MAX_REDUCTION) != 1
            or source.count(_SUM_REDUCTION) != 1 or _WARP_MAX_REDUCTION in source
            or _WARP_SUM_REDUCTION in source):
        raise ValueError('Router reduction source contract changed')
    return source.replace(_MAX_REDUCTION, _WARP_MAX_REDUCTION, 1).replace(
        _SUM_REDUCTION, _WARP_SUM_REDUCTION, 1)


_REDUCE_SOURCE = _warp_reduction_source(_WARP_SOURCE)


def _selection_modes(warp_select, warp_reduce, *, environment=False):
    for name, value in (('warp_select', warp_select), ('warp_reduce', warp_reduce)):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f'{name} must be bool or None')
    if warp_reduce is None:
        # Explicit 0327 overrides retain their old meaning even with REDUCE=1.
        warp_reduce = (environment and warp_select is None
                       and os.environ.get('MINI_LLM_ROUTER_WARP_REDUCE', '0') == '1')
    if warp_reduce:
        if warp_select is False:
            raise ValueError('warp_reduce=True conflicts with warp_select=False')
        return True, True
    if warp_select is None:
        warp_select = environment and os.environ.get('MINI_LLM_ROUTER_WARP_SELECT', '0') == '1'
    return warp_select, False


def _get_kernel(device, *, warp_select=None, warp_reduce=False):
    """Explicit raw variant, ignoring flags; default original, True reduce implies select."""
    warp_select, warp_reduce = _selection_modes(warp_select, warp_reduce)
    # Preserve the two existing cache keys; the full variant has a third key.
    key = (device, 'reduce' if warp_reduce else warp_select)
    if key not in _KERNELS:
        source = _REDUCE_SOURCE if warp_reduce else _WARP_SOURCE if warp_select else _SOURCE
        _KERNELS[key] = xp.RawKernel(source, 'normalized_topk', options=_OPTIONS)
    return _KERNELS[key]


def decode_enabled():
    return BACKEND_NAME == 'cupy' and os.environ.get(
        'MINI_LLM_ROUTER_DECODE', '1').lower() not in {'0', 'false', 'off', 'no'}


def try_normalized_topk(logits, k, normalization_floor, *, warp_select=None, warp_reduce=None):
    """Return weights/IDs or None for unsupported metadata.

    Both None follows default-off SELECT/REDUCE flags; REDUCE implies selection.
    Explicit warp_select=False/True alone forces original/selection-only and
    ignores both flags. warp_reduce=False disables reductions, leaving selection
    automatic if None. warp_reduce=True implies selection unless explicitly
    contradicted by warp_select=False (ValueError). Raw _get_kernel ignores flags.
    """
    warp_select, warp_reduce = _selection_modes(warp_select, warp_reduce, environment=True)
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
    kernel = _get_kernel(device, warp_select=warp_select, warp_reduce=warp_reduce)
    shape = logits.shape[:-1] + (k,)
    weights = xp.empty(shape, dtype=xp.float32)
    ids = xp.empty(shape, dtype=xp.int64)
    kernel((1,), (128,), (logits, weights, ids, np.int32(logits.shape[-1]),
                          np.int32(k), np.float32(normalization_floor)))
    return weights, ids
