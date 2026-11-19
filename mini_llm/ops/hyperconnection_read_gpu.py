"""Opt-in three-launch, single-token FP32 HC read (no decoder dispatch changes).

Only compiled kernels are cached per current device. Scratch and outputs belong
exclusively to each call; launches use the caller's current stream without host
readback or synchronization. No activation or weight precision conversion occurs.
"""

from functools import lru_cache
import math

import numpy as np

from ..backend import xp
from .hyperconnection_decode_gpu import _OPTIONS, _SIGMOID, _kernels as _decode_kernels


_SOURCE = _SIGMOID + r'''
__device__ float hc_weight(const float* p, int i) { return p[i]; }
__device__ float hc_weight(const unsigned short* p, int i) {
    return __uint_as_float((unsigned int)p[i] << 16);
}
extern "C" __global__ void hc_read_down_inject(
    const float* n, const HC_W* down, const HC_W* inject,
    float* low, float* gains, int wide, int rank) {
    int row = blockIdx.x, lane = threadIdx.x;
    bool gain = row >= rank;
    const HC_W* w = gain ? inject + (long long)(row - rank) * wide
                          : down + (long long)row * wide;
    float sum = 0.0f;
    for (int j = lane; j < wide; j += 256) {
        float product = n[j] * hc_weight(w,j);
        sum += product;
    }
    __shared__ float sums[256];
    sums[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) sums[lane] += sums[lane + step];
        __syncthreads();
    }
    if (!lane) {
        float a = sums[0] * 0.25f;
        if (gain) gains[row - rank] = 2.0f * hc_sigmoid(a);
        else low[row] = a * hc_sigmoid(a);
    }
}

__device__ float row_sum(float v) {
    for (int step = 16; step; step >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, step);
    return v;
}

extern "C" __global__ void hc_read_up_mix(
    const float* n, const float* low, const HC_W* up,
    float* mixed, int d, int rank) {
    int lane = threadIdx.x & 31;
    int feature = blockIdx.x * 8 + threadIdx.x / 32;
    if (feature >= d) return;
    float mix = 0.0f;
    for (int c = 0; c < 4; ++c) {
        int k = c * d + feature;
        const HC_W* w = up + (long long)k * rank;
        float dot = 0.0f;
        for (int r = lane; r < rank; r += 32) {
            float product = low[r] * hc_weight(w,r);
            dot += product;
        }
        dot = row_sum(dot);
        if (!lane) {
            float product = n[k] * hc_sigmoid(dot);
            mix += product;
        }
    }
    if (!lane) mixed[feature] = mix * 0.25f;
}
'''


def _validate(streams, gamma, down, up, inject, epsilon):
    if xp.__name__ != "cupy":
        raise TypeError("read_gpu requires the CuPy backend")
    device = xp.cuda.runtime.getDevice()
    weight_dtype=xp.uint16 if isinstance(down,xp.ndarray) and down.dtype==xp.uint16 else xp.float32
    for name, a in (("streams", streams), ("gamma", gamma), ("down", down),
                    ("up", up), ("inject", inject)):
        if name == "inject" and a is None:
            continue
        dtype=xp.float32 if name in ('streams','gamma') else weight_dtype
        if not isinstance(a, xp.ndarray) or a.dtype != dtype:
            raise TypeError(f"{name} must be a CuPy {dtype} array")
        if not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if a.device.id != device:
            raise ValueError(f"{name} must be on the current device")
    if (streams.ndim < 2 or streams.shape[-2] != 4
            or streams.shape[-1] <= 0 or streams.size != 4 * streams.shape[-1]):
        raise ValueError("streams must have single-token shape [..., 4, D], D > 0")
    d = streams.shape[-1]
    if gamma.shape != (4, d):
        raise ValueError("gamma must have shape [4, D]")
    if down.ndim != 2 or down.shape[0] <= 0 or down.shape[1] != 4 * d:
        raise ValueError("down must have shape [R, 4*D], R > 0")
    rank = down.shape[0]
    if up.shape != (4 * d, rank):
        raise ValueError("up must have shape [4*D, R]")
    if inject is not None and inject.shape != (4, 4 * d):
        raise ValueError("inject must have shape [4, 4*D]")
    if max(4 * d, rank + 4) > np.iinfo(np.int32).max:
        raise ValueError("dimensions exceed int32 kernel indexing")
    # Accept host scalars only: converting a device scalar would read back.
    if not isinstance(epsilon, (int, float, np.integer, np.floating)):
        raise TypeError("epsilon must be a host scalar")
    epsilon = float(epsilon)
    if (not math.isfinite(epsilon) or epsilon < float(np.nextafter(np.float32(0), np.float32(1)))
            or epsilon > float(np.finfo(np.float32).max)):
        raise ValueError("epsilon must be representable as positive float32")
    return d, rank, np.float32(epsilon), device


def supports_read_gpu(streams, gamma, down, up, inject, epsilon):
    """Metadata/host-scalar-only eligibility check, with the read API's contract."""
    try:
        _validate(streams, gamma, down, up, inject, epsilon)
    except (TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=None)
def _kernels(device):
    return (_decode_kernels(device)[0],
            xp.RawKernel('#define HC_W float\n'+_SOURCE, "hc_read_down_inject", options=_OPTIONS),
            xp.RawKernel('#define HC_W float\n'+_SOURCE, "hc_read_up_mix", options=_OPTIONS))


@lru_cache(maxsize=None)
def _bf16_kernels(device):
    source='#define HC_W unsigned short\n'+_SOURCE
    return (_decode_kernels(device)[0],
            xp.RawKernel(source,'hc_read_down_inject',options=_OPTIONS),
            xp.RawKernel(source,'hc_read_up_mix',options=_OPTIONS))


def read_gpu(streams, gamma, down, up, inject, epsilon):
    """Return fresh ``(mixed [..., D], gains [..., 4] or None)``.

    Requires contiguous current-device CuPy FP32 arrays, canonical output-first
    weights and exactly one token (all leading dimensions have size one).
    Unsupported inputs raise rather than silently selecting another graph.
    """
    d, rank, epsilon, device = _validate(streams, gamma, down, up, inject, epsilon)
    norm, project, mix = (_bf16_kernels(device) if down.dtype==xp.uint16 else _kernels(device))
    leading = streams.shape[:-2]
    normalized = xp.empty((4 * d,), dtype=xp.float32)
    low = xp.empty((rank,), dtype=xp.float32)
    mixed = xp.empty(leading + (d,), dtype=xp.float32)
    gains = None if inject is None else xp.empty(leading + (4,), dtype=xp.float32)
    norm((4,), (256,), (streams, gamma, normalized, np.int32(d), epsilon))
    # Non-null dummy pointers are never dereferenced when injection is absent.
    project((rank + (4 if inject is not None else 0),), (256,),
            (normalized, down, down if inject is None else inject, low,
             low if gains is None else gains, np.int32(4 * d), np.int32(rank)))
    mix(((d + 7) // 8,), (256,),
        (normalized, low, up, mixed, np.int32(d), np.int32(rank)))
    return mixed, gains
