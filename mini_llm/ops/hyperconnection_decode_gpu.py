"""Opt-in single-token HC fusion; FP32 dense matmuls remain CuPy operations.

Only compiled kernels are cached, never activations. All launches use the
caller's current stream, with no host transfers or synchronization.
"""

from functools import lru_cache

import numpy as np

from ..backend import xp


_OPTIONS = ("--std=c++17", "--fmad=false")
_SIGMOID = r'''
__device__ float hc_sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
'''
_NORM = r'''
extern "C" __global__ void hc_norm(
    const float* h, const float* gamma, float* out, int d, float epsilon) {
    int c = blockIdx.x, lane = threadIdx.x;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (int j = lane; j < d; j += 256) {
        float v = h[c * d + j];
        sum += v * v;
    }
    sums[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) sums[lane] += sums[lane + step];
        __syncthreads();
    }
    float rms = sqrtf(sums[0] / (float)d + epsilon);
    for (int j = lane; j < d; j += 256) {
        int k = c * d + j;
        // Preserve the explicit divide, then gamma multiply (not rsqrt).
        out[k] = (h[k] / rms) * gamma[k];
    }
}
'''


def eligible(streams, *arrays):
    """Metadata-only guard; callers retain their existing shape validation."""
    if xp.__name__ != "cupy" or streams.size != 4 * streams.shape[-1]:
        return False
    device = xp.cuda.runtime.getDevice()
    return all(isinstance(a, xp.ndarray) and a.dtype == xp.float32
               and a.flags.c_contiguous and a.device.id == device
               for a in (streams,) + arrays if a is not None)


@lru_cache(maxsize=None)
def _kernels(device):
    # Device keys isolate compiled objects; no device/stream context is changed.
    norm = xp.RawKernel(_NORM, "hc_norm", options=_OPTIONS)
    silu = xp.ElementwiseKernel(
        "float32 x", "float32 y",
        "float a = x * 0.25f; y = a * hc_sigmoid(a);",
        "hc_decode_silu", preamble=_SIGMOID, options=_OPTIONS)
    mix = xp.ElementwiseKernel(
        "raw float32 n, raw float32 gates, int32 d", "float32 y",
        """float sum = 0.0f;
        for (int c = 0; c < 4; ++c) {
            int k = c * d + i;
            sum += n[k] * hc_sigmoid(gates[k]);
        }
        y = sum * 0.25f;""",
        "hc_decode_mix", preamble=_SIGMOID, options=_OPTIONS)
    gain = xp.ElementwiseKernel(
        "float32 x", "float32 y", "y = 2.0f * hc_sigmoid(x * 0.25f);",
        "hc_decode_gain", preamble=_SIGMOID, options=_OPTIONS)
    write = xp.ElementwiseKernel(
        "float32 h, raw float32 output, raw float32 gains, int32 d",
        "float32 y", "float delta = gains[i / d] * output[i % d]; y = h + delta;",
        "hc_decode_write", options=_OPTIONS)
    return norm, silu, mix, gain, write


def mix_gpu(streams, gamma, down, up, epsilon, *, project=None):
    """Return the read mix and normalized flat features for one token."""
    d = streams.shape[-1]
    leading = streams.shape[:-2]
    norm, silu, mix, _, _ = _kernels(streams.device.id)
    flat = xp.empty(leading + (4 * d,), dtype=xp.float32)
    norm((4,), (256,), (streams, gamma, flat, np.int32(d), epsilon))
    low = silu(flat @ down.T if project is None else project(flat, down))
    gates = low @ up.T if project is None else project(low, up)
    mixed = xp.empty(leading + (d,), dtype=xp.float32)
    mix(flat, gates, np.int32(d), mixed)
    return mixed, flat


def gains_gpu(flat, inject, *, project=None):
    return _kernels(flat.device.id)[3](flat @ inject.T if project is None else project(flat, inject))


def write_gpu(streams, output, gains):
    return _kernels(streams.device.id)[4](streams, output, gains,
                                         np.int32(streams.shape[-1]))
