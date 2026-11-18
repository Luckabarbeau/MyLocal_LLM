"""Opt-in warp-per-row grouped IQ execution; exact packed lane decoder reused.

Four independent row warps share a CTA. No block-wide barriers/shared-memory
reduction are needed. FP32 products remain unchanged; the within-row reduction
order differs from the block oracle. Routing multiply/storage/ordered expert
addition remain the established kernels. No activation quantization is added.
"""
from functools import lru_cache

from . import iq_packed_gpu as iq


_SOURCE = r'''
__device__ __forceinline__ float row_sum(float v) {
    for (int step = 16; step; step >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, step);
    return v;
}
extern "C" __global__ void iq_warp_gate_up(
    const float* x, const unsigned long long* table, float* hidden,
    int d, int f, int width, int gf, int gqk, int gqb,
    int uf, int uqk, int uqb, int tile, int experts) {
    const long long pair = (long long)blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (pair >= (long long)experts * f) return;
    const int expert = pair / f, row = pair % f;
    const unsigned char* gr = (const unsigned char*)table[(long long)expert * width]
                           + (long long)row * (d / gqk) * gqb;
    const unsigned char* ur = (const unsigned char*)table[(long long)expert * width + 1]
                           + (long long)row * (d / uqk) * uqb;
    float ag = 0.0f, au = 0.0f;
    for (int col = lane; col < d; col += 32) {
        float xv = x[col];
        ag += xv * lane_weight(gr, col, gf, gqk, gqb);
        au += xv * lane_weight(ur, col, uf, uqk, uqb);
    }
    ag = row_sum(ag); au = row_sum(au);
    if (!lane) hidden[pair] = (ag / (1.0f + expf(-ag))) * au;
}
extern "C" __global__ void iq_warp_down_weighted(
    const float* hidden, const unsigned long long* table, const float* routes,
    float* terms, int d, int f, int width, long long route_stride,
    int df, int dqk, int dqb, int tile, int experts) {
    const long long pair = (long long)blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (pair >= (long long)experts * d) return;
    const int expert = pair / d, row = pair % d;
    const unsigned char* wr = (const unsigned char*)table[(long long)expert * width + 2]
                           + (long long)row * (f / dqk) * dqb;
    const float* h = hidden + (long long)expert * f;
    float value = 0.0f;
    for (int col = lane; col < f; col += 32)
        value += h[col] * lane_weight(wr, col, df, dqk, dqb);
    value = row_sum(value);
    if (!lane) {
        const long long slot = width == 4 ? table[(long long)expert * width + 3] : expert;
        terms[pair] = value * routes[slot * route_stride];
    }
}
'''


@lru_cache(maxsize=None)
def module(device_id, storage):
    # The canonical source includes codebooks, lane equations and MIT notices.
    # Original block/reduce symbols remain available as an independent path.
    with iq.cp.cuda.Device(device_id):
        return iq.cp.RawModule(code=iq._cuda_source(storage).replace('INPUT_T', 'float')
                               + iq._GROUPED_CUDA + _SOURCE,
                               options=('--std=c++11', '--fmad=false'))
