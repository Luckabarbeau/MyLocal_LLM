// Reference graph: external/qwen4exp_hyperconnection.py; CUDA reduction/layout:
// ops/hyperconnection_{decode,read}_gpu.py. See WP4_HC_PRIMITIVES.md for provenance.
#include "hyperconnection.h"
#include <cmath>
#include <climits>
#include <cstddef>

namespace {
constexpr int threads = 256;
__device__ float checked(float x, int* status) {
    if (!isfinite(x)) atomicExch(status, 1);
    return x;
}
__device__ float sigmoid(float x, int* status) {
    checked(x, status);
    float z = checked(expf(-fabsf(x)), status);
    float denominator = __fadd_rn(1.0f, z);
    return checked(x >= 0.0f ? __fdiv_rn(1.0f, denominator)
                              : __fdiv_rn(z, denominator), status);
}
__device__ float product(float a, float b, int* status) {
    return checked(__fmul_rn(checked(a, status), checked(b, status)), status);
}
__device__ float add(float a, float b, int* status) {
    return checked(__fadd_rn(a, b), status);
}
__device__ float reduce(float value, float* sums, int* status) {
    const int lane = threadIdx.x;
    sums[lane] = value;
    __syncthreads();
    for (int step = threads / 2; step; step >>= 1) {
        if (lane < step) sums[lane] = add(sums[lane], sums[lane + step], status);
        __syncthreads();
    }
    return sums[0];
}
__global__ void normalize(const float* h, const float* gamma, float* n,
                          int d, float epsilon, int* status) {
    const size_t base = size_t(blockIdx.x) * d;
    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < size_t(d); j += threads) {
        float x = checked(h[base + j], status);
        sum = add(sum, product(x, x, status), status);
    }
    __shared__ float sums[threads];
    sum = reduce(sum, sums, status);
    float mean = checked(__fdiv_rn(sum, float(d)), status);
    float rms = checked(__fsqrt_rn(add(mean, epsilon, status)), status);
    for (size_t j = threadIdx.x; j < size_t(d); j += threads) {
        size_t k = base + j;
        // HC requires divide THEN gamma multiply, not x * inverse_rms.
        n[k] = product(checked(__fdiv_rn(h[k], rms), status), gamma[k], status);
    }
}
__global__ void down_inject(const float* n, const float* down,
    const float* inject, float* low, float* gains, int wide, int rank, int* status) {
    const int row = blockIdx.x;
    const bool gain = row >= rank;
    const float* w = gain ? inject + size_t(row - rank) * wide
                          : down + size_t(row) * wide;
    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < size_t(wide); j += threads)
        sum = add(sum, product(n[j], w[j], status), status);
    __shared__ float sums[threads];
    sum = reduce(sum, sums, status);
    if (threadIdx.x == 0) {
        float a = product(sum, 0.25f, status);
        if (gain) gains[row - rank] = product(2.0f, sigmoid(a, status), status);
        else low[row] = product(a, sigmoid(a, status), status);
    }
}
__global__ void up_mix(const float* n, const float* low, const float* up,
                       float* mixed, int d, int rank, int* status) {
    const int lane = threadIdx.x & 31;
    const size_t feature = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    if (feature >= size_t(d)) return; // Whole warp exits; shuffle mask remains full.
    float mix = 0.0f;
    for (int c = 0; c < 4; ++c) {
        size_t k = size_t(c) * d + feature;
        const float* w = up + k * rank;
        float dot = 0.0f;
        for (size_t r = lane; r < size_t(rank); r += 32)
            dot = add(dot, product(low[r], w[r], status), status);
        for (int step = 16; step; step >>= 1)
            dot = add(dot, __shfl_down_sync(0xffffffffu, dot, step), status);
        if (!lane)
            mix = add(mix, product(n[k], sigmoid(dot, status), status), status);
    }
    if (!lane) mixed[feature] = product(mix, 0.25f, status);
}
__global__ void write(const float* h, const float* output, const float* gains,
                      float* updated, int d, int* status) {
    size_t k = size_t(blockIdx.x) * threads + threadIdx.x;
    if (k < size_t(4) * d)
        updated[k] = add(checked(h[k], status),
                         product(gains[k / d], output[k % d], status), status);
}
__global__ void decode(const void* source, int type, uint64_t count,
                        float* destination, int* status) {
    for (uint64_t k = uint64_t(blockIdx.x) * threads + threadIdx.x;
         k < count; k += uint64_t(gridDim.x) * threads) {
        float x = type == 0 ? static_cast<const float*>(source)[k]
            : __uint_as_float(uint32_t(static_cast<const uint16_t*>(source)[k]) << 16);
        destination[k] = checked(x, status);
    }
}
bool valid_width(int d) { return d > 0 && d <= INT_MAX / 4; }
}

cudaError_t ml_hc_decode_launch(cudaStream_t stream, const void* source, int type,
    uint64_t count, float* destination, int* status) {
    if (!source || !destination || !status || !count ||
        (type != 0 && type != 30)) return cudaErrorInvalidValue;
    unsigned blocks = unsigned(count / threads + (count % threads != 0));
    if (count / threads >= 65535) blocks = 65535;
    decode<<<blocks, threads, 0, stream>>>(source, type, count, destination, status);
    return cudaGetLastError();
}

cudaError_t ml_hc_read_launch(cudaStream_t stream, int d, int rank, float epsilon,
    const float* h, const float* gamma, const float* down, const float* up,
    const float* inject, float* normalized, float* low, float* mixed,
    float* gains, int* status) {
    if (!valid_width(d) || rank <= 0 || rank > INT_MAX - 4 ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || !h || !gamma || !down ||
        !up || !normalized || !low || !mixed || !status ||
        ((inject == nullptr) != (gains == nullptr))) return cudaErrorInvalidValue;
    normalize<<<4, threads, 0, stream>>>(h, gamma, normalized, d, epsilon, status);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    down_inject<<<rank + (inject ? 4 : 0), threads, 0, stream>>>(
        normalized, down, inject, low, gains, 4 * d, rank, status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    up_mix<<<(d + 7) / 8, threads, 0, stream>>>(
        normalized, low, up, mixed, d, rank, status);
    return cudaGetLastError();
}

cudaError_t ml_hc_collapse_launch(cudaStream_t stream, int d, int rank,
    float epsilon, const float* h, const float* gamma, const float* down,
    const float* up, float* normalized, float* low, float* mixed, int* status) {
    return ml_hc_read_launch(stream, d, rank, epsilon, h, gamma, down, up,
                             nullptr, normalized, low, mixed, nullptr, status);
}

cudaError_t ml_hc_write_launch(cudaStream_t stream, int d, const float* h,
    const float* block_output, const float* gains, float* new_h, int* status) {
    if (!valid_width(d) || !h || !block_output || !gains || !new_h || !status)
        return cudaErrorInvalidValue;
    write<<<(size_t(4) * d + threads - 1) / threads, threads, 0, stream>>>(
        h, block_output, gains, new_h, d, status);
    return cudaGetLastError();
}
