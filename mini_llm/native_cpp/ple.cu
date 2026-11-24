// Equations: external/qwen4exp_ple.py:201-257. No runtime integration.
#include "ple.h"
#include <cmath>
#include <cstddef>

namespace {
constexpr int D = 2560, W = 4 * D, T = 256;
__device__ float check(float x, int* st) {
    if (!isfinite(x)) atomicExch(st, 1);
    return x;
}
__device__ float mul(float a, float b, int* st) {
    return check(__fmul_rn(check(a, st), check(b, st)), st);
}
__device__ float add(float a, float b, int* st) {
    return check(__fadd_rn(check(a, st), check(b, st)), st);
}
__device__ float sigmoid(float x, int* st) {
    check(x, st);
    float e = check(expf(-fabsf(x)), st);
    return check(__fdiv_rn(x >= 0 ? 1.0f : e, __fadd_rn(1.0f, e)), st);
}
__device__ float reduce(float v, float* sums, int* st) {
    int t = threadIdx.x;
    sums[t] = v;
    __syncthreads();
    for (int s = T / 2; s; s >>= 1) {
        if (t < s) sums[t] = add(sums[t], sums[t+s], st);
        __syncthreads();
    }
    return sums[0];
}
__global__ void project(const float* emb, const float* kw, const float* vw,
                        float* key, float* value, int* st) {
    int row = blockIdx.x;
    const float* w = row < W ? kw + size_t(row)*D : vw + size_t(row-W)*D;
    float sum = 0;
    for (int j = threadIdx.x; j < D; j += T)
        sum = add(sum, mul(w[j], emb[j], st), st);
    __shared__ float sums[T];
    sum = reduce(sum, sums, st);
    if (!threadIdx.x) {
        if (row < W) key[row] = sum;
        else value[row-W] = sum;
    }
}
__global__ void norm(const float* x, const float* g, float epsilon, float* y, int* st) {
    int base = blockIdx.x * D;
    float sum = 0;
    for (int j = threadIdx.x; j < D; j += T)
        sum = add(sum, mul(x[base+j], x[base+j], st), st);
    __shared__ float sums[T];
    sum = reduce(sum, sums, st);
    float rms = check(__fsqrt_rn(add(__fdiv_rn(sum, float(D)), epsilon, st)), st);
    for (int j = threadIdx.x; j < D; j += T)
        y[base+j] = mul(check(__fdiv_rn(x[base+j], rms), st), g[base+j], st);
}
__global__ void gate(const float* key, const float* query, float* gates, int* st) {
    int base = blockIdx.x * D;
    float sum = 0;
    for (int j = threadIdx.x; j < D; j += T)
        sum = add(sum, mul(key[base+j], query[base+j], st), st);
    __shared__ float sums[T];
    sum = reduce(sum, sums, st);
    if (!threadIdx.x) {
        float score = mul(sum, 1.0f / sqrtf(float(D)), st);
        float sign = score > 0 ? 1.0f : (score < 0 ? -1.0f : 0.0f);
        float root = check(__fsqrt_rn(fmaxf(fabsf(score), 1e-6f)), st);
        gates[blockIdx.x] = sigmoid(mul(sign, root, st), st);
    }
}
__global__ void gated_value(const float* value, const float* gates, float* u, int* st) {
    int k = blockIdx.x*T + threadIdx.x;
    if (k < W) u[k] = mul(gates[k/D], value[k%D], st);
}
__global__ void convolve(const float* h, const float* u, const float* z,
    const float* weights, const float* history, float* updated, float* next, int* st) {
    int k = blockIdx.x*T + threadIdx.x;
    if (k >= W) return;
    float conv = mul(history[k], weights[4*k], st);
    conv = add(conv, mul(history[3*W+k], weights[4*k+1], st), st);
    conv = add(conv, mul(history[6*W+k], weights[4*k+2], st), st);
    conv = add(conv, mul(z[k], weights[4*k+3], st), st);
    updated[k] = add(h[k], add(u[k], mul(conv, sigmoid(conv, st), st), st), st);
    for (int t = 0; t < 8; ++t) next[t*W+k] = check(history[(t+1)*W+k], st);
    next[8*W+k] = check(z[k], st);
}
}

cudaError_t ml_ple_project_launch(cudaStream_t stream, const float* embedding,
    const float* kw, const float* vw, float* key, float* value, int* status) {
    if (!embedding || !kw || !vw || !key || !value || !status) return cudaErrorInvalidValue;
    project<<<W+D,T,0,stream>>>(embedding,kw,vw,key,value,status);
    return cudaGetLastError();
}

cudaError_t ml_ple_apply_launch(cudaStream_t stream, float epsilon,
    const float* h, const float* key, const float* value,
    const float* gk, const float* gq, const float* gc,
    const float* weights, const float* history,
    float* kn, float* qn, float* u, float* gates, float* z,
    float* updated, float* next, int* status) {
    if (!std::isfinite(epsilon) || epsilon <= 0 || !h || !key || !value ||
        !gk || !gq || !gc || !weights || !history || !kn || !qn || !u ||
        !gates || !z || !updated || !next || !status) return cudaErrorInvalidValue;
    norm<<<4,T,0,stream>>>(key,gk,epsilon,kn,status);
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) return err;
    norm<<<4,T,0,stream>>>(h,gq,epsilon,qn,status);
    err = cudaGetLastError(); if (err != cudaSuccess) return err;
    gate<<<4,T,0,stream>>>(kn,qn,gates,status);
    err = cudaGetLastError(); if (err != cudaSuccess) return err;
    gated_value<<<(W+T-1)/T,T,0,stream>>>(value,gates,u,status);
    err = cudaGetLastError(); if (err != cudaSuccess) return err;
    norm<<<4,T,0,stream>>>(u,gc,epsilon,z,status);
    err = cudaGetLastError(); if (err != cudaSuccess) return err;
    convolve<<<(W+T-1)/T,T,0,stream>>>(h,u,z,weights,history,updated,next,status);
    return cudaGetLastError();
}
