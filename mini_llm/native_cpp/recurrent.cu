#include "recurrent.h"
#include <cuda_runtime.h>
// Reused verbatim arithmetic from ops/gated_delta_decode.py; static modulo
// mapping replaces its uploaded head_map. Runtime supplies staged state only.
namespace {

extern "C" __global__ void conv_silu_history(
    const float* x, const float* weight, const float* bias,
    float* history, float* convolved, int channels, int taps, int count,
    int has_bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int b = i / channels, c = i % channels;
    int base = b * (taps - 1) * channels + c;
    float current = x[i], z = 0.0f;
    // One thread owns all history slots of a channel: no shift races.
    for (int j = 0; j < taps - 1; ++j) {
        float raw = history[base + j * channels];
        z += raw * weight[c * taps + j];
    }
    z += current * weight[c * taps + taps - 1];
    if (has_bias) z += bias[c];
    convolved[i] = z * (1.0f / (1.0f + expf(-z)));
    for (int j = 0; j < taps - 2; ++j)
        history[base + j * channels] = history[base + (j + 1) * channels];
    if (taps > 1) history[base + (taps - 2) * channels] = current;
}

extern "C" __global__ void normalized_delta(
    const float* convolved, const float* log_decay, const float* beta,
    int hk, float* memory, float* output,
    int channels, int q_width, int hv, int dk, int dv,
    float eps, float query_scale, int* status) {
    // Four key partitions, 32 contiguous value columns per block. Rectangular
    // memory is coalesced along Dv; each tile exclusively owns its columns.
    int row = blockIdx.x, b = row / hv, h = row % hv;
    int lane = threadIdx.x & 31, part = threadIdx.x >> 5;
    int col = blockIdx.y * 32 + lane, tid = threadIdx.x;
    int kh = h % hk, packed = b * channels + kh * dk;
    __shared__ float q[256], k[256], sq[128], sk[128], partial[128];
    float qs = 0.0f, ks = 0.0f;
    for (int d = tid; d < dk; d += 128) {
        float a = convolved[packed + d];
        float z = convolved[packed + q_width + d];
        q[d] = a; k[d] = z;
        qs += a * a; ks += z * z;
    }
    sq[tid] = qs; sk[tid] = ks;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (tid < s) { sq[tid] += sq[tid + s]; sk[tid] += sk[tid + s]; }
        __syncthreads();
    }
    float qden = sqrtf(sq[0] + eps), kden = sqrtf(sk[0] + eps);
    if (tid==0 && (!isfinite(qden) || !isfinite(kden))) atomicExch(status,1);
    for (int d = tid; d < dk; d += 128) {
        // Preserve the FP32 normalized storage and scaled-query boundaries.
        q[d] = (q[d] / qden) * query_scale;
        k[d] = k[d] / kden;
    }
    __syncthreads();
    int base = row * dk * dv;
    float decay = expf(log_decay[row]), prediction = 0.0f;
    if (col < dv) {
        for (int d = part; d < dk; d += 4) {
            int index = base + d * dv + col;
            float decayed = memory[index] * decay;
            memory[index] = decayed;
            prediction += k[d] * decayed;
        }
    }
    partial[tid] = prediction;
    __syncthreads();
    prediction = ((partial[lane] + partial[32 + lane]) +
                  partial[64 + lane]) + partial[96 + lane];
    float correction = 0.0f;
    if (col < dv)
        correction = (convolved[b * channels + 2 * q_width + h * dv + col]
                      - prediction) * beta[row];
    // All partitions must finish reading prediction before shared reuse.
    __syncthreads();
    float readout = 0.0f;
    if (col < dv) {
        for (int d = part; d < dk; d += 4) {
            int index = base + d * dv + col;
            float updated = memory[index] + k[d] * correction;
            memory[index] = updated;
            readout += q[d] * updated;
        }
    }
    partial[tid] = readout;
    __syncthreads();
    if (part == 0 && col < dv)
        output[row * dv + col] = ((partial[lane] + partial[32 + lane]) +
                                  partial[64 + lane]) + partial[96 + lane];
}

extern "C" __global__ void gated_rms(
    const float* x, const float* gate, const float* gamma, float* output,
    int dv, float eps, int sigmoid_gate, int* status) {
    int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float sums[128];
    float sum = 0.0f;
    for (int d = tid; d < dv; d += 128) {
        float z = x[row * dv + d];
        sum += z * z;
    }
    sums[tid] = sum;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (tid < s) sums[tid] += sums[tid + s];
        __syncthreads();
    }
    if (tid==0 && !isfinite(sums[0])) atomicExch(status,1);
    float inv = 1.0f / sqrtf(sums[0] / (float)dv + eps);
    for (int d = tid; d < dv; d += 128) {
        int i = row * dv + d;
        float g = gate[i];
        float activated;
        if (sigmoid_gate) {
            float z = expf(-fabsf(g));
            activated = g >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
        } else {
            activated = g * (1.0f / (1.0f + expf(-g)));
        }
        output[i] = ((x[i] * inv) * gamma[d]) * activated;
    }
}

// Same FP32 logaddexp routine as CuPy and qwen4exp_elementwise_gpu.py.
// The equality branch intentionally adds the double log(2.0) literal.
__device__ float softplus(float x) {
    if (x==0.0f) return float(x + log(2.0));
    return fmaxf(0.0f,x) + log1pf(expf(-fabsf(x)));
}
__global__ void coefficients(const float* logits, const float* alpha,
    const float* a, const float* dt, float* beta, float* decay, int hv, int* status) {
    int h=threadIdx.x;
    if (h>=hv) return;
    if (!isfinite(logits[h]) || !isfinite(alpha[h]) || !isfinite(a[h]) ||
        !(a[h]<0) || !isfinite(dt[h])) atomicExch(status,1);
    beta[h]=expf(-softplus(-logits[h]));
    decay[h]=a[h]*softplus(alpha[h]+dt[h]);
    if (!isfinite(beta[h]) || !isfinite(decay[h])) atomicExch(status,1);
}
}
cudaError_t ml_recurrent_launch(cudaStream_t stream, int hk, int hv, int dim,
    int taps, float epsilon, const float* qkv, const float* gate,
    const float* beta_logits, const float* alpha, const float* conv_weight,
    const float* gamma, const float* a, const float* dt, float* history,
    float* memory, float* convolved, float* beta, float* log_decay,
    float* readout, float* gated, int* status) {
    int channels=(2*hk+hv)*dim;
    coefficients<<<1,128,0,stream>>>(beta_logits,alpha,a,dt,beta,log_decay,hv,status);
    auto result=cudaGetLastError(); if (result!=cudaSuccess) return result;
    conv_silu_history<<<(channels+127)/128,128,0,stream>>>(qkv,conv_weight,
        conv_weight,history,convolved,channels,taps,channels,0);
    result=cudaGetLastError(); if (result!=cudaSuccess) return result;
    normalized_delta<<<dim3(hv,(dim+31)/32),128,0,stream>>>(convolved,log_decay,
        beta,hk,memory,readout,channels,hk*dim,hv,dim,dim,epsilon,1.0f/sqrtf(float(dim)),status);
    result=cudaGetLastError(); if (result!=cudaSuccess) return result;
    gated_rms<<<hv,128,0,stream>>>(readout,gate,gamma,gated,dim,epsilon,1,status);
    return cudaGetLastError();
}
