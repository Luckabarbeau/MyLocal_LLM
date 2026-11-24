#include "qsa_attention.h"
#include <cuda_runtime.h>
#include <math.h>

namespace {
// Each existing history offset is validated once, including unselected rows.
// The same-stream attention launch and runtime status gate preserve output trust.
__global__ void validate_history(const uint64_t* parameter, const float* keys,
    const float* values, int* status) {
    const uint64_t n=*parameter;
    if (!n) { if (!blockIdx.x && !threadIdx.x) atomicExch(status,1); return; }
    const uint64_t count=(n-1)*512;
    const uint64_t stride=static_cast<uint64_t>(gridDim.x)*blockDim.x;
    for (uint64_t offset=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         offset<count; offset+=stride) {
        if (!isfinite(keys[offset]) || !isfinite(values[offset])) atomicExch(status,1);
    }
}

// Adapted from ops/qsa_decode_gpu.py: online tiled stable softmax, no KV gather.
__global__ void attention(const uint64_t* parameter, const int64_t* ids,
    const float* q, const float* gate, const float* k, const float* v,
    const float* keys, const float* values, float* contexts, float* gated,
    int* status, float scale) {
    const int h=blockIdx.x, kh=h/12, lane=threadIdx.x;
    const uint64_t n=*parameter;
    const int width=static_cast<int>(n<2051 ? n : 2051);
    __shared__ float reduction[128], weights[128];
    __shared__ float maximum, denominator, alpha;
    if (!n) { if (!lane) atomicExch(status,1); return; }
    if (!lane) { maximum=-INFINITY; denominator=0.0f; }
    for (int d=lane; d<256; d+=128) {
        if (!isfinite(q[h*256+d]) || !isfinite(gate[h*256+d]) ||
            !isfinite(k[kh*256+d]) || !isfinite(v[kh*256+d])) atomicExch(status,1);
        contexts[h*256+d]=0.0f;
    }

    __syncthreads();
    for (int base=0; base<width; base+=128) {
        int cell=base+lane, count=min(128,width-base);
        float logit=-INFINITY;
        if (cell<width) {
            int64_t token=ids[cell];
            if (token<0 || static_cast<uint64_t>(token)>=n) {
                atomicExch(status,1);
                token=n-1; // Memory-safe failure path; result remains unavailable.
            }
            const float* key=static_cast<uint64_t>(token)==n-1 ? k+kh*256 : keys+token*512+kh*256;
            float dot=0.0f;
            for (int d=0; d<256; ++d) dot+=q[h*256+d]*key[d];
            logit=dot*scale;
            if (!isfinite(logit)) atomicExch(status,1);
        }
        reduction[lane]=logit;
        __syncthreads();
        for (int s=64; s; s>>=1) {
            if (lane<s) reduction[lane]=fmaxf(reduction[lane],reduction[lane+s]);
            __syncthreads();
        }
        if (!lane) {
            float next=fmaxf(maximum,reduction[0]);
            alpha=expf(maximum-next); maximum=next;
        }
        __syncthreads();
        float weight=cell<width ? expf(logit-maximum) : 0.0f;
        weights[lane]=weight; reduction[lane]=weight;
        __syncthreads();
        for (int s=64; s; s>>=1) {
            if (lane<s) reduction[lane]+=reduction[lane+s];
            __syncthreads();
        }
        if (!lane) denominator=denominator*alpha+reduction[0];
        for (int d=lane; d<256; d+=128) {
            float sum=0.0f;
            for (int j=0; j<count; ++j) {
                int64_t token=ids[base+j];
                if (token<0 || static_cast<uint64_t>(token)>=n) token=n-1;
                float value=static_cast<uint64_t>(token)==n-1 ? v[kh*256+d] : values[token*512+kh*256+d];
                sum+=weights[j]*value;
            }
            float numerator=contexts[h*256+d]*alpha+sum;
            if (!isfinite(numerator)) atomicExch(status,1);
            contexts[h*256+d]=numerator;
        }
        __syncthreads();
    }
    for (int d=lane; d<256; d+=128) {
        float g=gate[h*256+d], z=expf(-fabsf(g));
        float sigmoid=g>=0.0f ? 1.0f/(1.0f+z) : z/(1.0f+z);
        float context=contexts[h*256+d]/denominator;
        float output=context*sigmoid;
        if (!isfinite(denominator) || denominator<=0.0f || !isfinite(context) || !isfinite(output)) atomicExch(status,1);
        contexts[h*256+d]=context; gated[h*256+d]=output;
    }
}
}
cudaError_t ml_qsa_attention_launch(cudaStream_t stream, const uint64_t* n,
    const int64_t* ids, const float* q, const float* gate, const float* k,
    const float* v, const float* keys, const float* values,
    float* contexts, float* gated, int* status, float scale) {
    // Caller clears status once per run, so repeats cannot erase a failure.
    // Bounds use device n, excluding the uncommitted current slot/future capacity.
    validate_history<<<256,256,0,stream>>>(n,keys,values,status);
    cudaError_t validation=cudaGetLastError();
    if (validation!=cudaSuccess) return validation;
    attention<<<24,128,0,stream>>>(n,ids,q,gate,k,v,keys,values,contexts,gated,status,scale);
    return cudaGetLastError();
}
