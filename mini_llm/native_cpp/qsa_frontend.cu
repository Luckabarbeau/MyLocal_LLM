#include "qsa_frontend.h"
#include <cuda_runtime.h>
#include <math.h>

namespace {
__device__ float checked(float value, int* status) {
    if (!isfinite(value)) atomicExch(status,1);
    return value;
}

// 24 query rows, 2 key rows, 4 index-query rows, dead row, pooled row.
__global__ void frontend(const uint64_t* parameter,
    const float* qg, const float* k, const float* raw, const float* iq,
    const float* first_raw, const float* partial,
    const float* gamma_q, const float* gamma_k,
    const float* gamma_iq, const float* gamma_ik, float epsilon,
    float* q, float* rotated_k, float* gate, float* rotated_iq, float* dead,
    int* new_pool_ready, float* new_pool, int* status) {
    const int row=blockIdx.x, lane=threadIdx.x;
    const uint64_t n=*parameter;
    if (row==31 && !lane) *new_pool_ready=(n!=0 && n%4==0) ? 1 : 0;
    if (!n) {
        if (!lane) atomicExch(status,1);
        return;
    }
    if (row==31 && n%4!=0) return;
    // The current raw projection is retained by the caller even without a pool.
    if (row==30 && lane<128) checked(raw[lane],status);

    const int width=row<26 ? 256 : 128;
    const float* input=nullptr;
    const float* gamma=nullptr;
    float* output=nullptr;
    uint64_t position=n-1;
    if (row<24) {
        input=qg+row*512; gamma=gamma_q; output=q+row*256;
        float g=checked(input[256+lane],status);
        gate[row*256+lane]=g;
    } else if (row<26) {
        input=k+(row-24)*256; gamma=gamma_k;
        output=rotated_k+(row-24)*256;
    } else if (row<30) {
        input=iq+(row-26)*128; gamma=gamma_iq;
        output=rotated_iq+(row-26)*128;
    } else if (row==30) {
        input=n==1 ? raw : first_raw; gamma=gamma_ik; output=dead;
    } else {
        gamma=gamma_ik; output=new_pool; position=n-4;
    }

    __shared__ float sums[256], normalized[256];
    float x=0.0f;
    if (lane<width) {
        if (row==31) {
            // Never reassociate this raw-key sum or normalize its terms first.
            float a=checked(partial[lane],status);
            float b=checked(partial[128+lane],status);
            float c=checked(partial[256+lane],status);
            float d=checked(raw[lane],status);
            float ab=checked(__fadd_rn(a,b),status);
            float abc=checked(__fadd_rn(ab,c),status);
            float abcd=checked(__fadd_rn(abc,d),status);
            x=checked(__fmul_rn(abcd,0.25f),status);
        } else {
            x=checked(input[lane],status);
        }
    }
    sums[lane]=checked(__fmul_rn(x,x),status);
    __syncthreads();
    for (int offset=128;offset;offset/=2) {
        if (lane<offset)
            sums[lane]=checked(__fadd_rn(sums[lane],sums[lane+offset]),status);
        __syncthreads();
    }
    if (lane<width) {
        float mean=__fdiv_rn(sums[0],float(width));
        float denominator=checked(__fadd_rn(mean,epsilon),status);
        float inv=checked(__fdiv_rn(1.0f,__fsqrt_rn(denominator)),status);
        float weight=checked(gamma[lane],status);
        normalized[lane]=checked(__fmul_rn(__fmul_rn(x,inv),weight),status);
    }
    __syncthreads();
    if (lane<width) {
        float value=normalized[lane];
        if (row!=30 && lane<64) {
            const int i=lane%32;
            float frequency=checked(powf(1.0e7f,-float(i)/32.0f),status);
            float angle=checked(__fmul_rn(__ull2float_rn(position),frequency),status);
            float cosine=checked(cosf(angle),status);
            float sine=checked(sinf(angle),status);
            float a=normalized[i], b=normalized[i+32];
            value=lane<32
                ? __fsub_rn(__fmul_rn(a,cosine),__fmul_rn(b,sine))
                : __fadd_rn(__fmul_rn(a,sine),__fmul_rn(b,cosine));
        }
        output[lane]=checked(value,status);
    }
}
}

cudaError_t ml_qsa_frontend_launch(cudaStream_t stream, const uint64_t* n,
    const float* qg, const float* k, const float* raw, const float* iq,
    const float* first_raw, const float* partial,
    const float* gamma_q, const float* gamma_k,
    const float* gamma_iq, const float* gamma_ik, float epsilon,
    float* q, float* rotated_k, float* gate, float* rotated_iq, float* dead,
    int* new_pool_ready, float* new_pool, int* status) {
    if (!n || !qg || !k || !raw || !iq || !first_raw || !partial ||
        !gamma_q || !gamma_k || !gamma_iq || !gamma_ik ||
        !q || !rotated_k || !gate || !rotated_iq || !dead ||
        !new_pool_ready || !new_pool || !status ||
        !std::isfinite(epsilon) || epsilon<=0.0f) return cudaErrorInvalidValue;
    frontend<<<32,256,0,stream>>>(n,qg,k,raw,iq,first_raw,partial,
        gamma_q,gamma_k,gamma_iq,gamma_ik,epsilon,q,rotated_k,gate,rotated_iq,
        dead,new_pool_ready,new_pool,status);
    return cudaGetLastError();
}
