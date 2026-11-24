#include "qsa_selection.h"
#include <math.h>

namespace {
// One thread per block score. Products and additions round separately; the dot
// uses a documented left-to-right FP32 reduction, not a BLAS-dependent tree.
__global__ void score_kernel(const uint64_t* parameter, const float* query,
                             const float* pooled, const float* dead,
                             float* scores, int* status) {
    uint64_t n=*parameter, b=blockIdx.x*blockDim.x+threadIdx.x;
    if (b >= (n+3)/4) return;
    const float* key=b<n/4 ? pooled+b*128 : dead;
    float score=0;
    for (int h=0; h<4; ++h) {
        float dot=0;
        for (int j=0; j<128; ++j) {
            float q=query[h*128+j], k=key[j];
            if (!isfinite(q) || !isfinite(k)) atomicExch(status,1);
            dot=__fadd_rn(dot,__fmul_rn(q,k));
        }
        score=__fadd_rn(score,fmaxf(dot,0.0f));
        if (!isfinite(dot) || !isfinite(score)) atomicExch(status,1);
    }
    if (b==n/4) score=__fadd_rn(score,1e9f);
    for (int j=0; j<4 && b*4+j<n; ++j) scores[b*4+j]=score;
}
__device__ bool better(int64_t a, int64_t b, const float* scores, int policy) {
    return scores[a]>scores[b] || (policy==1 && scores[a]==scores[b] && a<b);
}
// Exact value-only __adjust_heap transcription for ggml_cpu. For strata the
// comparator additionally defines lower IDs as better, making membership total.
__device__ void adjust(int hole, int64_t value, int width, int64_t* ids,
                       const float* scores, int policy) {
    int top=hole, child=hole;
    while (child<(width-1)/2) {
        child=2*(child+1);
        if (better(ids[child],ids[child-1],scores,policy)) --child;
        ids[hole]=ids[child]; hole=child;
    }
    if (width%2==0 && child==(width-2)/2) {
        child=2*(child+1); ids[hole]=ids[child-1]; hole=child-1;
    }
    int parent=(hole-1)/2;
    while (hole>top && better(ids[parent],value,scores,policy)) {
        ids[hole]=ids[parent]; hole=parent; parent=(hole-1)/2;
    }
    ids[hole]=value;
}
__global__ void select_kernel(const uint64_t* parameter, const float* scores,
                              int64_t* ids, const int* status, int policy) {
    if (*status) return;
    uint64_t n=*parameter;
    int width=n<2051 ? static_cast<int>(n) : 2051;
    for (int i=0; i<width; ++i) ids[i]=i;
    if (width>1) {
        for (int parent=(width-2)/2; parent>=0; --parent)
            adjust(parent,ids[parent],width,ids,scores,policy);
        for (uint64_t i=width; i<n; ++i)
            if (better(i,ids[0],scores,policy)) adjust(0,i,width,ids,scores,policy);
    }
    // In-place heapsort by integer ID, independent of the membership comparator.
    for (int start=width/2-1; start>=0; --start) {
        int root=start;
        while (root*2+1<width) {
            int child=root*2+1;
            if (child+1<width && ids[child]<ids[child+1]) ++child;
            if (ids[root]>=ids[child]) break;
            int64_t tmp=ids[root]; ids[root]=ids[child]; ids[child]=tmp; root=child;
        }
    }
    for (int end=width-1; end>0; --end) {
        int64_t tmp=ids[0]; ids[0]=ids[end]; ids[end]=tmp;
        int root=0;
        while (root*2+1<end) {
            int child=root*2+1;
            if (child+1<end && ids[child]<ids[child+1]) ++child;
            if (ids[root]>=ids[child]) break;
            tmp=ids[root]; ids[root]=ids[child]; ids[child]=tmp; root=child;
        }
    }
}
}
cudaError_t ml_qsa_launch(cudaStream_t stream, uint64_t capacity, const uint64_t* n,
                         const float* query, const float* pooled, const float* dead,
                         float* scores, int64_t* ids, int* status, int policy, bool clear_status) {
    auto result=clear_status ? cudaMemsetAsync(status,0,sizeof(int),stream) : cudaSuccess;
    if (result!=cudaSuccess) return result;
    score_kernel<<<static_cast<unsigned>((capacity/4+63)/64),64,0,stream>>>(n,query,pooled,dead,scores,status);
    result=cudaGetLastError();
    if (result!=cudaSuccess) return result;
    select_kernel<<<1,1,0,stream>>>(n,scores,ids,status,policy);
    return cudaGetLastError();
}
