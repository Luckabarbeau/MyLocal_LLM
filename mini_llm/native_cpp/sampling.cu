#include "sampling.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <cmath>

size_t ml_sampling_temp_bytes(int n) {
    // Runtime ABI calls report their CUDA failures immediately. CUB must not
    // inherit an already-reported last-error slot from an earlier invalid call.
    (void)cudaGetLastError();
    size_t bytes=0;
    auto err=cub::DeviceRadixSort::SortPairs(nullptr,bytes,
        static_cast<float*>(nullptr),static_cast<float*>(nullptr),
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),n);
    if(err!=cudaSuccess) return 0;
    size_t reduce_bytes=0;
    err=cub::DeviceReduce::ArgMax(nullptr,reduce_bytes,static_cast<float*>(nullptr),
        static_cast<cub::KeyValuePair<int,float>*>(nullptr),n);
    // Greedy reuses this allocation: a 256-byte aligned result header followed
    // by reduction scratch. Sorting and reduction never run concurrently.
    return err==cudaSuccess ? std::max(bytes,size_t(256)+reduce_bytes) : 0;
}
namespace {
__global__ void prepare(const float* x,int n,int k,float t,double p,float* keys,int* ids,ml_sample_result* r) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float v=x[i];
    if(!isfinite(v)) { atomicExch(&r->status,1); v=0; }
    // CuPy argpartition is a stable full ascending sort. For full-vocab
    // nucleus there is no partition: stable argsort acts on scaled values.
    float key=(t>0 && k==n && p<1) ? v/t : v;
    keys[i]=key==0 ? 0.0f : key; ids[i]=i;
}
__global__ void exponentiate(const float* x,int n,int k,float t,double p,
    const int* sorted_ids,int* ids,float* weights,ml_sample_result* r) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=k || r->status) return;
    int id=p<1 ? sorted_ids[n-1-i] : k==n ? i : sorted_ids[n-k+i];
    ids[i]=id;
    float v=x[id]/t, maximum=x[sorted_ids[n-1]]/t;
    if(!isfinite(v) || !isfinite(maximum)) atomicExch(&r->status,2);
    weights[i]=expf(v-maximum);
}
__device__ uint64_t splitmix(uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
__global__ void finish_greedy(const cub::KeyValuePair<int,float>* maximum,
    uint64_t seed,uint64_t counter,int supplied,double uniform,ml_sample_result* r) {
    if(r->status) return;
    r->uniform=supplied ? uniform : double(splitmix(seed+counter)>>11)*0x1.0p-53;
    r->token=maximum->key; r->kept=1; r->total=1; r->probability=1;
    r->cdf_low=0; r->cdf_high=1;
}
__global__ void finish(int k,double p,uint64_t seed,uint64_t counter,
    int supplied,double uniform,const int* ids,float* weights,float* cdf,ml_sample_result* r) {
    if(r->status) return;
    double u=supplied ? uniform : double(splitmix(seed+counter)>>11)*0x1.0p-53;
    r->uniform=u;

    float sum=0;
    // Preserve the exact serial FP32 addition order. The omitted n-k tail was
    // all +0, so it cannot change the positive cumulative mass.
    for(int i=0;i<k;++i) { sum+=weights[i]; cdf[i]=sum; }
    float threshold=float(p)*sum;
    if(p<1) for(int i=0;i<k;++i)
        if(!(cdf[i]-weights[i]<threshold)) weights[i]=0;
    sum=0; int kept=0;
    for(int i=0;i<k;++i) { sum+=weights[i]; cdf[i]=sum; kept+=weights[i]>0; }
    if(!isfinite(sum) || !(sum>0)) { r->status=3; return; }
    // Right-sided search, double uniform times FP32 total, like sample_token.
    double target=u*double(sum);
    int lo=0,hi=k;
    while(lo<hi) { int m=lo+(hi-lo)/2; if(double(cdf[m])<=target) lo=m+1; else hi=m; }
    int slot=lo<k ? lo : k-1;
    r->token=ids[slot]; r->kept=kept; r->total=sum;
    r->probability=weights[slot]/sum;
    r->cdf_low=(slot ? cdf[slot-1] : 0)/sum; r->cdf_high=cdf[slot]/sum;
}
__global__ void normalize(int n,int k,int* ids,float* weights,float* cdf,const ml_sample_result* r) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || r->status) return;
    if(i>=k) { ids[i]=-1; weights[i]=0; cdf[i]=1; return; }
    weights[i]/=r->total; cdf[i]/=r->total;
}
}
cudaError_t ml_sampling_launch(cudaStream_t s,const float* x,int n,float t,double p,int k,
    uint64_t seed,uint64_t counter,int supplied,double u,float* a,float* b,int* ia,int* ib,
    float* weights,float* cdf,void* temp,size_t bytes,ml_sample_result* result) {
    (void)cudaGetLastError(); // Same per-call error-slot isolation as the query.
    auto err=cudaMemsetAsync(result,0,sizeof(*result),s); if(err!=cudaSuccess) return err;
    prepare<<<(n+255)/256,256,0,s>>>(x,n,k,t,p,a,ia,result);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    if(t==0) {
        auto* maximum=static_cast<cub::KeyValuePair<int,float>*>(temp);
        size_t reduce_bytes=bytes-256;
        err=cub::DeviceReduce::ArgMax(static_cast<char*>(temp)+256,reduce_bytes,a,maximum,n,s);
        if(err!=cudaSuccess) return err;
        // CUB ArgMax chooses the lowest input offset for equal maxima.
        finish_greedy<<<1,1,0,s>>>(maximum,seed,counter,supplied,u,result);
        return cudaGetLastError();
    }
    err=cub::DeviceRadixSort::SortPairs(temp,bytes,a,b,ia,ib,n,0,32,s);
    if(err!=cudaSuccess) return err;
    exponentiate<<<(k+255)/256,256,0,s>>>(x,n,k,t,p,ib,ia,weights,result);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    finish<<<1,1,0,s>>>(k,p,seed,counter,supplied,u,ia,weights,cdf,result);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    normalize<<<(n+255)/256,256,0,s>>>(n,k,ia,weights,cdf,result);
    return cudaGetLastError();
}
