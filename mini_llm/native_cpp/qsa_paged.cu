#include "qsa_paged.h"
#include "kv_streamed.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace {
__global__ void pack(const float* k,const float* v,uint64_t rows,float* out) {
    const uint64_t row=blockIdx.x;
    for (int d=threadIdx.x;d<512;d+=blockDim.x) {
        out[row*1024+d]=k[row*512+d]; out[row*1024+512+d]=v[row*512+d];
    }
}
__device__ float decode(const unsigned char* row,int d,bool quantized) {
    if (!quantized) return reinterpret_cast<const float*>(row)[d];
    const unsigned char* group=row+(d/64)*66;
    return float(reinterpret_cast<const int8_t*>(group)[d%64]) *
        __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(group+64)));
}
__global__ void gather(const uint64_t* n,const int64_t* ids,uint64_t logical,
    uint64_t page_size,bool quantized,const unsigned char* source,float* scratch,
    uint64_t* compact_n,int64_t* compact_ids) {
    const uint64_t cell=blockIdx.x,width=*n<2051?*n:2051;
    if (cell>=width) return;
    const uint64_t token=ids[cell];
    if (logical==UINT64_MAX) {
        if (!threadIdx.x) { if (!cell) *compact_n=width; compact_ids[cell]=cell; }
        if (token!=*n-1) return;
    } else if (token==*n-1 || token/page_size!=logical) return;
    const unsigned char* row=source+(logical==UINT64_MAX?0:token%page_size)*(quantized?1056:4096);
    for (int d=threadIdx.x;d<512;d+=blockDim.x) {
        scratch[cell*512+d]=decode(row,d,quantized);
        scratch[2051*512+cell*512+d]=decode(row,512+d,quantized);
    }
}
}
cudaError_t ml_qsa_paged_pack(cudaStream_t stream,const float* k,const float* v,
    uint64_t rows,bool quantized,unsigned char* out,int* status) {
    if (quantized) return ml_kvi8_quantize(stream,k,v,rows,out,status);
    pack<<<rows,256,0,stream>>>(k,v,rows,reinterpret_cast<float*>(out));
    return cudaGetLastError();
}
cudaError_t ml_qsa_paged_gather(cudaStream_t stream,const uint64_t* n,const int64_t* ids,
    uint64_t logical,uint64_t page_size,bool quantized,const unsigned char* source,
    float* scratch,uint64_t* compact_n,int64_t* compact_ids) {
    gather<<<2051,256,0,stream>>>(n,ids,logical,page_size,quantized,source,scratch,compact_n,compact_ids);
    return cudaGetLastError();
}
