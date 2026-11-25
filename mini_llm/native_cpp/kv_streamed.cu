/*
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Quantization arithmetic from Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf,
// src/prefill/kernels.cu:2552-2564; local port native_qsa_prefill.cu:410-422.
// Packed layout and runtime paging are local; this is not Strata MMA attention.
#include "kv_streamed.h"
#include "qsa_attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace {
__global__ void quantize(const float* k, const float* v, uint64_t rows,
                         unsigned char* packed, int* status) {
    const uint64_t row=blockIdx.x;
    const int group=blockIdx.y, lane=threadIdx.x;
    if (row>=rows) return;
    const float x=(group<8?k:v)[row*512+(group%8)*64+lane];
    if (!isfinite(x)) atomicExch(status,1);
    float a=fabsf(x);
    for (int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,o));
    __shared__ float wm[2];
    if (!(lane&31)) wm[lane>>5]=a;
    __syncthreads();
    const float amax=fmaxf(wm[0],wm[1]);
    const uint16_t sb=__half_as_ushort(__float2half_rn(amax/127.0f));
    const float sf=__half2float(__ushort_as_half(sb));
    if (!isfinite(sf)) atomicExch(status,1);
    int q=0;
    if (sf>0.0f) { q=__float2int_rn(x/sf); q=q < -127 ? -127 : (q>127 ? 127 : q); }
    unsigned char* block=packed+row*ML_KV_I8_ROW+group*66;
    reinterpret_cast<int8_t*>(block)[lane]=static_cast<int8_t>(q);
    if (!lane) *reinterpret_cast<uint16_t*>(block+64)=sb;
}
__global__ void requests(const uint64_t* n, const int64_t* ids,
    const uint64_t* table, uint64_t page_size, uint64_t slots, KVPageRequests* out) {
    if (threadIdx.x || blockIdx.x) return;
    const uint64_t width=*n<2051?*n:2051;
    out->count=out->misses=out->error=0;
    int64_t previous=-1;
    for (uint64_t i=0;i<width;++i) {
        const int64_t token=ids[i];
        if (token<=previous || token<0 || static_cast<uint64_t>(token)>=*n) { out->error=1; return; }
        previous=token;
        if (static_cast<uint64_t>(token)==*n-1) continue;
        const uint64_t page=token/page_size;
        if (out->count && out->pages[out->count-1]==page) continue;
        out->pages[out->count++]=page;
        const uint64_t entry=table[page];
        if (entry>slots) { out->error=1; return; }
        if (!entry) out->missing[out->misses++]=page;
    }
}
__device__ float decode(const unsigned char* row, int component) {
    const unsigned char* group=row+(component/64)*66;
    return static_cast<float>(reinterpret_cast<const int8_t*>(group)[component%64]) *
        __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(group+64)));
}
__global__ void gather(const uint64_t* n, const int64_t* ids, const uint64_t* table,
    uint64_t page_size, uint64_t slots, const unsigned char* pages,
    const unsigned char* current, float* scratch, uint64_t* compact_n,
    int64_t* compact_ids, int* status) {
    const uint64_t width=*n<2051?*n:2051, cell=blockIdx.x;
    if (cell>=width) return;
    if (!threadIdx.x) { if (!cell) *compact_n=width; compact_ids[cell]=cell; }
    const int64_t token=ids[cell];
    bool valid=token>=0 && static_cast<uint64_t>(token)<*n;
    const unsigned char* source=current;
    if (valid && static_cast<uint64_t>(token)!=*n-1) {
        const uint64_t entry=table[token/page_size];
        valid=entry && entry<=slots;
        if (valid) source=pages+((entry-1)*page_size+token%page_size)*ML_KV_I8_ROW;
    }
    if (!valid && !threadIdx.x) atomicExch(status,1);
    for (int d=threadIdx.x;d<512;d+=blockDim.x) {
        scratch[cell*512+d]=valid?decode(source,d):0.0f;
        scratch[2051*512+cell*512+d]=valid?decode(source,512+d):0.0f;
    }
}
}
cudaError_t ml_kvi8_quantize(cudaStream_t stream, const float* k, const float* v,
    uint64_t rows, unsigned char* packed, int* status) {
    quantize<<<dim3(rows,16),64,0,stream>>>(k,v,rows,packed,status);
    return cudaGetLastError();
}
cudaError_t ml_kvi8_requests(cudaStream_t stream, const uint64_t* n, const int64_t* ids,
    const uint64_t* table, uint64_t page_size, uint64_t slots, KVPageRequests* out) {
    requests<<<1,1,0,stream>>>(n,ids,table,page_size,slots,out);
    return cudaGetLastError();
}
cudaError_t ml_kvi8_attention_launch(cudaStream_t stream, const uint64_t* n, const int64_t* ids,
    uint64_t active_n, const uint64_t* table, uint64_t page_size, uint64_t slots,
    const unsigned char* pages, const unsigned char* current, const float* q, const float* gate,
    float* scratch, uint64_t* compact_n, int64_t* compact_ids, float* contexts,
    float* gated, int* status, float scale) {
    gather<<<2051,256,0,stream>>>(n,ids,table,page_size,slots,pages,current,scratch,
        compact_n,compact_ids,status);
    auto error=cudaGetLastError(); if (error!=cudaSuccess) return error;
    const uint64_t width=active_n<2051?active_n:2051;
    return ml_qsa_attention_launch(stream,compact_n,compact_ids,q,gate,
        scratch+(width-1)*512,scratch+2051*512+(width-1)*512,
        scratch,scratch+2051*512,contexts,gated,status,scale);
}
