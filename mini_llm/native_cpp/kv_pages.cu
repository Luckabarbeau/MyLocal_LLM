#include "kv_pages.h"
#include "qsa_attention.h"
#include <cuda_runtime.h>
namespace {
__global__ void gather(const uint64_t* n, const int64_t* ids,
    const uint64_t* table, uint64_t page_size, uint64_t physical_pages,
    const float* pages, const float* k, const float* v, float* scratch,
    uint64_t* compact_n, int64_t* compact_ids, int* status) {
    const uint64_t width=*n<2051 ? *n : 2051;
    const uint64_t cell=blockIdx.x;
    if (cell>=width) return;
    if (!threadIdx.x) {
        if (!cell) *compact_n=width;
        compact_ids[cell]=cell;
    }
    const int64_t token=ids[cell];
    bool valid=token>=0 && static_cast<uint64_t>(token)<*n;
    uint64_t slot=0;
    const bool current=valid && static_cast<uint64_t>(token)==*n-1;
    if (valid && !current) {
        slot=table[token/page_size];
        valid=slot && slot<=physical_pages;
    }
    if (!valid && !threadIdx.x) atomicExch(status,1);
    for (int d=threadIdx.x; d<512; d+=blockDim.x) {
        uint64_t offset=valid && !current ? ((slot-1)*page_size+token%page_size)*1024 : 0;
        scratch[cell*512+d]=!valid ? 0.0f : current ? k[d] : pages[offset+d];
        scratch[2051*512+cell*512+d]=!valid ? 0.0f : current ? v[d] : pages[offset+512+d];
    }
}
}
cudaError_t ml_kv_selected_launch(cudaStream_t stream, const uint64_t* n,
    const int64_t* ids, uint64_t active_n, const uint64_t* table, uint64_t page_size,
    uint64_t physical_pages, const float* pages, const float* q, const float* gate,
    const float* k, const float* v, float* scratch, uint64_t* compact_n,
    int64_t* compact_ids, float* contexts, float* gated, int* status, float scale) {
    gather<<<2051,256,0,stream>>>(n,ids,table,page_size,physical_pages,pages,k,v,
        scratch,compact_n,compact_ids,status);
    auto error=cudaGetLastError();
    if (error!=cudaSuccess) return error;
    // The last compact row acts as the current operand, even when the original
    // current token is not selected. No synthetic row is added to the softmax.
    const uint64_t width=active_n<2051 ? active_n : 2051;
    return ml_qsa_attention_launch(stream,compact_n,compact_ids,q,gate,
        scratch+(width-1)*512,scratch+2051*512+(width-1)*512,
        scratch,scratch+2051*512,contexts,gated,status,scale);
}
