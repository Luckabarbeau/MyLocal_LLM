#include "qsa_projection.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// IQ4_XS equation: mini_llm/ops/iq_packed_gpu.py; table from
// mini_llm/ops/gguf_codebooks.py (llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f).
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
namespace {
__device__ __constant__ signed char iq4_values[16]={
    -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ float half_at(const unsigned char* b) {
    return __half2float(__ushort_as_half(unsigned(b[0]) | (unsigned(b[1])<<8)));
}
// Q4_K/Q5_K/Q6_K equations and bit indexing reused from
// mini_llm/ops/quantized_linear.py::_load_gguf_lanes (formats 2, 4 and 5).
__device__ float decode(const unsigned char* row, unsigned type, int col) {
    if (type==0) return reinterpret_cast<const float*>(row)[col];
    if (type==1) return half_at(row+2*col);
    if (type==20) {
        const unsigned char* b=row+(col/32)*18;
        int lane=col%32;
        return half_at(b)*float(iq4_values[(b[2+lane%16]>>(4*(lane/16)))&15]);
    }
    if (type==30) {
        unsigned bits=unsigned(row[2*col]) | (unsigned(row[2*col+1])<<8);
        return __uint_as_float(bits<<16);
    }
    int within=col%256;
    const unsigned char* b=row+(col/256)*(type==12?144:type==13?176:type==23?136:210);
    if (type==23) {
        int group=within/32, lane=within%32;
        int hi=unsigned(b[2]) | (unsigned(b[3])<<8);
        int scale=((b[4+group/2]>>(4*(group%2)))&15) | (((hi>>(2*group))&3)<<4);
        int v=(b[8+16*group+lane%16]>>(4*(lane/16)))&15;
        return (half_at(b)*float(scale-32))*float(iq4_values[v]);
    }
    if (type==12 || type==13) {
        int group=within/32;
        int sb=b[4+(group<4?group:group+4)];
        int mb=b[4+(group<4?group+4:group+4)];
        int sc=group<4 ? sb&63 : (sb&15)|((b[4+group-4]>>2)&48);
        int mn=group<4 ? mb&63 : (mb>>4)|((b[4+group]>>2)&48);
        int q=b[(type==13?48:16)+(within/64)*32+within%32];
        int high=type==13 ? ((b[16+within%32]>>(within/32))&1)<<4 : 0;
        int v=((q>>((within%64)/32*4))&15)|high;
        return (half_at(b)*float(sc))*float(v)-half_at(b+2)*float(mn);
    }
    int scale=static_cast<signed char>(b[192+within/16]);
    int low=(b[(within/128)*64+within%64]>>((within%128)/64*4))&15;
    int high=(b[128+(within/128)*32+within%32]>>((within%128)/32*2))&3;
    return (half_at(b+208)*float(scale))*float((low|(high<<4))-32);
}
__global__ void decode_rows(const unsigned char* source, unsigned type,
    uint64_t columns, uint64_t rb, uint64_t count, float* out, int* status) {
    for (uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x) {
        float v=decode(source+(i/columns)*rb,type,int(i%columns));
        if (!isfinite(v)) atomicExch(status,1);
        out[i]=v;
    }
}
__global__ void project(const unsigned char* weight, unsigned type,
                        uint64_t columns, uint64_t rb, const float* x, float* out, int* status) {
    __shared__ float sums[128];
    float sum=0;
    const unsigned char* row=weight+uint64_t(blockIdx.x)*rb;
    for (int j=threadIdx.x;j<columns;j+=128) {
        float w=decode(row,type,j), v=x[j];
        if (!isfinite(w) || !isfinite(v)) atomicExch(status,1);
        sum+=w*v;
    }
    sums[threadIdx.x]=sum;
    __syncthreads();
    for (int offset=64;offset;offset/=2) {
        if (threadIdx.x<offset) sums[threadIdx.x]+=sums[threadIdx.x+offset];
        __syncthreads();
    }
    if (!threadIdx.x) {
        if (!isfinite(sums[0])) atomicExch(status,1);
        out[blockIdx.x]=sums[0];
    }
}
}
cudaError_t ml_qsa_decode_rows_launch(cudaStream_t stream, const void* source,
    uint32_t type, uint64_t rows, uint64_t columns, uint64_t rb, float* output, int* status) {
    uint64_t expected=type==20?columns/32*18:columns*(type==0?4:2);
    if (!source || !output || !status || !rows || rows>12288 || !columns || columns>6144 ||
        (type!=0 && type!=1 && type!=20 && type!=30) || (type==20 && columns%32) || rb!=expected)
        return cudaErrorInvalidValue;
    uint64_t count=rows*columns;
    decode_rows<<<unsigned((count+255)/256),256,0,stream>>>(
        static_cast<const unsigned char*>(source),type,columns,rb,count,output,status);
    return cudaGetLastError();
}
cudaError_t ml_qsa_projection_launch(cudaStream_t stream, const void* weight,
    uint32_t type, uint64_t rows, uint64_t columns, uint64_t rb, const float* input,
    float* output, int* status) {
    uint64_t expected=0;
    switch (type) {
        case 0: expected=columns*4; break;
        case 30: expected=columns*2; break;
        case 12: expected=columns/256*144; break;
        case 13: expected=columns/256*176; break;
        case 14: expected=columns/256*210; break;
        case 23: expected=columns/256*136; break;
        default: return cudaErrorInvalidValue;
    }
    if (!weight || !input || !output || !status || !rows || rows>12288 ||
        !columns || columns>6144 || (type!=0 && type!=30 && columns%256) || rb!=expected)
        return cudaErrorInvalidValue;
    project<<<unsigned(rows),128,0,stream>>>(static_cast<const unsigned char*>(weight),
        type,columns,rb,input,output,status);
    return cudaGetLastError();
}
