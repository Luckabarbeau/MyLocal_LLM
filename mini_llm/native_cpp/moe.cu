#include "moe.h"
#include "qsa_projection.h"
// Keep CUDA overloads at global scope; otherwise the generated decoder's half
// shuffle overload hides the global FP32 overload and rounds reductions to FP16.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
#include "moe_formats.cuh"

__host__ __device__ bool layout(unsigned type, int& qk, int& qb, int& format) {
    qk=256; format=-1;
    switch (type) {
        case 22: qb=82; format=0; break;
        case 21: qb=110; format=1; break;
        case 18: qb=98; format=2; break;
        case 20: qk=32; qb=18; format=3; break;
        case 23: qb=136; format=4; break;
        case 42: qk=64; qb=18; format=5; break;
        case 8: qk=32; qb=34; break;
        default: return false;
    }
    return true;
}
__host__ __device__ bool matrix_ok(MLMoeMatrix m, int columns, bool down) {
    int qk,qb,format;
    if (!layout(m.type,qk,qb,format)) return false;
    bool allowed=down ? (m.type==20 || m.type==42 || m.type==8)
                      : (m.type==18 || m.type==21 || m.type==22 || m.type==23);
    return allowed && m.data && !m.reserved && columns%qk==0 &&
        m.row_bytes>=uint64_t(columns/qk)*qb && m.row_bytes<=uint64_t(1)<<30;
}
__device__ float weight_at(MLMoeMatrix m, int row, int col) {
    int qk,qb,format;
    layout(m.type,qk,qb,format);
    auto p=static_cast<const unsigned char*>(m.data)+uint64_t(row)*m.row_bytes;
    if (m.type==8) {
        // quantized_weight.py::dequantize_reference Q8_0, no activation quantization.
        p+=(col/32)*34;
        return __half2float(__ushort_as_half(u16(p))) *
            float(static_cast<signed char>(p[2+col%32]));
    }
    return lane_weight(p,col,format,qk,qb);
}
__device__ float checked(float v, int* status) {
    if (!isfinite(v)) atomicOr(status,1);
    return v;
}
__device__ float row_sum(float v) {
    // iq_grouped_warp_gpu.py row_sum: exact same warp tree.
    for (int step=16;step;step>>=1)
        v=__fadd_rn(v,__shfl_down_sync(0xffffffffu,v,step));
    return v;
}
__global__ void route(const float* logits, float scale, float* probs,
                      int* ids, float* weights, int* status) {
    // Serial bounded control oracle. All 512 probabilities are materialized;
    // deliberately not a selected-logit softmax shortcut.
    float maximum=-INFINITY;
    for (int e=0;e<512;++e) maximum=fmaxf(maximum,checked(logits[e],status));
    float total=0;
    for (int e=0;e<512;++e) {
        probs[e]=expf(logits[e]-maximum);
        total=__fadd_rn(total,probs[e]);
    }
    for (int e=0;e<512;++e) probs[e]=checked(probs[e]/total,status);
    int chosen[10];
    float selected=0;
    for (int k=0;k<10;++k) {
        int best=-1;
        for (int e=0;e<512;++e) {
            bool used=false;
            for (int j=0;j<k;++j) used|=chosen[j]==e;
            if (!used && (best<0 || logits[e]>logits[best])) best=e;
        }
        chosen[k]=best;
        selected=__fadd_rn(selected,probs[best]);
    }
    // Normalize in selection order, then sort descriptors/weights by ID.
    float denominator=fmaxf(selected,0x1p-14f);
    for (int k=0;k<10;++k) {
        ids[k]=chosen[k];
        weights[k]=checked(__fmul_rn(probs[chosen[k]]/denominator,scale),status);
    }
    for (int k=1;k<10;++k) {
        int id=ids[k]; float w=weights[k]; int j=k;
        while (j && ids[j-1]>id) { ids[j]=ids[j-1]; weights[j]=weights[j-1]; --j; }
        ids[j]=id; weights[j]=w;
    }
}
__device__ bool route_ok(const int* ids, int slot, int* status) {
    if (ids[slot]<0 || ids[slot]>=512 || (slot && ids[slot-1]>=ids[slot])) {
        atomicOr(status,2); return false;
    }
    return true;
}
__global__ void gate_up(int d, int f, const MLMoeExpert* experts,
    const int* ids, const float* x, float* hidden, int* status, const int* resident=nullptr) {
    int pair=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32;
    if (pair>=10*f) return;
    int slot=pair/f, row=pair%f;
    if (!route_ok(ids,slot,status)) { if (!lane) hidden[pair]=0; return; }
    if (resident && !resident[ids[slot]]) return;
        MLMoeExpert e=experts[ids[slot]];
    if (!matrix_ok(e.gate,d,false) || !matrix_ok(e.up,d,false)) {
        atomicOr(status,2); if (!lane) hidden[pair]=0; return;
    }
    float g=0,u=0;
    for (int col=lane;col<d;col+=32) {
        float v=checked(x[col],status);
        g=__fadd_rn(g,__fmul_rn(v,checked(weight_at(e.gate,row,col),status)));
        u=__fadd_rn(u,__fmul_rn(v,checked(weight_at(e.up,row,col),status)));
    }
    g=row_sum(g); u=row_sum(u);
    if (!lane) hidden[pair]=checked(__fmul_rn(g/(1.0f+expf(-g)),u),status);
}
__global__ void down(int d, int f, const MLMoeExpert* experts,
    const int* ids, const float* weights, MLMoeMatrix shared,
    const float* hidden, float* terms, int* status, const int* resident=nullptr) {
    int pair=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32;
    int count=experts?10:1;
    if (pair>=count*d) return;
    int slot=pair/d, row=pair%d;
    if (experts && !route_ok(ids,slot,status)) { if (!lane) terms[pair]=0; return; }
    if (experts && resident && !resident[ids[slot]]) return;
        MLMoeMatrix m=experts?experts[ids[slot]].down:shared;
    bool allowed=experts?(m.type==20 || m.type==42):(m.type==20 || m.type==8);
    if (!allowed || !matrix_ok(m,f,true)) {
        atomicOr(status,2); if (!lane) terms[pair]=0; return;
    }
    float v=0;
    for (int col=lane;col<f;col+=32)
        v=__fadd_rn(v,__fmul_rn(checked(hidden[slot*f+col],status),
                              checked(weight_at(m,row,col),status)));
    v=row_sum(v);
    if (!lane) terms[pair]=checked(experts?__fmul_rn(v,checked(weights[slot],status)):v,status);
}
__global__ void ordered_sum(int d, const float* terms, float* out, int* status) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<d) {
        float v=0;
        for (int slot=0;slot<10;++slot) v=__fadd_rn(v,terms[slot*d+row]);
        out[row]=checked(v,status);
    }
}
__global__ void shared_hidden(int f, const float* gate, const float* up,
                               float* hidden, int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<f) {
        // Qwen4ExpMoE._compute_shared: (gate * sigmoid(gate)) * up.
        float g=checked(gate[i],status), u=checked(up[i],status);
        hidden[i]=checked(__fmul_rn(__fmul_rn(g,1.0f/(1.0f+expf(-g))),u),status);
    }
}
__global__ void shared_output(int d, const float* raw, const float* score,
                               float* output, int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<d) output[i]=checked(__fmul_rn(1.0f/(1.0f+expf(-checked(score[0],status))),raw[i]),status);
}
__global__ void combine(int d,const float* routed,const float* shared,float* out,int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<d) out[i]=checked(__fadd_rn(routed[i],shared[i]),status);
}
__global__ void cpu_terms(int d, const int* ids, const int* resident,
    const float* weights, const float* rows, float* terms, int* status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<10*d && !resident[ids[i/d]])
        terms[i]=checked(__fmul_rn(rows[i],weights[i/d]),status);
}
bool dimensions(int d,int f) { return d>=256 && d<=6144 && f>=256 && f<=12288 && d%256==0 && f%64==0; }
bool shared_gu(MLMoeMatrix m,int d) {
    int bytes=m.type==12?144:m.type==13?176:m.type==14?210:m.type==23?136:0;
    return bytes && m.data && !m.reserved && m.row_bytes==uint64_t(d/256)*bytes;
}
}

cudaError_t ml_moe_route_launch(cudaStream_t stream,const float* logits,float scale,
    float* probabilities,int* ids,float* weights,int* status) {
    if (!logits || !probabilities || !ids || !weights || !status || !std::isfinite(scale) || scale<0)
        return cudaErrorInvalidValue;
    route<<<1,1,0,stream>>>(logits,scale,probabilities,ids,weights,status);
    return cudaGetLastError();
}
cudaError_t ml_moe_experts_launch(cudaStream_t stream,int d,int f,
    const MLMoeExpert* experts,const int* ids,const float* weights,const float* x,
    float* hidden,float* terms,float* output,int* status) {
    if (!dimensions(d,f) || !experts || !ids || !weights || !x || !hidden || !terms || !output || !status)
        return cudaErrorInvalidValue;
    gate_up<<<(10*f+3)/4,128,0,stream>>>(d,f,experts,ids,x,hidden,status);
    cudaError_t err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    down<<<(10*d+3)/4,128,0,stream>>>(d,f,experts,ids,weights,MLMoeMatrix{},hidden,terms,status);
    err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    ordered_sum<<<(d+255)/256,256,0,stream>>>(d,terms,output,status);
    return cudaGetLastError();
}
cudaError_t ml_moe_resident_launch(cudaStream_t stream,int d,int f,
    const MLMoeExpert* experts,const int* resident,const int* ids,const float* weights,
    const float* x,float* hidden,float* terms,int* status) {
    if (!dimensions(d,f) || !experts || !resident || !ids || !weights || !x || !hidden || !terms || !status)
        return cudaErrorInvalidValue;
    gate_up<<<(10*f+3)/4,128,0,stream>>>(d,f,experts,ids,x,hidden,status,resident);
    auto err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    down<<<(10*d+3)/4,128,0,stream>>>(d,f,experts,ids,weights,MLMoeMatrix{},hidden,terms,status,resident);
    return cudaGetLastError();
}
cudaError_t ml_moe_finish_launch(cudaStream_t stream,int d,const int* ids,
    const int* resident,const float* weights,const float* rows,float* terms,
    float* routed,const float* shared,float* output,int* status) {
    if (d<256 || d>6144 || !ids || !resident || !weights || !rows || !terms || !routed || !shared || !output || !status)
        return cudaErrorInvalidValue;
    cpu_terms<<<(10*d+255)/256,256,0,stream>>>(d,ids,resident,weights,rows,terms,status);
    auto err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    ordered_sum<<<(d+255)/256,256,0,stream>>>(d,terms,routed,status);
    err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    return ml_moe_combine_launch(stream,d,routed,shared,output,status);
}
cudaError_t ml_moe_shared_launch(cudaStream_t stream,int d,int f,MLMoeExpert w,
    const float* scalar_gate,const float* x,float* gate,float* up,float* hidden,
    float* score,float* raw,float* output,int* status) {
    if (!dimensions(d,f) || !shared_gu(w.gate,d) || !shared_gu(w.up,d) ||
        !matrix_ok(w.down,f,true) || (w.down.type!=20 && w.down.type!=8) ||
        !scalar_gate || !x || !gate || !up || !hidden || !score || !raw || !output || !status)
        return cudaErrorInvalidValue;
    cudaError_t err=ml_qsa_projection_launch(stream,w.gate.data,w.gate.type,f,d,w.gate.row_bytes,x,gate,status);
    if (err!=cudaSuccess) return err;
    err=ml_qsa_projection_launch(stream,w.up.data,w.up.type,f,d,w.up.row_bytes,x,up,status);
    if (err!=cudaSuccess) return err;
    err=ml_qsa_projection_launch(stream,scalar_gate,0,1,d,uint64_t(d)*4,x,score,status);
    if (err!=cudaSuccess) return err;
    shared_hidden<<<(f+255)/256,256,0,stream>>>(f,gate,up,hidden,status);
    err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    down<<<(d+3)/4,128,0,stream>>>(d,f,nullptr,nullptr,nullptr,w.down,hidden,raw,status);
    err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    shared_output<<<(d+255)/256,256,0,stream>>>(d,raw,score,output,status);
    return cudaGetLastError();
}
cudaError_t ml_moe_combine_launch(cudaStream_t stream,int d,const float* routed,
    const float* shared,float* output,int* status) {
    if (d<1 || d>6144 || !routed || !shared || !output || !status) return cudaErrorInvalidValue;
    combine<<<(d+255)/256,256,0,stream>>>(d,routed,shared,output,status);
    return cudaGetLastError();
}
