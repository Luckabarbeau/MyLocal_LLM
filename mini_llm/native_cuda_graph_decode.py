"""Opt-in CUDA Graph decode for static batch-one native BF16 inference."""

from __future__ import annotations

import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype, resolve_dtype
from mini_llm.ops.cublas_fp16_gemm import (
    bf16_matmul_into,
    prepare_fp16_gemm_context,
)
from mini_llm.ops.sparse_decode_attention import HEAD_LOCAL


_MODULE = None


def _truthy(name, default="0"):
    return os.environ.get(name, default).strip().lower() not in {
        "", "0", "false", "off", "no",
    }


def enabled():
    return BACKEND_NAME == "cupy" and _truthy(
        "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE", "0"
    )


def strict():
    return _truthy("MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT", "0")


_SOURCE = r"""
#include <cuda_bf16.h>

#define REDUCE_RMS(INPUT_EXPR) \
    float sum_sq = 0.0f; \
    for (int col = tid; col < d_model; col += (int)blockDim.x) { \
        const float value = (INPUT_EXPR); sum_sq += value * value; \
    } \
    for (int offset = 16; offset > 0; offset >>= 1) \
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
    __shared__ float warp_sums[8]; \
    const int lane_r = tid & 31; const int warp_r = tid >> 5; \
    if (lane_r == 0) warp_sums[warp_r] = sum_sq; __syncthreads(); \
    if (warp_r == 0) { \
        sum_sq = lane_r < 8 ? warp_sums[lane_r] : 0.0f; \
        for (int offset = 16; offset > 0; offset >>= 1) \
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset); \
        if (lane_r == 0) warp_sums[0] = rsqrtf(sum_sq / (float)d_model + eps); \
    } \
    __syncthreads(); const float inv_rms = warp_sums[0]

extern "C" __global__ void embedding_rmsnorm_bf16(
    const int* token, const __nv_bfloat16* embedding,
    const __nv_bfloat16* gamma, float* residual,
    __nv_bfloat16* normalized, int vocab_size, int d_model, float eps)
{
    const int tid = threadIdx.x; int row = token[0];
    if (row < 0 || row >= vocab_size) row = 0;
    const long long base = (long long)row * d_model;
    REDUCE_RMS(__bfloat162float(embedding[base + col]));
    for (int col = tid; col < d_model; col += blockDim.x) {
        float value = __bfloat162float(embedding[base + col]);
        residual[col] = value;
        normalized[col] = __float2bfloat16_rn(
            value * inv_rms * __bfloat162float(gamma[col]));
    }
}

extern "C" __global__ void qkv_rope_store_bf16_dynamic(
    const __nv_bfloat16* packed, const __nv_bfloat16* normalized,
    const __nv_bfloat16* cos_table, const __nv_bfloat16* sin_table,
    __nv_bfloat16* q_out, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache, __nv_bfloat16* router_cache,
    const int* position_ptr, int n_q_heads, int n_kv_heads,
    int cache_capacity, int d_model)
{
    const int tid = threadIdx.x; const int position = position_ptr[0];
    const int q_width = n_q_heads * 64, kv_width = n_kv_heads * 64;
    const long long trig_base = (long long)position * 32LL;
    for (int pair = tid; pair < (q_width >> 1); pair += blockDim.x) {
        int col = pair << 1, ph = pair & 31;
        float c = __bfloat162float(cos_table[trig_base + ph]);
        float s = __bfloat162float(sin_table[trig_base + ph]);
        float a = __bfloat162float(packed[col]);
        float b = __bfloat162float(packed[col + 1]);
        q_out[col] = __float2bfloat16_rn(a*c-b*s);
        q_out[col+1] = __float2bfloat16_rn(a*s+b*c);
    }
    for (int pair = tid; pair < (kv_width >> 1); pair += blockDim.x) {
        int col = pair << 1, kvh = col >> 6, lane = col & 63, ph = pair & 31;
        float c = __bfloat162float(cos_table[trig_base + ph]);
        float s = __bfloat162float(sin_table[trig_base + ph]);
        long long src = q_width + col;
        float a = __bfloat162float(packed[src]);
        float b = __bfloat162float(packed[src + 1]);
        long long dst = ((long long)kvh * cache_capacity + position) * 64LL + lane;
        k_cache[dst] = __float2bfloat16_rn(a*c-b*s);
        k_cache[dst+1] = __float2bfloat16_rn(a*s+b*c);
    }
    for (int col = tid; col < kv_width; col += blockDim.x) {
        int kvh = col >> 6, lane = col & 63;
        long long dst = ((long long)kvh * cache_capacity + position) * 64LL + lane;
        v_cache[dst] = packed[q_width + kv_width + col];
    }
    if (router_cache != nullptr)
        for (int col = tid; col < d_model; col += blockDim.x)
            router_cache[(long long)position * d_model + col] = normalized[col];
}

extern "C" __global__ void local_attention_bf16_dynamic(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, __nv_bfloat16* out,
    const int* kv_head, const int* window, const int* position_ptr,
    int n_q_heads, int n_kv_heads, int cache_capacity, float scale)
{
    const int lane = threadIdx.x; const int h = blockIdx.x;
    if (lane >= 32 || h >= n_q_heads) return;
    const int position = position_ptr[0], kvh = kv_head[h];
    const long long qbase = (long long)h * 64LL;
    float q0=__bfloat162float(q[qbase+lane]);
    float q1=__bfloat162float(q[qbase+lane+32]);
    float acc0=0.0f,acc1=0.0f,rmax=-3.402823466e+38F,rsum=0.0f;
    int seen=0; const unsigned mask=0xffffffffu;
    int first=position-window[h]+1; if(first<0)first=0;
    for(int key=first;key<=position;++key){
        long long base=((long long)kvh*cache_capacity+key)*64LL;
        float dot=q0*__bfloat162float(k_cache[base+lane])+
                  q1*__bfloat162float(k_cache[base+lane+32]);
        dot+=__shfl_down_sync(mask,dot,16);dot+=__shfl_down_sync(mask,dot,8);
        dot+=__shfl_down_sync(mask,dot,4);dot+=__shfl_down_sync(mask,dot,2);
        dot+=__shfl_down_sync(mask,dot,1);
        float score=__shfl_sync(mask,dot,0)*scale,alpha=0.0f,beta=1.0f;
        if(lane==0){if(seen==0){rmax=score;rsum=1.0f;seen=1;}else{
            float nm=rmax>score?rmax:score;alpha=__expf(rmax-nm);
            beta=__expf(score-nm);rsum=rsum*alpha+beta;rmax=nm;++seen;}}
        int s=__shfl_sync(mask,seen,0);
        if(s!=1){alpha=__shfl_sync(mask,alpha,0);beta=__shfl_sync(mask,beta,0);}
        acc0=acc0*alpha+beta*__bfloat162float(v_cache[base+lane]);
        acc1=acc1*alpha+beta*__bfloat162float(v_cache[base+lane+32]);
    }
    float inv=lane==0?1.0f/rsum:0.0f;inv=__shfl_sync(mask,inv,0);
    out[qbase+lane]=__float2bfloat16_rn(acc0*inv);
    out[qbase+lane+32]=__float2bfloat16_rn(acc1*inv);
}

extern "C" __global__ void residual_rmsnorm_bf16(
    const float* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* gamma, float* residual_out,
    __nv_bfloat16* normalized, int d_model, float alpha, float eps)
{
    const int tid=threadIdx.x;
    REDUCE_RMS(residual[col]+alpha*__bfloat162float(branch[col]));
    for(int col=tid;col<d_model;col+=blockDim.x){
        float value=residual[col]+alpha*__bfloat162float(branch[col]);
        residual_out[col]=value;
        normalized[col]=__float2bfloat16_rn(
            value*inv_rms*__bfloat162float(gamma[col]));
    }
}

extern "C" __global__ void swiglu_bf16(
    const __nv_bfloat16* gate,const __nv_bfloat16* up,
    __nv_bfloat16* hidden,int count)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float g=__bfloat162float(gate[i]),u=__bfloat162float(up[i]);
    float s=1.0f/(1.0f+expf(-g));hidden[i]=__float2bfloat16_rn((g*s)*u);
}
"""


def _module():
    global _MODULE
    if _MODULE is None:
        names=("embedding_rmsnorm_bf16","qkv_rope_store_bf16_dynamic",
               "local_attention_bf16_dynamic","residual_rmsnorm_bf16",
               "swiglu_bf16")
        module=xp.RawModule(code=_SOURCE,options=("--std=c++14",),name_expressions=names)
        for name in names: module.get_function(name)
        _MODULE=module
    return _MODULE


class NativeBF16CudaGraphDecoder:
    def __init__(self, engine, state):
        if BACKEND_NAME!="cupy" or not is_bfloat16_dtype(resolve_dtype(engine.dtype)):
            raise ValueError("native CUDA Graph decode requires CuPy BF16")
        if state.batch_size!=1 or state.long_memory_enabled or not state.has_capacity(1):
            raise ValueError("native CUDA Graph decode requires batch-one linear cache with capacity")
        if not engine.sparse_attention or any(block.d_head!=64 for block in engine.blocks):
            raise ValueError("native CUDA Graph decode requires static sparse d_head=64")
        if any(block.attention._retrieval_groups for block in engine.blocks):
            raise ValueError("native CUDA Graph decode does not yet support retrieval groups")
        if any(block.attention.rope_layout!="interleaved" for block in engine.blocks):
            raise ValueError("native CUDA Graph decode requires interleaved RoPE")
        for block in engine.blocks:
            kinds=block.attention._fused_head_kind
            if kinds is None or not bool(xp.all(kinds==HEAD_LOCAL).item()):
                raise ValueError("native CUDA Graph decode currently supports local heads only")
            if block.moe.expert.W_gate_up is not None:
                raise ValueError("native BF16 graph preserves separate gate/up GEMMs")
        self.engine,self.state=engine,state;self.module=_module()
        self.d_model=int(engine.d_model);self.vocab=int(engine.vocab_size)
        self.n_q=int(engine.blocks[0].n_q_heads);self.n_kv=int(engine.n_kv_heads)
        self.d_ff=int(engine.blocks[0].d_ff);self.capacity=int(state.k_cache.shape[3])
        self.bf=resolve_dtype(engine.dtype);self.stream=xp.cuda.Stream(non_blocking=True)
        self.ctx=prepare_fp16_gemm_context(self.stream)
        self.token=xp.zeros((1,),dtype=xp.int32);self.position=xp.asarray([state.length],dtype=xp.int32)
        self.pos_host=np.asarray([state.length],dtype=np.int32)
        self.ra=xp.empty((1,self.d_model),dtype=xp.float32);self.rb=xp.empty_like(self.ra)
        self.norm=xp.empty((1,self.d_model),dtype=self.bf);self.qkv=xp.empty((1,self.d_model+2*self.n_kv*64),dtype=self.bf)
        self.q=xp.empty((1,self.n_q*64),dtype=self.bf);self.context=xp.empty_like(self.q)
        self.gate=xp.empty((1,self.d_ff),dtype=self.bf);self.up=xp.empty_like(self.gate);self.hidden=xp.empty_like(self.gate)
        self.logits=xp.empty((1,self.vocab),dtype=self.bf);self.embedding=engine.embedding.W.data;self.lm=engine.lm_head
        self.layers=[]
        for i,b in enumerate(engine.blocks):
            a=b.attention;a.ensure_rope_capacity(self.capacity);e=b.moe.expert
            self.layers.append(dict(norm1=b.norm1.gamma,norm2=b.norm2.gamma,wqkv=a.Wqkv,wo=a.Wo,
                gate=e.W_gate,up=e.W_up,down=e.W_down,cos=a._rope_cos,sin=a._rope_sin,
                kv=a._fused_kv_head,window=a._fused_p0,k=state.k_cache[i],v=state.v_cache[i],
                router=None if state.router_input_cache is None else state.router_input_cache[i],
                eps=float(b.norm1.eps),scale=float(a.scale),aa=float(b.alpha_attn),am=float(b.alpha_mlp)))
        self.final_gamma=engine.final_norm.gamma
        with self.stream:self._enqueue()
        self.stream.synchronize();self.stream.begin_capture()
        with self.stream:self._enqueue()
        self.graph=self.stream.end_capture();self.graph.upload(stream=self.stream);self.stream.synchronize()

    def _enqueue(self):
        m=self.module;t=256
        m.get_function("embedding_rmsnorm_bf16")((1,),(t,),(self.token,self.embedding,self.layers[0]["norm1"],self.ra,self.norm,np.int32(self.vocab),np.int32(self.d_model),np.float32(self.layers[0]["eps"])))
        for i,l in enumerate(self.layers):
            bf16_matmul_into(self.norm,l["wqkv"],self.qkv,context=self.ctx)
            router=l["router"] if l["router"] is not None else np.uint64(0)
            m.get_function("qkv_rope_store_bf16_dynamic")((1,),(t,),(self.qkv,self.norm,l["cos"],l["sin"],self.q,l["k"],l["v"],router,self.position,np.int32(self.n_q),np.int32(self.n_kv),np.int32(self.capacity),np.int32(self.d_model)))
            m.get_function("local_attention_bf16_dynamic")((self.n_q,),(32,),(self.q,l["k"],l["v"],self.context,l["kv"],l["window"],self.position,np.int32(self.n_q),np.int32(self.n_kv),np.int32(self.capacity),np.float32(l["scale"])))
            bf16_matmul_into(self.context,l["wo"],self.q,context=self.ctx)
            m.get_function("residual_rmsnorm_bf16")((1,),(t,),(self.ra,self.q,l["norm2"],self.rb,self.norm,np.int32(self.d_model),np.float32(l["aa"]),np.float32(l["eps"])))
            bf16_matmul_into(self.norm,l["gate"],self.gate,context=self.ctx);bf16_matmul_into(self.norm,l["up"],self.up,context=self.ctx)
            m.get_function("swiglu_bf16")(((self.d_ff+t-1)//t,),(t,),(self.gate,self.up,self.hidden,np.int32(self.d_ff)))
            bf16_matmul_into(self.hidden,l["down"],self.q,context=self.ctx)
            ng=self.layers[i+1]["norm1"] if i+1<len(self.layers) else self.final_gamma
            m.get_function("residual_rmsnorm_bf16")((1,),(t,),(self.rb,self.q,ng,self.ra,self.norm,np.int32(self.d_model),np.float32(l["am"]),np.float32(l["eps"])))
        bf16_matmul_into(self.norm,self.lm,self.logits,context=self.ctx)

    def decode_one(self,next_ids,state):
        if state is not self.state or not state.has_capacity(1):raise ValueError("native CUDA Graph state mismatch or exhausted cache")
        ids=xp.asarray(next_ids,dtype=xp.int32).reshape(-1)
        if ids.size!=1:raise ValueError("native CUDA Graph decode requires one token")
        current=xp.cuda.get_current_stream();xp.cuda.runtime.memcpyAsync(int(self.token.data.ptr),int(ids.data.ptr),4,xp.cuda.runtime.memcpyDeviceToDevice,int(current.ptr))
        self.pos_host[0]=state.length;xp.cuda.runtime.memcpyAsync(int(self.position.data.ptr),int(self.pos_host.ctypes.data),4,xp.cuda.runtime.memcpyHostToDevice,int(current.ptr))
        self.graph.launch(stream=current);state.length+=1;self.engine.last_hidden=self.norm
        return self.logits
