"""Packed IQ/Q2 integer GPU dots with native Q8_1 activations.

Checkpoint bytes stay unchanged. Float products/scales and warp reduction are
FP32; IQ2_S/IQ3_XXS retain Strata's explicit integer divisions, not an assumed
float rescaling. CPU GGML Q8_K/Q8_0 remains a separate arithmetic path.
Source arithmetic: Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf,
`src/kernels/cuda/iq_kernels.cu` vec_dot_*_q8_1 and row_dot. Existing codebook
attribution is embedded by iq_packed_gpu. No whole-engine API/code is ported.
"""
# Arithmetic adapted from the pinned Strata/GGML source under the MIT License:
# Copyright (c) 2026 Niko1221 and the Strata contributors
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from functools import lru_cache
from types import SimpleNamespace
import os
import numpy as np
from . import iq_packed_gpu as iq
from . import q8_1_gpu as q8

cp=q8.cp
FORMATS=(*iq._FORMAT_ID,'Q4_K','Q5_K','Q6_K','Q8_0')
FORMAT_ID={fmt:i for i,fmt in enumerate(FORMATS)}
_SOURCE=r'''
__device__ __forceinline__ unsigned int signed_word(unsigned int v, unsigned int signs) {
    const unsigned int mask=__vcmpne4(((signs&3)<<7)|((signs&12)<<21),0);
    return __vsub4(v^mask,mask);
}
template<int F> __device__ __forceinline__ unsigned int grid_word(const unsigned char* shared,int offset) {
#ifdef SHARED_IQ_GRID
    if(shared) return *(const unsigned int*)(shared+offset);
#endif
    if constexpr(F==0) return u32(iq2s_grid+offset);
    if constexpr(F==1) return u32(iq3s_grid+offset);
    return u32(iq3xxs_grid+offset);
}
template<int F> __device__ __forceinline__ int codes4(const unsigned char* b,int g,int j,const unsigned char* grid=nullptr) {
    if constexpr(F==0) {
        int l=j/2;
        int idx=b[2+4*g+l]|(((b[66+g]>>(2*l))&3)<<8);
        return signed_word(grid_word<F>(grid,8*idx+4*(j%2)),b[34+4*g+l]>>(4*(j%2)));
    } else if constexpr(F==1 || F==2) {
        int idx=b[2+8*g+j];
        unsigned int signs;
        if constexpr(F==1) {
            idx|=((b[66+g]>>j)&1)<<8;
            signs=b[74+4*g+j/2]>>(4*(j%2));
            return signed_word(grid_word<F>(grid,4*idx),signs);
        } else {
            unsigned int aux=u32(b+66+4*g);
            signs=ksigns_iq2xs[(aux>>(7*(j/2)))&127]>>(4*(j%2));
            return signed_word(grid_word<F>(grid,4*idx),signs);
        }
    } else {
        unsigned int result=0;
        #pragma unroll
        for(int t=0;t<4;++t) {
            int col=4*j+t,code;
            if constexpr(F==5) code=((b[2+col/4]>>(2*(col%4)))&3)-1;
            else {
                int off=F==4 ? 8+16*g : 2;
                code=kvalues_iq4nl[(b[off+col%16]>>(4*(col/16)))&15];
            }
            result|=(unsigned int)(unsigned char)code<<(8*t);
        }
        return (int)result;
    }
}
template<int F> __device__ __forceinline__ float k_part(const unsigned char* w,const unsigned char* q,int p) {
    if constexpr(F==6 || F==7) {
        const unsigned char* b=w+(long long)(p/16)*(F==6?144:176);
        int group=2*((p%16)/4),chunk=p%4;
        const unsigned char* a=q+(long long)(p/16)*8*36;
        const unsigned char* qs=b+(F==6?16:48)+16*group+4*chunk;
        unsigned int v0=u32(qs),v1=u32(qs+16),h0=0,h1=0;
        if constexpr(F==7) { h0=u32(b+16+4*chunk)>>group; h1=u32(b+32+4*chunk)>>group; }
        float sd=0.f,sm=0.f;
        #pragma unroll
        for(int i=0;i<2;++i) {
            int g=group+i,sc,mi;
            const unsigned char* scales=b+4;
            if(g<4) { sc=scales[g]&63; mi=scales[g+4]&63; }
            else { sc=(scales[g+4]&15)|((scales[g-4]>>6)<<4); mi=(scales[g+4]>>4)|((scales[g]>>6)<<4); }
            int c0=(v0>>(4*i))&0x0f0f0f0f,c1=(v1>>(4*i))&0x0f0f0f0f;
            if constexpr(F==7) { c0|=((h0>>i)<<4)&0x10101010; c1|=((h1>>i)<<4)&0x10101010; }
            const unsigned char* ai=a+g*36;
            int u0=(int)u32(ai+4+4*chunk),u1=(int)u32(ai+20+4*chunk);
            int dot=__dp4a(c0,u0,__dp4a(c1,u1,0));
            int sum=__dp4a(0x01010101,u0,__dp4a(0x01010101,u1,0));
            float dx=__half2float(__ushort_as_half((unsigned short)u16(ai)));
            sd+=dx*(float)(dot*sc); sm+=dx*(float)(sum*mi);
        }
        float dw=__half2float(__ushort_as_half((unsigned short)u16(b)));
        float dm=__half2float(__ushort_as_half((unsigned short)u16(b+2)));
        return dw*sd-dm*sm;
    } else if constexpr(F==8) {
        const unsigned char* b=w+(long long)(p/32)*210;
        const unsigned char* a=q+(long long)(p/32)*8*36;
        int j=p%32,off=4*(j/16)+(j%16)/8,scale=8*(j/16)+(j%16)/4;
        unsigned int vl=u32(b+4*j),vh=u32(b+128+4*(8*(j/16)+j%8))>>(2*((j%16)/8));
        float sum=0.f;
        #pragma unroll
        for(int i=0;i<2;++i) {
            unsigned int lo=(vl>>(4*i))&0x0f0f0f0f,hi=((vh>>(4*i))<<4)&0x30303030;
            int codes=(int)__vsubss4(lo|hi,0x20202020);
            const unsigned char* ai=a+(off+2*i)*36;
            int dot=__dp4a(codes,(int)u32(ai+4+4*(j%8)),0);
            float dx=__half2float(__ushort_as_half((unsigned short)u16(ai)));
            sum+=dx*(float)(dot*(int)((const signed char*)b)[192+scale+4*i]);
        }
        return __half2float(__ushort_as_half((unsigned short)u16(b+208)))*sum;
    } else {
        const unsigned char* b=w+(long long)(p/4)*34;
        const unsigned char* a=q+(long long)(p/4)*36;
        int dot=0;
        #pragma unroll
        for(int i=0;i<2;++i) dot=__dp4a((int)u32(b+2+4*(2*(p%4)+i)),(int)u32(a+4+4*(2*(p%4)+i)),dot);
        float dw=__half2float(__ushort_as_half((unsigned short)u16(b)));
        float dx=__half2float(__ushort_as_half((unsigned short)u16(a)));
        return dw*dx*(float)dot;
    }
}
template<int F> __device__ __forceinline__ int n_parts(int k) {
    return k/((F==3 || F==6 || F==7)?16:F>=8?8:32);
}
template<int F> __device__ __forceinline__ float part(const unsigned char* w,const unsigned char* q,int p,const unsigned char* grid=nullptr) {
    if constexpr(F>=6) return k_part<F>(w,q,p);
    const int wb=F==3 ? p/2 : F==5 ? p/2 : p/8;
    const int bytes=F==0?82:F==1?110:F==2?98:F==3?18:F==4?136:18;
    const unsigned char* b=w+(long long)wb*bytes;
    const int g=(F==3 || F==5)?p%2:p%8;
    const unsigned char* a=q+(long long)(F==3?p/2:p)*36;
    int s0=0,s1=0;
    #pragma unroll
    for(int j=0;j<8;++j) {
        if constexpr(F==3) {
            // Two 16-value parts per IQ4_NL block, same call partition as Strata.
            if(j<4) {
                int chunk=(j/2)+2*g+4*(j%2);
                s0=__dp4a(codes4<F>(b,0,chunk),(int)u32(a+4+4*chunk),s0);
            }
        } else {
            int word;
            if constexpr(F==5) {
                unsigned int v=0;
                #pragma unroll
                for(int t=0;t<4;++t) {
                    int c=32*g+4*j+t;
                    int code=((b[2+c/4]>>(2*(c%4)))&3)-1;
                    v|=(unsigned int)(unsigned char)code<<(8*t);
                }
                word=(int)v;
            } else word=codes4<F>(b,g,j,grid);
            if constexpr(F==0) {
                if(j<4) s0=__dp4a(word,(int)u32(a+4+4*j),s0);
                else s1=__dp4a(word,(int)u32(a+4+4*j),s1);
            } else s0=__dp4a(word,(int)u32(a+4+4*j),s0);
        }
    }
    if constexpr(F==0) s0=(s0*(b[74+g]&15)+s1*(b[74+g]>>4)+(s0+s1)/2)/4;
    if constexpr(F==1) s0*=1+2*((b[106+g/2]>>(4*(g%2)))&15);
    if constexpr(F==2) s0=((int)(u32(b+66+4*g)>>28)*s0+s0/2)/2;
    if constexpr(F==4) s0*=(((b[4+g/2]>>(4*(g%2)))&15)|(((u16(b+2)>>(2*g))&3)<<4))-32;
    float wd=__half2float(__ushort_as_half((unsigned short)u16(b)));
    float xd=__half2float(__ushort_as_half((unsigned short)u16(a)));
    return __fmul_rn(__fmul_rn(wd,xd),(float)s0);
}
template<int F> __device__ __forceinline__ float row_dot(const unsigned char* w,const unsigned char* q,int k,int lane,const unsigned char* grid=nullptr) {
    float sum=0.f;
    for(int p=lane;p<n_parts<F>(k);p+=32) sum=__fadd_rn(sum,part<F>(w,q,p,grid));
    #pragma unroll
    for(int off=16;off;off>>=1) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,off));
    return sum;
}
template<int F> __device__ __forceinline__ void init_grid(unsigned char* grid) {
#ifdef SHARED_IQ_GRID
    if constexpr(F<=2) {
        constexpr int size=F==0?8192:F==1?2048:1024;
        const unsigned char* src=F==0?iq2s_grid:F==1?iq3s_grid:iq3xxs_grid;
        for(int i=threadIdx.x;i<size/4;i+=blockDim.x)
            ((unsigned int*)grid)[i]=((const unsigned int*)src)[i];
        __syncthreads();
    }
#endif
}
template<int F> __global__ void q8_linear(const unsigned char* w,const unsigned char* q,float* out,int n,int k,int rowbytes) {
#ifdef SHARED_IQ_GRID
    __shared__ __align__(16) unsigned char grid[F==0?8192:F==1?2048:F==2?1024:1];
    init_grid<F>(grid);
#else
    const unsigned char* grid=nullptr;
#endif
    int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=n) return;
    float sum=row_dot<F>(w+(long long)row*rowbytes,q+(long long)blockIdx.y*(k/32)*36,k,lane,grid);
    if(!lane) out[(long long)blockIdx.y*n+row]=sum;
}
template<int F> __global__ void q8_partials(const unsigned char* w,const unsigned char* q,float* out,int n,int k,int rowbytes) {
    int i=blockIdx.x*128+threadIdx.x,parts=n_parts<F>(k);
    if(i<n*parts) out[i]=part<F>(w+(long long)(i/parts)*rowbytes,q,i%parts);
}
template<int G,int U> __global__ void q8_gate_up(const unsigned char* x,const unsigned long long* table,float* h,
        int d,int f,int grb,int urb,int experts) {
#ifdef SHARED_IQ_GRID
    // Whole inactive expert blocks need no codebook upload. All threads take
    // this branch together; non-four-row-aligned geometries keep the safe path.
    int base=blockIdx.x*4;
    if(f%4==0 && base<experts*f && !table[(base/f)*3]) {
        int pair=base+threadIdx.x/32;
        if(!(threadIdx.x%32) && pair<experts*f) h[pair]=0.f;
        return;
    }
    __shared__ __align__(16) unsigned char grid[G==0?8192:G==1?2048:G==2?1024:1];
    init_grid<G>(grid);
#else
    const unsigned char* grid=nullptr;
#endif
    int pair=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(pair>=experts*f) return;
    int e=pair/f,r=pair%f;
    if(!table[e*3]) { if(!lane) h[pair]=0.f; return; }
    float g=row_dot<G>((const unsigned char*)table[e*3]+(long long)r*grb,x,d,lane,grid);
    float u=row_dot<U>((const unsigned char*)table[e*3+1]+(long long)r*urb,x,d,lane,G==U?grid:nullptr);
    if(!lane) h[pair]=(g/(1.f+expf(-g)))*u;
}
template<int F> __global__ void q8_down(const unsigned char* h,const unsigned long long* table,const float* routes,
        float* terms,int d,int f,int rb,int experts) {
    int pair=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(pair>=experts*d) return;
    int e=pair/d,r=pair%d;
    if(!table[e*3+2]) { if(!lane) terms[pair]=0.f; return; }
    float v=row_dot<F>((const unsigned char*)table[e*3+2]+(long long)r*rb,h+(long long)e*(f/32)*36,f,lane);
    if(!lane) terms[pair]=v*routes[e];
}
'''


@lru_cache(maxsize=32)
def module(device, formats, shared_grid=False):
    names=[]
    for fmt in set(formats):
        i=FORMAT_ID[fmt]
        names += [f'q8_linear<{i}>',f'q8_partials<{i}>',f'q8_down<{i}>']
    if len(formats)==3:
        names.append(f'q8_gate_up<{FORMAT_ID[formats[0]]},{FORMAT_ID[formats[1]]}>')
    with cp.cuda.Device(device):
        prefix='#define SHARED_IQ_GRID\n' if shared_grid else ''
        base=iq._cuda_source('global').replace('INPUT_T','float')
        if shared_grid: base=base.replace('__device__ const','__device__ __align__(16) const')
        return cp.RawModule(code=prefix+base+_SOURCE,
                            options=('--std=c++17','--fmad=false'),name_expressions=tuple(names))


def linear_packed(packed,weight,*,out=None):
    device=q8._validate(packed,np.uint8,'packed')
    # Reuse the public packed-weight layout checks; dummy activation shape is unnecessary.
    if weight.quant_type not in FORMATS:
        raise NotImplementedError(f'No Q8 GPU dot for {weight.quant_type}')
    n,k=weight.shape
    if weight.quant_type in iq._FORMAT_ID:
        iq._validate_weight(SimpleNamespace(device=packed.device,
            shape=packed.shape[:-1]+(k,)),weight)
    else:
        data=weight.data
        qk,qb=iq.BLOCK_FORMATS[weight.quant_type]
        if (not isinstance(data,cp.ndarray) or data.dtype!=cp.uint8 or
                not data.flags.c_contiguous or data.device.id!=device):
            raise ValueError('Expected contiguous packed bytes on the current device')
        if (min(n,k)<=0 or max(n,k)>2147483647 or k%qk or data.size!=n*(k//qk)*qb):
            raise ValueError('Invalid packed shape/alignment/byte length')
    if packed.shape[-1]!=(k//32)*36:
        raise ValueError('Q8_1 row width mismatch')
    result=q8._output(out,packed.shape[:-1]+(n,),np.float32,packed)
    if result.nbytes and result.data.ptr<weight.data.data.ptr+weight.data.nbytes and weight.data.data.ptr<result.data.ptr+result.nbytes:
        raise ValueError('out must not overlap weights')
    if result.size:
        m=packed.size//packed.shape[-1]
        if m>65535: raise ValueError('Too many Q8 rows')
        i=FORMAT_ID[weight.quant_type]
        module(device,(weight.quant_type,),os.environ.get('MINI_LLM_Q8_SHARED_GRID','0')=='1').get_function(f'q8_linear<{i}>')(((n+3)//4,m),(128,),
            (weight.data,packed,result,np.int32(n),np.int32(k),np.int32(weight.data.nbytes//n)))
    return result


def linear(x,weight,*,out=None):
    q8._validate(x,np.float32,'x')
    if x.shape[-1]!=weight.shape[-1]: raise ValueError('Input width mismatch')
    return linear_packed(q8.quantize_q8_1(x),weight,out=out)


class GroupedQ8Plan:
    """Own only O(D + selected*F) Q8 scratch; residency pointers stay in owner."""
    def __init__(self,plan):
        self.plan=plan
        self.input=cp.empty(plan.d//32*36,cp.uint8)
        self.hidden=cp.empty((plan.k,plan.f//32*36),cp.uint8)
        formats=plan.formats
        mod=module(plan.device,formats,os.environ.get('MINI_LLM_Q8_SHARED_GRID','0')=='1')
        self.gate_up=mod.get_function(f'q8_gate_up<{FORMAT_ID[formats[0]]},{FORMAT_ID[formats[1]]}>')
        self.down=mod.get_function(f'q8_down<{FORMAT_ID[formats[2]]}>')
        self.gu_args=tuple(np.int32(v) for v in (plan.d,plan.f,
            plan.resident_row_bytes[0],plan.resident_row_bytes[1],plan.k))
        self.down_args=tuple(np.int32(v) for v in (plan.d,plan.f,plan.resident_row_bytes[2],plan.k))

    def forward(self,x):
        p=self.plan
        q8.quantize_q8_1(x.reshape(-1),out=self.input)
        self.gate_up(((p.k*p.f+3)//4,),(128,),
            (self.input,p.selected,p.hidden)+self.gu_args)
        q8.quantize_q8_1(p.hidden,out=self.hidden)
        self.down(((p.k*p.d+3)//4,),(128,),
            (self.hidden,p.selected,p.routes,p.terms)+self.down_args)
