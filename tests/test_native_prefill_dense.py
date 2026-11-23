"""Bounded arithmetic tests: no checkpoint/model loads or external engine."""
import ctypes
import ctypes.util
from types import SimpleNamespace
from functools import lru_cache

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from mini_llm.ops import native_prefill_dense as dense

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu():
    if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceProperties(0)['major'] < 8:
        pytest.skip('NVIDIA sm80+ required')
    with cp.cuda.Device(0):
        yield
        cp.cuda.get_current_stream().synchronize()


def bits_equal(a, b):
    np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint8),
                                  cp.asnumpy(b).view(np.uint8))


# Independent CUDA oracle, not generated from the production CUDA/source file.
ORACLE = r'''
#include <cuda_fp16.h>
typedef unsigned short uint16_t;
__device__ uint16_t b16(float x) {
    unsigned u=__float_as_uint(x);
    return (u+32767+((u>>16)&1))>>16;
}
extern "C" __global__ void norm_ref(const float* r,const float* gamma,
    float* scale,uint16_t* image,int n,float eps) {
    __shared__ float partial[8];
    int tid=threadIdx.x, lane=tid%32, warp=tid/32, row=blockIdx.x;
    float a=0;
    for(int j=tid;j<n;j+=256) a=fmaf(r[row*n+j],r[row*n+j],a);
    for(int step=16;step;step/=2) a+=__shfl_xor_sync(0xffffffff,a,step);
    if(lane==0) partial[warp]=a;
    __syncthreads();
    if(warp==0) {
        a=lane<8?partial[lane]:0;
        for(int step=16;step;step/=2) a+=__shfl_xor_sync(0xffffffff,a,step);
        if(lane==0) partial[0]=rsqrtf(a/(float)n+eps);
    }
    __syncthreads();
    if(tid==0) scale[row]=partial[0];
    for(int j=tid;j<n;j+=256)
        image[row*n+j]=b16((r[row*n+j]*partial[0])*gamma[(row%4)*n+j]);
}
extern "C" __global__ void silu_ref(const float* x,uint16_t* y,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {float z=x[i]*0.25f;y[i]=b16(z/(1.0f+__expf(-z)));}
}
extern "C" __global__ void mix_ref(const float* r,const float* rs,const float* w,
    const float* gate,float* out,uint16_t* b,half* h,int n,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    int t=i/n,d=i%n;float a=0;
    for(int c=0;c<4;c++) {
        int j=(t*4+c)*n+d;
        float z=(r[j]*rs[t*4+c])*w[c*n+d];
        a=fmaf(z,1.0f/(1.0f+__expf(-gate[j])),a);
    }
    a*=0.25f;out[i]=a;b[i]=b16(a);h[i]=__float2half_rn(a);
}
extern "C" __global__ void gain_ref(const float* x,float* y,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=2.0f/(1.0f+__expf(-(x[i]*0.25f)));
}
'''


@lru_cache(maxsize=None)
def oracle(name):
    from cupy.cuda import compiler, function
    from cupy._core.core import assemble_cupy_compiler_options
    opts = assemble_cupy_compiler_options((
        '--std=c++17', '--ftz=false', '--prec-div=true',
        f'--gpu-architecture=sm_{cp.cuda.Device().compute_capability}'))
    cubin, _ = compiler._NVRTCProgram(ORACLE, method='cubin').compile(opts)
    mod = function.Module()
    mod.load(cubin)
    return mod.get_function(name)


def reference_gemm(x, w, bf16=False):
    """Separate ctypes cuBLAS entry point/handle; never calls project_matrix."""
    from cupy_backends.cuda.libs import cublas as cb
    lib = ctypes.CDLL(ctypes.util.find_library('cublas'))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    gemm = lib.cublasGemmEx
    gemm.argtypes = [ptr, integer, integer, integer, integer, integer,
        ptr, ptr, integer, integer, ptr, integer, integer, ptr, ptr,
        integer, integer, integer, integer]
    gemm.restype = integer
    ws_fn = lib.cublasSetWorkspace_v2
    ws_fn.argtypes = [ptr, ptr, ctypes.c_size_t]
    ws_fn.restype = integer
    t,k=x.shape; n=w.shape[0]
    y=cp.empty((t,n),cp.float32)
    a,b=ctypes.c_float(1),ctypes.c_float(0)
    ws=cp.empty(32*1024*1024,cp.uint8)
    handle=cb.create()
    try:
        cb.setStream(handle,cp.cuda.get_current_stream().ptr)
        cb.setMathMode(handle,0)
        assert ws_fn(handle,ws.data.ptr,ws.nbytes)==0
        typ=14 if bf16 else 2
        assert gemm(handle,1,0,n,t,k,ctypes.byref(a),w.data.ptr,typ,k,
            x.data.ptr,typ,k,ctypes.byref(b),y.data.ptr,0,n,68,-1)==0
    finally:
        cb.destroy(handle)
    return y


@pytest.mark.parametrize('n', [37, 256, 777, 2560])
def test_hc_boundary_bitwise(n):
    rng=np.random.default_rng(21)
    t,rank=3,32
    r=cp.asarray(rng.normal(size=(t,4,n)).astype(np.float32))
    gamma=cp.asarray(rng.normal(1,.1,size=(4,n)).astype(np.float32))
    make=lambda shape: dense.consumer_images(cp.asarray(
        rng.normal(0,.03,size=shape).astype(np.float32))).bf16
    hc=SimpleNamespace(gamma=gamma,down=make((rank,4*n)),
        up=make((4*n,rank)),inject=make((4,4*n)),epsilon=1e-6)
    rs=cp.empty((t,4),cp.float32); xn=cp.empty((t,4*n),cp.uint16)
    oracle('norm_ref')((t*4,),(256,),
        (r,gamma,rs,xn,np.int32(n),np.float32(hc.epsilon)))
    actual_rs=cp.empty_like(rs);actual_xn=cp.empty_like(xn)
    dense._kernel('gr_norm_rs_kernel',n)((t*4,),(256,),
        (r,gamma,np.float32(hc.epsilon),actual_rs,actual_xn,np.uint64(0),
         np.int64(4*n),np.int32(0)))
    bits_equal(actual_rs,rs);bits_equal(actual_xn,xn)
    low=reference_gemm(xn,hc.down,True)
    lo16=cp.empty_like(low,dtype=cp.uint16)
    oracle('silu_ref')(((low.size+255)//256,),(256,),
        (low,lo16,np.int32(low.size)))
    actual_lo=cp.empty_like(lo16)
    dense._kernel('gr_silu_kernel',n)(((low.size+255)//256,),(256,),
        (low,actual_lo,np.uint64(0),np.int64(low.size)))
    bits_equal(actual_lo,lo16)
    gate=reference_gemm(lo16,hc.up,True)
    logits=reference_gemm(xn,hc.inject,True)
    y=cp.empty((t,n),cp.float32);b=cp.empty((t,n),cp.uint16)
    h=cp.empty((t,n),cp.float16);g=cp.empty((t,4),cp.float32)
    oracle('mix_ref')(((y.size+255)//256,),(256,),
        (r,rs,gamma,gate,y,b,h,np.int32(n),np.int32(y.size)))
    oracle('gain_ref')((1,),(256,),(logits,g,np.int32(g.size)))
    got=dense.hc_read(hc,r)
    for actual,expected in ((got.hidden,y),(got.gains,g),
                           (got.images.bf16,b),(got.images.fp16,h)):
        bits_equal(actual,expected)
    # No injection, output HC; widened BF16 constructor matrices are accepted.
    hc.inject=None
    hc.down=(hc.down.astype(cp.uint32)<<16).view(cp.float32)
    noinj=dense.hc_read(hc,r)
    assert noinj.gains is None
    bits_equal(noinj.hidden,y)


def packed_bytes(kind, n=7, k=256):
    _,qk,qb=dense.FORMATS[kind]
    rng=np.random.default_rng(9)
    raw=rng.integers(0,256,(n*k//qk,qb),dtype=np.uint8)
    # Finite scales, exact powers of two isolate unpacking from oracle FMA policy.
    scale=np.array([0.03125],np.float16).view(np.uint8)
    off={'Q3_K':108,'Q6_K':208}.get(kind,0)
    raw[:,off:off+2]=scale
    if kind in ('Q4_K','Q5_K','Q5_1'):
        raw[:,2:4]=np.array([0.015625],np.float16).view(np.uint8)
    return raw.reshape(-1)


@pytest.mark.parametrize('kind', list(dense.FORMATS))
def test_packed_projection_vs_independent_reference(kind):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    n,k=7,256
    raw=packed_bytes(kind,n,k)
    # GGUF's independent CPU dequantizer, then explicit FP16 rounding.
    if kind == 'Q2_0':
        blocks = raw.reshape(-1, 18)
        scales = blocks[:, :2].copy().view('<f2').astype(np.float32)
        codes = ((blocks[:, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3)
        ref = (codes.reshape(-1, 64).astype(np.float32)-1)*scales
    else:
        ref=dequantize(raw.reshape(n,-1),getattr(GGMLQuantizationType,kind))
    w=cp.asarray(ref.reshape(n,k).astype(np.float16))
    gotw=dense.dequantize_fp16(cp.asarray(raw),kind,(n,k))
    bits_equal(gotw,w)
    x=cp.asarray(np.random.default_rng(4).normal(size=(3,k)).astype(np.float32))
    imgs=dense.consumer_images(x)
    expected=reference_gemm(imgs.fp16,w)
    store=SimpleNamespace(tensors={'p':SimpleNamespace(shape=(n,k),quant_type=kind)},
                          tensor_bytes=lambda name:raw)
    model=SimpleNamespace(weights=SimpleNamespace(store=store))
    bits_equal(dense.native_project(model,'p',imgs),expected)
    bits_equal(dense.native_project(model,'p',x),expected)


def test_bf16_projection_and_rejections():
    x=cp.arange(192,dtype=cp.float32).reshape(3,64)/127
    w=dense.consumer_images(cp.arange(320,dtype=cp.float32).reshape(5,64)/47).bf16
    imgs=dense.consumer_images(x)
    raw=cp.asnumpy(w).view(np.uint8).reshape(-1)
    store=SimpleNamespace(tensors={'p':SimpleNamespace(shape=(5,64),quant_type='BF16')},
                          tensor_bytes=lambda name:raw)
    model=SimpleNamespace(weights=SimpleNamespace(store=store))
    bits_equal(dense.native_project(model,'p',imgs),reference_gemm(imgs.bf16,w,True))
    with pytest.raises(NotImplementedError):
        dense.hc_read(None,None,mode='fp16')
    with pytest.raises(NotImplementedError):
        dense.dequantize_fp16(cp.empty(1,cp.uint8),'IQ3_S',(1,256))
    with pytest.raises(ValueError):
        dense.project_matrix(imgs.fp16[:,::2],w)
    with pytest.raises(ValueError):
        dense.dequantize_fp16(cp.empty(1,cp.uint8),'Q8_0',(1,64))
