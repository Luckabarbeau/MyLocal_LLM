"""Mixed packed GPU Q8 arithmetic versus pinned Strata dots and FP64 decoding."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import mixed_q8_gpu as gpu,q8_1_gpu as q8,gguf_codebooks as books
from mini_llm.ops.quantized_weight import BLOCK_FORMATS,PackedGGUFWeight

cp=pytest.importorskip('cupy')
pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA integer dots')


def weight(fmt,n=13,k=768,seed=281):
    qk,qb=BLOCK_FORMATS[fmt]
    rng=np.random.default_rng(seed)
    data=rng.integers(0,256,(n*k//qk,qb),dtype=np.uint8)
    scale_offset=208 if fmt=='Q6_K' else 0
    data[:,scale_offset:scale_offset+2]=rng.uniform(-.02,.02,len(data)).astype('<f2').view(np.uint8).reshape(-1,2)
    if fmt in ('Q4_K','Q5_K'):
        data[:,2:4]=rng.uniform(-.02,.02,len(data)).astype('<f2').view(np.uint8).reshape(-1,2)
    host=PackedGGUFWeight(data,(n,k),fmt)
    return host,SimpleNamespace(data=cp.asarray(host.data),shape=host.shape,quant_type=fmt)


@lru_cache(None)
def oracle(device,fmt):
    root=Path(__file__).parent/'fixtures'
    source=(root/'strata_iq3s_q8_reference.cuh').read_text()
    source+='''
    typedef short int16_t;
    typedef int int32_t;
    constexpr int QK_K=256;
    struct block_iq2_s { half d; uint8_t qs[64],qh[8],scales[8]; };
    struct block_iq3_xxs { half d; uint8_t qs[96]; };
    struct block_iq4_nl { half d; uint8_t qs[16]; };
    struct block_iq4_xs { half d; unsigned short scales_h; uint8_t scales_l[4],qs[128]; };
    struct block_q2_0 { half d; uint8_t qs[16]; };
    static_assert(sizeof(block_iq2_s)==82 && sizeof(block_iq3_xxs)==98 &&
                  sizeof(block_iq4_nl)==18 && sizeof(block_iq4_xs)==136 && sizeof(block_q2_0)==18,"layouts");
    '''
    for name,typ,dt in [('iq2s_grid','unsigned long long','<u8'),('iq3xxs_grid','unsigned int','<u4'),
                        ('kvalues_iq4nl','signed char','i1')]:
        table=getattr(books,name).reshape(-1).view(dt)
        source+=f'__device__ const {typ} {name}[{table.size}] = {{'+','.join(str(int(v)) for v in table)+'};\n'
    source+=(root/'strata_mixed_q8_reference.cuh').read_text()
    if fmt in ('Q4_K','Q5_K','Q6_K','Q8_0'):
        source+=(root/'strata_k_q8_reference.cuh').read_text()
    qk,qb=BLOCK_FORMATS[fmt]
    ipb={'IQ4_NL':2,'Q4_K':16,'Q5_K':16,'Q6_K':32,'Q8_0':4}.get(fmt,qk//32)
    step=4 if fmt=='IQ4_XS' else 1 if fmt in ('Q2_0','Q6_K') else 2
    function='vec_dot_'+fmt.lower().replace('_k','_K')+'_q8_1'
    source+=f'''
    __device__ float ref_part(const unsigned char* w,const unsigned char* q,int p) {{
        int block=p/{ipb},iqs={step}*(p%{ipb});
        return {function}(w,(const block_q8_1*)(q+(long long)block*({qk}/32)*36),block,iqs);
    }}
    extern "C" __global__ void reference_parts(const unsigned char* w,const unsigned char* q,float* out,int n,int k) {{
        int i=blockIdx.x*128+threadIdx.x, parts=k/{qk}*{ipb};
        if(i<n*parts) out[i]=ref_part(w+(long long)(i/parts)*(k/{qk})*{qb},q,i%parts);
    }}
    extern "C" __global__ void reference_rows(const unsigned char* w,const unsigned char* q,float* out,int n,int k) {{
        int r=blockIdx.x,l=threadIdx.x;
        q+=(long long)blockIdx.y*(k/32)*36;
        float s=0.f;
        for(int p=l;p<k/{qk}*{ipb};p+=32) s+=ref_part(w+(long long)r*(k/{qk})*{qb},q,p);
        for(int off=16;off;off>>=1) s+=__shfl_xor_sync(0xffffffffu,s,off);
        if(!l) out[(long long)blockIdx.y*n+r]=s;
    }}
    '''
    with cp.cuda.Device(device):
        return cp.RawModule(code=source,options=('--std=c++17','--fmad=false'))


def reference(p,w):
    n,k=w.shape
    m=p.size//p.shape[-1]
    out=cp.empty(p.shape[:-1]+(n,),cp.float32)
    oracle(p.device.id,w.quant_type).get_function('reference_rows')((n,m),(32,),
        (w.data,p,out,np.int32(n),np.int32(k)))
    return out


@pytest.mark.parametrize('fmt',tuple(gpu.iq._FORMAT_ID))
@pytest.mark.parametrize('k',[256,768,2560])
def test_exact_strata_parts_rows_and_fp64_baseline(fmt,k):
    host,dev=weight(fmt,k=k)
    x=cp.asarray(np.random.default_rng(2811).normal(0,.1,(3,k)).astype(np.float32))
    p=q8.quantize_q8_1(x)
    actual=gpu.linear(x,dev)
    expected=reference(p,dev)
    np.testing.assert_array_equal(actual.get().view(np.uint32),expected.get().view(np.uint32))
    n=host.shape[0]; parts=k//(16 if fmt in ('IQ4_NL','Q4_K','Q5_K') else 8 if fmt in ('Q6_K','Q8_0') else 32)
    a,b=cp.empty((n,parts),cp.float32),cp.empty((n,parts),cp.float32)
    gpu.module(x.device.id,(fmt,)).get_function(f'q8_partials<{gpu.FORMAT_ID[fmt]}>')(
        ((a.size+127)//128,),(128,),(dev.data,p[0],a,np.int32(n),np.int32(k),np.int32(host.nbytes//n)))
    oracle(x.device.id,fmt).get_function('reference_parts')(((a.size+127)//128,),(128,),
        (dev.data,p[0],b,np.int32(n),np.int32(k)))
    np.testing.assert_array_equal(a.get().view(np.uint32),b.get().view(np.uint32))
    # Independent mathematical operation, including Strata's small integer truncation.
    q=p.get().reshape(3,k//32,36)
    scales=q[...,:2].copy().view('<f2').astype(np.float64).reshape(3,-1)
    decoded=(scales[...,None]*q[...,4:].copy().view(np.int8)).reshape(3,k)
    matrix=host.dequantize_reference().astype(np.float64)
    baseline=decoded@matrix.T
    magnitude=np.abs(decoded)@np.abs(matrix).T
    bound=16*np.finfo(np.float32).eps*magnitude+1e-30
    if fmt in ('IQ2_S','IQ3_XXS'):
        wd=host.data.reshape(n,k//256,-1)[...,:2].copy().view('<f2').astype(np.float64).reshape(n,-1)
        block_scales=np.repeat(np.abs(wd),8,axis=1)
        bound+=(1.125 if fmt=='IQ2_S' else .5)*(np.abs(scales)@block_scales.T)
    error=np.abs(actual.get().astype(np.float64)-baseline)
    assert np.all(error<=bound),(fmt,error.max(),bound.max())


@pytest.mark.parametrize('fmt',['IQ4_NL','Q2_0'])
def test_expert_down_non_256_width_and_zero(fmt):
    host,dev=weight(fmt,k=640)
    x=cp.zeros((1,640),cp.float32)
    np.testing.assert_array_equal(gpu.linear(x,dev).get(),np.zeros((1,13),np.float32))
    x[:]=cp.arange(640,dtype=cp.float32)
    np.testing.assert_array_equal(gpu.linear(x,dev).get(),reference(q8.quantize_q8_1(x),dev).get())


def test_output_validation_and_graph_replay():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    _,w=weight('IQ3_S')
    x=cp.ones((1,768),cp.float32)
    out=cp.empty((1,13),cp.float32)
    assert gpu.linear(x,w,out=out) is out
    with pytest.raises(ValueError,match='exact shape'):
        gpu.linear(x,w,out=cp.empty(13,cp.float32))
    graph=CapturedArrayInference(lambda x,unused:gpu.linear(x,w),x,())
    for i in range(4):
        x.fill(i)
        np.testing.assert_array_equal(graph.forward(x,()).get(),gpu.linear(x,w).get())


from test_0256_ggml_packed_cpu import selected_library,packed


@pytest.mark.parametrize('gate,down',[('IQ2_S','IQ4_NL'),('IQ2_S','Q2_0'),
    ('IQ3_XXS','IQ4_NL'),('IQ3_XXS','Q2_0'),('IQ3_S','IQ4_NL'),('IQ3_S','Q2_0'),('IQ4_XS','IQ4_NL')])
@pytest.mark.parametrize('mode',['sync','async','graph'])
def test_mixed_plan_all_checkpoint_triples(selected_library,monkeypatch,gate,down,mode):
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor,ExpertKey
    monkeypatch.setenv('MINI_LLM_EXPERT_GPU_Q8','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_HANDOFF','0' if mode=='sync' else '1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH','1' if mode=='graph' else '0')
    sources={}
    def provider(key):
        e=key.expert_id
        if e not in sources:
            sources[e]=(packed(gate,640,256,2810+e),packed(gate,640,256,3810+e),
                        packed(down,256,640,4810+e))
        return sources[e]
    executor=HeterogeneousExpertExecutor(provider,model_id='0281',d_model=256,d_ff=640,
        compute_dtype='float32',device_budget_bytes=8*1024**2,cpu_workers=4,
        cpu_packed_backend='ggml_pool',promote_after=1,device_expert_plan=True)
    x=cp.asarray(np.random.default_rng(2812).normal(0,.1,(1,256)).astype(np.float32))
    ids=[19,0,31,4,7,11,3,1,13,9]
    weights=np.linspace(.01,.1,10,dtype=np.float32)
    # Independent captured Strata row dots plus separately rounded SwiGLU.
    swiglu=cp.RawKernel('''extern "C" __global__ void sw(const float* g,const float* u,float* h,int n) {
        int i=blockIdx.x*128+threadIdx.x; if(i<n) h[i]=(g[i]/(1.f+expf(-g[i])))*u[i];
    }''','sw',options=('--fmad=false',))
    try:
        for e in ids[:4]: executor._prepare(0,e,x)
        executor.cache.set_admission_budget(0)
        expected=cp.zeros_like(x)
        for e,r in sorted(zip(ids,weights)):
            key=ExpertKey('0281',0,e)
            if key in executor.cache.resident:
                expert=executor.cache.resident[key][0]
                p=q8.quantize_q8_1(x)
                g,u=reference(p,expert.gate),reference(p,expert.up)
                h=cp.empty_like(g)
                swiglu((5,),(128,),(g,u,h,np.int32(640)))
                y=reference(q8.quantize_q8_1(h),expert.down)
            else:
                y=cp.asarray(executor._ggml_cpu.expert(x.get(),provider(key)))
            expected+=y*cp.float32(r)
        actual=executor.try_forward_device_plan(0,x,cp.asarray(weights),cp.asarray(ids,dtype=cp.int64),32)
        np.testing.assert_array_equal(actual.get(),expected.get())
        saved=actual.get()
        executor.try_forward_device_plan(0,x*2,cp.asarray(weights),cp.asarray(ids,dtype=cp.int64),32)
        np.testing.assert_array_equal(actual.get(),saved)
    finally:
        executor.close()
