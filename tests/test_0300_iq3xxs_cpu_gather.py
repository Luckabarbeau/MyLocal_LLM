"""IQ3_XXS gather must preserve pinned GGML Q8_K arithmetic exactly."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0256_ggml_packed_cpu import selected_library, packed


@pytest.fixture(params=['gather','pair'])
def pair(selected_library,request):
    base=GGMLPackedCPUExecutor(selected_library,iq3xxs_gather=False,iq3xxs_pair=False)
    if not base._bridge.mylocal_iq3xxs_gather_available(): pytest.skip('AVX2/F16C/FMA unavailable')
    return base,GGMLPackedCPUExecutor(selected_library,iq3xxs_gather=request.param=='gather',
                                      iq3xxs_pair=request.param=='pair')


@pytest.mark.parametrize('width',[256,512,768,2560])
@pytest.mark.parametrize('seed',[0,1,2])
def test_linear_exact_pinned_ggml(pair,width,seed):
    base,fast=pair
    weight=packed('IQ3_XXS',37,width,300+seed)
    x=np.random.default_rng(seed).normal(0,.3,(3,width)).astype(np.float32)
    np.testing.assert_array_equal(fast.linear(x,weight).view(np.uint32),base.linear(x,weight).view(np.uint32))
    assert fast.provenance['iq3_xxs_dot'] in ('avx2_gather_q8_k_same_fma_order','avx2_pair_q8_k_same_fma_order')


def test_extremes_unaligned_readonly(pair):
    base,fast=pair
    w=packed('IQ3_XXS',17,512,300)
    raw=np.empty(w.data.size+1,np.uint8); raw[1:]=w.data.reshape(-1)
    rows=raw[1:].reshape(-1,98)
    scales=np.resize(np.array([0.,-0.,2**-24,2**-14,.01,-.125,1.,65504],np.float16),len(rows))
    rows[:,:2]=scales.view(np.uint8).reshape(-1,2)
    raw.flags.writeable=False
    weight=PackedGGUFWeight(raw[1:],w.shape,w.quant_type)
    x=np.random.default_rng(300).normal(0,.1,(1,512)).astype(np.float32)
    np.testing.assert_array_equal(fast.linear(x,weight).view(np.uint32),base.linear(x,weight).view(np.uint32))


@pytest.mark.parametrize('down',['IQ4_NL','Q2_0'])
def test_pool_parallel_ownership(pair,down):
    base,fast=pair
    bank=[(packed('IQ3_XXS',320,256,100+i),packed('IQ3_XXS',320,256,200+i),
           packed(down,256,320,300+i)) for i in range(10)]
    x=np.random.default_rng(300).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([base.expert(x,weights) for weights in bank])
    with GGMLPackedCPURowPool(fast,256,320,4,scheduler='tasks',host_help=True,spin_us=50) as pool:
        np.testing.assert_array_equal(pool.submit(x,bank).result().view(np.uint32),expected.view(np.uint32))
    with ThreadPoolExecutor(max_workers=4) as threads:
        actual=np.concatenate(list(threads.map(lambda w:fast.expert(x,w),bank)))
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
