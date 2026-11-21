"""IQ2_S gather retains GGML Q8_K, scales/signs and exact FP32 accumulation."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0256_ggml_packed_cpu import selected_library,packed


@pytest.fixture
def pair(selected_library):
    baseline=GGMLPackedCPUExecutor(selected_library,iq2_gather=False)
    if not baseline._bridge.mylocal_iq2_gather_available(): pytest.skip('AVX2/F16C/FMA unavailable')
    return baseline,GGMLPackedCPUExecutor(selected_library,iq2_gather=True)


@pytest.mark.parametrize('width',[256,512,768,2560])
@pytest.mark.parametrize('seed',[0,1,2])
def test_linear_exact_pinned_ggml(pair,width,seed):
    base,fast=pair
    weight=packed('IQ2_S',n=37,k=width,seed=331+seed)
    x=np.random.default_rng(seed).normal(0,.3,(3,width)).astype(np.float32)
    expected=base.linear(x,weight)
    actual=fast.linear(x,weight)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert fast.provenance['iq2_s_dot']=='avx2_gather_q8_k_same_fma_order'


def test_half_scale_extremes_and_readonly_unaligned_source(pair):
    base,fast=pair
    w=packed('IQ2_S',n=17,k=512,seed=331)
    raw=np.empty(w.data.size+1,np.uint8)
    raw[1:]=w.data.reshape(-1)
    rows=raw[1:].reshape(-1,82)
    scales=np.resize(np.array([0.,-0.,2**-24,2**-14,.01,-.125,1.,65504],np.float16),len(rows))
    rows[:,:2]=scales.view(np.uint8).reshape(-1,2)
    raw.flags.writeable=False
    weight=PackedGGUFWeight(raw[1:],w.shape,w.quant_type)
    x=np.random.default_rng(297).normal(0,.1,(1,512)).astype(np.float32)
    np.testing.assert_array_equal(fast.linear(x,weight).view(np.uint32),base.linear(x,weight).view(np.uint32))


@pytest.mark.parametrize('down',['IQ4_NL','Q2_0'])
def test_mixed_expert_pool_and_parallel_ownership(pair,down):
    base,fast=pair
    bank=[(packed('IQ2_S',320,256,100+i),packed('IQ2_S',320,256,200+i),
           packed(down,256,320,300+i)) for i in range(10)]
    x=np.random.default_rng(297).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([base.expert(x,weights) for weights in bank],axis=0)
    with GGMLPackedCPURowPool(fast,256,320,4,scheduler='tasks',host_help=True,spin_us=50) as pool:
        np.testing.assert_array_equal(pool.submit(x,bank).result().view(np.uint32),expected.view(np.uint32))
    with ThreadPoolExecutor(max_workers=4) as threads:
        results=list(threads.map(lambda weights:fast.expert(x,weights),bank))
    np.testing.assert_array_equal(np.concatenate(results).view(np.uint32),expected.view(np.uint32))
