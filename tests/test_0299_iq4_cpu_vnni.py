"""IQ4_NL AVX-VNNI exact parity, including odd-block tails and packed ownership."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0256_ggml_packed_cpu import selected_library,packed


@pytest.fixture
def pair(selected_library):
    base=GGMLPackedCPUExecutor(selected_library,iq4_vnni=False)
    if not base._bridge.mylocal_iq4_vnni_available(): pytest.skip('GCC AVX-VNNI/F16C/FMA unavailable')
    return base,GGMLPackedCPUExecutor(selected_library,iq4_vnni=True)


@pytest.mark.parametrize('width',[32,64,96,320,768,2560])
@pytest.mark.parametrize('seed',[0,1,2])
def test_linear_exact_pair_order_and_scalar_tail(pair,width,seed):
    base,fast=pair
    weight=packed('IQ4_NL',n=31,k=width,seed=299+seed)
    x=np.random.default_rng(seed).normal(0,.3,(3,width)).astype(np.float32)
    np.testing.assert_array_equal(fast.linear(x,weight).view(np.uint32),base.linear(x,weight).view(np.uint32))


def test_scale_extremes_unaligned_readonly_and_parallel_calls(pair):
    base,fast=pair
    w=packed('IQ4_NL',n=31,k=96,seed=299)
    raw=np.empty(w.data.size+1,np.uint8); raw[1:]=w.data.reshape(-1)
    blocks=raw[1:].reshape(-1,18)
    scales=np.resize(np.array([0.,-0.,2**-24,2**-14,.01,-.125,1.,65504],np.float16),len(blocks))
    blocks[:,:2]=scales.view(np.uint8).reshape(-1,2); raw.flags.writeable=False
    weight=PackedGGUFWeight(raw[1:],w.shape,w.quant_type)
    x=np.random.default_rng(299).normal(0,.1,(1,96)).astype(np.float32)
    expected=base.linear(x,weight)
    with ThreadPoolExecutor(max_workers=4) as threads:
        results=list(threads.map(lambda _:fast.linear(x,weight),range(8)))
    for result in results: np.testing.assert_array_equal(result.view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('hidden',[96,320,768])
def test_complete_experts_and_native_pool(pair,hidden):
    base,fast=pair
    bank=[(packed('IQ3_S',hidden,256,100+i),packed('IQ3_S',hidden,256,200+i),
           packed('IQ4_NL',256,hidden,300+i)) for i in range(10)]
    x=np.random.default_rng(299).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([base.expert(x,w) for w in bank])
    with GGMLPackedCPURowPool(fast,256,hidden,4,scheduler='tasks',host_help=True,spin_us=50) as pool:
        np.testing.assert_array_equal(pool.submit(x,bank).result().view(np.uint32),expected.view(np.uint32))
