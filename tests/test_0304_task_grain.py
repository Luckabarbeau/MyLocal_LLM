"""Task grain cannot alter arithmetic or bypass a live batch's descriptors."""
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import selected_library,packed


@pytest.mark.parametrize('factor',[1,2,3,4,8,16])
@pytest.mark.parametrize('fmt',['IQ2_S','IQ3_XXS'])
def test_exact_task_grain(selected_library,factor,fmt):
    executor=GGMLPackedCPUExecutor(selected_library)
    bank=[(packed(fmt,320,256,100+i),packed(fmt,320,256,200+i),packed('IQ4_NL',256,320,300+i)) for i in range(10)]
    x=np.random.default_rng(304).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([executor.expert(x,w) for w in bank])
    with GGMLPackedCPURowPool(executor,256,320,4,scheduler='tasks',host_help=True,tasks_per_worker=factor) as pool:
        for _ in range(3):
            np.testing.assert_array_equal(pool.submit(x,bank).result().view(np.uint32),expected.view(np.uint32))
        for invalid in (0,17):
            assert executor._bridge.mylocal_ggml_pool_set_task_factor(pool._handle,invalid)==1


@pytest.mark.parametrize('factor',[0,17,True,1.5])
def test_invalid_factor(selected_library,factor):
    with pytest.raises(ValueError,match='tasks_per_worker'):
        GGMLPackedCPURowPool(GGMLPackedCPUExecutor(selected_library),256,320,1,tasks_per_worker=factor)
