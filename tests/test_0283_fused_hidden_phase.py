"""Bounded native phase fusion: unchanged quantizers, dots and output ownership."""
import ctypes as C
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import executor,selected_library
from test_0259_ggml_row_pool import bundle


@pytest.mark.parametrize('workers',[1,4,19,23])
@pytest.mark.parametrize('count',[1,3,10])
@pytest.mark.parametrize('help',[False,True])
def test_fused_hidden_exact_epochs(executor,workers,count,help):
    weights=[bundle(i) for i in range(count)]
    inputs=np.random.default_rng(283).normal(0,.1,(4,256)).astype(np.float32)
    expected=[np.concatenate([executor.expert(x[None],w) for w in weights]) for x in inputs]
    with GGMLPackedCPURowPool(executor,256,320,workers,scheduler='tasks',
            fused_hidden=True,host_help=help,spin_us=50) as pool:
        assert pool.fused_hidden
        ids=np.asarray([pool.register(i,w) for i,w in enumerate(weights)],dtype=np.int32)
        for epoch in range(80):
            x=inputs[epoch%4:epoch%4+1]
            job=(pool._submit_registered(x,ids,np.empty((count,256),np.float32))
                 if epoch%2 else pool.submit(x,weights))
            np.testing.assert_array_equal(job.result(),expected[epoch%4])
        saved=job.result().copy()
        pool.submit(inputs[:1],weights).result()
        np.testing.assert_array_equal(job.result(),saved)


def test_configuration_validation(executor,monkeypatch):
    for value in (0,'1',None):
        if value is None:
            monkeypatch.setenv('MINI_LLM_GGML_POOL_FUSED_HIDDEN','bad')
            with pytest.raises(ValueError,match='FUSED_HIDDEN'):
                GGMLPackedCPURowPool(executor,256,320,1)
        else:
            with pytest.raises(TypeError,match='fused_hidden'):
                GGMLPackedCPURowPool(executor,256,320,1,fused_hidden=value)
    monkeypatch.delenv('MINI_LLM_GGML_POOL_FUSED_HIDDEN')
    bridge=executor._bridge
    assert bridge.mylocal_ggml_pool_set_fused_hidden(None,1)==1
    with GGMLPackedCPURowPool(executor,256,320,1) as pool:
        assert bridge.mylocal_ggml_pool_set_fused_hidden(pool._handle,2)==1
