"""Native descriptor reuse must preserve source identity and packed arithmetic."""
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import executor, selected_library
from test_0259_ggml_row_pool import bundle


@pytest.mark.parametrize('host_help',[False,True])
def test_registered_batches_exact_and_no_projection_revalidation(executor,host_help,monkeypatch):
    sources=[bundle(i) for i in range(10)]
    x=np.random.default_rng(271).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([executor.expert(x,ws) for ws in sources])
    with GGMLPackedCPURowPool(executor,256,320,4,host_help=host_help,spin_us=500) as pool:
        indices=np.array([pool.register(i,ws) for i,ws in enumerate(sources)],np.int32)
        assert pool.register(0,sources[0])==int(indices[0])
        def forbidden(*args):
            raise AssertionError('Registered submit must not rebuild projection descriptors')
        monkeypatch.setattr(executor,'_projection',forbidden)
        for step in range(50):
            count=1+step%10
            out=np.empty((count,256),np.float32)
            job=pool._submit_registered(x,indices[:count],out)
            np.testing.assert_array_equal(job.result(),expected[:count])


def test_registry_identity_busy_and_bounds(executor):
    ws=bundle()
    x=np.ones((1,256),np.float32)
    out=np.empty_like(x)
    with GGMLPackedCPURowPool(executor,256,320,1,host_help=True) as pool:
        index=pool.register('expert',ws)
        with pytest.raises(ValueError,match='source changed'):
            pool.register('expert',bundle(1))
        for invalid in (-1,32768):
            with pytest.raises(RuntimeError,match='code=1'):
                pool._submit_registered(x,np.array([invalid],np.int32),out)
        job=pool._submit_registered(x,np.array([index],np.int32),out)
        with pytest.raises(RuntimeError,match='idle'):
            pool.register('next',ws)
        job.result()
        with pytest.raises(ValueError,match='slab'):
            pool._submit_registered(x,np.array([index],np.int64),out)
