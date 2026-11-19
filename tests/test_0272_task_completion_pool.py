"""Generation-tagged task completion, not all-worker phase rendezvous."""
import numpy as np
import pytest
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import executor, selected_library
from test_0259_ggml_row_pool import bundle


@pytest.mark.parametrize('workers',[1,4,15])
@pytest.mark.parametrize('host_help',[False,True])
@pytest.mark.parametrize('spin',[0,500])
def test_task_epochs_mixed_formats_exact(executor,workers,host_help,spin):
    sources=[bundle(i) for i in range(10)]
    x=np.random.default_rng(272).normal(0,.1,(1,256)).astype(np.float32)
    expected=np.concatenate([executor.expert(x,w) for w in sources])
    with GGMLPackedCPURowPool(executor,256,320,workers,host_help=host_help,
                             spin_us=spin,scheduler='tasks') as pool:
        indices=np.array([pool.register(i,w) for i,w in enumerate(sources)],np.int32)
        for epoch in range(200):
            count=1+epoch%10
            out=np.empty((count,256),np.float32)
            job=(pool._submit_registered(x,indices[:count],out) if epoch%2 else
                 pool.submit(x,sources[:count],out=out))
            np.testing.assert_array_equal(job.result(),expected[:count])


def test_unknown_scheduler_rejected(executor):
    with pytest.raises(ValueError,match='scheduler'):
        GGMLPackedCPURowPool(executor,256,320,1,scheduler='unknown')
