"""Opt-in fused-kernel integration; requires an explicitly built local library."""
import os

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME,asnumpy,xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.strata_prefill_experts import try_strata_prefill,library
from test_qwen4exp_chunked_prefill import gpu_packed

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(BACKEND_NAME!='cupy',reason='CuPy required')]


@pytest.mark.parametrize('rows',[1,5,64])
@pytest.mark.parametrize('resident',[(),(0,1)])
def test_fused_chunk_matches_fused_singleton_without_skipping(monkeypatch,rows,resident):
    path=os.environ.get('MINI_LLM_STRATA_PREFILL_LIBRARY')
    if not path:
        pytest.skip('Explicit Strata fused library required')
    monkeypatch.setenv('STRATA_PF_FUSED','1')
    assert library(path).mylocal_strata_prefill_width()==2560
    sources=[(gpu_packed('IQ3_S',640,2560,5700+i),gpu_packed('IQ3_S',640,2560,5800+i),
              gpu_packed('Q2_0',2560,640,5900+i)) for i in range(4)]
    managed=HeterogeneousExpertExecutor(lambda key:sources[key.expert_id],model_id='strata-kernels',
        d_model=2560,d_ff=640,compute_dtype='float32',device_budget_bytes=16*1024**2,
        cpu_workers=0,promote_after=1)
    experts=ExpertsInference(2560,640,4,expert_executor=managed)
    x=xp.asarray(np.random.default_rng(1).normal(0,.1,(1,rows,2560)),dtype=xp.float32)
    ids=xp.asarray(np.array([[2,0],[1,3],[0,1],[3,2],[1,2]])[np.arange(rows)%5][None],dtype=xp.int64)
    routes=xp.asarray(np.array([[.3,.7],[.6,.4],[.2,.8],[.4,.6],[.9,.1]],np.float32)[np.arange(rows)%5][None])
    try:
        for i in resident:
            managed._prepare(0,i,x[:,0].reshape(1,2560))
        managed.cache.set_admission_budget(0)
        before=dict(managed.cache.resident)
        expected=xp.concatenate([try_strata_prefill(experts,x[:,i:i+1],routes[:,i:i+1],ids[:,i:i+1])
                                 for i in range(rows)],axis=1)
        actual=try_strata_prefill(experts,x,routes,ids)
        xp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(asnumpy(actual),asnumpy(expected))
        assert bool(xp.isfinite(actual).all())
        assert dict(managed.cache.resident)==before
        assert managed.counters['prefill_strata_routed_rows']==rows*2*2
        assert managed.counters['prefill_streamed_bytes']>0
    finally:
        managed.close()
