"""Shared branch captured after native publication; complete mixed routing retained."""
from types import SimpleNamespace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp
from test_0256_ggml_packed_cpu import selected_library

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual native CUDA service')


@pytest.fixture(autouse=True)
def configuration(monkeypatch):
    for name in ('MINI_LLM_NATIVE_MOE_SERVICE','MINI_LLM_NATIVE_MOE_HANDOFF',
                 'MINI_LLM_NATIVE_MOE_GRAPH','MINI_LLM_NATIVE_MOE_FUSED_SHARED',
                 'MINI_LLM_GRAPH_READONLY_INPUTS','MINI_LLM_GRAPH_DEVICE_COPY',
                 'MINI_LLM_GGML_POOL_HOST_HELP'):
        monkeypatch.setenv(name,'1')
    monkeypatch.setenv('MINI_LLM_GGML_POOL_SCHEDULER','tasks')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL','warp')


@pytest.mark.parametrize('hits',[0,3,10])
def test_exact_shared_mixed_routes_and_graph_retirement(selected_library,hits):
    from test_0268_device_expert_plan import managed,oracle
    from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
    from mini_llm.ops.experts_inference import ExpertsInference
    e=managed(selected_library)
    rng=np.random.default_rng(305)
    hidden=xp.asarray(rng.normal(0,.1,256),dtype=xp.float32)
    ids=[19,0,511,4,7,11,3,1,13,9]
    weights=np.linspace(.01,.1,10,dtype=np.float32)
    routes=xp.asarray(weights).reshape(1,1,10)
    indices=xp.asarray(ids,xp.int64).reshape(1,1,10)
    router=SimpleNamespace(forward=lambda x:(routes,indices))
    experts=ExpertsInference(256,320,512,expert_executor=e)
    moe=Qwen4ExpMoE(router,experts,lambda x:x*xp.float32(.3),lambda x:x*xp.float32(-.7),
        lambda x:x*xp.float32(.2),xp.asarray(rng.normal(0,.01,256),dtype=xp.float32),
        fast_elementwise=True,overlap_shared=True,shared_graph_decode=True,defer_shared=True)
    try:
        for expert in ids[:hits]: e._prepare(0,expert,hidden.reshape(1,-1))
        e.cache.set_admission_budget(0)
        saved=[]
        for i in range(6):
            hidden*=xp.float32(-.9)
            routes*=xp.float32(.97)
            expected_routed=oracle(e,hidden.reshape(1,-1),ids,routes.get().reshape(-1))
            expected=moe.combine_gpu_branch(expected_routed,moe.enqueue_gpu_branch(hidden)).get().reshape(-1)
            result=moe(hidden)
            np.testing.assert_array_equal(result.get().view(np.uint32),expected.view(np.uint32))
            saved.append((result,expected.copy()))
        e.check_native_service()
        handoff=e._device_expert_plans[0].native_handoff
        assert handoff.graph_branch is moe and handoff.graph.calls==6
        assert moe._shared_graph is None # no redundant shared graph/events in this path
        assert e.statistics()['execution']['native_handoff_dispatches']==6
        for result,expected in saved:
            np.testing.assert_array_equal(result.get(),expected)
    finally: e.close()
    assert handoff.graph is None and handoff.graph_branch is None and handoff.handle is None


def test_declines_fused_branch_without_native_service(selected_library,monkeypatch):
    from test_0268_device_expert_plan import managed
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_SERVICE','0')
    e=managed(selected_library)
    try:
        e.cache.set_admission_budget(0)
        assert e.try_forward_device_plan(0,xp.ones((1,256),xp.float32),
            xp.ones(10,xp.float32),xp.arange(10,dtype=xp.int64),512,gpu_branch=object()) is None
    finally: e.close()
