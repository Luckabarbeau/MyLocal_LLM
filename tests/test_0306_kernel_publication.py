"""Mapped input stores precede release publication; payload bits stay exact."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp
from test_0256_ggml_packed_cpu import selected_library

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual mapped CUDA memory')


@pytest.fixture(autouse=True)
def configure(monkeypatch):
    for name in ('MINI_LLM_NATIVE_MOE_HANDOFF','MINI_LLM_NATIVE_MOE_SERVICE',
                 'MINI_LLM_NATIVE_MOE_GRAPH','MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH',
                 'MINI_LLM_GRAPH_READONLY_INPUTS','MINI_LLM_GRAPH_DEVICE_COPY',
                 'MINI_LLM_GGML_POOL_HOST_HELP'):
        monkeypatch.setenv(name,'1')
    monkeypatch.setenv('MINI_LLM_GGML_POOL_SCHEDULER','tasks')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL','warp')


@pytest.mark.parametrize('hits',[0,3,10])
def test_dynamic_mixed_inputs_and_no_dma(selected_library,hits,monkeypatch):
    from test_0268_device_expert_plan import managed,oracle
    e=managed(selected_library)
    rng=np.random.default_rng(306)
    x=xp.asarray(rng.normal(0,.1,(1,256)),dtype=xp.float32)
    ids=[19,0,511,4,7,11,3,1,13,9]; routes=np.linspace(.01,.1,10,dtype=np.float32)
    try:
        for expert in ids[:hits]: e._prepare(0,expert,x)
        e.cache.set_admission_budget(0)
        for i in range(24):
            x*=xp.float32(-.97)
            current=np.roll(ids,i%10).copy(); weights=routes*(1+i*.01)
            expected=oracle(e,x,current,weights)
            device_weights=xp.asarray(weights); device_ids=xp.asarray(current,xp.int64)
            # Raw DMA input staging must not sneak back into captured replay.
            with monkeypatch.context() as context:
                context.setattr(xp.cuda.runtime,'memcpyAsync',lambda *a,**k:pytest.fail('Unexpected runtime DMA'))
                actual=e.try_forward_device_plan(0,x,device_weights,device_ids,512)
            np.testing.assert_array_equal(actual.get().view(np.uint32),expected.get().view(np.uint32))
        e.check_native_service()
        handoff=e._device_expert_plans[0].native_handoff
        assert handoff.kernel_publish and handoff.graph.calls==24
        np.testing.assert_array_equal(handoff.plan.host_input.view(np.uint32),x.get().view(np.uint32))
        np.testing.assert_array_equal(handoff.plan.host_metadata[:10],sorted(ids))
        assert e.statistics()['native_handoffs'][0]['kernel_input_publication']
    finally: e.close()


def test_publication_copies_bitpatterns_and_failure_is_closed():
    from mini_llm.ops.native_expert_service import _mapped,_kernels
    publish=_kernels(xp.cuda.runtime.getDevice())[-1]
    source=np.random.default_rng(306).integers(0,2**32,257,dtype=np.uint32)
    source[:6]=[0,0x80000000,0x7f800000,0xff800000,0x7fc00123,1]
    x=xp.asarray(source); metadata=xp.arange(11,dtype=xp.int32)
    target=_mapped(4*257); host_metadata=_mapped(4*11); flags=_mapped(128); failure=_mapped(64)
    for request in range(1,17):
        publish((1,),(256,),(x,metadata,target.device,host_metadata.device,flags.device,failure.device,np.int32(257),np.int32(10)))
        xp.cuda.get_current_stream().synchronize()
        assert flags.host[0]==request
        np.testing.assert_array_equal(target.host,source)
        np.testing.assert_array_equal(host_metadata.host,np.arange(11,dtype=np.uint32))
    failure.host[0]=2
    target.host.fill(17)
    publish((1,),(256,),(x,metadata,target.device,host_metadata.device,flags.device,failure.device,np.int32(257),np.int32(10)))
    xp.cuda.get_current_stream().synchronize()
    assert flags.host[0]==16 and (target.host==17).all()
