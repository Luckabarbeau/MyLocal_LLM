"""Native service epochs, captured replay, failure/timeout, and safe retirement."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp
from test_0256_ggml_packed_cpu import selected_library,executor

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA mapped doorbells')


@pytest.fixture(autouse=True)
def service_mode(monkeypatch):
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_SERVICE','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_HANDOFF','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH','1')
    monkeypatch.setenv('MINI_LLM_GGML_POOL_SCHEDULER','tasks')
    monkeypatch.setenv('MINI_LLM_GGML_POOL_HOST_HELP','1')


def test_shared_service_dynamic_layers_replay_and_close(selected_library):
    from test_0268_device_expert_plan import managed,oracle
    e=managed(selected_library)
    x=xp.asarray(np.random.default_rng(287).normal(0,.1,(1,256)).astype(np.float32))
    ids=list(range(10)); routes=np.linspace(.01,.1,10,dtype=np.float32)
    try:
        e.cache.set_admission_budget(0)
        want=oracle(e,x,ids,routes)
        saved=[]
        for i in range(32):
            y=e.try_forward_device_plan(i%2,x,xp.asarray(routes),xp.asarray(ids,xp.int64),512)
            saved.append(y)
        e.check_native_service()
        service=e._ggml_row_pool._native_service
        assert len(service.clients)==2
        for y in saved: np.testing.assert_array_equal(y.get(),want.get())
        for entry in service.clients.values():
            assert entry.host[0]==entry.host[16]==17 # one warmup +16 dynamic replays
        assert e.statistics()['execution']['native_handoff_dispatches']==32
    finally: e.close()
    assert service.handle is None and not service.clients


@pytest.mark.parametrize('failure',[1,3])
def test_bounded_wait_and_epoch_exhaustion(executor,failure):
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
    from mini_llm.ops.native_expert_service import NativeExpertService,_mapped
    with GGMLPackedCPURowPool(executor,256,320,1) as pool:
        service=NativeExpertService(pool)
        pool._native_service=service
        flags=_mapped(128)
        flags.host[0]=1 if failure==1 else 0xffffffff
        if failure==1:
            service.wait((1,),(32,),(flags.device,service.failure.device,np.uint64(1_000_000),service.device_failure))
        else:
            service.publish((1,),(32,),(flags.device,service.failure.device))
            service.wait((1,),(32,),
                (flags.device,service.failure.device,np.uint64(1_000_000),service.device_failure))
        out=xp.ones(256,xp.float32)
        # An incomplete CPU slab must not be touched, even if its pointer is
        # unavailable. A failed epoch leaves the destination unconsumed.
        service.copy_rows((1,),(256,),
            (np.uint64(0),out,service.device_failure,np.int32(256)))
        np.testing.assert_array_equal(out.get(),np.ones(256,np.float32))
        service.poison((1,),(256,),(out,service.device_failure,np.int32(256)))
        assert np.isnan(out.get()).all()
        assert int(service.failure.host[0])==failure
        with pytest.raises(RuntimeError,match='timed out' if failure==1 else 'epoch exhausted'):
            service.check()


def test_model_rejects_failure_before_advancing_state(monkeypatch):
    from types import SimpleNamespace
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel,Qwen4ExpLayer
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    hc=SimpleNamespace(initialize=lambda x:xp.tile(x,(4,1)),read=lambda x:(x[0],None),
        write=lambda x,y,g:x+y[None,:],collapse=lambda x:x[0])
    mixer=SimpleNamespace(create_state=lambda n:SimpleNamespace(conv=SimpleNamespace(length=0),
        delta=SimpleNamespace(length=0)),forward=lambda x,state:x)
    model=Qwen4ExpGGUFModel.from_components(width=7,vocab=8,
        blocks=[Qwen4ExpLayer(hc,mixer,hc,lambda x:xp.zeros_like(x),True)],
        final_hc=hc,embedding=lambda t:xp.ones(7,xp.float32),
        output=lambda x:xp.ones(8,xp.float32),max_length=8)
    e=HeterogeneousExpertExecutor(lambda key:None,model_id='287',d_model=7,d_ff=7,compute_dtype='float32')
    model.expert_executor=e
    def fail(): raise RuntimeError('Native expert service failed: injected')
    monkeypatch.setattr(e,'check_native_service',fail)
    state=model.create_generation_state(1,8)
    try:
        with pytest.raises(RuntimeError,match='service failed'): model.step(1,state)
        assert state.failed and state.length==0
    finally:
        e.close(); model.close()
