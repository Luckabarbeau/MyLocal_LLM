"""Exact per-site CPU/Q8 GPU split, dynamic captured epochs, retirement and failure."""
from types import SimpleNamespace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp
from mini_llm.ops.expert_residency import ExpertKey
from test_0256_ggml_packed_cpu import selected_library

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA native service staging')


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    for key,value in dict(MINI_LLM_NATIVE_MOE_SERVICE='1',MINI_LLM_NATIVE_MOE_HANDOFF='1',
        MINI_LLM_NATIVE_MOE_GRAPH='1',MINI_LLM_NATIVE_MOE_STAGE_LIMIT='1',MINI_LLM_EXPERT_GPU_Q8='1',
        MINI_LLM_GGML_POOL_SCHEDULER='tasks',MINI_LLM_GGML_POOL_HOST_HELP='1').items():
        monkeypatch.setenv(key,value)


def oracle(e,x,ids,routes,limit,layer=0,staged_override=None):
    from mini_llm.ops.quantized_linear import packed_to_device
    from mini_llm.ops.mixed_q8_gpu import GroupedQ8Plan
    pairs=sorted(zip(ids,routes))
    resident={key.expert_id for key in e.cache.resident if key.layer_id==layer}
    misses=[i for i,_ in pairs if i not in resident]
    staged=set(misses[:limit]) if len(misses)>=5 else set()
    if staged_override is not None: staged=set(staged_override)
    gpu=[(i,r) for i,r in pairs if i in resident or i in staged]
    terms={}
    if gpu:
        sources=[e._host_weights(ExpertKey(e.model_id,layer,i)) for i,_ in gpu]
        weights=[tuple(packed_to_device(w) for w in ws) for ws in sources]
        p=SimpleNamespace(d=e.d_model,f=e.d_ff,k=len(gpu),device=xp.cuda.runtime.getDevice(),
            formats=tuple(w.quant_type for w in sources[0]),
            resident_row_bytes=tuple(w.nbytes//w.shape[0] for w in sources[0]),
            selected=xp.asarray([[w.data.data.ptr for w in ws] for ws in weights],xp.uint64),
            routes=xp.asarray([r for _,r in gpu],xp.float32),
            hidden=xp.empty((len(gpu),e.d_ff),xp.float32),terms=xp.empty((len(gpu),e.d_model),xp.float32))
        GroupedQ8Plan(p).forward(x)
        values=p.terms.get()
        terms={i:values[n] for n,(i,_) in enumerate(gpu)}
    out=np.zeros((1,e.d_model),np.float32)
    hx=x.get()
    for i,r in pairs:
        if i in terms: term=terms[i]
        else:
            term=e._ggml_cpu.expert(hx,e._host_weights(ExpertKey(e.model_id,layer,i)))*np.float32(r)
        out+=term
    return out


@pytest.mark.parametrize('hits',[0,4,7,10])
@pytest.mark.parametrize('limit',[1,2])
@pytest.mark.parametrize('capture',[False,True])
def test_exact_same_site_complete_top10_dynamic_replay(selected_library,monkeypatch,hits,limit,capture):
    from test_0268_device_expert_plan import managed
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_STAGE_LIMIT',str(limit))
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH',str(int(capture)))
    e=managed(selected_library)
    ids=[19,0,511,4,7,11,3,1,13,9]
    x=xp.asarray(np.random.default_rng(293).normal(0,.1,(1,256)).astype(np.float32))
    try:
        for i in ids[:hits]: e._prepare(0,i,x)
        e.cache.set_admission_budget(0)
        initial_requests=e.cache.counters['requests']
        for repeat in range(4):
            xx=x*(1+repeat*.1)
            order=ids if repeat%2==0 else list(reversed(ids))
            routes=np.linspace(.01+repeat*.01,.1,10,dtype=np.float32)
            want=oracle(e,xx,order,routes,limit)
            y=e.try_forward_device_plan(0,xx,xp.asarray(routes),xp.asarray(order,xp.int64),512)
            actual=y.get()
            e.check_native_service()
            np.testing.assert_array_equal(actual,want)
        statistics=e.statistics()
        staged=4*limit if 10-hits>=5 else 0
        assert statistics['execution']['native_staged_experts']==staged
        assert statistics['execution']['cpu_submitted']==4*(10-hits)-staged
        assert statistics['execution']['gpu_experts']==4*hits+staged
        assert statistics['device']['requests']==initial_requests+40
        plan=e._device_expert_plans[0]
        assert plan.native_handoff.stage is not None
        service=e._ggml_row_pool._native_service
    finally:
        e.close()
    assert not service.clients and service.stage_arena is None


def test_shared_fixed_arena_across_layers_and_pending_close(selected_library):
    from test_0268_device_expert_plan import managed
    e=managed(selected_library)
    x=xp.ones((1,256),xp.float32)*.01
    routes=xp.full(10,.1,xp.float32); ids=xp.arange(10,dtype=xp.int64)
    try:
        e.cache.set_admission_budget(0)
        for i in range(12):
            e.try_forward_device_plan(i%2,x,routes,ids,512)
        plans=e._device_expert_plans
        first=plans[0].native_handoff.stage.arena
        assert plans[1].native_handoff.stage.arena is first
        assert first.device.nbytes==16*1024**2 and first.host.nbytes==16*1024**2
        service=e._ggml_row_pool._native_service
    finally:
        e.close() # No manual fence before retirement.
    assert service.handle is None and service.stage_arena is None


@pytest.mark.parametrize('eligible',[True,False])
def test_pinned_direct_source_skips_bounce_and_falls_back(selected_library,monkeypatch,eligible):
    from test_0268_device_expert_plan import managed
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_STAGE_SOURCE','pinned')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_STAGE_LIMIT','2')
    e=managed(selected_library)
    keys=[ExpertKey(e.model_id,0,i) for i in (0,1,3,4)]
    e.prepare_resident_host_source(keys,budget_bytes=2*1024**2,pin=True,reserve_bytes=0)
    x=xp.ones((1,256),xp.float32)*.01
    ids=list(range(10)) if eligible else list(range(100,110))
    routes=np.linspace(.01,.1,10,dtype=np.float32)
    try:
        e.cache.set_admission_budget(0)
        want=oracle(e,x,ids,routes,2,staged_override={0,1} if eligible else set())
        y=e.try_forward_device_plan(0,x,xp.asarray(routes),xp.asarray(ids,xp.int64),512)
        np.testing.assert_array_equal(y.get(),want)
        stage=e._device_expert_plans[0].native_handoff.stage
        # Canary proves the native worker did not copy into the bounce window.
        stage.arena.host.fill(173)
        y=e.try_forward_device_plan(0,x,xp.asarray(routes),xp.asarray(ids,xp.int64),512)
        np.testing.assert_array_equal(y.get(),want)
        e.check_native_service()
        assert np.all(stage.arena.host==173)
        assert bool(np.any(stage.host_sources))==eligible
        assert e.statistics()['execution']['native_staged_experts']==(4 if eligible else 0)
    finally: e.close()


@pytest.mark.parametrize('limit',[0,1])
@pytest.mark.parametrize('capture',[False,True])
def test_native_route_profile_exact_and_warmup_excluded(selected_library,monkeypatch,limit,capture):
    from test_0268_device_expert_plan import managed
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_STAGE_LIMIT',str(limit))
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_ROUTE_PROFILE','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH',str(int(capture)))
    e=managed(selected_library)
    x=xp.ones((1,256),xp.float32)*.01
    routes=xp.full(10,.1,xp.float32)
    expected=np.zeros((2,512),np.uint64)
    try:
        e.cache.set_admission_budget(0)
        for repeat in range(14):
            layer=repeat%2
            ids=np.arange(repeat,repeat+10,dtype=np.int64)
            expected[layer,ids]+=1
            e.try_forward_device_plan(layer,x,routes,xp.asarray(ids),512)
        stats=e.statistics()
        for h in stats['native_handoffs']:
            np.testing.assert_array_equal(h['route_heat'],expected[h['layer']])
            assert sum(h['route_heat'])==h['calls']*10
        assert e.statistics()['native_handoffs']==stats['native_handoffs']
    finally: e.close()


@pytest.mark.parametrize('capture',[False,True])
@pytest.mark.parametrize('calls',[3,2051])
def test_bounded_native_timeline_clocks_and_ring(selected_library,monkeypatch,capture,calls):
    from test_0268_device_expert_plan import managed
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_STAGE_LIMIT','0')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_TIMELINE','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH',str(int(capture)))
    e=managed(selected_library)
    x=xp.ones((1,256),xp.float32)*.01
    routes=xp.full(10,.1,xp.float32); ids=xp.arange(10,dtype=xp.int64)
    try:
        e.cache.set_admission_budget(0)
        for _ in range(calls): e.try_forward_device_plan(0,x,routes,ids,512)
        e.check_native_service()
        h=e._device_expert_plans[0].native_handoff
        snapshot=h.timeline_snapshot()
        rows=np.asarray(snapshot['rows'],np.uint64)
        total=calls+int(capture)
        assert len(rows)==min(total,2048)
        np.testing.assert_array_equal(rows[:,0],np.arange(total-len(rows)+1,total+1,dtype=np.uint64))
        assert (rows[:,2]>=rows[:,1]).all() and (rows[1:,1]>=rows[:-1,2]).all()
        assert (rows[:,3]==10).all() and (rows[:,4]==0).all()
        assert (rows[:,5]>=rows[:,1]).all() and (rows[:,8]<=rows[:,2]).all()
        assert (rows[:,6:]>=rows[:,5:-1]).all()
        before,native,after=snapshot['clock_sample']
        assert before<=native<=after
    finally: e.close()


def test_invalid_epoch_poison_and_cleanup(selected_library):
    from test_0268_device_expert_plan import managed
    e=managed(selected_library)
    x=xp.ones((1,256),xp.float32)
    routes=xp.full(10,.1,xp.float32); ids=xp.arange(10,dtype=xp.int64)
    e.cache.set_admission_budget(0)
    try:
        e.try_forward_device_plan(0,x,routes,ids,512).get()
        ids[1]=ids[0]
        result=e.try_forward_device_plan(0,x,routes,ids,512)
        assert np.isnan(result.get()).all()
        with pytest.raises(RuntimeError,match='native CPU batch failed'): e.check_native_service()
    finally:
        with pytest.raises(RuntimeError,match='invalid/failed dispatches'): e.close()
    assert e._ggml_row_pool is None
