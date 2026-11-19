"""Complete device finite checks and conditional QSA writes without layer fences."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention
from mini_llm.ops.finite_checks_gpu import finite_status_gpu,all_finite_gpu
from test_0230_qsa_gpu_decode import fixture

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Conditional QSA GPU state commit')


def test_status_primitive_still_checks_every_array():
    status=xp.zeros((),xp.uint32)
    arrays=[xp.ones(33,xp.float32) for _ in range(6)]
    for index in range(6):
        arrays[index][-1]=xp.nan
        assert int(finite_status_gpu(*arrays,out=status).get())==1
        assert not all_finite_gpu(*arrays)
        arrays[index][-1]=1
        assert int(finite_status_gpu(*arrays,out=status).get())==0


def test_deferred_commit_exact_across_positions_and_width_fallback():
    c,weights,rng=fixture()
    control=Qwen4ExpAttention(c,**weights,fast_decode=True)
    candidate=Qwen4ExpAttention(c,**weights,fast_decode=True)
    a,b=control.create_state(24),candidate.create_state(24)
    status=xp.zeros((),xp.uint32)
    for t in range(24):
        x=xp.asarray(rng.normal(size=7),dtype=xp.float32)
        expected=control.step(x,a,position=t)
        actual=candidate.step(x,b,position=t,validation_status=status)
        assert int(status.get())==0
        np.testing.assert_array_equal(actual.get(),expected.get())
        for name in ('keys','values','raw_index','pooled'):
            np.testing.assert_array_equal(getattr(a,name).get(),getattr(b,name).get())
        assert a.length==b.length==t+1


@pytest.mark.parametrize('bad_projection',['hidden','query_gate','key','value','index_key','output'])
def test_invalid_projection_never_commits_cache(bad_projection):
    c,weights,rng=fixture()
    model=Qwen4ExpAttention(c,**weights,fast_decode=True)
    state=model.create_state(8)
    x=xp.asarray(rng.normal(size=7),dtype=xp.float32)
    for t in range(3):
        model.step(x,state,position=t)
    before={name:getattr(state,name).copy() for name in ('keys','values','raw_index','pooled')}
    if bad_projection=='hidden':
        x=xp.full(7,xp.nan,xp.float32)
    else:
        weight,size=model.projections[bad_projection]
        model.projections[bad_projection]=(lambda _:xp.full(size,xp.nan,xp.float32),size)
    status=xp.zeros((),xp.uint32)
    model.step(x,state,position=3,validation_status=status)
    assert int(status.get())==1
    # Completing a pool block is conditional too, not only K/V and raw history.
    for name,expected in before.items():
        np.testing.assert_array_equal(getattr(state,name).get(),expected.get())
    # Owner must retire this failed request; speculative length is not resumable.
    assert state.length==4


def test_model_reports_first_invalid_layer_and_isolates_next_request():
    from types import SimpleNamespace
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel,Qwen4ExpLayer
    c,weights,rng=fixture()
    attention=Qwen4ExpAttention(c,**weights,fast_decode=True)
    hc=SimpleNamespace(initialize=lambda x:xp.tile(x,(4,1)),
        read=lambda x:(x[0],None),write=lambda x,y,g:x+y[None,:],collapse=lambda x:x[0])
    embedding=xp.asarray(rng.normal(size=(32,7)),dtype=xp.float32)
    head=xp.asarray(rng.normal(size=(7,32)),dtype=xp.float32)
    model=Qwen4ExpGGUFModel.from_components(width=7,vocab=32,
        blocks=[Qwen4ExpLayer(hc,attention,hc,lambda x:xp.zeros_like(x),False)],
        final_hc=hc,embedding=lambda t:embedding[t],output=lambda x:x@head,max_length=8)
    model._qsa_validation_status=xp.zeros(1,xp.uint32)
    model._qsa_validation_views={0:model._qsa_validation_status.reshape(())}
    try:
        state=model.create_generation_state(1,8)
        model.step(1,state)
        original=attention.projections['value']
        attention.projections['value']=(lambda x:xp.full(original[1],xp.nan,xp.float32),original[1])
        with pytest.raises(ValueError,match='first layer 0'):
            model.step(2,state)
        assert state.failed
        with pytest.raises(RuntimeError,match='failed'):
            model.step(3,state)
        attention.projections['value']=original
        fresh=model.create_generation_state(1,8)
        assert np.isfinite(model.step(1,fresh).get()).all()
        assert not fresh.failed and fresh.length==1
    finally:
        model.close()
