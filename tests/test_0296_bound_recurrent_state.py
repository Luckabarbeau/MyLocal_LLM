"""Stable state bindings never warm caller state or reuse another request's buffers."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA graph state ownership')


@pytest.fixture(autouse=True,params=[False,True])
def copy_mode(request,monkeypatch):
    monkeypatch.setenv('MINI_LLM_GRAPH_DEVICE_COPY',str(int(request.param)))


def enqueue(x,arrays):
    arrays[0][...]+=x+arrays[1]
    return arrays[0]*2


def test_private_warmup_direct_state_replay_and_pending_close():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    x=xp.arange(64,dtype=xp.float32)
    state=xp.ones(64,xp.float32)
    extra=xp.full(64,.5,xp.float32)
    graph=CapturedArrayInference(enqueue,x,(state,extra),bound_mutable=(0,))
    np.testing.assert_array_equal(state.get(),np.ones(64,np.float32))
    assert graph.mutable[0] is state
    assert graph.mutable[1].data.ptr!=extra.data.ptr
    y=graph.forward(x,(state,extra))
    want=np.ones(64,np.float32)+np.arange(64,dtype=np.float32)+.5
    np.testing.assert_array_equal(state.get(),want)
    np.testing.assert_array_equal(y.get(),want*2)
    graph.forward(x,(state,extra))
    graph.close() # Retire GPU work before releasing state/scratch ownership.
    np.testing.assert_array_equal(state.get(),2*want-1)
    np.testing.assert_array_equal(y.get(),want*2)
    assert graph.statistics()['closed']
    graph.close()
    with pytest.raises(ValueError,match='ownership'): graph.forward(x,(state,extra))


def test_same_shape_different_request_identity_rejected():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    x=xp.ones(16,xp.float32); state=xp.zeros_like(x); extra=xp.ones_like(x)
    graph=CapturedArrayInference(enqueue,x,(state,extra),bound_mutable=(0,))
    try:
        other=xp.zeros_like(state)
        assert graph.matches(x,(state,extra))
        assert not graph.matches(x,(other,extra))
        with pytest.raises(ValueError,match='ownership'): graph.forward(x,(other,extra))
        np.testing.assert_array_equal(other.get(),np.zeros(16,np.float32))
    finally: graph.close()


@pytest.mark.parametrize('indices',[(0,0),(-1,),(2,),(True,)])
def test_invalid_binding_indices(indices):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    x=xp.ones(8,xp.float32); state=xp.zeros_like(x)
    with pytest.raises(ValueError,match='indices'):
        CapturedArrayInference(lambda a,s:s[0]+a,x,(state,),bound_mutable=indices)


def test_bound_state_cannot_alias_input_or_other_operand():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    x=xp.ones(16,xp.float32)
    with pytest.raises(ValueError,match='alias'):
        CapturedArrayInference(enqueue,x,(x,xp.zeros_like(x)),bound_mutable=(0,))
    state=xp.zeros_like(x)
    with pytest.raises(ValueError,match='alias'):
        CapturedArrayInference(enqueue,x,(state,state),bound_mutable=(0,))


def test_device_copy_bytes_alignment_tails_and_no_copyto(monkeypatch):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    monkeypatch.setenv('MINI_LLM_GRAPH_DEVICE_COPY','1')
    x=xp.arange(16,dtype=xp.float32)
    graph=CapturedArrayInference(lambda a,s:a.copy(),x,())
    try:
        rng=np.random.default_rng(296)
        for n in (0,1,15,16,17,4095,4096,4097):
            raw=rng.integers(0,256,n+1,dtype=np.uint8)
            for offset in (0,1):
                source=xp.asarray(raw)[offset:offset+n]
                target=xp.empty(n+offset,xp.uint8)[offset:]
                graph._copy(target,source)
                np.testing.assert_array_equal(target.get(),raw[offset:offset+n])
        def forbidden(*args,**kwargs): pytest.fail('Replay must not call xp.copyto')
        monkeypatch.setattr(xp,'copyto',forbidden)
        result=graph.forward(x,())
        np.testing.assert_array_equal(result.get(),x.get())
    finally: graph.close()


def test_recurrent_boundary_and_alternating_request_state_oracle(monkeypatch):
    monkeypatch.setenv('MINI_LLM_BOUND_RECURRENT_STATE','1')
    from test_0270_pre_moe_graph import test_pre_moe_boundaries_and_sessions_exact
    test_pre_moe_boundaries_and_sessions_exact()
