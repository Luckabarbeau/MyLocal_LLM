"""0080BJ zero-param contextual cross-attention correctness tests (CPU)."""
import numpy as np
import pytest
from mini_llm.backend import xp
from mini_llm.block_proposal.context_attention import (
    context_attention_forward, context_attention_backward,
)


def test_shape_mask_and_no_future_access():
    rng=np.random.default_rng(7)
    q=xp.asarray(rng.normal(size=(3,8,16)).astype('float32'))
    m=xp.asarray(rng.normal(size=(3,5,16)).astype('float32'))
    valid=xp.asarray([[0,0,1,1,1],[0,0,0,0,1],[1,1,1,1,1]], dtype=xp.bool_)
    y,cache=context_attention_forward(q,m,valid)
    altered=m.copy()
    altered[~valid]=float(1e7)
    y2,_=context_attention_forward(q,altered,valid)
    assert np.allclose(np.asarray(y),np.asarray(y2),rtol=1e-6,atol=1e-6)
    dy=xp.asarray(rng.normal(size=(3,8,16)).astype('float32'))
    dq,dm=context_attention_backward(dy,cache)
    assert dq.shape==q.shape and dm.shape==m.shape
    assert np.allclose(np.asarray(dm[~valid]),0,atol=1e-6)


def test_exact_manual_backward_finite_difference():
    rng=np.random.default_rng(23)
    q=xp.asarray(rng.normal(size=(2,4,8)).astype('float32'))
    m=xp.asarray(rng.normal(size=(2,5,8)).astype('float32'))
    valid=xp.asarray([[0,1,1,1,1],[1,1,0,0,0]], dtype=xp.bool_)
    upstream=xp.asarray(rng.normal(size=q.shape).astype('float32'))
    y,c=context_attention_forward(q,m,valid,gain=.2)
    dq,dm=context_attention_backward(upstream,c)
    eps=1.e-3
    def scalar():
        o,_=context_attention_forward(q,m,valid,gain=.2)
        return float(xp.sum(o*upstream))
    for idx in [(0,0,0),(0,1,5),(1,3,2)]:
        old=float(q[idx]); q[idx]=old+eps; a=scalar(); q[idx]=old-eps; b=scalar(); q[idx]=old
        assert np.isclose(float(dq[idx]),(a-b)/(2*eps),atol=3e-3,rtol=3e-3)
    for idx in [(0,1,0),(0,3,4),(1,0,2)]:
        old=float(m[idx]);m[idx]=old+eps;a=scalar();m[idx]=old-eps;b=scalar();m[idx]=old
        assert np.isclose(float(dm[idx]),(a-b)/(2*eps),atol=3e-3,rtol=3e-3)


def test_reject_empty_context():
    with pytest.raises(ValueError,match='at least one'):
        context_attention_forward(xp.zeros((2,8,16)),xp.zeros((2,5,16)),xp.zeros((2,5),dtype=xp.bool_))


def test_causal_context_gather_stops_at_boundary_even_with_routed_prefix():
    from mini_llm.block_proposal.context_attention import gather_causal_working_context
    hidden=xp.asarray(np.arange(2*100*4).reshape(2,100,4),dtype=xp.float32)
    rows=xp.asarray([0,0,1,1],dtype=xp.int64)
    # routed prefix occupies [0,9]; working positions start at 10
    boundaries=xp.asarray([10,11,35,78],dtype=xp.int64)
    mem,mask,ids=gather_causal_working_context(hidden,rows,boundaries,10,32)
    assert mem.shape==(4,32,4)
    assert np.array_equal(np.asarray(ids)[:,-1],np.asarray(boundaries))
    assert np.all(np.asarray(ids) <= np.asarray(boundaries)[:,None])
    assert np.all(np.asarray(ids) >= 10)
    assert np.array_equal(np.asarray(mask).sum(axis=1),[1,2,26,32])
    assert np.array_equal(np.asarray(mem)[:, -1],np.asarray(hidden)[np.asarray(rows),np.asarray(boundaries)])
