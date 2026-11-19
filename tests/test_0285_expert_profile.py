"""Observed profile placement, bounded slots, and in-flight protection."""
from types import SimpleNamespace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp,asnumpy
from mini_llm.ops.expert_profile import frequency_placement
from mini_llm.ops.expert_residency import AdaptiveExpertCache,ExpertKey,HeterogeneousExpertExecutor


def key(e): return ExpertKey('0285',0,e)


def test_frequency_budget_order_and_invalid():
    counts={key(1):20,key(3):7,key(2):7,key(0):1}
    assert frequency_placement(counts,300,100)==(key(1),key(2),key(3))
    assert frequency_placement(dict(reversed(list(counts.items()))),300,100)==frequency_placement(counts,300,100)
    assert frequency_placement(counts,0,100)==()
    for value in (-1,np.nan,np.inf,True):
        with pytest.raises(ValueError): frequency_placement({key(0):value},100,100)
    with pytest.raises(TypeError): frequency_placement({0:1},100,100)


def test_retain_only_preserves_consumers_and_heat():
    c=AdaptiveExpertCache(300,promote_after=1,victim_policy='heap')
    for i in range(3): c.request(key(i),100,lambda:object())
    c.protect([key(0)])
    with pytest.raises(RuntimeError,match='protected'): c.retain_only([key(1)])
    assert c.used_bytes==300
    c.unprotect([key(0)])
    assert c.retain_only([key(1)])==2
    assert c.used_bytes==100 and len(c._usage)==3
    assert c.request(key(2),100,lambda:object()) is not None


@pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual GPU slots')
def test_profile_replaces_complete_slots_then_freezes():
    from test_0238_packed_expert_slots import weights,stride
    bank={i:weights(seed=285+i) for i in range(8)}
    step=stride(bank[0])
    e=HeterogeneousExpertExecutor(lambda k:bank[k.expert_id],model_id='0285',
        d_model=256,d_ff=256,compute_dtype='float32',device_budget_bytes=2*step,
        expert_slot_stride_bytes=step,promote_after=1,cpu_workers=1)
    try:
        x=xp.ones((1,256),xp.float32)*.01
        # Establish hot-but-currently-nonresident experts using real observed heat.
        for i in (1,2,3,4,1,1,2,2): e._prepare(0,i,x)
        for i in range(8):
            usage=e.cache._usage.get(key(i))
            if usage: usage.hits=100 if i in (1,2) else 1
        report=e.install_observed_profile()
        assert set(e.cache.resident)=={key(1),key(2)}
        assert report['selected']==report['resident']==2
        for k,(expert,_) in e.cache.resident.items():
            for a,b in zip((expert.gate,expert.up,expert.down),bank[k.expert_id]):
                np.testing.assert_array_equal(asnumpy(a.data),b.data.reshape(-1))
        e.cache.set_admission_budget(0)
        with pytest.raises(ValueError,match='before admission'): e.install_observed_profile()
    finally:
        e.close(); e.cache.clear()
