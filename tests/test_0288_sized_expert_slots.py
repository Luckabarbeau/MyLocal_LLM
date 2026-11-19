"""Sized fixed slots: header budgets, compatible replacement, and exact weights."""
from collections import Counter
import gc
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp,asnumpy
from mini_llm.ops.expert_slot_layout import sized_slot_layout
from mini_llm.ops.expert_residency import AdaptiveExpertCache,ExpertKey,HeterogeneousExpertExecutor


def test_layout_bounded_deterministic_fractions():
    sizes=[1000]*10+[2100]*5+[3000]
    layout=sized_slot_layout(1024*1024,sizes,512)
    assert sum(layout)<=1024*1024
    assert layout==sized_slot_layout(1024*1024,list(reversed(sizes)),512)
    assert all(s%256==0 for s in layout)
    assert len(layout)>(1024*1024)//3072
    counts=Counter(layout)
    fractions=[counts[s]/(n*512) for s,n in Counter((n+255)//256*256 for n in sizes).items()]
    assert max(fractions)-min(fractions)<.004
    for args in [(0,[100],1),(100,[],1),(100,[True],1)]:
        with pytest.raises(ValueError): sized_slot_layout(*args)


def test_compatible_eviction_with_spare_global_bytes():
    cache=AdaptiveExpertCache(1000,promote_after=1,victim_policy='heap')
    a,b,c=[ExpertKey('288',0,i) for i in range(3)]
    cache.request(a,100,lambda:object()); cache.request(b,200,lambda:object())
    for _ in range(3):
        cache.request(c,100,lambda:object(),needs_slot=True,
            compatible_victim=lambda k:cache.resident[k][1]==100)
        if c in cache.resident: break
    assert c in cache.resident and b in cache.resident and a not in cache.resident
    assert cache.used_bytes==300


@pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual GPU slot arena')
def test_mixed_size_cache_bytes_replacement_and_protected_view():
    from test_0238_packed_expert_slots import weights,stride
    from mini_llm.ops.quantized_linear import PackedExpertFFNInference,packed_to_device
    small=('IQ2_S','IQ2_S','Q2_0'); large=('IQ4_XS','IQ4_XS','IQ4_NL')
    bank={i:weights(large if i%2 else small,seed=288+i) for i in range(6)}
    s,l=stride(bank[0]),stride(bank[1])
    budget=2*s+l
    e=HeterogeneousExpertExecutor(lambda key:bank[key.expert_id],model_id='288',
        d_model=256,d_ff=256,compute_dtype='float32',device_budget_bytes=budget,
        expert_slot_stride_bytes=l,expert_slot_sizes=(s,s,l),promote_after=1,cpu_workers=1)
    x=xp.asarray(np.random.default_rng(288).normal(0,.1,(1,256)).astype(np.float32))
    try:
        for i in (0,2,1):
            got=e.forward(0,i,x)
            ref=PackedExpertFFNInference(*(packed_to_device(w) for w in bank[i])).forward(x)
            np.testing.assert_array_equal(asnumpy(got),asnumpy(ref))
        assert e.cache.used_bytes==e.resident_device_bytes==budget
        assert e.statistics()['slots']['capacity']==3
        assert not e.cache.counters['allocation_failures']
        for _ in range(4): e._prepare(0,4,x)
        assert ExpertKey('288',0,4) in e.cache.resident
        assert ExpertKey('288',0,1) in e.cache.resident # larger class unaffected
        xp.cuda.get_current_stream().synchronize()
        old=e.cache.resident[ExpertKey('288',0,1)][0].gate.data[:]
        e.cache.retain_only([k for k in e.cache.resident if k.expert_id!=1])
        for _ in range(4):
            assert e._prepare(0,3,x)[1] is None
        assert e.cache.counters['cpu_fallbacks']>=4 # no compatible free/reclaimable slot
        del old; gc.collect()
        assert e._prepare(0,3,x)[1] is not None
        assert e.cache.used_bytes<=budget
    finally: e.close(); e.cache.clear()
