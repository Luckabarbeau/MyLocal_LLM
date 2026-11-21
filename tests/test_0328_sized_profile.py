"""Setup-only sized profiles: pure selection, preflight, and real slot parity."""
from collections import Counter
from concurrent.futures import Future
from hashlib import sha256
from threading import Event
from types import MappingProxyType,SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME,xp,asnumpy
from mini_llm.ops import expert_residency as residency
from mini_llm.ops.expert_profile import _validated_counts,frequency_placement,sized_frequency_placement
from mini_llm.ops.expert_residency import ExpertKey,HeterogeneousExpertExecutor
from mini_llm.ops.quantized_weight import PackedGGUFWeight

GPU=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual mixed-size CUDA slot arena')


def key(i): return ExpertKey('0328',0,i)


def test_cpu_exact_class_budgets_alignment_and_resident_ties():
    counts={key(0):10,key(1):10,key(2):10,key(3):1000,key(4):999,key(5):0}
    sizes={key(0):1,key(1):256,key(2):255,key(3):257,key(4):513,key(5):1}
    before=dict(counts)
    selected=sized_frequency_placement(MappingProxyType(counts),sizes,(768,256,256),resident=(key(2),))
    assert selected==(key(2),key(0),key(3))
    assert selected==sized_frequency_placement(dict(reversed(list(counts.items()))),
        sizes,(256,768,256),resident=(key(2),))
    assert counts==before
    assert sum(next(s for s in (256,768) if sizes[k]<=s) for k in selected)==1280
    # Unused large slots cannot increase the small class's capacity.
    assert sized_frequency_placement({key(i):1 for i in range(3)},
        {key(i):1 for i in range(3)},(256,768,768))==(key(0),)
    assert sized_frequency_placement({key(0):0},{},(256,))==()
    assert sized_frequency_placement({}, {},(256,))==()
    # The established fixed-stride API still resolves ties by key alone.
    assert frequency_placement(counts,512,256)==(key(3),key(4))


def test_cpu_selection_maximizes_frequency_separately_per_class():
    from itertools import combinations
    sizes={key(i):size for i,size in enumerate((256,250,1,257,500,700,768))}
    counts={key(i):count for i,count in enumerate((3,11,7,100,2,19,21))}
    selected=sized_frequency_placement(counts,sizes,(256,256,768,768))
    assert Counter(256 if sizes[k]<=256 else 768 for k in selected)=={256:2,768:2}
    feasible=[subset for subset in combinations(counts,4)
        if Counter(256 if sizes[k]<=256 else 768 for k in subset)=={256:2,768:2}]
    assert sum(counts[k] for k in selected)==max(sum(counts[k] for k in subset) for subset in feasible)


@pytest.mark.parametrize('count',[-1,True,np.bool_(True),np.nan,np.inf,-np.inf,1j,'1'])
def test_cpu_invalid_frequencies(count):
    with pytest.raises(ValueError,match='frequencies'):
        sized_frequency_placement({key(0):count},{key(0):1},(256,))


@pytest.mark.parametrize('sizes,slots',[
    ({key(0):0},(256,)),({key(0):-1},(256,)),({key(0):True},(256,)),
    ({key(0):1.5},(256,)),({key(0):769},(256,768)),({},(256,)),
    ({key(0):1},()),({key(0):1},(255,)),({key(0):1},(True,)),
    ({key(0):1},(256.0,)),({key(0):1},(0,)),
])
def test_cpu_invalid_complete_sizes_and_slot_layout(sizes,slots):
    with pytest.raises(ValueError): sized_frequency_placement({key(0):1},sizes,slots)


def test_cpu_count_snapshot_and_identity_validation():
    counts={key(0):np.int64(2),key(1):np.float64(.5)}
    snapshot=_validated_counts(MappingProxyType(counts),model_id='0328')
    snapshot[key(0)]=99
    assert counts[key(0)]==2
    unsigned={key(0):np.uint64(10),key(1):5}
    assert sized_frequency_placement(unsigned,{key(0):1,key(1):1},(256,))==(key(0),)
    assert isinstance(unsigned[key(0)],np.uint64)
    for bad in ({0:1},[(key(0),1)]):
        with pytest.raises(TypeError): _validated_counts(bad,model_id='0328')
    for bad in (ExpertKey('other',0,0),ExpertKey('0328',-1,0),
                ExpertKey('0328',0,True),ExpertKey('0328',0,.5)):
        with pytest.raises(ValueError): _validated_counts({bad:1},model_id='0328')


def test_cpu_setup_guards_before_any_source_read(monkeypatch):
    reads=[]
    e=HeterogeneousExpertExecutor(lambda k:reads.append(k),model_id='0328',
        d_model=256,d_ff=256,device_budget_bytes=1024,expert_slot_stride_bytes=512,
        expert_slot_sizes=(256,256,512),cpu_workers=1)
    monkeypatch.setattr(residency,'BACKEND_NAME','cupy')
    future=Future()
    try:
        cases=(
            ('_dispatch_depth',1,ValueError),('_device_expert_plans',{0:object()},ValueError),
            ('_resident_source',object(),ValueError),
            ('_ggml_row_pool',SimpleNamespace(_pending=object(),_native_service=None),RuntimeError),
            ('_ggml_row_pool',SimpleNamespace(_pending=None,_native_service=object()),ValueError),
        )
        for attr,value,error in cases:
            old=getattr(e,attr)
            setattr(e,attr,value)
            try:
                with pytest.raises(error): e.install_observed_profile(counts={key(0):1})
            finally: setattr(e,attr,old)
        for limit in (0,1):
            e.cache.set_admission_budget(limit)
            with pytest.raises(ValueError,match='before admission'): e.install_observed_profile()
        e.cache.set_admission_budget(None)
        e._cpu_futures.add(future)
        with pytest.raises(RuntimeError,match='pending CPU'): e.install_observed_profile()
        future.set_result(None)
        e.cache.protect([key(0)])
        with pytest.raises(RuntimeError,match='protected'): e.install_observed_profile()
        e.cache.unprotect([key(0)])
        e.cache._inflight_groups.append((SimpleNamespace(query=lambda:False),frozenset((key(0),))))
        with pytest.raises(RuntimeError,match='in-flight'): e.install_observed_profile()
        e.cache._inflight_groups.clear()
        for counts in ({ExpertKey('wrong',0,0):1},{key(0):-1},{key(0):np.nan}):
            with pytest.raises(ValueError): e.install_observed_profile(counts=counts)
        with pytest.raises(TypeError): e.install_observed_profile(counts=[(key(0),1)])
        assert not reads and not e.cache.resident and e._slot_arena is None
    finally:
        e.close(); e.cache.clear()


def test_cpu_submitted_work_is_tracked_until_completion(monkeypatch):
    e=HeterogeneousExpertExecutor(lambda k:pytest.fail('No source reads'),model_id='0328',
        d_model=256,d_ff=256,device_budget_bytes=256,expert_slot_stride_bytes=256,cpu_workers=1)
    gate=Event()
    job=e._submit_cpu(gate.wait)
    monkeypatch.setattr(residency,'BACKEND_NAME','cupy')
    try:
        assert job in e._cpu_futures and not job.done()
        with pytest.raises(RuntimeError,match='pending CPU'): e.install_observed_profile(counts={})
        gate.set(); job.result(timeout=5)
        assert e.install_observed_profile(counts={})['selected']==0
    finally:
        gate.set(); e.close(); e.cache.clear()


@pytest.fixture
def sources(tmp_path):
    from test_0228_iq_gpu_integration import packed
    formats=(('IQ2_S','IQ2_S','Q2_0'),('IQ4_XS','IQ4_XS','IQ4_NL'))
    triples={key(i):tuple(packed(fmt,256,256,328+i*3+p)
        for p,fmt in enumerate(formats[i%2])) for i in range(8)}
    path=tmp_path/'canonical.bin'
    with path.open('wb') as f:
        for triple in triples.values():
            for w in triple: f.write(w.data.tobytes())
    mmap=np.memmap(path,mode='r',dtype=np.uint8)
    bank={}
    offset=0
    for k,triple in triples.items():
        views=[]
        for w in triple:
            views.append(PackedGGUFWeight(mmap[offset:offset+w.nbytes],w.shape,w.quant_type))
            offset+=w.nbytes
        bank[k]=tuple(views)
    s,l=((sum(w.nbytes for w in bank[key(i)])+255)//256*256 for i in (0,1))
    return bank,mmap,(s,s,l)


def executor(bank,slots):
    return HeterogeneousExpertExecutor(bank.__getitem__,model_id='0328',d_model=256,d_ff=256,
        compute_dtype='float32',device_budget_bytes=sum(slots)+255,
        expert_slot_stride_bytes=max(slots),expert_slot_sizes=slots,promote_after=1,
        expert_victim_policy='heap',cpu_workers=1)


@GPU
def test_gpu_mixed_profile_exact_bytes_same_forward_site_and_immutable_mmap(sources):
    bank,mmap,slots=sources
    original=sha256(mmap).hexdigest()
    e=executor(bank,slots)
    x=xp.asarray(np.random.default_rng(328).normal(0,.04,(1,256)).astype(np.float32))
    targets=(0,4,3)
    try:
        e.install_observed_profile(counts={key(i):10 for i in targets})
        # Baseline and post-install outputs use the identical executor.forward
        # execution site, precision, shapes, and packed GPU kernels (not CPU).
        expected={i:asnumpy(e.forward(0,i,x)) for i in targets}
        xp.cuda.get_current_stream().synchronize()
        e.install_observed_profile(counts={key(i):10 for i in (0,2,1)})
        arena=e._slot_arena
        kept=e.cache.resident[key(0)][0]
        counts={key(0):50,key(4):50,key(6):50,key(3):100,key(5):100,key(1):2}
        before=dict(counts)
        heat={k:(v.score,v.last_step,v.hits) for k,v in e.cache._usage.items()}
        step=e.cache._step
        report=e.install_observed_profile(counts=MappingProxyType(counts))
        assert counts==before and report['count_source']=='supplied'
        assert set(e.cache.resident)=={key(i) for i in targets}
        assert report['selected']==report['resident']==3
        assert report['retained']==1 and report['evicted']==report['promotions']==2
        assert e.cache.resident[key(0)][0] is kept and e._slot_arena is arena
        assert e.cache._step==step
        assert all((e.cache._usage[k].score,e.cache._usage[k].last_step,e.cache._usage[k].hits)==v
                   for k,v in heat.items())
        assert e.cache._admission_limit is None and e.cache.promote_after==1
        payload=lambda i:sum(w.nbytes for w in bank[key(i)])
        assert report['selected_bytes']==sum(payload(i) for i in targets)
        assert report['retained_bytes']==payload(0)
        assert report['replaced_bytes']==payload(4)+payload(3)
        assert report['selected_slot_bytes']==e.cache.used_bytes==sum(slots)
        assert report['retained_slot_bytes']==slots[0]
        assert report['evicted_slot_bytes']==slots[0]+slots[-1]
        assert e.resident_device_bytes==sum(slots)<=e.cache.device_budget_bytes
        stats=arena.statistics()
        for size,capacity in Counter(slots).items():
            assert report['class_occupancy'][str(size)]['capacity']==capacity
            assert report['class_occupancy'][str(size)]['resident']==capacity
            assert stats['size_classes'][str(size)]==dict(capacity=capacity,free=0)
        e.cache.set_admission_budget(0)
        for i in targets:
            np.testing.assert_array_equal(asnumpy(e.forward(0,i,x)),expected[i])
            expert=e.cache.resident[key(i)][0]
            for device,host in zip((expert.gate,expert.up,expert.down),bank[key(i)]):
                np.testing.assert_array_equal(asnumpy(device.data),host.data)
        xp.cuda.get_current_stream().synchronize()
        assert sha256(mmap).hexdigest()==original
        for k,triple in bank.items():
            for a,b in zip(e._host_weights(k),triple):
                assert a is b and np.shares_memory(a.data,mmap) and not a.data.flags.writeable
        with pytest.raises(ValueError,match='before admission'): e.install_observed_profile()
    finally:
        e.close(); e.cache.clear()


@GPU
def test_gpu_sized_default_uses_warmup_usage(sources):
    bank,_,slots=sources
    e=executor(bank,slots)
    try:
        e.install_observed_profile(counts={key(i):1 for i in (0,2,1)})
        for i in (0,2,1,4,3):
            e.cache._usage.setdefault(key(i),residency._Usage()).hits=100 if i in (0,4,3) else 1
        report=e.install_observed_profile()
        assert report['count_source']=='warmup_usage'
        assert set(e.cache.resident)=={key(i) for i in (0,4,3)}
        assert report['selected_observed_requests']==300
    finally:
        e.close(); e.cache.clear()


@GPU
@pytest.mark.parametrize('bad',['geometry','format','oversize','dense','provider','byte_length'])
def test_gpu_bad_source_rejected_before_eviction(sources,monkeypatch,bad):
    from test_0228_iq_gpu_integration import packed
    bank,mmap,slots=sources
    e=executor(bank,slots)
    original=sha256(mmap).hexdigest()
    try:
        e.install_observed_profile(counts={key(i):1 for i in (0,2,1)})
        resident={k:id(v[0]) for k,v in e.cache.resident.items()}
        before=(e.cache.used_bytes,dict(e.cache.counters),e._slot_arena.statistics())
        replacement=dict(bank)
        if bad=='geometry':
            replacement[key(3)]=(packed('IQ4_XS',128,256),)+bank[key(3)][1:]
        elif bad=='format':
            supported=residency.supports_packed_gpu_type
            monkeypatch.setattr(residency,'supports_packed_gpu_type',lambda f:f!='IQ4_XS' and supported(f))
        elif bad=='oversize':
            replacement[key(3)]=tuple(packed('Q8_0',256,256,328+p) for p in range(3))
        elif bad=='dense':
            replacement[key(3)]=tuple(np.zeros((256,256),np.float32) for _ in range(3))
        elif bad=='byte_length':
            w=PackedGGUFWeight(bank[key(3)][0].data,(256,256),'IQ4_XS')
            object.__setattr__(w,'data',w.data[:-1]) # Simulate a damaged source descriptor.
            replacement[key(3)]=(w,)+bank[key(3)][1:]
        def provider(k):
            if bad=='provider' and k==key(3): raise ValueError('Bad source provider')
            return replacement[k]
        e.host_provider=provider
        with pytest.raises((ValueError,TypeError,NotImplementedError)):
            e.install_observed_profile(counts={key(i):100 for i in (0,4,3)})
        assert {k:id(v[0]) for k,v in e.cache.resident.items()}==resident
        assert (e.cache.used_bytes,dict(e.cache.counters),e._slot_arena.statistics())==before
        assert sha256(mmap).hexdigest()==original
    finally:
        e.close(); e.cache.clear()


@GPU
def test_gpu_busy_plans_and_host_substitution_reject_without_mutation(sources,monkeypatch):
    bank,mmap,slots=sources
    e=executor(bank,slots)
    original=sha256(mmap).hexdigest()
    try:
        e.install_observed_profile(counts={key(i):1 for i in (0,2,1)})
        resident={k:id(v[0]) for k,v in e.cache.resident.items()}
        before=(e.cache.used_bytes,dict(e.cache.counters),e._slot_arena.statistics())
        monkeypatch.setattr(e,'host_provider',lambda k:pytest.fail('Guard must precede source access'))
        def rejected(error):
            with pytest.raises(error):
                e.install_observed_profile(counts={key(i):100 for i in (0,4,3)})
            assert {k:id(v[0]) for k,v in e.cache.resident.items()}==resident
            assert (e.cache.used_bytes,dict(e.cache.counters),e._slot_arena.statistics())==before
        with e.dispatch_scope(0,[0]): rejected(ValueError)
        xp.cuda.get_current_stream().synchronize()
        e.cache._busy_keys()
        e.cache.protect([key(0)])
        try: rejected(RuntimeError)
        finally: e.cache.unprotect([key(0)])
        with monkeypatch.context() as m:
            m.setattr(e.cache,'_inflight_groups',[(SimpleNamespace(query=lambda:False),frozenset((key(0),)))])
            rejected(RuntimeError)
        for attr,value,error in (
            ('_device_expert_plans',{0:object()},ValueError),
            ('_resident_source',object(),ValueError),
            ('_ggml_row_pool',SimpleNamespace(_pending=object(),_native_service=None),RuntimeError),
            ('_ggml_row_pool',SimpleNamespace(_pending=None,_native_service=object()),ValueError),
        ):
            with monkeypatch.context() as m:
                m.setattr(e,attr,value)
                rejected(error)
        pending=Future()
        e._cpu_futures.add(pending)
        try: rejected(RuntimeError)
        finally: pending.set_result(None)
        for limit in (0,1):
            e.cache.set_admission_budget(limit)
            rejected(ValueError)
        assert sha256(mmap).hexdigest()==original
    finally:
        e.close(); e.cache.clear()


@GPU
def test_gpu_held_projection_lease_causes_explicit_incomplete_install(sources):
    import gc
    bank,_,slots=sources
    e=executor(bank,slots)
    try:
        e.install_observed_profile(counts={key(i):1 for i in (0,2,1)})
        held=e.cache.resident[key(2)][0].gate.data[::2]
        counts={key(i):100 for i in (0,4,3)}
        with pytest.raises(RuntimeError,match='installation incomplete') as error:
            e.install_observed_profile(counts=counts)
        assert isinstance(error.value.__cause__,MemoryError)
        assert set(e.cache.resident)=={key(0)} and e.cache.used_bytes==slots[0]
        assert e.cache.counters['profile_failures']==1
        np.testing.assert_array_equal(asnumpy(held),bank[key(2)][0].data[::2])
        del held; gc.collect()
        report=e.install_observed_profile(counts=counts)
        assert report['promotions']==2 and report['resident']==3
        assert e._slot_arena.statistics()['quarantined_slots']==0
    finally:
        e.close(); e.cache.clear()


@GPU
def test_gpu_late_copy_failure_is_not_silent_partial_success(sources,monkeypatch):
    bank,_,slots=sources
    e=executor(bank,slots)
    try:
        e.install_observed_profile(counts={key(i):1 for i in (0,2,1)})
        materialize=e._device_expert
        calls=[]
        def fail_second(weights):
            calls.append(weights)
            if len(calls)==2: raise MemoryError('Injected complete-bundle copy failure')
            return materialize(weights)
        monkeypatch.setattr(e,'_device_expert',fail_second)
        with pytest.raises(RuntimeError,match='incomplete \\(1/2 promotions\\)'):
            e.install_observed_profile(counts={key(i):100 for i in (0,4,3)})
        assert set(e.cache.resident)=={key(0),key(4)}
        assert e.cache.used_bytes==2*slots[0]
        assert e._slot_arena.statistics()['leased_slots']==2
        expert=e.cache.resident[key(4)][0]
        for device,host in zip((expert.gate,expert.up,expert.down),bank[key(4)]):
            np.testing.assert_array_equal(asnumpy(device.data),host.data)
    finally:
        e.close(); e.cache.clear()
