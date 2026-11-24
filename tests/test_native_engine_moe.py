"""WP5 complete native owner, bounded synthetic and opt-in checkpoint oracles."""
from contextlib import ExitStack
import ctypes as C
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.native import Runtime, NativeError
from mini_llm.ops.ggml_packed_cpu import _load_bridge, GGMLPackedCPUExecutor
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0256_ggml_packed_cpu import selected_library
from test_0228_iq_gpu_integration import packed
from test_0201_gguf_store import write_gguf
from test_native_engine_cpu_experts import ACTUAL_TRIPLES

POLICY = 'cpu_ggml_q8_gpu_reference_fp32'
TYPES = dict(Q8_0=8,Q4_K=12,Q5_K=13,Q6_K=14,IQ3_XXS=18,IQ4_NL=20,IQ3_S=21,IQ2_S=22,IQ4_XS=23,Q2_0=42)
IDS = (0,1,7,33,64,128,255,300,500,511)


@pytest.fixture
def libraries(selected_library):
    return dict(bridge_library=_load_bridge()._name, ggml_library=selected_library)


@pytest.fixture
def cp(monkeypatch):
    if BACKEND_NAME != 'cupy':
        pytest.skip('MoE arithmetic oracle requires MINI_LLM_BACKEND=cupy')
    import cupy
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL','warp')
    monkeypatch.setenv('MINI_LLM_ROUTER_DECODE','0')
    return cupy


def fixture(path, formats=ACTUAL_TRIPLES[0], shared=('Q4_K','IQ4_NL'), ties=False, bad_scalar=False):
    d,f,sf=256,320,256
    experts=[tuple(packed(fmt,n,k,6100+e*3+p) for p,(fmt,n,k) in
        enumerate(zip(formats,(f,f,d),(d,d,f)))) for e in range(10)]
    sw=[packed(fmt,n,k,6200+p) for p,(fmt,n,k) in
        enumerate(zip((shared[0],shared[0],shared[1]),(sf,sf,d),(d,d,sf)))]
    router=np.zeros((512,d),np.float32)
    if not ties:
        router[:,0]=-100
        router[list(IDS),0]=np.arange(10,dtype=np.float32)*.1+1
    scalar=np.random.default_rng(63).normal(0,.02,d).astype(np.float32)
    if bad_scalar: scalar[0]=np.nan
    tensors=[]; offset=0
    def add(name,shape,kind,payload):
        nonlocal offset
        offset=(offset+31)//32*32
        tensors.append((name,shape,kind,offset,payload)); offset+=len(payload)
    add('router',(d,512),0,router.tobytes())
    for name,w in zip(('sg','su','sd'),sw): add(name,tuple(reversed(w.shape)),TYPES[w.quant_type],w.data.tobytes())
    add('scalar',(d,),0,scalar.tobytes())
    for p,(name,fmt,n,k) in enumerate(zip(('bg','bu','bd'),formats,(f,f,d),(d,d,f))):
        add(name,(k,n,512),TYPES[fmt],b''.join(experts[e%10][p].data.tobytes() for e in range(512)))
    write_gguf(path,tensors)
    x=np.random.default_rng(64).normal(0,.02,d).astype(np.float32); x[0]=1
    return dict(d=d,f=f,router=router,shared=sw,scalar=scalar,x=x,
                expert=lambda e:experts[e%10])


def native_setup(stack,path,data,libraries,placement,budget=511*2**20):
    r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=budget))
    store=stack.enter_context(r.gguf(path))
    leases=[stack.enter_context(store.lease(n)) for n in ('router','sg','su','sd','scalar','bg','bu','bd')]
    weights=[stack.enter_context(e.upload(l)) for l in leases[:5]]
    baseline=e.stats.device_bytes
    owner=stack.enter_context(e.moe(*weights,*leases[5:],d_model=data['d'],d_ff=data['f'],
        gpu_experts=placement,**libraries))
    return r,e,store,leases,weights,owner,baseline


def reference(cp,data,placement,libraries,monkeypatch,scale=.37):
    """Unchanged Qwen4ExpMoE + actual heterogeneous executor, frozen preparation."""
    from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
    from mini_llm.ops.router_inference import RouterInference
    from mini_llm.ops.experts_inference import ExpertsInference
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor, ExpertKey
    from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear, PackedExpertFFNInference
    router=RouterInference(data['d'],512,10,renormalize_selected=True,normalization_floor=2**-14)
    router._decode_fastpath_enabled=False
    router.W_router=cp.asarray(data['router'].T.copy()); router.b_router=cp.zeros(512,cp.float32)
    x=cp.asarray(data['x']); rw,ri=router.forward(x.reshape(1,1,-1))
    ids=tuple(sorted(cp.asnumpy(ri).ravel().tolist()))
    devices={i:PackedExpertFFNInference(*(packed_to_device(w) for w in data['expert'](i))) for i in ids if i in placement}
    executor=HeterogeneousExpertExecutor(lambda key:data['expert'](key.expert_id),model_id='wp5',
        d_model=data['d'],d_ff=data['f'],compute_dtype='float32',cpu_workers=4,
        cpu_packed_backend='ggml_pool',group_after_admission=True,group_mixed_decode=True)
    monkeypatch.setattr(executor,'_prepare',lambda layer,i,x:
        (ExpertKey('wp5',layer,i),devices.get(i),None if i in devices else data['expert'](i),None))
    experts=ExpertsInference(data['d'],data['f'],512,10,expert_executor=executor,layer_id=0)
    callbacks=[]
    for w in data['shared']:
        dev=packed_to_device(w)
        callbacks.append(lambda x,w=dev:quantized_linear(x.reshape(1,-1),w).reshape(-1))
    model=Qwen4ExpMoE(router,experts,*callbacks,cp.asarray(data['scalar']),scale=scale)
    try:
        result=cp.asnumpy(model(x)).ravel()
        order=np.argsort(cp.asnumpy(ri).ravel())
        return result,cp.asnumpy(ri).ravel()[order],cp.asnumpy(rw).ravel()[order]*np.float32(scale)
    finally:
        executor.close()


def compare(got,expected):
    diff=got.astype(np.float64)-expected.astype(np.float64)
    absolute=float(np.max(np.abs(diff)))
    relative=float(np.linalg.norm(diff)/max(np.linalg.norm(expected),1e-30))
    assert relative<1e-5 or absolute<2e-6, (absolute,relative)
    assert absolute < 2e-6+2e-5*np.max(np.abs(expected)), (absolute,relative)
    return absolute,relative


@pytest.mark.parametrize('formats',ACTUAL_TRIPLES)
@pytest.mark.parametrize('mode',('cpu','gpu','mixed'))
def test_full_owner_formats(cp,libraries,tmp_path,monkeypatch,formats,mode):
    data=fixture(tmp_path/'moe.gguf',formats)
    placement=() if mode=='cpu' else IDS if mode=='gpu' else IDS[::2]
    expected,ids,weights=reference(cp,data,placement,libraries,monkeypatch)
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,tmp_path/'moe.gguf',data,libraries,placement)
        assert e.stats.peak_bytes<512*2**20
        # Slot memory grows only with explicit placement, not all 512 banks.
        expert_bytes=sum(w.nbytes for w in data['expert'](0))
        assert len(placement)*expert_bytes < e.stats.device_bytes-base < len(placement)*expert_bytes+200000
        monkeypatch.setattr(GGMLPackedCPUExecutor,'expert',lambda *a:pytest.fail('Python expert dispatch'))
        with owner.submit(data['x'],scale=.37) as job:
            got,gi,gw=job.readback_for_test()
            np.testing.assert_array_equal(gi,ids)
            np.testing.assert_allclose(gw,weights,rtol=8e-6,atol=2e-7)
            compare(got,expected)
            np.testing.assert_array_equal(job.readback_for_test()[0],got)


@pytest.mark.parametrize('shared',list(itertools.product(('Q4_K','Q5_K','Q6_K','IQ4_XS'),('IQ4_NL','Q8_0'))))
def test_shared_combinations_and_ties(cp,libraries,tmp_path,monkeypatch,shared):
    data=fixture(tmp_path/'moe.gguf',shared=shared,ties=True)
    placement=(0,2,4,6,8)
    expected,ids,weights=reference(cp,data,placement,libraries,monkeypatch)
    np.testing.assert_array_equal(ids,np.arange(10))
    with ExitStack() as stack:
        *_,owner,base=native_setup(stack,tmp_path/'moe.gguf',data,libraries,placement)
        with owner.submit(data['x'],scale=.37) as job:
            got,gi,gw=job.readback_for_test()
            np.testing.assert_array_equal(gi,ids)
            np.testing.assert_allclose(gw,weights,rtol=8e-6,atol=2e-7)
            compare(got,expected)


def test_busy_lifetime_invalid_and_budget(cp,libraries,tmp_path):
    path=tmp_path/'moe.gguf'; data=fixture(path)
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,IDS[::2])
        x=data['x'].copy(); job=owner.submit(x); x.fill(999)
        with pytest.raises(NativeError,match='busy'): owner.submit(x)
        with pytest.raises(NativeError,match='busy'): owner.reset()
        for value in (*leases,*uploads): value.close()
        store.close(); e.close()
        assert str(path) in Path('/proc/self/maps').read_text()
        owner.close()
        got=job.readback_for_test()[0]
        stale=job._handle; job.close()
        assert r._lib.ml_moe_job_wait(stale)==1
        assert str(path) not in Path('/proc/self/maps').read_text()
        assert np.isfinite(got).all()
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,(),budget=8*2**20)
        before=e.stats.device_bytes
        for placement,policy in (((1,1),POLICY),((512,),POLICY),((), 'reference_fp32')):
            with pytest.raises((NativeError,ValueError)):
                e.moe(*uploads,*leases[5:],d_model=data['d'],d_ff=data['f'],gpu_experts=placement,
                    arithmetic=policy,**libraries)
            assert e.stats.device_bytes==before
        for x in (np.full(data['d'],np.nan,np.float32),):
            with pytest.raises(NativeError,match='nonfinite'): owner.submit(x)
        for scale in (-1,float('inf')):
            with pytest.raises(NativeError): owner.submit(data['x'],scale=scale)
        out=C.c_uint64(123)
        assert r._lib.ml_moe_submit(owner._handle,None,data['d'],1,C.byref(out))==1
        assert out.value==0
        owner.reset()
        # Job outputs remain snapshots across future owner epochs.
        with owner.submit(data['x']) as first:
            saved=first.readback_for_test()[0]
            owner.reset()
            with owner.submit(np.zeros(data['d'],np.float32)) as second:
                np.testing.assert_array_equal(second.readback_for_test()[0],np.zeros(data['d'],np.float32))
            np.testing.assert_array_equal(first.readback_for_test()[0],saved)
        # All-resident allocation must hit the explicit engine budget, rollback fully.
        with pytest.raises(NativeError,match='budget'):
            e.moe(*uploads,*leases[5:],d_model=data['d'],d_ff=data['f'],gpu_experts=range(512),**libraries)
        assert e.stats.device_bytes==before


def test_placement_is_static_across_route_changes(cp,libraries,tmp_path,monkeypatch):
    path=tmp_path/'changing.gguf'; data=fixture(path)
    first=data['x'].copy()
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,IDS)
        allocated=e.stats.device_bytes
        for x in (first, -first, first):
            data['x']=x
            expected,ids,weights=reference(cp,data,IDS,libraries,monkeypatch)
            with owner.submit(x,scale=.37) as job:
                got,gi,gw=job.readback_for_test()
                np.testing.assert_array_equal(gi,ids)
                compare(got,expected)
            assert e.stats.device_bytes==allocated
        # ABI failures zero handles and do not expose partial diagnostic results.
        out=C.c_uint64(123)
        assert r._lib.ml_moe_create(owner._handle,data['d'],data['f'],None,None,None,
            0,4,None,None,None,C.byref(out))==1
        assert out.value==0
        with owner.submit(first) as job:
            result=np.full(data['d'],123,np.float32)
            ids=np.full(10,-1,np.int32); weights=np.full(10,123,np.float32)
            assert r._lib.ml_moe_job_readback_for_test(job._handle,
                result.ctypes.data_as(C.POINTER(C.c_float)),1,
                ids.ctypes.data_as(C.POINTER(C.c_int32)),weights.ctypes.data_as(C.POINTER(C.c_float)))==1
            assert np.all(result==123) and np.all(ids==-1) and np.all(weights==123)
            job.wait()


def test_failed_device_job_is_not_consumable_and_can_close(cp,libraries,tmp_path):
    path=tmp_path/'bad.gguf'; data=fixture(path,bad_scalar=True)
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,IDS)
        job=owner.submit(data['x'])
        with pytest.raises(NativeError,match='MoE device result'): job.wait()
        with pytest.raises(NativeError,match='failed MoE job'): job.readback_for_test()
        owner.reset()
        stale=job._handle
        job.close()
        assert r._lib.ml_moe_job_wait(stale)==1
        with owner.submit(data['x']) as again:
            with pytest.raises(NativeError,match='MoE device result'): again.wait()


@pytest.mark.skipif(os.environ.get('MINI_LLM_MOE_CHECKPOINT')!='1',reason='opt-in bounded actual checkpoint')
@pytest.mark.parametrize('layer',(0,1,47))
@pytest.mark.parametrize('mode',('cpu','gpu','mixed'))
def test_actual_checkpoint(cp,libraries,tmp_path,monkeypatch,layer,mode):
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r=Runtime(); prefix=f'blk.{layer}.'
    names=('ffn_gate_inp.weight','ffn_gate_shexp.weight','ffn_up_shexp.weight',
           'ffn_down_shexp.weight','ffn_gate_inp_shexp.weight','ffn_gate_exps.weight',
           'ffn_up_exps.weight','ffn_down_exps.weight')
    kinds={v:k for k,v in TYPES.items()}
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=511*2**20))
        store=stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        leases=[stack.enter_context(store.lease(prefix+n)) for n in names]
        def packed_lease(l,offset=0,shape=None):
            t=l.descriptor
            shape=shape or tuple(reversed(t.ggml_shape))
            n=shape[0]*t.row_bytes
            return PackedGGUFWeight(np.frombuffer(l.read(offset,n),np.uint8).copy(),shape,kinds[t.type_id])
        d,f=2560,640
        def control(l):
            t=l.descriptor; raw=l.read(0,t.nbytes)
            if t.type_id==30:
                return (np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
            return np.frombuffer(raw,'<f4' if t.type_id==0 else '<f2').astype(np.float32)
        router=control(leases[0]).reshape(512,d)
        scalar=control(leases[4])
        shared=[packed_lease(l) for l in leases[1:4]]
        x=np.random.default_rng(6500+layer).normal(0,.1,d).astype(np.float32)
        # Select placement from the unchanged router with robust, non-tied logits.
        selected=np.argsort(-(router@x),kind='stable')[:10]
        placement=() if mode=='cpu' else tuple(sorted(int(i) for i in (selected if mode=='gpu' else selected[::2])))
        cache={}
        def expert(i):
            if i not in cache:
                cache[i]=tuple(packed_lease(l,i*(d if p==2 else f)*l.descriptor.row_bytes,
                    (d,f) if p==2 else (f,d)) for p,l in enumerate(leases[5:]))
            return cache[i]
        data=dict(d=d,f=f,router=router,scalar=scalar,shared=shared,x=x,expert=expert)
        expected,ids,weights=reference(cp,data,placement,libraries,monkeypatch)
        uploads=[stack.enter_context(e.upload(l)) for l in leases[:5]]
        owner=stack.enter_context(e.moe(*uploads,*leases[5:],d_model=d,d_ff=f,gpu_experts=placement,**libraries))
        with owner.submit(x,scale=.37) as job:
            got,gi,gw=job.readback_for_test()
            np.testing.assert_array_equal(gi,ids)
            np.testing.assert_allclose(gw,weights,rtol=1e-5,atol=3e-7)
            absolute,relative=compare(got,expected)
        assert e.stats.peak_bytes<512*2**20
        oracle_pool_bytes=cp.get_default_memory_pool().total_bytes()
        assert e.stats.device_bytes+oracle_pool_bytes<512*2**20
        print('\nWP5_CHECKPOINT='+json.dumps(dict(layer=layer,mode=mode,placement=placement,ids=gi.tolist(),
            max_absolute=absolute,relative_l2=relative,device_bytes=e.stats.device_bytes,
            peak_bytes=e.stats.peak_bytes,oracle_pool_bytes=oracle_pool_bytes,policy=POLICY)))
