"""Complete paged QSA: monolithic equivalence and explicit decoded-INT8 oracle."""
from contextlib import ExitStack
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.native import Runtime, NativeError
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, QSAConfig
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_native_engine_qsa import inputs
from test_native_engine_qsa_layer import synthetic, seed_reference, hidden, read, state_read, NAMES
from test_native_engine_qsa_hidden_frontend import STEMS, NORMS, KINDS
from test_native_engine_kv_streamed import quant_reference

pytestmark = pytest.mark.gpu
FORMATS = ['reference_fp32', 'strata_int8_per64_fp16']


def paged(e, tensors, policy='strata', capacity=4096, page=64, debug=True, budget=2**26, fmt=FORMATS[0]):
    return e.qsa_layer(capacity, *tensors, policy=policy, kv_policy=fmt, page_size=page,
                       physical_pages=1, host_budget_bytes=budget, raw_diagnostics=debug)


def import_both(stack, path, r, e, layers, state):
    n = state.length
    arrays = dict(raw=asnumpy(state.raw_index[:n]).copy(),
                  k=asnumpy(state.keys[:n]).reshape(n, 512).copy(),
                  v=asnumpy(state.values[:n]).reshape(n, 512).copy(),
                  p=asnumpy(state.pooled[:n//4]).copy())
    uploaded = stack.enter_context(inputs(path, r, e, arrays))
    for layer in layers: layer.import_reference_state(n, *uploaded)
    return arrays, uploaded


def assert_raw_dependencies(layer, raw):
    n = len(raw)
    first = np.zeros(128, np.float32) if not n else raw[0]
    window = np.zeros((4, 128), np.float32)
    for t in range(max(0, n-4), n): window[t%4] = raw[t]
    np.testing.assert_array_equal(read(layer, 'raw_first', (128,)), first)
    np.testing.assert_array_equal(read(layer, 'raw_window', (4,128)), window)


def exact_step(mono, page, upload, t):
    assert mono.step(upload, t) == page.step(upload, t) == t+1
    for name, shape, dtype in [('output',(2560,),np.float32), ('scores',(t+1,),np.float32),
                               ('ids',(min(t+1,2051),),np.int64)]:
        np.testing.assert_array_equal(read(page,name,shape,dtype),read(mono,name,shape,dtype))
    actual, expected = state_read(page), state_read(mono)
    for got, want in zip(actual, expected): np.testing.assert_array_equal(got, want)
    assert_raw_dependencies(page, expected[0])


@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
@pytest.mark.parametrize('start,steps', [(0,9),(63,5),(2048,6),(2050,5),(2051,4),(4092,4)])
def test_fp32_import_boundaries_exact_state(tmp_path, policy, start, steps):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        tensors, ref = synthetic(stack,tmp_path,r,e,policy)
        mono = stack.enter_context(e.qsa_layer(4096,*tensors,policy=policy))
        page = stack.enter_context(paged(e,tensors,policy))
        if start:
            state = seed_reference(ref,4096,start)
            import_both(stack,tmp_path,r,e,[mono,page],state)
            for got,want in zip(state_read(page),state_read(mono)): np.testing.assert_array_equal(got,want)
            assert_raw_dependencies(page, state_read(mono)[0])
        uploads = stack.enter_context(inputs(tmp_path,r,e,{f'h{i}':hidden(start+i) for i in range(steps)}))
        stable = e.stats.device_bytes
        for i,u in enumerate(uploads):
            exact_step(mono,page,u,start+i)
            assert e.stats.device_bytes == stable
            assert page.paged_stats.resident_pages <= 1 and page.paged_stats.pending == 0
        if start>=2048:
            ids = read(page,'ids',(min(page.position,2051),),np.int64)
            assert len(np.unique(ids[ids<page.position-1]//64)) > 1
            assert page.paged_stats.misses > 1  # Legal selected working set exceeds physical slots.
        generation = page.paged_stats.generation
        mono.reset(); page.reset()
        assert page.position == 0 and page.paged_stats.generation == generation+1
        assert_raw_dependencies(page,np.empty((0,128),np.float32))
        for name in ('output','ids','scores'):
            dtype = np.int64 if name=='ids' else np.float32
            with pytest.raises(NativeError,match='unavailable'): read(page,name,(2560,),dtype)
        exact_step(mono,page,uploads[0],0)


def quantized_oracle(mono, page, ref):
    n = mono.position
    source = state_read(mono)
    _, decoded, _ = quant_reference(source[1],source[2])
    np.testing.assert_array_equal(read(page,'keys',(n,2,256)),decoded[:,0].reshape(n,2,256))
    np.testing.assert_array_equal(read(page,'values',(n,2,256)),decoded[:,1].reshape(n,2,256))
    np.testing.assert_array_equal(read(page,'raw',(n,128)),source[0])
    np.testing.assert_array_equal(read(page,'pooled',(n//4,128)),source[3])
    ids = read(page,'ids',(min(n,2051),),np.int64)
    np.testing.assert_array_equal(ids,read(mono,'ids',ids.shape,np.int64))
    np.testing.assert_array_equal(read(page,'scores',(n,)),read(mono,'scores',(n,)))
    q, gate = read(page,'queries',(24,256)), read(page,'gates',(24,256))
    k = decoded[ids,0].reshape(-1,2,256); v = decoded[ids,1].reshape(-1,2,256)
    context = np.empty_like(q)
    for h in range(24):
        logits = (k[:,h//12] @ q[h])*np.float32(1/16)
        p = np.exp(logits-logits.max()); p /= p.sum(dtype=np.float32)
        context[h] = p @ v[:,h//12]
    z = np.exp(-np.abs(gate)); sigmoid = np.where(gate>=0,1/(1+z),z/(1+z))
    want = asnumpy(ref._project('output',xp.asarray((context*sigmoid).reshape(-1))))
    got = read(page,'output',(2560,))
    np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    assert_raw_dependencies(page,source[0])
    return float(np.max(np.abs(got-want))),float(np.max(np.abs(got-read(mono,'output',(2560,)))))


@pytest.mark.parametrize('policy',['ggml_cpu','strata'])
@pytest.mark.parametrize('start,steps',[(0,5),(2050,4)])
def test_int8_defined_decoded_oracle(tmp_path,policy,start,steps):
    r = Runtime(); errors=[]
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        tensors,ref = synthetic(stack,tmp_path,r,e,policy)
        mono = stack.enter_context(e.qsa_layer(4096,*tensors,policy=policy))
        page = stack.enter_context(paged(e,tensors,policy,fmt=FORMATS[1]))
        if start: import_both(stack,tmp_path,r,e,[mono,page],seed_reference(ref,4096,start))
        uploads = stack.enter_context(inputs(tmp_path,r,e,{f'h{i}':hidden(start+i) for i in range(steps)}))
        for i,u in enumerate(uploads):
            mono.step(u,start+i); page.step(u,start+i)
            errors.append(quantized_oracle(mono,page,ref))
        print('PAGED_INT8_ORACLE='+json.dumps(dict(start=start,policy=policy,
              decoded_maxabs=max(x[0] for x in errors),fp32_difference_maxabs=max(x[1] for x in errors))))


@pytest.mark.parametrize('fmt',FORMATS)
def test_configured_262k_short_active_bounded_and_raw_debug_off(tmp_path,fmt):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        tensors,ref = synthetic(stack,tmp_path,r,e,'strata')
        mono = stack.enter_context(e.qsa_layer(4096,*tensors,policy='strata'))
        small = stack.enter_context(paged(e,tensors,fmt=fmt))
        large = stack.enter_context(paged(e,tensors,capacity=262144,debug=False,budget=2**20,fmt=fmt))
        initial = large.paged_stats
        assert initial.host_bytes < 100_000
        assert initial.device_bytes < 80*2**20
        assert initial.device_bytes-small.paged_stats.device_bytes == (262144-4096)*260 + ((262144-4096)//64)*8
        state = seed_reference(ref,4096,31)
        import_both(stack,tmp_path,r,e,[mono,small,large],state)
        with pytest.raises(NativeError,match='raw diagnostics disabled'): read(large,'raw',(31,128))
        uploads = stack.enter_context(inputs(tmp_path,r,e,{f'h{i}':hidden(31+i) for i in range(4)}))
        stable = e.stats.device_bytes
        for i,u in enumerate(uploads):
            t=31+i; mono.step(u,t); small.step(u,t); large.step(u,t)
            for name,shape,dtype in [('output',(2560,),np.float32),('ids',(t+1,),np.int64),('scores',(t+1,),np.float32)]:
                np.testing.assert_array_equal(read(large,name,shape,dtype),read(small,name,shape,dtype))
            if fmt==FORMATS[0]: np.testing.assert_array_equal(read(large,'output',(2560,)),read(mono,'output',(2560,)))
            assert_raw_dependencies(large,state_read(mono)[0])
            assert e.stats.device_bytes == stable
            assert large.paged_stats.host_bytes < 2**20
        print('PAGED_QSA_262K_SHORT='+json.dumps(dict(format=fmt,**{k:int(getattr(large.paged_stats,k))
              for k in ('length','device_bytes','host_bytes','host_budget','resident_pages','hits','misses')})))


def test_error_budget_import_poison_reset_lifetime(tmp_path):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        tensors,ref = synthetic(stack,tmp_path,r,e,'strata')
        # Exactly metadata and one debug page, not an implicit full-context allocation.
        budget = 32840+8*(1024+2)+4*(4096+512)
        page = stack.enter_context(paged(e,tensors,page=4,budget=budget))
        mono = stack.enter_context(e.qsa_layer(4096,*tensors,policy='strata'))
        u = stack.enter_context(inputs(tmp_path,r,e,dict(h=hidden(0),bad=np.full(2560,np.nan,np.float32))))
        for t in range(4): exact_step(mono,page,u[0],t)
        saved = state_read(page)
        for source,position,match in [(u[0],4,'host page budget'),(u[1],3,'position'),(u[0],0,'position')]:
            with pytest.raises(NativeError,match=match): page.step(source,position)
            assert page.position==4
            for got,want in zip(state_read(page),saved): np.testing.assert_array_equal(got,want)
        page.reset(); mono.reset()
        with pytest.raises(NativeError,match='nonfinite'): page.step(u[1],0)
        assert page.position==0
        exact_step(mono,page,u[0],0)
        page.reset()
        reference = seed_reference(ref,4096,8)
        arrays,imports = import_both(stack,tmp_path,r,e,[],reference)
        with pytest.raises(NativeError,match='host page budget'): page.import_reference_state(8,*imports)
        with pytest.raises(NativeError,match='reset required'): _=page.position
        page.reset()
        short = seed_reference(ref,4096,4)
        import_both(stack,tmp_path,r,e,[page],short)
        assert page.position==4
        page.reset(); mono.reset()
        # Public source and engine close do not invalidate retained layer owners.
        for tensor in tensors: tensor.close()
        e.close()
        exact_step(mono,page,u[0],0)
        stale=page._handle; page.close()
        assert r._lib.ml_qsa_layer_reset(stale)==1


def checkpoint_tensors(stack,r,e,store):
    tensors=[]; callbacks={}
    for name,stem in zip(NAMES,STEMS+['attn_output.weight']):
        lease=stack.enter_context(store.lease('blk.3.'+stem)); d=lease.descriptor
        raw=lease.read(0,d.nbytes)
        tensors.append(stack.enter_context(e.upload(lease)))
        cache={}
        def project(x,raw=raw,d=d,cache=cache):
            x=asnumpy(x); key=x.tobytes()
            if key not in cache:
                rows,width=d.ggml_shape[1],d.ggml_shape[0]; y=np.empty(rows,np.float32)
                for start in range(0,rows,32):
                    count=min(32,rows-start); chunk=raw[start*d.row_bytes:(start+count)*d.row_bytes]
                    if d.type_id==0: w=np.frombuffer(chunk,'<f4').reshape(count,width)
                    elif d.type_id==30: w=(np.frombuffer(chunk,'<u2').astype(np.uint32)<<16).view(np.float32).reshape(count,width)
                    else: w=PackedGGUFWeight(np.frombuffer(chunk,np.uint8),(count,width),KINDS[d.type_id]).dequantize_reference()
                    y[start:start+count]=w@x
                cache[key]=y
            return xp.asarray(cache[key])
        callbacks[name]=project
    gammas=[]
    for stem in NORMS:
        lease=stack.enter_context(store.lease('blk.3.'+stem))
        gammas.append(np.frombuffer(lease.read(0,lease.descriptor.nbytes),'<f4').copy())
        tensors.append(stack.enter_context(e.upload(lease)))
    ref=Qwen4ExpAttention(QSAConfig(2560,tie_policy='strata'),**callbacks,
        gamma_q=xp.asarray(gammas[0]),gamma_k=xp.asarray(gammas[1]),
        gamma_index_q=xp.asarray(gammas[2]),gamma_index_k=xp.asarray(gammas[3]))
    return tensors,ref


@pytest.mark.skipif(os.environ.get('MINI_LLM_QSA_LAYER_CHECKPOINT')!='1',reason='bounded actual checkpoint paged composition')
@pytest.mark.parametrize('fmt',FORMATS)
def test_checkpoint_layer3_short_sparse_262k(tmp_path,fmt):
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r=Runtime(); errors=[]
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=384*2**20))
        store=stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        tensors,ref=checkpoint_tensors(stack,r,e,store)
        mono=stack.enter_context(e.qsa_layer(4096,*tensors,policy='strata'))
        page=stack.enter_context(paged(e,tensors,capacity=262144,fmt=fmt,budget=16*2**20))
        for start,steps in [(0,5),(2050,3)]:
            mono.reset(); page.reset()
            if start: import_both(stack,tmp_path,r,e,[mono,page],seed_reference(ref,4096,start))
            uploads=stack.enter_context(inputs(tmp_path,r,e,{f'h{start}_{i}':hidden(start+i) for i in range(steps)}))
            for i,u in enumerate(uploads):
                if fmt==FORMATS[0]: exact_step(mono,page,u,start+i); errors.append((0.,0.))
                else:
                    mono.step(u,start+i); page.step(u,start+i); errors.append(quantized_oracle(mono,page,ref))
            if start:
                ids=read(page,'ids',(2051,),np.int64)
                assert len(np.unique(ids[ids<page.position-1]//64))>1
                assert page.paged_stats.misses>1
        stats=page.paged_stats
        assert stats.device_bytes<80*2**20 and stats.host_bytes<16*2**20
        print('PAGED_QSA_CHECKPOINT='+json.dumps(dict(layer=3,format=fmt,capacity=262144,steps=8,
              decoded_maxabs=max(x[0] for x in errors),fp32_difference_maxabs=max(x[1] for x in errors),
              device_bytes=stats.device_bytes,host_bytes=stats.host_bytes,misses=stats.misses,
              engine_peak_bytes=e.stats.peak_bytes)))
