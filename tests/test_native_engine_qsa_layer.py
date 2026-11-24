"""Complete native attention trajectories against unchanged Python QSA.step.

Native steps never consume any of the diagnostic arrays used by these tests.
"""
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.native import Runtime, NativeError
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_native_engine_qsa import inputs
from test_native_engine_qsa_projection import weight_upload
from test_native_engine_qsa_hidden_frontend import STEMS, NORMS, KINDS

pytestmark = pytest.mark.gpu
NAMES = ['query_gate', 'key', 'value', 'index_query', 'index_key', 'output']
ROWS = [12288, 512, 512, 512, 128, 2560]


def hidden(t):
    return (np.sin(np.arange(2560, dtype=np.float32)*np.float32(.03125) +
                   np.float32(t % 11)*np.float32(.125))*.2).astype(np.float32)


def synthetic(stack, path, r, e, policy, flat=False):
    weights, callbacks = [], {}
    for j, (name, rows) in enumerate(zip(NAMES, ROWS)):
        width = 6144 if j == 5 else 2560
        cols = (np.arange(rows)*17+j*31) % width
        scale = np.resize(np.array([.25, -.125, .5, .0625], np.float32), rows)
        w = np.zeros((rows, width), np.float32)
        w[np.arange(rows), cols] = scale
        raw = (w.view(np.uint32)>>16).astype('<u2').tobytes()
        weights.append(weight_upload(stack, path, r, e, raw, 30, rows, width))
        callbacks[name] = lambda x, cols=cols, scale=scale: xp.asarray(asnumpy(x)[cols]*scale)
    gammas = [np.linspace(.5, 1.5, n, dtype=np.float32) for n in [256,256,128,128]]
    # Nonrotary suffix-only scoring yields stable controlled cutoff fixtures.
    gammas[2][:64] = 0
    if flat:
        gammas[2].fill(0)
    u = stack.enter_context(inputs(path, r, e, dict(zip(['gq','gk','giq','gik'],gammas))))
    ref = Qwen4ExpAttention(QSAConfig(2560, tie_policy=policy), **callbacks,
        gamma_q=xp.asarray(gammas[0]), gamma_k=xp.asarray(gammas[1]),
                gamma_index_q=xp.asarray(gammas[2]), gamma_index_k=xp.asarray(gammas[3]))
    return weights+u, ref


def seed_reference(ref, capacity, length):
    """Build real reference caches without performing thousands of attention steps.

    Every raw/K/V row is projected from hidden(t), normalized/rotated with the
    reference operators; pools use explicit chronological FP32 additions.
    """
    state = ref.create_state(capacity)
    for t in range(length):
        x = xp.asarray(hidden(t))
        state.raw_index[t] = ref._project('index_key', x)
        k = ref._project('key', x).reshape(2,256)
        state.keys[t] = ref._rotate(ref.norms[1].forward(k), t)
        state.values[t] = ref._project('value', x).reshape(2,256)
        if (t+1) % 4 == 0:
            p = state.raw_index[t-3].copy()
            p += state.raw_index[t-2]
            p += state.raw_index[t-1]
            p += state.raw_index[t]
            p *= np.float32(.25)
            state.pooled[t//4] = ref._rotate(ref.norms[3].forward(p[None]), t-3)[0]
    state.length = length
    ref._state(state)
    return state


def import_state(stack, path, r, e, layer, state):
    n = state.length
    arrays = dict(raw=state.raw_index[:n].copy(), keys=state.keys[:n].reshape(n,512).copy(),
                  values=state.values[:n].reshape(n,512).copy(), pooled=state.pooled[:n//4].copy())
    u = stack.enter_context(inputs(path, r, e, {k:asnumpy(v) for k,v in arrays.items()}))
    layer.import_reference_state(n, *u)
    return u


def read(layer, name, shape, dtype=np.float32):
    a = np.empty(shape, dtype)
    layer.readback_for_test(name,a)
    return a


def state_read(layer):
    n = layer.position
    return [read(layer,name,shape) for name,shape in
            [('raw',(n,128)),('keys',(n,2,256)),('values',(n,2,256)),('pooled',(n//4,128))]]


def compare_state(layer, state):
    n = state.length
    wants = [state.raw_index[:n],state.keys[:n],state.values[:n],state.pooled[:n//4]]
    for j,(got,want) in enumerate(zip(state_read(layer),wants)):
        want=asnumpy(want)
        # Frontend RMS/reduction + long-position powf angle rounding envelope.
        bound = 5e-5*(1+np.abs(want))
        if j in (1,3) and n:
            pos = np.arange(n, dtype=np.float32) if j == 1 else np.arange(n//4,dtype=np.float32)*4
            pos = pos.reshape((-1,)+(1,)*(want.ndim-1))
            f = np.power(np.float32(1e7), -np.arange(32,dtype=np.float32)/32)
            angle = 8*np.finfo(np.float32).eps*pos*f*(np.abs(want[...,:32])+np.abs(want[...,32:64]))
            bound[...,:32] += angle
            bound[...,32:64] += angle
        assert np.all(np.abs(got-want)<=bound), (j,float(np.max(np.abs(got-want))))


def compare_step(layer, ref, state, x, upload):
    t = state.length
    want, diag = ref.step(xp.asarray(x),state,position=t,diagnostics=True)
    want=asnumpy(want)
    assert layer.step(upload,t) == t+1 == layer.position
    got = read(layer,'output',(2560,))
    # Includes matvec, norm/RoPE, softmax, gate and projection reduction errors.
    np.testing.assert_allclose(got,want,rtol=2e-4,atol=2e-4)
    np.testing.assert_array_equal(read(layer,'ids',(min(t+1,2051),),np.int64),asnumpy(diag['selected_ids']))
    np.testing.assert_allclose(read(layer,'scores',(t+1,)),asnumpy(diag['cell_scores']),rtol=5e-5,atol=2e-4)
    compare_state(layer,state)
    return float(np.max(np.abs(got-want)))


@pytest.mark.parametrize('policy',['ggml_cpu','strata'])
@pytest.mark.parametrize('flat',[False,True])
@pytest.mark.parametrize('capacity,start,steps',[(4096,0,9),(4096,2050,5),(4096,4092,4),(16384,16380,4)])
def test_trajectories(tmp_path,policy,flat,capacity,start,steps):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        tensors,ref = synthetic(stack,tmp_path,r,e,policy,flat)
        baseline = e.stats.device_bytes
        layer = stack.enter_context(e.qsa_layer(capacity,*tensors,policy=policy))
        state = seed_reference(ref,capacity,start)
        if start:
            import_state(stack,tmp_path,r,e,layer,state)
            compare_state(layer,state)
        before = bytes(e.stats)
        # Pre-upload all hidden inputs; neither step nor reference uses readback.
        xs = [hidden(start+i) for i in range(steps)]
        uploads = stack.enter_context(inputs(tmp_path,r,e,{f'h{i}':x for i,x in enumerate(xs)}))
        stable = bytes(e.stats)
        for x,u in zip(xs,uploads):
            compare_step(layer,ref,state,x,u)
            assert bytes(e.stats) == stable
        assert e.stats.peak_bytes < 512*2**20
        saved = state_read(layer)
        for bad in [layer.position-1,capacity]:
            with pytest.raises(NativeError,match='position'):
                layer.step(uploads[0],bad)
            for a,b in zip(saved,state_read(layer)): np.testing.assert_array_equal(a,b)
        layer.reset()
        assert layer.position == 0
        with pytest.raises(NativeError,match='unavailable'):
            read(layer,'output',(2560,))
        ref.reset(state)
        compare_step(layer,ref,state,xs[0],uploads[0])
        # Native weight ownership survives all public weight/upload/engine closes.
        for tensor in tensors: tensor.close()
        e.close()
        compare_step(layer,ref,state,xs[1],uploads[1])
        assert baseline > 0 and before


def test_invalid_inputs_import_and_budget(tmp_path):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=384*2**20))
        other = stack.enter_context(r.engine(budget_bytes=2**20))
        tensors,ref = synthetic(stack,tmp_path,r,e,'strata')
        layer = stack.enter_context(e.qsa_layer(4096,*tensors,policy='strata'))
        state = ref.create_state(4096)
        x = hidden(0)
        u = stack.enter_context(inputs(tmp_path,r,e,dict(h=x,nan=np.full(2560,np.nan,np.float32),wrong=np.zeros(3,np.float32))))
        cross = stack.enter_context(inputs(tmp_path,r,other,dict(h=x)))[0]
        compare_step(layer,ref,state,x,u[0])
        saved = state_read(layer)
        stale = u[2]._handle
        u[2].close()
        for source,match in [(u[1],'nonfinite'),(cross,'cross-engine')]:
            with pytest.raises(NativeError,match=match): layer.step(source,1)
            assert layer.position == 1
            for a,b in zip(saved,state_read(layer)): np.testing.assert_array_equal(a,b)
        done = C.c_uint64()
        assert r._lib.ml_qsa_layer_step(layer._handle,stale,1,C.byref(done)) == 1
        assert layer.position == 1
        with pytest.raises(ValueError): layer.readback_for_test('output',np.empty(2560,np.float64))
        compare_step(layer,ref,state,x,u[0])
        with pytest.raises(NativeError,match='capacity'): read(layer,'output',(1,))
        layer.reset()
        reference = seed_reference(ref,4096,8)
        imports = import_state(stack,tmp_path,r,e,layer,reference)
        with pytest.raises(NativeError,match='empty'): layer.import_reference_state(8,*imports)
        layer.reset()
        bad = stack.enter_context(inputs(tmp_path,r,e,dict(raw=np.full((8,128),np.nan,np.float32))))[0]
        with pytest.raises(NativeError,match='nonfinite'):
            layer.import_reference_state(8,bad,*imports[1:])
        assert layer.position == 0
        import_state(stack,tmp_path,r,e,layer,reference)
        compare_state(layer,reference)
        # Admission failure doesn't alter accounting; no sessions are allocated.
        before = bytes(e.stats)
        small = stack.enter_context(r.engine(budget_bytes=12))
        # Native cross-engine checks precede allocation.
        with pytest.raises(NativeError,match='cross-engine'):
            small.qsa_layer(4096,*tensors,policy='strata')
        assert bytes(e.stats)==before
        for cap,policy in [(8,'strata'),(4096,'unknown')]:
            with pytest.raises((NativeError,ValueError)):
                e.qsa_layer(cap,*tensors,policy=policy)
        stale_layer = layer._handle
        layer.close()
        assert r._lib.ml_qsa_layer_reset(stale_layer)==1


@pytest.mark.parametrize('spare',[-1,0])
def test_exact_admission_and_release(tmp_path,spare):
    c=4096
    payload=8+2051*8+8+4*(c*1217+13952+6144+512+6144+512+128+128+6144+6144+2560)
    weights=13952*2560*2+2560*6144*2+(256+256+128+128)*4
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=weights+payload+16+spare))
        tensors,_=synthetic(stack,tmp_path,r,e,'strata')
        before=e.stats
        assert before.device_bytes==weights and before.pinned_bytes==0
        if spare<0:
            with pytest.raises(NativeError,match='budget'): e.qsa_layer(c,*tensors,policy='strata')
        else:
            layer=e.qsa_layer(c,*tensors,policy='strata')
            assert e.stats.device_bytes-before.device_bytes==payload
            assert e.stats.pinned_bytes==16
            layer.reset()
            layer.close()
        assert e.stats.device_bytes==before.device_bytes and e.stats.pinned_bytes==0


@pytest.mark.skipif(os.environ.get('MINI_LLM_QSA_LAYER_CHECKPOINT')!='1',reason='optional bounded actual checkpoint layer')
@pytest.mark.parametrize('layer_index', range(3, 48, 4))
@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
def test_actual_checkpoint(tmp_path, layer_index, policy):
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=256*2**20))
        store = stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        prefix = f'blk.{layer_index}.'
        tensors,callbacks = [],{}
        for name,stem in zip(NAMES,STEMS+['attn_output.weight']):
            lease = stack.enter_context(store.lease(prefix+stem))
            d = lease.descriptor
            raw = lease.read(0,d.nbytes)
            tensors.append(stack.enter_context(e.upload(lease)))
            # Bounded row tiles; no full dequantized matrix, including output.
            def project(x,raw=raw,d=d):
                x=asnumpy(x)
                rows,width = d.ggml_shape[1],d.ggml_shape[0]
                y = np.empty(rows,np.float32)
                for start in range(0,rows,32):
                    count = min(32,rows-start)
                    chunk = raw[start*d.row_bytes:(start+count)*d.row_bytes]
                    if d.type_id==0: w=np.frombuffer(chunk,'<f4').reshape(count,width)
                    elif d.type_id==30: w=(np.frombuffer(chunk,'<u2').astype(np.uint32)<<16).view(np.float32).reshape(count,width)
                    else: w=PackedGGUFWeight(np.frombuffer(chunk,np.uint8),(count,width),KINDS[d.type_id]).dequantize_reference()
                    y[start:start+count]=w@x
                return xp.asarray(y)
            callbacks[name]=project
        gammas=[]
        for stem in NORMS:
            lease=stack.enter_context(store.lease(prefix+stem))
            gammas.append(np.frombuffer(lease.read(0,lease.descriptor.nbytes),'<f4').copy())
            tensors.append(stack.enter_context(e.upload(lease)))
        ref=Qwen4ExpAttention(QSAConfig(2560,tie_policy=policy),**callbacks,
            gamma_q=xp.asarray(gammas[0]),gamma_k=xp.asarray(gammas[1]),
                        gamma_index_q=xp.asarray(gammas[2]),gamma_index_k=xp.asarray(gammas[3]))
        layer=stack.enter_context(e.qsa_layer(4096,*tensors,policy=policy))
        state=ref.create_state(4096)
        xs=[hidden(t) for t in range(9)]
        uploads=stack.enter_context(inputs(tmp_path,r,e,{f'h{i}':x for i,x in enumerate(xs)}))
        maximum=max(compare_step(layer,ref,state,x,u) for x,u in zip(xs,uploads))
        assert e.stats.peak_bytes<512*2**20
        print('\nWP3_LAYER_CHECKPOINT='+json.dumps(dict(layer=layer_index,policy=policy,steps=9,max_output_abs=maximum,
              device_bytes=e.stats.device_bytes,pinned_bytes=e.stats.pinned_bytes,peak_bytes=e.stats.peak_bytes)))
