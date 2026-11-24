"""WP4 full recurrent layer; native step never consumes diagnostic readbacks."""
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec, Qwen4ExpRecurrentMixerInference
from mini_llm.native import Runtime, NativeError
from test_native_engine_hc import upload

pytestmark = pytest.mark.gpu
NAMES = ('qkv', 'z', 'beta', 'alpha', 'out')
STEMS = ('attn_qkv.weight', 'attn_gate.weight', 'ssm_beta.weight', 'ssm_alpha.weight',
         'ssm_out.weight', 'ssm_conv1d.weight', 'ssm_norm.weight', 'ssm_a', 'ssm_dt.bias')
# FP32 reductions differ between native tiles and BLAS/Triton. Fixed before execution.
RTOL, ATOL = 3e-5, 3e-6
CHECKPOINT_RTOL, CHECKPOINT_ATOL = 2e-4, 2e-4


def fixture(spec, seed=9):
    rng = np.random.default_rng(seed)
    d, inner, hv = spec.d_model, spec.inner_size, spec.n_value_heads
    matrices = [rng.normal(0, .3/np.sqrt(width), (rows, width)).astype(np.float32)
                for rows,width in [(spec.qkv_channels,d),(inner,d),(hv,d),(hv,d),(d,inner)]]
    return matrices + [rng.normal(0,.3,(spec.qkv_channels,spec.conv_kernel_size)).astype(np.float32),
                       rng.uniform(.5,1.5,spec.head_dim).astype(np.float32),
                       -rng.uniform(.1,1,hv).astype(np.float32),
                       rng.uniform(-2,1,hv).astype(np.float32)]


def reference(spec, values):
    return Qwen4ExpRecurrentMixerInference(spec, **dict(zip(NAMES,values[:5])),
        conv_weight=values[5], norm_weight=values[6], ssm_a=values[7], dt_bias=values[8])


def create(e, spec, tensors, capacity=32):
    return e.recurrent_layer(spec, capacity, *tensors, arithmetic='reference_fp32')


def read(layer, name, shape):
    result = np.empty(shape,np.float32)
    layer.readback_for_test(name,result)
    return result


def compare(layer, ref, state, x, u, *, checkpoint=False):
    want = ref.step(xp.asarray(x)[None],state)[0]
    assert layer.step(u,layer.position) == state.length == layer.position
    rtol,atol = (CHECKPOINT_RTOL,CHECKPOINT_ATOL) if checkpoint else (RTOL,ATOL)
    pairs = [(read(layer,'output',(ref.spec.d_model,)),asnumpy(want)),
             (read(layer,'history',state.conv.history.shape),asnumpy(state.conv.history)),
             (read(layer,'matrix',state.delta.memory.shape),asnumpy(state.delta.memory))]
    maxima = []
    for got,want in pairs:
        np.testing.assert_allclose(got,want,rtol=rtol,atol=atol)
        maxima.append(float(np.max(np.abs(got-want))) if got.size else 0.)
    # Raw convolution history has no arithmetic; all storage/order must be exact
    # when both paths used the same dense projection reduction (not guaranteed).
    return maxima


@pytest.mark.parametrize('kind',[0,30])
@pytest.mark.parametrize('dims',[(1,1,1,1,1),(7,2,6,5,4),(33,2,6,33,3),(65,3,9,128,4)])
def test_trajectory_reset_ownership(tmp_path,kind,dims):
    spec=Qwen4ExpMixerSpec(*dims,1e-6)
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=32*2**20))
        arrays=fixture(spec)
        tensors,decoded=upload(stack,tmp_path,r,e,arrays,[kind]*5+[0]*4)
        ref=reference(spec,[xp.asarray(a) for a in decoded]); state=ref.create_state(1)
        layer=stack.enter_context(create(e,spec,tensors,capacity=12))
        rng=np.random.default_rng(15)
        xs=[rng.normal(0,.3,spec.d_model).astype(np.float32) for _ in range(12)]
        us,_=upload(stack,tmp_path,r,e,xs)
        stable=bytes(e.stats)
        for x,u in zip(xs,us):
            compare(layer,ref,state,x,u)
            assert bytes(e.stats)==stable
        saved=[read(layer,'history',state.conv.history.shape),read(layer,'matrix',state.delta.memory.shape)]
        with pytest.raises(NativeError,match='capacity') as err: layer.step(us[0],12)
        assert err.value.status==2 and layer.position==12
        for name,old in zip(('history','matrix'),saved): np.testing.assert_array_equal(read(layer,name,old.shape),old)
        layer.reset(); state.reset()
        assert layer.position==0
        for name,old in zip(('history','matrix'),saved): np.testing.assert_array_equal(read(layer,name,old.shape),np.zeros_like(old))
        with pytest.raises(NativeError,match='unavailable'): read(layer,'output',(spec.d_model,))
        for t in tensors: t.close()
        e.close()
        for x,u in zip(xs[:3],us[:3]): compare(layer,ref,state,x,u)
        handle=layer._handle; layer.close()
        assert r._lib.ml_recurrent_reset(handle)==1
        assert r._lib.ml_recurrent_destroy(0)==0


@pytest.mark.parametrize('failure',['nan','overflow','position','shape','stale','cross'])
def test_failed_step_is_atomic(tmp_path,failure):
    spec=Qwen4ExpMixerSpec(7,2,6,5,4,1e-6); r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=2**20))
        tensors,values=upload(stack,tmp_path,r,e,fixture(spec))
        layer=stack.enter_context(create(e,spec,tensors))
        ref=reference(spec,[xp.asarray(a) for a in values]); state=ref.create_state(1)
        x=np.ones(7,np.float32)*.2
        us,_=upload(stack,tmp_path,r,e,[x,np.full(7,np.nan),np.full(7,np.finfo(np.float32).max),np.zeros(3)])
        compare(layer,ref,state,x,us[0])
        old=[read(layer,'history',state.conv.history.shape),read(layer,'matrix',state.delta.memory.shape)]
        bad={'nan':us[1],'overflow':us[2],'shape':us[3]}.get(failure,us[0]); pos=1
        if failure=='position': pos=0
        if failure=='stale':
            handle=us[3]._handle; us[3].close()
        if failure=='cross':
            other=stack.enter_context(r.engine(budget_bytes=4096))
            bad=upload(stack,tmp_path,r,other,[x])[0][0]
        if failure=='stale':
            done=C.c_uint64(77)
            assert r._lib.ml_recurrent_step(layer._handle,handle,pos,C.byref(done))==1
            assert done.value==0
        else:
            with pytest.raises(NativeError): layer.step(bad,pos)
        assert layer.position==1
        for name,a in zip(('history','matrix'),old): np.testing.assert_array_equal(read(layer,name,a.shape),a)
        with pytest.raises(NativeError,match='unavailable'): read(layer,'output',(7,))
        compare(layer,ref,state,x,us[0])
        with pytest.raises(NativeError,match='capacity'): read(layer,'matrix',(1,))
        with pytest.raises(ValueError): layer.readback_for_test('matrix',np.empty(150,np.float64))


@pytest.mark.parametrize('spare',[-1,0])
def test_exact_budget_release(tmp_path,spare):
    spec=Qwen4ExpMixerSpec(7,2,6,5,4,1e-6); arrays=fixture(spec)
    weights=sum(a.nbytes for a in arrays)
    c,i=spec.qkv_channels,spec.inner_size
    payload=4*(2*3*c+2*6*25+2*c+3*i+4*6+7+1)
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=weights+payload+4+spare))
        tensors,_=upload(stack,tmp_path,r,e,arrays)
        before=e.stats
        if spare<0:
            with pytest.raises(NativeError,match='budget'): create(e,spec,tensors)
        else:
            layer=create(e,spec,tensors)
            assert e.stats.device_bytes==weights+payload and e.stats.pinned_bytes==4
            layer.reset(); layer.close()
        assert e.stats.device_bytes==before.device_bytes and e.stats.pinned_bytes==0


@pytest.mark.parametrize('bad',['arithmetic','shape','format','nan_control','A','zero_capacity','geometry'])
def test_create_and_coefficient_contract(tmp_path,bad):
    spec=Qwen4ExpMixerSpec(7,2,6,5,4,1e-6); arrays=fixture(spec); kinds=[0]*9
    if bad=='shape': arrays[6]=np.ones(6,np.float32)
    if bad=='format': kinds[5]=30
    if bad=='nan_control': arrays[8][0]=np.nan
    if bad=='A': arrays[7][0]=1
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=2**20))
        tensors,_=upload(stack,tmp_path,r,e,arrays,kinds)
        before=e.stats
        if bad=='A':
            layer=stack.enter_context(create(e,spec,tensors))
            u=upload(stack,tmp_path,r,e,[np.ones(7)])[0][0]
            with pytest.raises(NativeError,match='converted A'): layer.step(u,0)
            assert layer.position==0
            np.testing.assert_array_equal(read(layer,'matrix',(6,5,5)),np.zeros((6,5,5),np.float32))
            return
        if bad=='geometry': spec=Qwen4ExpMixerSpec(2561,2,6,5,4,1e-6)
        with pytest.raises((ValueError,NativeError)):
            e.recurrent_layer(spec,0 if bad=='zero_capacity' else 32,*tensors,
                              arithmetic='q8_1' if bad=='arithmetic' else 'reference_fp32')
        assert e.stats.device_bytes==before.device_bytes and e.stats.pinned_bytes==0


def test_actual_checkpoint(tmp_path):
    if xp.__name__!='cupy':
        assert os.environ.get('MINI_LLM_RECURRENT_CHECKPOINT_CHILD')!='1'
        subprocess.run([sys.executable,'-m','pytest','-q','-s',str(Path(__file__).resolve())+'::test_actual_checkpoint','-x'],
                       env=dict(os.environ,MINI_LLM_BACKEND='cupy',MINI_LLM_RECURRENT_CHECKPOINT_CHILD='1',NVIDIA_TF32_OVERRIDE='0'),
                       check=True,timeout=120)
        return
    import cupy as cp
    from mini_llm.ops.quantized_linear import DevicePackedGGUFWeight, quantized_linear
    assert xp is cp
    cp.cuda.get_current_stream().synchronize(); free_before=cp.cuda.runtime.memGetInfo()[0]
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=128*2**20))
        store=stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        spec=Qwen4ExpMixerSpec.from_metadata(store.metadata)
        assert (spec.d_model,spec.n_key_heads,spec.n_value_heads,spec.head_dim,spec.conv_kernel_size)==(2560,16,48,128,4)
        tensors,values=[],[]
        kinds=[14,12,30,30,14,0,0,0,0]
        for stem,kind in zip(STEMS,kinds):
            lease=stack.enter_context(store.lease('blk.0.'+stem)); desc=lease.descriptor
            assert desc.type_id==kind
            tensors.append(stack.enter_context(e.upload(lease)))
            raw=lease.read(0,desc.nbytes)
            if kind in (12,14):
                w=DevicePackedGGUFWeight(cp.asarray(np.frombuffer(raw,np.uint8)),desc.shape,{12:'Q4_K',14:'Q6_K'}[kind])
                # Existing independent GPU packed reference_fp32 projection; no CPU decode.
                values.append(lambda x,w=w: quantized_linear(x,w))
            elif kind==30:
                bits=cp.asarray(np.frombuffer(raw,'<u2'))
                values.append((bits.astype(cp.uint32)<<cp.uint32(16)).view(cp.float32).reshape(desc.shape))
            else: values.append(cp.asarray(np.frombuffer(raw,'<f4')).reshape(desc.shape))
        ref=reference(spec,values); state=ref.create_state(1)
        layer=stack.enter_context(create(e,spec,tensors,capacity=16))
        xs=[(np.sin(np.arange(2560,dtype=np.float32)*.03125+t*.125)*.2).astype(np.float32) for t in range(16)]
        us,_=upload(stack,tmp_path,r,e,xs)
        maximum=np.zeros(3)
        stable=bytes(e.stats)
        for x,u in zip(xs,us): maximum=np.maximum(maximum,compare(layer,ref,state,x,u,checkpoint=True))
        layer.reset(); state.reset()
        for x,u in zip(xs[:4],us[:4]): maximum=np.maximum(maximum,compare(layer,ref,state,x,u,checkpoint=True))
        assert bytes(e.stats)==stable and e.stats.peak_bytes<128*2**20
        cp.cuda.get_current_stream().synchronize()
        extra=free_before-cp.cuda.runtime.memGetInfo()[0]
        assert extra<512*2**20
        print('WP4_RECURRENT_CHECKPOINT='+json.dumps(dict(layer=0,steps=16,reset_steps=4,
            max_output_history_matrix_abs=maximum.tolist(),device_bytes=e.stats.device_bytes,
            pinned_bytes=e.stats.pinned_bytes,peak_bytes=e.stats.peak_bytes,observed_gpu_extra_bytes=extra)))
