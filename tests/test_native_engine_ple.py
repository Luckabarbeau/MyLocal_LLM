"""WP4 bounded native PLE integration; Python reference is oracle only."""
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
from mini_llm.native import Runtime, NativeError
from mini_llm.external.gguf_store import MultiShardTensorStore
from mini_llm.external.qwen4exp_ple import PLEConfig, Qwen4ExpPLE
from test_0201_gguf_store import write_gguf
from test_native_engine_hc import upload

pytestmark = pytest.mark.gpu
D, W = 2560, 10240
ATOL, RTOL = 8e-6, 8e-6


def read(state, name):
    shapes = dict(streams=(4,D), history=(9,4,D), embedding=(D,), gates=(4,))
    out = np.empty(shapes[name], np.float32)
    state.readback_for_test(name, out)
    return out


def compare(got, expected):
    np.testing.assert_allclose(got, asnumpy(expected), atol=ATOL, rtol=RTOL)


def synthetic(stack, tmp_path, runtime, engine, kind=1, rows=2**31):
    # A huge sparse immutable source with only selected rows written. No full payload.
    config = PLEConfig((2**64-1, 2**63+37, 0), tuple(i*(rows//16) for i in range(16)),
                       (rows//16,)*16, 7, rows)
    tokens = [1,7,7,2,3,248044,0,2**31-1,4,5,6,9,10,11,12,13]
    table_path = tmp_path/'table.gguf'
    base = write_gguf(table_path, [('table',(160,rows),20,0,b'')])
    selected, previous = set(), (-1,-1)
    for token in tokens:
        selected.update(config.rows(token,previous)); previous=(previous[1],token)
    # Covers every nibble and both halves of every IQ4_NL block, varied signed scale.
    with table_path.open('r+b') as f:
        f.truncate(base+rows*90)
        for row in selected:
            raw = bytearray()
            for block in range(5):
                raw += np.float16((1 if row%2 else -1)*(.001+block*.0002)).tobytes()
                raw += bytes((row+block+j*17)%256 for j in range(16))
            f.seek(base+row*90); f.write(raw)
    store=stack.enter_context(runtime.gguf(table_path)); lease=stack.enter_context(store.lease('table'))
    rng=np.random.default_rng(4)
    arrays=[rng.normal(0,.01,(W,D)).astype(np.float32),rng.normal(0,.01,(D,D)).astype(np.float32)]
    arrays += [rng.uniform(.5,1.5,(4,D)).astype(np.float32) for _ in range(3)]
    arrays += [rng.normal(0,.1,(W,4)).astype(np.float32)]
    tensors,decoded=upload(stack,tmp_path,runtime,engine,arrays,[30,30,kind,kind,kind,kind])
    weights=stack.enter_context(engine.ple_weights(lease,config,*tensors,epsilon=1e-6,arithmetic='reference_fp32'))
    oracle_store=stack.enter_context(MultiShardTensorStore.open(table_path))
    reference=Qwen4ExpPLE(oracle_store,'table',config,*(xp.asarray(a) for a in decoded),epsilon=1e-6)
    return weights,reference,config,tokens,store,lease,tensors


@pytest.mark.parametrize('kind',[0,1,30])
def test_sparse_16_steps_reset_lifetimes_and_hc(tmp_path,kind):
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=128*2**20))
        weights,ref,config,tokens,store,lease,tensors=synthetic(stack,tmp_path,r,e,kind)
        state=stack.enter_context(weights.state(16))
        second=stack.enter_context(weights.state(16))
        rng=np.random.default_rng(99)
        streams=rng.normal(0,.2,(4,D)).astype(np.float32)
        inputs,_=upload(stack,tmp_path,r,e,[streams])
        hc=stack.enter_context(e.hc_state(D,1)); hc.initialize(inputs[0])
        # Close all public native sources before any table fetch. The sparse mmap
        # lease and device uploads must remain owned by PLE, not Python wrappers.
        store.close(); lease.close()
        for t in tensors: t.close()
        weights.close()
        baseline=(e.stats.device_bytes,e.stats.pinned_bytes,e.stats.peak_bytes)
        maximum=0.
        for replay in range(2):
            expected=ref.fresh_state(); hs=xp.asarray(streams)
            for j,token in enumerate(tokens):
                old=expected.predecessors
                want,expected=ref.step(token,xp.asarray(streams),expected)
                state.step(inputs[0],token,j)
                compare(read(state,'streams'),want); compare(read(state,'history'),expected.normalized_history)
                compare(read(state,'embedding'),ref.embedding_reference(token,old))
                assert state.metadata==(expected.predecessors,config.rows(token,old),j+1)
                maximum=max(maximum,float(np.max(np.abs(read(state,'streams')-asnumpy(want)))))
                # Independent second sequence applies directly to engine HC streams.
                hwant,hstate=ref.step(token,hs,ref.fresh_state() if j==0 else hstate)
                second.step(hc,token,j); hs=hwant
                actual=np.empty((4,D),np.float32); hc.readback_for_test('streams',actual)
                compare(actual,hwant); compare(read(second,'streams'),hwant)
                compare(read(second,'history'),hstate.normalized_history)
                assert (e.stats.device_bytes,e.stats.pinned_bytes,e.stats.peak_bytes)==baseline
            state.reset(); second.reset(); hc.reset(); hc.initialize(inputs[0])
            assert state.metadata==((-1,-1),(0,)*16,0)
            np.testing.assert_array_equal(read(state,'history'),0)
        ref.close()
        print('WP4_PLE_SYNTHETIC='+json.dumps(dict(kind=kind,max_abs=maximum,device=baseline[0],pinned=baseline[1])))
        # Allocations remain independently releasable after direct HC use.
        hc.close(); state.close(); second.close(); inputs[0].close()
        assert e.stats.device_bytes==e.stats.pinned_bytes==0


def test_failure_rollback_and_budget(tmp_path):
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=128*2**20-1))
        weights,ref,config,tokens,store,lease,tensors=synthetic(stack,tmp_path,r,e)
        state=stack.enter_context(weights.state(2))
        values=[np.ones((4,D),np.float32),np.full((4,D),np.nan,np.float32),np.ones((D,),np.float32)]
        u,_=upload(stack,tmp_path,r,e,values)
        other=stack.enter_context(r.engine(budget_bytes=2**20))
        cross,_=upload(stack,tmp_path,r,other,[values[0]])
        state.step(u[0],1,0)
        history=read(state,'history'); metadata=state.metadata
        for action in [lambda: state.step(u[1],7,1),lambda: state.step(u[2],7,1),
                       lambda: state.step(cross[0],7,1),lambda: state.step(u[0],7,0),
                       lambda: r._call('ml_ple_step',state._handle,u[0]._handle,-1,1,0),
                       lambda: r._call('ml_ple_step',state._handle,0,1,1,0)]:
            with pytest.raises(NativeError): action()
            assert state.metadata==metadata
            np.testing.assert_array_equal(read(state,'history'),history)
            with pytest.raises(NativeError,match='unavailable'): read(state,'streams')
        state.step(u[0],7,1) # failure was not an EOS/history commit
        assert state.metadata[0]==(1,7)
        with pytest.raises(NativeError,match='capacity'): state.step(u[0],2,2)
        state.reset(); state.step(u[0],1,0)
        with pytest.raises(NativeError,match='capacity'):
            state.readback_for_test('streams',np.empty(1,np.float32))
        with pytest.raises(ValueError): state.readback_for_test('streams',np.empty((4,D),np.float64))
        with pytest.raises(ValueError): state.step(u[0],2**31,1)
        with pytest.raises(ValueError): state.step(u[0],True,1)
        # Direct C ABI config validation: no Python pre-validation can mask it.
        h=C.c_uint64(123); handles=(C.c_uint64*6)(*(x._handle for x in tensors))
        def create(vocabs, offsets=config.offsets, arithmetic=0, eps=1e-6):
            return r._call('ml_ple_weights_create',e._handle,lease._handle,handles,
                (C.c_uint64*3)(*config.multipliers),(C.c_uint64*16)(*offsets),
                (C.c_uint64*16)(*vocabs),config.eos,eps,arithmetic,C.byref(h))
        for kwargs in [dict(vocabs=(0,)*16),dict(vocabs=(1,)*16,offsets=(2**64-1,)*16),
                       dict(vocabs=config.vocabs,arithmetic=1),dict(vocabs=config.vocabs,eps=float('nan'))]:
            with pytest.raises(NativeError): create(**kwargs)
            assert h.value==0
        # Exact state budget boundary using the same retained weight owner.
        before=e.stats.device_bytes+e.stats.pinned_bytes
        probe=weights.state(1); delta=e.stats.device_bytes+e.stats.pinned_bytes-before; probe.close()
        # Capacity itself never scales the allocation.
        probe=weights.state(2**64-1); assert e.stats.device_bytes+e.stats.pinned_bytes-before==delta; probe.close()
        # Fill to exactly one byte short of state and weight-owner admission.

        required = [delta, 40960*4+3*10240*4+8]
        for need in required:
            free=e.stats.budget_bytes-e.stats.device_bytes-e.stats.pinned_bytes
            fill=(free-need)//4+1
            filler,_=upload(stack,tmp_path,r,e,[np.zeros(fill,np.float32)])
            snapshot=(e.stats.device_bytes,e.stats.pinned_bytes)
            assert e.stats.budget_bytes-sum(snapshot)==need-1
            with pytest.raises(NativeError,match='budget'):
                if need==delta: weights.state(1)
                else: create(config.vocabs)
            assert (e.stats.device_bytes,e.stats.pinned_bytes)==snapshot
            filler[0].close()
        # HC initialization/transaction constraints are rejected before any commit.
        hc=stack.enter_context(e.hc_state(D,1))
        state.reset()
        with pytest.raises(NativeError): state.step(hc,1,0)
        hc.initialize(u[0])
        gamma,down,up,inject=upload(stack,tmp_path,r,e,
            [np.ones(W,np.float32),np.zeros((1,W),np.float32),
             np.zeros((W,1),np.float32),np.zeros((4,W),np.float32)])[0]
        hw=stack.enter_context(e.hc_weights(gamma,down,up,inject,epsilon=1e-6,arithmetic='reference_fp32'))
        ticket=hc.read(hw)
        with pytest.raises(NativeError,match='transaction'): state.step(hc,1,0)
        assert state.metadata==((-1,-1),(0,)*16,0)
        hc.write(ticket,u[2]); state.step(hc,1,0)
        # All native dependencies survive public engine and weight-owner close.
        e.close(); weights.close(); store.close(); lease.close()
        for tensor in tensors: tensor.close()
        state.reset(); state.step(u[0],1,0)
        ref.close()


def test_invalid_handles_and_partial_construction():
    r=Runtime(); out=C.c_uint64(123)
    with pytest.raises(NativeError,match='handle'):
        r._call('ml_ple_create',0,1,C.byref(out))
    assert out.value==0
    for name in ['ml_ple_destroy','ml_ple_weights_destroy']:
        r._call(name,0)
        with pytest.raises(NativeError,match='handle'): r._call(name,2**64-1)
    for name in ['ml_ple_reset','ml_ple_step','ml_ple_metadata','ml_ple_readback_for_test']:
        args={'ml_ple_reset':[], 'ml_ple_step':[0,1,0,0],
              'ml_ple_metadata':[None,None,None], 'ml_ple_readback_for_test':[0,None,0]}[name]
        with pytest.raises(NativeError,match='handle'): r._call(name,2**64-1,*args)


def test_actual_checkpoint(tmp_path):
    if xp.__name__!='cupy':
        assert os.environ.get('MINI_LLM_PLE_CHECKPOINT_CHILD')!='1'
        subprocess.run([sys.executable,'-m','pytest','-q','-s',str(Path(__file__).resolve())+'::test_actual_checkpoint','-x'],
            check=True,timeout=180,env=dict(os.environ,MINI_LLM_BACKEND='cupy',MINI_LLM_PLE_CHECKPOINT_CHILD='1',NVIDIA_TF32_OVERRIDE='0'))
        return
    import cupy as cp
    assert xp is cp
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    path=inventory['shards'][0]['path']; r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=256*2**20))
        store=stack.enter_context(r.gguf(path)); table=stack.enter_context(store.lease('per_layer_token_embd.weight'))
        assert table.descriptor.type_id==20 and table.descriptor.row_bytes==90
        assert table.descriptor.shape==(320001536,160)
        config=PLEConfig.from_metadata(store.metadata,table.descriptor.shape[0])
        tensors,decoded=[],[]
        suffixes=['key','value','norm_key','norm_query','norm_conv','conv1d']
        for suffix in suffixes:
            lease=stack.enter_context(store.lease('blk.1.ple_'+suffix+'.weight')); desc=lease.descriptor
            tensors.append(stack.enter_context(e.upload(lease)))
            raw=lease.read(0,desc.nbytes)
            if desc.type_id==30:
                value=(cp.asarray(np.frombuffer(raw,'<u2')).astype(cp.uint32)<<cp.uint32(16)).view(cp.float32)
            else: value=cp.asarray(np.frombuffer(raw,'<f2' if desc.type_id==1 else '<f4')).astype(cp.float32)
            shape=(4,D) if suffix.startswith('norm_') else desc.shape
            decoded.append(value.reshape(shape))
        eps=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
        weights=stack.enter_context(e.ple_weights(table,config,*tensors,epsilon=eps,arithmetic='reference_fp32'))
        state=stack.enter_context(weights.state(16))
        oracle=stack.enter_context(MultiShardTensorStore.open(path))
        ref=Qwen4ExpPLE(oracle,'per_layer_token_embd.weight',config,*decoded,epsilon=eps)
        tokens=[1,2,config.eos,3,config.eos,config.eos,0,248056,17,88,1234,5,6,7,8,9]
        streams=(np.sin(np.arange(W,dtype=np.float32)*.03125)*.2).reshape(4,D)
        u,_=upload(stack,tmp_path,r,e,[streams])
        # Native table and weight leases survive closing BOTH public sources.
        store.close(); table.close()
        for tensor in tensors: tensor.close()
        weights.close()
        baseline=(e.stats.device_bytes,e.stats.pinned_bytes,e.stats.peak_bytes)
        assert baseline[2]<512*2**20
        maximum=history_max=0.
        for replay in range(2):
            expected=ref.fresh_state()
            for j,token in enumerate(tokens):
                old=expected.predecessors
                want,expected=ref.step(token,cp.asarray(streams),expected)
                state.step(u[0],token,j)
                got=read(state,'streams'); hist=read(state,'history')
                compare(got,want); compare(hist,expected.normalized_history)
                np.testing.assert_array_equal(read(state,'embedding'),asnumpy(ref.embedding_reference(token,old)))
                assert state.metadata==(expected.predecessors,config.rows(token,old),j+1)
                maximum=max(maximum,float(np.max(np.abs(got-asnumpy(want)))))
                history_max=max(history_max,float(np.max(np.abs(hist-asnumpy(expected.normalized_history)))))
                assert (e.stats.device_bytes,e.stats.pinned_bytes,e.stats.peak_bytes)==baseline
            state.reset()
            np.testing.assert_array_equal(read(state,'history'),0)
            assert state.metadata==((-1,-1),(0,)*16,0)
        ref.close()
        print('WP4_PLE_CHECKPOINT='+json.dumps(dict(steps=32,resets=2,max_abs=maximum,history_max_abs=history_max,
            device_bytes=baseline[0],pinned_bytes=baseline[1],peak_bytes=baseline[2],atol=ATOL,rtol=RTOL)))
