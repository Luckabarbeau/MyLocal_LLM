"""WP4 engine-owned HC transactions, no production Python changes or fallback.

The checkpoint gate always executes (GPU reference in a CuPy child when the broad
suite uses NumPy). Only three selected layers' HC and final HC are held at once.
Synthetic block outputs are pre-uploaded independently of diagnostic readbacks.
"""
from contextlib import ExitStack
import ctypes as C
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.native import Runtime, NativeError
from test_0201_gguf_store import write_gguf

pytestmark = pytest.mark.gpu
SERIAL = itertools.count()
RTOL, ATOL = 2e-6, 2e-6


def upload(stack, path, runtime, engine, arrays, kinds=None):
    """Unique immutable fixture file per upload group (never overwrite live mmap)."""
    tensors, offset, expected = [], 0, []
    kinds = [0] * len(arrays) if kinds is None else kinds
    for j, (a, kind) in enumerate(zip(arrays, kinds)):
        a = np.asarray(a, np.float32)
        if kind == 30:
            bits = (a.view(np.uint32) >> 16).astype('<u2')
            raw = bits.tobytes()
            decoded = (bits.astype(np.uint32) << 16).view(np.float32)
        elif kind == 1:
            raw, decoded = a.astype('<f2').tobytes(), a.astype(np.float16).astype(np.float32)
        else:
            raw, decoded = a.tobytes(), a.copy()
        tensors.append((str(j), a.shape[::-1], kind, offset, raw))
        expected.append(decoded)
        offset += (len(raw) + 31) // 32 * 32
    file = path / f'hc-{next(SERIAL)}.gguf'
    write_gguf(file, tensors)
    store = stack.enter_context(runtime.gguf(file))
    result = []
    for j in range(len(arrays)):
        lease = stack.enter_context(store.lease(str(j)))
        result.append(stack.enter_context(engine.upload(lease)))
    return result, expected


def make_weights(stack, path, runtime, engine, d=7, rank=5, kind=0,
                 grouped=False, inject=True, seed=4, exact=False):
    rng = np.random.default_rng(seed)
    arrays = [rng.uniform(.4, 1.6, (4, d)).astype(np.float32),
              rng.normal(0, .3 / np.sqrt(d), (rank, 4*d)).astype(np.float32),
              rng.normal(0, .4 / np.sqrt(rank), (4*d, rank)).astype(np.float32)]
    if inject:
        arrays.append(rng.normal(0, .4 / np.sqrt(d), (4, 4*d)).astype(np.float32))
    if exact:
        arrays[0][:] = np.array([[2], [4], [6], [8]], np.float32)
        for a in arrays[1:]: a.fill(0)
    if not grouped: arrays[0] = arrays[0].reshape(-1)
    tensors, decoded = upload(stack, path, runtime, engine, arrays, [kind]*len(arrays))
    weights = stack.enter_context(engine.hc_weights(*tensors, epsilon=3 if exact else 1e-6,
                                                    arithmetic='reference_fp32'))
    reference = Qwen4ExpHyperConnection(*(xp.asarray(a) for a in decoded),
                                       epsilon=3 if exact else 1e-6)
    return weights, reference, tensors


def read(state, which, d):
    shape = (4, d) if which == 'streams' else (d,) if which == 'hidden' else (4,)
    a = np.empty(shape, np.float32)
    state.readback_for_test(which, a)
    return a


def close(got, expected):
    np.testing.assert_allclose(got, asnumpy(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('kind', [0, 30])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('d,rank', [(1, 1), (7, 5), (33, 35), (257, 257)])
def test_paired_attention_ffn_collapse(tmp_path, kind, grouped, d, rank):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=32*2**20))
        attn, ar, _ = make_weights(stack, tmp_path, r, e, d, rank, kind, grouped, seed=4)
        ffn, fr, _ = make_weights(stack, tmp_path, r, e, d, rank, kind, grouped, seed=5)
        final, final_ref, _ = make_weights(stack, tmp_path, r, e, d, rank, kind, grouped,
                                           inject=False, seed=6)
        state = stack.enter_context(e.hc_state(d, rank))
        embedding = np.sin(np.arange(d, dtype=np.float32)*.125).astype(np.float32)
        outputs = [(np.cos(np.arange(d, dtype=np.float32)*.07+i)*.2).astype(np.float32)
                   for i in range(6)]
        u, _ = upload(stack, tmp_path, r, e, [embedding]+outputs)
        baseline = bytes(e.stats)
        for replay in range(2):
            state.initialize(u[0])
            streams = ar.initialize(xp.asarray(embedding))
            for j in range(6):
                w, ref = (attn, ar) if j % 2 == 0 else (ffn, fr)
                hidden, gains = ref.read(streams)
                ticket = state.read(w)
                close(read(state, 'hidden', d), hidden)
                close(read(state, 'gains', d), gains)
                close(read(state, 'streams', d), streams)
                # Diagnostics are not a source of any native input or block output.
                saved_gains = read(state, 'gains', d)
                np.testing.assert_array_equal(read(state, 'gains', d), saved_gains)
                state.write(ticket, u[j+1])
                streams = ref.write(streams, xp.asarray(outputs[j]), gains)
                close(read(state, 'streams', d), streams)
                with pytest.raises(NativeError, match='unavailable'):
                    read(state, 'gains', d)
                with pytest.raises(NativeError, match='unavailable'):
                    read(state, 'hidden', d)
                assert bytes(e.stats) == baseline
            state.collapse(final)
            close(read(state, 'hidden', d), final_ref.collapse(streams))
            close(read(state, 'streams', d), streams)
            assert bytes(e.stats) == baseline
            state.reset()
            with pytest.raises(NativeError, match='uninitialized'):
                read(state, 'streams', d)


def test_exact_graph_and_lifecycle(tmp_path):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**20))
        weights, _, tensors = make_weights(stack, tmp_path, r, e, 1, 1, 30, exact=True)
        final, _, final_tensors = make_weights(stack, tmp_path, r, e, 1, 1, 30, inject=False, exact=True)
        u, _ = upload(stack, tmp_path, r, e, [np.ones((4,1)), np.array([2])])
        state = stack.enter_context(e.hc_state(1, 1))
        state.initialize(u[0])
        # Weight owners and state retain engine/upload ownership independently.
        for t in tensors + final_tensors: t.close()
        e.close()
        ticket = state.read(weights)
        np.testing.assert_array_equal(read(state, 'hidden', 1), [1.25])
        np.testing.assert_array_equal(read(state, 'gains', 1), [1,1,1,1])
        weights.close()  # Pending transaction retains its exact native weight owner.
        state.write(ticket, u[1])
        np.testing.assert_array_equal(read(state, 'streams', 1), np.full((4,1),3))
        state.collapse(final)
        close(read(state, 'hidden', 1), [7.5/np.sqrt(12)])
        old = state._handle
        state.close()
        assert r._lib.ml_hc_reset(old) == 1
        assert r._lib.ml_hc_destroy(0) == 0
        assert r._lib.ml_hc_weights_destroy(0) == 0


@pytest.mark.parametrize('operation', ['write_without_read', 'double_read', 'wrong_ticket',
    'stale_reset_ticket', 'other_state_ticket', 'collapse_pending', 'read_final',
    'collapse_injecting', 'after_collapse', 'wrong_output', 'cross_output',
    'cross_weights', 'stale_output', 'null_snapshot', 'reinitialize', 'nan_output',
    'overflow_output', 'nan_initialize', 'rank_mismatch'])
def test_invalid_transactions_require_reset(tmp_path, operation):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**20))
        other = stack.enter_context(r.engine(budget_bytes=2**20))
        w, _, _ = make_weights(stack, tmp_path, r, e)
        final, _, _ = make_weights(stack, tmp_path, r, e, inject=False)
        cross_w, _, _ = make_weights(stack, tmp_path, r, other)
        arrays = [np.ones((4,7), np.float32), np.ones(7,np.float32),
                  np.ones(3,np.float32), np.full(7,np.nan,np.float32),
                  np.full(7,np.finfo(np.float32).max,np.float32), np.full((4,7),np.nan,np.float32)]
        u, _ = upload(stack, tmp_path, r, e, arrays)
        cross, _ = upload(stack, tmp_path, r, other, [arrays[1]])
        state = stack.enter_context(e.hc_state(7, 5))
        state.initialize(u[0])
        ticket = state.read(w)
        if operation == 'write_without_read':
            state.write(ticket, u[1])
            action = lambda: state.write(ticket, u[1])
        elif operation == 'double_read': action = lambda: state.read(w)
        elif operation == 'wrong_ticket': action = lambda: state.write(ticket+1, u[1])
        elif operation == 'stale_reset_ticket':
            state.reset(); state.initialize(u[0]); state.read(w)
            action = lambda: state.write(ticket, u[1])
        elif operation == 'other_state_ticket':
            second = stack.enter_context(e.hc_state(7,5)); second.initialize(u[0])
            other_ticket = second.read(w)
            assert ticket != other_ticket
            action = lambda: state.write(other_ticket, u[1])
        elif operation == 'collapse_pending': action = lambda: state.collapse(final)
        elif operation == 'read_final':
            state.write(ticket,u[1]); action = lambda: state.read(final)
        elif operation == 'collapse_injecting':
            state.write(ticket,u[1]); action = lambda: state.collapse(w)
        elif operation == 'after_collapse':
            state.write(ticket,u[1]); state.collapse(final); action = lambda: state.read(w)
        elif operation == 'wrong_output': action = lambda: state.write(ticket,u[2])
        elif operation == 'cross_output': action = lambda: state.write(ticket,cross[0])
        elif operation == 'cross_weights':
            state.write(ticket,u[1]); action = lambda: state.read(cross_w)
        elif operation == 'stale_output':
            stale = u[2]._handle; u[2].close()
            action = lambda: r._call('ml_hc_write',state._handle,ticket,stale)
        elif operation == 'null_snapshot':
            state.write(ticket,u[1])
            action = lambda: r._call('ml_hc_read',state._handle,w._handle,None)
        elif operation == 'reinitialize': action = lambda: state.initialize(u[0])
        elif operation == 'nan_output': action = lambda: state.write(ticket,u[3])
        elif operation == 'overflow_output':
            # Force gains above 1 with an exact deterministic injection fixture.
            state.reset()
            a = [np.ones(28), np.ones((5,28)), np.ones((28,5)), np.ones((4,28))]
            t, _ = upload(stack,tmp_path,r,e,a)
            positive = stack.enter_context(e.hc_weights(*t,epsilon=1e-6,arithmetic='reference_fp32'))
            state.initialize(u[0]); ticket = state.read(positive)
            action = lambda: state.write(ticket,u[4])
        elif operation == 'nan_initialize':
            state.reset(); action = lambda: state.initialize(u[5])
        else:
            state.reset(); state.close()
            state = stack.enter_context(e.hc_state(7,4)); state.initialize(u[0])
            action = lambda: state.read(w)
        with pytest.raises(NativeError): action()
        for resume in (lambda: state.read(w), lambda: state.write(ticket,u[1]),
                       lambda: state.collapse(final), lambda: state.initialize(u[0]),
                       lambda: read(state,'streams',7)):
            with pytest.raises(NativeError, match='invalid|reset'): resume()
        state.reset(); state.initialize(u[0])
        if operation != 'rank_mismatch':
            fresh = state.read(w)
            assert fresh != ticket
            state.write(fresh,u[1]); state.collapse(final)


@pytest.mark.parametrize('bad', ['gamma_shape','down_shape','up_shape','inject_shape',
                                 'f16','nan','inf_bf16','epsilon','cross_engine','arithmetic'])
def test_weight_validation(tmp_path,bad):
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=2**20))
        other=stack.enter_context(r.engine(budget_bytes=2**20))
        a=[np.ones(28,np.float32),np.ones((5,28),np.float32),np.ones((28,5),np.float32),np.ones((4,28),np.float32)]
        if bad.endswith('_shape'):
            j=['gamma_shape','down_shape','up_shape','inject_shape'].index(bad)
            a[j]=np.ones((2,2),np.float32)
        kinds=[0]*4
        if bad=='f16': kinds[1]=1
        if bad=='nan': a[2][0,0]=np.nan
        if bad=='inf_bf16': a[1][0,0]=np.inf; kinds[1]=30
        u,_=upload(stack,tmp_path,r,e,a,kinds)
        if bad=='cross_engine':
            v,_=upload(stack,tmp_path,r,other,[a[0]]); u[0]=v[0]
        before=e.stats
        with pytest.raises((NativeError,ValueError)):
            e.hc_weights(*u,epsilon=0 if bad=='epsilon' else 1e-6,
                         arithmetic='invalid' if bad=='arithmetic' else 'reference_fp32')
        after=e.stats
        assert (after.device_bytes,after.pinned_bytes)==(before.device_bytes,before.pinned_bytes)


@pytest.mark.parametrize('kind',[0,30])
@pytest.mark.parametrize('spare',[-1,0])
def test_budget_ownership(tmp_path,kind,spare):
    d,rank=7,5
    count=4*d+2*rank*4*d+4*4*d
    stored=count*(2 if kind==30 else 4)
    converted=(count*4 if kind==30 else 0)+4
    state_bytes=(13*d+rank+5)*4
    r=Runtime()
    with ExitStack() as stack:
        # Admit exact uploaded bytes + weight payload, then exact state admission.
        e=stack.enter_context(r.engine(budget_bytes=stored+converted+4+state_bytes+4+spare))
        w,_,t=make_weights(stack,tmp_path,r,e,d,rank,kind)
        before=e.stats
        if spare<0:
            with pytest.raises(NativeError,match='budget'): e.hc_state(d,rank)
            assert bytes(e.stats)==bytes(before)
        else:
            state=e.hc_state(d,rank)
            assert e.stats.device_bytes==stored+converted+state_bytes
            assert e.stats.pinned_bytes==8
            state.close()
            assert e.stats.device_bytes==before.device_bytes
            assert e.stats.pinned_bytes==before.pinned_bytes
        for item in t: item.close()
        assert e.stats.device_bytes==stored+converted # retained by weight owner
        w.close()
        assert e.stats.device_bytes==e.stats.pinned_bytes==0


def test_weight_conversion_budget_and_diagnostics(tmp_path):
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=240))
        a=[np.ones(4),np.ones((1,4)),np.ones((4,1)),np.ones((4,4))]
        u,_=upload(stack,tmp_path,r,e,a,[30]*4) # 56 uploaded bytes, conversion 112+8
        # 240 is enough for this weight owner; one byte short tested on other owner.
        weights=stack.enter_context(e.hc_weights(*u,epsilon=1e-6,arithmetic='reference_fp32'))
        assert e.stats.device_bytes==172 and e.stats.pinned_bytes==4
        small=stack.enter_context(r.engine(budget_bytes=175))
        v,_=upload(stack,tmp_path,r,small,a,[30]*4)
        before=bytes(small.stats)
        with pytest.raises(NativeError,match='budget'):
            small.hc_weights(*v,epsilon=1e-6,arithmetic='reference_fp32')
        assert bytes(small.stats)==before
        big=stack.enter_context(r.engine(budget_bytes=2**20))
        w,_,_=make_weights(stack,tmp_path,r,big)
        h,_=upload(stack,tmp_path,r,big,[np.ones(7)])
        state=stack.enter_context(big.hc_state(7,5)); state.initialize(h[0]); ticket=state.read(w)
        with pytest.raises(NativeError,match='capacity'):
            state.readback_for_test('streams',np.empty(1,np.float32))
        with pytest.raises(ValueError): state.readback_for_test('hidden',np.empty(7,np.float64))
        # Diagnostics do not poison or replace the pending snapshot.
        state.write(ticket,h[0])


def test_actual_checkpoint(tmp_path):
    if xp.__name__ != 'cupy':
        assert os.environ.get('MINI_LLM_HC_CHECKPOINT_CHILD') != '1', 'CuPy backend was not selected'
        env=dict(os.environ,MINI_LLM_BACKEND='cupy',MINI_LLM_HC_CHECKPOINT_CHILD='1',NVIDIA_TF32_OVERRIDE='0')
        subprocess.run([sys.executable,'-m','pytest','-q','-s',
                        str(Path(__file__).resolve())+'::test_actual_checkpoint','-x'],
                       check=True,env=env,timeout=120)
        return
    import cupy as cp
    assert xp is cp # real GPU oracle, never NumPy or a CPU fallback
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r=Runtime()
    for layer in [0,3,47]:
        with ExitStack() as stack:
            e=stack.enter_context(r.engine(budget_bytes=192*2**20))
            store=stack.enter_context(r.gguf(inventory['shards'][0]['path']))
            weights,refs=[],[]
            shapes=[(10240,),(320,10240),(10240,320),(4,10240)]
            for stem in (f'blk.{layer}.hc_attn_',f'blk.{layer}.hc_ffn_','output_hc_'):
                tensors,decoded=[],[]
                for suffix,shape in zip(('norm.weight','down.weight','up.weight','inject.weight'),shapes):
                    if stem=='output_hc_' and suffix=='inject.weight': continue
                    lease=stack.enter_context(store.lease(stem+suffix)); desc=lease.descriptor
                    assert desc.shape in ([shape,(4,2560)] if suffix=='norm.weight' else [shape])
                    assert desc.type_id==(0 if suffix=='norm.weight' else 30)
                    tensors.append(stack.enter_context(e.upload(lease)))
                    # Oracle gets the same stored bytes; expansion is on GPU too.
                    raw=lease.read(0,desc.nbytes)
                    if desc.type_id==30:
                        bits=cp.asarray(np.frombuffer(raw,'<u2'))
                        value=(bits.astype(cp.uint32)<<cp.uint32(16)).view(cp.float32)
                    else: value=cp.asarray(np.frombuffer(raw,'<f4'))
                    decoded.append(value.reshape(desc.shape))
                eps=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
                weights.append(stack.enter_context(e.hc_weights(*tensors,epsilon=eps,arithmetic='reference_fp32')))
                refs.append(Qwen4ExpHyperConnection(*decoded,epsilon=eps))
            state=stack.enter_context(e.hc_state(2560,320))
            embedding=(np.sin(np.arange(2560,dtype=np.float32)*.03125)*.2).astype(np.float32)
            outputs=[(np.cos(np.arange(2560,dtype=np.float32)*.02+i)*.1).astype(np.float32) for i in range(6)]
            u,_=upload(stack,tmp_path,r,e,[embedding]+outputs)
            state.initialize(u[0]); streams=refs[0].initialize(cp.asarray(embedding))
            maximum=0.
            before=bytes(e.stats)
            for j in range(6):
                ref=refs[j%2]; hidden,gains=ref.read(streams)
                ticket=state.read(weights[j%2])
                got=read(state,'hidden',2560); close(got,hidden); close(read(state,'gains',2560),gains)
                maximum=max(maximum,float(np.max(np.abs(got-asnumpy(hidden)))))
                state.write(ticket,u[j+1])
                streams=ref.write(streams,cp.asarray(outputs[j]),gains)
                close(read(state,'streams',2560),streams)
            state.collapse(weights[2]); want=refs[2].collapse(streams)
            got=read(state,'hidden',2560); close(got,want)
            maximum=max(maximum,float(np.max(np.abs(got-asnumpy(want)))))
            assert bytes(e.stats)==before
            assert e.stats.peak_bytes<192*2**20
            print('WP4_HC_CHECKPOINT='+json.dumps(dict(layer=layer,pairs=6,max_hidden_abs=maximum,
                  device_bytes=e.stats.device_bytes,pinned_bytes=e.stats.pinned_bytes,peak_bytes=e.stats.peak_bytes)))
