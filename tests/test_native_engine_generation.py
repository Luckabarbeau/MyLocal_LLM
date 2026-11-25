"""WP8 worker vs manual native replay, NOT strict Python-model parity."""
from contextlib import ExitStack
import ctypes as C
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from mini_llm.native import NativeError, SampleResult, Sampler
from mini_llm.native.generation import NativeGeneration, GenerationEvent
from test_native_engine_decoder import small_setup, forbidden
from test_native_engine_moe import libraries, selected_library

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def retain_ggml_library(selected_library):
    # This external build asserts if its C++ terminate-handler initializer runs
    # again after dlclose. Retain one test-only loader reference across fixtures.
    library = C.CDLL(selected_library)
    yield library


def collect(g):
    events = []
    deadline = time.monotonic()+25
    while time.monotonic() < deadline:
        event = g.wait(100)
        if event is not None:
            if event.kind == 2:
                return events, event
            events.append(event)
    pytest.fail('generation did not terminate within 25 seconds')


def await_state(g, predicate):
    deadline = time.monotonic()+25
    while time.monotonic() < deadline:
        state = g.state
        if predicate(*state):
            return state
        time.sleep(.005)
    pytest.fail('generation state deadline')


@pytest.mark.parametrize('temperature,top_p,top_k', [(0., 1., None), (.7, .85, 5)])
def test_native_replay_bitwise(tmp_path, libraries, monkeypatch, temperature, top_p, top_k):
    with ExitStack() as s:
        d, (r,e,_), _ = small_setup(s, tmp_path, libraries)
        sampler = s.enter_context(e.sampler(7, seed=981))
        kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k)
        token = 1; expected = []
        for _ in range(3):
            d.step_device(token)
            result = sampler.sample_decoder(d, metadata=True, **kwargs)
            expected.append(bytes(result)); token = result.token
        expected_trace = d.trace_for_test(47, 'post_moe').tobytes()
        # Replaying the pending token must consume it exactly once, not at sample time.
        expected_logits = d.step(token).tobytes()
        d.reset(); sampler.reset()
        with monkeypatch.context() as patch:
            patch.setattr(d, 'step', forbidden)
            patch.setattr(d, 'step_device', forbidden)
            patch.setattr(sampler, 'sample_decoder', forbidden)
            with NativeGeneration.start(d, sampler, 1, 3, queue_capacity=1, **kwargs) as g:
                got, final = collect(g)
                assert [bytes(x.sample) for x in got] == expected
                assert [(x.position,x.sampled,x.pending) for x in got] == [(1,1,1),(2,2,1),(3,3,1)]
                assert final.termination == 'max_tokens'
                assert (final.position,final.sampled,final.pending_token,final.pending,final.failed) == (3,3,token,1,0)
                assert g.poll() is None
                final_close = g.close()
                assert bytes(final_close) == bytes(final)
        assert d.state == (3,False)
        with pytest.raises(NativeError, match='unavailable'):
            d.trace_for_test(47, 'post_moe')  # No stale pre-generation trace exposed.
        assert d.step(final.pending_token).tobytes() == expected_logits
        # The pre-generation trace is only an oracle artifact, not a worker transfer.
        assert expected_trace
        assert g.close() is final_close


def test_full_queue_cancel_ownership_and_resume(tmp_path, libraries):
    with ExitStack() as s:
        d,(r,e,_),_ = small_setup(s,tmp_path,libraries)
        sampler=s.enter_context(e.sampler(7,seed=23))
        old_d,old_s=d._handle,sampler._handle
        with NativeGeneration.start(d,sampler,1,4,temperature=1,queue_capacity=1) as g:
            state,queued,done=await_state(g,lambda x,q,done:q==1)
            assert (state.position,state.sampled,queued,done)==(1,1,1,False)
            time.sleep(.03)
            assert g.state[0].position==1  # Backpressure reserves space before decode.
            for call in (lambda:d.step_device(2),d.reset,lambda:sampler.sample_decoder(d),sampler.reset,d.close,sampler.close):
                with pytest.raises(NativeError,match='handle'): call()
            # A caller blocked in poll/step must not serialize cancellation in Python.
            start=time.monotonic()
            thread=threading.Thread(target=g.cancel); thread.start(); thread.join(1)
            assert not thread.is_alive() and time.monotonic()-start<1
            events,final=collect(g)
            assert len(events)==1 and final.termination=='cancelled'
            assert (final.position,final.sampled,final.pending)==(1,1,1)
        assert (d._handle,sampler._handle)==(old_d,old_s)
        assert d.state==(1,False)
        assert d.step_device(final.pending_token)==2
        stale=g._runtime._lib.ml_generation_cancel(g._handle or 999999999)
        assert stale==1


@pytest.mark.parametrize('case', ['eos','capacity','zero','close_unread','cancel_idle'])
def test_termination_and_close(tmp_path,libraries,case):
    with ExitStack() as s:
        d,(r,e,_),_=small_setup(s,tmp_path,libraries)
        sampler=s.enter_context(e.sampler(7,seed=9))
        maximum=0 if case=='zero' else 20
        eos=range(7) if case=='eos' else ()
        g=NativeGeneration.start(d,sampler,1,maximum,eos_ids=eos,queue_capacity=1)
        try:
            if case=='close_unread':
                await_state(g,lambda x,q,done:q==1)
                e.close()  # Outstanding owners retain engine and weights.
                start=time.monotonic(); final=g.close()
                assert time.monotonic()-start<5
                assert final.termination=='cancelled' and final.position==1
            elif case=='cancel_idle':
                g.cancel(); _,final=collect(g)
                assert final.termination=='cancelled'
                assert final.position in (0,1)  # Dispatch race: running work is drained.
                assert final.sampled==final.position
            else:
                events,final=collect(g)
                reason={'eos':'eos','capacity':'capacity','zero':'max_tokens'}[case]
                count={'eos':1,'capacity':4,'zero':0}[case]
                assert final.termination==reason
                assert len(events)==count and final.position==count
                assert final.sampled==count and final.pending==bool(count)
                assert final.pending_token==(events[-1].sample.token if count else -1)
        finally:
            g.close()
        assert d.state==(final.position,False)


def test_late_layer_failure_and_terminal_drain(tmp_path,libraries):
    with ExitStack() as s:
        d,(r,e,_),_=small_setup(s,tmp_path,libraries,bad_last=True)
        sampler=s.enter_context(e.sampler(7,seed=1))
        with NativeGeneration.start(d,sampler,1,4) as g:
            events,final=collect(g)
            assert not events
            assert final.termination=='error' and final.status==1
            assert b'MoE' in final.error
            assert (final.position,final.sampled,final.pending,final.failed)==(0,0,0,1)
        assert d.state==(0,True)
        with pytest.raises(NativeError,match='unresumable'): d.step_device(1)
        d.reset()
        assert d.state==(0,False)


def test_validation_wrong_stale_handles_and_shutdown(tmp_path,libraries):
    with ExitStack() as s:
        d,(r,e,_),_=small_setup(s,tmp_path,libraries)
        sampler=s.enter_context(e.sampler(7,seed=1))
        lib=r._lib; out=C.c_uint64(); event=GenerationEvent()
        args=(1,2,None,0,0.,1.,0,1,C.byref(out))
        for dh,sh in ((sampler._handle,d._handle),(999999,sampler._handle),(d._handle,999999)):
            assert lib.ml_generation_start(dh,sh,*args)==1 and out.value==0
        for kwargs in ({'queue_capacity':0},{'temperature':-1},{'top_p':0},{'eos_ids':[7]}):
            with pytest.raises((NativeError,ValueError)):
                NativeGeneration.start(d,sampler,1,2,**kwargs)
        with e.sampler(6,seed=1) as wrong:
            with pytest.raises(NativeError): NativeGeneration.start(d,wrong,1,2)
        with r.engine() as other,other.sampler(7,seed=1) as wrong:
            with pytest.raises(NativeError): NativeGeneration.start(d,wrong,1,2)
        assert d.state==(0,False)
        g=NativeGeneration.start(d,sampler,1,4)
        handle=g._handle
        await_state(g,lambda x,q,done:q==1)
        assert lib.ml_generation_poll(handle,120001,C.byref(event))==1
        assert lib.ml_generation_poll(handle,0,None)==1
        e.close()
        del g; gc.collect()  # Python owner destructor must cancel/join full queue.
        assert lib.ml_generation_cancel(handle)==1
        assert lib.ml_generation_poll(handle,0,C.byref(event))==1
        assert d.state==(1,False)
        assert sampler.sample_decoder(d)>=0


def test_process_teardown_with_outstanding_native_worker(tmp_path,libraries):
    # Bypass the Python Generation wrapper: exercise native static-owner teardown
    # with a full queue and no close call at all, not merely Python GC cleanup.
    code = '''
import ctypes as C, sys, time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0, 'tests')
from test_native_engine_decoder import small_setup
s=ExitStack()
d,(r,e,_),_=small_setup(s,Path(sys.argv[1]),__import__('json').loads(sys.argv[2]))
sampler=s.enter_context(e.sampler(7,seed=1))
out=C.c_uint64()
r._call('ml_generation_start',d._handle,sampler._handle,1,4,None,0,0.,1.,0,1,C.byref(out))
from mini_llm.native.generation import GenerationEvent
state=GenerationEvent(); queued=C.c_uint64(); done=C.c_int32()
deadline=time.monotonic()+20
while not queued.value:
    assert time.monotonic()<deadline
    assert r._lib.ml_generation_state(out.value,C.byref(state),C.byref(queued),C.byref(done))==0
    time.sleep(.005)
print('outstanding full queue', flush=True)
'''
    completed=subprocess.run([sys.executable,'-c',code,str(tmp_path),json.dumps(libraries)],
                             cwd=Path(__file__).resolve().parents[1],capture_output=True,
                             text=True,timeout=30)
    assert completed.returncode==0, completed.stdout+completed.stderr
    assert 'outstanding full queue' in completed.stdout


def test_cancel_while_waiting_poll_and_global_gpu_lock(tmp_path,libraries):
    with ExitStack() as s:
        d,(r,e,_),_=small_setup(s,tmp_path,libraries)
        sampler=s.enter_context(e.sampler(7,seed=1))
        with NativeGeneration.start(d,sampler,1,4) as g:
            await_state(g,lambda x,q,done:x.in_flight)
            result=[]
            waiter=threading.Thread(target=lambda:result.append(g.wait(5000)))
            waiter.start()
            # Cancellation deliberately bypasses both Python Runtime._lock and
            # native runtime mutex; a queued event or in-flight work may win.
            start=time.monotonic(); g.cancel()
            assert time.monotonic()-start<.25
            waiter.join(10)
            assert not waiter.is_alive()
            # A bounded poll may time out while already-running GPU work drains,
            # especially under memcheck. Cancellation does not interrupt kernels.
            if result[0] is None or result[0].kind!=2:
                _,final=collect(g)
            else:
                final=result[0]
            assert final.termination=='cancelled'
            assert final.sampled==final.position
