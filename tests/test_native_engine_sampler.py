"""Independent WP8 gates: GPU results vs Python FP32 probabilities/CDF, not RNG identity."""
from contextlib import ExitStack
import ctypes as C
import json
import time

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError, SampleResult
from test_native_engine_hc import upload

pytestmark = pytest.mark.gpu


def reference(x, temperature, top_p, top_k):
    """sample_token FP32 arithmetic with installed CuPy stable sorting semantics."""
    x = np.asarray(x, np.float32).reshape(-1)
    k = min(len(x), len(x) if top_k is None else top_k)
    ids = (np.arange(len(x)) if k == len(x) else
           np.argsort(x, kind='stable')[-k:])
    values = x[ids] / np.float32(temperature)
    if top_p < 1:
        order = np.argsort(values, kind='stable')[::-1]
        ids, values = ids[order], values[order]
    weights = np.exp(values - values.max()).astype(np.float32)
    if top_p < 1:
        cumulative = np.cumsum(weights, dtype=np.float32)
        weights = np.where(cumulative - weights < np.float32(top_p)*cumulative[-1],
                           weights, np.float32(0))
    # Diagnostics pad unused capacity only; this is not additional candidate mass.
    all_weights = np.zeros(len(x), np.float32); all_weights[:k] = weights
    all_ids = np.full(len(x), -1, np.int32); all_ids[:k] = ids
    cdf = np.cumsum(all_weights, dtype=np.float32)
    total = cdf[-1]
    return all_ids, all_weights/total, cdf/total, all_weights, total


@pytest.mark.parametrize('n', [1, 9, 513, 248320])
@pytest.mark.parametrize('temperature,top_p,top_k', [
    (1., 1., None), (.7, .87, 37), (1e-20, .99, 4),
    (1e20, 1., 17), (2., .5, None), (.4, .95, 1), (1., 1.-1e-10, None), (1., 1., 4), (1e20, .8, None)])
def test_probabilities_cdf_uniform(tmp_path, n, temperature, top_p, top_k):
    x = np.random.default_rng(n).normal(0, 1, n).astype(np.float32)
    if n > 1:
        x[:min(n, 10)] = 2  # ties include the top-k boundary
    ids, probs, cdf, raw, total = reference(x, temperature, top_p, top_k)
    with ExitStack() as s:
        r = Runtime(); e = s.enter_context(r.engine(budget_bytes=32*2**20))
        u = upload(s, tmp_path, r, e, [x])[0][0]
        sampler = s.enter_context(e.sampler(n, seed=123))
        baseline = e.stats.device_bytes
        for uniform in (0., .1234567, .5, .999999999999):
            result = sampler.sample(u, temperature=temperature, top_p=top_p,
                                    top_k=top_k, uniform=uniform, metadata=True)
            got_ids, got_probs, got_cdf = sampler.readback_for_test()
            np.testing.assert_array_equal(got_ids, ids)
            np.testing.assert_allclose(got_probs, probs, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(got_cdf, cdf, rtol=2e-6, atol=2e-7)
            slot = min(np.searchsorted(np.cumsum(raw, dtype=np.float32),
                                       uniform*float(total), side='right'), n-1)
            assert result.token == ids[slot]
            assert result.status == 0
            assert result.kept == np.count_nonzero(raw)
        assert e.stats.device_bytes == baseline
        read = bytearray(x.nbytes); u.readback_for_test(0, x.nbytes, read)
        assert bytes(read) == x.tobytes()


def test_greedy_ties_and_right_boundary(tmp_path):
    with ExitStack() as s:
        r = Runtime(); e = s.enter_context(r.engine())
        u = upload(s,tmp_path,r,e,[np.array([-0.,0.,-1.,0.],np.float32)])[0][0]
        sampler = s.enter_context(e.sampler(4,seed=0))
        assert sampler.sample(u,temperature=0,top_k=0) == 0
        for uniform, expected in [(0.,0),(1/3,1),(2/3,3)]:
            assert sampler.sample(u,temperature=1e-20,uniform=uniform) == expected
        # Exact binary CDF boundary: search side='right' skips the first token.
        equal = upload(s,tmp_path,r,e,[np.zeros(4,np.float32)])[0][0]
        assert sampler.sample(equal,temperature=1,uniform=.25) == 1


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_and_recovery(tmp_path,bad):
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine())
        good,bad_u=upload(s,tmp_path,r,e,[np.ones(9,np.float32),np.array([bad]+[0.]*8,np.float32)])[0]
        sampler=s.enter_context(e.sampler(9,seed=0))
        for temp in (0,1):
            with pytest.raises(NativeError,match='nonfinite'):
                sampler.sample(bad_u,temperature=temp)
            with pytest.raises(NativeError,match='unavailable'):
                sampler.readback_for_test()
            assert sampler.sample(good) == 0
        huge=upload(s,tmp_path,r,e,[np.full(9,np.finfo(np.float32).max,np.float32)])[0][0]
        with pytest.raises(NativeError,match='overflow'):
            sampler.sample(huge,temperature=1e-20)
        assert sampler.sample(good)==0


def test_budget_validation_lifetime_reset(tmp_path):
    r=Runtime()
    with r.engine(budget_bytes=12) as e:
        with pytest.raises(NativeError,match='budget'): e.sampler(9,seed=0)
        assert e.stats.device_bytes==0
    with ExitStack() as s:
        e=s.enter_context(r.engine(budget_bytes=32*2**20)); other=s.enter_context(r.engine())
        u=upload(s,tmp_path,r,e,[np.zeros(9,np.float32)])[0][0]
        wrong=upload(s,tmp_path,r,other,[np.zeros(9,np.float32)])[0][0]
        base=e.stats.device_bytes
        sampler=s.enter_context(e.sampler(9,seed=987))
        allocated=e.stats.device_bytes
        assert allocated>base+24*9
        for kwargs in ({'temperature':-1},{'temperature':np.nan},{'temperature':np.inf},
                               {'temperature':1e-50},{'temperature':1e40},
                       {'top_p':0},{'top_p':1.1},{'top_p':np.nan},{'uniform':-1},
                       {'uniform':1},{'uniform':np.nan}):
            with pytest.raises(NativeError): sampler.sample(u,**kwargs)
        for k in (0,-1,True,1.5):
            with pytest.raises(ValueError): sampler.sample(u,temperature=1,top_k=k)
        with pytest.raises(NativeError,match='same-engine'): sampler.sample(wrong)
        wrong_shape=upload(s,tmp_path,r,e,[np.zeros((3,3),np.float32)])[0][0]
        with pytest.raises(NativeError,match='vector'): sampler.sample(wrong_shape)
        wrong_type=upload(s,tmp_path,r,e,[np.zeros(9,np.float32)],kinds=[1])[0][0]
        with pytest.raises(NativeError,match='F32'): sampler.sample(wrong_type)
        wrong_size=upload(s,tmp_path,r,e,[np.zeros(8,np.float32)])[0][0]
        with pytest.raises(NativeError): sampler.sample(wrong_size)
        for vocab in (0,248321):
            with pytest.raises((ValueError,NativeError)): e.sampler(vocab,seed=0)
        lib=r._lib; out=SampleResult()
        assert lib.ml_sampler_sample(sampler._handle,u._handle,2,1,1,0,0,0,C.byref(out))==1
        assert lib.ml_sampler_sample(sampler._handle,u._handle,0,1,1,0,0,0,None)==1
        assert lib.ml_sampler_readback_for_test(sampler._handle,None,None,None,0)==1
        assert lib.ml_sampler_sample(sampler._handle,999999,1,1,1,0,0,0,C.byref(out))==1
        # Scalar sequence is explicitly SplitMix64(seed+counter), not backend RNG.
        def sm(x):
            mask=2**64-1; x=(x+0x9e3779b97f4a7c15)&mask
            x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
            x=((x^(x>>27))*0x94d049bb133111eb)&mask
            return (x^(x>>31))>>11
        first=[sampler.sample(u,temperature=1,metadata=True).uniform for _ in range(12)]
        assert first==[sm(987+i)*2**-53 for i in range(12)]
        sampler.reset()
        sampler.sample(u,temperature=1,uniform=.5)  # diagnostic must not advance RNG
        assert first==[sampler.sample(u,temperature=1,metadata=True).uniform for _ in range(12)]
        for _ in range(4): sampler.reset(seed=987)
        e.close()  # native owner retains engine even with no public engine handle
        assert sampler.sample(u,temperature=1,metadata=True).uniform==first[0]
        old=sampler._handle; sampler.close(); sampler.close()
        assert lib.ml_sampler_reset(old,0)==1
        with pytest.raises(NativeError,match='closed'): sampler.sample(u)
        u.close()
        with pytest.raises(NativeError): r.engine(device=2147483647)


def test_release_accounting(tmp_path):
    r=Runtime()
    with r.engine(budget_bytes=32*2**20) as e:
        for n in (1,9,248320):
            with e.sampler(n,seed=0): assert e.stats.device_bytes>24*n
            assert e.stats.device_bytes==0


def test_decoder_resident(tmp_path,libraries):
    from test_native_engine_decoder import small_setup
    with ExitStack() as s:
        decoder,(r,e,_),_=small_setup(s,tmp_path,libraries)
        sampler=s.enter_context(e.sampler(7,seed=1))
        with pytest.raises(NativeError,match='valid'): sampler.sample_decoder(decoder)
        first=None
        for repeat in range(2):
            assert decoder.step_device(1)==1
            token=sampler.sample_decoder(decoder)
            if first is None: first=token
            assert token==first
            decoder.reset()
            with pytest.raises(NativeError,match='valid'): sampler.sample_decoder(decoder)
        host=decoder.step(1)
        assert sampler.sample_decoder(decoder)==int(np.argmax(host))
        with r.engine() as other, other.sampler(7,seed=0) as wrong:
            with pytest.raises(NativeError,match='same-engine'): wrong.sample_decoder(decoder)
        decoder.reset()
        with pytest.raises(NativeError): decoder.step_device(100)
        with pytest.raises(NativeError,match='valid'): sampler.sample_decoder(decoder)
        decoder.close()
        with pytest.raises(NativeError,match='closed'): sampler.sample_decoder(decoder)


def test_existing_python_reference(tmp_path, monkeypatch):
    from mini_llm import generation
    # Exercise the actual imported source, independently of the diagnostic oracle.
    # No ties here: argpartition membership and sorted-nucleus order are unique.
    monkeypatch.setattr(generation, 'xp', np)
    x=np.array([.1,2.2,-.7,1.6,3.4,-2.,.9],np.float32)
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=2**20))
        u=upload(s,tmp_path,r,e,[x])[0][0]
        sampler=s.enter_context(e.sampler(len(x),seed=0))
        for top_p,top_k in [(1.,None),(.8,None),(.8,3),(1.-1e-10,4)]:
            for uniform in (0.,.2,.51,.99):
                monkeypatch.setattr(np.random,'random',lambda:uniform)
                expected=generation.sample_token(x,temperature=.7,top_p=top_p,top_k=top_k)
                assert sampler.sample(u,temperature=.7,top_p=top_p,top_k=top_k,uniform=uniform)==expected


def test_actual_cupy_tie_order(tmp_path,monkeypatch):
    import cupy as cp
    from mini_llm import generation
    monkeypatch.setattr(generation,'xp',cp)
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=2**20))
        x=np.array([2.,2.,2.,1.,1.,0.,-1.,-2.,2.],np.float32)
        u=upload(s,tmp_path,r,e,[x])[0][0]; device=cp.asarray(x)
        sampler=s.enter_context(e.sampler(len(x),seed=0))
        for temperature in (.7,1e20,1e-20):
            for top_p in (1.,.8,1.-1e-10):
                for top_k in (None,3,7):
                    for uniform in (0.,.213,.567,.999):
                        monkeypatch.setattr(cp.random,'random',lambda:cp.asarray(uniform,cp.float64))
                        expected=generation.sample_token(device,temperature=temperature,top_p=top_p,top_k=top_k)
                        assert sampler.sample(u,temperature=temperature,top_p=top_p,top_k=top_k,uniform=uniform)==expected
        assert sampler.sample(u)==int(cp.argmax(device).item())


def test_exact_budget():
    r=Runtime()
    with r.engine(budget_bytes=32*2**20) as e:
        with e.sampler(248320,seed=0): size=e.stats.device_bytes
    for budget,ok in [(size-1,False),(size,True)]:
        with r.engine(budget_bytes=budget) as e:
            if ok:
                with e.sampler(248320,seed=0): assert e.stats.device_bytes==size
            else:
                with pytest.raises(NativeError,match='budget'): e.sampler(248320,seed=0)
                assert e.stats.device_bytes==0


def test_sampler_only_benchmark(tmp_path):
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=32*2**20))
        x=np.random.default_rng(8).normal(size=248320).astype(np.float32)
        u=upload(s,tmp_path,r,e,[x])[0][0]
        sampler=s.enter_context(e.sampler(len(x),seed=8))
        workspace=e.stats.device_bytes-x.nbytes
        results={}
        for name,kwargs in [('greedy',{}),('nucleus',dict(temperature=.7,top_p=.9,top_k=50)),
                            ('full',dict(temperature=1))]:
            sampler.sample(u,**kwargs)
            times=[]
            for _ in range(10):
                start=time.perf_counter(); sampler.sample(u,**kwargs)
                times.append((time.perf_counter()-start)*1000)
            results[name+'_median_host_ms']=float(np.median(times))
        print('WP8 sampler-only',json.dumps(dict(vocab=len(x),workspace_bytes=workspace,**results)))


# The existing selected GGML library fixture is reused, not a replacement oracle.
from test_native_engine_moe import libraries, selected_library
