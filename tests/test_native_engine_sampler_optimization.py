"""WP8 performance-only regression gates. Original 47 sampler tests are untouched."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import NativeError, Runtime
from test_native_engine_hc import upload
from test_native_engine_sampler import reference
from native_sampler_optimization_probe import V, capture, snapshot_input

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT/'docs/native_engine/WP8_SAMPLER_OPTIMIZATION_BEFORE.json'


@pytest.mark.parametrize('k', [1,50,500])
@pytest.mark.parametrize('p', [.9,1.])
def test_bitwise_before_snapshot(tmp_path,k,p):
    before=json.loads(BEFORE.read_text())
    expected=next(row for row in before['snapshots'] if row['k']==k and row['top_p']==p)
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=32*2**20))
        u=upload(s,tmp_path,r,e,[snapshot_input()])[0][0]
        sampler=s.enter_context(e.sampler(V,seed=8))
        # Hashes cover every element, including the V-k padding, bit-for-bit.
        assert capture(sampler,u,k,p)==expected
        ids,weights,cdf=sampler.readback_for_test()
        assert np.all(ids[k:]==-1)
        assert np.all(weights[k:].view(np.uint32)==0)
        assert np.all(cdf[k:].view(np.uint32)==np.float32(1).view(np.uint32))
        assert np.all(cdf[1:]>=cdf[:-1])


@pytest.mark.parametrize('k', [50,V])
@pytest.mark.parametrize('p', [.9,1.])
def test_random_cdf_bitwise_before_snapshot(tmp_path,k,p):
    evidence=json.loads((ROOT/'docs/native_engine/WP8_SAMPLER_OPTIMIZATION_BITWISE.json').read_text())
    assert evidence['bitwise_identical']
    expected=next(row for row in evidence['runs'][0]['snapshots'] if row['k']==k and row['top_p']==p)
    x=np.random.default_rng(8).normal(size=V).astype(np.float32)
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=32*2**20))
        u=upload(s,tmp_path,r,e,[x])[0][0]
        sampler=s.enter_context(e.sampler(V,seed=8))
        actual=capture(sampler,u,k,p)
        for field in ('active_ids','active_probability_bits','active_cdf_bits'):
            actual[field]=actual[field][:64]
        assert actual==expected


@pytest.mark.parametrize('k', [1,50,500])
def test_full_vocab_order_cdf_and_seed_errors(tmp_path,k):
    x=snapshot_input()
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=32*2**20))
        bad=x.copy(); bad[-1]=np.nan  # outside selected top-k, still must validate
        huge=x.copy(); huge[0]=np.finfo(np.float32).max
        good_u,bad_u,huge_u=upload(s,tmp_path,r,e,[x,bad,huge])[0]
        sampler=s.enter_context(e.sampler(V,seed=18446744073709551615))
        allocated=e.stats.device_bytes
        for p in (.9,1.,1.-1e-10):
            ids,weights,cdf,raw,total=reference(x,.7,p,k)
            for uniform in (0.,.213,.567,.999999999999):
                result=sampler.sample(good_u,temperature=.7,top_k=k,top_p=p,uniform=uniform,metadata=True)
                actual=sampler.readback_for_test()
                np.testing.assert_array_equal(actual[0],ids)
                np.testing.assert_allclose(actual[1],weights,rtol=2e-6,atol=2e-7)
                np.testing.assert_allclose(actual[2],cdf,rtol=2e-6,atol=2e-7)
                slot=np.searchsorted(np.cumsum(raw[:k],dtype=np.float32),uniform*float(total),side='right')
                assert result.token==ids[min(slot,k-1)]
        def draw():
            return bytes(sampler.sample(good_u,temperature=.7,top_k=k,top_p=.9,metadata=True))
        sampler.reset()
        expected=[draw() for _ in range(4)]  # counter arithmetic also wraps seed+counter
        sampler.reset()
        for t in (0.,.7):
            with pytest.raises(NativeError,match='nonfinite'):
                sampler.sample(bad_u,temperature=t,top_k=k,top_p=.9)
            with pytest.raises(NativeError,match='unavailable'): sampler.readback_for_test()
        with pytest.raises(NativeError,match='overflow'):
            sampler.sample(huge_u,temperature=1e-20,top_k=k,top_p=.9)
        for uniform in (-1.,1.,np.nan):
            with pytest.raises(NativeError):
                sampler.sample(good_u,temperature=.7,top_k=k,uniform=uniform)
        assert sampler.sample(good_u)==0  # lower-ID tie in the parallel argmax
        sampler.sample(good_u,temperature=.7,top_k=k,uniform=.3)
        assert [draw() for _ in range(4)]==expected
        sampler.reset()
        assert [draw() for _ in range(4)]==expected
        assert e.stats.device_bytes==allocated
        read=bytearray(x.nbytes); good_u.readback_for_test(0,x.nbytes,read)
        assert bytes(read)==x.tobytes()


@pytest.mark.parametrize('value', [np.float32(0), np.float32(-0.), np.finfo(np.float32).min])
def test_argmax_all_equal_and_tail_invalid(tmp_path,value):
    x=np.full(V,value,np.float32)
    with ExitStack() as s:
        r=Runtime(); e=s.enter_context(r.engine(budget_bytes=32*2**20))
        bad=x.copy(); bad[-1]=np.inf
        good_u,bad_u=upload(s,tmp_path,r,e,[x,bad])[0]
        sampler=s.enter_context(e.sampler(V,seed=9))
        for repeat in range(2):
            result=sampler.sample(good_u,metadata=True)
            assert result.token==0 and result.kept==1 and result.probability==1
            assert result.cdf_low==0 and result.cdf_high==1
            with pytest.raises(NativeError,match='nonfinite'): sampler.sample(bad_u)
            assert sampler.sample(good_u)==0
            sampler.reset()


def test_original_test_file_unchanged():
    before=json.loads(BEFORE.read_text())
    name='tests/test_native_engine_sampler.py'
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==before['source_sha256'][name]
