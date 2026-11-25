"""WP6 populated-layer gate, separate from the still-open WP7 strict-logit gate."""
from contextlib import ExitStack
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from test_native_engine_qsa import inputs
from test_native_engine_qsa_layer import synthetic, hidden, read
from test_native_engine_qsa_paged import FORMATS, paged, assert_raw_dependencies
from native_qsa_full_capacity import run_case, range_read

pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('fmt',FORMATS)
def test_prepared_chunk_transport_and_bounded_diagnostics(tmp_path,fmt):
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=256*2**20))
        tensors,_=synthetic(stack,tmp_path,r,e,'strata')
        whole=stack.enter_context(paged(e,tensors,fmt=fmt,page=3))
        chunks=stack.enter_context(paged(e,tensors,fmt=fmt,page=3))
        rng=np.random.default_rng(65536)
        source=dict(raw=rng.normal(size=(8,128)).astype(np.float32),
                    k=rng.normal(size=(8,512)).astype(np.float32),
                    v=rng.normal(size=(8,512)).astype(np.float32),
                    p=rng.normal(size=(2,128)).astype(np.float32))
        source_dir=tmp_path/'source'; source_dir.mkdir()
        batch_dir=tmp_path/'batch'; batch_dir.mkdir()
        u=stack.enter_context(inputs(source_dir,r,e,source))
        whole.import_reference_state(8,*u)
        for start,end in [(0,3),(3,8)]:
            batch={k:a[start:end] for k,a in source.items() if k!='p'}
            first,last=start//4,end//4
            if last>first: batch['p']=source['p'][first:last]
            with inputs(batch_dir,r,e,batch) as batch_upload:
                chunks.import_reference_chunk(start,*batch_upload[:3],batch_upload[3] if len(batch_upload)>3 else None)
        for which in ('raw','keys','values','pooled'):
            rows=2 if which=='pooled' else 8
            np.testing.assert_array_equal(range_read(chunks,which,0,rows),range_read(whole,which,0,rows))
        assert_raw_dependencies(chunks,source['raw'])
        for start,rows in [(8,1),(7,2)]:
            with pytest.raises(NativeError,match='bounds'): range_read(chunks,'keys',start,rows)
        with pytest.raises(NativeError,match='exact position'): chunks.import_reference_chunk(0,*u)
        bad={k:a[:3].copy() for k,a in source.items() if k!='p'}; bad['raw'][0,0]=np.nan
        with inputs(batch_dir,r,e,bad) as bad_upload:
            with pytest.raises(NativeError,match='nonfinite'): chunks.import_reference_chunk(8,*bad_upload)
        assert chunks.position==8
        oversized=stack.enter_context(inputs(tmp_path,r,e,dict(raw=np.zeros((4097,128),np.float32))))[0]
        with pytest.raises(NativeError,match='4096'): chunks.import_reference_chunk(8,oversized,u[1],u[2])
        hidden_dir=tmp_path/'hidden'; hidden_dir.mkdir()
        h=stack.enter_context(inputs(hidden_dir,r,e,dict(h=hidden(8))))[0]
        assert whole.step(h,8)==chunks.step(h,8)==9
        np.testing.assert_array_equal(read(chunks,'output',(2560,)),read(whole,'output',(2560,)))
        np.testing.assert_array_equal(range_read(chunks,'keys',7,2),range_read(whole,'keys',7,2))
        chunks.reset()
        assert chunks.position==0
        assert range_read(chunks,'keys',0,0).size==0
        mono=stack.enter_context(e.qsa_layer(4096,*tensors,policy='strata'))
        with pytest.raises(NativeError,match='paged QSA chunk'): mono.import_reference_chunk(0,*u)


@pytest.mark.skipif(os.environ.get('MINI_LLM_WP6_FULL_CAPACITY')!='1',reason='explicit bounded populated checkpoint layer acceptance')
@pytest.mark.parametrize('policy',['ggml_cpu','strata'])
@pytest.mark.parametrize('fmt',FORMATS)
def test_actual_populated_checkpoint_layer(tmp_path,fmt,policy):
    capacity=int(os.environ.get('MINI_LLM_WP6_TEST_CAPACITY','262144'))
    assert 64<=capacity<=262144
    output=Path(os.environ.get('MINI_LLM_WP6_FULL_OUTPUT',str(tmp_path)))
    assert output.is_dir()
    result=run_case(tmp_path,fmt,policy,capacity,output)
    assert result['status']=='passed' and result['overflow_rejected_without_state_change']
    print('WP6_FULL_CAPACITY_RESULT='+str(output/(fmt+'_'+policy+'.result.json')))
