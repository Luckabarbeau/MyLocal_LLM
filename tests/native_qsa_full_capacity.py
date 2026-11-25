"""Bounded CPU prepared-snapshot oracle for WP6; never a full-model trajectory."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from mini_llm.native import Runtime, NativeError
from mini_llm.external.qwen4exp_attention import select_cells
from test_native_engine_qsa import inputs
from test_native_engine_qsa_layer import hidden, read
from test_native_engine_qsa_paged import checkpoint_tensors, FORMATS
from test_native_engine_kv_streamed import quant_reference

UUID = 'GPU-e961516f-d77d-813c-98e9-2f5403f71631'
GPU_BUDGET = 256*2**20
HOST_BUDGET = {FORMATS[0]: 1_200_000_000, FORMATS[1]: 350_000_000}
SOURCE_FILES = ['mini_llm/native_cpp/qsa_layer_runtime.inc',
                'mini_llm/native_cpp/qsa_paged_runtime.inc',
                'mini_llm/native_cpp/qsa_paged_import_runtime.inc',
                'mini_llm/native_cpp/qsa_projection.cu',
                'mini_llm/external/qwen4exp_attention.py',
                'tests/native_qsa_full_capacity.py', 'tests/test_native_engine_qsa_full_capacity.py']


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def snapshot():
    import cupy as cp
    free,total = cp.cuda.runtime.memGetInfo()
    status = {k:v.strip() for k,v in (line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines())
              if k in ('VmRSS','VmHWM','RssAnon','RssFile')}
    info = subprocess.run(['nvidia-smi','--id='+UUID,'--query-compute-apps=pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    own = [line.strip() for line in info.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    return dict(time_ns=time.time_ns(),pid=os.getpid(),cuda_free_bytes=int(free),cuda_total_bytes=int(total),
                rss=status,ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                own_gpu_process_rows=own)


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('MINI_LLM_BACKEND')!='numpy':
        raise RuntimeError('WP6 full capacity requires the exact GPU1 UUID mask and CPU NumPy reference')
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount()==1
    props=cp.cuda.runtime.getDeviceProperties(0)
    assert props['uuid'].hex()=='e961516fd77d813c98e92f5403f71631'
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    state=snapshot()
    # Leave room for the existing training owner; never stop it or reserve its free memory.
    if state['cuda_free_bytes'] < GPU_BUDGET+1536*2**20:
        raise RuntimeError('insufficient GPU1 headroom for bounded WP6 acceptance; training is untouched')
    if available < 15*2**30:
        raise RuntimeError('insufficient host headroom: require 12GiB reserve plus bounded fixture/backing')
    return dict(memory=state,host_mem_available_bytes=available,visible_name=props['name'].decode(),
                uuid=UUID,gpu_budget_bytes=GPU_BUDGET,minimum_gpu_headroom_bytes=GPU_BUDGET+1536*2**20,
                minimum_host_available_bytes=15*2**30)


def range_read(layer,which,start,rows):
    width=128 if which in ('raw','pooled') else 512
    out=np.empty((rows,width),np.float32)
    layer.readback_range_for_test(which,start,rows,out)
    return out


def stats(layer):
    s=layer.paged_stats
    return {name:int(getattr(s,name)) for name,_ in s._fields_}


def cpu_front(ref,x,n,partial):
    qg=ref._project('query_gate',x).reshape(24,512)
    raw=ref._project('index_key',x)
    q=ref._rotate(ref.norms[0].forward(qg[:,:256]),n-1)
    k=ref._rotate(ref.norms[1].forward(ref._project('key',x).reshape(2,256)),n-1)
    v=ref._project('value',x).reshape(2,256)
    iq=ref._rotate(ref.norms[2].forward(ref._project('index_query',x).reshape(4,128)),n-1)
    pooled=partial[0].copy(); pooled+=partial[1]; pooled+=partial[2]; pooled+=raw; pooled*=np.float32(.25)
    pool=ref._rotate(ref.norms[3].forward(pooled[None]),n-4)[0]
    return dict(q=q,k=k,v=v,gate=qg[:,256:],iq=iq,pool=pool,raw=raw)


def frontend_bound(expected,position):
    # Existing WP3 frontend/complete-layer norm + long-angle envelope.
    bound=5e-5*(1+np.abs(expected))
    f=np.power(np.float32(1e7),-np.arange(32,dtype=np.float32)/32)
    angle=8*np.finfo(np.float32).eps*np.float32(position)*f*(np.abs(expected[...,:32])+np.abs(expected[...,32:64]))
    bound[...,:32]+=angle; bound[...,32:64]+=angle
    return bound


class Prepared:
    """Explicit post-projection snapshot: exact dyadic K/V, raw one-hot windows.

    Historical K rotary prefixes are zero, keeping independent CPU/GPU long-angle
    error from masquerading as an attention/storage defect. Raw/pools are consistent.
    This is not a claim that historical K/V came from checkpoint hidden projections.
    """
    def __init__(self,ref,milestones,capacity):
        self.ref=ref; self.milestones=milestones
        self.window=min(4096,capacity//4)
        self.windows=[(m-self.window,m) for m in milestones]
        queries=[]
        for m in milestones:
            x=hidden(m-1)
            queries.append(ref._rotate(ref.norms[2].forward(ref._project('index_query',x).reshape(4,128)),m-1))
        candidates=[]
        for d in range(64,128):
            raw=np.zeros(128,np.float32); raw[d]=1
            pool=ref.norms[3].forward(raw[None])[0]
            minimum=min(float(np.maximum(iq@pool,0).sum(dtype=np.float32)) for iq in queries)
            candidates.append((minimum,d,pool))
        minimum,self.axis,self.basis_pool=max(candidates,key=lambda a:a[0])
        assert minimum>1e-3,'prepared tied window must have a positive score separated from zero'
        self.minimum_positive_score=minimum
        self.cpu_overrides={}; self.native_overrides={}; self.cpu_pools={}; self.native_pools={}

    def active(self,positions):
        result=np.zeros(len(positions),bool)
        for lo,hi in self.windows: result |= (positions>=lo)&(positions<hi)
        return result

    def chunk(self,start,rows):
        positions=np.arange(start,start+rows,dtype=np.int64)
        d=np.arange(512,dtype=np.int64)[None]
        k=(((positions[:,None]%251)*17+d*13)%257-128).astype(np.float32)*np.float32(1/512)
        k.reshape(rows,2,256)[...,:64]=0
        v=(((positions[:,None]%509)*7+d*19)%511-255).astype(np.float32)*np.float32(1/1024)
        raw=np.zeros((rows,128),np.float32); raw[self.active(positions),self.axis]=1
        first,last=start//4,(start+rows)//4
        pools=np.zeros((last-first,128),np.float32)
        block_positions=np.arange(first,last,dtype=np.int64)*4
        pools[self.active(block_positions)]=self.basis_pool
        return dict(raw=raw,k=k,v=v,p=pools)

    def history(self,ids,which,quantized=False):
        ids=np.asarray(ids,dtype=np.int64)
        d=np.arange(512,dtype=np.int64)[None]
        k=(((ids[:,None]%251)*17+d*13)%257-128).astype(np.float32)*np.float32(1/512)
        k.reshape(-1,2,256)[...,:64]=0
        v=(((ids[:,None]%509)*7+d*19)%511-255).astype(np.float32)*np.float32(1/1024)
        overrides=self.native_overrides if which=='native' else self.cpu_overrides
        for token,(ok,ov) in overrides.items():
            hit=ids==token
            if np.any(hit): k[hit]=ok.reshape(512); v[hit]=ov.reshape(512)
        if quantized:
            _,decoded,_=quant_reference(k,v)
            k,v=decoded[:,0],decoded[:,1]
        return k.reshape(-1,2,256),v.reshape(-1,2,256)

    def scores(self,iq,n,which,current_pool):
        block=np.zeros(n//4,np.float32)
        dot=np.maximum(iq@self.basis_pool,0)
        common=dot[0].copy()
        for h in range(1,4): common+=dot[h]
        block[self.active(np.arange(n//4)*4)]=common
        pools=dict(self.native_pools if which=='native' else self.cpu_pools)
        pools[n//4-1]=current_pool
        for b,pool in pools.items():
            dots=np.maximum(iq@pool,0); score=dots[0].copy()
            for h in range(1,4): score+=dots[h]
            block[b]=score
        return np.repeat(block,4),float(common)


def attention(ref,q,gate,k,v):
    context=np.empty((24,256),np.float32)
    for h in range(24):
        logits=(k[:,h//12]@q[h])*np.float32(1/16)
        p=np.exp(logits-logits.max()); p/=p.sum(dtype=np.float32)
        context[h]=p@v[:,h//12]
    z=np.exp(-np.abs(gate)); sigmoid=np.where(gate>=0,1/(1+z),z/(1+z))
    return ref._project('output',(context*sigmoid).reshape(-1))


def run_case(tmp_path,fmt,policy,capacity,output):
    pre=preflight()
    milestones=[capacity//4,capacity//2,capacity]
    assert capacity%16==0 and capacity>=64
    label=fmt+'_'+policy
    prefix=output/label
    config=dict(schema='WP6-full-capacity-v1',format=fmt,tie_policy=policy,capacity=capacity,
                milestones=milestones,import_chunk_max_rows=4096,page_size=min(1024,capacity//16),
                physical_pages=1,host_budget_bytes=HOST_BUDGET[fmt],gpu_budget_bytes=GPU_BUDGET,
                raw_diagnostics=False,reference_backend='numpy',preflight=pre,
                argv=sys.argv,environment={k:v for k,v in os.environ.items() if k.startswith(('MINI_LLM_','CUDA_VISIBLE','OPENBLAS','NVIDIA_TF32'))},
                sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES},
                native_library_sha256=hashlib.sha256(Path(os.environ['MINI_LLM_NATIVE_LIBRARY']).read_bytes()).hexdigest(),
                strict_wp7_status='OPEN: documented strict whole-model failure, not tested or suppressed here')
    with Path(str(prefix)+'.config.json').open('x') as f: json.dump(config,f,indent=2,sort_keys=True)
    records=[]; hashes={k:hashlib.sha256() for k in ('raw','k','v','p')}
    events=Path(str(prefix)+'.events.jsonl').open('x',buffering=1)
    def emit(kind,**data): events.write(json.dumps(dict(kind=kind,**data),sort_keys=True)+'\n')
    result=dict(config_file=str(prefix)+'.config.json',status='running',records=records)
    try:
        inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
        shard_paths=[Path(s['path']) for s in inventory['shards']]
        identities={str(p):[p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns] for p in shard_paths}
        result['checkpoint_identities_before']=identities
        with ExitStack() as stack:
            r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=GPU_BUDGET))
            store=stack.enter_context(r.gguf(inventory['shards'][0]['path']))
            tensors,ref=checkpoint_tensors(stack,r,e,store)
            fixture=Prepared(ref,milestones,capacity)
            result['fixture']=dict(version='dyadic-suffix-KV-onehot-index-windows-v1',axis=fixture.axis,
                                   windows=fixture.windows,minimum_positive_score=fixture.minimum_positive_score,
                                   historical_KV_is_prepared_snapshot_not_checkpoint_trajectory=True)
            weights={}
            from test_native_engine_qsa_hidden_frontend import STEMS,NORMS
            for stem in STEMS+['attn_output.weight']+NORMS:
                lease=stack.enter_context(store.lease('blk.3.'+stem)); d=lease.descriptor
                weights[stem]=dict(type_id=d.type_id,ggml_shape=d.ggml_shape,nbytes=d.nbytes,
                                   sha256=hashlib.sha256(lease.read(0,d.nbytes)).hexdigest())
            result['checkpoint_layer3_weights']=weights
            page=stack.enter_context(e.qsa_layer(capacity,*tensors,policy=policy,kv_policy=fmt,
                page_size=config['page_size'],physical_pages=1,host_budget_bytes=HOST_BUDGET[fmt],raw_diagnostics=False))
            front=stack.enter_context(e.qsa_frontend())
            # These source directories are separate from transient chunk uploads.
            current_dir=tmp_path/'current'; current_dir.mkdir()
            batch_dir=tmp_path/'chunks'; batch_dir.mkdir()
            partial=np.zeros((3,128),np.float32); partial[:,fixture.axis]=1
            current=stack.enter_context(inputs(current_dir,r,e,dict(h=hidden(0),first=np.zeros(128,np.float32),partial=partial)))
            # Hidden uploads for all milestone steps are small and immutable.
            hidden_dir=tmp_path/'hidden'; hidden_dir.mkdir()
            hidden_uploads=stack.enter_context(inputs(hidden_dir,r,e,{f'h{n}':hidden(n-1) for n in milestones}))
            position=0
            for n,hidden_upload in zip(milestones,hidden_uploads):
                while position<n-1:
                    count=min(4096,n-1-position)
                    source=fixture.chunk(position,count)
                    for key,array in source.items(): hashes[key].update(memoryview(array).cast('B'))
                    present={k:v for k,v in source.items() if k!='p' or len(v)}
                    with inputs(batch_dir,r,e,present) as uploaded:
                        page.import_reference_chunk(position,*uploaded[:3],uploaded[3] if len(uploaded)>3 else None)
                    expected_k,expected_v=source['k'],source['v']
                    if fmt==FORMATS[1]:
                        _,decoded,_=quant_reference(expected_k,expected_v)
                        expected_k,expected_v=decoded[:,0],decoded[:,1]
                    np.testing.assert_array_equal(range_read(page,'keys',position,count),expected_k)
                    np.testing.assert_array_equal(range_read(page,'values',position,count),expected_v)
                    position+=count
                    emit('import_chunk',start=position-count,rows=count,source_hashes={k:digest(v) for k,v in source.items()},
                         position=position,stats=stats(page))
                assert page.position==n-1
                before_pool=range_read(page,'pooled',0,(n-1)//4)
                before_hash=digest(before_pool)
                before=stats(page)
                cpu=cpu_front(ref,hidden(n-1),n,partial)
                # Standalone validated frontend is a factorization oracle for exact
                # post-quantization commit bytes; independent CPU math is checked too.
                front.from_hidden(hidden_upload,*tensors[:5],*tensors[6:],current[1],current[2],n)
                native={k:np.empty(shape,np.float32) for k,shape in
                        [('q',(24,256)),('k',(2,256)),('gate',(24,256)),('iq',(4,128)),('dead',(128,)),('pool',(128,))]}
                actual_n,ready=front.readback_for_test(*native.values()); assert actual_n==n and ready
                qg0=np.empty((24,512),np.float32); k0=np.empty((2,256),np.float32)
                v0=np.empty_like(k0); raw0=np.empty(128,np.float32); iq0=np.empty((4,128),np.float32)
                front.hidden_readback_for_test(qg0,k0,v0,raw0,iq0)
                native['v']=v0; native['raw']=raw0
                for key,pos in [('q',n-1),('k',n-1),('iq',n-1),('pool',n-4)]:
                    assert np.all(np.abs(native[key]-cpu[key])<=frontend_bound(cpu[key],pos)), key
                np.testing.assert_allclose(native['gate'],cpu['gate'],rtol=2e-4,atol=2e-4)
                np.testing.assert_allclose(native['v'],cpu['v'],rtol=2e-4,atol=2e-4)
                cpu_scores,common=fixture.scores(cpu['iq'],n,'cpu',cpu['pool'])
                expected_ids=np.asarray(select_cells(cpu_scores,min(n,2051),policy),dtype=np.int64)
                assert page.step(hidden_upload,n-1)==n
                fixture.cpu_overrides[n-1]=(cpu['k'].copy(),cpu['v'].copy())
                fixture.native_overrides[n-1]=(native['k'].copy(),native['v'].copy())
                fixture.cpu_pools[n//4-1]=cpu['pool'].copy(); fixture.native_pools[n//4-1]=native['pool'].copy()
                ids=read(page,'ids',(min(n,2051),),np.int64)
                np.testing.assert_array_equal(ids,expected_ids)
                if capacity==262144: assert ids.min()>0  # No compulsory historical token0 load.
                actual=read(page,'output',(2560,))
                ck,cv=fixture.history(ids,'cpu',False)
                independent_fp32=attention(ref,cpu['q'],cpu['gate'],ck,cv)
                nk,nv=fixture.history(ids,'native',False)
                factored_fp32=attention(ref,native['q'],native['gate'],nk,nv)
                if fmt==FORMATS[0]:
                    expected=independent_fp32
                    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
                else:
                    qk,qv=fixture.history(ids,'native',True)
                    expected=attention(ref,native['q'],native['gate'],qk,qv)
                    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=3e-6)
                last_k,last_v=native['k'].reshape(1,512),native['v'].reshape(1,512)
                if fmt==FORMATS[1]:
                    _,last,_=quant_reference(last_k,last_v); last_k,last_v=last[:,0],last[:,1]
                np.testing.assert_array_equal(range_read(page,'keys',n-1,1),last_k)
                np.testing.assert_array_equal(range_read(page,'values',n-1,1),last_v)
                after_pool=range_read(page,'pooled',0,n//4)
                assert digest(after_pool[:-1])==before_hash
                full_pool_hash=digest(after_pool)
                prepared_pools=fixture.chunk(0,n)['p'] if n<=4096 else None
                # Reconstruct only the index table, never full reference raw/K/V.
                if prepared_pools is None:
                    prepared_pools=np.zeros_like(after_pool)
                    prepared_pools[fixture.active(np.arange(n//4)*4)]=fixture.basis_pool
                for block,value in fixture.native_pools.items(): prepared_pools[block]=value
                np.testing.assert_array_equal(after_pool,prepared_pools)
                after=stats(page)
                requested_pages=np.unique(ids[ids<n-1]//config['page_size'])
                assert after['hits']-before['hits']+after['misses']-before['misses']==len(requested_pages)
                assert after['resident_pages']<=1 and after['pending']==0
                assert after['populated_bytes']==n*(4096 if fmt==FORMATS[0] else 1056)
                memory=snapshot()
                assert e.stats.peak_bytes<GPU_BUDGET and memory['cuda_free_bytes']>512*2**20
                record=dict(n=n,position_before=n-1,exact_selection=True,selected_ids_sha256=digest(ids),
                    selected_min=int(ids.min()),selected_max=int(ids.max()),distinct_selected_history_pages=len(requested_pages),
                    ids=ids.tolist(),cpu_common_tie_score=common,
                    score_max_abs=float(np.max(np.abs(read(page,'scores',(n,))-cpu_scores))),
                    output_max_abs=float(np.max(np.abs(actual-expected))),
                    independent_fp32_output_max_abs=float(np.max(np.abs(actual-independent_fp32))),
                    factored_fp32_policy_difference_max_abs=float(np.max(np.abs(actual-factored_fp32))),
                    full_pooled_sha256=full_pool_hash,unchanged_pooled_prefix_sha256=before_hash,
                    committed_last_key_sha256=digest(last_k),committed_last_value_sha256=digest(last_v),
                    stats=after,delta_cache={key:after[key]-before[key] for key in ('hits','misses','h2d_bytes','metadata_bytes')},
                    engine_requested_peak_bytes=e.stats.peak_bytes,memory=memory,
                    source_cumulative_sha256={k:v.hexdigest() for k,v in hashes.items()})
                records.append(record); emit('milestone',**record)
                position=n
            saved_pool=records[-1]['full_pooled_sha256']; saved_output=digest(read(page,'output',(2560,)))
            saved_stats=stats(page); saved_last=range_read(page,'keys',capacity-1,1)
            try: page.step(hidden_uploads[-1],capacity)
            except NativeError as error:
                assert 'capacity' in str(error); result['overflow_error']=str(error)
            else: raise AssertionError('step beyond maximum capacity was accepted')
            assert page.position==capacity and stats(page)==saved_stats
            assert digest(read(page,'output',(2560,)))==saved_output
            assert digest(range_read(page,'pooled',0,capacity//4))==saved_pool
            np.testing.assert_array_equal(range_read(page,'keys',capacity-1,1),saved_last)
            result['overflow_rejected_without_state_change']=True
            result['final_memory']=snapshot()
        result['checkpoint_identities_after']={str(p):[p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns] for p in shard_paths}
        assert result['checkpoint_identities_after']==identities
        result['status']='passed'
    except BaseException:
        result['status']='failed'; result['traceback']=traceback.format_exc(); raise
    finally:
        events.close()
        with Path(str(prefix)+'.result.json').open('x') as f: json.dump(result,f,indent=2,sort_keys=True)
    return result
