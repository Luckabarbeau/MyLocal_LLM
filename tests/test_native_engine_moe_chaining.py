"""Direct native HC FFN read -> static MoE -> HC write, no Python dispatch seam."""
from contextlib import ExitStack
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError, MoE, MoEJob, CPUExperts
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, QSAConfig
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
from test_native_engine_moe import cp, libraries, selected_library, fixture, native_setup, reference as moe_reference, IDS
from test_native_engine_hc import make_weights, upload, read as hc_read
from test_native_engine_chaining import checkpoint_upload, check_branch, close, REC_STEMS, STEMS, NORMS, NAMES
from test_native_engine_recurrent import create, reference as recurrent_reference
from native_moe_checkpoint_helpers import CheckpointMoE, control, POLICY


def forbidden(*a,**k):
    pytest.fail('Python per-op/expert dispatch inside native HC MoE mix')


def native_only(patch,hc,owner,e):
    for obj,name in ((hc,'read'),(hc,'write'),(owner,'submit'),(e,'upload'),
                     (MoEJob,'readback_for_test'),(owner,'input_readback_for_test'),(CPUExperts,'submit'),(GGMLPackedCPUExecutor,'expert')):
        patch.setattr(obj,name,forbidden)


@pytest.fixture(scope='module')
def report():
    rows=[]
    yield rows
    path=os.environ.get('MINI_LLM_MOE_CHAIN_REPORT')
    if path:
        library=Path(os.environ['MINI_LLM_NATIVE_LIBRARY'])
        Path(path).write_text(json.dumps(dict(command=[sys.executable,*sys.argv],policy=POLICY,
            native_library=str(library),native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            environment={k:v for k,v in os.environ.items() if k.startswith('MINI_LLM_') or k in
                ('NVIDIA_TF32_OVERRIDE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
            cases=rows),indent=2)+'\n')


@pytest.mark.parametrize('mode',('cpu','gpu','mixed'))
def test_direct_synthetic_replay_and_lifetime(cp,libraries,tmp_path,monkeypatch,mode):
    path=tmp_path/'moe.gguf'; data=fixture(path)
    placement=() if mode=='cpu' else IDS if mode=='gpu' else IDS[::2]
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,placement)
        weights,ref,ht=make_weights(stack,tmp_path,r,e,256,5)
        hc=stack.enter_context(e.hc_state(256,5))
        xs=[data['x'].copy(),-data['x'],data['x']*.5]
        us,_=upload(stack,tmp_path,r,e,xs)
        baseline=bytes(e.stats)
        for replay in range(2):
            for t,x in enumerate(xs):
                hc.initialize(us[t]); streams=ref.initialize(cp.asarray(x))
                h,g=ref.read(streams); data['x']=cp.asnumpy(h)
                expected,_,_=moe_reference(cp,data,placement,libraries,monkeypatch)
                with monkeypatch.context() as patch:
                    native_only(patch,hc,owner,e)
                    assert hc.mix(owner,weights,t,scale=.37)==t+1
                close(hc_read(hc,'streams',256),ref.write(streams,cp.asarray(expected),g))
                hc.reset()
            owner.reset()
        assert bytes(e.stats)==baseline
        # Public sources, uploads and engine may close before the synchronous mix.
        hc.initialize(us[0])
        for value in (*leases,*uploads,*ht,*us): value.close()
        store.close(); e.close()
        with monkeypatch.context() as patch:
            native_only(patch,hc,owner,e)
            assert hc.mix(owner,weights,0)==1


@pytest.mark.parametrize('failure',('busy','pending','position','scale','stale','numeric'))
def test_direct_moe_failure_contract(cp,libraries,tmp_path,failure):
    path=tmp_path/'bad.gguf'; data=fixture(path,bad_scalar=failure=='numeric')
    with ExitStack() as stack:
        r,e,store,leases,uploads,owner,base=native_setup(stack,path,data,libraries,IDS[::2])
        weights,_,_=make_weights(stack,tmp_path,r,e,256,5)
        hc=stack.enter_context(e.hc_state(256,5)); us,_=upload(stack,tmp_path,r,e,[data['x']])
        hc.initialize(us[0]); job=owner.submit(data['x']) if failure=='busy' else None
        if failure=='pending': hc.read(weights)
        handle=owner._handle
        if failure=='stale': owner.close()
        done=C.c_uint64(123)
        with pytest.raises(NativeError):
            r._call('ml_hc_mix_scaled',hc._handle,weights._handle,handle,2,
                1 if failure=='position' else 0,float('nan') if failure=='scale' else .37,C.byref(done))
        assert done.value==0
        with pytest.raises(NativeError,match='invalid'): hc_read(hc,'streams',256)
        if job: job.close()
        if failure=='numeric':
            with pytest.raises(NativeError,match='invalid'): owner.submit(data['x'])
        hc.reset(); hc.initialize(us[0])
        if failure not in ('stale','numeric'):
            owner.reset(); assert hc.mix(owner,weights,0,scale=.37)==1
        # Legacy unscaled ABI also supports MoE without exporting read tickets.
        if failure=='position':
            hc.reset(); hc.initialize(us[0]); owner.reset(); done.value=0
            r._call('ml_hc_mix',hc._handle,weights._handle,owner._handle,2,0,C.byref(done))
            assert done.value==1


def actual_ple(stack,r,e,store,path,eps):
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.qwen4exp_ple import PLEConfig, Qwen4ExpPLE
    import cupy as cp
    table=stack.enter_context(store.lease('per_layer_token_embd.weight'))
    config=PLEConfig.from_metadata(store.metadata,table.descriptor.shape[0])
    tensors=[]; decoded=[]
    for suffix in ('key','value','norm_key','norm_query','norm_conv','conv1d'):
        l=stack.enter_context(store.lease('blk.1.ple_'+suffix+'.weight'))
        tensors.append(stack.enter_context(e.upload(l)))
        decoded.append(cp.asarray(control(l)).reshape((4,2560) if suffix.startswith('norm_') else l.descriptor.shape))
    weights=stack.enter_context(e.ple_weights(table,config,*tensors,epsilon=eps,arithmetic='reference_fp32'))
    state=stack.enter_context(weights.state(8))
    oracle=stack.enter_context(MultiShardTensorStore.open(path))
    ref=Qwen4ExpPLE(oracle,'per_layer_token_embd.weight',config,*decoded,epsilon=eps)
    stack.callback(ref.close)
    return state,ref


@pytest.mark.skipif(os.environ.get('MINI_LLM_MOE_CHECKPOINT')!='1',reason='opt in actual MoE chaining')
@pytest.mark.parametrize('layer,mode,with_ple',[(l,m,False) for l in (0,1,3,47) for m in ('cpu','gpu','mixed')]+[(1,'mixed',True)])
def test_actual_layered_order(cp,libraries,tmp_path,monkeypatch,report,layer,mode,with_ple):
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    inventory=json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    path=inventory['shards'][0]['path']; r=Runtime(); tokens=(1,7,7,2,3)
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=511*2**20)); store=stack.enter_context(r.gguf(path))
        eps=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
        weights=[]; refs=[]
        for stem in (f'blk.{layer}.hc_attn_',f'blk.{layer}.hc_ffn_','output_hc_'):
            pairs=[checkpoint_upload(stack,e,store,stem+s) for s in
                (('norm.weight','down.weight','up.weight') if stem=='output_hc_' else
                 ('norm.weight','down.weight','up.weight','inject.weight'))]
            weights.append(stack.enter_context(e.hc_weights(*(p[0] for p in pairs),epsilon=eps,arithmetic='reference_fp32')))
            refs.append(Qwen4ExpHyperConnection(*(p[1] for p in pairs),epsilon=eps))
        recurrent=layer%4!=3
        stems=REC_STEMS if recurrent else STEMS+['attn_output.weight']+NORMS
        pairs=[checkpoint_upload(stack,e,store,f'blk.{layer}.'+s) for s in stems]
        ts,values=[p[0] for p in pairs],[p[1] for p in pairs]
        if recurrent:
            spec=Qwen4ExpMixerSpec.from_metadata(store.metadata)
            ref=recurrent_reference(spec,values); branch=stack.enter_context(create(e,spec,ts,capacity=8))
        else:
            ref=Qwen4ExpAttention(QSAConfig(2560,epsilon=eps),**dict(zip(NAMES,values[:6])),
                **dict(zip(('gamma_q','gamma_k','gamma_index_q','gamma_index_k'),values[6:])))
            branch=stack.enter_context(e.qsa_layer(4096,*ts,policy='ggml_cpu',epsilon=eps))
        hc=stack.enter_context(e.hc_state(2560,320))
        ple,pr=actual_ple(stack,r,e,store,path,eps) if with_ple else (None,None)
        xs=[(np.sin(np.arange(2560,dtype=np.float32)*.03125+t*.125)*.2).astype(np.float32) for t in range(len(tokens))]
        us,_=upload(stack,tmp_path,r,e,xs)
        source=CheckpointMoE(stack,e,store,layer)
        # Freeze placement using only a setup prepass, never adaptive execution.
        state=ref.create_state(1 if recurrent else 4096); ps=pr.fresh_state() if pr else None
        selected=set(); first_ids=None
        for t,x in enumerate(xs):
            streams=refs[0].initialize(cp.asarray(x))
            if pr: streams,ps=pr.step(tokens[t],streams,ps)
            h,g=refs[0].read(streams)
            y=ref.step(h[None],state)[0] if recurrent else ref.step(h,state,position=t)
            streams=refs[0].write(streams,y,g); fh,_=refs[1].read(streams)
            _,(_,ids)=source.route(cp,fh)
            route=cp.asnumpy(ids).ravel().tolist(); selected.update(route)
            if first_ids is None: first_ids=route
        placement=() if mode=='cpu' else tuple(sorted(selected)) if mode=='gpu' else tuple(sorted(first_ids[::2]))
        del state,ps
        owner=source.native(stack,placement,libraries)
        model,executor=stack.enter_context(source.oracle(cp,placement))
        baseline=bytes(e.stats); maximum=0.; final_max=0.; replay_outputs=[]; cutpoints=[]
        for replay in range(2):
            state=ref.create_state(1 if recurrent else 4096); ps=pr.fresh_state() if pr else None
            output=[]
            for t,x in enumerate(xs):
                hc.initialize(us[t]); streams=refs[0].initialize(cp.asarray(x))
                if ple:
                    streams,ps=pr.step(tokens[t],streams,ps); ple.step(hc,tokens[t],t)
                    close(hc_read(hc,'streams',2560),streams)
                h,g=refs[0].read(streams)
                with monkeypatch.context() as patch:
                    patch.setattr(branch,'step',forbidden); patch.setattr(hc,'read',forbidden); patch.setattr(hc,'write',forbidden)
                    assert hc.mix(branch,weights[0],t)==t+1
                y=check_branch(branch,ref,state,h,recurrent)
                streams=refs[0].write(streams,y,g)
                close(hc_read(hc,'streams',2560),streams)
                before_ffn=streams
                fh,fg=refs[1].read(streams)
                want=model(fh)
                streams=refs[1].write(streams,want,fg)
                with monkeypatch.context() as patch:
                    native_only(patch,hc,owner,e)
                    assert hc.mix(owner,weights[1],t,scale=source.scale)==t+1
                got=hc_read(hc,'streams',2560); close(got,streams)
                maximum=max(maximum,float(np.max(np.abs(got-cp.asnumpy(streams)))))
                same_input=refs[2].collapse(cp.asarray(got))
                native_fh=owner.input_readback_for_test()
                quantum_target=refs[2].collapse(refs[1].write(before_ffn,model(cp.asarray(native_fh)),fg))
                hc.collapse(weights[2]); got=hc_read(hc,'hidden',2560); target=refs[2].collapse(streams)
                close(native_fh,fh)
                close(got,same_input)
                close(got,quantum_target)
                a=got.astype(np.float64); b=cp.asnumpy(target).astype(np.float64)
                absolute=float(np.max(np.abs(a-b)))
                relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
                # CPU Q8 activation rounding is discontinuous at HC reduction ulps.
                # Keep strict same-input checks above; bound the complete chain in
                # norm/scale, not relative to individual nearly cancelled elements.
                assert relative<2e-4
                assert absolute<2e-4+2e-4*np.max(np.abs(b))
                final_max=max(final_max,absolute)
                if replay==0:
                    cutpoints.append(dict(token_index=t,hidden_max_abs=float(np.max(np.abs(native_fh-cp.asnumpy(fh)))),
                        same_collapse_input_max_abs=float(np.max(np.abs(got-cp.asnumpy(same_input)))),
                        same_moe_input_final_max_abs=float(np.max(np.abs(got-cp.asnumpy(quantum_target)))),
                        whole_final_max_abs=absolute,whole_final_relative_l2=relative))
                output.append(got.copy()); hc.reset()
            replay_outputs.append(np.stack(output)); branch.reset(); owner.reset()
            if ple: ple.reset()
            del state,ps
        np.testing.assert_array_equal(*replay_outputs)
        assert bytes(e.stats)==baseline
        cp.cuda.get_current_stream().synchronize()
        reserved_before_trim=cp.get_default_memory_pool().total_bytes()
        cp.get_default_memory_pool().free_all_blocks()
        pool=cp.get_default_memory_pool().total_bytes()
        assert e.stats.peak_bytes<512*2**20
        live=cp.get_default_memory_pool().used_bytes()
        assert e.stats.device_bytes+live<512*2**20
        row=dict(layer=layer,attention='GDN' if recurrent else 'QSA',mode=mode,ple=with_ple,
            tokens=list(tokens),replays=2,placement=placement,scale=source.scale,
            max_stream_abs=maximum,max_final_abs=final_max,device_bytes=e.stats.device_bytes,
            pinned_bytes=e.stats.pinned_bytes,peak_bytes=e.stats.peak_bytes,oracle_pool_bytes=pool,oracle_live_bytes=live,oracle_reserved_before_trim=reserved_before_trim,cutpoints=cutpoints)
        report.append(row); print('\nWP5_CHAIN='+json.dumps(row))
