"""Bounded WP7 diagnostic experiment, not a replacement numerical oracle.

Run independent processes for actual native capture, Python capture, and native
identical-input replay. Only selected expert rows are materialized by the oracle.
"""
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import sys

import cupy as cp
import numpy as np

from mini_llm.native import Runtime
from mini_llm.native.decoder import NativeDecoder
from mini_llm.ops.ggml_packed_cpu import _load_bridge

FIELDS = ('ids','weights','logits','probabilities','cpu_rows','terms','routed','shared',
          'output','input','gate','up','hidden','score','raw')
STAGES = ('hc_attention_read','post_mixer','hc_ffn_read','post_moe')


def libraries():
    return dict(bridge_library=_load_bridge()._name,
                ggml_library=os.environ['MINI_LLM_GGML_CPU_LIBRARY'])


def snapshot(owner, d, sf, layer=None):
    shapes = [(10,),(10,),(512,),(512,),(10,d),(10,d),(d,),(d,),(d,),(d,),
              (sf,),(sf,),(sf,),(1,),(d,)]
    result = {}
    for which,(name,shape) in enumerate(zip(FIELDS,shapes)):
        a = np.empty(shape,np.int32 if which == 0 else np.float32)
        args = (owner._handle,which,a.ctypes.data,a.size) if layer is None else (
            owner._handle,layer,which,a.ctypes.data,a.size)
        owner._runtime._call('ml_moe_diagnostic_for_test' if layer is None else
                            'ml_decoder_moe_diagnostic_for_test',*args)
        result[name] = a
    return result


def python_snapshot(moe, x):
    """Recompute diagnostics through existing unchanged router, pool and branches."""
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.external.qwen4exp_hyperconnection import _sigmoid
    x = cp.asarray(x)
    router, executor = moe.router,moe.experts.expert_executor
    logits = (x[None]@router.W_router+router.b_router).reshape(-1)
    probabilities = cp.exp(logits-cp.max(logits)); probabilities /= cp.sum(probabilities)
    rw,ri = router.forward(x.reshape(1,1,-1))
    order = cp.argsort(ri.reshape(-1)); ids = cp.asnumpy(ri.reshape(-1)[order]).astype(np.int32)
    weights = (rw.reshape(-1)[order]*moe.scale).astype(cp.float32)
    bundles = [executor.host_provider(ExpertKey(executor.model_id,moe.experts.layer_id,int(i))) for i in ids]
    rows = executor._ggml_row_pool.submit(cp.asnumpy(x)[None],bundles).result().copy()
    terms = cp.asarray(rows)*weights[:,None]
    ordered = cp.zeros_like(x)
    for term in terms:
        ordered += term
    routed = moe.experts.forward(x.reshape(1,1,-1),rw*moe.scale,ri).reshape(-1)
    # Establish that the diagnostic decomposition matches the production path.
    cp.testing.assert_array_equal(ordered,routed)
    gate,up = moe.gate(x),moe.up(x)
    hidden = (gate*_sigmoid(gate))*up
    raw,score = moe._run_shared(x)
    shared = _sigmoid(score)*raw
    output = moe(x)
    cp.testing.assert_array_equal(output,routed+shared)
    result = dict(ids=ids,weights=weights,logits=logits,probabilities=probabilities,
        cpu_rows=rows,terms=terms,routed=routed,shared=shared,output=output,input=x,
        gate=gate,up=up,hidden=hidden,score=score.reshape(1),raw=raw)
    # Raw input Q8 encodings provide a cheap threshold-sensitivity check.
    for name in ('quantize_row_q8_K','quantize_row_q8_0'):
        q = getattr(executor._ggml_cpu._ggml,name)
        q.argtypes = [C.c_void_p,C.c_void_p,C.c_int64]; q.restype = None
        row = cp.asnumpy(x).astype(np.float32)
        qb,qk = (292,256) if name.endswith('_K') else (34,32)
        encoded = np.empty(row.size//qk*qb,np.uint8)
        q(row.ctypes.data,encoded.ctypes.data,row.size)
        result[name] = encoded
    return {k:cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for k,v in result.items()}


def save_capture(destination, decoder, layer, **extras):
    saved = snapshot(decoder,decoder.width,640 if decoder.width==2560 else 256,layer)
    for i in range(decoder.layers):
        for s in STAGES:
            saved[f'{i}_{s}'] = decoder.trace_for_test(i,s)
    np.testing.assert_array_equal(saved['input'],saved[f'{layer}_hc_ffn_read'])
    saved.update(extras)
    np.savez(destination,**saved)


def actual_path():
    return json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())['shards'][0]['path']


def full_capture_preflight(root, phase):
    import time
    expected='GPU-e961516f-d77d-813c-98e9-2f5403f71631'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('full capture requires the exact GPU1 UUID mask')
    if cp.cuda.runtime.getDeviceCount()!=1 or cp.cuda.runtime.getDevice()!=0:
        raise RuntimeError('full capture requires sole visible logical CUDA device 0')
    free,total=cp.cuda.runtime.memGetInfo()
    budget,headroom=10*1024**3,2*1024**3
    record=dict(phase=phase,unix_time=time.time(),gpu_mask=expected,free_bytes=free,
                total_bytes=total,budget_bytes=budget,headroom_bytes=headroom,
                admitted=free>=budget+headroom)
    (root/f'{phase}_preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print('WP7 preflight',json.dumps(record),flush=True)
    if not record['admitted']:
        raise RuntimeError('insufficient GPU1 free memory for fixed 10 GiB budget + 2 GiB headroom')


def actual_native(root):
    full_capture_preflight(root,'native')
    saved_logits=[]
    with NativeDecoder.open(Runtime(),actual_path(),**libraries(),capacity=4,physical_pages=1,
                            threads=2,trace=True,budget_bytes=10*1024**3) as decoder:
        for t,token in enumerate((1,7)):
            logits=decoder.step(token); saved_logits.append(logits.copy())
            assert decoder.state==(t+1,False)
            target=root if t==0 else root/'token1'
            save_capture(target/'native.npz',decoder,6,full_logits=logits,
                token=np.array([token],np.int32),position=np.array([t+1],np.int64),
                full_model_layers=np.array([48],np.int32))
        decoder.reset(); assert decoder.state==(0,False)
        for t,token in enumerate((1,7)):
            np.testing.assert_array_equal(decoder.step(token),saved_logits[t])
            assert decoder.state==(t+1,False)
        decoder.reset(); assert decoder.state==(0,False)
    # Context exit closes all native ownership; the phase process then exits.
    (root/'native_replay.json').write_text(json.dumps(dict(tokens=[1,7],layers=48,
        exact_logits_after_reset=True,final_position=0),indent=2)+'\n')


def actual_python(root):
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    full_capture_preflight(root,'python')
    traces = {}
    with Qwen4ExpGGUFModel.open(actual_path(),max_length=4,expert_cpu_backend='ggml_pool',
        expert_promote_after=1000000,cache_bytes=0,host_expert_bytes=0,cpu_workers=2,
        static_device_budget_bytes=7*1024**3,device_budget_bytes=10*1024**3,
        host_budget_bytes=4*1024**3,
        diagnostic_callback=lambda i,s,x:traces.__setitem__(f'{i}_{s}',cp.asnumpy(x).copy())) as model:
        assert len(model.blocks)==48 and model.plan['cache_bytes']==0
        state=model.create_generation_state(max_length=4)
        for t,token in enumerate((1,7)):
            target=root if t==0 else root/'token1'
            native=dict(np.load(target/'native.npz',allow_pickle=False))
            logits=cp.asnumpy(model.step(token,state))
            b = model.blocks[6]
            saved = dict(traces)
            saved.update(full_logits=logits,token=np.array([token],np.int32),
                         full_model_layers=np.array([48],np.int32))
            embedding=model.embedding(token)
            streams0=model.final_hc.initialize(embedding)
            hidden0,gains0=model.blocks[0].attention_hc.read(streams0)
            np.testing.assert_array_equal(cp.asnumpy(hidden0),saved['0_hc_attention_read'])
            saved.update(first_hc_input=cp.asnumpy(embedding),first_hc_hidden=cp.asnumpy(hidden0),
                         first_hc_gains=cp.asnumpy(gains0))
            for lineage,hidden in [('native',native['input']),('python',traces['6_hc_ffn_read'])]:
                diag = python_snapshot(b.moe,hidden)
                saved.update({f'{lineage}_{k}':v for k,v in diag.items()})
                streams = cp.asarray(native['6_post_mixer'] if lineage=='native' else traces['6_post_mixer'])
                h,gains = b.ffn_hc.read(streams)
                saved[f'{lineage}_hc_hidden'] = cp.asnumpy(h)
                saved[f'{lineage}_gains'] = cp.asnumpy(gains)
                saved[f'{lineage}_hc_write'] = cp.asnumpy(b.ffn_hc.write(streams,cp.asarray(diag['output']),gains))
            np.savez(target/'python.npz',**saved)


def native_replay(root, synthetic=False, layer=6):
    from native_moe_checkpoint_helpers import CheckpointMoE
    from test_native_engine_moe import fixture,native_setup
    from test_native_engine_hc import upload,make_weights
    from test_native_engine_chaining import checkpoint_upload
    n,p = dict(np.load(root/'native.npz')),dict(np.load(root/'python.npz'))
    d,layer = (256,15) if synthetic else (2560,layer)
    with ExitStack() as stack:
        if synthetic:
            path=root/'replay.gguf'; data=fixture(path)
            r,e,store,leases,weights,owner,_ = native_setup(stack,path,data,libraries(),())
            hc_weights,_,_ = make_weights(stack,root,r,e,256,5,seed=5)
            sf=256
        else:
            r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=512*2**20))
            store=stack.enter_context(r.gguf(actual_path()))
            selected=CheckpointMoE(stack,e,store,layer)
            owner=selected.native(stack,(),libraries(),threads=2)
            pairs=[checkpoint_upload(stack,e,store,f'blk.{layer}.hc_ffn_'+s) for s in
                   ('norm.weight','down.weight','up.weight','inject.weight')]
            hc_weights=stack.enter_context(e.hc_weights(*(v[0] for v in pairs),
                epsilon=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon'],arithmetic='reference_fp32'))
            sf=640
        saved={}
        hc=stack.enter_context(e.hc_state(d,5 if synthetic else 320))
        for lineage,x in [('native',n['input']),('python',p[f'python_input'])]:
            with owner.submit(np.ascontiguousarray(x)) as job:
                job.wait()
                diag=snapshot(owner,d,sf)
            saved.update({f'{lineage}_{k}':v for k,v in diag.items()})
            streams = n[f'{layer}_post_mixer'] if lineage=='native' else p[f'{layer}_post_mixer']
            us,_=upload(stack,root,r,e,[streams,diag['output']])
            hc.initialize(us[0]); ticket=hc.read(hc_weights)
            for field,shape in [('hidden',(d,)),('gains',(4,))]:
                a=np.empty(shape,np.float32); hc.readback_for_test(field,a)
                saved[f'{lineage}_hc_{field}']=a
            hc.write(ticket,us[1]); a=np.empty((4,d),np.float32); hc.readback_for_test('streams',a)
            saved[f'{lineage}_hc_write']=a; hc.reset()
        if not synthetic and layer==6 and 'first_hc_input' in p:
            pairs=[checkpoint_upload(stack,e,store,'blk.0.hc_attn_'+s) for s in
                   ('norm.weight','down.weight','up.weight','inject.weight')]
            first_weights=stack.enter_context(e.hc_weights(*(v[0] for v in pairs),
                epsilon=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon'],arithmetic='reference_fp32'))
            us,_=upload(stack,root,r,e,[p['first_hc_input']])
            hc.initialize(us[0]); hc.read(first_weights)
            for field,shape in [('hidden',(d,)),('gains',(4,))]:
                a=np.empty(shape,np.float32); hc.readback_for_test(field,a)
                saved[f'first_hc_{field}']=a
            np.testing.assert_array_equal(saved['first_hc_hidden'],n['0_hc_attention_read'])
            hc.reset()
        np.savez(root/'replay.npz',**saved)


def actual_layer7_token1(root):
    """Four-way replay of a later amplification using retained full trajectories.

    No model step or boundary injection: source inputs/streams are immutable full
    token-1 captures. Unlike layer 6, original owner internal buffers were not
    retained here; exact paired HC-write replay checks its original aggregate.
    """
    from native_moe_checkpoint_helpers import CheckpointMoE
    from test_native_engine_chaining import checkpoint_upload
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    source_dir=root/'token1'; target=root/'layer7_token1'; layer=7
    n,p=(dict(np.load(source_dir/f'{s}.npz',allow_pickle=False)) for s in ('native','python'))
    keep=lambda data:{k:v for k,v in data.items() if k.split('_',1)[0].isdigit() or k.startswith('full_') or k=='token'}
    saved_native=keep(n); saved_native['input']=n['7_hc_ffn_read']
    np.savez(target/'native.npz',**saved_native)
    saved=keep(p)
    ggml_keepalive=C.CDLL(os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
    with ExitStack() as stack:
        r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=512*2**20))
        store=stack.enter_context(r.gguf(actual_path())); store.validate_qwen4exp()
        source=CheckpointMoE(stack,e,store,layer)
        moe,_=stack.enter_context(source.oracle(cp,(),threads=2))
        moe.router.W_router=cp.asarray(source.router).T
        pairs=[checkpoint_upload(stack,e,store,f'blk.{layer}.hc_ffn_'+s) for s in
               ('norm.weight','down.weight','up.weight','inject.weight')]
        hc=Qwen4ExpHyperConnection(*(v[1] for v in pairs),
            epsilon=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon'])
        for lineage,data in [('native',n),('python',p)]:
            diag=python_snapshot(moe,data['7_hc_ffn_read'])
            saved.update({f'{lineage}_{k}':v for k,v in diag.items()})
            streams=cp.asarray(data['7_post_mixer']); h,g=hc.read(streams)
            saved[f'{lineage}_hc_hidden']=cp.asnumpy(h); saved[f'{lineage}_gains']=cp.asnumpy(g)
            saved[f'{lineage}_hc_write']=cp.asnumpy(hc.write(streams,cp.asarray(diag['output']),g))
    np.savez(target/'python.npz',**saved)
    native_replay(target,layer=7)
    report(target,7)


def actual_gdn8_state_replay(root):
    """Separate current input from carried recurrent state at token-1 layer 8."""
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from test_native_engine_chaining import checkpoint_upload,REC_STEMS
    from test_native_engine_recurrent import reference as rec_reference
    from test_native_engine_hc import upload
    target=root/'gdn8'; captures={lineage:[dict(np.load(folder/f'{name}.npz',allow_pickle=False))
        for folder in (root,root/'token1')] for lineage,name in [('native','native'),('python','python')]}
    saved={}; summary=dict(layer=8,conditioned={},captured_hc_write={})
    with ExitStack() as stack:
        r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=512*2**20))
        store=stack.enter_context(r.gguf(actual_path())); store.validate_qwen4exp()
        spec=Qwen4ExpMixerSpec.from_metadata(store.metadata)
        pairs=[checkpoint_upload(stack,e,store,'blk.8.'+s) for s in REC_STEMS]
        ts,values=[v[0] for v in pairs],[v[1] for v in pairs]
        branch=stack.enter_context(e.recurrent_layer(spec,4,*ts,arithmetic='reference_fp32'))
        reference=rec_reference(spec,values)
        pairs=[checkpoint_upload(stack,e,store,'blk.8.hc_attn_'+s) for s in
               ('norm.weight','down.weight','up.weight','inject.weight')]
        weights=stack.enter_context(e.hc_weights(*(v[0] for v in pairs),epsilon=spec.rms_eps,arithmetic='reference_fp32'))
        hc_reference=Qwen4ExpHyperConnection(*(v[1] for v in pairs),epsilon=spec.rms_eps)
        hc=stack.enter_context(e.hc_state(2560,320))
        for lineage,sequence in captures.items():
            state=reference.create_state(1); summary['conditioned'][lineage]=[]
            summary['captured_hc_write'][lineage]=[]
            for t,capture in enumerate(sequence):
                x=capture['8_hc_attention_read']; streams=capture['7_post_moe']
                us,_=upload(stack,target,r,e,[x,streams])
                expected=cp.asnumpy(reference.forward(cp.asarray(x)[None,None],state)[0,0])
                branch.step(us[0],t)
                output=np.empty(2560,np.float32); branch.readback_for_test('output',output)
                row=dict(output=metric(output,expected))
                for name,refvalue in [('history',state.conv.history),('matrix',state.delta.memory)]:
                    refvalue=cp.asnumpy(refvalue); value=np.empty_like(refvalue)
                    branch.readback_for_test(name,value)
                    row[name]=metric(value,refvalue)
                    np.testing.assert_allclose(value,refvalue,rtol=2e-4,atol=2e-4)
                    saved[f'{lineage}_{t}_native_{name}']=value
                    saved[f'{lineage}_{t}_python_{name}']=refvalue
                np.testing.assert_allclose(output,expected,rtol=2e-4,atol=2e-4)
                saved[f'{lineage}_{t}_native_output']=output
                saved[f'{lineage}_{t}_python_output']=expected
                summary['conditioned'][lineage].append(row)
                # The native state/input lineage should reconstruct its original
                # full-decoder HC write exactly, without any full-model replay.
                out_upload,_=upload(stack,target,r,e,[output])
                hc.initialize(us[1]); ticket=hc.read(weights)
                hidden=np.empty(2560,np.float32); hc.readback_for_test('hidden',hidden)
                if lineage=='native': np.testing.assert_array_equal(hidden,x)
                hc.write(ticket,out_upload[0]); actual=np.empty((4,2560),np.float32)
                hc.readback_for_test('streams',actual); hc.reset()
                summary['captured_hc_write'][lineage].append(metric(actual,capture['8_post_mixer']))
                if lineage=='native': np.testing.assert_array_equal(actual,capture['8_post_mixer'])
            branch.reset()
        # Hold current token input identical while changing only token-0 input
        # history, hence its committed state. This is not a full-trajectory gate.
        common=cp.asarray(captures['native'][1]['8_hc_attention_read'])
        controlled=[]
        for lineage in ('native','python'):
            state=reference.create_state(1)
            previous=cp.asarray(captures[lineage][0]['8_hc_attention_read'])
            reference.forward(previous[None,None],state)
            result=cp.asnumpy(reference.forward(common[None,None],state)[0,0])
            controlled.append(result); saved[f'{lineage}_history_common_current_output']=result
        summary['same_current_input_different_prior_history']=metric(*controlled)
        summary['previous_input_difference']=metric(captures['native'][0]['8_hc_attention_read'],captures['python'][0]['8_hc_attention_read'])
        summary['current_input_difference']=metric(captures['native'][1]['8_hc_attention_read'],captures['python'][1]['8_hc_attention_read'])
        summary['actual_lineage_output_difference']=metric(saved['native_1_native_output'],saved['python_1_native_output'])
    np.savez(target/'states.npz',**saved)
    (target/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def actual_prefix_capture(root):
    """Actual first seven layers; same token dependency chain through layer 6.

    Only used when another process occupies GPU1. No surrogate geometry, no
    numerical boundary injection, and no claim that this is a full-model gate.
    """
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel,Qwen4ExpLayer
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention,QSAConfig
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    from test_native_engine_recurrent import reference as rec_reference
    from test_native_engine_chaining import checkpoint_upload,REC_STEMS,STEMS,NORMS,NAMES
    from test_native_engine_moe_chaining import actual_ple
    from native_moe_checkpoint_helpers import CheckpointMoE
    traces={}
    class HCProbe:
        def __init__(self,base,i,stage): self.base,self.i,self.stage=base,i,stage
        def read(self,x):
            h,g=self.base.read(x); traces[f'{self.i}_{STAGES[self.stage]}']=cp.asnumpy(h).copy()
            return h,g
        def write(self,x,y,g):
            out=self.base.write(x,y,g); traces[f'{self.i}_{STAGES[self.stage+1]}']=cp.asnumpy(out).copy()
            return out
    with ExitStack() as stack:
        r=Runtime(); e=stack.enter_context(r.engine(budget_bytes=3*1024**3,max_sessions=1))
        store=stack.enter_context(r.gguf(actual_path())); headers=store.validate_qwen4exp()
        eps=headers.mixer.rms_eps
        def hc(stem,inject=True):
            pairs=[checkpoint_upload(stack,e,store,stem+s) for s in
                (('norm.weight','down.weight','up.weight','inject.weight') if inject else
                 ('norm.weight','down.weight','up.weight'))]
            native=stack.enter_context(e.hc_weights(*(p[0] for p in pairs),epsilon=eps,arithmetic='reference_fp32'))
            return native,Qwen4ExpHyperConnection(*(p[1] for p in pairs),epsilon=eps)
        layers=[]; blocks=[]; ffns=[]
        for i in range(7):
            stem=f'blk.{i}.'; a,ar=hc(stem+'hc_attn_'); f,fr=hc(stem+'hc_ffn_'); ffns.append(fr)
            rec=headers.recurrent[i]
            pairs=[checkpoint_upload(stack,e,store,stem+s) for s in (REC_STEMS if rec else STEMS+['attn_output.weight']+NORMS)]
            ts,values=[p[0] for p in pairs],[p[1] for p in pairs]
            if rec:
                ref=rec_reference(headers.mixer,values)
                branch=stack.enter_context(e.recurrent_layer(headers.mixer,4,*ts,arithmetic='reference_fp32'))
            else:
                ref=Qwen4ExpAttention(QSAConfig(2560,epsilon=eps),**dict(zip(NAMES,values[:6])),
                    **dict(zip(('gamma_q','gamma_k','gamma_index_q','gamma_index_k'),values[6:])))
                branch=stack.enter_context(e.qsa_layer(4,*ts,policy='ggml_cpu',epsilon=eps,
                    kv_policy='reference_fp32',page_size=16,physical_pages=1,host_budget_bytes=2**20))
            ple,pr=actual_ple(stack,r,e,store,actual_path(),eps) if i in headers.ple_layers else (None,None)
            source=CheckpointMoE(stack,e,store,i); owner=source.native(stack,(),libraries(),threads=2)
            moe,_=stack.enter_context(source.oracle(cp,(),threads=2))
            # Match qwen4exp_model._build: dense router upload, then transpose
            # view. The existing selected-layer helper instead copies W.T to C.
            moe.router.W_router=cp.asarray(source.router).T
            layers.append((a,branch,f,owner,ple))
            blocks.append(Qwen4ExpLayer(HCProbe(ar,i,0),ref,HCProbe(fr,i,2),moe,rec,pr))
        final,final_ref=hc('output_hc_',False)
        emb=stack.enter_context(store.lease('token_embd.weight'))
        out=stack.enter_context(e.upload(stack.enter_context(store.lease('output.weight'))))
        decoder=stack.enter_context(e.decoder(emb,out,final,stack.enter_context(e.hc_state(2560,320)),layers,capacity=4,trace=True))
        decoder.step(1); save_capture(root/'native.npz',decoder,6)
        def embedding(t):
            d=emb.descriptor
            w=PackedGGUFWeight(np.frombuffer(emb.read(t*d.row_bytes,d.row_bytes),np.uint8),(1,2560),'IQ4_XS')
            return cp.asarray(w.dequantize_rows_reference([0])[0])
        def no_logits(x): raise AssertionError('prefix oracle must not project dummy logits')
        model=Qwen4ExpGGUFModel.from_components(width=2560,vocab=248320,blocks=blocks,
            final_hc=final_ref,embedding=embedding,output=no_logits,max_length=4)
        model._step(1,model.create_generation_state(max_length=4),project_logits=False)
        native=dict(np.load(root/'native.npz')); saved=dict(traces)
        for lineage,x in [('native',native['input']),('python',traces['6_hc_ffn_read'])]:
            diag=python_snapshot(blocks[6].moe,x)
            saved.update({f'{lineage}_{k}':v for k,v in diag.items()})
            streams=cp.asarray(native['6_post_mixer'] if lineage=='native' else traces['6_post_mixer'])
            h,g=ffns[6].read(streams)
            saved[f'{lineage}_hc_hidden']=cp.asnumpy(h); saved[f'{lineage}_gains']=cp.asnumpy(g)
            saved[f'{lineage}_hc_write']=cp.asnumpy(ffns[6].write(streams,cp.asarray(diag['output']),g))
        saved['prefix_layers']=np.array([7],np.int32)
        np.savez(root/'python.npz',**saved)


def synthetic_capture(root):
    from test_native_engine_decoder import small_setup,small_reference
    with ExitStack() as stack:
        decoder,_,refs=small_setup(stack,root,libraries())
        model,traces=small_reference(cp,refs,stack)
        model.step(1,model.create_generation_state(max_length=4)); decoder.step(1)
        save_capture(root/'native.npz',decoder,15)
        native=dict(np.load(root/'native.npz'))
        saved={f'{i}_{s}':v for (i,s),v in traces.items()}
        # Immutable synthetic MoE is shared across all 48 layers, not its state.
        b=model.blocks[15]
        for lineage,x in [('native',native['input']),('python',traces[15,'hc_ffn_read'])]:
            diag=python_snapshot(b.moe,x)
            saved.update({f'{lineage}_{k}':v for k,v in diag.items()})
            streams=cp.asarray(native['15_post_mixer'] if lineage=='native' else traces[15,'post_mixer'])
            h,g=b.ffn_hc.read(streams)
            saved[f'{lineage}_hc_hidden']=cp.asnumpy(h); saved[f'{lineage}_gains']=cp.asnumpy(g)
            saved[f'{lineage}_hc_write']=cp.asnumpy(b.ffn_hc.write(streams,cp.asarray(diag['output']),g))
        np.savez(root/'python.npz',**saved)


def metric(a,b):
    delta=a.astype(np.float64)-b.astype(np.float64)
    at=np.unravel_index(np.argmax(np.abs(delta)),a.shape)
    return dict(dtype=str(a.dtype),reference_dtype=str(b.dtype),rtol=2e-4,atol=2e-4,
        maxabs=float(np.max(np.abs(delta))),relL2=float(np.linalg.norm(delta)/max(np.linalg.norm(b.astype(np.float64)),1e-30)),
        maxrel=float(np.max(np.abs(delta)/np.maximum(np.abs(b.astype(np.float64)),1e-30))),
        violations=int(np.count_nonzero(np.abs(delta)>2e-4+2e-4*np.abs(b))),
        index=[int(i) for i in at],native=float(a[at]),reference=float(b[at]),exact=bool(np.array_equal(a,b)))


def report(root,layer):
    n,p,r=(dict(np.load(root/f'{s}.npz')) for s in ('native','python','replay'))
    result=dict(layer=layer,input=metric(n['input'],p['python_input']),same_input={},lineage={},routes={})
    fields=FIELDS+('hc_write',)
    for lineage in ('native','python'):
        result['same_input'][lineage]={k:metric(r[f'{lineage}_{k}'],p[f'{lineage}_{k}']) for k in fields if k!='ids'}
        nl,pl=r[f'{lineage}_logits'],p[f'{lineage}_logits']
        def margin(v):
            indices=np.argsort(-v,kind='stable'); return dict(ids=indices[:11].tolist(),
                tenth=float(v[indices[9]]),eleventh=float(v[indices[10]]),margin=float(v[indices[9]]-v[indices[10]]))
        result['routes'][lineage]=dict(native_ids=r[f'{lineage}_ids'].tolist(),python_ids=p[f'{lineage}_ids'].tolist(),
            native_weights=r[f'{lineage}_weights'].tolist(),python_weights=p[f'{lineage}_weights'].tolist(),
            ids_equal=bool(np.array_equal(r[f'{lineage}_ids'],p[f'{lineage}_ids'])),
            native_margin=margin(nl),python_margin=margin(pl))
        result['same_input'][lineage]['each_cpu_row']=[dict(expert=int(i),**metric(a,b)) for i,a,b in
            zip(r[f'{lineage}_ids'],r[f'{lineage}_cpu_rows'],p[f'{lineage}_cpu_rows'])]
    for path,source in [('native',r),('python',p)]:
        result['lineage'][path]={k:metric(source[f'native_{k}'],source[f'python_{k}']) for k in fields if k!='ids'}
        result['lineage'][path]['ids_equal']=bool(np.array_equal(source['native_ids'],source['python_ids']))
        result['lineage'][path]['each_cpu_row']=[dict(expert=int(i),**metric(a,b)) for i,a,b in
            zip(source['native_ids'],source['native_cpu_rows'],source['python_cpu_rows'])]
    result['capture_replay']={k:metric(n[k],r[f'native_{k}']) for k in FIELDS if k!='ids' and k in n}
    result['original_owner_buffers_retained']=all(k in n for k in FIELDS)
    result['conditioned_hc_read']={lineage:dict(
        hidden=metric(r[f'{lineage}_hc_hidden'],p[f'{lineage}_hc_hidden']),
        gains=metric(r[f'{lineage}_hc_gains'],p[f'{lineage}_gains'])) for lineage in ('native','python')}
    if 'first_hc_hidden' in r:
        result['first_dense_hc_identical_embedding']=dict(
            hidden=metric(r['first_hc_hidden'],p['first_hc_hidden']),
            gains=metric(r['first_hc_gains'],p['first_hc_gains']))
    result['captured_native_hc_write']=metric(n[f'{layer}_post_moe'],r['native_hc_write'])
    result['captured_python_hc_write']=metric(p[f'{layer}_post_moe'],p['python_hc_write'])
    result['boundaries']={f'{i}_{s}':metric(n[f'{i}_{s}'],p[f'{i}_{s}']) for i in range(48) for s in STAGES if f'{i}_{s}' in n and f'{i}_{s}' in p}
    result['quantized_input_changed']={name:int(np.count_nonzero(p[f'native_{name}']!=p[f'python_{name}']))
                                      for name in ('quantize_row_q8_K','quantize_row_q8_0')}
    result['q8_k_changes']=[]
    qa,qb=(p[f'{lineage}_quantize_row_q8_K'].reshape(-1,292) for lineage in ('native','python'))
    for block,(a,b) in enumerate(zip(qa,qb)):
        if np.array_equal(a,b): continue
        da,db=(float(np.frombuffer(v[:4].tobytes(),'<f4')[0]) for v in (a,b))
        changed=[]
        for j in np.flatnonzero(a[4:260]!=b[4:260]):
            xn,xp=float(p['native_input'][block*256+j]),float(p['python_input'][block*256+j])
            changed.append(dict(lane=int(j),global_index=int(block*256+j),
                native_code=int(a[4:260].view('i1')[j]),python_code=int(b[4:260].view('i1')[j]),
                native_input=xn,python_input=xp,native_x_over_stored_d=xn/da,python_x_over_stored_d=xp/db))
        result['q8_k_changes'].append(dict(block=block,native_d=da,python_d=db,code_changes=changed))
    # Ratios above are diagnostic reconstructions from stored d, not a new quantizer.
    if 'full_logits' in n and 'full_logits' in p:
        ln,lp=n['full_logits'].reshape(-1),p['full_logits'].reshape(-1)
        result['full_logits']=metric(ln,lp)
        def log_probs(x):
            x=x.astype(np.float64); shifted=x-np.max(x)
            return shifted-np.log(np.sum(np.exp(shifted)))
        an,ap=log_probs(ln),log_probs(lp); pn,pp=np.exp(an),np.exp(ap)
        topn,topp=np.argsort(-ln,kind='stable'),np.argsort(-lp,kind='stable')
        result['short_token_distribution_diagnostics']=dict(
            accepted_quality_gate=False,token_id=int(n['token'][0]),
            native_top10=topn[:10].tolist(),python_top10=topp[:10].tolist(),
            top1_equal=bool(topn[0]==topp[0]),top10_overlap=len(set(topn[:10])&set(topp[:10])),
            python_top1_native_rank=int(np.flatnonzero(topn==topp[0])[0])+1,
            kl_python_to_native=float(np.sum(pp*(ap-an))),
            kl_native_to_python=float(np.sum(pn*(an-ap))),
            probability_total_variation=float(np.sum(np.abs(pn-pp))/2),
            note='Two fixed tokens only; not perplexity, generated-text quality or accepted throughput.')
        result['strict_trajectory_gate_passed']=(result['full_logits']['violations']==0 and
            all(v['violations']==0 for v in result['boundaries'].values()))
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={k:result[k] for k in ('layer','input','routes','quantized_input_changed','captured_native_hc_write')}
    summary['same_input']={l:{k:result['same_input'][l][k] for k in ('cpu_rows','routed','shared','output','hc_write')} for l in ('native','python')}
    summary['lineage']={l:{k:result['lineage'][l][k] for k in ('ids_equal','cpu_rows','routed','shared','output')} for l in ('native','python')}
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    phase,folder=sys.argv[1:]; root=Path(folder)
    if phase=='actual-native': actual_native(root)
    elif phase=='actual-python': actual_python(root)
    elif phase=='actual-prefix': actual_prefix_capture(root)
    elif phase=='actual-layer7-token1': actual_layer7_token1(root)
    elif phase=='actual-gdn8': actual_gdn8_state_replay(root)
    elif phase=='actual-replay':
        # Keep the selected GGML backend loaded across the two standalone epochs.
        ggml_keepalive=C.CDLL(os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
        native_replay(root)
        if (root/'token1/python.npz').is_file(): native_replay(root/'token1')
    elif phase=='synthetic-capture': synthetic_capture(root)
    elif phase=='synthetic-replay': native_replay(root,True)
    elif phase=='actual-report':
        report(root,6)
        if (root/'token1/replay.npz').is_file(): report(root/'token1',6)
    elif phase=='synthetic-report': report(root,15)
    else: raise ValueError(phase)
