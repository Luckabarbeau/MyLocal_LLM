"""Identical-input real-checkpoint MoE vs linked GGML; selected banks only."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from probe_0218_hc_identity import COPY, MODEL, provenance, bounded
from probe_0220_layer_evolution import associate, metrics
from probe_0222_matched_activations import Codec, compile_codec, TYPES


def sigmoid(x):return np.exp(-np.logaddexp(np.float32(0),-x))


def normalized_route(logits,k):
    from mini_llm.ops.router_inference import RouterInference
    r=RouterInference(1,len(logits),k,renormalize_selected=True,normalization_floor=2**-14)
    r.W_router=np.asarray(logits,np.float32)[None];r.b_router=np.zeros(len(logits),np.float32)
    r._decode_fastpath_enabled=False
    weights,ids=r.forward(np.ones((1,1,1),np.float32))
    shifted=logits-logits.max();p=np.exp(shifted);p/=p.sum(dtype=np.float32)
    return p,ids.reshape(-1),weights.reshape(-1)


def sum_routes(values,weights,ids,order='slot'):
    result=np.zeros(values.shape[1],np.float32)
    slots=range(len(ids)) if order=='slot' else np.argsort(ids,kind='stable')
    for i in slots:result+=np.asarray(values[i]*weights[i],np.float32)
    return result


def compile_driver(tmp):
    binary=Path(tmp)/'moe';lib=COPY/'build-cpu/bin'
    subprocess.run(['c++','-std=c++17','-O2',str(ROOT/'tests/fixtures/qwen4exp_moe_isolated.cpp'),
        '-I'+str(COPY/'ggml/include'),'-I'+str(COPY/'ggml/src'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
        '-lggml','-lggml-base','-lggml-cpu','-o',str(binary)],check=True,capture_output=True,text=True,timeout=60)
    return binary


def experiment(store,binary,codec,work,x,layer,position,reference_dir,reference_input_id):
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    from mini_llm.ops.native_packed_cpu import packed_cpu_linear
    stem=f'blk.{layer}.';d=2560;f=640;e=512;k=10
    def desc(part):return store.tensors[stem+part+'.weight']
    def raw(part,expert=None):
        t=desc(part)
        return store.tensor_bytes(t.name) if expert is None else store.rows(t.name,0,t.shape[-2],outer=(int(expert),))
    def dense(part):
        t=desc(part);data=raw(part)
        if t.quant_type!='BF16':raise ValueError('router must be BF16')
        return (data.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(t.shape)
    def linear(part,v,rounded=False,expert=None):
        t=desc(part);v=codec(v,t.format.type_id) if rounded else np.asarray(v,np.float32)
        if t.quant_type=='BF16':return v@dense(part).T
        w=PackedGGUFWeight(raw(part,expert),t.shape if expert is None else t.shape[1:],t.quant_type)
        return packed_cpu_linear(v.reshape(1,-1),w)[0]
    types=[desc(p).format.type_id for p in ['ffn_gate_exps','ffn_up_exps','ffn_down_exps','ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp']]
    (work/'spec.txt').write_text(' '.join(map(str,[d,f,e,k]+types)))
    x.tofile(work/'input.bin');raw('ffn_gate_inp').tofile(work/'router.bin')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([str(binary),str(work),'route'],check=True,capture_output=True,text=True,timeout=30,env=env)
    refids=np.fromfile(work/'ref_ids.i32',np.int32)
    if refids.shape!=(10,) or len(set(map(int,refids)))!=10:raise ValueError('GGML top-k IDs invalid')
    refweights=np.fromfile(work/'ref_weights.f32',np.float32)
    refweights.tofile(work/'weights.bin');np.arange(k,dtype=np.int32).tofile(work/'local_ids.bin')
    for p,file in [('ffn_gate_exps','gate'),('ffn_up_exps','up'),('ffn_down_exps','down')]:
        with (work/(file+'.bin')).open('wb') as out:
            for expert in refids:out.write(np.asarray(raw(p,expert),np.uint8).tobytes())
    for p,file in [('ffn_gate_shexp','shared_gate'),('ffn_up_shexp','shared_up'),('ffn_down_shexp','shared_down'),('ffn_gate_inp_shexp','shared_router')]:
        np.asarray(raw(p),np.uint8).tofile(work/(file+'.bin'))
    subprocess.run([str(binary),str(work),'experts'],check=True,capture_output=True,text=True,timeout=30,env=env)
    def reference(name,shape):return np.fromfile(work/f'ref_{name}.f32',np.float32).reshape(shape)
    rg,ru,rh,ry=[reference(name,(k,n)) for name,n in [('gate',f),('up',f),('swiglu',f),('down',d)]]
    record=dict(layer=layer,position=position,input_source=f'retained reference eval_{reference_input_id}',
        input_norm=float(np.linalg.norm(x.astype(np.float64))),input_dtype='F32',
        formats={p:desc(p).quant_type for p in ['ffn_gate_exps','ffn_up_exps','ffn_down_exps','ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp']},
        reference_ids=refids.tolist(),reference_weights=refweights.tolist(),routes={},paths={})
    for mode,rounded in [('fp32',False),('activation_matched',True)]:
        logits=linear('ffn_gate_inp',x,rounded)
        probs,ids,weights=normalized_route(logits,k)
        by_id={int(i):float(w) for i,w in zip(ids,weights)}
        conditional=probs[refids]/np.maximum(probs[refids].sum(dtype=np.float32),np.float32(2**-14))
        record['routes'][mode]=dict(logits=metrics(logits,reference('route_logits',(e,))),
            probabilities=metrics(probs,reference('probabilities',(e,))),ids=ids.tolist(),same_top10_set=set(map(int,ids))==set(map(int,refids)),
            same_slot_order=bool(np.array_equal(ids,refids)),conditional_weights=metrics(conditional,refweights))
        g=np.stack([linear('ffn_gate_exps',x,rounded,i) for i in refids])
        u=np.stack([linear('ffn_up_exps',x,rounded,i) for i in refids])
        h=(g*sigmoid(g))*u
        y=np.stack([linear('ffn_down_exps',v,rounded,i) for v,i in zip(h,refids)])
        sg=linear('ffn_gate_shexp',x,rounded);su=linear('ffn_up_shexp',x,rounded)
        sh=(sg*sigmoid(sg))*su;sd=linear('ffn_down_shexp',sh,rounded)
        scalar=linear('ffn_gate_inp_shexp',x,rounded).reshape(-1)
        gain=sigmoid(scalar);shared=sd*gain
        routed=sum_routes(y,conditional,refids,'expert_id')
        values=dict(gate=g,up=u,swiglu=h,down=y,routed=routed,shared_gate=sg,shared_up=su,
                    shared_swiglu=sh,shared_down=sd,shared_router=scalar,shared_gain=gain,shared=shared,final=routed+shared)
        record['paths'][mode]={name:metrics(a.reshape(-1),reference(name,a.shape).reshape(-1)) for name,a in values.items()}
        record['paths'][mode]['scope']='Forced reference expert set; local routing weights; ascending-expert combination. Selection set differences reported separately.'
        if mode=='activation_matched':matched_h=h;matched_y=y
    # Teacher-forced boundaries prevent upstream error/quantization thresholds from propagating.
    forced_h=(rg*sigmoid(rg))*ru
    forced_y=np.stack([linear('ffn_down_exps',v,True,i) for v,i in zip(rh,refids)])
    sy=reference('shared_down',(d,));sscalar=reference('shared_router',(1,))
    teacher_shared_down=linear('ffn_down_shexp',reference('shared_swiglu',(f,)),True)
    teacher_routed=sum_routes(ry,refweights,refids,'expert_id')
    teacher_final=teacher_routed+sy*sigmoid(sscalar)
    record['teacher_forced']=dict(swiglu_same_gate_up=metrics(forced_h.reshape(-1),rh.reshape(-1)),
        down_same_reference_swiglu=metrics(forced_y.reshape(-1),ry.reshape(-1)),
        shared_down_same_reference_swiglu=metrics(teacher_shared_down,sy),
        shared_sigmoid_same_reference_logit=metrics(sigmoid(sscalar),reference('shared_gain',(1,))),
        combine_same_reference_outputs_weights=metrics(teacher_final,reference('final',(d,))))
    perturb=[]
    delta=matched_h-rh
    for factor in [0.,.125,.5,1.,2.]:
        hp=np.asarray(rh+np.float32(factor)*delta,np.float32)
        yp=np.stack([linear('ffn_down_exps',v,True,i) for v,i in zip(hp,refids)])
        q=codec(hp,types[2]);qr=codec(rh,types[2])
        perturb.append(dict(factor=factor,swiglu_input_relative_l2=metrics(hp.reshape(-1),rh.reshape(-1))['relative_l2'],
            rounded_input_changed_elements=int(np.count_nonzero(q!=qr)),down=metrics(yp.reshape(-1),ry.reshape(-1))))
    record['controlled_swiglu_perturbations']=perturb
    # Keep small stage vectors, not the staged packed matrices.
    output=reference_dir/f'p{position}_l{layer}';output.mkdir()
    for file in work.glob('ref_*'):output.joinpath(file.name).write_bytes(file.read_bytes())
    np.save(output/'input.npy',x)
    return record


def main():
    from mini_llm.external.gguf_store import MultiShardTensorStore
    docs=ROOT/'docs/qwen4exp';output=docs/'0224_isolated_moe_v1.json';vectors=docs/'diagnostics_0224_v1'
    if output.exists() or vectors.exists():raise FileExistsError('preserve versioned artifacts')
    events=[json.loads(s) for s in (docs/'diagnostics_0220_v2/reference/events.jsonl').read_text().splitlines()]
    identities=associate(events);lookup={identities[(e['position'],e['tensor']['ptr'])]:e['eval_id'] for e in events}
    report=dict(scope='Identical-input isolated GGML routing/MUL_MAT_ID/SwiGLU/down/shared/weighted output',
        provenance=provenance(),production_files_modified=False,reference_math_modified=False,threads=1,
        staged_experts_per_case=10,contexts_or_full_models_loaded=False,cases=[])
    vectors.mkdir()
    with MultiShardTensorStore.open(MODEL) as store,tempfile.TemporaryDirectory(prefix='isolated-moe-') as tmp:
        tmp=Path(tmp);binary=compile_driver(tmp);codec=Codec(compile_codec(tmp))
        if any('clamp' in k for k in store.metadata):raise ValueError('unhandled clamp metadata')
        for p,l in [(0,0),(1,0),(0,1),(0,2),(0,17),(0,45)]:
            eid=lookup[(p,l,'hc_ffn_read')]
            x=np.fromfile(docs/f'diagnostics_0220_v2/reference/eval_{eid}.f32',np.float32)
            if x.shape!=(2560,):raise ValueError('FFN input layout mismatch')
            report['cases'].append(experiment(store,binary,codec,tmp,x,l,p,vectors,eid))
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps([dict(position=c['position'],layer=c['layer'],routes={m:r['same_top10_set'] for m,r in c['routes'].items()},
        fp32_final=c['paths']['fp32']['final']['relative_l2'],matched_final=c['paths']['activation_matched']['final']['relative_l2'],
        forced=c['teacher_forced']['combine_same_reference_outputs_weights']['relative_l2'],
        down_teacher=c['teacher_forced']['down_same_reference_swiglu']['relative_l2']) for c in report['cases']],indent=2))

if __name__=='__main__':main()
