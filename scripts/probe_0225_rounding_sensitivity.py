"""Controlled diagnostic perturbations; no production or reference math changes."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from probe_0218_hc_identity import COPY, MODEL, provenance, bounded
from probe_0220_layer_evolution import metrics, associate
from probe_0210_qwen4exp_reference import logits_metrics
from probe_0222_matched_activations import Codec, compile_codec, install_diagnostic_arithmetic


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0),-x))


def shared_perturbation(store,codec,layer,case_dir):
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    from mini_llm.ops.native_packed_cpu import packed_cpu_linear
    x=np.load(case_dir/'input.npy')
    def project(part,v):
        name=f'blk.{layer}.ffn_{part}_shexp.weight';d=store.tensors[name]
        w=PackedGGUFWeight(store.tensor_bytes(name),d.shape,d.quant_type)
        return packed_cpu_linear(codec(v,d.format.type_id)[None],w)[0]
    g,u=project('gate',x),project('up',x)
    h=(g*sigmoid(g))*u
    rh=np.fromfile(case_dir/'ref_shared_swiglu.f32',np.float32)
    ry=np.fromfile(case_dir/'ref_shared_down.f32',np.float32)
    down=store.tensors[f'blk.{layer}.ffn_down_shexp.weight']
    qr=codec(rh,down.format.type_id)
    delta=h-rh
    trials=[]
    for factor in [0.,.125,.5,1.,2.]:
        value=np.asarray(rh+np.float32(factor)*delta,np.float32)
        q=codec(value,down.format.type_id)
        trials.append(dict(factor=factor,input_relative_l2=metrics(value,rh)['relative_l2'],
            changed_rounded_elements=int(np.count_nonzero(q!=qr)),
            rounded_input=metrics(q,qr),output=metrics(project('down',value),ry)))
    return dict(layer=layer,format=down.quant_type,scope='shared expert only, same captured FFN input',trials=trials)


def child(vectors,codec_path):
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    codec=Codec(codec_path)
    install_diagnostic_arithmetic(codec)
    delta=np.load(vectors/'delta.npy')
    baseline=np.load(ROOT/'docs/qwen4exp/diagnostics_0222_v1/p0_l0_hc_ffn_read.npy')
    with Qwen4ExpGGUFModel.open(MODEL,max_length=32,cpu_workers=1) as model:
        original=model.blocks[0].moe
        results=[]
        for sign in [1,-1]:
            position=0
            def perturbed(hidden):
                if position==0:
                    np.testing.assert_array_equal(hidden,baseline)
                    return original(np.asarray(hidden+np.float32(sign)*delta,np.float32))
                return original(hidden)
            model.blocks[0].moe=perturbed
            state=model.create_generation_state(1,32)
            for position,token in enumerate([1,2]):
                value=model.step(token,state)[0]
                np.save(vectors/f'sign{sign}_logits_{position}.npy',value)
            results.append(dict(sign=sign,consumed_ids=[1,2],length=state.length,failed=state.failed))
            del state
            gc.collect()
        model.blocks[0].moe=original
    (vectors/'child.json').write_text(json.dumps(results))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--child',type=Path)
    parser.add_argument('--codec',type=Path)
    args=parser.parse_args()
    if args.child:
        child(args.child,args.codec)
        return
    from mini_llm.external.gguf_store import MultiShardTensorStore
    docs=ROOT/'docs/qwen4exp';output=docs/'0225_rounding_sensitivity_v1.json';vectors=docs/'diagnostics_0225_v1'
    if output.exists() or vectors.exists():raise FileExistsError('preserve prior artifacts')
    report=dict(scope='controlled diagnostic sensitivity, not proof of whole-model correctness',
        provenance=provenance(),production_files_modified=False,shared_experiments=[])
    vectors.mkdir()
    env=dict(os.environ,MINI_LLM_BACKEND='numpy',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='rounding-sensitivity-') as tmp:
        so=compile_codec(tmp);codec=Codec(so)
        with MultiShardTensorStore.open(MODEL) as store:
            for layer in [0,2]:
                report['shared_experiments'].append(shared_perturbation(store,codec,layer,docs/f'diagnostics_0224_v1/p0_l{layer}'))
        baseline=np.load(docs/'diagnostics_0222_v1/p0_l0_hc_ffn_read.npy')
        events=[json.loads(s) for s in (docs/'diagnostics_0220_v2/reference/events.jsonl').read_text().splitlines()]
        ids=associate(events)
        event=next(e for e in events if ids[(e['position'],e['tensor']['ptr'])]==(0,0,'hc_ffn_read'))
        reference=np.fromfile(docs/f'diagnostics_0220_v2/reference/eval_{event["eval_id"]}.f32',np.float32)
        direction=baseline-reference
        sweep=[];chosen=None
        q8=codec(baseline,18);bf16=codec(baseline,30)
        for factor in [.25,1.,4.,16.,64.]:
            delta=np.float32(factor)*direction
            changed=np.asarray(baseline+delta,np.float32)
            error=metrics(changed,baseline)['relative_l2']
            q8_rounded=codec(changed,18)
            q8_changes=int(np.count_nonzero(q8_rounded!=q8))
            q8_error=metrics(q8_rounded,q8)['relative_l2']
            bf16_changes=int(np.count_nonzero(codec(changed,30)!=bf16))
            sweep.append(dict(factor=factor,input_relative_l2=error,q8_decoded_changes=q8_changes,q8_relative_change=q8_error,bf16_changes=bf16_changes))
            # Decoded Q8 values can change only because their block scale moves;
            # do not describe that alone as an integer-code threshold crossing.
            if chosen is None and error<=1e-4 and (bf16_changes or q8_error>32*error):chosen=(factor,delta.copy())
        report['input_sweep']=sweep
        if chosen is None:
            report['full_forward_status']='not_run_no_bounded_threshold_crossing'
        else:
            factor,delta=chosen;np.save(vectors/'delta.npy',delta)
            report['intervention']=dict(layer=0,stage='MoE input only',position=0,factor=factor,
                delta_norm=float(np.linalg.norm(delta.astype(np.float64))),
                delta_maxabs=float(np.max(np.abs(delta))),delta_relative_l2=float(np.linalg.norm(delta.astype(np.float64))/np.linalg.norm(baseline.astype(np.float64))),
                direction='observed activation-matched minus reference FFN input',
                arithmetic='existing diagnostic activation-matched path; injection once, then autoregressive state continues on forced IDs [1,2]')
            report['run']=bounded([sys.executable,str(Path(__file__)),'--child',str(vectors),'--codec',str(so)],
                time.monotonic()+120,Path(tmp),'perturbation',env)
            report['child']=json.loads((vectors/'child.json').read_text())
            report['full_forward_status']='completed_two_signs'
            report['logits']=[]
            for sign in [1,-1]:
                for p in range(2):
                    a=np.load(vectors/f'sign{sign}_logits_{p}.npy')
                    b=np.load(docs/f'diagnostics_0222_v1/logits_{p}.npy')
                    r=np.fromfile(docs/f'diagnostics_0220_v2/reference/logits_{p}.f32',np.float32)
                    report['logits'].append(dict(sign=sign,position=p,versus_unperturbed_matched=logits_metrics(a,b),versus_reference=logits_metrics(a,r)))
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['full_forward_status'],intervention=report.get('intervention'),
        outputs=[dict(sign=r['sign'],position=r['position'],delta_relative_l2=r['versus_unperturbed_matched']['relative_l2'],reference_relative_l2=r['versus_reference']['relative_l2']) for r in report.get('logits',[])],
        shared=[dict(layer=s['layer'],trials=[dict(factor=t['factor'],input=t['input_relative_l2'],changed=t['changed_rounded_elements'],output=t['output']['relative_l2']) for t in s['trials']]) for s in report['shared_experiments']]),indent=2))

if __name__=='__main__':main()
