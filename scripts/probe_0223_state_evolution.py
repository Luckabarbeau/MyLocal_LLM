"""Sampled fixed-prefix state comparison; no production arithmetic changes."""
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
from probe_0220_layer_evolution import metrics
from probe_0210_qwen4exp_reference import logits_metrics


def snapshot(state,output,p):
    np.save(output/f'p{p}_state.npy',state.recurrent[0].delta.memory)
    np.save(output/f'p{p}_conv0.npy',state.recurrent[0].conv.history)
    np.save(output/f'p{p}_conv1.npy',state.recurrent[1].conv.history)
    np.save(output/f'p{p}_ple_history.npy',state.ple[1].normalized_history)


def child(output):
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    with Qwen4ExpGGUFModel.open(MODEL,max_length=32,cpu_workers=1) as model:
        state=model.create_generation_state(1,32)
        for p,token in enumerate([1,2,3,4,5]):
            snapshot(state,output,p)
            np.save(output/f'p{p}_ple_indices.npy',np.array(model.blocks[1].ple.config.rows(token,state.ple[1].predecessors),np.int32))
            logits=model.step(token,state)[0]
            np.save(output/f'logits_{p}.npy',logits)
            qsa=state.qsa[3]
            for label,value in [('Kcur-3',qsa.keys[p]),('Vcur-3',qsa.values[p]),('indexer_k_raw-3',qsa.raw_index[p])]:
                np.save(output/f'p{p}_{label}.npy',value)
        del state
        gc.collect()
        fresh=model.create_generation_state(1,32)
        logits=model.prefill([1,2,3,4],fresh)[0]
        np.testing.assert_array_equal(logits,np.load(output/'logits_3.npy'))
        for label,value in [('state',fresh.recurrent[0].delta.memory),('conv0',fresh.recurrent[0].conv.history),
                            ('conv1',fresh.recurrent[1].conv.history),('ple_history',fresh.ple[1].normalized_history)]:
            np.testing.assert_array_equal(value,np.load(output/f'p4_{label}.npy'))
        (output/'self_consistency.json').write_text(json.dumps(dict(prefix=[1,2,3,4],logits_and_sampled_states_bit_exact=True)))


def canonical(reference,event):
    name=event['tensor']['name'];ne=event['tensor']['ne'];a=np.fromfile(reference/event['file'],np.float32)
    if name=='state_predelta-0':
        if ne!=[128,128,48,1]:raise ValueError('recurrence layout mismatch')
        return 'state',a.reshape(1,48,128,128).swapaxes(-1,-2)
    if name.startswith('conv_state_at-'):
        if ne not in ([3,10240,1,1],[9,10240,1,1]):raise ValueError('convolution layout mismatch')
        if ne[0]==9:return 'ple_history',a.reshape(10240,9).T.reshape(9,4,2560)
        return 'conv0' if name.endswith('-0') else 'conv1',a.reshape(10240,3).T[None]
    if name in ('Kcur-3','Vcur-3'):
        if ne!=[256,2,1,1]:raise ValueError('KV projection layout mismatch')
        return name,a.reshape(2,256)
    if name=='indexer_k_raw-3':
        if ne!=[128,1,1,1]:raise ValueError('indexer layout mismatch')
        return name,a.reshape(128)
    if name=='ple_embd': return None,a
    raise ValueError('unrecognized state boundary')


def verified_events(events):
    """Drop only a raw V projection proven to feed its captured head reshape."""
    retained=[]
    for e in events:
        t=e['tensor']
        if t['name']=='Vcur-3' and t['ne']==[512,1,1,1]:
            partners=[q for q in events if q['position']==e['position'] and q['tensor']['name']=='Vcur-3'
                      and q['tensor']['ne']==[256,2,1,1] and t['ptr'] in q['tensor']['src']]
            if len(partners)!=1:raise ValueError('unverified V projection/reshape association')
            continue
        retained.append(e)
    return retained


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path)
    parser.add_argument('--reanalyze',action='store_true');args=parser.parse_args()
    if args.child:child(args.child);return
    docs=ROOT/'docs/qwen4exp';result=docs/('0223_state_evolution_v2.json' if args.reanalyze else '0223_state_evolution_v1.json');vectors=docs/'diagnostics_0223_v1'
    if result.exists() or (vectors.exists() and not args.reanalyze):raise FileExistsError('preserve prior state captures')
    report=dict(scope='sampled actual states; fixed prefixes 1,2,4, fifth sentinel exposes final pre-state',
        provenance=provenance(),tokens=[1,2,3,4,5],sampled_layers=[0,1,3],full_state_equivalence_claim=False,
        production_files_modified=False,context=32,threads=1,kv_dtype='F32',comparisons=[])
    if not args.reanalyze: vectors.mkdir()
    ref=vectors/'reference';actual=vectors/'production'
    if not args.reanalyze: ref.mkdir();actual.mkdir()
    report['retained_capture_reanalysis_only']=args.reanalyze
    env=dict(os.environ,MINI_LLM_BACKEND='numpy',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    deadline=time.monotonic()+120
    try:
        if not args.reanalyze:
            with tempfile.TemporaryDirectory(prefix='state-evolution-') as tmp:
                tmp=Path(tmp);binary=tmp/'states';lib=COPY/'build-cpu/bin'
                report['compile']=bounded(['c++','-std=c++17','-O2',str(ROOT/'tests/fixtures/qwen4exp_state_reference.cpp'),
                    '-I'+str(COPY/'include'),'-I'+str(COPY/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
                    '-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)],deadline,tmp,'compile',env)
                report['reference_run']=bounded([str(binary),str(MODEL),str(ref)],deadline,tmp,'reference',env)
                report['production_run']=bounded([sys.executable,str(Path(__file__)),'--child',str(actual)],deadline,tmp,'production',env)
        events=verified_events([json.loads(line) for line in (ref/'events.jsonl').read_text().splitlines()])
        observed=set()
        for event in events:
            p=event['position'];name,reference=canonical(ref,event)
            if name is None:continue
            key=(p,name)
            if key in observed:raise ValueError('duplicate sampled state identity')
            observed.add(key)
            ours=np.load(actual/f'p{p}_{name}.npy')
            if ours.shape!=reference.shape:raise ValueError('state shape mismatch')
            report['comparisons'].append(dict(position=p,state=name,tensor=event['tensor'],**metrics(ours.reshape(-1),reference.reshape(-1))))
        expected={(p,name) for p in range(5) for name in ['state','conv0','conv1','ple_history','Kcur-3','Vcur-3','indexer_k_raw-3']}
        if observed!=expected:raise ValueError(f'missing state boundaries: {expected-observed}')
        report['ple_row_indices']=[dict(position=p,exact=bool(np.array_equal(np.load(actual/f'p{p}_ple_indices.npy'),np.fromfile(ref/f'p{p}_ple_indices.i32',np.int32)))) for p in range(5)]
        if not all(r['exact'] for r in report['ple_row_indices']):raise ValueError('PLE selected rows mismatch')
        report['logits']=[dict(prefix_length=p+1,**logits_metrics(np.load(actual/f'logits_{p}.npy'),np.fromfile(ref/f'logits_{p}.f32',np.float32))) for p in [0,1,3]]
        report['self_consistency']=json.loads((actual/'self_consistency.json').read_text())
        report['status']='sampled_states_compared_no_equivalence_claim'
    except Exception as e:
        report['status']='blocked';report['error']=str(e)
        if hasattr(e,'result'):report['failure']=e.result
    with result.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','compile','reference_run','production_run')},indent=2))
    if report['status']=='blocked':raise SystemExit(1)

if __name__=='__main__':main()
