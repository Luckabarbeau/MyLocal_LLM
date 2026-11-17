"""Identical-reference-input/state DeltaNet and QSA diagnostics, 16 fixed tokens.

The reference executes only decoder prefix layers0..3. No logits or full-model
parity are claimed; later layers cannot feed state back into this prefix.
"""
import argparse
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
from probe_0222_matched_activations import Codec, compile_codec


def recurrence_layout(raw):
    return raw.reshape(1,48,128,128).swapaxes(-1,-2)


def history_layout(raw):
    return raw.reshape(10240,3).T[None]


class Capture:
    def __init__(self,path):
        self.path=path
        self.events={}
        for line in (path/'events.jsonl').read_text().splitlines():
            e=json.loads(line);key=(e['position'],e['key'])
            if key in self.events:raise ValueError('duplicate reference object boundary')
            self.events[key]=e
    def get(self,p,key,count=None):
        e=self.events[(p,key)];t=e['tensor']
        if t['type'] not in (0,26):raise ValueError('capture type must be F32/I32')
        dtype=np.float32 if t['type']==0 else np.int32
        a=np.fromfile(self.path/e['file'],dtype)
        if len(a)!=int(np.prod(t['ne'])) or (count is not None and len(a)!=count):
            raise ValueError(f'capture shape mismatch: {p}, {key}, {t["ne"]}')
        if dtype==np.float32 and not np.isfinite(a).all():
            raise ValueError('nonfinite reference activation')
        return a


def replay(path,codec_path,*,force_reference_state=True):
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec,Qwen4ExpRecurrentMixerInference
    from mini_llm.external.qwen4exp_attention import QSAConfig,Qwen4ExpAttention
    from mini_llm.ops.native_packed_cpu import packed_cpu_linear
    c=Capture(path);codec=Codec(codec_path);rows=[];seen={};p=0
    def check(kind,stage,actual,reference,exact=False):
        a=np.asarray(actual,np.float32).reshape(-1);r=np.asarray(reference,np.float32).reshape(-1)
        row=dict(position=p,kind=kind,stage=stage,**metrics(a,r));rows.append(row)
        if exact:np.testing.assert_array_equal(a,r,err_msg=f'{kind}/{stage}/p{p}')
        else:np.testing.assert_allclose(a,r,atol=2e-5,rtol=3e-5,err_msg=f'{kind}/{stage}/p{p}')
    with MultiShardTensorStore.open(MODEL) as store:
        m=store.metadata
        def dense(name):
            t=store.tensors[name];raw=store.tensor_bytes(name)
            if t.quant_type=='F32':return raw.view('<f4').reshape(t.shape).copy()
            if t.quant_type=='BF16':return (raw.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(t.shape)
            raise ValueError('small control tensor requires F32/BF16')
        def project_ref(key,width):
            def call(x):
                result=c.get(p,key,width).copy()
                return result.reshape(x.shape[:-1]+(width,))
            return call
        def output(name,kind):
            w=store.packed_weight(name);type_id=store.tensors[name].format.type_id
            def call(x):
                seen[kind]=np.asarray(x,np.float32).copy()
                rounded=codec(x,type_id).reshape(-1,w.shape[-1])
                return packed_cpu_linear(rounded,w).reshape(x.shape[:-1]+(w.shape[0],))
            return call
        spec=Qwen4ExpMixerSpec.from_metadata(m)
        gdn=Qwen4ExpRecurrentMixerInference(spec,
            qkv=project_ref('linear_attn_qkv_mixed-0',10240),z=project_ref('z-0',6144),
            beta=project_ref('beta-0',48),alpha=project_ref('alpha-0',48),
            out=output('blk.0.ssm_out.weight','gdn'),
            conv_weight=dense('blk.0.ssm_conv1d.weight'),norm_weight=dense('blk.0.ssm_norm.weight'),
            ssm_a=dense('blk.0.ssm_a'),dt_bias=dense('blk.0.ssm_dt.bias'))
        cfg=QSAConfig(2560,epsilon=float(m['qwen4exp.attention.layer_norm_rms_epsilon']))
        qsa=Qwen4ExpAttention(cfg,
            query_gate=project_ref('Qcur_full-3',12288),key=project_ref('K_projection-3',512),
            value=project_ref('Vcur-3',512),index_query=project_ref('indexer_q_projection-3',512),
            index_key=project_ref('indexer_k_raw-3',128),output=output('blk.3.attn_output.weight','qsa'),
            gamma_q=dense('blk.3.attn_q_norm.weight'),gamma_k=dense('blk.3.attn_k_norm.weight'),
            gamma_index_q=dense('blk.3.indexer.q_norm.weight'),gamma_index_k=dense('blk.3.indexer.k_norm.weight'))
        gd=gdn.create_state(1);qs=qsa.create_state(32)
        for p in range(16):
            if force_reference_state:
                gd.delta.memory[:]=recurrence_layout(c.get(p,'state_predelta-0',786432))
                gd.conv.history[:]=history_layout(c.get(p,'conv_state_at-0',30720))
                gd.delta.length=gd.conv.length=p
            old_history=gd.conv.history.copy()
            result=gdn.step(c.get(p,'hc_mixed-0',2560)[None],gd)
            check('gdn','gated_normalized_output',seen['gdn'],c.get(p,'final_output-0',6144))
            check('gdn','new_state',gd.delta.memory,recurrence_layout(c.get(p,'new_state-0',786432)))
            expected=np.concatenate((old_history[:,1:],c.get(p,'linear_attn_qkv_mixed-0',10240).reshape(1,1,-1)),axis=1)
            check('gdn','convolution_update',gd.conv.history,expected,exact=True)
            if p<15:check('gdn','next_reference_history',gd.conv.history,history_layout(c.get(p+1,'conv_state_at-0',30720)),exact=True)
            projected=c.get(p,'linear_attn_out-0',2560)
            rows.append(dict(position=p,kind='gdn',stage='projected_output_descriptive',**metrics(result.reshape(-1),projected)))
            w=store.packed_weight('blk.0.ssm_out.weight')
            exact_input=c.get(p,'final_output-0',6144)[None]
            teacher=packed_cpu_linear(codec(exact_input,store.tensors['blk.0.ssm_out.weight'].format.type_id),w)
            check('gdn','projection_identical_input',teacher,projected)
            if force_reference_state:
                qs.length=p
                if p:
                    qs.keys[:p]=np.stack([c.get(j,'Kcur-3',512).reshape(2,256) for j in range(p)])
                    qs.values[:p]=np.stack([c.get(j,'Vcur-3',512).reshape(2,256) for j in range(p)])
                    qs.raw_index[:p]=np.stack([c.get(j,'indexer_k_raw-3',128) for j in range(p)])
            pooled=c.get(p,'indexer_k-3').reshape(-1,128)
            if force_reference_state and p//4:qs.pooled[:p//4]=pooled[:p//4]
            result,diag=qsa.step(c.get(p,'hc_mixed-3',2560),qs,position=p,diagnostics=True)
            check('qsa','gated_attention_output',seen['qsa'],c.get(p,'attn_gated-3',6144))
            check('qsa','key_update',qs.keys[p],c.get(p,'Kcur-3',512))
            check('qsa','value_update',qs.values[p],c.get(p,'Vcur-3',512),exact=True)
            check('qsa','raw_index_update',qs.raw_index[p],c.get(p,'indexer_k_raw-3',128),exact=True)
            complete=(p+1)//4
            if complete:check('qsa','completed_pools',qs.pooled[:complete],pooled[:complete])
            blocks=complete+bool((p+1)%4)
            check('qsa','indexer_scores',diag['block_scores'],c.get(p,'indexer_score-3')[:blocks])
            reference_ids=c.get(p,'indexer_top_k-3')
            valid=np.sort(reference_ids[reference_ids<p+1])
            np.testing.assert_array_equal(diag['selected_ids'],valid)
            rows.append(dict(position=p,kind='qsa',stage='selected_causal_cells',exact=True,
                count=len(valid),reference_physical_topk=len(reference_ids)))
            projected=c.get(p,'attn_output-3',2560)
            rows.append(dict(position=p,kind='qsa',stage='projected_output_descriptive',**metrics(result,projected)))
            w=store.packed_weight('blk.3.attn_output.weight')
            teacher=packed_cpu_linear(codec(c.get(p,'attn_gated-3',6144)[None],store.tensors['blk.3.attn_output.weight'].format.type_id),w)
            check('qsa','projection_identical_input',teacher,projected)
            assert gd.length==qs.length==p+1
    return dict(status='passed',positions=16,consumed_ids=list(range(1,17)),comparisons=rows,
        state_policy='identical reference pre-state every step' if force_reference_state else 'local continuous state from zero, no per-step resets',
        tolerances=dict(atol=2e-5,rtol=3e-5,dtype='F32',projected_output='Descriptive when input differs; identical-input projection is asserted'),
        scope='Real checkpoint blocks, reference projection operands; state_policy specifies per-step reference or continuous local state; no full-model equivalence claim',
        qsa_selection_scope='At <=16 positions reference padded cache includes masked cells; compare selected causal cells, all real cells fit default2051 budget')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',type=Path);parser.add_argument('--codec',type=Path);args=parser.parse_args()
    if args.replay:
        result=replay(args.replay,args.codec)
        with (args.replay/'replay.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False)
        return
    docs=ROOT/'docs/qwen4exp';report_path=docs/'0226_identical_block_states_v1.json';vectors=docs/'diagnostics_0226_v1'
    if report_path.exists() or vectors.exists():raise FileExistsError('preserve prior artifacts')
    report=dict(provenance=provenance(),production_files_modified=False,reference_math_modified=False,
        context=32,threads=1,kv_dtype='F32',reference_execution='decoder prefix0..3 only; no logits requested from unfinished graph')
    env=dict(os.environ,MINI_LLM_BACKEND='numpy',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    vectors.mkdir();deadline=time.monotonic()+120
    try:
        with tempfile.TemporaryDirectory(prefix='identical-block-states-') as tmp:
            tmp=Path(tmp);binary=tmp/'blocks';lib=COPY/'build-cpu/bin'
            report['compile']=bounded(['c++','-std=c++17','-O2',str(ROOT/'tests/fixtures/qwen4exp_block_state_reference.cpp'),
                '-I'+str(COPY/'include'),'-I'+str(COPY/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
                '-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)],deadline,tmp,'compile',env)
            report['reference_run']=bounded([str(binary),str(MODEL),str(vectors)],deadline,tmp,'reference',env)
            codec=compile_codec(tmp)
            report['replay_run']=bounded([sys.executable,str(Path(__file__)),'--replay',str(vectors),'--codec',str(codec)],deadline,tmp,'replay',env)
        report['result']=json.loads((vectors/'replay.json').read_text());report['status']='passed'
    except Exception as e:
        report['status']='blocked';report['error']=str(e)
        if hasattr(e,'result'):report['failure']=e.result
    with report_path.open('x') as out:json.dump(report,out,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('compile','reference_run','replay_run','result')},indent=2))
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
