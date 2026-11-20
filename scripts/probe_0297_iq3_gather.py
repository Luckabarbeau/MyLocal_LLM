"""Actual packed CPU kernel A/B, one shared registered row pool."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=19)
    parser.add_argument('--kernel',choices=('iq3_gather','iq4_vnni','iq3xxs_gather','iq3xxs_pair'),default='iq3_gather')
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    if not 1<=args.workers<=23: parser.error('workers1..23 only')
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MINI_LLM_BACKEND']='numpy'
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from probe_0218_hc_identity import MODEL
    index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0297')
    base=GGMLPackedCPUExecutor(q2_simd=True,iq3_gather=False,iq4_vnni=False,iq3xxs_gather=False,iq3xxs_pair=False)
    fast=GGMLPackedCPUExecutor(q2_simd=True,iq3_gather=args.kernel=='iq3_gather',
                              iq4_vnni=args.kernel=='iq4_vnni',iq3xxs_gather=args.kernel=='iq3xxs_gather',
                              iq3xxs_pair=args.kernel=='iq3xxs_pair')
    target_format={'iq3_gather':'IQ3_S','iq4_vnni':'IQ4_NL','iq3xxs_gather':'IQ3_XXS','iq3xxs_pair':'IQ3_XXS'}[args.kernel]
    label='pair' if args.kernel=='iq3xxs_pair' else 'vnni' if args.kernel=='iq4_vnni' else 'gather'
    x=np.random.default_rng(297).normal(0,.1,(1,index.d_model)).astype(np.float32)
    rows=[]; seen=set(); inventory=[]
    try:
        with GGMLPackedCPURowPool(base,index.d_model,index.d_ff,args.workers,
                                 scheduler='tasks',host_help=True,spin_us=50) as pool:
            for layer in range(index.n_layers):
                formats=tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type
                              for p in ('gate','up','down'))
                inventory.append(dict(layer=layer,formats=formats))
                eligible=(formats[2]=='IQ4_NL' if args.kernel=='iq4_vnni' else formats[:2]==(target_format,target_format))
                if not eligible or formats in seen: continue
                seen.add(formats)
                experts=[index.provider(ExpertKey(index.model_id,layer,int(e)))
                         for e in np.linspace(0,index.n_experts-1,10,dtype=int)]
                ids={}
                # Setup only: descriptors capture their dot function addresses.
                # Restore the executor before submitting any consumer. Thus both
                # variants share identical workers/spin/scratch and canonical bytes.
                original=base._functions[target_format]
                try:
                    for kind,executor in (('ggml',base),(label,fast)):
                        base._functions[target_format]=original if kind=='ggml' else fast._functions[target_format]
                        ids[kind]=np.asarray([pool.register((layer,kind,i),ws)
                                             for i,ws in enumerate(experts)],np.int32)
                finally: base._functions[target_format]=original
                for count in (1,2,5,10):
                    output={kind:np.empty((count,index.d_model),np.float32) for kind in ids}
                    def execute(kind):
                        pool._submit_registered(x,ids[kind][:count],output[kind]).result()
                    for _ in range(5): execute('ggml'); execute(label)
                    np.testing.assert_array_equal(output['ggml'].view(np.uint32),output[label].view(np.uint32))
                    samples={kind:[] for kind in ids}
                    for trial in range(10):
                        order=('ggml',label) if trial%2==0 else (label,'ggml')
                        for kind in order:
                            started=time.perf_counter_ns()
                            for _ in range(40): execute(kind)
                            samples[kind].append((time.perf_counter_ns()-started)/40/1e6)
                    np.testing.assert_array_equal(output['ggml'].view(np.uint32),output[label].view(np.uint32))
                    medians={kind:float(np.median(v)) for kind,v in samples.items()}
                    row=dict(layer=layer,formats=formats,experts=count,bit_exact=True,
                             median_ms=medians,samples_ms=samples,speedup=medians['ggml']/medians[label])
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
        (args.output/'result.json').write_text(json.dumps(dict(scope='Warm actual CPU expert operators, not model throughput',
            workers=args.workers,host_help=True,spin_us=50,inventory=inventory,kernel=args.kernel,
            baseline=base.provenance,candidate=fast.provenance,rows=rows),indent=2,allow_nan=False))
    finally:
        index.clear_source_cache(); index.store.close()


if __name__=='__main__': main()
