"""Actual packed checkpoint GPU expert A/B with unchanged Q8_1 arithmetic."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MINI_LLM_BACKEND']='cupy'
    import cupy as cp
    from types import SimpleNamespace
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.ops.quantized_linear import packed_to_device
    from mini_llm.ops.mixed_q8_gpu import GroupedQ8Plan
    from mini_llm.ops.inference_graph import CapturedArrayInference
    from probe_0218_hc_identity import MODEL
    index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0307')
    x=cp.asarray(np.random.default_rng(307).normal(0,.1,(1,index.d_model)),dtype=cp.float32)
    rows=[]; seen=set()
    try:
        for layer in range(index.n_layers):
            formats=tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type for p in ('gate','up','down'))
            if formats in seen or formats[0] not in ('IQ2_S','IQ3_S','IQ3_XXS'): continue
            seen.add(formats)
            bank=[tuple(packed_to_device(w) for w in index.provider(ExpertKey(index.model_id,layer,int(e))))
                  for e in np.linspace(0,index.n_experts-1,10,dtype=int)]
            pointers=np.asarray([[w.data.data.ptr for w in weights] for weights in bank],np.uint64)
            for hits in (2,5,10):
                selected=pointers.copy(); selected[hits:]=0
                plan=SimpleNamespace(device=x.device.id,d=index.d_model,f=index.d_ff,k=10,formats=formats,
                    resident_row_bytes=tuple(w.nbytes//w.shape[0] for w in bank[0]),
                    selected=cp.asarray(selected),hidden=cp.empty((10,index.d_ff),cp.float32),
                    terms=cp.empty((10,index.d_model),cp.float32),routes=cp.linspace(.01,.1,10,dtype=cp.float32))
                graphs={}; samples={key:[] for key in ('global','shared')}; outputs={}
                try:
                    for key,value in [('global','0'),('shared','1')]:
                        os.environ['MINI_LLM_Q8_SHARED_GRID']=value
                        operator=GroupedQ8Plan(plan)
                        def enqueue(value,unused,operator=operator):
                            operator.forward(value)
                            return plan.terms
                        graphs[key]=CapturedArrayInference(enqueue,x,(),pool_limit_bytes=4*1024**2)
                        outputs[key]=graphs[key].forward(x,()).get()
                    np.testing.assert_array_equal(outputs['global'].view(np.uint32),outputs['shared'].view(np.uint32))
                    for trial in range(10):
                        for key in (('global','shared') if trial%2==0 else ('shared','global')):
                            start,end=cp.cuda.Event(),cp.cuda.Event()
                            start.record()
                            for _ in range(100): graphs[key].forward(x,())
                            end.record(); end.synchronize()
                            samples[key].append(cp.cuda.get_elapsed_time(start,end)/100)
                    medians={key:float(np.median(v)) for key,v in samples.items()}
                    row=dict(layer=layer,formats=formats,hits=hits,exact=True,median_ms=medians,
                        speedup=medians['global']/medians['shared'],samples_ms=samples)
                    rows.append(row); print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
                finally:
                    for graph in graphs.values(): graph.close()
            del bank
            cp.cuda.get_current_stream().synchronize()
        (args.output/'result.json').write_text(json.dumps(dict(scope='Actual resident packed GPU Q8 experts, not model throughput',rows=rows),indent=2))
    finally:
        index.clear_source_cache(); index.store.close()


if __name__=='__main__': main()
