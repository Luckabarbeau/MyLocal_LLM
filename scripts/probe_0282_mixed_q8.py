"""Bounded same-input real packed GPU Q8 operator audit/timing; not throughput."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    os.environ.update(MINI_LLM_BACKEND='cupy',MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
                      MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp')
    import cupy as cp
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops import mixed_q8_gpu as gpu,q8_1_gpu as q8
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    from mini_llm.ops.quantized_linear import packed_to_device,quantized_linear,PackedExpertFFNInference
    from mini_llm.ops.grouped_quantized_experts import grouped_quantized_experts_forward
    from mini_llm.ops.expert_residency import ExpertKey
    from probe_0218_hc_identity import MODEL
    from test_0281_mixed_q8_gpu import reference
    deadline=time.monotonic()+150
    def timed(call):
        for _ in range(3): call()
        samples=[]
        for _ in range(5):
            if time.monotonic()>deadline: raise TimeoutError('Operator audit time budget')
            cp.cuda.get_current_stream().synchronize()
            a,b=cp.cuda.Event(),cp.cuda.Event()
            a.record(); start=time.perf_counter()
            for _ in range(20): call()
            b.record(); b.synchronize()
            samples.append(dict(event_ms=cp.cuda.get_elapsed_time(a,b)/20,
                                wall_ms=(time.perf_counter()-start)*50))
        return samples
    index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0282')
    report=dict(scope='Same-input real packed operators, not full-model speedup',linear=[],experts=[])
    try:
        seen=set()
        for name,desc in index.store.tensors.items():
            if len(desc.shape)!=2 or desc.quant_type not in gpu.FORMATS or desc.quant_type in seen:
                continue
            # Avoid embedding/PLE/output tables: use bounded projection matrices only.
            if not name.startswith('blk.'): continue
            seen.add(desc.quant_type)
            n,k=desc.shape
            small=PackedGGUFWeight(index.store.rows(name,0,min(n,32)),(min(n,32),k),desc.quant_type)
            w=packed_to_device(small)
            x=cp.asarray(np.random.default_rng(282).normal(0,.1,(1,k)).astype(np.float32))
            p=q8.quantize_q8_1(x)
            actual=gpu.linear(x,w).get()
            expected=reference(p,w).get()
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            raw=p.get().reshape(-1,36)
            decoded=(raw[:,:2].copy().view('<f2').astype(np.float64)*raw[:,4:].copy().view(np.int8)).reshape(1,k)
            baseline=decoded@small.dequantize_reference().astype(np.float64).T
            error=actual.astype(np.float64)-baseline
            os.environ['MINI_LLM_LINEAR_GPU_Q8']='0'
            fp32=quantized_linear(x,w).get()
            delta=actual.astype(np.float64)-fp32
            # Full packed projection only; no full floating-point expansion.
            full=packed_to_device(index.store.packed_weight(name))
            old=timed(lambda:quantized_linear(x,full))
            new=timed(lambda:gpu.linear(x,full))
            report['linear'].append(dict(name=name,format=desc.quant_type,shape=desc.shape,
                oracle_rows=min(n,32),strata_bit_exact=True,
                fp64_q8_max_abs=float(np.abs(error).max()),
                fp64_q8_relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(baseline),1e-30)),
                q8_vs_fp32_max_abs=float(np.abs(delta).max()),
                q8_vs_fp32_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(fp32),1e-30)),
                fp32=old,q8=new))
            del full,w,small
            gc.collect()
        seen=set()
        x=cp.asarray(np.random.default_rng(2821).normal(0,.1,(1,index.d_model)).astype(np.float32))
        routes=cp.asarray(np.arange(1,11,dtype=np.float32)/55)
        for layer in range(index.n_layers):
            formats=tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type for p in ('gate','up','down'))
            if formats in seen: continue
            seen.add(formats)
            experts=[PackedExpertFFNInference(*(packed_to_device(w) for w in index.provider(
                ExpertKey(index.model_id,layer,int(e))))) for e in np.linspace(0,511,10,dtype=int)]
            plan=SimpleNamespace(d=index.d_model,f=index.d_ff,k=10,device=cp.cuda.runtime.getDevice(),formats=formats,
                resident_row_bytes=tuple(w.nbytes//w.shape[0] for w in
                    index.provider(ExpertKey(index.model_id,layer,0))),
                selected=cp.asarray([[w.data.data.ptr for w in (e.gate,e.up,e.down)] for e in experts],dtype=cp.uint64),
                hidden=cp.empty((10,index.d_ff),cp.float32),routes=routes,
                terms=cp.empty((10,index.d_model),cp.float32))
            candidate=gpu.GroupedQ8Plan(plan)
            # Include the separate deterministic combine in both timing paths.
            def call():
                candidate.forward(x)
                return cp.sum(plan.terms,axis=0,keepdims=True)
            old=timed(lambda:grouped_quantized_experts_forward(x,experts,routes))
            new=timed(call)
            report['experts'].append(dict(layer=layer,formats=formats,fp32=old,q8=new))
            del candidate,plan,experts
            gc.collect()
        report['gpu_pool_reserved_bytes']=cp.get_default_memory_pool().total_bytes()
    finally:
        index.clear_source_cache()
        index.store.close()
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    for kind in ('linear','experts'):
        for r in report[kind]:
            old=np.median([s['event_ms'] for s in r['fp32']]); new=np.median([s['event_ms'] for s in r['q8']])
            print(kind,r.get('name',r.get('layer')),r.get('format',r.get('formats')),
                  'FP32 ms',round(old,4),'Q8 ms',round(new,4),'ratio',round(old/new,3))


if __name__=='__main__': main()
