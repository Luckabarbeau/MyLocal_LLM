"""Alternating native task-grain experiment on real checkpoint mixed experts."""
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
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MINI_LLM_BACKEND']='numpy'
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from probe_0218_hc_identity import MODEL
    index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0304')
    executor=GGMLPackedCPUExecutor(q2_simd=True,iq3_gather=True,iq3xxs_gather=False,iq3xxs_pair=False,iq4_vnni=False)
    x=np.random.default_rng(304).normal(0,.1,(1,index.d_model)).astype(np.float32)
    factors=(1,2,3,4,6,8,12,16); rows=[]; seen=set()
    try:
        with GGMLPackedCPURowPool(executor,index.d_model,index.d_ff,19,scheduler='tasks',host_help=True,spin_us=50) as pool:
            for layer in range(index.n_layers):
                formats=tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type for p in ('gate','up','down'))
                if formats in seen: continue
                seen.add(formats)
                bank=[index.provider(ExpertKey(index.model_id,layer,int(e))) for e in np.linspace(0,index.n_experts-1,10,dtype=int)]
                ids=np.array([pool.register((layer,i),weights) for i,weights in enumerate(bank)],np.int32)
                for count in (2,5,10):
                    out=np.empty((count,index.d_model),np.float32)
                    def execute(): pool._submit_registered(x,ids[:count],out).result()
                    for _ in range(5): execute()
                    expected=out.copy(); samples={f:[] for f in factors}
                    for trial in range(8):
                        for factor in (factors if trial%2==0 else tuple(reversed(factors))):
                            executor._check(executor._bridge.mylocal_ggml_pool_set_task_factor(pool._handle,factor))
                            started=time.perf_counter_ns()
                            for _ in range(25): execute()
                            samples[factor].append((time.perf_counter_ns()-started)/25/1e6)
                            np.testing.assert_array_equal(out.view(np.uint32),expected.view(np.uint32))
                    medians={f:float(np.median(v)) for f,v in samples.items()}
                    row=dict(layer=layer,formats=formats,experts=count,exact=True,median_ms=medians,samples_ms=samples)
                    rows.append(row); print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
        (args.output/'result.json').write_text(json.dumps(dict(scope='Warm packed expert operator; not model throughput',workers=19,rows=rows),indent=2))
    finally:
        index.clear_source_cache(); index.store.close()


if __name__=='__main__': main()
