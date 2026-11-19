"""Real same-input CPU/VRAM/mapped-host expert cost, bounded operators only."""
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
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=19)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    os.environ['MINI_LLM_BACKEND']='cupy'
    import cupy as cp
    from mini_llm.ops.mapped_packed_experts import MappedPackedExpertTier
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor,GGMLPackedCPURowPool
    from mini_llm.ops.quantized_linear import PackedExpertFFNInference,packed_to_device,DevicePackedGGUFWeight
    from mini_llm.ops.mixed_q8_gpu import GroupedQ8Plan
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from probe_0218_hc_identity import MODEL
    index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0286')
    cpu=GGMLPackedCPUExecutor()
    x=np.random.default_rng(286).normal(0,.1,(1,index.d_model)).astype(np.float32)
    gx=cp.asarray(x)
    deadline=time.monotonic()+150
    def timed(call,gpu=False):
        for _ in range(3): call()
        cp.cuda.get_current_stream().synchronize()
        samples=[]
        for _ in range(5):
            if time.monotonic()>deadline: raise TimeoutError('Operator probe time limit')
            a,b=cp.cuda.Event(),cp.cuda.Event(); a.record(); start=time.perf_counter()
            for _ in range(20): call()
            b.record(); b.synchronize()
            samples.append(dict(wall_ms=(time.perf_counter()-start)*50,
                event_ms=cp.cuda.get_elapsed_time(a,b)/20 if gpu else None))
        return samples
    def plan(experts,source,formats):
        count=len(experts)
        p=SimpleNamespace(d=index.d_model,f=index.d_ff,k=count,device=cp.cuda.runtime.getDevice(),
            formats=formats,resident_row_bytes=tuple(w.nbytes//w.shape[0] for w in source[0]),
            selected=cp.asarray([[w.data.data.ptr for w in (e.gate,e.up,e.down)] for e in experts],cp.uint64),
            hidden=cp.empty((count,index.d_ff),cp.float32),terms=cp.empty((count,index.d_model),cp.float32),
            routes=cp.ones(count,cp.float32))
        return p,GroupedQ8Plan(p)
    rows=[]; seen=set()
    try:
        with GGMLPackedCPURowPool(cpu,index.d_model,index.d_ff,args.workers,host_help=True,
                                 scheduler='tasks',spin_us=50) as pool:
            for layer in range(index.n_layers):
                formats=tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type for p in ('gate','up','down'))
                if formats in seen: continue
                seen.add(formats)
                sources=[index.provider(ExpertKey(index.model_id,layer,int(e))) for e in np.linspace(0,511,10,dtype=int)]
                # 32MiB ceiling, a complete ten-expert tier, no full bank copy.
                tier=MappedPackedExpertTier(32*1024**2)
                remote=[tier.add(i,source) for i,source in enumerate(sources)]
                local=[PackedExpertFFNInference(*(packed_to_device(w) for w in s)) for s in sources]
                staging=cp.empty(tier.used_bytes,cp.uint8)
                offsets=[e.gate.data.data.ptr-tier.device_pointer for e in remote]
                staged=[]
                for expert in remote:
                    projections=[]
                    for w in (expert.gate,expert.up,expert.down):
                        off=w.data.data.ptr-tier.device_pointer
                        projections.append(DevicePackedGGUFWeight(staging[off:off+w.nbytes],w.shape,w.quant_type))
                    staged.append(PackedExpertFFNInference(*projections))
                for count in (1,2,3,5,10):
                    rp,r=plan(remote[:count],sources,formats)
                    lp,l=plan(local[:count],sources,formats)
                    sp,s=plan(staged[:count],sources,formats)
                    def stage():
                        for i in range(count):
                            cp.cuda.runtime.memcpyAsync(staging.data.ptr+offsets[i],tier.host.ctypes.data+offsets[i],
                                sum(w.nbytes for w in sources[i]),cp.cuda.runtime.memcpyHostToDevice,cp.cuda.get_current_stream().ptr)
                        s.forward(gx)
                    stage()
                    r.forward(gx); l.forward(gx)
                    np.testing.assert_array_equal(rp.terms.get(),lp.terms.get())
                    np.testing.assert_array_equal(sp.terms.get(),lp.terms.get())
                    cpu_times=timed(lambda:pool.submit(x,sources[:count]).result())
                    local_times=timed(lambda:l.forward(gx),True)
                    remote_times=timed(lambda:r.forward(gx),True)
                    staged_times=timed(stage,True)
                    row=dict(layer=layer,formats=formats,experts=count,remote_vs_vram_bit_exact=True,
                        cpu=cpu_times,vram=local_times,mapped_host=remote_times,staged=staged_times)
                    rows.append(row)
                    print(layer,formats,count,'CPU ms',round(np.median([s['wall_ms'] for s in cpu_times]),4),
                        'VRAM ms',round(np.median([s['event_ms'] for s in local_times]),4),
                        'PCIe ms',round(np.median([s['event_ms'] for s in remote_times]),4),
                                                'staged ms',round(np.median([s['event_ms'] for s in staged_times]),4),flush=True)
                    del rp,r,lp,l,sp,s
                cp.cuda.get_current_stream().synchronize()
                tier.close(); del tier,remote,local,sources,staged,staging
                gc.collect()
    finally:
        index.clear_source_cache(); index.store.close()
    args.output.write_text(json.dumps(dict(scope='Operator cost only, not integrated model throughput',
        workers=args.workers,spin_us=50,host_help=True,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
