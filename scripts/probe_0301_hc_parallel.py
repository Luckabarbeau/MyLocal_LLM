"""Bounded alternating captured HC read A/B; synthetic weights, not model rate."""
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
    import cupy as cp
    from mini_llm.ops.hyperconnection_read_gpu import read_gpu
    from mini_llm.ops.inference_graph import CapturedArrayInference
    rng=np.random.default_rng(301)
    d,rank=2560,320
    h=cp.asarray(rng.normal(size=(1,4,d)),dtype=cp.float32)
    gamma=cp.ones((4,d),cp.float32)
    weights=[]
    for shape in [(rank,4*d),(4*d,rank),(4,4*d)]:
        bits=rng.normal(0,.04,shape).astype(np.float32).view(np.uint32)
        bits=bits+np.uint32(0x7fff)+((bits>>16)&1)
        weights.append(cp.asarray((bits>>16).astype(np.uint16)))
    graphs={}; outputs={}; samples={v:[] for v in ('serial','parallel')}
    try:
        for key,value in [('serial','0'),('parallel','1')]:
            os.environ['MINI_LLM_HC_PARALLEL_READ']=value
            graphs[key]=CapturedArrayInference(lambda x,unused:read_gpu(x,gamma,*weights,1e-5)[0],h,())
            outputs[key]=cp.asnumpy(graphs[key].forward(h,()))
        np.testing.assert_array_equal(outputs['serial'].view(np.uint32),outputs['parallel'].view(np.uint32))
        for trial in range(12):
            for key in (('serial','parallel') if trial%2==0 else ('parallel','serial')):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                start.record()
                for _ in range(200): graphs[key].forward(h,())
                end.record(); end.synchronize()
                samples[key].append(cp.cuda.get_elapsed_time(start,end)/200)
        medians={k:float(np.median(v)) for k,v in samples.items()}
        result=dict(scope='Captured synthetic BF16 HC read, not full-model throughput',d=d,rank=rank,
                    exact=True,median_ms=medians,samples_ms=samples,speedup=medians['serial']/medians['parallel'])
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        for graph in graphs.values(): graph.close()


if __name__=='__main__': main()
