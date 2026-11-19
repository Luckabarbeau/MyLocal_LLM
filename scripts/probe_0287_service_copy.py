"""Isolate completion-copy cost: mapped versus device-local failure status."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    os.environ['MINI_LLM_BACKEND']='cupy'
    import cupy as cp
    from mini_llm.ops.native_expert_service import _mapped,_kernels,mapped_output
    owner=cp.cuda.alloc_pinned_memory(102400)
    host=np.frombuffer(owner,np.float32)
    host[:]=np.arange(host.size,dtype=np.float32)
    src=mapped_output(host,owner); dst=cp.empty_like(src)
    flags=_mapped(64)
    kernel=_kernels(cp.cuda.runtime.getDevice())[3]
    rows=[]
    for name,status in [('mapped',flags.device),('device',cp.zeros(1,cp.uint32))]:
        samples=[]
        for _ in range(5):
            a,b=cp.cuda.Event(),cp.cuda.Event(); a.record()
            for _ in range(200):
                kernel(((host.size+255)//256,),(256,),(src,dst,status,np.int32(host.size)))
            b.record(); b.synchronize()
            samples.append(cp.cuda.get_elapsed_time(a,b)/200)
        np.testing.assert_array_equal(dst.get(),host)
        rows.append(dict(status_storage=name,event_ms=samples))
    args.output.write_text(json.dumps(dict(rows=rows,bytes=host.nbytes),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
