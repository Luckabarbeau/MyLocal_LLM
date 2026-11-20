"""Read-only Linux mincore census; no checkpoint payload reads, warming or pinning."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def resident_bytes(address,length,*,chunk_pages=8192):
    """Return byte-weighted resident payload, including partial edge pages.

    Caller owns the live mapping; this function never dereferences its payload.
    Linux may conservatively return all-resident bits for unowned/unwritable files;
    the checkpoint CLI refuses that ambiguous ownership case.
    """
    if (isinstance(address,bool) or not isinstance(address,int) or address<=0 or
        isinstance(length,bool) or not isinstance(length,int) or length<0 or
        isinstance(chunk_pages,bool) or not isinstance(chunk_pages,int) or not 1<=chunk_pages<=8192):
        raise ValueError('Require a valid owned mapping range and bounded chunk size')
    if not length: return 0
    page=os.sysconf('SC_PAGE_SIZE')
    first=address//page*page; end=address+length
    stop=(end+page-1)//page*page
    libc=C.CDLL(None,use_errno=True)
    mincore=libc.mincore
    mincore.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p]; mincore.restype=C.c_int
    count=0
    for start in range(first,stop,chunk_pages*page):
        size=min(chunk_pages*page,stop-start); pages=size//page
        vector=(C.c_ubyte*pages)()
        if mincore(start,size,vector):
            code=C.get_errno(); raise OSError(code,os.strerror(code))
        flags=np.ctypeslib.as_array(vector)
        count+=int(np.count_nonzero(flags&1))*page
        if start==first and flags[0]&1: count-=address-first
        if start+size==stop and flags[-1]&1: count-=stop-end
    return count


def tensor_group(name):
    if '_exps.weight' in name: return 'routed_experts'
    if name=='per_layer_token_embd.weight' or 'ple' in name: return 'ple'
    return 'other'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    os.environ['MINI_LLM_BACKEND']='numpy'
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from probe_0218_hc_identity import MODEL
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); index=Qwen4ExpGGUFIndex.open(MODEL,model_id='0308')
    try:
        shards=[]
        for path in index.store.paths:
            stat=path.stat()
            if stat.st_uid!=os.geteuid():
                raise RuntimeError('Refusing potentially redacted mincore bits for an unowned checkpoint')
            shards.append(dict(path=str(path),bytes=stat.st_size,uid=stat.st_uid,mtime_ns=stat.st_mtime_ns))
        before=Path('/proc/self/io').read_text()
        rows=[]; groups={}
        for name,desc in index.store.tensors.items():
            owner=index.store._owners[desc.shard_index]
            if desc.data_offset<0 or desc.data_offset+desc.nbytes>owner.nbytes:
                raise ValueError('Tensor extends beyond canonical mapping')
            resident=resident_bytes(int(owner.ctypes.data)+desc.data_offset,int(desc.nbytes))
            group=tensor_group(name)
            row=dict(name=name,group=group,format=desc.quant_type,bytes=desc.nbytes,resident_bytes=resident,
                     nonresident_bytes=desc.nbytes-resident,resident_fraction=resident/max(desc.nbytes,1))
            rows.append(row)
            totals=groups.setdefault(group,dict(bytes=0,resident_bytes=0,nonresident_bytes=0))
            for key in totals: totals[key]+=row[key]
        result=dict(scope='Snapshot of file page residency, not per-token I/O attribution or unique physical RAM',
            method='Linux mincore, owned readonly GGUF mappings, <=8192 page bits per syscall',
            payload_reads=False,checkpoint_writes=False,elapsed_seconds=time.monotonic()-started,
            shards=shards,groups=groups,tensors=rows,io_before_census=before,io_after_census=Path('/proc/self/io').read_text())
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(groups,indent=2))
    finally:
        index.clear_source_cache(); index.store.close()


if __name__=='__main__': main()
