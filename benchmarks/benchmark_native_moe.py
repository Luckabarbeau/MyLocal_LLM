"""Whole-MoE host-to-host latency, not kernel time or full-model throughput.

Run from the repository root with MINI_LLM_BACKEND=cupy and explicit native/GGML
libraries. Fixed inputs come from actual embedding -> attention HC -> first
QSA/GDN token -> FFN HC read. JSON embeds their exact FP32 bytes for replay.
"""
import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from mini_llm.backend import BACKEND_NAME
from mini_llm.native import Runtime
from mini_llm.ops.ggml_packed_cpu import _load_bridge
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, QSAConfig
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_native_engine_chaining import checkpoint_upload, REC_STEMS, STEMS, NORMS, NAMES
from test_native_engine_recurrent import reference as recurrent_reference
from native_moe_checkpoint_helpers import CheckpointMoE, KINDS, POLICY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_hidden(cp,path,layer,token):
    """Setup only. All native attention-upload owners are released before timing."""
    r=Runtime()
    with ExitStack() as stack:
        e=stack.enter_context(r.engine(budget_bytes=511*2**20)); store=stack.enter_context(r.gguf(path))
        eps=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']; refs=[]; provenance=[]
        def pair(name):
            lease=stack.enter_context(store.lease(name)); d=lease.descriptor
            provenance.append(dict(name=name,type_id=d.type_id,ggml_shape=d.ggml_shape,
                sha256=hashlib.sha256(lease.read(0,d.nbytes)).hexdigest()))
            return checkpoint_upload(stack,e,store,name)
        for stem in (f'blk.{layer}.hc_attn_',f'blk.{layer}.hc_ffn_'):
            pairs=[pair(stem+s) for s in ('norm.weight','down.weight','up.weight','inject.weight')]
            refs.append(Qwen4ExpHyperConnection(*(p[1] for p in pairs),epsilon=eps))
        recurrent=layer%4!=3
        values=[pair(f'blk.{layer}.'+n)[1] for n in (REC_STEMS if recurrent else STEMS+['attn_output.weight']+NORMS)]
        if recurrent:
            ref=recurrent_reference(Qwen4ExpMixerSpec.from_metadata(store.metadata),values)
        else:
            ref=Qwen4ExpAttention(QSAConfig(2560,epsilon=eps),**dict(zip(NAMES,values[:6])),
                **dict(zip(('gamma_q','gamma_k','gamma_index_q','gamma_index_k'),values[6:])))
        lease=stack.enter_context(store.lease('token_embd.weight')); d=lease.descriptor
        raw=lease.read(token*d.row_bytes,d.row_bytes)
        if d.type_id in (0,1,30):
            if d.type_id==30: x=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
            else: x=np.frombuffer(raw,'<f4' if d.type_id==0 else '<f2').astype(np.float32)
        else:
            x=PackedGGUFWeight(np.frombuffer(raw,np.uint8),(1,2560),KINDS[d.type_id]).dequantize_reference().ravel()
        streams=refs[0].initialize(cp.asarray(x)); h,g=refs[0].read(streams)
        state=ref.create_state(1 if recurrent else 4096)
        y=ref.step(h[None],state)[0] if recurrent else ref.step(h,state,position=0)
        streams=refs[0].write(streams,y,g); hidden,_=refs[1].read(streams)
        result=cp.asnumpy(hidden).astype('<f4')
        identity=dict(token=token,position=0,ple=False,embedding_type=d.type_id,
            embedding_row_sha256=hashlib.sha256(raw).hexdigest(),setup_weights=provenance,
            graph='actual embedding -> attention HC read -> QSA/GDN -> attention HC write -> FFN HC read')
    return result,identity


def distribution(samples):
    a=np.asarray(samples,np.float64)
    return dict(samples=len(a),median_ms=float(np.median(a)),p90_ms=float(np.percentile(a,90)),
        p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),
        min_ms=float(a.min()),max_ms=float(a.max()),samples_ms=a.tolist())


def discrepancy(a,b):
    diff=a.astype(np.float64)-b.astype(np.float64)
    absolute=float(np.max(np.abs(diff))); relative=float(np.linalg.norm(diff)/max(np.linalg.norm(b),1e-30))
    assert relative<1e-5 or absolute<2e-6,(absolute,relative)
    assert absolute<2e-6+2e-5*np.max(np.abs(b)),(absolute,relative)
    return dict(max_absolute=absolute,relative_l2=relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',nargs='+',type=int,default=[0,1,3,47])
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--token',type=int,default=7)
    parser.add_argument('--replay-json',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if BACKEND_NAME!='cupy': parser.error('MINI_LLM_BACKEND=cupy is required')
    if args.samples<1 or args.warmup<0 or args.threads<1: parser.error('invalid sample/warmup/thread count')
    import cupy as cp
    os.environ['MINI_LLM_IQ_GPU_GROUPED_KERNEL']='warp'
    inventory_path=ROOT/'docs/native_engine/WP2_GGUF_INVENTORY.json'
    inventory=json.loads(inventory_path.read_text()); path=inventory['shards'][0]['path']
    libraries=dict(bridge_library=_load_bridge()._name,ggml_library=os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
    replay=json.loads(args.replay_json.read_text()) if args.replay_json else None
    rows=[]; inputs=[]
    for layer in args.layers:
        if replay:
            item=next(i for i in replay['inputs'] if i['layer']==layer)
            hidden=np.frombuffer(base64.b64decode(item['f32_le_base64']),'<f4').copy(); lineage=item['lineage']
            assert hashlib.sha256(hidden.tobytes()).hexdigest()==item['sha256']
        else:
            hidden,lineage=actual_hidden(cp,path,layer,args.token)
        inputs.append(dict(layer=layer,sha256=hashlib.sha256(hidden.tobytes()).hexdigest(),
            f32_le_base64=base64.b64encode(hidden.tobytes()).decode(),lineage=lineage))
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()
        for mode in ('cpu','gpu','mixed'):
            r=Runtime()
            with ExitStack() as stack:
                e=stack.enter_context(r.engine(budget_bytes=511*2**20)); store=stack.enter_context(r.gguf(path))
                source=CheckpointMoE(stack,e,store,layer)
                _,(rw,ri)=source.route(cp,hidden); selected=cp.asnumpy(ri).ravel().tolist()
                placement=() if mode=='cpu' else tuple(sorted(selected if mode=='gpu' else selected[::2]))
                owner=source.native(stack,placement,libraries,args.threads)
                model,executor=stack.enter_context(source.oracle(cp,placement,args.threads))
                def native_call():
                    with owner.submit(hidden,scale=source.scale) as job:
                        return job.readback_for_test()  # Includes wait, status and host result.
                def python_call():
                    # Same host-to-host boundary, including activation upload and final D2H.
                    return cp.asnumpy(model(cp.asarray(hidden))).ravel()
                got,ids,weights=native_call(); expected=python_call()
                error=discrepancy(got,expected)
                order=np.argsort(selected)
                np.testing.assert_array_equal(ids,np.asarray(selected)[order])
                np.testing.assert_allclose(weights,cp.asnumpy(rw).ravel()[order]*np.float32(source.scale),rtol=1e-5,atol=3e-7)
                for _ in range(args.warmup): native_call(); python_call()
                stable=bytes(e.stats); samples={'native':[],'python':[]}
                for i in range(args.samples):
                    for name,fn in ((('native',native_call),('python',python_call)) if i%2==0 else
                                    (('python',python_call),('native',native_call))):
                        cp.cuda.get_current_stream().synchronize()
                        start=perf_counter_ns(); fn(); elapsed=(perf_counter_ns()-start)/1e6
                        samples[name].append(elapsed)
                assert bytes(e.stats)==stable
                cp.get_default_memory_pool().free_all_blocks()
                memory=dict(native_device_bytes=e.stats.device_bytes,native_pinned_bytes=e.stats.pinned_bytes,
                    native_peak_bytes=e.stats.peak_bytes,oracle_pool_bytes=cp.get_default_memory_pool().total_bytes(),
                    oracle_live_bytes=cp.get_default_memory_pool().used_bytes(),
                    host_packed_selected_bytes=sum(w.nbytes for ws in source.cache.values() for w in ws))
                assert e.stats.peak_bytes<512*2**20
                assert e.stats.device_bytes+memory['oracle_pool_bytes']<512*2**20
                total_calls=1+args.warmup+args.samples
                cpu_count=10-len(placement)
                assert executor.counters['ggml_cpu_experts']==cpu_count*total_calls
                assert executor.counters['gpu_experts']==len(placement)*total_calls
                row=dict(layer=layer,mode=mode,placement=placement,ids=ids.tolist(),scaled_weights=weights.tolist(),
                    scale=source.scale,error=error,memory=memory,arithmetic=POLICY,
                    native=distribution(samples['native']),python=distribution(samples['python']),
                    python_counters=dict(executor.counters),payload_identity=source.identity)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('payload_identity','python_counters')
                    and k not in ('native','python')} | dict(native_median_ms=row['native']['median_ms'],
                    python_median_ms=row['python']['median_ms'])),flush=True)
            del owner,source,model,executor
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    files=['mini_llm/native_cpp/moe_runtime.inc','mini_llm/native_cpp/moe.cu',
           'mini_llm/native_cpp/chaining_runtime.inc','tests/native_moe_checkpoint_helpers.py',
           'mini_llm/external/qwen4exp_model.py','mini_llm/ops/expert_residency.py',
           'benchmarks/benchmark_native_moe.py']
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        benchmark='whole single-token MoE host-to-host; not full model; no HC timing',
        boundary='native submit + finish + detached result + close vs Python upload + Qwen4ExpMoE + detached result',
        warmup=args.warmup,samples=args.samples,threads=args.threads,
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cupy=cp.__version__,numpy=np.__version__,
        environment={k:v for k,v in os.environ.items() if k.startswith('MINI_LLM_') or k in
            ('NVIDIA_TF32_OVERRIDE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
        libraries={k:dict(path=str(v),sha256=sha(v)) for k,v in
            dict(native=os.environ['MINI_LLM_NATIVE_LIBRARY'],**libraries).items()},
        inventory_sha256=sha(inventory_path),shards=[dict(path=s['path'],size=Path(s['path']).stat().st_size,
            mtime_ns=Path(s['path']).stat().st_mtime_ns,inode=Path(s['path']).stat().st_ino) for s in inventory['shards']],
        source_sha256={p:sha(ROOT/p) for p in files},inputs=inputs,cases=rows,
        memory_exclusions='CUDA context/modules/constants, allocator overhead and process RSS; native engine budget is enforced',
        caveat='One fixed real first-token hidden per layer, warm pages and frozen placement. Not model/generation speedup.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
