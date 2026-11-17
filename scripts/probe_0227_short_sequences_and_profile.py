"""Bounded full-model short-sequence gate and initial synchronized decode profile."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from probe_0218_hc_identity import MODEL

PREFIXES=(1,2,4,8,16)
RSS_LIMIT=12*1024**3


def digest_state(state,asnumpy):
    h=hashlib.sha256()
    def add(label,array):
        a=np.ascontiguousarray(asnumpy(array))
        h.update(label.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    h.update(str(state.length).encode())
    for i,s in sorted(state.recurrent.items()):
        if s.delta.length!=state.length or s.conv.length!=state.length:raise ValueError('recurrent positions disagree')
        add(f'r{i}',s.delta.memory);add(f'c{i}',s.conv.history)
    for i,s in sorted(state.qsa.items()):
        if s.length!=state.length:raise ValueError('QSA positions disagree')
        add(f'k{i}',s.keys[:s.length]);add(f'v{i}',s.values[:s.length])
        add(f'i{i}',s.raw_index[:s.length]);add(f'p{i}',s.pooled[:s.length//4])
    for i,s in sorted(state.ple.items()):
        add(f'ple{i}',s.normalized_history);h.update(str(s.predecessors).encode())
    return h.hexdigest()


class Profile:
    def __init__(self,xp):
        self.xp=xp;self.position=0;self.records=[]
    def wrap(self,call,layer,stage):
        def measured(*args,**kwargs):
            if self.position<8:return call(*args,**kwargs)
            begin=self.xp.cuda.Event();end=self.xp.cuda.Event()
            begin.record();start=time.perf_counter()
            result=call(*args,**kwargs)
            elapsed=time.perf_counter()-start;end.record()
            self.records.append((layer,stage,elapsed,begin,end))
            return result
        return measured
    def install(self,model):
        model.embedding=self.wrap(model.embedding,-1,'embedding')
        model.output=self.wrap(model.output,-1,'output_head')
        model.final_hc.collapse=self.wrap(model.final_hc.collapse,-1,'final_hc')
        for i,b in enumerate(model.blocks):
            for hc,label in [(b.attention_hc,'attention_hc'),(b.ffn_hc,'ffn_hc')]:
                hc.read=self.wrap(hc.read,i,label+'_read');hc.write=self.wrap(hc.write,i,label+'_write')
            if b.recurrent:b.mixer.forward=self.wrap(b.mixer.forward,i,'gated_delta')
            else:b.mixer.step=self.wrap(b.mixer.step,i,'qsa')
            if b.ple is not None:
                b.ple.step=self.wrap(b.ple.step,i,'ple_total')
                b.ple.embedding=self.wrap(b.ple.embedding,i,'ple_lookup')
            moe=b.moe
            moe.router.forward=self.wrap(moe.router.forward,i,'router')
            moe.experts.forward=self.wrap(moe.experts.forward,i,'routed_experts')
            for name in ['gate','up','down']:
                setattr(moe,name,self.wrap(getattr(moe,name),i,'shared_'+name))
            b.moe=self.wrap(moe,i,'moe_total')
    def summary(self):
        self.xp.cuda.get_current_stream().synchronize()
        grouped=defaultdict(lambda:dict(calls=0,host_seconds=0.,device_span_seconds=0.))
        layers=defaultdict(lambda:dict(calls=0,host_seconds=0.,device_span_seconds=0.))
        for layer,stage,elapsed,start,end in self.records:
            device=float(self.xp.cuda.get_elapsed_time(start,end))/1000
            for target in [grouped[stage],layers[(layer,stage)]]:
                target['calls']+=1;target['host_seconds']+=elapsed;target['device_span_seconds']+=device
        return dict(stages=dict(grouped),per_layer=[dict(layer=i,stage=s,**v) for (i,s),v in sorted(layers.items())],
            interpretation='Host durations include implicit waits; CUDA event spans may include idle CPU gaps. Nested moe_total/ple_total overlap their children: do not add all rows. Events are collected without per-operation synchronization; tokens synchronize at boundaries. This is an instrumented single-request profile, not a steady-state throughput benchmark.')


def child(mode,output):
    from mini_llm.backend import BACKEND_NAME,asnumpy,xp
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    if BACKEND_NAME!='cupy':raise ValueError('This full-model diagnostic explicitly requires CuPy')
    start=time.perf_counter()
    with Qwen4ExpGGUFModel.open(MODEL,max_length=16,cpu_workers=4) as model:
        built=time.perf_counter();state=model.create_generation_state(1,16)
        profile=Profile(xp)
        if mode=='incremental':profile.install(model)
        snapshots=[];latencies=[];previous=0
        for prefix in PREFIXES:
            if mode=='prefill':
                t=time.perf_counter();logits=model.prefill(list(range(previous+1,prefix+1)),state)
                xp.cuda.get_current_stream().synchronize();latencies.append(dict(tokens=prefix-previous,seconds=time.perf_counter()-t))
            else:
                for p in range(previous,prefix):
                    profile.position=p;t=time.perf_counter();logits=model.step(p+1,state)
                    xp.cuda.get_current_stream().synchronize();latencies.append(dict(position=p,seconds=time.perf_counter()-t))
            a=asnumpy(logits).copy()
            if a.shape!=(1,248320) or not np.isfinite(a).all():raise ValueError('invalid full-model logits')
            np.save(output/f'{mode}_logits_{prefix}.npy',a)
            snapshots.append(dict(prefix=prefix,state_sha256=digest_state(state,asnumpy),argmax=int(a.argmax())))
            previous=prefix
        result=dict(mode=mode,consumed_ids=list(range(1,17)),snapshots=snapshots,latencies=latencies,
            build_seconds=built-start,total_seconds=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            device_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
            device_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
            expert_statistics=model.expert_executor.statistics(),plan=model.plan)
        if mode=='incremental':result['profile']=profile.summary()
    with (output/f'{mode}.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


def child_limits():
    # CUDA reserves large virtual address ranges; a CPU RLIMIT_AS would reject
    # them despite bounded actual VRAM. Device construction has its own budgets.
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def bounded_gpu(command,output,mode):
    env=dict(os.environ,MINI_LLM_BACKEND='cupy',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.monotonic();peak=0
    with (output/f'{mode}.log').open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=child_limits)
        reason=None
        try:
            while proc.poll() is None:
                try:
                    lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    rss=next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                    if rss>RSS_LIMIT:reason='12 GiB sampled host RSS limit'
                except FileNotFoundError:pass
                if time.monotonic()-start>120:reason='120-second child deadline'
                if reason:proc.kill();break
                time.sleep(.05)
        finally:
            if proc.poll() is None:proc.kill()
            proc.wait()
    record=dict(returncode=proc.returncode,seconds=time.monotonic()-start,sampled_peak_rss_bytes=peak,stop_reason=reason)
    if proc.returncode or reason:raise RuntimeError(json.dumps(record)+'; inspect '+str(output/f'{mode}.log'))
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',choices=['incremental','prefill']);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.child:
        child(args.child,args.output);return
    docs=ROOT/'docs/qwen4exp';output=docs/'diagnostics_0227_v1';result=docs/'0227_short_sequences_profile_v1.json'
    if output.exists() or result.exists():raise FileExistsError('preserve previous artifacts')
    # Do not enter the performance phase without the independent state gate.
    gate=json.loads((docs/'0226_identical_block_states_v2.json').read_text())
    if gate['status']!='passed':raise ValueError('identical-input/state gate has not passed')
    output.mkdir();report=dict(identical_block_state_gate='0226_identical_block_states_v2.json',
        production_files_modified=False,model_ids=list(range(1,17)),backend='cupy',modes_sequential=True,
        note='Full-model within-backend self-consistency plus profiling; not independent full-model logit equality')
    try:
        for mode in ['incremental','prefill']:
            report[mode+'_run']=bounded_gpu([sys.executable,str(Path(__file__)),'--child',mode,'--output',str(output)],output,mode)
            report[mode]=json.loads((output/f'{mode}.json').read_text())
        report['comparisons']=[]
        for i,prefix in enumerate(PREFIXES):
            a=np.load(output/f'incremental_logits_{prefix}.npy');b=np.load(output/f'prefill_logits_{prefix}.npy')
            np.testing.assert_array_equal(a,b)
            if report['incremental']['snapshots'][i]['state_sha256']!=report['prefill']['snapshots'][i]['state_sha256']:
                raise AssertionError(f'full active-state digest mismatch at prefix {prefix}')
            report['comparisons'].append(dict(prefix=prefix,logits_bit_exact=True,all_active_states_sha256_equal=True))
        report['status']='passed_validation_gate_enter_initial_profiling'
    except Exception as e:
        report['status']='blocked';report['error']=str(e)
    with result.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('incremental','prefill')},indent=2))
    if report['status']=='blocked':raise SystemExit(1)

if __name__=='__main__':main()
