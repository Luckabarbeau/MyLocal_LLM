"""Reproduce the bounded long-response full-top10 configuration (0297 onward)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=19,choices=range(1,24))
    parser.add_argument('--task-factor',type=int,default=3,choices=range(1,17))
    parser.add_argument('--expert-cache-gib',type=float,default=7.25,choices=(7,7.25))
    parser.add_argument('--qsa-frontend',action='store_true')
    parser.add_argument('--hc-parallel',action='store_true')
    parser.add_argument('--fused-shared',action='store_true')
    parser.add_argument('--kernel-publish',action='store_true')
    parser.add_argument('--shared-grid',action='store_true')
    parser.add_argument('--natural',action='store_true')
    parser.add_argument('--trace',action='store_true',help='16-token node-level Nsight diagnostic, not a rate claim')
    parser.add_argument('--telemetry',action='store_true')
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    settings={
        'MINI_LLM_GGML_IQ3XXS_GATHER':'0','MINI_LLM_GGML_IQ3XXS_PAIR':'0',
        'MINI_LLM_GGML_POOL_TASK_FACTOR':str(args.task_factor),'MINI_LLM_GGML_IQ4_VNNI':'0',
        'MINI_LLM_NATIVE_MOE_TIMELINE':'0','MINI_LLM_GGML_IQ3_GATHER':'1',
        'MINI_LLM_BOUND_RECURRENT_STATE':'1','MINI_LLM_GRAPH_DEVICE_COPY':'1',
        'MINI_LLM_EXPERT_GPU_Q8':'1','MINI_LLM_LINEAR_GPU_Q8':'1',
        'MINI_LLM_HC_BF16_WEIGHTS':'1','MINI_LLM_NATIVE_MOE_HANDOFF':'1',
        'MINI_LLM_NATIVE_MOE_GRAPH':'1','MINI_LLM_NATIVE_MOE_SERVICE':'1',
        'MINI_LLM_NATIVE_MOE_FUSED_SHARED':str(int(args.fused_shared)),
        'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH':str(int(args.kernel_publish)),
        'MINI_LLM_Q8_SHARED_GRID':str(int(args.shared_grid)),
        'MINI_LLM_NATIVE_MOE_STAGE_LIMIT':'0','MINI_LLM_NATIVE_MOE_ROUTE_PROFILE':'0',
        'MINI_LLM_SIZED_EXPERT_SLOTS':'1','MINI_LLM_DEVICE_PLAN_SPECIALIZED':'0',
        'MINI_LLM_QSA_FRONTEND_GRAPH':str(int(args.qsa_frontend)),
        'MINI_LLM_HC_PARALLEL_READ':str(int(args.hc_parallel)),
        'MINI_LLM_GRAPH_READONLY_INPUTS':'1','MINI_LLM_QSA_DEFER_VALIDATION':'1',
        'MINI_LLM_GGML_POOL_FUSED_HIDDEN':'0','MINI_LLM_GGML_POOL_SCHEDULER':'tasks',
        'MINI_LLM_GGML_POOL_HOST_HELP':'1','MINI_LLM_GGML_POOL_CPUS':'',
        'MINI_LLM_HOST_CPU':'','MINI_LLM_GGML_POOL_SPIN_US':'50','MINI_LLM_GGML_Q2_SIMD':'1',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
        'MINI_LLM_GGML_CPU_LIBRARY':'/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so',
    }
    command=[sys.executable,str(ROOT/'scripts/probe_0254_strata_workload.py'),
        '--output',str(args.output),'--expert-cache-gib',str(args.expert_cache_gib),'--host-warm-gib','0',
        '--mem-available-floor-gib','12','--victim-policy','heap','--cpu-backend','ggml_pool',
        '--cpu-workers',str(args.workers),'--promotion-limit','0','--overlap-shared',
        '--pre-moe-graph','--device-expert-plan','--shared-graph-decode','--defer-shared']
    if not args.natural:
        command+=['--force-tokens-from',str(ROOT/'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json')]

    if args.trace:
        if args.natural: parser.error('Trace requires the controlled token trajectory')
        command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace=node',
            '--output',str(args.output)+'_trace',sys.executable,
            str(ROOT/'scripts/probe_0266_critical_path.py'),'--output',str(args.output),
            '--cpu-workers',str(args.workers),'--expert-cache-gib',str(args.expert_cache_gib),'--host-warm-gib','0',
            '--mem-available-floor-gib','12','--device-expert-plan','--pre-moe-graph',
            '--shared-graph-decode','--defer-shared','--force-tokens-from',
            str(ROOT/'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json')]
    stop=threading.Event(); readings=[]
    def sample():
        for _ in range(370):
            if stop.is_set(): return
            started=time.monotonic()
            try:
                row=subprocess.run(['nvidia-smi','--query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                readings.append(dict(monotonic=started,returncode=row.returncode,csv=row.stdout.strip()))
            except (OSError,subprocess.TimeoutExpired) as exc:
                readings.append(dict(monotonic=started,error=str(exc)))
            if stop.wait(.5): return
    thread=threading.Thread(target=sample) if args.telemetry else None
    if thread is not None: thread.start()
    try:
        result=subprocess.run(command,cwd=ROOT,env={**os.environ,**settings},timeout=185)
    finally:
        stop.set()
        if thread is not None: thread.join(timeout=5)
        if args.output.is_dir() and args.telemetry:
            (args.output/'device_telemetry.json').write_text(json.dumps(readings,indent=2))
    if args.output.is_dir():
        (args.output/'reproduce.json').write_text(json.dumps(dict(command=command,environment=settings),indent=2))
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
