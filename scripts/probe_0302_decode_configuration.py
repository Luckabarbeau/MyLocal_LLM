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
    parser.add_argument('--q2-rows',action='store_true',help='Opt-in Q2 down rows with existing GGML Q8_0 activations')
    parser.add_argument('--q2-rows-exact',action='store_true',help='Exact-order precomputed Q2 down rows')
    parser.add_argument('--spin-us',type=int,default=50,choices=range(2001))
    parser.add_argument('--strata-affinity',action='store_true',help='Captured 19-worker CPU mask and submitting CPU5')
    parser.add_argument('--expert-cache-gib',type=float,default=7.25,choices=(7,7.25))
    parser.add_argument('--resident-host-gib',type=float,default=0,choices=(0,1,2,4))
    parser.add_argument('--qsa-frontend',action='store_true')
    parser.add_argument('--recurrent-block-graph',action='store_true')
    parser.add_argument('--qsa-cooperative',action='store_true')
    parser.add_argument('--qsa-warp-dots',action='store_true')
    parser.add_argument('--token-graph',action='store_true')
    parser.add_argument('--token-graph-baseline',action='store_true',help='Same capture-ready PLE arithmetic without whole-token replay')
    parser.add_argument('--recurrent-chain-graph',action='store_true')
    parser.add_argument('--qsa-block-graph',action='store_true')
    parser.add_argument('--qsa-block-baseline',action='store_true',help='Same capture-ready QSA arithmetic without complete-block replay')
    parser.add_argument('--hc-parallel',action='store_true')
    parser.add_argument('--fused-shared',action='store_true')
    parser.add_argument('--kernel-publish',action='store_true')
    parser.add_argument('--shared-grid',action='store_true')
    parser.add_argument('--recurrent-pair',action='store_true',help='Exact same-input fused QKV/Z packed GPU projection')
    parser.add_argument('--natural',action='store_true')
    parser.add_argument('--route-profile',action='store_true',help='Diagnostic native expert counters; not a clean throughput control')
    parser.add_argument('--tokens-from',type=Path,help='Explicit retained trajectory for matched placement comparisons')
    parser.add_argument('--expert-profile-from',type=Path,help='Setup-only placement from a prior completed request')
    parser.add_argument('--profile-case-index',type=int,default=1)
    parser.add_argument('--warmup-frequency-placement',action='store_true')
    parser.add_argument('--response-repeats',type=int,choices=range(1,5),default=1)
    parser.add_argument('--trace',action='store_true',help='16-token node-level Nsight diagnostic, not a rate claim')
    parser.add_argument('--trace-response',type=int,choices=(1,2),default=1,help='Capture the first or warmed second response')
    parser.add_argument('--native-timeline',action='store_true',help='Diagnostic CPU phase timestamps; not a clean throughput control')
    parser.add_argument('--telemetry',action='store_true')
    parser.add_argument('--parity',action='store_true',help='Two fixed 16-token requests, not a throughput run')
    args=parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    if args.strata_affinity and args.workers != 19:
        parser.error('Captured Strata affinity requires exactly 19 workers')
    if args.natural and args.tokens_from is not None:
        parser.error('Natural generation cannot use a forced trajectory')
    if args.profile_case_index < 0 or (args.expert_profile_from is not None and args.warmup_frequency_placement):
        parser.error('Require nonnegative profile case index and only one profile count source')
    if args.parity and (args.tokens_from is not None or args.expert_profile_from is not None or args.warmup_frequency_placement):
        parser.error('Parity wrapper does not support trajectory/profile overrides')
    if args.trace and args.warmup_frequency_placement:
        parser.error('Trace wrapper does not support warmup-frequency placement')
    if not args.trace and args.trace_response != 1:
        parser.error('Selecting a trace response requires --trace')
    settings={
        'MINI_LLM_GGML_IQ3XXS_GATHER':'0','MINI_LLM_GGML_IQ3XXS_PAIR':'0',
        'MINI_LLM_GGML_Q2_ROWS':str(int(args.q2_rows)),
        'MINI_LLM_GGML_Q2_ROWS_EXACT':str(int(args.q2_rows_exact)),
        'MINI_LLM_GGML_POOL_TASK_FACTOR':str(args.task_factor),'MINI_LLM_GGML_IQ4_VNNI':'0',
        'MINI_LLM_NATIVE_MOE_TIMELINE':str(int(args.native_timeline)),'MINI_LLM_GGML_IQ3_GATHER':'1',
        'MINI_LLM_BOUND_RECURRENT_STATE':'1','MINI_LLM_GRAPH_DEVICE_COPY':'1',
        'MINI_LLM_EXPERT_GPU_Q8':'1','MINI_LLM_LINEAR_GPU_Q8':'1',
        'MINI_LLM_HC_BF16_WEIGHTS':'1','MINI_LLM_NATIVE_MOE_HANDOFF':'1',
        'MINI_LLM_NATIVE_MOE_GRAPH':'1','MINI_LLM_NATIVE_MOE_SERVICE':'1',
        'MINI_LLM_NATIVE_MOE_FUSED_SHARED':str(int(args.fused_shared)),
        'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH':str(int(args.kernel_publish)),
        'MINI_LLM_Q8_SHARED_GRID':str(int(args.shared_grid)),
        'MINI_LLM_Q8_RECURRENT_PAIR':str(int(args.recurrent_pair)),
        'MINI_LLM_NATIVE_MOE_STAGE_LIMIT':'0','MINI_LLM_NATIVE_MOE_ROUTE_PROFILE':str(int(args.route_profile)),
        'MINI_LLM_SIZED_EXPERT_SLOTS':'1','MINI_LLM_DEVICE_PLAN_SPECIALIZED':'0',
        'MINI_LLM_QSA_FRONTEND_GRAPH':str(int(args.qsa_frontend or args.qsa_block_graph or args.qsa_block_baseline or args.token_graph or args.token_graph_baseline)),
        'MINI_LLM_QSA_BLOCK_GRAPH':str(int(args.qsa_block_graph or args.qsa_block_baseline or args.token_graph or args.token_graph_baseline)),
        'MINI_LLM_TOKEN_GRAPH':str(int(args.token_graph)),
        'MINI_LLM_TOKEN_GRAPH_PREP':str(int(args.token_graph or args.token_graph_baseline)),
        'MINI_LLM_QSA_BLOCK_REPLAY':str(int(not args.qsa_block_baseline)),
        'MINI_LLM_RECURRENT_BLOCK_GRAPH':str(int(args.recurrent_block_graph or args.recurrent_chain_graph or args.token_graph or args.token_graph_baseline)),
        'MINI_LLM_RECURRENT_CHAIN_GRAPH':str(int(args.recurrent_chain_graph)),
        'MINI_LLM_QSA_COOPERATIVE':str(int(args.qsa_cooperative)),
        'MINI_LLM_QSA_WARP_DOTS':str(int(args.qsa_warp_dots)),
        'MINI_LLM_HC_PARALLEL_READ':str(int(args.hc_parallel)),
        'MINI_LLM_GRAPH_READONLY_INPUTS':'1','MINI_LLM_QSA_DEFER_VALIDATION':'1',
        'MINI_LLM_GGML_POOL_FUSED_HIDDEN':'0','MINI_LLM_GGML_POOL_SCHEDULER':'tasks',
        'MINI_LLM_GGML_POOL_HOST_HELP':'1',
        'MINI_LLM_GGML_POOL_CPUS':'7,0,1,2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19' if args.strata_affinity else '',
        'MINI_LLM_HOST_CPU':'5' if args.strata_affinity else '','MINI_LLM_GGML_POOL_SPIN_US':str(args.spin_us),'MINI_LLM_GGML_Q2_SIMD':'1',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
        'MINI_LLM_GGML_CPU_LIBRARY':'/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so',
    }
    command=[sys.executable,str(ROOT/'scripts/probe_0254_strata_workload.py'),
        '--output',str(args.output),'--expert-cache-gib',str(args.expert_cache_gib),'--host-warm-gib','0',
        '--mem-available-floor-gib','12','--victim-policy','heap','--cpu-backend','ggml_pool',
        '--cpu-workers',str(args.workers),'--resident-host-gib',str(args.resident_host_gib),
        '--promotion-limit','0','--overlap-shared','--response-repeats',str(args.response_repeats),
        '--pre-moe-graph','--device-expert-plan','--shared-graph-decode','--defer-shared']
    if not args.natural:
        command+=['--force-tokens-from',str(args.tokens_from if args.tokens_from is not None else
            ROOT/'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json')]
    if args.expert_profile_from is not None:
        command+=['--expert-profile-from',str(args.expert_profile_from),
                  '--profile-case-index',str(args.profile_case_index)]
    if args.warmup_frequency_placement:
        command.append('--warmup-frequency-placement')

    if args.trace:
        if args.resident_host_gib: parser.error('Trace wrapper does not support host source overrides')
        if args.natural: parser.error('Trace requires the controlled token trajectory')
        if args.response_repeats != 1: parser.error('Trace does not support repeated responses')
        command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace=node',
            '--output',str(args.output)+'_trace',sys.executable,
            str(ROOT/'scripts/probe_0266_critical_path.py'),'--output',str(args.output),
            '--cpu-workers',str(args.workers),'--expert-cache-gib',str(args.expert_cache_gib),'--host-warm-gib','0',
            '--mem-available-floor-gib','12','--device-expert-plan','--pre-moe-graph',
            '--shared-graph-decode','--defer-shared','--capture-response',str(args.trace_response),
            '--force-tokens-from',str(args.tokens_from if args.tokens_from is not None else
                ROOT/'docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json')]
        if args.expert_profile_from is not None:
            command+=['--expert-profile-from',str(args.expert_profile_from),
                      '--profile-case-index',str(args.profile_case_index)]
    if args.parity:
        if args.trace or args.natural or args.resident_host_gib or args.response_repeats != 1:
            parser.error('Parity does not support trace, natural, host source or response repeats')
        command=[sys.executable,str(ROOT/'scripts/probe_0265_gdn_graph_parity.py'),
            '--output',str(args.output),'--mode','graph','--segment','pre',
            '--shared-graph','--defer-shared','--native-handoff']
        if args.qsa_frontend or args.qsa_block_graph or args.qsa_block_baseline or args.token_graph or args.token_graph_baseline:
            command.append('--qsa-frontend-graph')
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
        (args.output/'reproduce.json').write_text(json.dumps(dict(command=command,environment=settings,
                    route_profile_diagnostic=args.route_profile),indent=2))
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
