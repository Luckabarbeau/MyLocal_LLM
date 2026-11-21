"""Nsight diagnostic: 16 sampled decode tokens after 64 response decode steps.

Run inside0290 isolation with nsys --capture-range=cudaProfilerApi. Uses0254
child guards too; defaults to no large mmap warmup and the first response.
Optionally capture the second response after an untouched first replay, using
setup-only static expert placement. No throughput claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-expert-plan', action='store_true')
    parser.add_argument('--pre-moe-graph', action='store_true')
    parser.add_argument('--shared-graph-decode', action='store_true')
    parser.add_argument('--defer-shared', action='store_true')
    parser.add_argument('--cpu-workers', type=int, default=15)
    parser.add_argument('--expert-cache-gib', type=float, default=6)
    parser.add_argument('--host-warm-gib', type=float, default=0)
    parser.add_argument('--mem-available-floor-gib', type=float, default=12)
    parser.add_argument('--force-tokens-from', type=Path)
    parser.add_argument('--capture-response', type=int, choices=(1, 2), default=1,
        help='1-based non-warmup response to instrument; earlier responses are untouched')
    parser.add_argument('--expert-profile-from', type=Path)
    parser.add_argument('--profile-case-index', type=int, default=1,
        help='0-based completed case in the expert profile source')
    args = parser.parse_args(argv)
    if args.profile_case_index < 0:
        parser.error('Profile case index must be nonnegative')
    return args


def main():
    args = parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    from mini_llm.backend import xp

    import probe_0254_strata_workload as runner
    active = [False]
    boundaries = []
    clock_anchors=[]
    response_count = 0
    captured_response = None
    capture_completed = False
    def marked(call, name):
        def wrapped(*values, **kwargs):
            if not active[0]:
                return call(*values, **kwargs)
            if name.startswith('token/'):
                before=time.perf_counter_ns()
                xp.cuda.nvtx.RangePush(name)
                after=time.perf_counter_ns()
                clock_anchors.append(dict(name=name,before_ns=before,after_ns=after))
            else:
                xp.cuda.nvtx.RangePush(name)
            try:
                return call(*values, **kwargs)
            finally:
                xp.cuda.nvtx.RangePop()
        return wrapped
    original = runner.replay_case
    def replay(model, tokenizer, row, prompt, replay_xp, sample_token, *tail):
        nonlocal response_count, captured_response, capture_completed
        if row['request_id'] != 'warmup':
            response_count += 1
        if row['request_id'] == 'warmup' or response_count != args.capture_response:
            return original(model, tokenizer, row, prompt, replay_xp, sample_token, *tail)
        captured_response = response_count
        restorations = []
        patched = set()
        def install(owner, name, wrapped):
            key = (id(owner), name)
            if key not in patched:
                restorations.append((owner, name, name in vars(owner), getattr(owner, name)))
                setattr(owner, name, wrapped)
                patched.add(key)
        def mark(owner, name, label):
            if (id(owner), name) not in patched:
                install(owner, name, marked(getattr(owner, name), label))
        original_step = model._step
        base = len(prompt['token_ids'])
        def step(token, state, **kwargs):
            position = state.length-base
            if position == 64:
                xp.cuda.profiler.start()
                active[0] = True
                boundaries.append(dict(event='start', position=position, seconds=time.perf_counter()))
            if position == 80 and active[0]:
                boundaries.append(dict(event='stop', position=position, seconds=time.perf_counter()))
                active[0] = False
                xp.cuda.profiler.stop()
            return marked(original_step, f'token/{position}')(token, state, **kwargs)
        try:
            executor = model.expert_executor
            for name in ('try_forward_device_plan', 'try_forward_decode_batch', '_prepare'):
                mark(executor, name, name)
            pool = executor._ggml_row_pool
            mark(pool, 'submit', 'cpu_submit_quantize')
            mark(pool, '_submit_registered', 'cpu_registered_submit')
            mark(pool, '_finish', 'cpu_completion_wait')
            for layer, block in enumerate(model.blocks):
                mark(block.attention_hc, 'read', f'L{layer}/hc_attention_read')
                mark(block.ffn_hc, 'read', f'L{layer}/hc_ffn_read')
                mark(block.mixer, 'step', f'L{layer}/mixer')
                mark(block.moe.router, 'forward', f'L{layer}/router')
            install(model, '_step', step)
            result = original(model, tokenizer, row, prompt, replay_xp,
                marked(sample_token, 'sampler_and_output_wait'), *tail)
            capture_completed = True
            return result
        finally:
            try:
                if active[0]:
                    try:
                        xp.cuda.get_current_stream().synchronize()
                    finally:
                        active[0] = False
                        xp.cuda.profiler.stop()
            finally:
                for owner, name, had_local, call in reversed(restorations):
                    if had_local:
                        setattr(owner, name, call)
                    else:
                        delattr(owner, name)
                (args.output/'capture_window.json').write_text(json.dumps(boundaries, indent=2))
                (args.output/'clock_anchors.json').write_text(json.dumps(clock_anchors,indent=2))
    runner.replay_case = replay
    try:
        runner.child(SimpleNamespace(output=args.output, cases=['warmup', 'performance-short-r1'],
        response_repeats=args.capture_response, expert_profile_from=args.expert_profile_from,
        profile_case_index=args.profile_case_index,
        host_warm_gib=args.host_warm_gib, mem_available_floor_gib=args.mem_available_floor_gib,
        expert_cache_gib=args.expert_cache_gib, force_tokens_from=args.force_tokens_from,
        rss_anon_limit_gib=8,
        victim_policy='heap', cpu_backend='ggml_pool', cpu_workers=args.cpu_workers,
        group_mixed_decode=False, promotion_limit=0, overlap_shared=True,
        gdn_graph_decode=not args.pre_moe_graph, pre_moe_graph=args.pre_moe_graph,
        shared_graph_decode=args.shared_graph_decode, defer_shared=args.defer_shared,
        device_expert_plan=args.device_expert_plan))
    finally:
        runner.replay_case = original
        metadata = dict(capture_response=args.capture_response,
            response_repeats=args.capture_response, observed_response_count=response_count,
            captured_response_index=captured_response,
            captured_case_index=captured_response,  # result.json includes the leading warmup.
            request_id='performance-short-r1', start_position=64, stop_position=80,
            requested_token_count=16, captured_token_count=len(clock_anchors),
            complete=(capture_completed and len(clock_anchors) == 16 and
                bool(boundaries) and boundaries[-1]['event'] == 'stop'),
            expert_profile_from=(None if args.expert_profile_from is None else
                str(args.expert_profile_from.resolve())),
            profile_case_index=args.profile_case_index,
            placement_profile_source='placement_profile_source.json' if args.expert_profile_from else None,
            force_tokens_from=(None if args.force_tokens_from is None else
                str(args.force_tokens_from.resolve())),
            profiling_overhead_included=True, rate_claim=False)
        (args.output/'capture_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
