"""Nsight diagnostic: 16 sampled decode tokens after 64 response decode steps.

Run with nsys --capture-range=cudaProfilerApi. Uses0254 child memory/time guards;
this direct diagnostic does not use the0254 parent supervisor. No throughput claim.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-expert-plan', action='store_true')
    parser.add_argument('--pre-moe-graph', action='store_true')
    parser.add_argument('--shared-graph-decode', action='store_true')
    parser.add_argument('--defer-shared', action='store_true')
    parser.add_argument('--cpu-workers', type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    from mini_llm.backend import xp
    import mini_llm.generation as generation
    import probe_0254_strata_workload as runner
    active = [False]
    boundaries = []
    def marked(call, name):
        def wrapped(*values, **kwargs):
            if not active[0]:
                return call(*values, **kwargs)
            xp.cuda.nvtx.RangePush(name)
            try:
                return call(*values, **kwargs)
            finally:
                xp.cuda.nvtx.RangePop()
        return wrapped
    generation.sample_token = marked(generation.sample_token, 'sampler_and_output_wait')
    original = runner.replay_case
    def replay(model, tokenizer, row, prompt, *tail):
        if row['request_id'] == 'warmup':
            return original(model, tokenizer, row, prompt, *tail)
        executor = model.expert_executor
        for name in ('try_forward_device_plan', 'try_forward_decode_batch', '_prepare'):
            setattr(executor, name, marked(getattr(executor, name), name))
        pool = executor._ggml_row_pool
        pool.submit = marked(pool.submit, 'cpu_submit_quantize')
        pool._submit_registered = marked(pool._submit_registered, 'cpu_registered_submit')
        pool._finish = marked(pool._finish, 'cpu_completion_wait')
        for layer, block in enumerate(model.blocks):
            block.attention_hc.read = marked(block.attention_hc.read, f'L{layer}/hc_attention_read')
            block.ffn_hc.read = marked(block.ffn_hc.read, f'L{layer}/hc_ffn_read')
            block.mixer.step = marked(block.mixer.step, f'L{layer}/mixer')
            block.moe.router.forward = marked(block.moe.router.forward, f'L{layer}/router')
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
        model._step = step
        try:
            return original(model, tokenizer, row, prompt, *tail)
        finally:
            if active[0]:
                xp.cuda.get_current_stream().synchronize()
                active[0] = False
                xp.cuda.profiler.stop()
            (args.output/'capture_window.json').write_text(json.dumps(boundaries, indent=2))
    runner.replay_case = replay
    runner.child(SimpleNamespace(output=args.output, cases=['warmup', 'performance-short-r1'],
        host_warm_gib=50, mem_available_floor_gib=4, rss_anon_limit_gib=8,
        victim_policy='heap', cpu_backend='ggml_pool', cpu_workers=args.cpu_workers,
        group_mixed_decode=False, promotion_limit=0, overlap_shared=True,
        gdn_graph_decode=not args.pre_moe_graph, pre_moe_graph=args.pre_moe_graph,
        shared_graph_decode=args.shared_graph_decode, defer_shared=args.defer_shared,
        device_expert_plan=args.device_expert_plan))


if __name__ == '__main__':
    main()
