"""Bounded real-checkpoint same-arithmetic GDN capture parity, fixed IDs1..16.

Run modes sequentially. No checkpoint copy, bank expansion or eager host warmup.
Both modes construct capture-capable projections; uncaptured disables replay only.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('graph', 'uncaptured'), required=True)
    parser.add_argument('--segment', choices=('gdn','pre'), default='gdn')
    parser.add_argument('--shared-graph',action='store_true')
    parser.add_argument('--defer-shared',action='store_true')
    parser.add_argument('--native-handoff',action='store_true')
    parser.add_argument('--qsa-frontend-graph',action='store_true')
    args = parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(MINI_LLM_BACKEND='cupy', MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
                      MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp')
    if args.native_handoff:
        os.environ['MINI_LLM_NATIVE_MOE_HANDOFF'] = '1'
    if args.qsa_frontend_graph:
        os.environ['MINI_LLM_QSA_FRONTEND_GRAPH'] = '1'
    import numpy as np
    from mini_llm.backend import xp, asnumpy
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from probe_0218_hc_identity import MODEL
    from probe_0227_short_sequences_and_profile import digest_state
    from probe_0254_strata_workload import MODEL_OPTIONS, memory_snapshot, enforce_limits, GiB
    deadline = time.monotonic()+150
    def guard():
        m = memory_snapshot()
        enforce_limits(m, 12*GiB, 8*GiB, deadline)
        return m
    guard()
    options = dict(MODEL_OPTIONS, max_length=32, cpu_workers=15, expert_cpu_backend='ggml_pool',
                   expert_victim_policy='heap', gdn_graph_decode=args.segment=='gdn',
                   pre_moe_graph=args.segment=='pre',shared_graph_decode=args.shared_graph,
                                      overlap_shared=args.defer_shared, defer_shared=args.defer_shared,
                                                         device_expert_plan=args.native_handoff)
    reports = []
    with Qwen4ExpGGUFModel.open(MODEL, **options) as model:
        # Freeze empty placement: same complete CPU Q8 experts in both processes.
        model.expert_executor.cache.set_admission_budget(0)
        if args.mode == 'uncaptured':
            model._pre_moe_segments = {}
            for block in model.blocks:
                block.moe._shared_graph_enabled=False
                if block.recurrent:
                    block.mixer.graph_decode = False
                else:
                    block.mixer._frontend_graph_enabled = False
        for repeat in range(2):
            state = model.create_generation_state(1, 32)
            records = []
            for prefix in range(1, 17):
                guard()
                logits = model.step(prefix, state)
                if prefix in (1, 2, 4, 8, 16):
                    array = asnumpy(logits)
                    if not np.isfinite(array).all():
                        raise ValueError('Nonfinite logits')
                    np.save(args.output/f'logits_{repeat}_{prefix}.npy', array)
                    records.append(dict(prefix=prefix, state_sha256=digest_state(state, asnumpy)))
            reports.append(records)
            del state, logits
            gc.collect()
        xp.cuda.get_current_stream().synchronize()
        report = dict(mode=args.mode, segment=args.segment, defer_shared=args.defer_shared,
                    native_handoff=args.native_handoff, qsa_frontend_graph=args.qsa_frontend_graph,
                    token_ids=list(range(1,17)), requests=reports,
            graph_statistics=[m.graph_statistics() for m in model._gdn_graph_owners],
            pre_graph_statistics=[m.graph_statistics() for m in model._pre_moe_segments.values()],
            shared_graph_statistics=[m.graph_statistics() for m in model._shared_graph_owners],
            final_memory=guard(), plan=model.plan,
            same_arithmetic=True, routing='no GPU admissions; complete CPU/Q8 staged expert split',
            native_stage_limit=int(os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_LIMIT','0')),
            optimization_environment={name:os.environ.get(name,'0') for name in (
                'MINI_LLM_NATIVE_MOE_FUSED_SHARED','MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH',
                'MINI_LLM_Q8_SHARED_GRID','MINI_LLM_BOUND_RECURRENT_STATE','MINI_LLM_GRAPH_DEVICE_COPY')},
            expert_statistics=model.expert_executor.statistics())
        (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(f'Completed {args.mode}: {args.output}')


if __name__ == '__main__':
    main()
