"""Bounded optimized-path prefill/incremental, reset and greedy smoke check."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from probe_0218_hc_identity import MODEL
from probe_0227_short_sequences_and_profile import digest_state, bounded_gpu


def child(output, *, hc_graph=False):
    os.environ['MINI_LLM_IQ_GPU_CODEBOOK_STORAGE'] = 'global'
    os.environ['MINI_LLM_IQ_GPU_GROUPED_KERNEL'] = 'warp'
    from mini_llm.backend import asnumpy, xp
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    with Qwen4ExpGGUFModel.open(MODEL, max_length=16, cpu_workers=4,
            static_device_budget_bytes=6*1024**3, device_budget_bytes=14*1024**3,
            cache_bytes=6*1024**3, expert_promote_after=1,
            qsa_fast_decode=True, hc_fast_decode=True, router_fast_decode=True,
            gdn_fast_decode=True, group_after_admission=True, expert_cache_slots=True,
            hc_graph_decode=hc_graph) as model:
        ids = list(range(1, 17))
        state = model.create_generation_state(1, 16)
        incremental = None
        for token in ids:
            incremental = model.step(token, state)
        incremental = asnumpy(incremental).copy()
        state_hash = digest_state(state, asnumpy)
        del state
        gc.collect()
        fresh = model.create_generation_state(1, 16)
        prefill = asnumpy(model.prefill(ids, fresh)).copy()
        np.testing.assert_array_equal(prefill, incremental)
        fresh_hash = digest_state(fresh, asnumpy)
        if state_hash != fresh_hash:
            raise AssertionError('optimized incremental/prefill active-state mismatch')
        del fresh
        gc.collect()
        generated = model.greedy([1, 2], max_new_tokens=3)
        if generated != [220, 100561, 198]:
            raise AssertionError(f'greedy smoke differs: {generated}')
        repeated = model.greedy([1, 2], max_new_tokens=3)
        if repeated != generated:
            raise AssertionError('fresh request contamination')
        xp.cuda.get_current_stream().synchronize()
        report = dict(status='passed', input_ids=ids, backend='cupy', activation_dtype='float32',
                      codebook_storage='global', grouped_kernel='warp', expert_cache_slots=True,
                      hc_graph_decode=hc_graph,
                      full_top10=True, incremental_prefill_logits_bit_exact=True,
                      all_active_state_hash_equal=True, active_state_sha256=state_hash,
                      greedy_ids=generated, repeated_request_ids=repeated,
                      expert_statistics=model.expert_executor.statistics(),
                      pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      scope='Optimized-path self-consistency and retained three-token greedy smoke, not independent full-model reference equality.')
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--version', default='v2')
    parser.add_argument('--hc-graph', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args.child, hc_graph=args.hc_graph)
        return
    docs = ROOT/'docs/qwen4exp'
    output = docs/f'diagnostics_0233_lifecycle_{args.version}'
    result = docs/f'0233_fast_lifecycle_{args.version}.json'
    if output.exists() or result.exists():
        raise FileExistsError('Preserve previous artifacts')
    output.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), '--child', str(output)]
    if args.hc_graph:
        command.append('--hc-graph')
    run = bounded_gpu(command, output, 'lifecycle')
    report = json.loads((output/'result.json').read_text())
    report['run'] = run
    with result.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
