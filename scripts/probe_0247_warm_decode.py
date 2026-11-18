"""Bounded warm-cache fixed-prefix decode; optional Nsight capture window.

Run under probe_0227.bounded_gpu (20 GiB RSS / 120 s) for full checkpoints.
No model download, warm-all-expert sweep, changed arithmetic, or route skipping.
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
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--passes', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.passes <= 4:
        parser.error('passes must be 1..4')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(MINI_LLM_BACKEND='cupy', MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
                      MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp')
    import numpy as np
    from mini_llm.backend import xp, asnumpy
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from probe_0218_hc_identity import MODEL
    from probe_0227_short_sequences_and_profile import digest_state
    records, snapshots = [], []
    with Qwen4ExpGGUFModel.open(MODEL, max_length=16, cpu_workers=4,
            static_device_budget_bytes=6*1024**3, device_budget_bytes=14*1024**3,
            cache_bytes=6*1024**3, expert_promote_after=1, group_after_admission=True,
            expert_cache_slots=True, qsa_fast_decode=True, hc_fast_decode=True,
            router_fast_decode=True, gdn_fast_decode=True) as model:
        for repeat in range(args.passes + 1):
            state = model.create_generation_state(1, 16)
            before = model.expert_executor.statistics()
            if args.capture and repeat == 1:
                xp.cuda.profiler.start()
            times = []
            for position in range(16):
                xp.cuda.nvtx.RangePush(f'pass{repeat}/token{position}')
                start = time.perf_counter()
                logits = model.step(position + 1, state)
                xp.cuda.get_current_stream().synchronize()
                times.append(time.perf_counter() - start)
                xp.cuda.nvtx.RangePop()
            if args.capture and repeat == 1:
                xp.cuda.profiler.stop()
            values = asnumpy(logits)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite logits')
            np.save(args.output / f'logits_{repeat}.npy', values)
            snapshots.append(digest_state(state, asnumpy))
            records.append(dict(repeat=repeat, cold=repeat == 0, seconds=times,
                                tokens_per_second=16/sum(times), before=before,
                                after=model.expert_executor.statistics()))
            del state
            gc.collect()
        for repeat in range(1, args.passes + 1):
            np.testing.assert_array_equal(np.load(args.output/'logits_0.npy'),
                                          np.load(args.output/f'logits_{repeat}.npy'))
            if snapshots[repeat] != snapshots[0]:
                raise AssertionError('Warm repeat state differs')
        report = dict(records=records, state_hashes=snapshots,
                      repeated_logits_and_states_bit_exact=True,
                      scope='forced IDs1..16, FP32 full top10, no speculation; NOT prolonged chat',
                      device_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes())
    (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(rates=[r['tokens_per_second'] for r in records],
                         repeated_logits_and_states_bit_exact=True)))


if __name__ == '__main__':
    main()
