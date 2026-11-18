"""Bounded identical-input real packed expert specialization A/B; no full model."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous measurements')
    os.environ.update(MINI_LLM_BACKEND='cupy', MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global')
    from mini_llm.backend import xp, asnumpy
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
    from mini_llm.ops.grouped_quantized_experts import grouped_quantized_experts_forward
    from probe_0218_hc_identity import MODEL
    index = Qwen4ExpGGUFIndex.open(MODEL, model_id='0249')
    records, seen = [], set()
    x = xp.asarray(np.random.default_rng(249).normal(0, .1, (1,2560)).astype(np.float32))
    routes = xp.asarray(np.arange(1,11,dtype=np.float32)/55)
    try:
        for layer in range(index.n_layers):
            formats = tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type
                            for p in ('gate','up','down'))
            if formats in seen:
                continue
            seen.add(formats)
            experts = [PackedExpertFFNInference(*(packed_to_device(w) for w in
                       index.provider(ExpertKey(index.model_id, layer, int(e)))))
                       for e in np.linspace(0,511,10,dtype=int)]
            values, times = {}, {'warp': [], 'warp_specialized': []}
            def call(mode):
                os.environ['MINI_LLM_IQ_GPU_GROUPED_KERNEL'] = mode
                return grouped_quantized_experts_forward(x, experts, routes)
            for mode in times:
                values[mode] = asnumpy(call(mode))
            a, b = values['warp_specialized'].astype(np.float64), values['warp'].astype(np.float64)
            absolute = float(np.max(np.abs(a-b)))
            relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
            np.testing.assert_allclose(a,b,atol=1e-7,rtol=2e-5)
            for repeat in range(8):
                for mode in (('warp','warp_specialized') if repeat%2 == 0
                             else ('warp_specialized','warp')):
                    xp.cuda.get_current_stream().synchronize()
                    start, end = xp.cuda.Event(), xp.cuda.Event()
                    start.record()
                    wall = time.perf_counter()
                    for _ in range(20):
                        call(mode)
                    end.record()
                    end.synchronize()
                    times[mode].append(dict(event_ms=xp.cuda.get_elapsed_time(start,end)/20,
                                           wall_ms=(time.perf_counter()-wall)*1000/20))
            records.append(dict(layer=layer,formats=formats,max_absolute=absolute,relative_l2=relative,
                                bit_exact=bool(np.array_equal(a,b)),samples=times))
            del experts
            gc.collect()
        if len(records) != 7:
            raise AssertionError(f'Expected seven checkpoint format triples, got {len(records)}')
    finally:
        index.store.close()
    report = dict(records=records,scope='Resident packed top10 operators; not model throughput',
                  device_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes())
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps([dict(layer=r['layer'],formats=r['formats'],bit_exact=r['bit_exact'],
        speedup=float(np.median([s['wall_ms'] for s in r['samples']['warp']]) /
                      np.median([s['wall_ms'] for s in r['samples']['warp_specialized']])))
                      for r in records],indent=2))


if __name__ == '__main__':
    main()
