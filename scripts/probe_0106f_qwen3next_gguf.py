#!/usr/bin/env python3
"""Probe a local Qwen3-Coder-Next GGUF without loading it into RAM/VRAM.

Optional --smoke-expert tests one real routed expert on the shared backend.
This is NOT a text-generation entrypoint.
"""

import argparse
import json
import sys
from pathlib import Path
import time

# Runnable directly from source checkout; no editable install necessary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Absolute path to existing GGUF')
    parser.add_argument('--device-budget-gib', type=float, default=6)
    parser.add_argument('--ram-budget-gib', type=float, default=32)
    parser.add_argument('--smoke-expert', action='store_true',
                        help='Execute one real routed expert on CPU (no model decoding)')
    parser.add_argument('--gpu-smoke', action='store_true',
                        help='Run one packed expert on CuPy/Triton GPU, compare with CPU reference')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--cpu-backend', choices=('reference', 'auto', 'native'), default='auto')
    args = parser.parse_args()
    index = Qwen3NextGGUFIndex.open(args.model)
    report = index.report(device_budget_gib=args.device_budget_gib,
                          ram_budget_gib=args.ram_budget_gib)
    if args.smoke_expert or args.gpu_smoke:
        import numpy as np
        from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
        x = np.random.default_rng(42).normal(0, 0.05, (1, index.d_model)).astype(np.float32)
        executor = HeterogeneousExpertExecutor(
            index.provider, model_id='qwen3-coder-next-gguf',
            d_model=index.d_model, d_ff=index.d_ff,
            device_budget_bytes=0, cpu_packed_backend=args.cpu_backend,
        )
        try:
            before = time.perf_counter()
            from mini_llm.backend import BACKEND_NAME, asnumpy, xp
            execution_input = xp.asarray(x) if BACKEND_NAME == 'cupy' else x
            y = asnumpy(executor.forward(args.layer, args.expert, execution_input))
            report['smoke_expert'] = {
                'layer': args.layer, 'expert': args.expert,
                'output_shape': list(y.shape), 'output_finite': bool(np.isfinite(y).all()),
                'elapsed_ms': round((time.perf_counter() - before) * 1000, 3),
                'execution_counters': executor.statistics()['execution'],
            }
        finally:
            executor.close()
        if args.gpu_smoke:
            from mini_llm.backend import BACKEND_NAME, asnumpy, xp
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('GPU smoke requires MINI_LLM_BACKEND=cupy')
            packed = index.expert_weights(args.layer, args.expert)
            expert_bytes = sum(w.nbytes for w in packed)
            gpu_executor = HeterogeneousExpertExecutor(
                index.provider, model_id='qwen3-coder-next-gguf',
                d_model=index.d_model, d_ff=index.d_ff, promote_after=1,
                device_budget_bytes=max(expert_bytes * 2, 16 * 1024**2),
            )
            try:
                gpu_x = xp.asarray(x.astype(np.float16))
                begin = time.perf_counter()
                gpu_y = gpu_executor.forward(args.layer, args.expert, gpu_x)
                xp.cuda.get_current_stream().synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                gpu_arr = asnumpy(gpu_y).astype(np.float32)
                report['gpu_smoke_expert'] = {
                    'elapsed_ms_including_first_compile_and_transfer': round(elapsed, 3),
                    'output_finite': bool(np.isfinite(gpu_arr).all()),
                    'max_absolute_difference_from_cpu': float(np.max(np.abs(gpu_arr - y))),
                    'tolerance_pass': bool(np.allclose(gpu_arr, y, rtol=0.02, atol=0.02)),
                    'cache': gpu_executor.statistics()['device'],
                }
            finally:
                gpu_executor.cache.clear()
                gpu_executor.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, IndexError, NotImplementedError) as exc:
        print(f'Qwen3Next GGUF probe: {exc}', file=sys.stderr)
        raise SystemExit(2)
