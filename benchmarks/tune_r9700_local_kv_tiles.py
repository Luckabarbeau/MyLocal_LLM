#!/usr/bin/env python3
"""One-shot HIP local-KV tile sweep; production kernels remain unchanged."""
import json
import os
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'logs/r9700_local_kv_tiles_20261113.json'
BASELINE = (32, 32)
CANDIDATES = ((16, 32), (32, 16), (32, 64), (64, 32), (64, 64))


def main():
    # Reserve exclusively before importing GPU libraries or doing any GPU work.
    with OUTPUT.open('x') as handle:
        result = {'status': 'initializing', 'measurements': []}

        def save():
            handle.seek(0)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

        save()
        original = None
        try:
            for name, expected in (('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '2'),
                                   ('MINI_LLM_HIP_DQ_LOOP_STAGES', '1')):
                if os.environ.get(name, expected) != expected:
                    raise RuntimeError(f'{name} must be unset or {expected}')
            import cupy as cp
            import triton
            import mini_llm.ops.triton_dual_resolution_local as local
            import mini_llm.ops.triton_hip_local_kv as kv

            if not cp.cuda.runtime.is_hip:
                raise RuntimeError('HIP only')
            stream = cp.cuda.get_current_stream()
            result.update(status='running', shape={'B': 1, 'T': 8192, 'Q': 16, 'KV': 4, 'D': 64},
                          dtype='bfloat16', upstream_dtype='float32', seed=107,
                          window=512, long_heads=1, long_window=4096,
                          baseline=list(BASELINE), candidates=list(CANDIDATES),
                          num_warps=4, num_stages=1, actual_loop_stages=1,
                          warmup=2, repeats=7, atol=1e-5, rtol=1e-5,
                          normwise_tolerance=1e-3, stream=int(stream.ptr),
                          device=str(cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name']),
                          reference='unwrapped production complete backward',
                          methodology='Independent DK and DV sweeps; other kernel stays (32,32). GPU events on current stream; no dQ tuning.')
            save()
            rng = np.random.default_rng(107)

            def values(heads):
                return cp.asarray(rng.normal(0, 0.2, (1, 8192, heads, 64)),
                                  dtype=cp.float32).astype('bfloat16')

            q, k, v = values(16), values(4), values(4)
            docs = cp.zeros((1, 8192), dtype=cp.int32)
            dy = cp.asarray(rng.normal(0, 0.1, q.shape), dtype=cp.float32)
            _, cache = local.local_attention_forward(q, k, v, docs, window=512,
                                                     long_window=4096, long_heads=1)
            reference = local.local_attention_backward(dy, cache)
            stream.synchronize()
            reference_cpu = [array.get(stream=stream).astype(np.float64) for array in reference]
            original = kv._kv_kernel
            tiles = {'dk': BASELINE, 'dv': BASELINE}
            events, resources = [], {}

            class WrappedKernel:
                def __getitem__(self, grid):
                    def launch(*args, **options):
                        label = 'dk' if options['compute_dk'] else 'dv'
                        block_m, block_n = tiles[label]
                        options.update(block_m=block_m, block_n=block_n,
                                       num_warps=4, num_stages=1)
                        actual_grid = (triton.cdiv(options['n_ctx'], block_n), *grid[1:])
                        start, end = cp.cuda.Event(), cp.cuda.Event()
                        start.record(stream)
                        compiled = original[actual_grid](*args, **options)
                        end.record(stream)
                        events.append((label, start, end))
                        resources[label] = {
                            'block_m': block_m, 'block_n': block_n, 'grid': list(actual_grid),
                            'nregs': getattr(compiled, 'n_regs', None),
                            'nspills': getattr(compiled, 'n_spills', None),
                            'shared_bytes': getattr(compiled.metadata, 'shared', None),
                            'num_warps': 4, 'launch_num_stages': 1,
                            'compiled_num_stages': getattr(compiled.metadata, 'num_stages', None),
                            'actual_loop_stages': 1,
                        }
                        return compiled
                    return launch

            def parity(actual):
                checks = {}
                for name, array, expected in zip(('dq', 'dk', 'dv'), actual, reference_cpu):
                    observed = array.get(stream=stream).astype(np.float64)
                    absolute = np.abs(observed - expected)
                    limit = 1e-5 + 1e-5 * np.abs(expected)
                    finite = np.isfinite(observed) & np.isfinite(expected)
                    failed = (~finite) | (absolute > limit)
                    error_norm = float(np.linalg.norm((observed - expected).ravel()))
                    reference_norm = float(np.linalg.norm(expected.ravel()))
                    ratio = error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else float('inf'))
                    check = {'dtype': str(array.dtype), 'elements': int(expected.size),
                             'max_abs_error': float(absolute.max()),
                             'error_l2': error_norm, 'reference_l2': reference_norm,
                             'relative_l2': ratio, 'allclose': bool(not failed.any()),
                             'normwise_pass': bool(np.isfinite(ratio) and ratio <= 1e-3),
                             'mismatch_count': int(failed.sum())}
                    if failed.any() or not check['normwise_pass']:
                        # Include every offending element, not a truncated assertion message.
                        indices = np.argwhere(failed if failed.any() else absolute != 0)
                        check['discrepancies'] = [
                            {'index': index.tolist(), 'actual': float(observed[tuple(index)]),
                             'reference': float(expected[tuple(index)]),
                             'absolute_error': float(absolute[tuple(index)]),
                             'allowed_error': float(limit[tuple(index)])}
                            for index in indices]
                    checks[name] = check
                return checks

            kv._kv_kernel = WrappedKernel()
            schedule = [('baseline', BASELINE)] + [(label, config) for label in ('dk', 'dv') for config in CANDIDATES]
            for target, config in schedule:
                tiles.update(dk=BASELINE, dv=BASELINE)
                if target != 'baseline':
                    tiles[target] = config
                events.clear()
                actual = local.local_attention_backward(dy, cache)
                stream.synchronize()
                checks = parity(actual)
                row = {'target': target, 'tiles': {key: list(value) for key, value in tiles.items()},
                       'parity': checks, 'resources': dict(resources)}
                result['measurements'].append(row)
                if not all(check['allclose'] and check['normwise_pass'] for check in checks.values()):
                    result['status'] = 'numerical_failure'
                    save()
                    print(f'Numerical failure: {target} {config}; full discrepancies saved to {OUTPUT}', flush=True)
                    return 1
                samples = {'dk': [], 'dv': []}
                for index in range(2 + 7):
                    events.clear()
                    actual = local.local_attention_backward(dy, cache)
                    stream.synchronize()
                    if [label for label, _, _ in events] != ['dk', 'dv']:
                        raise RuntimeError('unexpected KV launch sequence')
                    if index >= 2:
                        for label, start, end in events:
                            samples[label].append(float(cp.cuda.get_elapsed_time(start, end)))
                row.update(samples_ms=samples, median_ms={key: median(value) for key, value in samples.items()})
                save()
                print(json.dumps({'target': target, 'config': config, 'median_ms': row['median_ms']}), flush=True)
            result['best'] = {}
            for label in ('dk', 'dv'):
                eligible = [row for row in result['measurements'] if row['target'] in ('baseline', label)]
                best = min(eligible, key=lambda row: row['median_ms'][label])
                baseline_ms = result['measurements'][0]['median_ms'][label]
                result['best'][label] = {'config': best['tiles'][label],
                                         'median_ms': best['median_ms'][label],
                                         'baseline_ms': baseline_ms,
                                         'speedup': baseline_ms / best['median_ms'][label]}
            result['status'] = 'complete'
            save()
            print(json.dumps(result['best'], indent=2), flush=True)
            return 0
        except Exception as exc:
            result.update(status='error', error_type=type(exc).__name__, error=str(exc))
            save()
            raise
        finally:
            if original is not None:
                kv._kv_kernel = original


if __name__ == '__main__':
    raise SystemExit(main())
