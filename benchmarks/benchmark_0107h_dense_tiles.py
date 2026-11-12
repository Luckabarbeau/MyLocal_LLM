#!/usr/bin/env python3
"""One HIP-only, process-local dense backward 32x32 experiment; source unchanged."""
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
OUTPUT = ROOT / 'logs/r9700_dense_tile32_stage1_candidate.json'


def main():
    import cupy as cp
    import mini_llm.ops.triton_dual_resolution_dense as dense
    import benchmark_dual_resolution_full_model_0102d as full

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('HIP required; no CUDA launch policy is changed')
    for key, expected in {'MINI_LLM_HIP_SPLIT_KV_BACKWARD': '2',
                          'MINI_LLM_HIP_FP32_PROJECTION_WGRAD': '1'}.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f'{key}={expected} required')
    result = {
        'candidate': 'HIP global/sparse backward only BM32 BN32',
        'environment': {k: v for k, v in os.environ.items() if k.startswith('MINI_LLM_')},
        'hip': True, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'seed': 107, 'atol': 1e-5, 'rtol': 1e-5,
        'warmup': 1, 'repeats': 3, 'correctness': [], 'compiled_resources': {},
        'isolated_backward': {}, 'status': 'running',
        'reference_note': 'Independent baseline executions/storage, unchanged installed launches; not a mathematical oracle.',
        'graph': 'P=masked_softmax(QK^T*scale); O=PV; delta=sum(O*dO); dS=P*(dO*V^T-delta); dQ=scale*dS*K; dK=scale*dS^T*Q; dV=P^T*dO. BF16 dot operands retain installed casts.',
    }
    # The comparison must retain the original combined reference even after
    # production adopts HIP-specific defaults.
    os.environ['MINI_LLM_HIP_DENSE_KV_BACKWARD'] = '0'
    os.environ['MINI_LLM_HIP_DENSE_BACKWARD_TILES'] = '0'
    original = {name: getattr(dense, name) for name in ('_dq', '_dkdv')}
    start_time = perf_counter()

    def save():
        result['elapsed_s'] = perf_counter() - start_time
        OUTPUT.write_text(json.dumps(result, indent=2) + '\n')

    class Kernel:
        def __init__(self, name, mode, events):
            self.name, self.mode, self.events = name, mode, events

        def __getitem__(self, grid):
            def launch(*values, **options):
                is_candidate = (self.mode == 'candidate' and cp.cuda.runtime.is_hip
                                and options['GLOBAL'])
                if is_candidate:
                    old_bm = options['BM']
                    first_query = options['Q_TILE_OFFSET'] * old_bm
                    assert first_query % 32 == 0
                    options['Q_TILE_OFFSET'] = first_query // 32
                    options['BM'] = options['BN'] = 32
                    options['num_stages'] = 1
                    if self.name == '_dq':
                        count = (max(0, options['NQ'] - first_query) + 31) // 32
                    else:
                        count = (options['NK'] + 31) // 32
                    grid_new = (count, *grid[1:])
                else:
                    grid_new = grid
                branch = 'sparse' if options['SPARSE_STRIDE'] else 'global' if options['GLOBAL'] else 'compressor'
                label = f'{branch}.{self.name}'
                resource_key = (f'{self.mode}.{label}.NQ{options["NQ"]}.NK{options["NK"]}'
                                f'.group{options["GROUP"]}.phase{options["SPARSE_PHASE"]}'
                                f'.bounded{int(options["BOUNDED"])}'
                                f'.add{int(options.get("ADD_DKDV", False))}')
                if self.events is not None:
                    before, after = cp.cuda.Event(), cp.cuda.Event()
                    before.record()
                compiled = original[self.name][grid_new](*values, **options)
                if self.events is not None:
                    after.record()
                    self.events.append((label, before, after))
                if resource_key not in result['compiled_resources']:
                    result['compiled_resources'][resource_key] = {
                        'n_regs': getattr(compiled, 'n_regs', None),
                        'n_spills': getattr(compiled, 'n_spills', None),
                        'shared_bytes': getattr(compiled.metadata, 'shared', None),
                        'BM': options['BM'], 'BN': options['BN'],
                        'grid': list(grid_new), 'Q_TILE_OFFSET': options['Q_TILE_OFFSET'],
                        'num_warps': options['num_warps'], 'num_stages': options['num_stages'],
                        'BF16': options['BF16'], 'SKIP_EMPTY': options['SKIP_EMPTY'],
                        'ADD_DKDV': options.get('ADD_DKDV'),
                    }
                return compiled
            return launch

    def wrappers(mode, events=None):
        stack = ExitStack()
        for name in original:
            stack.enter_context(patch.object(dense, name, Kernel(name, mode, events)))
        return stack

    def compare(got, want):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        error = np.abs(got - want)
        threshold = 1e-5 + 1e-5 * np.abs(want)
        index = np.unravel_index(int(np.argmax(error)), error.shape)
        return {
            'max_absolute': float(error.max()),
            'max_pointwise_relative': float((error / np.maximum(np.abs(want), 1e-300)).max()),
            'relative_l2': float(np.linalg.norm(error) / max(np.linalg.norm(want), 1e-300)),
            'worst_index': list(map(int, index)),
            'worst_actual': float(got[index]), 'worst_reference': float(want[index]),
            'worst_absolute_relative': float(error[index] / max(abs(want[index]), 1e-300)),
            'max_tolerance_ratio': float((error / threshold).max()),
            'outside_tolerance': int((error > threshold).sum()),
            'allclose': bool(np.allclose(got, want, atol=1e-5, rtol=1e-5)),
            'dtype': 'float32', 'inputs': 'bfloat16', 'finite_difference_step': None,
        }

    rng = np.random.default_rng(107)

    def bf16(shape):
        return cp.asarray(rng.normal(0, .2, shape), dtype=cp.float32).astype('bfloat16')

    def make_case(branch, length, packed=False, bounded=True):
        batch = 2 if packed else 1
        docs_host = np.zeros((batch, length), np.int32)
        if packed:
            docs_host[:, 257:] = 1
            docs_host[:, 599:] = 2
        docs = cp.asarray(docs_host)
        if branch == 'sparse':
            phase, stride = 5 if packed else 0, 16
            # The model's query head and stride-selected shared KV head are views.
            qkv = bf16((batch, length * 2, 24, 64))[:, ::2]
            q, k, v = qkv[:, :, 1:2], qkv[:, phase::stride, 16:17], qkv[:, phase::stride, 20:21]
            kwargs = dict(documents=docs, bounded_causal=bounded,
                          sparse_stride=stride, sparse_phase=phase)
            dy = cp.asarray(rng.normal(0, .1, (batch, length * 2, 16, 64)),
                            dtype=cp.float32)[:, ::2, 1:2]
        else:
            heads = 4
            nk = length // 128 * 4
            q = bf16((batch, length * 2, heads * 2, 64))[:, ::2, ::2]
            kv = bf16((batch, nk * 2, heads * 2, 64))[:, ::2]
            k, v = kv[:, :, :heads], kv[:, :, heads:]
            positions = np.zeros_like(docs_host)
            block_docs = np.zeros((batch, nk), np.int32)
            ends = np.zeros_like(block_docs)
            valid = np.zeros_like(block_docs)
            for row in range(batch):
                first = 0
                for i in range(length):
                    if i and docs_host[row, i] != docs_host[row, i - 1]:
                        first = i
                    positions[row, i] = i - first
                for key in range(nk):
                    begin = key // 4 * 128
                    end = begin + 127
                    block_docs[row, key] = docs_host[row, end]
                    ends[row, key] = positions[row, end]
                    valid[row, key] = int(docs_host[row, begin] == docs_host[row, end])
            kwargs = dict(documents=docs, positions=cp.asarray(positions),
                          block_documents=cp.asarray(block_docs),
                          block_ends=cp.asarray(ends), block_valid=cp.asarray(valid),
                          bounded_causal=bounded, history_block_size=128, summaries_per_block=4)
            dy = cp.asarray(rng.normal(0, .1, (batch, length * 2, heads, 64)),
                            dtype=cp.float32)[:, ::2]

        def execute(add=False):
            output, cache = dense.forward(q, k, v, scale=.125, **kwargs)
            if branch == 'sparse':
                # Nonzero shared-local gradients, plus canaries outside owned views.
                qs = cp.full((batch, length * 2, 16, 64), 17., cp.float32)
                ks = cp.full((batch, length * 2, 4, 64), .25, cp.float32)
                vs = cp.full((batch, length * 2, 4, 64), -.5, cp.float32)
                dq = qs[:, ::2, 1:2]
                dk = ks[:, ::2][:, phase::stride, :1]
                dv = vs[:, ::2][:, phase::stride, :1]
                grads = dense.backward(dy, cache, dq=dq, dk=dk, dv=dv, add_dkdv=add)
                assert all(a is b for a, b in zip(grads, (dq, dk, dv)))
                # Check every unowned location without erasing the owned result.
                qmask = cp.ones(qs.shape[:-1], cp.bool_)
                kmask = cp.ones(ks.shape[:-1], cp.bool_)
                qmask[:, ::2, 1:2] = False
                kmask[:, ::2][:, phase::stride, :1] = False
                assert bool(cp.all(qs[qmask] == 17.).item())
                assert bool(cp.all(ks[kmask] == .25).item())
                assert bool(cp.all(vs[kmask] == -.5).item())
            else:
                grads = dense.backward(dy, cache)
            return output, cache, grads
        return execute

    try:
        # Partial tiles, packed EOS with no valid early history, bounded offset,
        # and sparse ADD_DKDV both false/true; no tolerance fallback.
        specs = [('global', 641, False, True), ('global', 641, True, False),
                 ('sparse', 641, True, True), ('global', 8192, False, True),
                 ('sparse', 8192, False, True)]
        for branch, length, packed, bounded in specs:
            execute = make_case(branch, length, packed, bounded)
            for add in ((False, True) if branch == 'sparse' else (False,)):
                label = f'{branch}_{length}_packed{int(packed)}_bounded{int(bounded)}_add{int(add)}'
                with wrappers('baseline'):
                    output, cache, grads = execute(add)
                    cp.cuda.runtime.deviceSynchronize()
                    # CPU snapshots precede candidate allocation/execution.
                    references = [cp.asnumpy(a) for a in (output, *grads)]
                del output, cache, grads
                with wrappers('candidate'):
                    output, cache, grads = execute(add)
                    cp.cuda.runtime.deviceSynchronize()
                    errors = {name: compare(cp.asnumpy(a), b) for name, a, b in
                              zip(('forward', 'dq', 'dk', 'dv'), (output, *grads), references)}
                    repeat = execute(add)
                    deterministic = all(np.array_equal(cp.asnumpy(a), cp.asnumpy(b))
                                        for a, b in zip(grads, repeat[2]))
                row = {'case': label, 'errors': errors, 'repeat_bitwise_equal': deterministic,
                       'passed': all(e['allclose'] for e in errors.values()) and deterministic}
                result['correctness'].append(row)
                print(json.dumps(row), flush=True)
                save()
                if not row['passed']:
                    raise AssertionError(f'{label}: gradient/forward mismatch; no tolerance relaxation')
                del output, cache, grads, repeat, references

            if length == 8192:
                # Representative device-only launches, separate from unprofiled model.
                output, cache, _ = execute(branch == 'sparse')
                dy = cp.asarray(rng.normal(0, .1, cache.q.shape), dtype=cp.float32)
                for mode in ('baseline', 'candidate'):
                    events = []
                    with wrappers(mode, events):
                        dense.backward(dy, cache)  # warm once for this launch shape
                        cp.cuda.runtime.deviceSynchronize()
                        events.clear()
                        for _ in range(3):
                            dense.backward(dy, cache)
                        cp.cuda.runtime.deviceSynchronize()
                    times = {}
                    for name, before, after in events:
                        times.setdefault(name, []).append(float(cp.cuda.get_elapsed_time(before, after)))
                    result['isolated_backward'][f'{mode}.{branch}'] = {
                        'samples_ms': times, 'kernel_median_ms': {k: median(v) for k, v in times.items()},
                        'note': 'dQ+dKdV device events only; fresh non-additive output buffers',
                    }
                del output, cache, dy
            del execute
        result['correctness_passed'] = True
        save()
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        args = full.parse_args([
            '--architecture', 'matrix-pool-sparse', '--context', '8192',
            '--model-context', '65536', '--score-last-k', '8192',
            '--head-chunk-tokens', '8192', '--activation-checkpoint', 'none',
            '--runtime-profile', 'consumer-gpu', '--query-chunk', '1536',
            '--warmup-iterations', '1', '--repeat-iterations', '3',
        ])
        for mode in ('baseline', 'candidate'):
            print(f'FULL MODEL {mode}: warmup 1, repeats 3', flush=True)
            with wrappers(mode):
                result[f'full_model_{mode}'] = full.run(args)
            save()
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        baseline, candidate = result['full_model_baseline'], result['full_model_candidate']
        result['comparison'] = {
            'throughput_percent': 100 * (candidate['full_model_forward_backward_tok_per_s'] /
                                        baseline['full_model_forward_backward_tok_per_s'] - 1),
            'backward_ms_difference': candidate['backward_ms'] - baseline['backward_ms'],
            'loss_difference': candidate['loss'] - baseline['loss'],
            'historical_baseline_report': 'logs/r9700_8k_all_hip_candidates.json',
        }
        result['status'] = 'passed'
        save()
        print(json.dumps({'status': result['status'], 'comparison': result['comparison'],
                          'report': str(OUTPUT)}, indent=2), flush=True)
    except BaseException as exc:
        result['status'] = 'failed'
        result['failure'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
