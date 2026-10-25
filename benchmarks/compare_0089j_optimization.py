#!/usr/bin/env python3
"""Fail-closed baseline-vs-opt-in comparison of matched 0089A production probes.

Only an explicitly selected single optimization flag is allowed to differ.
Reports must identify exactly the same sampled inputs, loss masks, router seed,
checkpoint, backend and safe no-update execution. No GPU is needed to compare.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FLAGS = {
    'history': 'MINI_LLM_HISTORY_BF16_BACKWARD',
    'ffn-dx': 'MINI_LLM_EXPERT_DX_GEMM_BETA',
    'ffn-wgrad': 'MINI_LLM_EXPERT_WGRAD_FP32_GEMM',
    'cross-block': 'MINI_LLM_FUSED_CROSS_BLOCK_NORM1',
}
HASHES = ('inputs', 'targets', 'loss_mask', 'history_valid_starts')
MODEL_FIELDS = ('checkpoint', 'parameter_count', 'layers', 'd_model', 'd_ff',
                'q_heads', 'kv_heads', 'dtype', 'activation_checkpoint',
                'lm_head_chunk_tokens')
SAFETY = ('optimizer_constructed', 'optimizer_update', 'checkpoint_written',
          'weights_modified')


def _enabled(value):
    return str(value).strip().lower() not in ('none', '', '0', 'no', 'off', 'false')


def _identities(report):
    values = []
    for batch in report['workload']['batches']:
        values.append({
            'working_length': batch['compact_working_length'],
            'computed_rows': batch['computed_rows'],
            'supervised_rows': batch['supervised_rows'],
            'seed': batch['router_gumbel_seed'],
            'hashes': {key: batch['fixture_hashes'][key] for key in HASHES},
        })
    return values


def compare(baseline, candidate, flag, *, loss_abs_tol=5e-3,
            loss_rel_tol=2e-3):
    if flag not in FLAGS:
        raise ValueError(f'unsupported optimization {flag!r}')
    selected = FLAGS[flag]
    problems = []
    for label, r in [('baseline', baseline), ('candidate', candidate)]:
        if r.get('schema_version') != '0089A-v1' or r.get('status') != 'measured':
            problems.append(f'{label}: unsupported or unmeasured report')
        if any(bool(r.get('safety', {}).get(key, True)) for key in SAFETY):
            problems.append(f'{label}: unsafe model state changes')
        if r.get('timing', {}).get('profiler_perturbed', False):
            problems.append(f'{label}: profiler-perturbed timing')
    if any(baseline.get('model', {}).get(key) != candidate.get('model', {}).get(key)
           for key in MODEL_FIELDS):
        problems.append('model/checkpoint/config differs')
    if baseline.get('system') != candidate.get('system'):
        problems.append('hardware/software environment differs')
    for key in ('attention_backend', 'attention_strict', 'profile_resolved'):
        if baseline['runtime'].get(key) != candidate['runtime'].get(key):
            problems.append(f'runtime configuration differs: {key}')
    if baseline['runtime'].get('attention_backend') != 'triton' or not baseline['runtime'].get('attention_strict'):
        problems.append('expected strict Triton attention in both reports')
    for key in ('batch_size', 'microbatches_per_sample', 'requested_working_length',
                'total_supervised_rows_per_sample', 'total_computed_rows_per_sample'):
        if baseline['workload'].get(key) != candidate['workload'].get(key):
            problems.append(f'workload differs: {key}')
    if _identities(baseline) != _identities(candidate):
        problems.append('fixture batch hashes/seed/computed rows differ')
    lhs = baseline['runtime']['effective_flags']
    rhs = candidate['runtime']['effective_flags']
    if selected not in lhs or selected not in rhs:
        problems.append(f'optimization flag {selected} missing from reports')
    elif _enabled(lhs[selected]) or not _enabled(rhs[selected]):
        problems.append(f'{selected} must be disabled in baseline, enabled in candidate')
    for name in set(lhs) | set(rhs):
        if name != selected and lhs.get(name) != rhs.get(name):
            problems.append(f'confounding runtime flag differs: {name}')
    loss_a = float(baseline['last_loss_device_resident_until_boundary'])
    loss_b = float(candidate['last_loss_device_resident_until_boundary'])
    loss_abs = abs(loss_b - loss_a)
    loss_rel = loss_abs / max(abs(loss_a), abs(loss_b), 1e-12)
    if not all(math.isfinite(x) for x in (loss_a, loss_b, loss_abs, loss_rel)) or (
            loss_abs > loss_abs_tol or loss_rel > loss_rel_tol):
        problems.append('last scalar loss fails declared parity tolerance')
    ref_time = float(baseline['timing']['gpu_event']['median_ms'])
    new_time = float(candidate['timing']['gpu_event']['median_ms'])
    if not all(math.isfinite(x) and x > 0 for x in (ref_time, new_time)):
        problems.append('invalid GPU-event timing')
    count = baseline['workload'].get('total_supervised_rows_per_sample')
    accepted = not problems
    return {
        'schema_version': '0089J-optin-comparison-v1',
        'optimization': flag,
        'flag': selected,
        'status': 'valid' if accepted else 'invalid',
        'errors': problems,
        'parity': {
            'scalar_loss_abs': loss_abs,
            'scalar_loss_rel': loss_rel,
            'abs_tol': loss_abs_tol,
            'rel_tol': loss_rel_tol,
            'passed': loss_abs <= loss_abs_tol and loss_rel <= loss_rel_tol,
            'note': 'scalar loss alone does not establish parameter-gradient parity',
        },
        'timing': {
            'baseline_gpu_median_ms': ref_time,
            'candidate_gpu_median_ms': new_time,
            'speedup': ref_time / new_time if accepted else None,
            'latency_reduction_fraction': (ref_time - new_time) / ref_time if accepted else None,
            'baseline_supervised_tokens_per_s': count * 1000 / ref_time if accepted else None,
            'candidate_supervised_tokens_per_s': count * 1000 / new_time if accepted else None,
            'note': 'no optimizer, zero_grad or checkpoint I/O in measured scope',
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--optimization', required=True, choices=sorted(FLAGS))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--loss-abs-tol', type=float, default=5e-3)
    parser.add_argument('--loss-rel-tol', type=float, default=2e-3)
    args = parser.parse_args(argv)
    if args.loss_abs_tol < 0 or args.loss_rel_tol < 0:
        parser.error('loss tolerances must be nonnegative')
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.optimization,
        loss_abs_tol=args.loss_abs_tol,
        loss_rel_tol=args.loss_rel_tol,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'valid' else 2


if __name__ == '__main__':
    raise SystemExit(main())
