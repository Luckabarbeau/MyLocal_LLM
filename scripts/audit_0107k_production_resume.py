#!/usr/bin/env python3
"""Audit one real Model B training update without changing its CLI options.

Run through scripts/run_r9700_preflight.sh, supplying --audit-report and the
usual train_model.py options. Requires a fresh destination and --max-run-steps 1.
Hooks fail on fused fallback; the report records actual, variable-shape counts.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', type=Path, required=True)
    args, training_args = parser.parse_known_args()
    if training_args[:1] == ['--']:
        training_args = training_args[1:]

    def option(name):
        for index, value in enumerate(training_args):
            if value.startswith(name + '='):
                return value.split('=', 1)[1]
            if value == name and index + 1 < len(training_args):
                return training_args[index + 1]
        raise ValueError(f'{name} must be explicitly supplied')

    if option('--max-run-steps') != '1':
        raise ValueError('This production audit is restricted to one optimizer update')
    source = Path(option('--resume-from')).resolve()
    destination = Path(option('--checkpoint-dir')).resolve()
    if source == destination or destination.exists():
        raise ValueError('Use a fresh output checkpoint directory, not the resume source')
    if args.audit_report.exists():
        raise FileExistsError(args.audit_report)
    for name in ('--log-file', '--component-update-output'):
        if Path(option(name)).exists():
            raise FileExistsError(option(name))

    from probe_0107j_amd_fastpaths import install_audit
    counts = install_audit()
    import mini_llm.ops.matrix_pool_dual_resolution_attention as matrix
    original_pool = matrix.MatrixPoolDualResolutionAttention._pool_backward
    def checked_pool(self, *values, **options):
        counts['pool_backward_dispatch.calls'] += 1
        return original_pool(self, *values, **options)
    matrix.MatrixPoolDualResolutionAttention._pool_backward = checked_pool

    required = (
        'qkv_rope_forward.executed', 'norm_backward.executed',
        'rope_backward.executed', 'swiglu_forward_or_recompute.executed',
        'swiglu_backward.executed', 'pool_backward.executed',
        'fp32_projection_wgrad.executed', 'local_separate_kv.executed',
        'global_or_sparse_segmented_kv.executed', 'global.bf16_forward',
        'sparse.bf16_forward', 'ce.bf16_cross_entropy_loss_only',
        'ce.bf16_cross_entropy_grad_from_logits',
    )
    start = time.perf_counter()
    report = {'status': 'failed', 'resume_from': str(source),
              'checkpoint_dir': str(destination), 'training_args': training_args}
    try:
        sys.argv = [str(ROOT / 'train_model.py'), *training_args]
        runpy.run_path(str(ROOT / 'train_model.py'), run_name='__main__')
        missing = [name for name in required if counts[name] < 1]
        assert not missing, f'Optimizations not observed: {missing}'
        assert counts['pool_backward_dispatch.calls'] == counts['pool_backward.executed'], (
            'Pooling reference fallback observed', dict(counts))
        report['status'] = 'passed'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        report['counts'] = dict(counts)
        report['hip_runtime_policy'] = {
            k: v for k, v in os.environ.items() if k.startswith('MINI_LLM_HIP_')
        }
        with args.audit_report.open('x') as handle:
            json.dump(report, handle, indent=2)
        print('Production fast-path audit:', json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
