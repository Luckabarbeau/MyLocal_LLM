#!/usr/bin/env python3
"""Fail-on-fallback audit of the executed full-8K Model B HIP training paths.

Synthetic weights/tokens only; no checkpoint I/O. Instrumented timings are not
throughput results. Run with scripts/run_r9700_preflight.sh.
"""
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_audit():
    """Install process-local fail-on-fallback hooks and return execution counters."""
    from mini_llm.backend import xp
    if not xp.cuda.runtime.is_hip:
        raise RuntimeError('HIP required')
    counts = Counter()
    def module(name):
        return importlib.import_module('mini_llm.ops.' + name)

    def watch(owner, name, label):
        original = getattr(owner, name)
        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            counts[label + '.calls'] += 1
            assert result is not None, f'{label}: fused path fell back'
            counts[label + '.executed'] += 1
            return result
        setattr(owner, name, checked)

    norm = module('rmsnorm')
    original_norm = norm.RMSNorm.backward
    def checked_norm(self, dy, cache):
        assert cache.get('fused_bf16_residual') is True
        assert self._can_use_fused_bf16_backward(dy, cache)
        result = original_norm(self, dy, cache)
        counts['norm_backward.executed'] += 1
        return result
    norm.RMSNorm.backward = checked_norm
    watch(module('rope'), '_fused_rope_backward', 'rope_backward')
    watch(module('matrix_pool_dual_resolution_attention'),
          '_fused_packed_qkv_rope_forward_bf16', 'qkv_rope_forward')
    watch(module('experts'), '_fused_swiglu_forward', 'swiglu_forward_or_recompute')
    watch(module('experts'), '_fused_swiglu_backward', 'swiglu_backward')
    watch(module('triton_matrix_pool_backward_0102u'), 'backward', 'pool_backward')
    watch(module('hip_projection_wgrad'), 'projection_wgrad', 'fp32_projection_wgrad')
    watch(module('triton_hip_local_kv'), 'backward', 'local_separate_kv')
    watch(module('triton_hip_dense_kv'), 'backward', 'global_or_sparse_segmented_kv')

    dense = module('triton_dual_resolution_dense')
    original_dense = dense.forward
    def checked_dense(*args, **kwargs):
        result = original_dense(*args, **kwargs)
        cache = result[1]
        assert all(str(a.dtype) == 'bfloat16' for a in (cache.q, cache.k, cache.v))
        counts[('sparse' if cache.sparse_stride else 'global') + '.bf16_forward'] += 1
        return result
    dense.forward = checked_dense

    loss = module('loss')
    original_getter = loss._get_fused_bf16_ce_module
    class CountedModule:
        def __init__(self, original):
            self.original = original
        def get_function(self, name):
            function = self.original.get_function(name)
            def launch(*args, **kwargs):
                counts['ce.' + name] += 1
                return function(*args, **kwargs)
            return launch
        def __getattr__(self, name):
            return getattr(self.original, name)
    def checked_getter():
        compiled = original_getter()
        assert compiled is not None, 'Fused CE unavailable'
        return CountedModule(compiled)
    loss._get_fused_bf16_ce_module = checked_getter
    return counts


def main():
    from benchmarks.benchmark_dual_resolution_full_model_0102d import parse_args, run
    counts = install_audit()
    args = parse_args([
        '--architecture', 'matrix-pool-sparse', '--context', '8192',
        '--model-context', '65536', '--score-last-k', '8192',
        '--head-chunk-tokens', '8192', '--activation-checkpoint', 'none',
        '--runtime-profile', 'consumer-gpu', '--query-chunk', '1536',
        '--warmup-iterations', '0', '--repeat-iterations', '1',
    ])
    run(args)
    expected = {
        'qkv_rope_forward.executed': 20, 'norm_backward.executed': 41,
        'rope_backward.executed': 80, 'swiglu_forward_or_recompute.executed': 40,
        'swiglu_backward.executed': 20, 'pool_backward.executed': 20,
        'fp32_projection_wgrad.executed': 40, 'local_separate_kv.executed': 20,
        'global_or_sparse_segmented_kv.executed': 36,
        'global.bf16_forward': 20, 'sparse.bf16_forward': 16,
        'ce.bf16_cross_entropy_loss_only': 1,
        'ce.bf16_cross_entropy_grad_from_logits': 1,
    }
    for name, expected_count in expected.items():
        assert counts[name] == expected_count, (name, counts[name], expected_count)
    report = {'status': 'passed', 'counts': dict(counts), 'expected': expected}
    Path('logs/r9700_fastpaths_final.json').write_text(json.dumps(report, indent=2))
    print('PASS: full-model fast-path activation', json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    main()
