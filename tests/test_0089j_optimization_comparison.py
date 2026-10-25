"""0089J: matched report comparison, fail-closed on confounders."""
import copy
from benchmarks.compare_0089j_optimization import FLAGS, compare


def _report(flag='history', enabled=False):
    return {
        'schema_version': '0089A-v1', 'status': 'measured',
        'safety': {key: False for key in (
            'optimizer_constructed', 'optimizer_update',
            'checkpoint_written', 'weights_modified')},
        'system': {'device': 'RTX 5080', 'cupy': '14.2', 'cuda_runtime': 13200},
        'model': {
            'checkpoint': 'checkpoints/338m', 'parameter_count': 338000000,
            'layers': 20, 'd_model': 1024, 'd_ff': 4096, 'q_heads': 16,
            'kv_heads': 4, 'dtype': 'bfloat16', 'activation_checkpoint': 'none',
            'lm_head_chunk_tokens': 8192,
        },
        'runtime': {
            'attention_backend': 'triton', 'attention_strict': True,
            'profile_resolved': 'gpu-fast',
            'effective_flags': {
                **{f: '0' for f in FLAGS.values()},
                'MINI_LLM_FUSED_EXPERT_GEMM': '0',
                FLAGS[flag]: '1' if enabled else '0',
            },
        },
        'workload': {
            'batch_size': 3, 'microbatches_per_sample': 1,
            'requested_working_length': 4096,
            'total_supervised_rows_per_sample': 8000,
            'total_computed_rows_per_sample': 12288,
            'batches': [{'compact_working_length': 4096,
                         'computed_rows': 12288, 'supervised_rows': 8000,
                         'router_gumbel_seed': 42,
                         'fixture_hashes': {'inputs': 'hash-x', 'targets': 'hash-y',
                                            'loss_mask': 'hash-mask',
                                            'history_valid_starts': 'hash-h'}}],
        },
        'timing': {'gpu_event': {'median_ms': 850.0},
                   'profiler_perturbed': False},
        'last_loss_device_resident_until_boundary': 2.5,
    }


def test_accepts_single_isolated_optimization():
    a = _report('history', False)
    b = _report('history', True)
    b['timing']['gpu_event']['median_ms'] = 650.0
    b['last_loss_device_resident_until_boundary'] += 1e-4
    result = compare(a, b, 'history')
    assert result['status'] == 'valid'
    assert result['timing']['speedup'] > 1.3


def test_rejects_confounded_flags_and_changed_samples():
    a, b = _report('ffn-dx', False), _report('ffn-dx', True)
    b['runtime']['effective_flags']['MINI_LLM_FUSED_EXPERT_GEMM'] = '1'
    assert compare(a, b, 'ffn-dx')['status'] == 'invalid'
    b = _report('ffn-dx', True)
    b['workload']['batches'][0]['fixture_hashes']['inputs'] = 'wrong'
    assert compare(a, b, 'ffn-dx')['status'] == 'invalid'


def test_rejects_wrong_toggle_or_large_loss_difference():
    a, b = _report('ffn-wgrad', False), _report('ffn-wgrad', False)
    assert compare(a, b, 'ffn-wgrad')['status'] == 'invalid'
    b = _report('ffn-wgrad', True)
    b['last_loss_device_resident_until_boundary'] = 2.6
    assert compare(a, b, 'ffn-wgrad')['status'] == 'invalid'


def test_rejects_gpu_changes_or_non_timing_capture():
    a, b = _report('cross-block', False), _report('cross-block', True)
    b['system']['device'] = 'RTX 4090'
    assert compare(a, b, 'cross-block')['status'] == 'invalid'
    b = _report('cross-block', True)
    b['status'] = 'captured'
    assert compare(a, b, 'cross-block')['status'] == 'invalid'
