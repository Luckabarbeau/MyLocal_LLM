"""All-48 retained conditioned contract, separate from legacy strict parity.

These tests deliberately fail (not skip) when the requested evidence is absent.
No model rerun and no changed production tolerance or acceptance flag.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('logs/wp7_conditioned_all_layers_v2')
GPU = 'GPU-e961516f-d77d-813c-98e9-2f5403f71631'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_pair(got, want, row):
    assert got.shape == want.shape
    assert np.isfinite(got).all() and np.isfinite(want).all()
    if not got.size:
        assert row['empty'] and row['exact']
        return
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    if row.get('requires_exact'):
        np.testing.assert_array_equal(got, want)
    delta = got.astype(np.float64)-want.astype(np.float64)
    assert row['maxabs'] == float(np.max(np.abs(delta)))
    assert row['violations'] == 0
    assert row['rtol'] == row['atol'] == 2e-4


@pytest.mark.parametrize('index', range(48))
def test_all48_conditioned_evidence(index):
    folder = ROOT/f'layer{index:02d}'
    report = json.loads((folder/'report.json').read_text())
    assert report['layer'] == index and report['passed'] and not report['failures']
    assert report['policy'] == 'cpu_ggml_q8_gpu_reference_fp32'
    assert report['gpu_experts'] == []
    admission = report['preflight']
    assert admission['gpu_mask'] == GPU and admission['admitted']
    assert admission['free_bytes'] >= admission['target_bytes'] + admission['reserve_bytes']
    assert sum(report['memory'].values()) <= 1024**3
    assert digest(folder/'evidence.npz') == report['evidence_sha256']
    recurrent = index % 4 != 3
    assert report['recurrent'] == recurrent
    with np.load(folder/'evidence.npz', allow_pickle=False) as evidence:
        for lineage in ('native', 'python'):
            for t in (0, 1):
                prefix = f'{lineage}_{t}_'
                required = ['attention_hidden', 'attention_gains', 'attention_write',
                            'ffn_hidden', 'ffn_gains', 'ffn_write', 'mixer_output',
                            'moe_ids', 'moe_weights', 'moe_logits', 'moe_probabilities',
                            'moe_cpu_rows', 'moe_terms', 'moe_routed', 'moe_shared', 'moe_output',
                            'native_ordered_combine', 'python_ordered_combine',
                            'native_weight_once', 'python_weight_once']
                required += ['mixer_'+s for s in (('history', 'matrix') if recurrent else
                             ('raw', 'keys', 'values', 'pooled', 'ids'))]
                required += [f'cpu_row{i}' for i in range(10)]
                if index == 1:
                    required += ['ple_streams', 'ple_history']
                assert all(prefix+name in report['checks'] for name in required)
                for name in ('quantize_row_q8_K', 'quantize_row_q8_0'):
                    assert evidence[prefix+name].dtype == np.uint8
                with np.load(Path('logs/wp7_divergence/actual_full')/('token1' if t else '')/f'{lineage}.npz') as capture:
                    np.testing.assert_array_equal(evidence[prefix+'moe_input_native'], capture[f'{index}_hc_ffn_read'])
                assert evidence[prefix+'moe_cpu_rows_native'].shape == (10, 2560)
                assert evidence[prefix+'moe_ids_native'].shape == (10,)
        for key, row in report['checks'].items():
            if 'captured_' not in key:
                assert_pair(evidence[key+'_native'], evidence[key+'_python'], row)
        for t, cross in enumerate(report['cross_lineage']):
            for name in ('quantize_row_q8_K', 'quantize_row_q8_0'):
                n, p = (evidence[f'{lineage}_{t}_'+name] for lineage in ('native', 'python'))
                assert cross['quantized_input_bytes_changed'][name] == np.count_nonzero(n != p)
            n, p = (evidence[f'{lineage}_{t}_moe_ids_native'] for lineage in ('native', 'python'))
            assert cross['ids_changed'] == (not np.array_equal(n, p))


def test_batches_and_original_capture_identity():
    historical = json.loads(Path('docs/native_engine/WP7_FULL_MODEL_DIAGNOSTICS.json').read_text())
    for start in range(0, 48, 8):
        batch = json.loads((ROOT/f'batch{start:02d}-{start+8:02d}.json').read_text())
        assert batch['passed'] and batch['layers'] == list(range(start, start+8))
        for path, expected in batch['provenance']['artifacts'].items():
            assert digest(Path(path)) == expected == historical['artifact_sha256'][path]
        for path, expected in batch['provenance']['sources'].items():
            assert digest(Path(path)) == expected
        for path, expected in batch['provenance']['libraries'].items():
            assert digest(Path(path)) == expected
    assert historical['strict_whole_model_acceptance_passed'] is False
    for name in ('token0', 'token1'):
        assert historical[name]['strict_trajectory_gate_passed'] is False
        assert historical[name]['short_token_distribution_diagnostics']['accepted_quality_gate'] is False


def test_head_identical_collapsed_input():
    report = json.loads((ROOT/'head/report.json').read_text())
    assert report['passed']
    assert sum(report['memory'].values()) <= 1024**3
    assert report['preflight']['gpu_mask'] == GPU and report['preflight']['admitted']
    assert digest(ROOT/'head/evidence.npz') == report['evidence_sha256']
    with np.load(ROOT/'head/evidence.npz', allow_pickle=False) as evidence:
        assert len(report['checks']) == 8
        for key, row in report['checks'].items():
            assert_pair(evidence[key+'_native'], evidence[key+'_python'], row)
            if key.endswith('vocab'):
                assert evidence[key+'_native'].shape == (248320,)
        for lineage in ('native', 'python'):
            for t in (0, 1):
                assert evidence[f'{lineage}_{t}_collapsed_identical_input'].shape == (2560,)


def test_published_summary_is_complete_and_scoped():
    summary = json.loads(Path('docs/native_engine/WP7_CONDITIONED_ALL_LAYERS.json').read_text())
    assert summary['layers_executed'] == list(range(48))
    assert summary['conditioned_all_layers_passed']
    assert summary['same_input_cpu_rows_checked'] == 1920
    assert summary['groups']['hc_read']['comparisons'] == 384
    assert summary['captures_reconstructed']['exact_native'] == 192
    assert not summary['legacy_strict']['strict_whole_model_acceptance_passed']
    assert not summary['parent_status_modified']
    for name, expected in summary['provenance']['new_source_sha256'].items():
        assert digest(Path(name)) == expected
    for name, expected in summary['provenance']['layer_reports'].items():
        assert digest(Path(name)) == expected


def test_conditioned_contract_rejects_discrete_and_continuous_defects():
    with pytest.raises(AssertionError):
        assert_pair(np.array([1.], np.float32), np.array([1.01], np.float32),
                    dict(maxabs=.01, violations=1, rtol=2e-4, atol=2e-4))
    # An allclose-compatible difference is still invalid for exact expert rows.
    with pytest.raises(AssertionError):
        assert_pair(np.array([1.], np.float32), np.array([1.000001], np.float32),
                    dict(requires_exact=True))
