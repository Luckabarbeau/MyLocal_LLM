"""Independent-reference protocol, identity, metrics and optional linked integration."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.benchmark_0106q_reference import compare_logits, read_reference

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'benchmarks/benchmark_0106q_reference.py'


def test_identity_strict_precision_and_characterization_are_distinct():
    reference = np.array([[1., 2., 4.], [-2., 0., 1.]], np.float32)
    ids = np.array([1, 2], np.int32)
    exact = compare_logits(reference, reference.copy(), ids, ids, dict(rtol=1e-4, atol=1e-5))
    assert exact['strict_parity_passed'] and exact['max_abs_error'] == 0
    assert exact['greedy_ids_equal'] and all(p['top5_overlap'] == 3 for p in exact['positions'])
    changed = reference + np.float32(.01)
    characterized = compare_logits(changed, reference, ids, ids)
    assert characterized['strict_parity_passed'] is None
    assert characterized['greedy_ids_equal']
    assert characterized['max_abs_error'] > .009
    assert characterized['centered_rms_error'] < 1e-7
    strict = compare_logits(changed, reference, ids, ids, dict(rtol=1e-4, atol=1e-5))
    assert strict['strict_parity_passed'] is False and strict['max_tolerance_ratio'] > 1


@pytest.mark.parametrize('mutation', ['ids', 'shape', 'dtype', 'nan', 'inf', 'float_ids'])
def test_mismatched_or_invalid_inputs_never_establish_parity(mutation):
    actual = np.array([[1., 2., 3.]], np.float32)
    reference, ids, other = actual.copy(), np.array([1]), np.array([1])
    if mutation == 'ids':
        other = np.array([2])
    elif mutation == 'shape':
        actual = actual[:, :2]
    elif mutation == 'dtype':
        actual = actual.astype(np.float64)
    elif mutation == 'nan':
        actual[0, 0] = np.nan
    elif mutation == 'inf':
        reference[0, 0] = np.inf
    else:
        ids = ids.astype(np.float32)
    with pytest.raises(ValueError):
        compare_logits(actual, reference, ids, other)


def test_greedy_difference_records_both_margins():
    metric = compare_logits(np.array([[3, 2, 1]], np.float32),
                            np.array([[2, 4, 1]], np.float32), np.array([1]), np.array([1]))
    assert metric['greedy_ids_equal'] is False
    row = metric['positions'][0]
    assert row['actual_top_ids'][0] == 0 and row['reference_top_ids'][0] == 1
    assert row['actual_top1_margin'] == 1 and row['reference_top1_margin'] == 2


def test_binary_protocol_roundtrip(tmp_path):
    path = tmp_path / 'reference.bin'
    ids = np.array([1, 2], '<i4')
    logits = np.arange(6, dtype='<f4').reshape(2, 3)
    path.write_bytes(b'MLLQREF1' + struct.pack('<II', 2, 3) + ids.tobytes() + logits.tobytes())
    actual_ids, actual = read_reference(path)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual, logits)


@pytest.mark.parametrize('bad', ['magic', 'truncated', 'oversized', 'zero', 'nan', 'ids'])
def test_binary_protocol_rejects_malformed_inputs(tmp_path, bad):
    path = tmp_path / 'bad.bin'
    header = b'MLLQREF1' + struct.pack('<II', 1, 3)
    payload = np.array([1], '<i4').tobytes() + np.array([1, 2, 3], '<f4').tobytes()
    if bad == 'magic':
        header = b'INVALID!' + header[8:]
    elif bad == 'truncated':
        payload = payload[:-1]
    elif bad == 'oversized':
        header = b'MLLQREF1' + struct.pack('<II', 65, 3)
    elif bad == 'zero':
        header = b'MLLQREF1' + struct.pack('<II', 0, 3)
    elif bad == 'nan':
        payload = np.array([1], '<i4').tobytes() + np.array([1, np.nan, 3], '<f4').tobytes()
    else:
        payload = np.array([3], '<i4').tobytes() + np.array([1, 2, 3], '<f4').tobytes()
    path.write_bytes(header + payload)
    with pytest.raises(ValueError):
        read_reference(path)


def test_import_help_are_cpu_safe():
    env = dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1')
    script = ('import runpy,sys; runpy.run_path(sys.argv[1]); '
              'assert "mini_llm.backend" not in sys.modules; '
              'assert "cupy" not in sys.modules; assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', script, str(BENCHMARK)], env=env,
                            capture_output=True, text=True, cwd=ROOT, timeout=20)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(BENCHMARK), '--help'], env=env,
                            capture_output=True, text=True, cwd=ROOT, timeout=20)
    assert result.returncode == 0 and '--expected-revision' in result.stdout


@pytest.mark.skipif(not os.environ.get('MYLOCAL_LLAMA_REFERENCE_ROOT'),
                    reason='Optional source-built independent llama.cpp integration')
def test_independent_source_built_fixture_integration(tmp_path):
    path = tmp_path / 'strict.json'
    result = subprocess.run([sys.executable, str(BENCHMARK), '--llama-root',
        os.environ['MYLOCAL_LLAMA_REFERENCE_ROOT'], '--output', str(path)],
        env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1'),
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report['status'] == 'strict_fixture_verified'
    assert report['strict_reference_parity_passed'] and report['greedy_agreement_all_comparisons']
    assert all(m['strict_parity_passed'] for m in report['comparisons'].values())
    assert report['reference_full']['metadata']['cpu_only_execution']
    assert report['reference_full']['metadata']['kv_dtype'] == 'f32'
    assert report['fixture']['tensor_count'] == 73
