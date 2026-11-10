"""CPU-only decoder benchmark subprocess/trace contracts; no CUDA initialization."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'benchmarks/benchmark_0106o_decoder_layer.py'


def test_import_and_help_do_not_initialize_backend_or_cuda():
    env = dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1')
    script = ('import runpy,sys; runpy.run_path(sys.argv[1]); '
              'assert "mini_llm.backend" not in sys.modules; '
              'assert "cupy" not in sys.modules; assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', script, str(BENCHMARK)],
                            env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(BENCHMARK), '--help'],
                            env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and '--memory-budget-mib' in result.stdout


@pytest.mark.parametrize('profile,workers', [('summary', 0), ('detailed', 2)])
def test_synthetic_full_decoder_smoke_and_routing_report(tmp_path, profile, workers):
    path = tmp_path / 'decoder.json'
    result = subprocess.run([sys.executable, str(BENCHMARK), '--tokens', '4',
        '--profile', profile, '--cpu-workers', str(workers), '--output', str(path)],
        env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1'),
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report['passed'] and report['independent_fp64_reference']
    assert report['layer_ids'] == [0, 1, 2, 3]
    assert report['full_model_decode_available'] is False
    assert report['executor']['cpu_workers'] == workers
    assert report['memory_preflight']['estimate_bytes'] < report['memory_preflight']['limit_bytes']
    for run in report['runs'].values():
        assert run['passed'] and run['lengths'] == [6] * 4
        assert all(metric['passed'] for metric in run['comparisons'].values())
        trace = run['routing_trace']
        assert trace['tokens_by_layer'] == {str(i): 8 for i in range(4)}
        assert trace['counters']['routed_assignments'] == 64
        assert bool(trace['routes']) == (profile == 'detailed')
        assert run['observed_cpu_wall_ms'] >= 0
        if profile == 'detailed':
            assert all(record['path'] == 'cpu' and record['transferred_bytes'] == 0
                       for record in trace['executions'])


@pytest.mark.parametrize('arguments', [('--tokens', '0'), ('--cpu-workers', '5'),
                                        ('--memory-budget-mib', '0')])
def test_invalid_smoke_limits_rejected(tmp_path, arguments):
    result = subprocess.run([sys.executable, str(BENCHMARK), *arguments,
        '--output', str(tmp_path / 'invalid.json')],
        env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1'),
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'Require 1..32 tokens' in result.stderr
    assert not (tmp_path / 'invalid.json').exists()
