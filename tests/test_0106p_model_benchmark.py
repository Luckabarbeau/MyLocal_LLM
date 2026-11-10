"""Full-model benchmark CLI, isolated CPU oracle and bounded real CUDA smoke."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'benchmarks/benchmark_0106p_qwen_model.py'


def test_benchmark_import_and_help_do_not_initialize_cuda():
    script = ('import runpy,sys; runpy.run_path(sys.argv[1]); '
              'assert "mini_llm.backend" not in sys.modules; '
              'assert "cupy" not in sys.modules; assert "torch" not in sys.modules')
    env = dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', script, str(BENCHMARK)], env=env,
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(BENCHMARK), '--help'], env=env,
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and '--device-budget-mib' in result.stdout


def test_synthetic_cpu_cli_uses_existing_session_and_fp64_reference(tmp_path):
    path = tmp_path / 'model.json'
    result = subprocess.run([sys.executable, str(BENCHMARK), '--output', str(path)],
        env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1'),
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report['passed'] and report['synthetic']
    assert report['generated_token_ids'] == [48, 28]
    assert all(metric['passed'] for metric in report['comparisons'].values())
    assert 'independent_fp64' in report['comparisons']


@pytest.mark.skipif(os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'cupy',
                    reason='Requires real CuPy backend and free GPU')
def test_synthetic_cuda_matches_isolated_cpu_and_fp64(tmp_path):
    path = tmp_path / 'cuda.json'
    result = subprocess.run([sys.executable, str(BENCHMARK), '--output', str(path)],
        env=dict(os.environ, MINI_LLM_BACKEND='cupy', OPENBLAS_NUM_THREADS='1'),
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report['passed'] and report['cpu_reference']['passed']
    assert report['non_default_stream']
    assert report['matched_input_greedy_ids'] == report['cpu_reference']['generated_token_ids']
    assert 'full_independent_fp64' in report['comparisons']
    assert report['pool_reserved_bytes'] < report['pool_limit_bytes']


@pytest.mark.skipif(os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'cupy',
                    reason='Requires real CuPy backend and free GPU')
def test_cuda_budget_rejects_before_model_execution(tmp_path):
    path = tmp_path / 'budget.json'
    result = subprocess.run([sys.executable, str(BENCHMARK), '--device-budget-mib', '1',
        '--output', str(path)], env=dict(os.environ, MINI_LLM_BACKEND='cupy', OPENBLAS_NUM_THREADS='1'),
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    report = json.loads(path.read_text())
    assert report['passed'] is False and report['comparisons'] == {}
    assert report['error']['type'] == 'MemoryError'
    assert 'explicit device budget' in report['error']['message']
