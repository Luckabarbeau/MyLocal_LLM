"""0106N CuPy validation only; existing CPU-only fixture guards stay untouched.

CPU: MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q tests/test_0106n_recurrent_gpu.py
GPU: MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -x -q tests/test_0106n_recurrent_gpu.py

Each GPU case exports CPU references first, then uses a private <256 MiB pool.
No actual checkpoint is needed; the benchmark CLI optionally tests layer 0.
"""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/benchmark_0106n_recurrent_gpu.py"
_spec = importlib.util.spec_from_file_location("recurrent_gpu_validation_0106n", BENCHMARK)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def test_cpu_safe_syntax_import_and_cli_help():
    for path in (Path(__file__), BENCHMARK):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # A fresh interpreter proves importing the helper does not initialize CUDA,
    # import torch, or import the backend before the CPU subprocess is selected.
    script = ("import runpy, sys; runpy.run_path(sys.argv[1]); "
              "assert 'cupy' not in sys.modules; "
              "assert 'torch' not in sys.modules; "
              "assert 'mini_llm.backend' not in sys.modules")
    env = dict(os.environ, MINI_LLM_BACKEND="numpy")
    completed = subprocess.run([sys.executable, "-c", script, str(BENCHMARK)],
                               env=env, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    help_result = subprocess.run([sys.executable, str(BENCHMARK), "--help"],
                                env=env, capture_output=True, text=True, timeout=20)
    assert help_result.returncode == 0, help_result.stderr
    assert "--model" in help_result.stdout and "--output" in help_result.stdout


def test_cpu_fixture_npz_manifest_without_pickle(tmp_path):
    manifest, arrays = validation.create_cpu_fixture(tmp_path / "dense.npz", "mixer-dense")
    assert manifest["kind"] == "mixer" and manifest["dtype"] == "float32"
    assert manifest["spec"]["n_key_heads"] != manifest["spec"]["n_value_heads"]
    assert manifest["spec"]["key_dim"] != manifest["spec"]["value_dim"]
    assert arrays["ref_output"].dtype == np.float64
    assert arrays["cpu_output"].dtype == np.float32
    assert np.any(arrays["initial_history"]) and np.any(arrays["initial_memory"])
    assert all(value.dtype.kind != "O" for value in arrays.values())
    metric = validation.numerical_metrics(arrays["cpu_output"], arrays["ref_output"],
                                         rtol=3e-4, atol=5e-6, path="cpu/fp64/output")
    assert metric["passed"], metric


GPU_CASES = (
    [(case, dtype) for case in ("delta-default", "delta-map", "conv")
     for dtype in ("float32", "float64")]
    + [(case, dtype) for case in ("norm-silu", "norm-sigmoid")
       for dtype in ("float16", "float32", "float64")]
    + [("mixer-dense", "float32"), ("mixer-packed", "float32")]
)


@pytest.mark.skipif(os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "cupy",
                    reason="Real GPU validation requires MINI_LLM_BACKEND=cupy; CPU import is safe")
@pytest.mark.parametrize("case,dtype", GPU_CASES, ids=[f"{case}-{dtype}" for case, dtype in GPU_CASES])
def test_recurrent_real_cupy_current_stream_cpu_and_fp64_parity(tmp_path, case, dtype):
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if devices == 0:
        pytest.skip("No CUDA device")
    # Import the backend only after selecting real CuPy. No monkeypatching or
    # changing process backend is permitted: CPU exports are separate processes.
    from mini_llm.backend import BACKEND_NAME, xp
    assert BACKEND_NAME == "cupy" and xp is cp
    manifest, arrays = validation.create_cpu_fixture(tmp_path / "fixture.npz", case, dtype=dtype)
    report_path = tmp_path / "validation.json"
    print(f"0106N diagnostics: {report_path}")
    result = validation.validate_gpu(manifest, arrays, report_path, non_default=True)
    assert result["passed"] and result["non_default_stream"]
    assert result["pool_limit_bytes"] == validation.BUDGET
    assert result["pool_peak_reserved_bytes"] < validation.BUDGET
    assert result["timings"]["full"]["cuda_event_ms"] >= 0
    assert all(value["passed"] for value in result["comparisons"].values())
