"""0106N synthetic attention GPU validation; no actual checkpoint execution.

MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -x -q tests/test_0106n_attention_gpu.py
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -x -q tests/test_0106n_attention_gpu.py

Fixtures/references are exported in isolated CPU subprocesses before GPU use.
An unselected CuPy backend skips GPU cases without importing CuPy or torch.
"""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/benchmark_0106n_attention_gpu.py"
_spec = importlib.util.spec_from_file_location("attention_gpu_validation_0106n", BENCHMARK)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def test_cpu_safe_syntax_import_help_and_required_output():
    for path in (Path(__file__), BENCHMARK):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env = dict(os.environ, MINI_LLM_BACKEND="numpy")
    script = ("import runpy, sys; runpy.run_path(sys.argv[1]); "
              "assert 'cupy' not in sys.modules; assert 'torch' not in sys.modules; "
              "assert 'mini_llm.backend' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script, str(BENCHMARK)],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(BENCHMARK), "--help"],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout and "--output" in result.stdout
    result = subprocess.run([sys.executable, str(BENCHMARK)],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and "--output is required" in result.stderr


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "q8"])
@pytest.mark.parametrize("positions", ["implicit", "prepared"])
def test_cpu_fixture_npz_format_oracle_and_unread_expert_payloads(tmp_path, packed, positions):
    manifest, arrays = validation.create_cpu_fixture(tmp_path / "fixture.npz",
                                                    packed=packed, positions=positions)
    spec = manifest["spec"]
    assert spec["rotary_dim"] < spec["key_dim"]
    assert spec["d_model"] != spec["n_q_heads"] * spec["key_dim"]
    assert spec["n_q_heads"] > spec["n_kv_heads"]
    assert manifest["dtype"] == "float32" and manifest["capacity"] == 8
    assert manifest["prefix_tokens"] == 2 and manifest["cpu_packed_backend"] == "reference"
    assert manifest["budget"]["conservative_bytes"] < validation.BUDGET
    assert arrays["hidden"].dtype == np.float32
    assert arrays["ref_output"].dtype == np.float64
    assert arrays["unused_payload"].shape == (0,) and arrays["unused_payload"].dtype == np.uint8
    assert all(value.dtype.kind != "O" for value in arrays.values())
    requested = {header["name"] for header in manifest["budget"]["headers"]}
    assert set(manifest["payload_reads"]) == requested
    for tensor in manifest["index"]["tensors"]:
        payload = arrays[tensor["payload"]]
        if tensor["name"] in requested:
            assert payload.size > 0
            if packed and not tensor["name"].endswith("_norm.weight"):
                assert tensor["kind"] == "Q8_0" and payload.dtype == np.uint8
        else:
            assert tensor["payload"] == "unused_payload" and payload.size == 0
    if positions != "implicit":
        assert arrays["positions"].shape == (2, 4)
        assert not np.array_equal(arrays["positions"][0], arrays["positions"][1])
        assert np.any(np.diff(arrays["positions"], axis=1) < 0)
    metrics = {}
    for prefix, length in (("prefix_", 2), ("", 6), ("reset_", 4)):
        metrics[prefix + "output"] = validation.numerical_metrics(
            arrays["cpu_" + prefix + "output"], arrays["ref_" + prefix + "output"],
            rtol=validation.OUTPUT_TOL[0], atol=validation.OUTPUT_TOL[1], path="cpu/" + prefix + "output")
        for cache in ("k_cache", "v_cache"):
            metrics[prefix + cache] = validation.numerical_metrics(
                arrays["cpu_" + prefix + cache][:, :, :length], arrays["ref_" + prefix + cache],
                rtol=validation.KV_TOL[0], atol=validation.KV_TOL[1], path="cpu/" + prefix + cache)
    report_path = tmp_path / "cpu-validation.json"
    report_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert all(metric["passed"] for metric in metrics.values()), f"{report_path}: {metrics}"


GPU_CASES = [(packed, positions) for packed in (False, True)
             for positions in ("implicit", "explicit", "prepared")]


@pytest.mark.skipif(os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "cupy",
                    reason="Real GPU validation requires MINI_LLM_BACKEND=cupy")
@pytest.mark.parametrize("packed,positions", GPU_CASES,
                         ids=[f"{'q8' if packed else 'dense'}-{positions}" for packed, positions in GPU_CASES])
def test_attention_cupy_nondefault_stream_cpu_fp64_and_state_contracts(tmp_path, packed, positions):
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if not devices:
        pytest.skip("No CUDA device")
    from mini_llm.backend import BACKEND_NAME, xp
    assert BACKEND_NAME == "cupy" and xp is cp
    manifest, arrays = validation.create_cpu_fixture(tmp_path / "fixture.npz",
                                                    packed=packed, positions=positions)
    report_path = tmp_path / "validation.json"
    print(f"0106N attention diagnostics: {report_path}")
    report = validation.validate_gpu(manifest, arrays, report_path)
    assert report["passed"] and report["non_default_stream"]
    assert report["pool_limit_bytes"] == validation.BUDGET
    assert report["pool_peak_reserved_bytes"] < validation.BUDGET
    assert all(metric["passed"] for metric in report["comparisons"].values())
    assert set(report["timings"]) == {"prefix_cold", "full", "chunks", "steps"}
    assert all(timing["cuda_event_ms"] >= 0 and timing["wall_ms"] >= 0
               for timing in report["timings"].values())
