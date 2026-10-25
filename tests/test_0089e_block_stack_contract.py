"""CPU-fast contracts and optional GPU smoke for PATCH 0089E."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _load_runner():
    path = ROOT / "benchmarks" / "benchmark_0089e_block_stack.py"
    spec = importlib.util.spec_from_file_location("benchmark_0089e_block_stack", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cupy_benchmark_environment_covers_required_fairness_contract():
    required = {
        "MINI_LLM_BACKEND": "cupy",
        "MINI_LLM_LOCAL_DETAIL_PROFILE": "0",
        "MINI_LLM_MOE_DETAIL_PROFILE": "0",
        "MINI_LLM_OPTIMIZER_OFFLOAD": "full",
        "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB": "16",
        "MINI_LLM_FUSED_EXPERT_GEMM": "0",
        "MINI_LLM_AR_COMPACT_TARGET": "1",
        "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS": "1",
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM": "1",
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT": "0",
        "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD": "1",
        "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS": "1",
        "MINI_LLM_ROPE_PREPARED_POSITIONS": "1",
        "MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL": "5",
        "MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION": "1",
        "MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT": "1",
        "MINI_LLM_GPU_MEMORY_AUDIT": "0",
    }
    assert {
        name: runner.CUPY_BENCHMARK_ENV.get(name) for name in required
    } == required

    inactive_in_block_stack = {
        "MINI_LLM_OPTIMIZER_OFFLOAD",
        "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB",
        "MINI_LLM_AR_COMPACT_TARGET",
        "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS",
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM",
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT",
        "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD",
        "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS",
        "MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL",
        "MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION",
        "MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT",
    }
    assert inactive_in_block_stack <= set(runner.CUPY_BENCHMARK_ENV_APPLICABILITY)
    assert "prepared once outside timing" in runner.CUPY_BENCHMARK_ENV_APPLICABILITY[
        "MINI_LLM_ROPE_PREPARED_POSITIONS"
    ]


def test_valid_depths_are_fail_closed():
    assert runner.VALID_DEPTHS == (1, 2, 4, 8, 20)
    for depth in runner.VALID_DEPTHS:
        assert runner.validate_depth(depth) == depth
    for invalid in (0, 3, 16, 21, -1):
        with pytest.raises(ValueError, match="depth must be one of"):
            runner.validate_depth(invalid)


def test_depth_one_reuses_exact_0089b_tensor_contract():
    reference = __import__(
        "benchmarks.reference_0089b_torch_block",
        fromlist=["RESULT_TENSOR_NAMES"],
    )
    assert runner.result_tensor_names(1) == tuple(reference.RESULT_TENSOR_NAMES)
    assert len(runner.result_tensor_names(1)) == 11


def test_depth_two_tensor_contract_has_22_unique_tensors():
    names = runner.result_tensor_names(2)
    assert names[:4] == (
        "output",
        "d_input",
        "boundary_output",
        "boundary_grad",
    )
    assert len(names) == 22
    assert len(set(names)) == 22
    for layer_index in range(2):
        layer_names = [name for name in names if name.startswith(f"layer{layer_index}_")]
        assert len(layer_names) == 9
        assert f"layer{layer_index}_grad_norm1_gamma" in layer_names
        assert f"layer{layer_index}_grad_wq" in layer_names
        assert f"layer{layer_index}_grad_w_down" in layer_names
    assert runner.result_tensor_names(4) == ()
    assert runner.result_tensor_names(8) == ()
    assert runner.result_tensor_names(20) == ()


def test_leaf_and_logical_gradient_counts_scale_with_depth():
    for depth in runner.VALID_DEPTHS:
        assert runner.torch_leaf_count(depth) == 1 + 7 * depth
        assert runner.logical_gradient_count(depth) == 9 * depth


def test_timing_summary_retains_raw_samples_and_mad():
    summary = runner._summary([1.0, 2.0, 100.0])
    assert summary["samples_ms"] == [1.0, 2.0, 100.0]
    assert summary["median_ms"] == 2.0
    assert summary["mad_ms"] == 1.0
    assert summary["count"] == 3
    with pytest.raises(ValueError, match="at least one sample"):
        runner._summary([])


def test_parity_accepts_all_tensors_and_fails_closed():
    names = runner.result_tensor_names(2)
    expected = {
        name: np.asarray([1.0, 2.0, 3.0], dtype=np.float32) for name in names
    }
    actual = {name: value.copy() for name, value in expected.items()}
    accepted = runner._parity_report(actual, expected, names)
    assert accepted["accepted"] is True
    assert set(accepted["metrics"]) == set(names)

    actual[names[-1]][0] = -10.0
    rejected = runner._parity_report(actual, expected, names)
    assert rejected["accepted"] is False
    assert rejected["worst_pair"]["tensor"] == names[-1]

    missing = dict(actual)
    missing.pop(names[0])
    with pytest.raises(ValueError, match="contract mismatch"):
        runner._parity_report(missing, expected, names)

    extra = dict(expected)
    extra["unexpected"] = np.ones((3,), dtype=np.float32)
    with pytest.raises(ValueError, match="contract mismatch"):
        runner._parity_report(expected, extra, names)

    wrong_shape = {name: value.copy() for name, value in expected.items()}
    wrong_shape[names[0]] = np.ones((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        runner._parity_report(wrong_shape, expected, names)


def test_timing_only_io_contract_rejects_reference_and_full_exports(tmp_path):
    output = tmp_path / "tensors.npz"
    reference = tmp_path / "reference.npz"
    runner._validate_io_contract(1, None, output)
    runner._validate_io_contract(2, reference, output)
    runner._validate_io_contract(4, None, None)
    with pytest.raises(ValueError, match="require --output-tensors"):
        runner._validate_io_contract(2, None, None)
    with pytest.raises(ValueError, match="reference tensors are unsupported"):
        runner._validate_io_contract(8, reference, None)
    with pytest.raises(ValueError, match="full tensor export is unsupported"):
        runner._validate_io_contract(20, None, output)


def test_memory_gate_records_fraction_and_fails_closed():
    accepted = runner._memory_gate(
        total_bytes=1000,
        observed_bytes=899,
        limit_fraction=0.9,
        semantics="test",
    )
    rejected = runner._memory_gate(
        total_bytes=1000,
        observed_bytes=901,
        limit_fraction=0.9,
        semantics="test",
    )
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["observed_fraction"] == pytest.approx(0.901)


@pytest.mark.skipif(
    os.environ.get("RUN_0089E_GPU_SMOKE") != "1",
    reason="set RUN_0089E_GPU_SMOKE=1 and PATCH_0089E_GPU_FIXTURE to opt in",
)
def test_optional_depth_two_cupy_vs_compiled_torch_smoke(tmp_path):
    fixture_value = os.environ.get("PATCH_0089E_GPU_FIXTURE")
    if not fixture_value:
        pytest.skip("PATCH_0089E_GPU_FIXTURE is not set")
    fixture = Path(fixture_value)
    if not fixture.is_file():
        pytest.skip(f"fixture does not exist: {fixture}")
    torch_python = ROOT / ".venv-0089b-torch" / "bin" / "python"
    if not torch_python.is_file():
        pytest.skip("isolated 0089B Torch environment is not installed")

    benchmark = ROOT / "benchmarks" / "benchmark_0089e_block_stack.py"
    cupy_tensors = tmp_path / "cupy_depth2.npz"
    cupy_json = tmp_path / "cupy_depth2.json"
    torch_tensors = tmp_path / "torch_depth2.npz"
    torch_json = tmp_path / "torch_depth2.json"

    cupy = subprocess.run(
        [
            sys.executable,
            str(benchmark),
            "--backend",
            "cupy",
            "--fixture",
            str(fixture),
            "--depth",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output-tensors",
            str(cupy_tensors),
            "--output-json",
            str(cupy_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    assert cupy.returncode == 0, cupy.stderr

    compiled = subprocess.run(
        [
            str(torch_python),
            str(benchmark),
            "--backend",
            "torch-compile",
            "--fixture",
            str(fixture),
            "--depth",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--reference-tensors",
            str(cupy_tensors),
            "--output-tensors",
            str(torch_tensors),
            "--output-json",
            str(torch_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
