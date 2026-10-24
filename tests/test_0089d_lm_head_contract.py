"""CPU-fast contracts and optional isolated-Torch smoke for PATCH 0089D."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reference = _load(
    "reference_0089d_torch_lm_head",
    "benchmarks/reference_0089d_torch_lm_head.py",
)
exporter = _load(
    "export_0089d_lm_head_fixture",
    "benchmarks/export_0089d_lm_head_fixture.py",
)
runner = _load(
    "benchmark_0089d_lm_head",
    "benchmarks/benchmark_0089d_lm_head.py",
)


def _manifest():
    return {
        "schema_version": "0089D-lm-head-fixture-v1",
        "batch": 3,
        "seq": 2304,
        "rows": 6912,
        "d_model": 1024,
        "vocab": 32768,
        "chunk_tokens": 8192,
        "target_slice": [0, 2304],
        "valid_lengths": [2304, 1792, 1024],
        "normalizer_count": 5120.0,
        "grad_scale": 1.0,
        "array_names": ["head_bits", "weight_bits", "targets", "loss_mask"],
        "graph": reference.graph_contract(),
    }


def test_exact_production_geometry_and_one_tile_contract():
    assert (
        reference.BATCH,
        reference.SEQ,
        reference.D_MODEL,
        reference.VOCAB,
    ) == (3, 2304, 1024, 32768)
    assert reference.ROWS == 6912 < reference.CHUNK_TOKENS == 8192
    assert reference.NORMALIZER_COUNT == 5120.0
    contract = reference.graph_contract()
    assert contract["effective_tiles"] == [[0, 6912]]
    assert contract["forward_retains_logits"] is False
    assert contract["backward_recomputes_logits"] is True
    assert contract["probability_rounding"] == (
        "float32_to_bfloat16_before_target_subtraction"
    )
    assert contract["dweight_tile_dtype"] == "bfloat16"
    assert contract["dweight_accumulator_dtype"] == "float32"
    assert contract["target_slice_copy"] is True
    assert contract["target_gradient_zero_scatter"] is True


def test_manifest_and_fixture_layout_are_fail_closed_without_allocating_fixture():
    manifest = _manifest()
    reference.validate_manifest(manifest)
    assert reference.expected_array_shapes() == {
        "head_bits": (3, 2304, 1024),
        "weight_bits": (1024, 32768),
        "targets": (3, 2304),
        "loss_mask": (3, 2304),
    }
    assert reference.expected_array_dtypes() == {
        "head_bits": np.dtype(np.uint16),
        "weight_bits": np.dtype(np.uint16),
        "targets": np.dtype(np.int32),
        "loss_mask": np.dtype(np.float32),
    }
    broken = _manifest()
    broken["chunk_tokens"] = 6912
    with pytest.raises(ValueError, match="chunk_tokens"):
        reference.validate_manifest(broken)
    broken = _manifest()
    broken["graph"] = dict(broken["graph"], backward_recomputes_logits=False)
    with pytest.raises(ValueError, match="graph contract"):
        reference.validate_manifest(broken)


def test_synthetic_mask_is_prefix_valid_and_uses_masked_normalizer():
    mask = exporter._synthetic_loss_mask()
    assert mask.shape == (3, 2304)
    assert mask.dtype == np.float32
    assert tuple(mask.sum(axis=1).astype(np.int64)) == reference.VALID_LENGTHS
    assert float(np.sum(mask, dtype=np.float64)) == reference.NORMALIZER_COUNT
    assert np.all(mask[1, :1792] == 1.0)
    assert np.all(mask[1, 1792:] == 0.0)


def test_fixture_hash_covers_dtype_shape_and_content():
    value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert reference.array_sha256(value) == reference.array_sha256(value.copy())
    assert reference.array_sha256(value) != reference.array_sha256(
        value.reshape(2, 6)
    )
    assert reference.array_sha256(value) != reference.array_sha256(
        value.view(np.int16)
    )
    changed = value.copy()
    changed[0, 0] = 99
    assert reference.array_sha256(value) != reference.array_sha256(changed)


def test_portable_bf16_encoding_is_ties_to_even():
    values = np.asarray(
        [0.0, 1.0, -2.0, 1.00390625, 1.01171875], dtype=np.float32
    )
    bits = reference.bf16_bits_from_float32(values)
    restored = reference.bf16_bits_to_float32(bits)
    assert bits.dtype == np.uint16
    assert np.array_equal(restored[:3], values[:3])
    assert restored[3] == np.float32(1.0)
    assert restored[4] == np.float32(1.015625)


def test_timing_summary_retains_raw_samples_and_mad():
    summary = runner._summary([1.0, 2.0, 100.0])
    assert summary["samples_ms"] == [1.0, 2.0, 100.0]
    assert summary["median_ms"] == 2.0
    assert summary["mad_ms"] == 1.0
    assert summary["count"] == 3


def test_parity_requires_loss_dx_and_dweight(tmp_path):
    names = reference.RESULT_TENSOR_NAMES
    expected = {
        "loss": np.asarray(2.0, dtype=np.float32),
        "dx": np.ones((2, 3), dtype=np.float32),
        "dweight": np.ones((3, 4), dtype=np.float32),
    }
    reference_path = tmp_path / "cupy_reference.npz"
    runner._save_tensors(reference_path, names, expected, {"backend": "cupy"})
    actual = {name: value.copy() for name, value in expected.items()}
    accepted = runner._compare(
        actual, reference_path, names, reference.error_metrics
    )
    assert accepted is not None and accepted["accepted"] is True

    actual["dweight"][0, 0] = -1.0
    rejected = runner._compare(
        actual, reference_path, names, reference.error_metrics
    )
    assert rejected is not None and rejected["accepted"] is False
    assert rejected["worst_tensor"] == "dweight"

    actual = {name: value.copy() for name, value in expected.items()}
    actual["loss"] = np.asarray(2.02, dtype=np.float32)
    rejected_loss = runner._compare(
        actual, reference_path, names, reference.error_metrics
    )
    assert rejected_loss is not None and rejected_loss["accepted"] is False
    assert rejected_loss["loss"]["accepted"] is False


def test_optional_isolated_torch_manual_compiled_smoke():
    if os.environ.get("RUN_0089D_TORCH_SMOKE") != "1":
        pytest.skip("set RUN_0089D_TORCH_SMOKE=1 for isolated CUDA/Torch smoke")
    python = ROOT / ".venv-0089b-torch" / "bin" / "python"
    if not python.is_file():
        pytest.skip("isolated Torch environment is not installed")
    code = """
import torch
from benchmarks.reference_0089d_torch_lm_head import build_torch_functions
assert torch.cuda.is_available()
fwd,bwd=build_torch_functions(torch,batch=1,seq=3,d_model=8,vocab=16,
    chunk_tokens=8,target_slice=(0,3),normalizer_count=2.0,grad_scale=1.0)
fwd=torch.compile(fwd,fullgraph=True,dynamic=False)
bwd=torch.compile(bwd,fullgraph=True,dynamic=False)
x=torch.randn(1,3,8,device='cuda',dtype=torch.bfloat16)
w=torch.randn(8,16,device='cuda',dtype=torch.bfloat16)
y=torch.tensor([[1,2,3]],device='cuda',dtype=torch.int64)
m=torch.tensor([[1.,1.,0.]],device='cuda')
dw=torch.zeros(8,16,device='cuda',dtype=torch.float32)
loss,selected=fwd(x,w,y,m)
dx,dw_out=bwd(selected,w,y,m,dw)
torch.cuda.synchronize()
assert loss.dtype==torch.float32 and loss.ndim==0
assert selected.dtype==torch.bfloat16 and selected.shape==(1,3,8)
assert dx.dtype==torch.bfloat16 and dx.shape==(1,3,8)
assert dw_out.dtype==torch.float32 and dw_out.shape==(8,16)
assert torch.count_nonzero(dx[:,2:])==0
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
