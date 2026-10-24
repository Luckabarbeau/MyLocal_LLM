"""CPU contracts and optional isolated-Torch smoke test for PATCH 0089B."""

from __future__ import annotations

import importlib.util
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
    "reference_0089b_torch_block",
    "benchmarks/reference_0089b_torch_block.py",
)
exporter = _load(
    "export_0089b_block_fixture",
    "benchmarks/export_0089b_block_fixture.py",
)
runner = _load(
    "benchmark_0089b_block",
    "benchmarks/benchmark_0089b_block.py",
)


def _manifest(batch=1, seq=17):
    return {
        "schema_version": "0089B-block-fixture-v1",
        "batch": batch,
        "seq": seq,
        "parameter_names": list(reference.PARAMETER_NAMES),
        "config": {
            "d_model": 1024,
            "d_ff": 4096,
            "q_heads": 16,
            "kv_heads": 4,
            "d_head": 64,
            "local_window": 512,
            "rope_base": 1_000_000.0,
            "rms_eps": 1e-6,
        },
    }


def test_parameter_layout_is_the_exact_dense_block_contract():
    assert len(reference.PARAMETER_NAMES) == 9
    assert tuple(reference.PARAMETER_SHAPES.values()) == (
        (1024,),
        (1024, 1024),
        (1024, 256),
        (1024, 256),
        (1024, 1024),
        (1024,),
        (1024, 4096),
        (1024, 4096),
        (4096, 1024),
    )
    assert len(reference.RESULT_TENSOR_NAMES) == 11
    assert reference.RESULT_TENSOR_NAMES[:2] == ("output", "d_input")
    assert set(reference.PARAMETER_KEYS) == set(reference.PARAMETER_NAMES)


def test_manifest_and_expected_fixture_shapes_are_fail_closed():
    manifest = _manifest(batch=3, seq=2304)
    reference.validate_manifest(manifest)
    shapes = reference.expected_array_shapes(3, 2304)
    assert shapes["x"] == (3, 2304, 1024)
    assert shapes["position_ids"] == (2304,)
    assert shapes["w_down_bits"] == (4096, 1024)
    broken = _manifest()
    broken["config"]["local_window"] = 511
    with pytest.raises(ValueError, match="local_window"):
        reference.validate_manifest(broken)


def test_fixture_hash_covers_dtype_shape_and_content():
    value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert reference.array_sha256(value) == reference.array_sha256(value.copy())
    assert reference.array_sha256(value) != reference.array_sha256(value.reshape(2, 6))
    assert reference.array_sha256(value) != reference.array_sha256(value.view(np.int16))
    changed = value.copy()
    changed[0, 0] = 99
    assert reference.array_sha256(value) != reference.array_sha256(changed)


def test_portable_bf16_encoding_is_ties_to_even():
    values = np.asarray([0.0, 1.0, -2.0, 1.00390625, 1.01171875], dtype=np.float32)
    bits = reference.bf16_bits_from_float32(values)
    restored = reference.bf16_bits_to_float32(bits)
    assert bits.dtype == np.uint16
    assert np.array_equal(restored[:3], values[:3])
    assert restored[3] == np.float32(1.0)
    assert restored[4] == np.float32(1.015625)


def test_full_and_local_attention_boundary_is_preserved():
    boundary = reference.boundary_contract(sequence_length=520)
    assert boundary == {
        "query": 519,
        "local_key_7_allowed": False,
        "local_key_8_allowed": True,
        "local_future_key_520_allowed": False,
        "full_key_7_allowed": True,
        "full_future_key_520_allowed": False,
        "sequence_length": 520,
    }


def test_checkpoint_names_are_layer_specific_but_fixture_names_are_stable():
    assert exporter._source_name("block.norm1.gamma", 0) == "blocks.0.norm1.gamma"
    assert (
        exporter._source_name("block.dense_ffn.expert.W_down", 7)
        == "blocks.7.dense_ffn.expert.W_down"
    )


def test_timing_summary_retains_raw_samples_and_mad():
    summary = runner._summary([1.0, 2.0, 100.0])
    assert summary["samples_ms"] == [1.0, 2.0, 100.0]
    assert summary["median_ms"] == 2.0
    assert summary["mad_ms"] == 1.0
    assert summary["count"] == 3


def test_parity_is_fail_closed_across_all_eleven_tensors(tmp_path):
    names = reference.RESULT_TENSOR_NAMES
    expected = {name: np.ones((2,), dtype=np.float32) for name in names}
    reference_path = tmp_path / "reference.npz"
    runner._save_tensors(reference_path, names, expected, {"backend": "cupy"})
    actual = {name: value.copy() for name, value in expected.items()}
    accepted = runner._compare(actual, reference_path, names, reference.error_metrics)
    assert accepted is not None and accepted["accepted"] is True
    actual[names[-1]][0] = -1.0
    rejected = runner._compare(actual, reference_path, names, reference.error_metrics)
    assert rejected is not None and rejected["accepted"] is False
    assert rejected["metrics"][names[-1]]["worst_pair"]["index"] == [0]


def test_optional_isolated_torch_full_block_smoke():
    python = ROOT / ".venv-0089b-torch" / "bin" / "python"
    if not python.is_file():
        pytest.skip("isolated 0089B Torch environment is not installed")
    code = """
import torch
from benchmarks.reference_0089b_torch_block import build_block_mask, build_torch_forward
T=17
mask=build_block_mask(torch,T)
pos=torch.arange(61440,61440+T,device='cuda')
fn=build_torch_forward(torch,mask,pos)
x=torch.randn(1,T,1024,device='cuda',dtype=torch.float32,requires_grad=True)
shapes=((1024,),(1024,1536),(1024,1024),(1024,),(1024,4096),(1024,4096),(4096,1024))
weights=[torch.randn(shape,device='cuda',dtype=torch.bfloat16,requires_grad=True)*0.01 for shape in shapes]
weights=[value.detach().requires_grad_(True) for value in weights]
y=fn(x,weights[0],weights[1],weights[2],weights[3],weights[4],weights[5],weights[6])
grads=torch.autograd.grad(y,(x,*weights),torch.randn_like(y))
torch.cuda.synchronize()
assert y.shape==(1,T,1024)
assert len(grads)==8
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
