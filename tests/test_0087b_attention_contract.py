import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reference = _load(
    "reference_0087b_torch_attention",
    "benchmarks/reference_0087b_torch_attention.py",
)


def test_original_gqa_mapping_and_patterns_preserve_head_numbers():
    assert [reference.original_q_to_kv(head) for head in range(16)] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
    ]
    assert reference.resolve_active_heads("15") == tuple(range(1, 16))
    assert reference.resolve_active_heads("13") == tuple(range(1, 14))
    assert reference.resolve_active_heads("9") == tuple(range(1, 10))
    assert reference.resolve_active_heads("scattered") == (1, 2, 4, 5, 7, 8, 10, 12, 15)
    with pytest.raises(ValueError):
        reference.resolve_active_heads("2,1")
    with pytest.raises(ValueError):
        reference.resolve_active_heads("0,1")


def test_mask_boundary_full_local_future_and_historical_substitution():
    active = reference.resolve_active_heads("13")
    boundary = reference.boundary_contract(active)
    assert boundary == {
        "query": 519,
        "local_key_7_allowed": False,
        "local_key_8_allowed": True,
        "local_future_key_520_allowed": False,
        "full_key_7_allowed": True,
        "full_future_key_520_allowed": False,
    }
    assert not reference.mask_allows(14, 519, 519, active)
    assert not reference.mask_allows(15, 519, 519, active)
    assert reference.mask_allows(13, 519, 519, active)


def test_portable_bf16_roundtrip_and_ties_to_even():
    values = np.asarray([0.0, 1.0, -2.0, 1.00390625, 1.01171875], dtype=np.float32)
    bits = reference.bf16_bits_from_float32(values)
    restored = reference.bf16_bits_to_float32(bits)
    assert bits.dtype == np.uint16
    assert restored.dtype == np.float32
    assert np.array_equal(restored[:3], values[:3])
    # Exactly halfway values round to an even low BF16 significand bit.
    assert restored[3] == np.float32(1.0)
    assert restored[4] == np.float32(1.015625)


def test_fixture_hash_and_shape_validation(tmp_path):
    arrays = {
        "q_bits": np.zeros((1, 17, 16, 64), dtype=np.uint16),
        "k_bits": np.zeros((1, 17, 4, 64), dtype=np.uint16),
        "v_bits": np.zeros((1, 17, 4, 64), dtype=np.uint16),
        "d_out": np.zeros((1, 17, 16, 64), dtype=np.float32),
    }
    manifest = {
        "batch": 1,
        "seq": 17,
        "sha256": {name: reference.array_sha256(array) for name, array in arrays.items()},
    }
    fixture = tmp_path / "fixture.npz"
    np.savez(fixture, manifest_json=np.asarray(json.dumps(manifest)), **arrays)
    loaded_manifest, loaded = reference.load_fixture(fixture)
    assert loaded_manifest == manifest
    assert all(np.array_equal(loaded[name], arrays[name]) for name in arrays)
    loaded["d_out"][0, 0, 0, 0] = 1.0
    broken = tmp_path / "broken.npz"
    np.savez(broken, manifest_json=np.asarray(json.dumps(manifest)), **loaded)
    with pytest.raises(ValueError, match="hash mismatch"):
        reference.load_fixture(broken)


def test_error_metrics_report_worst_index_and_bias():
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    actual = expected.copy()
    actual[1, 0] += 0.5
    metrics = reference.error_metrics(actual, expected)
    assert metrics["max_abs"] == pytest.approx(0.5)
    assert metrics["worst_index"] == [1, 0]
    assert metrics["bias"] == pytest.approx(0.125)
    assert metrics["cosine"] < 1.0


def test_optional_isolated_torch_flex_smoke():
    python = ROOT / ".venv-0087b-torch" / "bin" / "python"
    if not python.is_file():
        pytest.skip("isolated 0087B Torch environment is not installed")
    code = """
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
q=torch.randn(1,4,17,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(1,1,17,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(1,1,17,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
mask=create_block_mask(lambda b,h,qi,ki:(ki<=qi)&((h==0)|(ki>=qi-7)),B=None,H=4,Q_LEN=17,KV_LEN=17,device='cuda')
fn=torch.compile(flex_attention,fullgraph=True)
y=fn(q,k,v,block_mask=mask,enable_gqa=True)
y.float().sum().backward()
torch.cuda.synchronize()
assert y.shape==(1,4,17,64)
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
