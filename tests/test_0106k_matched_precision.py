"""Matched-storage FP64 oracle; no dependence on GPU reduction ordering."""

import numpy as np
import pytest

from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.ops.quantized_weight import PackedGGUFWeight


def q8_matrix(seed, rows=32, cols=32):
    values = np.random.default_rng(seed).integers(-8, 9, (rows, cols), dtype=np.int8)
    raw = np.empty((rows * cols // 32, 34), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(.0625).tobytes(), dtype=np.uint8)
    raw[:, 2:] = values.reshape(-1, 32).view(np.uint8)
    return PackedGGUFWeight(raw.ravel(), (rows, cols), "Q8_0"), values.astype(np.float64) * .0625


@pytest.mark.parametrize("shape", [(6, 32), (2, 3, 32)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_oracle_matches_independent_dense_graph_and_storage_boundaries(shape, dtype):
    gate, gd = q8_matrix(1)
    up, ud = q8_matrix(2)
    down, dd = q8_matrix(3)
    x = np.random.default_rng(4).normal(0, .3, shape).astype(np.float32)
    stages = packed_swiglu_reference(x, gate, up, down, storage_dtype=dtype)
    effective = x.astype(dtype).astype(np.float64)
    g, u = effective @ gd.T, effective @ ud.T
    h64 = (g / (1 + np.exp(-g))) * u
    h = h64.astype(dtype)
    y64 = h.astype(np.float64) @ dd.T
    np.testing.assert_array_equal(stages["input"], effective)
    np.testing.assert_array_equal(stages["gate"], g)
    np.testing.assert_array_equal(stages["up"], u)
    np.testing.assert_allclose(stages["hidden_unrounded"], h64, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(stages["output_unrounded"], y64, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(stages["output"], y64.astype(dtype), rtol=1e-14, atol=1e-14)
    assert stages["hidden"].dtype == stages["output"].dtype == dtype
    assert stages["gate"].dtype == stages["up"].dtype == np.float64
    assert stages["output"].shape == shape
    if dtype == np.float16:
        assert np.any(g != g.astype(np.float16).astype(np.float64))
        # Down must consume the stored hidden, not an unrounded FP64 shortcut.
        assert np.any(y64 != h64 @ dd.T)


def test_dtype_defaults_to_actual_input_and_fused_output_is_stored_hidden():
    gate, _ = q8_matrix(1)
    up, _ = q8_matrix(2)
    x = np.full((2, 32), 1.0001, dtype=np.float32)
    matched = packed_swiglu_reference(x, gate, up, storage_dtype="float16")
    actual_input = packed_swiglu_reference(x.astype(np.float16), gate, up)
    for name in matched:
        np.testing.assert_array_equal(matched[name], actual_input[name])
    assert actual_input["output"] is actual_input["hidden"]
    assert np.any(matched["input"] != x.astype(np.float64))


def test_large_negative_gate_does_not_overflow_reference_sigmoid():
    raw = np.empty((32, 34), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
    raw[:, 2:] = np.int8(-127).view(np.uint8)
    gate = PackedGGUFWeight(raw.ravel(), (32, 32), "Q8_0")
    up, _ = q8_matrix(2)
    with np.errstate(over="raise", invalid="raise"):
        stages = packed_swiglu_reference(np.full((1, 32), 1000, dtype=np.float64), gate, up)
    np.testing.assert_array_equal(stages["output"], np.zeros((1, 32)))
    assert np.isfinite(stages["gate"]).all()


@pytest.mark.parametrize("dtype", ["int8", "uint8", "complex128"])
def test_oracle_rejects_nonfloating_storage(dtype):
    weight, _ = q8_matrix(1)
    with pytest.raises(TypeError, match="dtype"):
        packed_swiglu_reference(np.zeros((1, 32)), weight, weight, storage_dtype=dtype)


@pytest.mark.parametrize("x", [np.zeros(32), np.zeros((1, 32), dtype=np.int32),
                              np.full((1, 32), np.nan)])
def test_oracle_rejects_invalid_input(x):
    weight, _ = q8_matrix(1)
    with pytest.raises(ValueError, match="input"):
        packed_swiglu_reference(x, weight, weight)


def test_oracle_rejects_mismatched_projection_contracts():
    weight, _ = q8_matrix(1)
    other, _ = q8_matrix(2, rows=16)
    x = np.zeros((1, 32))
    with pytest.raises(TypeError, match="host-packed"):
        packed_swiglu_reference(x, np.eye(32), weight)
    with pytest.raises(ValueError, match="dimensions"):
        packed_swiglu_reference(x, weight, other)
    with pytest.raises(ValueError, match="down"):
        packed_swiglu_reference(x, weight, weight, other)
