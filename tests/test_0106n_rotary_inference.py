"""CPU scalar oracles and contracts for generic inference-only partial RoPE."""

from dataclasses import replace
import math

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import rope
from mini_llm.ops import rotary_inference as rotary
from mini_llm.ops.rotary_inference import RotaryEmbeddingInference


pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU NumPy tests only")


def _reference(x, rotary_dim, base, layout, positions):
    """Scalar indexing oracle, with the shared helper's FP32 angle convention.

    Trig is independently evaluated with math, rounded to FP32, then to x's
    dtype. Each product/sum is rounded to x's dtype, including for FP16.
    """
    result = x.copy()
    scalar = x.dtype.type
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            pos = positions[t] if positions.ndim == 1 else positions[b, t]
            for j in range(rotary_dim // 2):
                exponent = np.float32(2 * j) / np.float32(rotary_dim)
                denominator = np.power(np.float32(base), exponent, dtype=np.float32)
                frequency = np.float32(1) / denominator
                angle = np.float32(np.float32(pos) * frequency)
                c = scalar(np.float32(math.cos(float(angle))))
                s = scalar(np.float32(math.sin(float(angle))))
                i, k = ((2 * j, 2 * j + 1) if layout == "interleaved"
                        else (j, j + rotary_dim // 2))
                for h in range(x.shape[2]):
                    a, v = x[b, t, h, i], x[b, t, h, k]
                    result[b, t, h, i] = scalar(scalar(a * c) - scalar(v * s))
                    result[b, t, h, k] = scalar(scalar(a * s) + scalar(v * c))
    return result


def _assert_close(actual, expected):
    tolerance = {np.dtype("float16"): 4e-3, np.dtype("float32"): 2e-6,
                 np.dtype("float64"): 2e-6}[actual.dtype]
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("layout", ["interleaved", "half"])
@pytest.mark.parametrize("rotary_dim,width,base", [
    (2, 7, 3.0), (6, 6, 10000.0), (6, 10, 10000.0), (64, 256, 5e6),
])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_scalar_rotation_full_partial_generic_qwen_geometry(layout, rotary_dim, width, base, dtype):
    x = np.random.default_rng(106).normal(size=(2, 4, 3, width)).astype(dtype)
    before = x.copy()
    layer = RotaryEmbeddingInference(rotary_dim, base=base, layout=layout)
    actual = layer.forward(x, start_pos=11)
    expected = _reference(x, rotary_dim, base, layout, np.arange(11, 15))
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert not np.shares_memory(actual, x)
    _assert_close(actual, expected)
    np.testing.assert_array_equal(x, before)
    assert actual[..., rotary_dim:].tobytes() == x[..., rotary_dim:].tobytes()
    np.testing.assert_allclose(
        np.sum(actual[..., :rotary_dim].astype(np.float64) ** 2, axis=-1),
        np.sum(x[..., :rotary_dim].astype(np.float64) ** 2, axis=-1),
        rtol=3e-3 if dtype == "float16" else 4e-7, atol=1e-6,
    )


@pytest.mark.parametrize("layout", ["interleaved", "half"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_noncontiguous_explicit_and_prepared_positions(layout, batched, dtype, monkeypatch):
    x = np.random.default_rng(12).normal(size=(2, 4, 2, 11)).astype(dtype)
    positions = np.array([[0, 7, 3, 12345], [101, 2, 9, 65535]], dtype=np.int64)
    if not batched:
        positions = positions[0]
    layer = RotaryEmbeddingInference(6, base=5e6, layout=layout)
    prepared = rope.prepare_rope_positions(positions, d_head=6, base=5e6, dtype=x.dtype)
    saved_cos, saved_sin = prepared.cos.copy(), prepared.sin.copy()
    raw = layer.forward(x, start_pos=-1, position_ids=positions)
    expected = _reference(x, 6, 5e6, layout, positions)
    _assert_close(raw, expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared positions must not be recomputed or reduced")

    with monkeypatch.context() as checks:
        checks.setattr(rotary, "prepare_rope_positions", forbidden)
        checks.setattr(rotary, "get_rope_cos_sin", forbidden)
        checks.setattr(rotary.xp, "any", forbidden)
        checks.setattr(rotary.xp, "max", forbidden)
        actual = layer.forward(x, position_ids=prepared)
    np.testing.assert_array_equal(actual, raw)
    np.testing.assert_array_equal(prepared.cos, saved_cos)
    np.testing.assert_array_equal(prepared.sin, saved_sin)
    assert layer._tables == {}


def test_chunks_steps_dtype_switch_and_geometric_global_cache(monkeypatch):
    layer = RotaryEmbeddingInference(6, layout="half")
    calls = []
    original = rotary.get_rope_cos_sin

    def recorded(capacity, dim, base, dtype):
        calls.append((capacity, str(dtype)))
        return original(capacity, dim, base, dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("implicit decode must not query array values")

    monkeypatch.setattr(rotary, "get_rope_cos_sin", recorded)
    x = np.random.default_rng(2).normal(size=(2, 65, 2, 9)).astype(np.float32)
    with monkeypatch.context() as checks:
        checks.setattr(rotary.xp, "any", forbidden)
        checks.setattr(rotary.xp, "max", forbidden)
        checks.setattr(rotary, "prepare_rope_positions", forbidden)
        stepped = np.concatenate([layer.forward(x[:, t:t + 1], np.int64(t))
                                  for t in range(65)], axis=1)
    capacities = [1, 2, 4, 8, 16, 32, 64, 128]
    assert calls == [(n, "float32") for n in capacities]
    assert sum(capacities) < 2 * capacities[-1]
    for n in capacities:
        assert (n, 6, 10000.0, "float32") in rope._ROPE_TABLE_CACHE
    sequence = layer.forward(x)
    chunked = np.concatenate([layer.forward(x[:, a:b], a)
                              for a, b in [(0, 3), (3, 3), (3, 17), (17, 65)]], axis=1)
    np.testing.assert_array_equal(stepped, sequence)
    np.testing.assert_array_equal(chunked, sequence)
    for dtype in ("float64", "float16", "float32", "float64"):
        actual = layer.forward(x[:, :2].astype(dtype), 3)
        _assert_close(actual, _reference(x[:, :2].astype(dtype), 6, 10000.0,
                                        "half", np.array([3, 4])))
        assert layer._tables[dtype][1].dtype == np.dtype(dtype)
    assert calls == ([(n, "float32") for n in capacities]
                     + [(8, "float64"), (8, "float16")])


def test_noncontiguous_readonly_input_and_bit_exact_suffix():
    storage = np.arange(2 * 5 * 3 * 24, dtype=np.float64).reshape(2, 5, 3, 24)
    x = storage[:, ::2, :, ::2]
    # Include signed zero, infinities and a NaN payload in the untouched suffix.
    x[..., 6:] = np.array([0., -0., np.inf, -np.inf, np.nan, 17.])
    x[0, 0, 0, 10:11].view(np.uint64)[0] = 0x7ff8000000000123
    before = storage.tobytes()
    x.flags.writeable = False
    actual = RotaryEmbeddingInference(6, layout="half").forward(x, 7)
    assert actual[..., 6:].tobytes() == x[..., 6:].tobytes()
    assert storage.tobytes() == before
    assert not np.shares_memory(actual, x)
    _assert_close(actual[..., :6], _reference(x, 6, 10000.0, "half", np.arange(7, 10))[..., :6])


def test_rotation_never_reads_suffix_or_other_heads_batches():
    x = np.ones((2, 3, 2, 10), dtype=np.float32)
    layer = RotaryEmbeddingInference(6, layout="half")
    baseline = layer.forward(x, 2)
    changed = x.copy()
    changed[0, :, 0, :6] *= 5
    changed[..., 6:] = 999
    actual = layer.forward(changed, 2)
    np.testing.assert_array_equal(actual[1, :, :, :6], baseline[1, :, :, :6])
    np.testing.assert_array_equal(actual[0, :, 1, :6], baseline[0, :, 1, :6])
    np.testing.assert_array_equal(actual[..., 6:], changed[..., 6:])


@pytest.mark.parametrize("layout", ["half", "interleaved"])
@pytest.mark.parametrize("mode", ["implicit", "explicit", "batched", "prepared"])
def test_empty_chunks(layout, mode, monkeypatch):
    layer = RotaryEmbeddingInference(6, layout=layout)
    x = np.empty((2, 0, 3, 10), dtype=np.float16)
    ids = np.empty((2, 0) if mode == "batched" else (0,), dtype=np.int64)
    positions = (None if mode == "implicit" else
                 rope.prepare_rope_positions(ids, d_head=6, dtype=x.dtype)
                 if mode == "prepared" else ids)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty chunks must not allocate an implicit table")

    monkeypatch.setattr(rotary, "get_rope_cos_sin", forbidden)
    result = layer.forward(x, start_pos=10**9, position_ids=positions)
    assert result is not x and result.shape == x.shape and result.dtype == x.dtype
    assert layer._tables == {}


@pytest.mark.parametrize("kwargs,error", [
    ({"rotary_dim": 0}, ValueError), ({"rotary_dim": -2}, ValueError),
    ({"rotary_dim": 3}, ValueError), ({"rotary_dim": True}, TypeError),
    ({"rotary_dim": 4.0}, TypeError), ({"rotary_dim": np.array(4)}, TypeError),
    ({"rotary_dim": 4, "base": 0}, ValueError),
    ({"rotary_dim": 4, "base": -1}, ValueError),
    ({"rotary_dim": 4, "base": np.inf}, ValueError),
    ({"rotary_dim": 4, "base": np.nan}, ValueError),
    ({"rotary_dim": 4, "base": True}, TypeError),
    ({"rotary_dim": 4, "base": "10000"}, TypeError),
    ({"rotary_dim": 4, "layout": "qwen"}, ValueError),
])
def test_invalid_constructor(kwargs, error):
    with pytest.raises(error):
        RotaryEmbeddingInference(**kwargs)


@pytest.mark.parametrize("start,error", [(-1, ValueError), (1.0, TypeError),
                                          (True, TypeError), (np.array(2), TypeError)])
@pytest.mark.parametrize("time", [0, 2])
def test_invalid_start_is_rejected_without_input_mutation(start, error, time):
    x = np.ones((2, time, 3, 8), dtype=np.float32)
    before = x.tobytes()
    layer = RotaryEmbeddingInference(4)
    with pytest.raises(error, match="start_pos"):
        layer.forward(x, start)
    assert x.tobytes() == before and layer._tables == {}


@pytest.mark.parametrize("x,error", [
    ([[[[1.]]]], TypeError), (np.ones((2, 3, 8)), ValueError),
    (np.ones((1, 2, 1, 2)), ValueError),
    (np.ones((1, 2, 1, 8), dtype=np.int32), TypeError),
    (np.ones((1, 2, 1, 8), dtype=np.complex64), TypeError),
])
def test_invalid_input(x, error):
    with pytest.raises(error):
        RotaryEmbeddingInference(4).forward(x)


@pytest.mark.parametrize("positions,error", [
    (np.array([0, -1]), ValueError), (np.array([0., 1.]), TypeError),
    (np.array([True, False]), TypeError), (np.array([0]), ValueError),
    (np.zeros((1, 2), dtype=np.int64), ValueError),
    (np.zeros((2, 2, 1), dtype=np.int64), ValueError),
])
def test_invalid_explicit_positions(positions, error):
    x = np.ones((2, 2, 1, 8), dtype=np.float32)
    before = x.tobytes()
    layer = RotaryEmbeddingInference(4)
    with pytest.raises(error):
        layer.forward(x, position_ids=positions)
    assert x.tobytes() == before and layer._tables == {}


@pytest.mark.parametrize("defect", [
    "dim", "base", "dtype", "shape", "rank", "cos_shape", "sin_shape",
    "cos_dtype", "sin_dtype", "cos_backend",
])
@pytest.mark.parametrize("time", [0, 2])
def test_prepared_metadata_and_factors_validated_before_output(defect, time):
    x = np.ones((2, time, 1, 8), dtype=np.float32)
    prepared = rope.prepare_rope_positions(np.arange(time), d_head=4, dtype=x.dtype)
    changes = {
        "dim": {"d_head": 8}, "base": {"base": 5e6},
        "dtype": {"dtype": "float16"}, "shape": {"shape": (time + 1,)},
        "rank": {"shape": (2, time, 1)},
        "cos_shape": {"cos": np.ones((1, time, 1, 4), dtype=np.float32)},
        "sin_shape": {"sin": np.ones((1, time, 2, 2), dtype=np.float32)},
        "cos_dtype": {"cos": prepared.cos.astype(np.float64)},
        "sin_dtype": {"sin": prepared.sin.astype(np.float16)},
        "cos_backend": {"cos": prepared.cos.tolist()},
    }
    malformed = replace(prepared, **changes[defect])
    before = x.tobytes()
    layer = RotaryEmbeddingInference(4)
    with pytest.raises((TypeError, ValueError), match="prepared"):
        layer.forward(x, position_ids=malformed)
    assert x.tobytes() == before and layer._tables == {}


def test_fp64_factors_follow_fp32_angles_not_fp64_trigonometry():
    position = 65535
    x = np.zeros((1, 1, 1, 12), dtype=np.float64)
    x[..., :3] = 1
    layer = RotaryEmbeddingInference(6, base=5e6, layout="half")
    actual = layer.forward(x, position_ids=np.array([position]))
    _assert_close(actual, _reference(x, 6, 5e6, "half", np.array([position])))
    true_fp64_cos = math.cos(position / (5e6 ** (2 / 6)))
    assert abs(actual[0, 0, 0, 1] - true_fp64_cos) > 1e-6
    assert actual.dtype == np.float64
