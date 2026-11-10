"""CPU-only scalar oracles and streaming contracts for causal convolution."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, resolve_dtype, xp
from mini_llm.ops.causal_conv_inference import (
    CausalConvState,
    CausalDepthwiseConv1DInference,
)


pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU NumPy tests only")


def _reference(x, weight, bias=None, activation=None, history=None):
    """Independent scalar indexing oracle, not the implementation's shift loop."""
    batch, time, channels = x.shape
    kernel = weight.shape[1]
    prefix = (np.zeros((batch, kernel - 1, channels)) if history is None
              else np.asarray(history, dtype=np.float64))
    raw = np.concatenate((prefix, np.asarray(x, dtype=np.float64)), axis=1)
    result = np.empty(x.shape, dtype=np.float64)
    for b in range(batch):
        for t in range(time):
            for c in range(channels):
                total = 0.0
                for j in range(kernel):
                    total += float(weight[c, j]) * float(raw[b, t + j, c])
                if bias is not None:
                    total += float(bias[c])
                if activation == "silu":
                    total = total * (1.0 / (1.0 + np.exp(-total)))
                result[b, t, c] = total
    return result


def _layer(channels=3, kernel=4, **kwargs):
    layer = CausalDepthwiseConv1DInference(channels, kernel, **kwargs)
    weight = np.arange(1, channels * kernel + 1, dtype=np.float32).reshape(channels, kernel)
    layer.set_weights(xp.asarray(weight / 10))
    return layer


def test_distinguishable_taps_oldest_to_current():
    layer = CausalDepthwiseConv1DInference(1, 3)
    layer.set_weights(xp.asarray([[1., 10., 100.]]))
    x = xp.asarray([[[1.], [2.], [3.], [4.]]])
    state = layer.create_state(1)
    np.testing.assert_array_equal(layer.forward(x, state).ravel(), [100, 210, 321, 432])
    np.testing.assert_array_equal(state.history.ravel(), [3, 4])
    assert state.length == 4


@pytest.mark.parametrize("kernel", [1, 2, 5, 11])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("compute_dtype", ["float32", "float64"])
def test_scalar_reference_sequence_chunks_steps_and_initial_history(kernel, activation, compute_dtype):
    rng = np.random.default_rng(106)
    x = xp.asarray(rng.normal(size=(3, 9, 4)).astype(np.float32))
    weight = xp.asarray(rng.normal(scale=.2, size=(4, kernel)).astype(np.float32))
    bias = xp.asarray(rng.normal(scale=.1, size=4).astype(np.float32))
    layer = CausalDepthwiseConv1DInference(4, kernel, activation=activation,
                                         compute_dtype=compute_dtype)
    layer.set_weights(weight, bias)
    initial = layer.create_state(3)
    initial.history[...] = rng.normal(size=initial.history.shape)
    initial.length = 17
    saved_history = initial.history.copy()
    expected = _reference(x, weight, bias, activation, saved_history)
    sequence, chunks, steps = initial.copy(), initial.copy(), initial.copy()
    actual = layer.forward(x, sequence)
    chunked = xp.concatenate([layer.forward(x[:, a:b], chunks)
                              for a, b in [(0, 2), (2, 2), (2, 3), (3, 8), (8, 9)]], axis=1)
    stepped = xp.stack([layer.step(x[:, t], steps) for t in range(x.shape[1])], axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual, chunked)
    np.testing.assert_array_equal(actual, stepped)
    for state in (sequence, chunks, steps):
        assert state.length == 26
        assert state.history.dtype == np.dtype(compute_dtype)
        raw = np.concatenate((saved_history, x.astype(compute_dtype)), axis=1)
        np.testing.assert_array_equal(state.history, raw[:, raw.shape[1] - kernel + 1:])
    np.testing.assert_array_equal(initial.history, saved_history)
    assert initial.length == 17


def test_reset_copy_sessions_and_buffer_aliases():
    layer = _layer()
    state = layer.create_state(2)
    alias = state.history
    other = layer.create_state(2)
    x = xp.arange(30, dtype=xp.float32).reshape(2, 5, 3)
    layer.forward(x, state)
    assert state.history is alias
    fork = state.copy()
    assert fork.length == state.length == 5
    assert not np.shares_memory(fork.history, state.history)
    expected_fork = fork.history.copy()
    np.testing.assert_array_equal(other.history, 0)
    assert other.length == 0
    state.reset()
    assert state.history is alias
    assert state.length == 0
    np.testing.assert_array_equal(alias, 0)
    np.testing.assert_array_equal(fork.history, expected_fork)
    fork.history.fill(-12)
    np.testing.assert_array_equal(state.history, 0)
    fresh = layer.create_state(2)
    np.testing.assert_array_equal(layer.forward(x, state), layer.forward(x, fresh))


def test_empty_chunk_is_noop_and_kernel_one_has_empty_history():
    for kernel in (1, 4):
        layer = _layer(kernel=kernel)
        state = layer.create_state(2)
        state.history.fill(3)
        state.length = 13
        alias, before = state.history, state.history.copy()
        result = layer.forward(xp.empty((2, 0, 3), dtype=xp.float16), state)
        assert result.shape == (2, 0, 3)
        assert result.dtype == xp.float16
        assert state.history is alias
        np.testing.assert_array_equal(state.history, before)
        assert state.length == 13
        layer.step(xp.ones((2, 3), dtype=xp.float32), state)
        assert state.history.shape == (2, kernel - 1, 3)
        assert state.length == 14


def test_causality_channel_and_batch_isolation():
    rng = np.random.default_rng(49)
    layer = _layer()
    x = xp.asarray(rng.normal(size=(2, 8, 3)).astype(np.float32))
    baseline = layer.forward(x, layer.create_state(2))
    future = x.copy()
    future[:, 5:] += 19
    changed = layer.forward(future, layer.create_state(2))
    np.testing.assert_array_equal(changed[:, :5], baseline[:, :5])
    isolated = x.copy()
    isolated[0, :, 1] += 7
    changed = layer.forward(isolated, layer.create_state(2))
    np.testing.assert_array_equal(changed[1], baseline[1])
    np.testing.assert_array_equal(changed[0, :, [0, 2]], baseline[0, :, [0, 2]])
    assert np.any(changed[0, :, 1] != baseline[0, :, 1])


@pytest.mark.parametrize("input_dtype", ["float16", "float32", "float64", "bfloat16"])
@pytest.mark.parametrize("compute_dtype", ["float32", "float64"])
def test_dtype_raw_history_and_activation_before_output_cast(input_dtype, compute_dtype):
    try:
        dtype = resolve_dtype(input_dtype)
    except RuntimeError:
        pytest.skip("optional bfloat16 dtype is unavailable")
    x = xp.asarray([[[.371, -.829], [.613, .127], [-.299, .911]]], dtype=dtype)
    weight = xp.asarray([[1.109, -.719], [.333, .887]], dtype=dtype)
    bias = xp.asarray([.173, -.241], dtype=dtype)
    layer = CausalDepthwiseConv1DInference(2, 2, activation="silu", compute_dtype=compute_dtype)
    layer.set_weights(weight, bias)
    state = layer.create_state(1)
    actual = layer.forward(x, state)
    # Compute in the configured dtype, using the existing SiLU formula, and
    # round only once. The preceding scalar-oracle tests cover convolution.
    pre = (x[:, 1].astype(compute_dtype) * weight[:, 0].astype(compute_dtype)
           + x[:, 2].astype(compute_dtype) * weight[:, 1].astype(compute_dtype)
           + bias.astype(compute_dtype))
    expected_last = (pre * (1.0 / (1.0 + np.exp(-pre)))).astype(dtype)
    assert actual.dtype == np.dtype(dtype)
    assert state.history.dtype == np.dtype(compute_dtype)
    np.testing.assert_array_equal(actual[:, -1], expected_last)
    np.testing.assert_array_equal(state.history, x[:, -1:, :].astype(compute_dtype))
    assert layer.weight.dtype == np.dtype(compute_dtype)
    assert layer.bias.dtype == np.dtype(compute_dtype)


def test_minimum_fp32_avoids_fp16_product_overflow():
    layer = CausalDepthwiseConv1DInference(1, 2)
    layer.set_weights(xp.asarray([[300, -300]], dtype=xp.float16))
    state = layer.create_state(1)
    state.history.fill(300)
    result = layer.forward(xp.asarray([[[300]]], dtype=xp.float16), state)
    np.testing.assert_array_equal(result, 0)
    assert result.dtype == xp.float16


@pytest.mark.parametrize("channels,kernel,kwargs", [
    (0, 2, {}), (-1, 2, {}), (2, 0, {}), (2, -1, {}),
    (True, 2, {}), (2, 1.5, {}), (2, 2, {"activation": "relu"}),
    (2, 2, {"compute_dtype": "float16"}), (2, 2, {"compute_dtype": "int32"}),
    (2, 2, {"compute_dtype": "complex64"}), (2, 2, {"compute_dtype": "invalid"}),
])
def test_invalid_constructor(channels, kernel, kwargs):
    with pytest.raises((TypeError, ValueError)):
        CausalDepthwiseConv1DInference(channels, kernel, **kwargs)


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch_size(batch):
    with pytest.raises((TypeError, ValueError)):
        _layer().create_state(batch)


@pytest.mark.parametrize("weight,bias", [
    ([[1., 2.]], None), (np.ones((3, 4), dtype=np.int32), None),
    (np.ones((3, 4), dtype=np.complex64), None),
    (np.ones((3, 4), dtype=bool), None),
    (np.ones((4, 3)), None), (np.ones((3, 4, 1)), None),
    (np.ones((3, 4)), [1., 2., 3.]),
    (np.ones((3, 4)), np.ones(3, dtype=np.int32)),
    (np.ones((3, 4)), np.ones((1, 3))),
    (np.ones((3, 4)), np.ones(2)),
])
def test_invalid_weights_preserve_existing_parameters(weight, bias):
    layer = _layer()
    original_weight, original_bias = layer.weight, layer.bias
    with pytest.raises((TypeError, ValueError)):
        layer.set_weights(weight, bias)
    assert layer.weight is original_weight
    assert layer.bias is original_bias


def test_weight_snapshot_and_optional_bias_removal():
    layer = _layer()
    weight = xp.ones((3, 4), dtype=xp.float32)
    bias = xp.ones(3, dtype=xp.float32)
    layer.set_weights(weight, bias)
    weight.fill(8)
    bias.fill(9)
    np.testing.assert_array_equal(layer.weight, 1)
    np.testing.assert_array_equal(layer.bias, 1)
    layer.set_weights(weight)
    assert layer.bias is None


@pytest.mark.parametrize("case", [
    "list_input", "integer_input", "complex_input", "bool_input", "rank_input",
    "channels_input", "empty_batch", "batch_state", "kernel_state", "channels_state",
    "rank_state", "list_history", "integer_history", "half_history", "double_history",
    "negative_length", "float_length", "bool_length", "readonly_history", "unset_weights",
])
def test_invalid_input_or_state_fails_before_mutation(case):
    layer = _layer()
    state = layer.create_state(2)
    state.history.fill(.75)
    state.length = 5
    x = xp.ones((2, 3, 3), dtype=xp.float32)
    if case == "list_input":
        x = x.tolist()
    elif case in ("integer_input", "complex_input", "bool_input"):
        dtype = {"integer_input": "int32", "complex_input": "complex64", "bool_input": "bool"}[case]
        x = x.astype(dtype)
    elif case == "rank_input":
        x = x[:, 0]
    elif case == "channels_input":
        x = x[:, :, :2]
    elif case == "empty_batch":
        x = x[:0]
    elif case in ("batch_state", "kernel_state", "channels_state", "rank_state"):
        shape = {"batch_state": (1, 3, 3), "kernel_state": (2, 2, 3),
                 "channels_state": (2, 3, 2), "rank_state": (2, 9)}[case]
        state.history = xp.full(shape, .75, dtype=xp.float32)
    elif case == "list_history":
        state.history = state.history.tolist()
    elif case in ("integer_history", "half_history", "double_history"):
        dtype = {"integer_history": "int32", "half_history": "float16", "double_history": "float64"}[case]
        state.history = state.history.astype(dtype)
    elif case in ("negative_length", "float_length", "bool_length"):
        state.length = {"negative_length": -1, "float_length": 1.5, "bool_length": True}[case]
    elif case == "readonly_history":
        state.history.flags.writeable = False
    elif case == "unset_weights":
        layer = CausalDepthwiseConv1DInference(3, 4)
    history_alias = state.history
    before = np.array(state.history, copy=True)
    length_before = state.length
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        layer.forward(x, state)
    assert state.history is history_alias
    np.testing.assert_array_equal(state.history, before)
    assert state.length == length_before


def test_wrong_state_type_and_step_validation_do_not_mutate():
    layer = _layer()
    state = layer.create_state(2)
    state.history.fill(4)
    before = state.history.copy()
    with pytest.raises(TypeError):
        layer.forward(xp.ones((2, 1, 3), dtype=xp.float32), object())
    for bad in (xp.ones((2, 1, 3)), xp.ones((2, 2)), xp.ones((1, 3)),
                xp.ones((2, 3), dtype=xp.int32), [[1., 2., 3.]]):
        with pytest.raises((TypeError, ValueError)):
            layer.step(bad, state)
        np.testing.assert_array_equal(state.history, before)
        assert state.length == 0


def test_empty_chunk_still_validates_and_noncontiguous_input_works():
    layer = _layer()
    state = layer.create_state(2)
    state.history = state.history.astype(np.float16)
    with pytest.raises(TypeError):
        layer.forward(xp.empty((2, 0, 3), dtype=xp.float32), state)
    state = layer.create_state(2)
    x = xp.arange(72, dtype=xp.float32).reshape(2, 6, 6)[:, ::2, ::2]
    assert not x.flags.c_contiguous
    actual = layer.forward(x, state)
    np.testing.assert_allclose(actual, _reference(x, layer.weight), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(state.history, x)


def test_manual_state_and_numpy_integer_metadata():
    layer = CausalDepthwiseConv1DInference(np.int64(1), np.int64(2), compute_dtype=np.float64)
    layer.set_weights(xp.asarray([[2., 3.]]))
    state = CausalConvState(xp.asarray([[[7.]]], dtype=xp.float64), np.int64(12))
    np.testing.assert_array_equal(layer.step(xp.asarray([[5.]]), state), [[29.]])
    assert state.length == 13
    assert isinstance(state.length, int)
