"""General delta-rule math, explicit state ownership and normalization contracts."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.inference_state import GenerationState
from mini_llm.ops.gated_delta_inference import (
    GatedDeltaRuleInference, GatedDeltaMixerInference, l2_normalize,
)
from mini_llm.ops.gated_norm_inference import GatedRMSNormInference


def oracle(q, k, v, log_decay, beta, initial, head_map, scale):
    """Alternative affine matrix recurrence, in FP64 and scalar batch/head loops."""
    state = np.array(initial, dtype=np.float64, copy=True)
    result = np.empty(v.shape, dtype=np.float64)
    for t in range(q.shape[1]):
        for b in range(q.shape[0]):
            for h, kh in enumerate(head_map):
                key = k[b, t, kh].astype(np.float64)
                query = q[b, t, kh].astype(np.float64)
                rate = float(beta[b, t, h])
                transform = np.exp(float(log_decay[b, t, h])) * (
                    np.eye(key.size) - rate * np.outer(key, key))
                state[b, h] = transform @ state[b, h] + rate * np.outer(key, v[b, t, h])
                result[b, t, h] = (query * scale) @ state[b, h]
    return result, state


def fixture(dtype=np.float64, heads=(2, 6), head_map=None):
    rng = np.random.default_rng(10613)
    hk, hv = heads
    rule = GatedDeltaRuleInference(hk, hv, 3, 5, head_map=head_map,
                                  state_dtype=dtype)
    arrays = [rng.normal(0, .2, shape).astype(dtype) for shape in
              ((2, 9, hk, 3), (2, 9, hk, 3), (2, 9, hv, 5))]
    arrays += [rng.uniform(-1.5, -.01, (2, 9, hv)).astype(dtype),
               rng.uniform(0, 1, (2, 9, hv)).astype(dtype)]
    state = rule.create_state(2)
    state.memory[...] = xp.asarray(rng.normal(0, .3, state.memory.shape), dtype=dtype)
    return rule, [xp.asarray(a) for a in arrays], state


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("heads,mapping", [((2, 6), None), ((3, 5), (2, 0, 2, 1, 0))])
def test_delta_matches_independent_affine_graph_and_group_mapping(dtype, heads, mapping):
    rule, arrays, state = fixture(dtype, heads, mapping)
    initial = asnumpy(state.memory).copy()
    expected, end = oracle(*(asnumpy(a) for a in arrays), initial,
                            rule.head_map, rule.query_scale)
    actual = rule.forward(*arrays, state)
    tolerance = 2e-6 if dtype == np.float32 else 2e-14
    np.testing.assert_allclose(asnumpy(actual), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(asnumpy(state.memory), end, rtol=tolerance, atol=tolerance)
    assert state.length == 9 and actual.dtype == dtype
    assert state.memory.shape == (2, heads[1], 3, 5)


def test_decay_precedes_prediction_and_read_uses_updated_state():
    rule = GatedDeltaRuleInference(1, 1, 1, 1, state_dtype="float64", query_scale=1)
    state = rule.create_state(1)
    state.memory.fill(2)
    q = k = xp.ones((1, 1, 1, 1), dtype=xp.float64)
    v = xp.full_like(q, 4)
    decay = xp.full((1, 1, 1), np.log(.5), dtype=xp.float64)
    beta = xp.ones_like(decay)
    result = rule.forward(q, k, v, decay, beta, state)
    np.testing.assert_allclose(asnumpy(state.memory), 4, rtol=0, atol=1e-14)
    np.testing.assert_allclose(asnumpy(result), 4, rtol=0, atol=1e-14)
    # No write: only the old memory decays; no extra correction decay allowed.
    beta.fill(0)
    rule.forward(q, k, v, decay, beta, state)
    np.testing.assert_allclose(asnumpy(state.memory), 2, rtol=0, atol=1e-14)


def test_chunk_step_copy_reset_empty_and_storage_identity():
    rule, arrays, initial = fixture()
    full, chunks, steps = (initial.copy() for _ in range(3))
    memory_id = id(full.memory)
    expected = rule.forward(*arrays, full)
    actual = xp.concatenate([rule.forward(*(a[:, :4] for a in arrays), chunks),
                             rule.forward(*(a[:, 4:] for a in arrays), chunks)], axis=1)
    stepwise = xp.stack([rule.step(*(a[:, t] for a in arrays), steps) for t in range(9)], axis=1)
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    np.testing.assert_array_equal(asnumpy(stepwise), asnumpy(expected))
    assert id(full.memory) == memory_id
    np.testing.assert_array_equal(asnumpy(chunks.memory), asnumpy(full.memory))
    before = asnumpy(full.memory).copy()
    empty = rule.forward(*(a[:, :0] for a in arrays), full)
    assert empty.shape == (2, 0, 6, 5) and full.length == 9
    np.testing.assert_array_equal(asnumpy(full.memory), before)
    full.reset()
    assert id(full.memory) == memory_id and full.length == 0
    assert np.count_nonzero(asnumpy(full.memory)) == 0
    assert np.count_nonzero(asnumpy(initial.memory)) > 0


@pytest.mark.parametrize("invalid", ["coefficient_shape", "memory_shape", "memory_dtype", "length", "integer_query"])
def test_delta_invalid_input_does_not_mutate_state(invalid):
    rule, arrays, state = fixture()
    if invalid == "coefficient_shape": arrays[-1] = arrays[-1][:, :, :1]
    elif invalid == "memory_shape": state.memory = state.memory[:, :1].copy()
    elif invalid == "memory_dtype": state.memory = state.memory.astype(xp.float16)
    elif invalid == "length": state.length = -1
    else: arrays[0] = arrays[0].astype(xp.int32)
    before = asnumpy(state.memory).copy()
    with pytest.raises((ValueError, TypeError)):
        rule.forward(*arrays, state)
    np.testing.assert_array_equal(asnumpy(state.memory), before)


@pytest.mark.parametrize("kwargs", [dict(n_key_heads=0), dict(n_value_heads=5),
                                     dict(head_map=(0, 1)), dict(head_map=(0, 0, 0, 0, 0, 2)),
                                     dict(state_dtype="float16"), dict(query_scale=np.inf)])
def test_delta_invalid_configuration(kwargs):
    params = dict(n_key_heads=2, n_value_heads=6, key_dim=3, value_dim=5)
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        GatedDeltaRuleInference(**params)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_l2_sum_epsilon_and_zero_vectors(dtype):
    x = np.array([[0, 0, 0], [1e-4, -2e-4, 1e-4]], dtype=dtype)
    actual = l2_normalize(xp.asarray(x), 1e-6)
    reference = x.astype(np.float64) / np.sqrt(np.sum(x.astype(np.float64)**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(asnumpy(actual), reference, rtol=2e-7, atol=1e-8)
    wrong_mean = x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True) + 1e-6)
    assert np.max(np.abs(reference - wrong_mean)) > 1e-3
    assert actual.dtype == (np.float64 if dtype == np.float64 else np.float32)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_gated_norm_per_row_norm_before_gate_and_direct_gamma(activation, dtype):
    compute = "float64" if dtype == np.float64 else "float32"
    norm = GatedRMSNormInference(5, gate_activation=activation, compute_dtype=compute)
    gamma = np.linspace(.3, 1.5, 5).astype(dtype)
    x = np.random.default_rng(9).normal(0, .3, (2, 3, 4, 5)).astype(dtype)
    gate = np.random.default_rng(10).normal(0, .8, x.shape).astype(dtype)
    norm.set_weights(xp.asarray(gamma))
    x64, g64 = x.astype(np.float64), gate.astype(np.float64)
    expected = x64 / np.sqrt(np.mean(x64*x64, axis=-1, keepdims=True) + 1e-6)
    expected *= gamma.astype(np.float64)
    act = 1 / (1 + np.exp(-g64))
    if activation == "silu": act *= g64
    expected = (expected * act).astype(dtype)
    np.testing.assert_allclose(asnumpy(norm.forward(xp.asarray(x), xp.asarray(gate))), expected,
                               rtol=.002 if dtype == np.float16 else 2e-6,
                               atol=.0005 if dtype == np.float16 else 2e-7)


def test_composite_validates_all_state_before_conv_mutation_and_generation_reset():
    mixer = GatedDeltaMixerInference(2, 4, 3, 5, 3)
    mixer.set_weights(xp.ones((32, 3), dtype=xp.float32), xp.ones(5, dtype=xp.float32))
    state = mixer.create_state(2)
    state.conv.history.fill(.2)
    state.delta.memory.fill(.3)
    qkv = xp.ones((2, 2, 32), dtype=xp.float32)
    gate = xp.ones((2, 2, 4, 5), dtype=xp.float32)
    rates = xp.zeros((2, 2, 4), dtype=xp.float32)
    initial_history = asnumpy(state.conv.history).copy()
    with pytest.raises(ValueError):
        mixer.forward(qkv, gate, rates, rates[:, :, :1], state)
    np.testing.assert_array_equal(asnumpy(state.conv.history), initial_history)
    assert state.length == 0
    mixer.forward(qkv, gate, rates, xp.ones_like(rates), state)
    conv_id, delta_id = id(state.conv.history), id(state.delta.memory)
    generation = GenerationState(xp.zeros(1), xp.zeros(1), length=2, max_length=10, batch_size=2)
    generation.recurrent_states[7] = state
    generation.reset()
    assert generation.length == state.length == 0
    assert generation.recurrent_states[7] is state
    assert id(state.conv.history) == conv_id and id(state.delta.memory) == delta_id
    assert not np.count_nonzero(asnumpy(state.conv.history))
    assert not np.count_nonzero(asnumpy(state.delta.memory))


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU packed adapter regression")
def test_native_packed_empty_mixer_chunk_never_launches_cpp(monkeypatch):
    from types import SimpleNamespace
    from test_0106m_qwen3next_mixer import _fixture
    from mini_llm.external.qwen3next_mixer import Qwen3NextRecurrentMixerInference
    from mini_llm.ops import native_packed_cpu
    index, _, _ = _fixture(packed=True)
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0, "native")
    state = mixer.create_state(2)
    state.conv.history.fill(.2)
    state.delta.memory.fill(.3)
    state.conv.length = state.delta.length = 7
    before_history, before_memory = state.conv.history.copy(), state.delta.memory.copy()
    def forbidden(*args):
        pytest.fail("Empty GEMV must not call C++ with tokens=0")
    monkeypatch.setattr(native_packed_cpu, "_load_native", lambda: SimpleNamespace(
        mylocal_packed_gemv=forbidden))
    result = mixer.forward(np.empty((2, 0, mixer.spec.d_model), np.float32), state)
    assert result.shape == (2, 0, mixer.spec.d_model) and result.dtype == np.float32
    assert state.length == 7
    np.testing.assert_array_equal(state.conv.history, before_history)
    np.testing.assert_array_equal(state.delta.memory, before_memory)
